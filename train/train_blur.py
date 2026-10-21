"""
A1 人脸分类 · 模糊鲁棒版训练脚本
================================

针对 test 集比 train 略模糊的观察,在基线(train_resnet.py)基础上加入:
  1. RandomBlur:训练时以概率 p 对图片做高斯模糊,让模型见过模糊脸
  2. 标签平滑(Label Smoothing):压过拟合,提升泛化

其余设置与 train_resnet.py 相同(ResNet 风格 + 类别加权采样 + 余弦退火)。

输出:
  - best_blur.pth:验证集准确率最高的 checkpoint
  - submission.csv:test 集预测结果
"""

import argparse
import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# ────────────────────────────── 基础设置 ──────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN, STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

# ────────────────────────────── 数据增强 ──────────────────────────────
class RandomBlur:
    """以概率 p 做高斯模糊,半径在 radius 范围内随机,模拟测试集的模糊。"""

    def __init__(self, p=0.3, radius=(0.8, 1.6)):
        self.p = p
        self.radius = radius

    def __call__(self, image):
        if random.random() < self.p:
            blur_radius = random.uniform(*self.radius)
            image = image.filter(ImageFilter.GaussianBlur(blur_radius))
        return image


train_transform = transforms.Compose([
    RandomBlur(p=0.3),  # 关键:随机模糊,让模型适应模糊输入
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class TestSet(Dataset):
    """test 集:按 test_list.txt 的顺序读取图片,无标签。"""

    def __init__(self, root, list_file, transform):
        self.root = root
        with open(list_file) as f:
            self.names = [line.strip() for line in f if line.strip()]
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        image = Image.open(os.path.join(self.root, name)).convert("RGB")
        return self.transform(image), name


# ────────────────────────────── 模型结构 ──────────────────────────────
class BasicBlock(nn.Module):
    """残差块:y = ReLU(F(x) + x),不改变特征图尺寸和通道数。"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)  # 残差连接


class DownBlock(nn.Module):
    """下采样残差块:通道数翻倍,特征图尺寸减半。"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class MiniResNet(nn.Module):
    """轻量 ResNet:输入 64×64,输出 num_classes 类。"""

    def __init__(self, num_classes=1000, drop=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.stage1 = nn.Sequential(BasicBlock(32), BasicBlock(32))
        self.stage2 = nn.Sequential(DownBlock(32, 64), BasicBlock(64))
        self.stage3 = nn.Sequential(DownBlock(64, 128), BasicBlock(128))
        self.stage4 = nn.Sequential(DownBlock(128, 256), BasicBlock(256))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)


# ────────────────────────────── 标签平滑损失 ──────────────────────────────
class LabelSmoothingLoss(nn.Module):
    """把 one-hot 标签 (0/1) 平滑成 (eps/(K-1), 1-eps),防止模型过度自信。"""

    def __init__(self, num_classes, eps=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, predictions, labels):
        log_probs = F.log_softmax(predictions, dim=1)
        with torch.no_grad():
            one_hot = torch.zeros_like(predictions)
            one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
            smooth_labels = one_hot * (1 - self.eps) + self.eps / self.num_classes
        return -(smooth_labels * log_probs).sum(dim=1).mean()


# ────────────────────────────── 评估函数 ──────────────────────────────
@torch.no_grad()
def evaluate(model, loader):
    """在给定 loader 上计算分类准确率。"""
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        predictions = model(images).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    return correct / total


# ────────────────────────────── 训练主流程 ──────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--blur-p", type=float, default=0.3)
    parser.add_argument("--label-smooth", type=float, default=0.1)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--data", type=str, default="/home/room115/jhy/a1_data")
    args = parser.parse_args()

    train_set = datasets.ImageFolder(os.path.join(args.data, "train"), train_transform)
    val_set = datasets.ImageFolder(os.path.join(args.data, "val"), eval_transform)
    test_set = TestSet(os.path.join(args.data, "test"),
                       os.path.join(args.data, "test_list.txt"), eval_transform)

    class_counts = Counter(train_set.targets)
    sample_weights = [1.0 / class_counts[label] for label in train_set.targets]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_set, batch_size=args.batch, sampler=sampler, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=8)

    model = MiniResNet(num_classes=len(train_set.classes)).to(device)
    print(f"[模型] 参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[配置] blur_p={args.blur_p} label_smoothing={args.label_smooth} epochs={args.epochs}")

    criterion = LabelSmoothingLoss(len(train_set.classes), eps=args.label_smooth)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=5e-4, nesterov=True)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = 5 * len(train_loader)

    def lr_at(step):
        if step < warmup_steps:
            return args.lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.lr * 0.5 * (1 + np.cos(np.pi * progress))

    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["opt"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint.get("best", 0.0)
        print(f"[resume] 从 epoch {start_epoch} 继续, 历史 best={best_acc:.4f}")

    step = start_epoch * len(train_loader)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step)
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            step += 1

        train_acc = evaluate(model, train_loader)
        val_acc = evaluate(model, val_loader)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch+1:3d}/{args.epochs} | loss {running_loss/len(train_loader):.4f} "
              f"| train {train_acc:.3f} | val {val_acc:.4f} | lr {lr_now:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "opt": optimizer.state_dict(),
                        "epoch": epoch, "best": best_acc}, "best_blur.pth")
            print(f"    ✓ 新最优 val={best_acc:.4f}, 已保存 best_blur.pth")

    # ───────────── 用最优 checkpoint 预测 test 集 ─────────────
    checkpoint = torch.load("best_blur.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    rows = []
    seen = set()
    with torch.no_grad():
        for images, names in test_loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu().numpy()
            for name, pred in zip(names, predictions):
                if name in seen:
                    continue
                seen.add(name)
                rows.append((name, train_set.classes[pred]))
    rows.sort(key=lambda row: test_set.names.index(row[0]))
    submission = pd.DataFrame(rows, columns=["id", "label"])
    submission.to_csv("submission.csv", index=False)
    print(f"[完成] best val={best_acc:.4f}, submission.csv 共 {len(submission)} 行")


if __name__ == "__main__":
    main()
