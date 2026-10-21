"""
A1 人脸分类 · 二阶段自蒸馏训练(128px 输入,预生成软标签版)
===========================================================

与 train_student.py 的区别:
  1. 输入分辨率 64 → 128(双线性插值放大)
  2. 教师改为 dist64(best_student.pth),构成二阶段蒸馏
  3. 教师软标签提前在 64 分辨率上生成存盘(teacher_logits_student_fp16.pt),
     训练时直接查表,省去每个 batch 的 teacher 前向(速度提升约 4 倍)
  4. 128px 显存占用大,batch=16 + 梯度累积 4 步(等效 batch=64)

输出:
  - best_final.pth:验证集准确率最高的 checkpoint
  - submission_final.csv:test 集预测结果
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
IMG_SIZE = 128

# ────────────────────────────── 数据增强 ──────────────────────────────
class RandomBlur:
    """以概率 p 做高斯模糊,半径在 radius 范围内随机。"""

    def __init__(self, p=0.3, radius=(0.8, 1.6)):
        self.p = p
        self.radius = radius

    def __call__(self, image):
        if random.random() < self.p:
            blur_radius = random.uniform(*self.radius)
            image = image.filter(ImageFilter.GaussianBlur(blur_radius))
        return image


train_transform = transforms.Compose([
    RandomBlur(p=0.3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
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


class IndexedDataset(Dataset):
    """包装器:除了 (image, label) 之外再返回样本索引,用于取教师软标签。"""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]
        return image, label, index


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


class WideStudent(nn.Module):
    """加宽学生网络:通道数 64→128→256→512。"""

    def __init__(self, num_classes=1000, drop=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.stage1 = nn.Sequential(BasicBlock(64), BasicBlock(64))
        self.stage2 = nn.Sequential(DownBlock(64, 128), BasicBlock(128))
        self.stage3 = nn.Sequential(DownBlock(128, 256), BasicBlock(256))
        self.stage4 = nn.Sequential(DownBlock(256, 512), BasicBlock(512))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)


# ────────────────────────────── 评估函数 ──────────────────────────────
@torch.no_grad()
def evaluate(model, loader):
    """在给定 loader 上计算分类准确率。"""
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        images = batch[0].to(device)
        labels = batch[1].to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total


# ────────────────────────────── 训练主流程 ──────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--T", type=float, default=4.0)
    parser.add_argument("--teacher", type=str, default="best_student.pth")
    parser.add_argument("--resume", type=str, default="", help="从 checkpoint 继续训练")
    parser.add_argument("--data", type=str, default="/home/room115/jhy/a1_data")
    args = parser.parse_args()

    base_set = datasets.ImageFolder(os.path.join(args.data, "train"), train_transform)
    val_set = datasets.ImageFolder(os.path.join(args.data, "val"), eval_transform)
    test_set = TestSet(os.path.join(args.data, "test"),
                       os.path.join(args.data, "test_list.txt"), eval_transform)
    train_set = IndexedDataset(base_set)

    class_counts = Counter(base_set.targets)
    sample_weights = [1.0 / class_counts[label] for label in base_set.targets]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_set, batch_size=args.batch, sampler=sampler, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=8)

    teacher_logits = torch.load("teacher_logits_student_fp16.pt", map_location="cpu").float()
    assert teacher_logits.size(0) == len(base_set), "教师软标签与训练集样本数不一致"
    print(f"[蒸馏] 教师软标签 {tuple(teacher_logits.shape)} (来自 {args.teacher}), T={args.T}, α={args.alpha}")

    model = WideStudent(num_classes=len(base_set.classes)).to(device)
    print(f"[模型] 学生参数量: {sum(p.numel() for p in model.parameters()):,}, "
          f"输入 {IMG_SIZE}x{IMG_SIZE}, 梯度累积 {args.accum} 步(等效 batch={args.batch*args.accum})")

    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    def distill_loss(logits, labels, teacher_logits_batch):
        hard_loss = ce_loss(logits, labels)
        soft_loss = F.kl_div(F.log_softmax(logits / args.T, dim=1),
                             F.softmax(teacher_logits_batch / args.T, dim=1),
                             reduction="batchmean") * (args.T ** 2)
        return args.alpha * hard_loss + (1 - args.alpha) * soft_loss

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=5e-4, nesterov=True)
    # 梯度累积:lr 调度按"有效 step"(每 accum 步算 1 步)计
    total_steps = args.epochs * len(train_loader) // args.accum
    warmup_steps = 5 * len(train_loader) // args.accum

    def lr_at(step):
        if step < warmup_steps:
            return args.lr * (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.lr * 0.5 * (1 + np.cos(np.pi * progress))

    best_acc = 0.0
    step = 0
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["opt"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint.get("best", 0.0)
        print(f"[resume] 从 epoch {start_epoch} 继续, 历史 best={best_acc:.4f}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        for batch_index, (images, labels, indices) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            teacher_logits_batch = teacher_logits[indices].to(device)
            loss = distill_loss(model(images), labels, teacher_logits_batch) / args.accum
            loss.backward()
            if (batch_index + 1) % args.accum == 0:
                for group in optimizer.param_groups:
                    group["lr"] = lr_at(step)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
            running_loss += loss.item() * args.accum

        train_acc = evaluate(model, train_loader)
        val_acc = evaluate(model, val_loader)
        print(f"epoch {epoch+1:3d}/{args.epochs} | loss {running_loss/len(train_loader):.4f} "
              f"| train {train_acc:.3f} | val {val_acc:.4f} | lr {optimizer.param_groups[0]['lr']:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "opt": optimizer.state_dict(),
                        "epoch": epoch, "best": best_acc}, "best_final.pth")
            print(f"    ✓ 新最优 val={best_acc:.4f}")

    # ───────────── 用最优 checkpoint 预测 test 集 ─────────────
    checkpoint = torch.load("best_final.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    rows = []
    seen = set()
    with torch.no_grad():
        for images, names in test_loader:
            predictions = model(images.to(device)).argmax(1).cpu().numpy()
            for name, pred in zip(names, predictions):
                if name in seen:
                    continue
                seen.add(name)
                rows.append((name, base_set.classes[pred]))
    rows.sort(key=lambda row: test_set.names.index(row[0]))
    pd.DataFrame(rows, columns=["id", "label"]).to_csv("submission_final.csv", index=False)
    print(f"[完成] best val={best_acc:.4f}, submission_final.csv 共 {len(rows)} 行")


if __name__ == "__main__":
    main()
