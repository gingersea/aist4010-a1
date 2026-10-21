"""
A1 人脸分类 · AdamW 优化器版训练(64px, 自蒸馏保留)
====================================================

与 train_bagging.py 同构,区别:
  1. 优化器 SGD → AdamW(weight_decay=5e-2)
  2. --sample-frac 默认 1.0(用全量数据)
  3. 默认 lr 1e-3

用途:作为集成成员提供"优化器多样性"(SGD 与 AdamW 收敛到不同的解)。

输出:
  - best_adam.pth:验证集准确率最高的 checkpoint
  - submission_adam.csv:test 集预测结果
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN, STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
IMG_SIZE = 64

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
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
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


class IndexedSubset(Dataset):
    """子集包装器:按给定索引取子集,并返回原始索引(用于取教师软标签)。"""

    def __init__(self, base, indices):
        self.base = base
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original_index = self.indices[index]
        image, label = self.base[original_index]
        return image, label, original_index


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
    parser.add_argument("--fold", type=int, required=True, help="0/1/2 第几个成员")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--sample-frac", type=float, default=1.0, help="抽样比例(adam 版用全量)")
    parser.add_argument("--resume", type=str, default="", help="从 checkpoint 继续训练")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--T", type=float, default=4.0)
    parser.add_argument("--data", type=str, default="/home/room115/jhy/a1_data")
    args = parser.parse_args()

    # 每个 fold 用不同的种子(与 bagging 一致)
    torch.manual_seed(42 + args.fold * 1000)
    np.random.seed(42 + args.fold * 1000)
    random.seed(42 + args.fold * 1000)

    base_set = datasets.ImageFolder(os.path.join(args.data, "train"), train_transform)
    val_set = datasets.ImageFolder(os.path.join(args.data, "val"), eval_transform)
    test_set = TestSet(os.path.join(args.data, "test"),
                       os.path.join(args.data, "test_list.txt"), eval_transform)

    # 按比例随机抽样(adam 版默认全量;样本顺序排序后固定)
    num_samples = len(base_set)
    indices = np.random.choice(num_samples, int(num_samples * args.sample_frac), replace=False)
    indices = sorted(indices.tolist())
    train_set = IndexedSubset(base_set, indices)

    class_counts = Counter(base_set.targets[i] for i in indices)
    sample_weights = [1.0 / class_counts[base_set.targets[i]] for i in indices]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(indices))

    train_loader = DataLoader(train_set, batch_size=args.batch, sampler=sampler, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=8)

    teacher_logits = torch.load("teacher_logits_student_fp16.pt", map_location="cpu").float()
    print(f"[fold {args.fold}] 样本 {len(indices)}/{num_samples} ({args.sample_frac:.0%}), "
          f"种子 {42 + args.fold*1000}")

    model = WideStudent(num_classes=len(base_set.classes)).to(device)
    print(f"[fold {args.fold}] 参数量: {sum(p.numel() for p in model.parameters()):,}")

    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    def distill_loss(logits, labels, teacher_logits_batch):
        hard_loss = ce_loss(logits, labels)
        soft_loss = F.kl_div(F.log_softmax(logits / args.T, dim=1),
                             F.softmax(teacher_logits_batch / args.T, dim=1),
                             reduction="batchmean") * (args.T ** 2)
        return args.alpha * hard_loss + (1 - args.alpha) * soft_loss

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-2)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = 5 * len(train_loader)

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
        for images, labels, indices_batch in train_loader:
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step)
            images = images.to(device)
            labels = labels.to(device)
            teacher_logits_batch = teacher_logits[indices_batch].to(device)
            optimizer.zero_grad()
            loss = distill_loss(model(images), labels, teacher_logits_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            step += 1

        train_acc = evaluate(model, train_loader)
        val_acc = evaluate(model, val_loader)
        print(f"[fold {args.fold}] epoch {epoch+1:3d}/{args.epochs} | loss {running_loss/len(train_loader):.4f} "
              f"| train {train_acc:.3f} | val {val_acc:.4f} | lr {optimizer.param_groups[0]['lr']:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "opt": optimizer.state_dict(),
                        "epoch": epoch, "best": best_acc}, "best_adam.pth")
            print(f"    ✓ [fold {args.fold}] 新最优 val={best_acc:.4f}")

    # ───────────── 用最优 checkpoint 预测 test 集 ─────────────
    checkpoint = torch.load("best_adam.pth", map_location=device, weights_only=False)
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
    pd.DataFrame(rows, columns=["id", "label"]).to_csv("submission_adam.csv", index=False)
    print(f"[fold {args.fold}] 完成 best val={best_acc:.4f}, submission_adam.csv")


if __name__ == "__main__":
    main()
