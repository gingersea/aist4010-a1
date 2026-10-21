"""
A1 人脸分类 · 主力训练脚本(参考用)
=====================================
数据:
  train/  1000 人, 93134 张(每人 24~221 张)
  val/    1000 人, 每类 2 张, 共 2000 张
  test/   2394 张, 无标签

模型:自己实现的 ResNet 风格 CNN(32/64/128/256 通道,每 stage 2 个 BasicBlock)
防过拟合:数据增强 + Dropout + 权重衰减 + 类别权重(缓解类不平衡)
调度:SGD + momentum + 余弦退火 + 5 epoch 热身
输出:best checkpoint + submission.csv(按 test_list.txt 顺序)
"""

import argparse, os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────── 数据 ──────────────────────────────
MEAN, STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

class TestSet(Dataset):
    """test/ 目录:按 test_list.txt 顺序给出文件名,无标签"""
    def __init__(self, root, list_file, transform):
        self.root = root
        with open(list_file) as f:
            self.names = [l.strip() for l in f if l.strip()]
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        img = Image.open(os.path.join(self.root, name)).convert("RGB")
        return self.transform(img), name

# ────────────────────────────── 模型 ──────────────────────────────
class BasicBlock(nn.Module):
    """残差块:y = ReLU(F(x) + x),不改变尺寸和通道"""
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)   # 残差连接

class DownBlock(nn.Module):
    """下采样残差块:通道翻倍,尺寸减半"""
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride=2, bias=False),
            nn.BatchNorm2d(cout),
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))

class MiniResNet(nn.Module):
    """输入 64×64 → 输出 1000 类"""
    def __init__(self, num_classes=1000, drop=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(),
        )                                   # 64×64×32
        self.stage1 = nn.Sequential(BasicBlock(32), BasicBlock(32))
        self.stage2 = nn.Sequential(DownBlock(32, 64), BasicBlock(64))    # 32×32×64
        self.stage3 = nn.Sequential(DownBlock(64, 128), BasicBlock(128))  # 16×16×128
        self.stage4 = nn.Sequential(DownBlock(128, 256), BasicBlock(256)) # 8×8×256
        self.gap = nn.AdaptiveAvgPool2d(1)   # 全局平均池化 → 256
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

# ────────────────────────────── 训练 ──────────────────────────────
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--data", type=str, default="/home/room115/jhy/a1_data")
    args = ap.parse_args()

    train_set = datasets.ImageFolder(os.path.join(args.data, "train"), train_tf)
    val_set   = datasets.ImageFolder(os.path.join(args.data, "val"),   eval_tf)
    test_set  = TestSet(os.path.join(args.data, "test"),
                        os.path.join(args.data, "test_list.txt"), eval_tf)

    # 类别权重:每类样本数越少,权重越大(缓解类不平衡)
    from collections import Counter
    cnt = Counter(train_set.targets)
    weights = [1.0 / cnt[t] for t in train_set.targets]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_set, batch_size=args.batch, sampler=sampler, num_workers=8)
    val_loader   = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=8)
    test_loader  = DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=8)

    model = MiniResNet(num_classes=len(train_set.classes)).to(device)
    print(f"[模型] 参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=5e-4, nesterov=True)

    total_steps = args.epochs * len(train_loader)
    warmup = 5 * len(train_loader)   # 前 5 个 epoch 线性热身

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * 0.5 * (1 + np.cos(np.pi * p))

    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["opt"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best", 0.0)
        print(f"[resume] 从 epoch {start_epoch} 继续, 历史 best={best_acc:.4f}")

    step = start_epoch * len(train_loader)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = 0.0
        for images, labels in train_loader:
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
            step += 1

        train_acc = evaluate(model, train_loader)
        val_acc = evaluate(model, val_loader)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch+1:3d}/{args.epochs} | loss {running/len(train_loader):.4f} "
              f"| train {train_acc:.3f} | val {val_acc:.4f} | lr {lr_now:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "opt": optimizer.state_dict(),
                        "epoch": epoch, "best": best_acc}, "best.pth")
            print(f"    ✓ 新最优 val={best_acc:.4f}, 已保存 best.pth")

    # ───────────── 预测 test ─────────────
    ckpt = torch.load("best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    rows, seen = [], set()
    with torch.no_grad():
        for images, names in test_loader:
            preds = model(images.to(device)).argmax(dim=1).cpu().numpy()
            for name, p in zip(names, preds):
                if name in seen: continue
                seen.add(name)
                rows.append((name, train_set.classes[p]))
    rows.sort(key=lambda r: test_set.names.index(r[0]))   # 严格按 test_list 顺序
    sub = pd.DataFrame(rows, columns=["id", "label"])
    sub.to_csv("submission.csv", index=False)
    print(f"[完成] best val={best_acc:.4f}, submission.csv 共 {len(sub)} 行")
    print(sub.head(3).to_string(index=False))

if __name__ == "__main__":
    main()
