"""
A1 人脸分类 · 二阶段自蒸馏(128 输入,预生成软标签版)
=====================================================
优化:教师软标签提前在 64 分辨率上生成存盘(teacher_logits_student_fp16.pt),
训练时直接查表,省去每个 batch 的 teacher 前向(速度提升 ~4 倍)。
"""

import argparse, os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image, ImageFilter

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────── 数据 ──────────────────────────────
MEAN, STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
IMG_SIZE = 128

class RandomBlur:
    def __init__(self, p=0.3, radius=(0.8, 1.6)):
        self.p = p; self.radius = radius
    def __call__(self, img):
        if random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(*self.radius)))
        return img

train_tf = transforms.Compose([
    RandomBlur(p=0.3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

class TestSet(Dataset):
    def __init__(self, root, list_file, transform):
        self.root = root
        with open(list_file) as f:
            self.names = [l.strip() for l in f if l.strip()]
        self.transform = transform
    def __len__(self): return len(self.names)
    def __getitem__(self, i):
        name = self.names[i]
        img = Image.open(os.path.join(self.root, name)).convert("RGB")
        return self.transform(img), name

class IndexedDataset(Dataset):
    def __init__(self, base):
        self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        img, label = self.base[i]
        return img, label, i

# ────────────────────────────── 模型 ──────────────────────────────
class BasicBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)

class DownBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride=2, bias=False),
            nn.BatchNorm2d(cout))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))

class WideStudent(nn.Module):
    def __init__(self, num_classes=1000, drop=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU())
        self.stage1 = nn.Sequential(BasicBlock(64), BasicBlock(64))
        self.stage2 = nn.Sequential(DownBlock(64, 128), BasicBlock(128))
        self.stage3 = nn.Sequential(DownBlock(128, 256), BasicBlock(256))
        self.stage4 = nn.Sequential(DownBlock(256, 512), BasicBlock(512))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x); x = self.stage2(x)
        x = self.stage3(x); x = self.stage4(x)
        x = self.gap(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)

# ────────────────────────────── 训练 ──────────────────────────────
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for batch in loader:
        images, labels = batch[0].to(device), batch[1].to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--teacher", type=str, default="best_student.pth")
    ap.add_argument("--resume", type=str, default="", help="从 checkpoint 继续训练")
    ap.add_argument("--data", type=str, default="/home/room115/jhy/a1_data")
    args = ap.parse_args()

    base_set = datasets.ImageFolder(os.path.join(args.data, "train"), train_tf)
    val_set  = datasets.ImageFolder(os.path.join(args.data, "val"),   eval_tf)
    test_set = TestSet(os.path.join(args.data, "test"),
                       os.path.join(args.data, "test_list.txt"), eval_tf)
    train_set = IndexedDataset(base_set)

    from collections import Counter
    cnt = Counter(base_set.targets)
    weights = [1.0 / cnt[t] for t in base_set.targets]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_set, batch_size=args.batch, sampler=sampler, num_workers=8)
    val_loader   = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=8)
    test_loader  = DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=8)

    teacher_logits = torch.load("teacher_logits_student_fp16.pt", map_location="cpu").float()
    assert teacher_logits.size(0) == len(base_set), "教师软标签与训练集样本数不一致"
    print(f"[蒸馏] 教师软标签 {tuple(teacher_logits.shape)} (来自 {args.teacher}), T={args.T}, α={args.alpha}")

    model = WideStudent(num_classes=len(base_set.classes)).to(device)
    print(f"[模型] 学生参数量: {sum(p.numel() for p in model.parameters()):,}, "
          f"输入 {IMG_SIZE}x{IMG_SIZE}, 梯度累积 {args.accum} 步(等效 batch={args.batch*args.accum})")

    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    def distill_loss(logits, labels, teacher_l):
        hard = ce(logits, labels)
        soft = F.kl_div(F.log_softmax(logits / args.T, dim=1),
                        F.softmax(teacher_l / args.T, dim=1),
                        reduction="batchmean") * (args.T ** 2)
        return args.alpha * hard + (1 - args.alpha) * soft

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=5e-4, nesterov=True)
    total_steps = args.epochs * len(train_loader) // args.accum
    warmup = 5 * len(train_loader) // args.accum

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / max(1, warmup)
        p = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * 0.5 * (1 + np.cos(np.pi * p))

    best_acc = 0.0
    step = 0
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["opt"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best", 0.0)
        print(f"[resume] 从 epoch {start_epoch} 继续, 历史 best={best_acc:.4f}")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = 0.0
        optimizer.zero_grad()
        for i, (images, labels, idxs) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            t_l = teacher_logits[idxs].to(device)
            loss = distill_loss(model(images), labels, t_l) / args.accum
            loss.backward()
            if (i + 1) % args.accum == 0:
                for g in optimizer.param_groups:
                    g["lr"] = lr_at(step)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
            running += loss.item() * args.accum

        train_acc = evaluate(model, train_loader)
        val_acc = evaluate(model, val_loader)
        print(f"epoch {epoch+1:3d}/{args.epochs} | loss {running/len(train_loader):.4f} "
              f"| train {train_acc:.3f} | val {val_acc:.4f} | lr {optimizer.param_groups[0]['lr']:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "opt": optimizer.state_dict(),
                        "epoch": epoch, "best": best_acc}, "best_final.pth")
            print(f"    ✓ 新最优 val={best_acc:.4f}")

    ckpt = torch.load("best_final.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    rows, seen = [], set()
    with torch.no_grad():
        for images, names in test_loader:
            preds = model(images.to(device)).argmax(1).cpu().numpy()
            for name, p in zip(names, preds):
                if name in seen: continue
                seen.add(name)
                rows.append((name, base_set.classes[p]))
    rows.sort(key=lambda r: test_set.names.index(r[0]))
    pd.DataFrame(rows, columns=["id", "label"]).to_csv("submission_final.csv", index=False)
    print(f"[完成] best val={best_acc:.4f}, submission_final.csv 共 {len(rows)} 行")

if __name__ == "__main__":
    main()
