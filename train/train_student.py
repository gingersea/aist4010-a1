"""
A1 人脸分类 · 自我蒸馏学生训练
===============================
学生:通道翻倍的 WideResNet 风格(64→128→256→512, ~950 万参数)
损失 = α·硬标签交叉熵(带标签平滑) + (1-α)·KL(学生软概率, 教师软概率)·T²
教师软标签 = 两个教师(best.pth + best_blur.pth)logits 的平均,存于 teacher_logits_fp16.pt
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

class RandomBlur:
    def __init__(self, p=0.3, radius=(0.8, 1.6)):
        self.p = p; self.radius = radius
    def __call__(self, img):
        if random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(*self.radius)))
        return img

train_tf = transforms.Compose([
    RandomBlur(p=0.3),
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
    """包装器:除了 (img, label) 之外再返回样本索引 i,用于取教师软标签"""
    def __init__(self, base):
        self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        img, label = self.base[i]
        return img, label, i

# ────────────────────────────── 学生模型(加宽) ──────────────────────────────
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
    """通道翻倍版:64→128→256→512"""
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
        images, labels = batch[0], batch[1]
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=0.5, help="硬标签损失权重")
    ap.add_argument("--T", type=float, default=4.0, help="蒸馏温度")
    ap.add_argument("--resume", type=str, default="")
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

    teacher_logits = torch.load("teacher_logits_fp16.pt", map_location="cpu").float()  # (N,1000)
    assert teacher_logits.size(0) == len(base_set), "教师软标签与训练集样本数不一致"
    print(f"[蒸馏] 教师软标签 {tuple(teacher_logits.shape)}, T={args.T}, α={args.alpha}")

    model = WideStudent(num_classes=len(base_set.classes)).to(device)
    print(f"[模型] 学生参数量: {sum(p.numel() for p in model.parameters()):,}")

    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    def distill_loss(logits, labels, teacher_l):
        hard = ce(logits, labels)
        soft = F.kl_div(F.log_softmax(logits / args.T, dim=1),
                        F.softmax(teacher_l / args.T, dim=1),
                        reduction="batchmean") * (args.T ** 2)
        return args.alpha * hard + (1 - args.alpha) * soft

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=5e-4, nesterov=True)
    total_steps = args.epochs * len(train_loader)
    warmup = 5 * len(train_loader)

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * 0.5 * (1 + np.cos(np.pi * p))

    start_epoch, best_acc = 0, 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["opt"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best", 0.0)

    step = start_epoch * len(train_loader)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = 0.0
        for images, labels, idxs in train_loader:
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step)
            images, labels = images.to(device), labels.to(device)
            t_l = teacher_logits[idxs].to(device)     # 该批样本的教师软标签
            optimizer.zero_grad()
            loss = distill_loss(model(images), labels, t_l)
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
                        "epoch": epoch, "best": best_acc}, "best_student.pth")
            print(f"    ✓ 新最优 val={best_acc:.4f}")

    # 预测 test
    ckpt = torch.load("best_student.pth", map_location=device, weights_only=False)
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
    pd.DataFrame(rows, columns=["id", "label"]).to_csv("submission_student.csv", index=False)
    print(f"[完成] best val={best_acc:.4f}, submission_student.csv 共 {len(rows)} 行")

if __name__ == "__main__":
    main()
