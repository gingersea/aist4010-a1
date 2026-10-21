"""
AIST4010 A1 · 最终集成提交脚本(独立版)
==========================================
最终方案:四模型 TTA 加权集成(val 86.40%)

模型成员与权重:
  - mutual(相互蒸馏 SGD+Adam) : 10%
  - adam(Adam 优化器版)        : 20%
  - dist64(自蒸馏 64px)        : 25%
  - dist128(二阶段蒸馏 128px)  : 45%

每个模型对 test 集做 TTA(原图 + 水平翻转)取平均 logits,
再按权重加权求和,argmax 得到预测。

依赖:4 个 checkpoint + test 数据 + test_list.txt
输出:submission_final.csv(可直接提交 Kaggle)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import os

MEAN, STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────── 模型定义(WideStudent) ──────────────────────────────
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
            nn.BatchNorm2d(cout),
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class WideStudent(nn.Module):
    """ResNet 风格骨干:64→128→256→512 通道"""
    def __init__(self, num_classes=1000, drop=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
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


# ────────────────────────────── 数据集 ──────────────────────────────
class TestSet(Dataset):
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


def tta_transforms(base_size=None):
    """TTA 变体:原图 + 水平翻转"""
    tfs = []
    if base_size:  # 128px 模型
        tfs.append(transforms.Compose([
            transforms.Resize((base_size, base_size)),
            transforms.ToTensor(), transforms.Normalize(MEAN, STD)]))
        tfs.append(transforms.Compose([
            transforms.Resize((base_size, base_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(), transforms.Normalize(MEAN, STD)]))
    else:  # 64px 模型
        tfs.append(transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(MEAN, STD)]))
        tfs.append(transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(), transforms.Normalize(MEAN, STD)]))
    return tfs


# ────────────────────────────── 推理 ──────────────────────────────
@torch.no_grad()
def logits_tta(model, tfs, root, list_file, batch_size=32):
    """对每个 TTA 变体推理,平均 logits"""
    model.eval()
    acc = None
    for tf in tfs:
        ds = TestSet(root, list_file, tf)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
        outs = []
        for images, _ in loader:
            outs.append(model(images.to(DEVICE)).cpu())
        logits = torch.cat(outs)
        acc = logits if acc is None else acc + logits
    return acc / len(tfs)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="/home/room115/jhy/a1_data",
                    help="数据根目录(含 test/ 和 test_list.txt)")
    ap.add_argument("--out", type=str, default="submission_final.csv")
    args = ap.parse_args()

    test_root = os.path.join(args.data, "test")
    list_file = os.path.join(args.data, "test_list.txt")
    train_root = os.path.join(args.data, "train")

    # 类别映射(train 目录的文件夹名即 ID)
    class_names = sorted(os.listdir(train_root))

    # 模型配置:(checkpoint, TTA transforms, 权重)
    members = [
        ("best_mutual.pth", tta_transforms(), 0.10),
        ("best_adam.pth", tta_transforms(), 0.20),
        ("best_student.pth", tta_transforms(), 0.25),
        ("best_final.pth", tta_transforms(128), 0.45),
    ]

    final_logits = None
    for ckpt, tfs, weight in members:
        model = WideStudent(num_classes=len(class_names)).to(DEVICE)
        state = torch.load(ckpt, map_location=DEVICE, weights_only=False)["model"]
        model.load_state_dict(state)
        print(f"[推理] {ckpt} (权重 {weight:.0%})")
        logits = logits_tta(model, tfs, test_root, list_file)
        del model
        torch.cuda.empty_cache()
        final_logits = logits * weight if final_logits is None else final_logits + logits * weight

    preds = final_logits.argmax(dim=1)

    # 按 test_list.txt 顺序生成提交
    with open(list_file) as f:
        names = [l.strip() for l in f if l.strip()]
    rows = [(name, class_names[p]) for name, p in zip(names, preds.tolist())]
    sub = pd.DataFrame(rows, columns=["id", "label"])
    sub.to_csv(args.out, index=False)
    print(f"[完成] {args.out} 共 {len(sub)} 行")


if __name__ == "__main__":
    main()
