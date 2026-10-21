"""
AIST4010 A1 · 最终集成提交脚本(独立版)
==========================================

最终方案:四模型 TTA 加权集成(val 86.40%)

模型成员与权重:
  - mutual(相互蒸馏 SGD+Adam):10%
  - adam(Adam 优化器版)       :20%
  - dist64(自蒸馏 64px)       :25%
  - dist128(二阶段蒸馏 128px) :45%

每个模型对 test 集做 TTA(原图 + 水平翻转)取平均 logits,
再按权重加权求和,argmax 得到预测。

依赖:4 个 checkpoint + test 数据 + test_list.txt
输出:submission_final.csv(可直接提交 Kaggle)
"""

import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

MEAN, STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────── 模型定义(WideStudent) ──────────────────────────────
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
    """加宽学生网络(ResNet 风格骨干):64→128→256→512 通道。"""

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


# ────────────────────────────── 数据集 ──────────────────────────────
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


def tta_transforms(base_size=None):
    """TTA 变换对:原图 + 水平翻转(128px 模型先 Resize)。"""
    transforms_list = []
    if base_size:  # 128px 模型
        transforms_list.append(transforms.Compose([
            transforms.Resize((base_size, base_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]))
        transforms_list.append(transforms.Compose([
            transforms.Resize((base_size, base_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]))
    else:  # 64px 模型
        transforms_list.append(transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]))
        transforms_list.append(transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]))
    return transforms_list


# ────────────────────────────── 推理 ──────────────────────────────
@torch.no_grad()
def logits_tta(model, transforms_list, root, list_file, batch_size=32):
    """对每个 TTA 变体推理一遍,取平均 logits。"""
    model.eval()
    averaged_logits = None
    for transform in transforms_list:
        dataset = TestSet(root, list_file, transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        batch_outputs = []
        for images, _ in loader:
            batch_outputs.append(model(images.to(DEVICE)).cpu())
        logits = torch.cat(batch_outputs)
        averaged_logits = logits if averaged_logits is None else averaged_logits + logits
    return averaged_logits / len(transforms_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/home/room115/jhy/a1_data",
                        help="数据根目录(含 test/ 和 test_list.txt)")
    parser.add_argument("--out", type=str, default="submission_final.csv")
    args = parser.parse_args()

    test_root = os.path.join(args.data, "test")
    list_file = os.path.join(args.data, "test_list.txt")
    train_root = os.path.join(args.data, "train")

    # 类别映射:train 目录的文件夹名即类别 ID
    class_names = sorted(os.listdir(train_root))

    # 集成成员:(checkpoint, TTA 变换, 权重)
    members = [
        ("best_mutual.pth", tta_transforms(), 0.10),
        ("best_adam.pth", tta_transforms(), 0.20),
        ("best_student.pth", tta_transforms(), 0.25),
        ("best_final.pth", tta_transforms(128), 0.45),
    ]

    final_logits = None
    for checkpoint_path, transforms_list, weight in members:
        model = WideStudent(num_classes=len(class_names)).to(DEVICE)
        state_dict = torch.load(checkpoint_path, map_location=DEVICE,
                                weights_only=False)["model"]
        model.load_state_dict(state_dict)
        print(f"[推理] {checkpoint_path} (权重 {weight:.0%})")
        logits = logits_tta(model, transforms_list, test_root, list_file)
        del model
        torch.cuda.empty_cache()
        final_logits = (logits * weight if final_logits is None
                        else final_logits + logits * weight)

    predictions = final_logits.argmax(dim=1)

    # 按 test_list.txt 的顺序生成提交文件
    with open(list_file) as f:
        names = [line.strip() for line in f if line.strip()]
    rows = [(name, class_names[pred]) for name, pred in zip(names, predictions.tolist())]
    submission = pd.DataFrame(rows, columns=["id", "label"])
    submission.to_csv(args.out, index=False)
    print(f"[完成] {args.out} 共 {len(submission)} 行")


if __name__ == "__main__":
    main()
