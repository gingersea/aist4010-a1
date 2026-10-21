"""
A1 人脸分类 · 自蒸馏工具:生成教师软标签
========================================

在 train 集上跑一遍教师模型,把 logits 存成 fp16 tensor,
供后续学生训练(train_student.py 等)按样本索引查表使用,
避免训练时每个 batch 都跑一次教师前向(提速约 4 倍)。

教师:best.pth(清晰版)+ best_blur.pth(模糊鲁棒版)两个模型,
取两者 logits 的平均作为软标签。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train_resnet import MiniResNet

# ────────────────────────────── 基础设置 ──────────────────────────────
MEAN, STD = [0.5] * 3, [0.5] * 3
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

device = torch.device("cuda")

train_dataset = datasets.ImageFolder("/home/room115/jhy/a1_data/train", transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=8)

val_dataset = datasets.ImageFolder("/home/room115/jhy/a1_data/val", transform)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=8)

# ────────────────────────────── 加载两个教师 ──────────────────────────────
teachers = []
for path in ["best.pth", "best_blur.pth"]:
    model = MiniResNet(1000).to(device).eval()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    teachers.append(model)
print("已加载教师: best.pth(清晰) + best_blur.pth(模糊鲁棒)")

# 教师集成的 val 准确率(参考值)
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        logits = sum(model(images) for model in teachers) / len(teachers)
        correct += (logits.argmax(1).cpu() == labels).sum().item()
        total += labels.size(0)
print(f"教师集成 val 准确率: {correct/total:.4f}")

# ───────────── 生成 train 集软标签(两个教师 logits 平均) ─────────────
all_logits = []
with torch.no_grad():
    for images, _ in train_loader:
        images = images.to(device)
        logits = sum(model(images) for model in teachers) / len(teachers)
        all_logits.append(logits.half().cpu())
teacher_logits = torch.cat(all_logits)  # (N, 1000) fp16
torch.save(teacher_logits, "teacher_logits_fp16.pt")
print(f"教师软标签已保存: {tuple(teacher_logits.shape)} "
      f"(fp16, {teacher_logits.numel()*2/1e6:.0f} MB)")
