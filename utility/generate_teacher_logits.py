"""
A1 人脸分类 · 自我蒸馏
第一步:生成教师软标签
=======================
用已训练的两个教师(best.pth 清晰版 + best_blur.pth 模糊鲁棒版)
在 train 集上跑一遍,取两个教师 logits 的平均,存成 fp16 tensor。
学生训练时,对每张图同时提供:硬标签 + 教师软标签。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from train_resnet import MiniResNet

MEAN, STD = [0.5] * 3, [0.5] * 3
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

device = torch.device("cuda")

train_ds = datasets.ImageFolder("/home/room115/jhy/a1_data/train", tf)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=8)

val_ds = datasets.ImageFolder("/home/room115/jhy/a1_data/val", tf)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=8)

# 加载两个教师
teachers = []
for p in ["best.pth", "best_blur.pth"]:
    m = MiniResNet(1000).to(device).eval()
    m.load_state_dict(torch.load(p, map_location=device, weights_only=False)["model"])
    teachers.append(m)
print("已加载教师: best.pth(清晰) + best_blur.pth(模糊鲁棒)")

# 教师集成的 val 准确率(参考)
correct = total = 0
with torch.no_grad():
    for x, y in val_loader:
        x = x.to(device)
        logits = sum(m(x) for m in teachers) / len(teachers)
        correct += (logits.argmax(1).cpu() == y).sum().item()
        total += y.size(0)
print(f"教师集成 val 准确率: {correct/total:.4f}")

# 生成 train 集软标签(两个教师 logits 平均)
all_logits = []
with torch.no_grad():
    for x, y in train_loader:
        x = x.to(device)
        logits = sum(m(x) for m in teachers) / len(teachers)
        all_logits.append(logits.half().cpu())
teacher_logits = torch.cat(all_logits)   # (93134, 1000) fp16
torch.save(teacher_logits, "teacher_logits_fp16.pt")
print(f"教师软标签已保存: {tuple(teacher_logits.shape)} (fp16, {teacher_logits.numel()*2/1e6:.0f} MB)")
