# AIST4010 A1 — Face Classification (1000 classes)

Final result: **val 86.40%** (4-model TTA weighted ensemble).

This repo contains only the code needed to reproduce the final submission, plus the Word report.

## Files

```
├── A1实验报告.docx              # 实验报告 (Word, 中文)
├── ensemble_final.py            # 最终集成提交脚本 (4模型 TTA 加权集成)
├── train/
│   ├── train_resnet.py          # ① 基线 ResNet (教师1: 清晰版)
│   ├── train_blur.py            # ② 模糊增强版 (教师2: 模糊鲁棒)
│   ├── train_student.py         # ③ 一阶段自蒸馏 → dist64
│   ├── train_final.py           # ④ 二阶段自蒸馏 128px → dist128
│   ├── train_adam.py            # ⑤ AdamW 版 (集成多样性)
│   └── train_mutual.py          # ⑥ 相互蒸馏 (集成多样性)
└── utility/
    └── generate_teacher_logits.py  # 教师软标签预生成 (蒸馏管线)
```

## Pipeline

```
train_resnet.py ─┐
                 ├→ generate_teacher_logits.py → teacher_logits_fp16.pt
train_blur.py  ──┘        │
                          ▼
                 train_student.py → best_student.pth (dist64, 83.5%)
                          │  ┌→ teacher_logits_student_fp16.pt
                          ▼  ▼
                 train_final.py (128px) → best_final.pth (dist128, 83.9%)
                 train_adam.py → best_adam.pth (82.0%)
                 train_mutual.py → best_mutual.pth (82.65%)
                          │
                          ▼
                 ensemble_final.py → submission_final.csv (86.40%)
```

## Reproduce

```bash
# 0. 数据目录需含: train/ val/ test/ test_list.txt (默认 /home/room115/jhy/a1_data, 用 --data 修改)

# 1. 两个教师
python train/train_resnet.py --epochs 60 --batch 64
python train/train_blur.py   --epochs 80 --batch 64

# 2. 教师软标签
python utility/generate_teacher_logits.py

# 3. 一阶段蒸馏 (dist64)
python train/train_student.py --epochs 80 --batch 64

# 4. 用 dist64 生成第二版软标签 (教师路径改为 best_student.pth, 输出改存 teacher_logits_student_fp16.pt)

# 5. 二阶段蒸馏 128px (dist128)
python train/train_final.py --epochs 100 --batch 16 --accum 4

# 6. 多样性成员
python train/train_adam.py --fold 0 --sample-frac 1.0
python train/train_mutual.py --epochs 80 --batch 64

# 7. 集成提交 (4个 checkpoint 需放当前目录)
python ensemble_final.py --data <数据目录> --out submission_final.csv
```

### GPU 环境注意事项

服务器驱动 595.84/595.99 版本错配导致 NVML 崩溃, 训练前需设置:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_NVML_BASED_CUDA_CALL=0
```

64px 训练 batch≤64; 128px 训练 batch=16 + 梯度累积 4 步 (显存峰值 < 2GB)。

### 依赖

torch, torchvision, numpy, pandas, PIL

### 集成成员与权重

| 模型 | checkpoint | val | 权重 |
|---|---|---|---|
| mutual | best_mutual.pth | 82.65% | 10% |
| adam | best_adam.pth | 82.00% | 20% |
| dist64 | best_student.pth | 83.50% | 25% |
| dist128 | best_final.pth | 83.90% | 45% |
