# Vision Transformers (ViT)

A minimal PyTorch implementation of [Vision Transformers(ViT)](https://arxiv.org/pdf/2010.11929) and its varient [Data efficient Image Transformers (DeiT)](https://arxiv.org/pdf/2012.12877). Experimented with CIFAR-100 and Tiny-Imagenet datasets with a small ViT-T/8 and DeiT-T/8 (Tiny with patch size of 8) because of compute reasons, but supports other varients as well. Uses hydra for config and wandb for optional logging. Sidenote: There is also a minimal implemention of [Stochastic Depth Regularizer](https://arxiv.org/pdf/1603.09382) that is commonly used as ViTs as regulizer.

## CIFAR-100 Training Plots

![CIFAR-100 Train Plots](https://raw.githubusercontent.com/mnjm/vision-transformers/refs/heads/assets/train-plots.png)

## Setup

- Install [uv](https://docs.astral.sh/uv/) and run
```bash
uv sync
```

## Training Runs

### Train ViT-T/8 on CIFAR-100

```bash
uv run train.py +run=vit-cifar100
```

### Train DeiT-T/8 on CIFAR-100

```bash
uv run train.py +run=deit-cifar100
```
Uses frozen `resnet18_cifar100` (via timm) as Teacher and is used for hard distillation (as it is showen to work well in DeiT paper)

### Train ViT-T/8 on Tiny-Imagenet

```bash
uv run train.py
```