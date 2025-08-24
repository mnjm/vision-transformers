# Vision Transformers (ViT)

A minimal PyTorch implementation of [Vision Transformer](https://arxiv.org/pdf/2010.11929) for image classification. Experimented with Tiny-ImageNet dataset on ViT-T/8 (Tiny with patch size of 8) because of compute reasons, but supports other varients as well with matching param counts from ViT models available in `torchvision.models.vision_transformers`. Uses hydra for config and wandb for logging. Sidenote: There is also a minimal implemention of [Stochastic Depth Regularizer](https://arxiv.org/pdf/1603.09382) that is used in ViT papers like [DeIT](https://arxiv.org/pdf/2012.12877).

![Train Plots](https://raw.githubusercontent.com/mnjm/vision-transformers/refs/heads/assets/train-plots.png)

## Setup

Dependencies
```bash
pip install -r requirements.txt
```

Training
```bash
python train.py
```
or
```bash
python train.py +device=auto # if cuda is not available
```
