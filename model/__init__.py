import torch
import detectors  # noqa: F401
import timm
from .vit import ViTConfig, ViT
from .deit import DeitConfig, DeiT

def init_model(cfg):
    if getattr(cfg, 'model', False) and getattr(cfg.model, 'use_dist_token', False):
        model_cfg = DeitConfig(**cfg.model)
        model = DeiT(model_cfg)
    else:
        model_cfg = ViTConfig(**cfg.model)
        model = ViT(model_cfg)
    return model

def init_deit(model, cfg, device, logger):
    assert isinstance(model, DeiT), "Model should be DeiT. Make sure to set 'use_dist_token = true' in model config"
    teacher_name = cfg.deit.teacher_name
    # load teacher model
    teacher = timm.create_model(teacher_name, pretrained=True)
    teacher.to(device)
    rand_img = torch.rand((1, cfg.dataset.img_chls, cfg.dataset.img_size, cfg.dataset.img_size), device=device)
    out = teacher(rand_img)
    assert out.shape == (1, cfg.dataset.n_class), f"Invalid teacher model, model's {out.shape=}"
    # freeze teacher
    teacher.requires_grad_(False)
    teacher.eval()
    model.set_teacher(teacher)
    logger.info(f"Loaded teacher model {teacher_name=}")
    return teacher

__all__ = [
    init_model, init_deit
]