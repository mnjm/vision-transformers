"""
Minimal DeiT (Data-efficient Image Transformer) implementation.

Extends ViT with a distillation token
"""

import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from .vit import ViTConfig, ViT

@dataclass
class DeitConfig(ViTConfig):
    """Configuration for DeiT.

    Args:
        use_dist_token: Enable distillation token.
    """
    use_dist_token: bool = True

class DeiT(ViT):
    """DeiT model with distillation token and head."""

    def __init__(self, cfg, use_sdpa_attn=True):
        super().__init__(cfg, use_sdpa_attn)
        assert getattr(cfg, "use_dist_token", False), "DeiT expects use_dist_token=True"
        self.dist_token = nn.Parameter(torch.zeros(1, 1, cfg.n_embd))
        self.head_dist = nn.Linear(cfg.n_embd, cfg.n_class)

        with torch.no_grad():
            # init new params
            nn.init.trunc_normal_(self.dist_token, std=0.02)
            nn.init.trunc_normal_(self.head_dist.weight, std=0.02)
            if self.head_dist.bias is not None:
                nn.init.zeros_(self.head_dist.bias)

            pe = self.pos_embed
            # add place [DIST] token, [CLS] + [DIST] + patches
            self.pos_embed = nn.Parameter(torch.zeros(1, 2 + cfg.num_patches, cfg.n_embd))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            # copy over cls + patches weights in case of adapting ViT to DeiT
            self.pos_embed[:, :1, :].copy_(pe[:, :1, :])
            self.pos_embed[:, 2:, :].copy_(pe[:, 1:, :])

            self.teacher = None

    def set_teacher(self, teacher: nn.Module):
        teacher.requires_grad_(False)
        teacher.eval()
        self.teacher = teacher

    def loss_fn(self, pred, lbls, imgs=None, weight=.5):
        """Deit loss.
         Args:
            pred: Logits or (cls_logits, dist_logits).
            lbls: Ground-truth labels.
            imgs: Input images for teacher.
            weight: Distillation loss weight.
        """
        if isinstance(pred, tuple):
            logits_cls, logits_dist = pred
            if self.teacher is None:
                # use dataset lbls as hard lbls and return avg loss between cls and distil tokens
                loss = .5 * (F.cross_entropy(logits_cls, lbls) + F.cross_entropy(logits_dist, lbls))
            else:
                assert isinstance(imgs, torch.Tensor), "Teacher model is set, pass teacher model input to calculate loss"
                # Uses hard distillation from DeiT as it in paper produced slightly better results.
                with torch.no_grad():
                    self.teacher.eval()
                    t_logits = self.teacher(imgs)
                    t_hard_lbls = torch.argmax(t_logits, dim=1)
                loss_cls = F.cross_entropy(logits_cls, lbls)
                loss_dist = F.cross_entropy(logits_dist, t_hard_lbls)
                loss = (1. - weight) * loss_cls + weight * loss_dist
        else:
            # No distil fallback to simple CE loss
            loss = F.cross_entropy(pred, lbls)
        return loss

    def forward(self, imgs, lbls=None):
        B = imgs.shape[0]

        x = self.patch_embed(imgs)                                   # (B,num_patches,n_embd)
        cls_tokens = self.cls_token.expand(B, -1, -1)                # (B,1,n_embd)
        dist_tokens = self.dist_token.expand(B, -1, -1)              # (B,1,n_embd)
        x = torch.cat((cls_tokens, dist_tokens, x), dim=1)           # prepend [CLS] + [DIST]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0] # take [CLS]
        dist_out = x[:, 1] # take [DIST]
        logits_cls = self.head(cls_out)
        logits_dist = self.head_dist(dist_out)

        logits = (logits_cls + logits_dist) * .5

        if lbls is not None:
            loss = self.loss_fn((logits_cls, logits_dist), lbls, imgs)
            return logits, loss
        return logits
