import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops.layers.torch import Rearrange

@dataclass
class ViTConfig:
    name: str = "ViT-B-16"
    img_size: int = 224
    patch_size: int = 16
    img_chls: int = 3
    n_class: int = 1_000
    n_layer: int = 12
    n_heads: int = 12
    n_embd: int = 768
    mlp_dim: int = 3_072
    drop_rate: float = 0.
    attn_drop_rate: float = 0.
    stoch_depth_drop_rate: float = 0.

    @property
    def num_patches(self) -> int:
        return (self.img_size // self.patch_size) ** 2

class PatchEmbed(nn.Module):

    def __init__(self, cfg: ViTConfig):
        super().__init__()
        assert cfg.img_size%cfg.patch_size==0
        patch_dim = cfg.patch_size * cfg.patch_size * cfg.img_chls
        self.to_patches = Rearrange(
            "b c (h ph) (w pw) -> b (h w) (c ph pw)",
            ph=cfg.patch_size, pw=cfg.patch_size
        )  # (B, num_patches, patch_dim)
        self.proj = nn.Linear(patch_dim, cfg.n_embd, bias=True)

    def forward(self, x):
        x = self.to_patches(x)
        x = self.proj(x)
        return x

class MLP(nn.Module):

    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd, cfg.mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(cfg.mlp_dim, cfg.n_embd)
        self.drop = nn.Dropout(cfg.drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, cfg: ViTConfig, enable_flash_attn=True):
        super().__init__()
        assert cfg.n_embd % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        head_dim = cfg.n_embd // cfg.n_heads
        self.flash_attn = enable_flash_attn and hasattr(F, 'scaled_dot_product_attention')
        if self.flash_attn:
            self.attn_drop_rate = cfg.attn_drop_rate
        else:
            self.scale = head_dim ** -0.5
            self.attn_drop = nn.Dropout(cfg.attn_drop_rate)
        self.qkv = nn.Linear(cfg.n_embd, cfg.n_embd * 3)
        self.re_qkv = Rearrange("b n (t h d) -> t b h n d", t=3, h=cfg.n_heads, d=head_dim)
        self.re_merge = Rearrange("b h n d -> b n (h d)")
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.proj_drop = nn.Dropout(cfg.drop_rate)

    def forward(self, x):
        q, k, v = self.re_qkv(self.qkv(x))
        if self.flash_attn:
            x = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=self.attn_drop_rate)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = self.re_merge(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class StochDepthDrop(nn.Module):
    """
    Stoch. Depth Paper: https://arxiv.org/pdf/1603.09382
    DeIT uses it: https://arxiv.org/pdf/2012.12877
    """
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        drop_prob = self.drop_prob
        if drop_prob == 0. or not self.training:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random = torch.rand(shape, dtype=x.dtype, device=x.device)
        mask = (random > drop_prob).float()
        out = x * mask / (1. - drop_prob)
        return out

class Block(nn.Module):

    def __init__(self, cfg: ViTConfig, drop_path_prob: float, enable_flash_attn=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.n_embd)
        self.attn = MultiHeadSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)
        self.drop_path_attn = StochDepthDrop(drop_path_prob)
        self.drop_path_mlp = StochDepthDrop(drop_path_prob)

    def forward(self, x):
        x = x + self.drop_path_attn(self.attn(self.norm1(x))) # residual with stocastic drop path regularizer
        x = x + self.drop_path_mlp(self.mlp(self.norm2(x))) # residual with stocastic drop path regularizer
        return x

class VisionTransformer(nn.Module):

    def __init__(self, cfg: ViTConfig, enable_flash_attn=True):
        super().__init__()
        self.cfg = cfg
        L = cfg.n_layer
        self.patch_embed = PatchEmbed(cfg)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.n_embd))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + cfg.num_patches, cfg.n_embd))
        self.pos_drop = nn.Dropout(cfg.drop_rate)
        self.blocks = nn.ModuleList(
            Block(
                cfg,
                (l + 1) / L * cfg.stoch_depth_drop_rate,
                enable_flash_attn=enable_flash_attn,
            )
            for l in range(L) # noqa: E741
        )
        self.norm = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.n_class)

        # Init weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                         # (B,num_patches,n_embd)
        cls_tokens = self.cls_token.expand(B, -1, -1)   # (B,1,n_embd)
        x = torch.cat((cls_tokens, x), dim=1)           # prepend [CLS]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0] # take [CLS]
        return self.head(cls_out) # (B,num_classes)

if __name__ == "__main__":
    from torchvision import models
    from omegaconf import OmegaConf
    from pathlib import Path

    cfg = ViTConfig()
    my_model = VisionTransformer(cfg)
    dummy = torch.randn(2, 3, cfg.img_size, cfg.img_size)
    out = my_model(dummy)
    assert out.shape == (2, 1000), f"Output mismatch {out.shape}"

    # Test all available vit models from `torchvision.models`
    model_map = {
        'ViT-B-16': models.vit_b_16,
        'ViT-B-32': models.vit_b_32,
        'ViT-L-16': models.vit_l_16,
        'ViT-L-32': models.vit_l_32,
        'ViT-H-14': models.vit_h_14,
    }
    yml_files = Path("./config/model").glob("*.yaml")
    for yml_file in yml_files:
        cfg = OmegaConf.load(yml_file)
        cfg.img_size = 224
        cfg.img_chls = 3
        cfg.n_class = 1000
        name = cfg.name
        kwargs = dict(cfg)
        if name not in model_map:
            continue
        pt_model = model_map[name]()
        pt_params = sum(p.numel() for p in pt_model.parameters())
        my_model = VisionTransformer(ViTConfig(**kwargs))
        my_params = sum(p.numel() for p in my_model.parameters())
        assert pt_params == my_params, f"{name} params mismatch {pt_params=} {my_params=}"

    print("="*20)
    print("Success")
    print("="*20)