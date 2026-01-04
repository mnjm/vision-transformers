import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops.layers.torch import Rearrange
from hydra.utils import instantiate

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

    def __init__(self, img_size, patch_size, in_dim, out_dim, norm_lyr=nn.Identity):
        super().__init__()
        assert img_size % patch_size == 0, f"{patch_size=} must evenly divide {img_size=}"
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.re = Rearrange("b c h w -> b (h w) c")
        self.norm = norm_lyr(out_dim)

    def forward(self, x):
        x = self.proj(x) # (B, n_embd, H/patch_size, W/patch_size)
        x = self.re(x) # (B, n_patches, n_embed)
        x = self.norm(x)
        return x

class MLP(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, act_fn=nn.GELU, drop_rate=.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_fn()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, dim, n_heads, attn_drop_rate=.0, proj_drop_rate=.0, enable_flash_attn=False):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        head_dim = dim // n_heads
        self.flash_attn = enable_flash_attn and hasattr(F, 'scaled_dot_product_attention')
        self.attn_drop_rate = attn_drop_rate
        self.scale = head_dim ** -0.5
        if not self.flash_attn:
            self.attn_drop = nn.Dropout(attn_drop_rate)
        self.qkv = nn.Linear(dim, dim * 3)
        self.split_qkv = Rearrange("b n (t h d) -> t b h n d", t=3, h=n_heads, d=head_dim)
        self.merge_qkv = Rearrange("b h n d -> b n (h d)")
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_rate)

    def forward(self, x):
        q, k, v = self.split_qkv(self.qkv(x))
        if self.flash_attn:
            drop_p = self.attn_drop_rate if self.training else 0.0
            x = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=drop_p)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = self.merge_qkv(x)
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

    def __init__(self, dim, n_heads, mlp_dim, attn_drop_rate, proj_drop_rate, drop_path_prob, enable_flash_attn=True, mlp_drop_rate=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim=dim, n_heads=n_heads, attn_drop_rate=attn_drop_rate, proj_drop_rate=proj_drop_rate, enable_flash_attn=enable_flash_attn)
        self.norm2 = nn.LayerNorm(dim)
        mlp_drop_rate = proj_drop_rate if mlp_drop_rate is None else mlp_drop_rate
        self.mlp = MLP(dim, mlp_dim, dim, drop_rate=mlp_drop_rate)
        self.drop_path_attn = StochDepthDrop(drop_path_prob)
        self.drop_path_mlp = StochDepthDrop(drop_path_prob)

    def forward(self, x):
        x = x + self.drop_path_attn(self.attn(self.norm1(x))) # residual with stocastic drop path regularizer
        x = x + self.drop_path_mlp(self.mlp(self.norm2(x))) # residual with stocastic drop path regularizer
        return x

class ViT(nn.Module):

    def __init__(self, cfg: ViTConfig, enable_flash_attn=False):
        super().__init__()
        self.cfg = cfg
        L = cfg.n_layer
        self.patch_embed = PatchEmbed(cfg.img_size, cfg.patch_size, cfg.img_chls, cfg.n_embd)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.n_embd))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + cfg.num_patches, cfg.n_embd))
        self.pos_drop = nn.Dropout(cfg.drop_rate)
        self.blocks = nn.ModuleList(
            Block(
                dim=cfg.n_embd, n_heads=cfg.n_heads, mlp_dim=cfg.mlp_dim,
                attn_drop_rate=cfg.attn_drop_rate, proj_drop_rate=cfg.drop_rate,
                drop_path_prob=(l + 1) / L * cfg.stoch_depth_drop_rate,
                enable_flash_attn=enable_flash_attn,
            )
            for l in range(L) # noqa: E741
        )
        self.norm = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.n_class)

        # Init weights
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def loss_fn(self, pred, lbls):
        loss = F.cross_entropy(pred, lbls)
        return loss

    def forward(self, imgs, lbls=None):
        B = imgs.shape[0]
        x = self.patch_embed(imgs)                         # (B,num_patches,n_embd)
        cls_tokens = self.cls_token.expand(B, -1, -1)      # (B,1,n_embd)
        x = torch.cat((cls_tokens, x), dim=1)              # prepend [CLS]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0] # take [CLS]
        logits = self.head(cls_out) # (B, num_classes)
        if lbls is not None:
            loss = self.loss_fn(logits, lbls)
            return logits, loss
        return logits

    def configure_optimizer(self, optim_cfg, device):
        return _configure_optimizer(self, optim_cfg, device)

def _configure_optimizer(model, optim_cfg, device):
    optim_cfg.fused = getattr(optim_cfg, 'fused', False) and device.type == "cuda"
    params_dict = { pn: p for pn, p in model.named_parameters() }
    params_dict = { pn:p for pn, p in params_dict.items() if p.requires_grad } # filter params that requires grad
    # create optim groups of any params that is 2D or more. This group will be weight decayed ie weight tensors in Linear and embeddings
    decay_params = [ p for p in params_dict.values() if p.dim() >= 2]
    # create optim groups of any params that is 1D. All biases and layernorm params
    no_decay_params = [ p for p in params_dict.values() if p.dim() < 2]
    weight_decay = getattr(optim_cfg, 'weight_decay', 0.0)
    optim_cfg.weight_decay = .0
    optim_groups = [
        { 'params': decay_params, 'weight_decay': weight_decay },
        { 'params': no_decay_params, 'weight_decay': 0.0 },
    ]
    optimizer = instantiate(optim_cfg, params=optim_groups, _convert_="all")
    return optimizer