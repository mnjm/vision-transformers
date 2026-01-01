# Ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
import torch
from torch import nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
from .vit import MLP, PatchEmbed, StochDepthDrop, _configure_optimizer
from dataclasses import dataclass, field

@dataclass
class SwinTransformerConfig:
    name: str = "Swin-T"
    img_size: int = 224
    patch_size: int = 4
    patch_norm: bool = True
    img_chls: int = 3
    n_class: int = 1000
    n_embed: int = 96
    depths: list = field(default_factory=lambda: [2, 2, 6, 2])
    n_heads: list = field(default_factory=lambda: [3, 6, 12, 24])
    window_size: int = 7
    mlp_ratio: float = 4.
    drop_rate: float = 0.
    attn_drop_rate: float = 0.
    stoch_depth_drop_rate: float = 0.

def to_2tuple(x):
    return x if isinstance(x, (list, tuple)) else (x, x)

def window_partition(x, window_size):
    """ Partition input feature maps into non-overlapping windows

    Args:
        x (Tensor): input feature maps of shape (B, H, W, C)
        window_size (init or tuple): window size

    Returns:
        Tensor: Tensor of shape (B * num_windows, Wh, Ww, C)
    """
    B, H, W, C = x.shape
    wsh, wsw = to_2tuple(window_size)
    nh, nw = H // wsh, W // wsw
    windows = einops.rearrange(
        x, "b (nh ws1) (nw ws2) c -> (b nh nw) ws1 ws2 c",
        ws1=wsh, ws2=wsw, nh=nh, nw=nw
    )
    return windows

def window_reverse(windows, window_size, H, W):
    """ Reverses the window partition to restruct the feature map

    Args:
        windows (Tensor): shape (B * num_window, Wh, Ww, C)
        window_size (int or tuple): (Wh, Wc)
        H (int): Height of the output feature
        W (int): Width of the output feature

    Returns:
        Tensor: reconstructed feature map
    """
    wsh, wsw = to_2tuple(window_size)
    nh, nw = H // wsh, W // wsw
    B = windows.shape[0] // (nh * nw)
    x = einops.rearrange(
        windows, "(b nh nw) ws1 ws2 c -> b (nh ws1) (nw ws2) c",
        ws1=wsh, ws2=wsw, nh=nh, nw=nw, b=B
    )
    return x

class ShiftedWindowMHSA(nn.Module):

    def __init__(self, dim, window_size, n_heads, attn_drop_rate=.0, proj_drop_rate=.0):
        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)
        self.n_heads = n_heads
        assert dim % n_heads == 0
        self.head_dim = dim // n_heads

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), n_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1 # 2d coords to 1d
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.split_qkv = Rearrange("b n (t h d) -> t b h n d", t=3, h=n_heads, d=self.head_dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.merge_qkv = Rearrange("b h n d -> b n (h d)")

        self.attn_drop_rate = attn_drop_rate
        self.scale = self.head_dim ** -0.5
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B, N, C = x.shape

        tbl = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ]
        rel_pos_bias = tbl.view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1
        ).permute(2, 0, 1).unsqueeze(0)

        # mask expected as (num_windows_total, N, N) and num_windows_total = batch_size * windows_per_img
        shift_mask = None
        if mask is not None:
            windows_per_img = B // mask.shape[0]
            assert B % mask.shape[0] == 0
            shift_mask = einops.repeat(mask, "nw h w -> (repeats nw) 1 h w", repeats=windows_per_img)

        q, k, v = self.split_qkv(self.qkv(x))

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn + rel_pos_bias
        if shift_mask is not None:
            attn = attn + shift_mask
        attn = attn.softmax(dim=-1)
        attn = F.dropout(attn, self.attn_drop_rate, training=self.training)
        x = attn @ v

        x = self.merge_qkv(x)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, input_res, n_heads, window_size, shift_size, mlp_ratio, attn_drop_rate,
                 proj_drop_rate, path_drop_rate, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_res = to_2tuple(input_res)
        self.n_heads = n_heads
        self.window_size = to_2tuple(window_size)
        self.shift_size = shift_size

        # if window size is larger than input resolution in either, do not partition windows
        if self.input_res[0] <= self.window_size[0] or self.input_res[1] <= self.window_size[1]:
            self.shift_size = 0
            self.window_size = self.input_res

        assert 0 <= self.shift_size <= min(self.window_size), "shift size should be within 0 - window_size"

        self.attn_norm = norm_layer(dim)
        self.attn = ShiftedWindowMHSA(
            dim=self.dim, window_size=self.window_size, n_heads=self.n_heads,
            attn_drop_rate=attn_drop_rate, proj_drop_rate=proj_drop_rate
        )

        self.drop_path = StochDepthDrop(drop_prob=path_drop_rate)

        self.mlp_norm = norm_layer(dim)
        self.mlp = MLP(in_features=self.dim, hidden_features=int(self.dim * mlp_ratio), out_features=self.dim, act_fn=act_layer, drop_rate=proj_drop_rate)


        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_res
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_res
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.attn_norm(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1,2))
        else:
            shifted_x = x
        x_windowed = window_partition(shifted_x, self.window_size) # nW * B, wsh, wsw, C

        x_windowed = x_windowed.view(-1, self.window_size[0] * self.window_size[1], C)

        attn = self.attn(x_windowed, mask=self.attn_mask) # nW * B, wsh * wsw, C

        attn = attn.view(-1, self.window_size[0], self.window_size[1], C)

        shifted_x = window_reverse(attn, self.window_size, H, W) # B, n_h_win * wsh, n_w_win * wsw, C
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1,2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.mlp_norm(x)))

        return x

class PatchMerge(nn.Module):

    def __init__(self, dim, input_res, norm_lyr=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_res = input_res
        H, W = self.input_res
        assert H % 2 == 0 and W % 2 == 0, f"input size is not even {H}x{W}"
        self.re = Rearrange('b (h ph) (w pw) c -> b (h w) (pw ph c)', ph=2, pw=2)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_lyr(4 * dim)

    def forward(self, x):
        H, W = self.input_res
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = self.re(x)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class SwinLayer(nn.Module):

    def __init__(self, dim, input_res, depth, n_heads, window_size, mlp_ratio, proj_drop_rate,
                 attn_drop_rate, path_drop_rate, norm_layer=nn.LayerNorm, downsample=False):
        super().__init__()
        self.dim = dim
        self.input_res = input_res
        self.depth = depth
        self.blks = nn.ModuleList(
            SwinTransformerBlock(
                dim=dim, input_res=input_res, n_heads=n_heads, window_size=window_size,
                shift_size= 0 if (l % 2 == 0) else window_size // 2, mlp_ratio=mlp_ratio,
                proj_drop_rate=proj_drop_rate, attn_drop_rate=attn_drop_rate,
                path_drop_rate=path_drop_rate[l] if isinstance(path_drop_rate, list) else path_drop_rate,
                norm_layer=norm_layer
            )
            for l in range(depth)
        )

        if downsample:
            self.downsample = PatchMerge(dim=dim, input_res=input_res, norm_lyr=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blks:
            x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x

class SwinTransformer(nn.Module):

    def __init__(self, cfg: SwinTransformerConfig):
        super().__init__()

        self.cfg = cfg

        self.patch_embed = PatchEmbed(
            img_size=cfg.img_size, patch_size=cfg.patch_size, in_dim=cfg.img_chls,
            out_dim=cfg.n_embed, norm_lyr=nn.LayerNorm if cfg.patch_norm else None
        )
        self.n_patches = (cfg.img_size // cfg.patch_size) ** 2
        patches_res = cfg.img_size // cfg.patch_size, cfg.img_size // cfg.patch_size

        stoch_depth_drop_rates = [x.item() for x in torch.linspace(0, cfg.stoch_depth_drop_rate, sum(cfg.depths))]

        self.n_layers = len(cfg.depths)

        self.layers = nn.ModuleList(
            SwinLayer(
                dim=int(cfg.n_embed * 2 ** l),
                input_res=(patches_res[0] // (2 ** l), patches_res[1] // (2 ** l)),
                depth=cfg.depths[l], n_heads=cfg.n_heads[l], window_size=cfg.window_size,
                mlp_ratio=cfg.mlp_ratio, proj_drop_rate=cfg.drop_rate,
                attn_drop_rate=cfg.attn_drop_rate, downsample=(l < self.n_layers - 1),
                path_drop_rate=stoch_depth_drop_rates[sum(cfg.depths[:l]):sum(cfg.depths[:l+1])]
            )
            for l in range(self.n_layers)
        )

        self.n_features = int(cfg.n_embed * 2 ** (self.n_layers - 1))
        self.norm = nn.LayerNorm(self.n_features)
        self.head = nn.Linear(self.n_features, cfg.n_class) if cfg.n_class > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def loss_fn(self, x, y, weight=None, label_smoothing=.0):
        return F.cross_entropy(x, y, weight=weight, label_smoothing=label_smoothing)

    def forward(self, x, y=None):
        x = self.patch_embed(x)

        for lyr in self.layers:
            x = lyr(x)

        x = self.norm(x) # B, L, C
        x = x.mean(dim=1)
        x = self.head(x) # B, n_class
        if y is not None:
            loss = self.loss_fn(x, y)
            return x, loss
        return x

    def configure_optimizer(self, optim_cfg, device):
        return _configure_optimizer(self, optim_cfg, device)