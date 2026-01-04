from model.swin import SwinTransformer, SwinTransformerConfig
from torchvision.models.swin_transformer import swin_b, swin_s, swin_t
from pathlib import Path
from omegaconf import OmegaConf
import pytest
import torch
import torch.nn as nn

model_map = {
    'Swin-B': swin_b,
    'Swin-T': swin_t,
    'Swin-S': swin_s,
}
yml_files = [ Path("./config/model") / f"{name}.yaml" for name in model_map.keys() ]

def test_swin_output_format():
    cfg = SwinTransformerConfig()
    model = SwinTransformer(cfg)

    dummy = torch.randn(2, 3, cfg.img_size, cfg.img_size)
    out = model(dummy)
    assert out.shape == (2, 1000), f"Output mismatch {out.shape}"

def test_swin_param_match():
    # Test all available Swin models from `torchvision.models`
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
        my_model = SwinTransformer(SwinTransformerConfig(**kwargs))
        my_params = sum(p.numel() for p in my_model.parameters())
        assert pt_params == my_params, f"{name} params mismatch {pt_params=} {my_params=}"

# There are architectural differences between my implementation and torchvision, so I decided not to test this.
# def load_weights_from_tv(my_model, tv_model):
#     my_model.patch_embed.proj.weight.copy_(tv_model.features[0][0].weight)
#     my_model.patch_embed.proj.bias.copy_(tv_model.features[0][0].bias)

#     my_model.patch_embed.norm.weight.copy_(tv_model.features[0][2].weight)
#     my_model.patch_embed.norm.bias.copy_(tv_model.features[0][2].bias)

#     tv_layer_indices = [1, 3, 5, 7]
#     tv_merging_indices = [2, 4, 6]

#     for stage_idx in range(my_model.n_layers):

#         tv_blks_seq = tv_model.features[tv_layer_indices[stage_idx]]

#         assert isinstance(tv_blks_seq, nn.Sequential) and len(tv_blks_seq) == my_model.cfg.depths[stage_idx]
#         for block_idx, (my_blk, tv_blk) in enumerate(zip(my_model.layers[stage_idx].blks, tv_blks_seq)):
#             my_blk.attn_norm.weight.copy_(tv_blk.norm1.weight)
#             my_blk.attn_norm.bias.copy_(tv_blk.norm1.bias)
#             my_blk.attn.qkv.weight.copy_(tv_blk.attn.qkv.weight)
#             my_blk.attn.qkv.bias.copy_(tv_blk.attn.qkv.bias)
#             my_blk.attn.proj.weight.copy_(tv_blk.attn.proj.weight)
#             my_blk.attn.proj.bias.copy_(tv_blk.attn.proj.bias)
#             my_blk.attn.relative_position_bias_table.copy_(tv_blk.attn.relative_position_bias_table)
#             my_blk.mlp_norm.weight.copy_(tv_blk.norm2.weight)
#             my_blk.mlp_norm.bias.copy_(tv_blk.norm2.bias)
#             my_blk.mlp.fc1.weight.copy_(tv_blk.mlp[0].weight)
#             my_blk.mlp.fc1.bias.copy_(tv_blk.mlp[0].bias)
#             my_blk.mlp.fc2.weight.copy_(tv_blk.mlp[3].weight)
#             my_blk.mlp.fc2.bias.copy_(tv_blk.mlp[3].bias)

#         if stage_idx < my_model.n_layers - 1:
#             tv_merge_idx = tv_merging_indices[stage_idx]
#             tv_merge_module = tv_model.features[tv_merge_idx]

#             my_model.layers[stage_idx].downsample.norm.weight.copy_(tv_merge_module.norm.weight)
#             my_model.layers[stage_idx].downsample.norm.bias.copy_(tv_merge_module.norm.bias)
#             my_model.layers[stage_idx].downsample.reduction.weight.copy_(tv_merge_module.reduction.weight)

#     my_model.norm.weight.copy_(tv_model.norm.weight)
#     my_model.norm.bias.copy_(tv_model.norm.bias)

#     my_model.head.weight.copy_(tv_model.head.weight)
#     my_model.head.bias.copy_(tv_model.head.bias)

# @torch.no_grad()
# def test_bit_match_output_with_tv():
#     for yml_file in yml_files:
#         cfg = OmegaConf.load(yml_file)
#         cfg.img_size = 224
#         cfg.img_chls = 3
#         cfg.n_class = 1000
#         name = cfg.name
#         kwargs = dict(cfg)
#         if name not in model_map:
#             continue
#         tv_model = model_map[name]()
#         my_model = SwinTransformer(SwinTransformerConfig(**kwargs))
#         tv_model.eval()
#         my_model.eval()
#         load_weights_from_tv(my_model, tv_model)

#         x = torch.randn(2, 3, 224, 224)

#         y_tv = tv_model(x)
#         y_my = my_model(x)
#         max_diff = (y_my - y_tv).abs().max().item()
#         assert torch.allclose(y_tv, y_my, atol=1e-5, rtol=1e-5), f"Not bit-matching! {max_diff}"

@torch.no_grad()
def test_swin_flex_attn():
    for yml_file in yml_files:
        cfg = OmegaConf.load(yml_file)
        cfg.img_size = 224
        cfg.img_chls = 3
        cfg.n_class = 1000
        name = cfg.name
        kwargs = dict(cfg)
        if name not in model_map:
            continue
        model = SwinTransformer(SwinTransformerConfig(**kwargs))
        flex_model = SwinTransformer(SwinTransformerConfig(**kwargs), use_flex_attn=True)
        model.eval()
        flex_model.eval()
        flex_model.load_state_dict(model.state_dict())

        x = torch.randn(2, 3, 224, 224)
        y = model(x)
        y_flex = flex_model(x)
        max_diff = (y_flex - y).abs().max().item()
        assert torch.allclose(y, y_flex, atol=1e-5, rtol=1e-5), f"Not bit-matching! {max_diff}"