from model.vit import ViTConfig, ViT
import pytest
from torchvision import models
from omegaconf import OmegaConf
from pathlib import Path
import torch

model_map = {
    'ViT-B-16': models.vit_b_16,
    'ViT-B-32': models.vit_b_32,
    'ViT-L-16': models.vit_l_16,
    'ViT-L-32': models.vit_l_32,
    'ViT-H-14': models.vit_h_14,
}
yml_files = [ Path("./config/model") / f"{name}.yaml" for name in model_map.keys() ]

def test_vit_output_format():
    cfg = ViTConfig()
    my_model = ViT(cfg)
    dummy = torch.randn(2, 3, cfg.img_size, cfg.img_size)
    out = my_model(dummy)
    assert out.shape == (2, 1000), f"Output mismatch {out.shape}"

def test_param_count_vit():
    # Test all available vit models from `torchvision.models`
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
        my_model = ViT(ViTConfig(**kwargs))
        my_params = sum(p.numel() for p in my_model.parameters())
        assert pt_params == my_params, f"{name} params mismatch {pt_params=} {my_params=}"

def load_weights_from_tv(my_model, tv_model):
    my_model.patch_embed.proj.weight.copy_(tv_model.conv_proj.weight)
    my_model.patch_embed.proj.bias.copy_(tv_model.conv_proj.bias)

    my_model.cls_token.copy_(tv_model.class_token)
    my_model.pos_embed.copy_(tv_model.encoder.pos_embedding)

    for i, (my_blk, tv_blk) in enumerate(zip(my_model.blocks, tv_model.encoder.layers)):
        my_blk.norm1.weight.copy_(tv_blk.ln_1.weight)
        my_blk.norm1.bias.copy_(tv_blk.ln_1.bias)

        my_blk.attn.qkv.weight.copy_(tv_blk.self_attention.in_proj_weight)
        my_blk.attn.qkv.bias.copy_(tv_blk.self_attention.in_proj_bias)

        my_blk.attn.proj.weight.copy_(tv_blk.self_attention.out_proj.weight)
        my_blk.attn.proj.bias.copy_(tv_blk.self_attention.out_proj.bias)

        my_blk.norm2.weight.copy_(tv_blk.ln_2.weight)
        my_blk.norm2.bias.copy_(tv_blk.ln_2.bias)

        my_blk.mlp.fc1.weight.copy_(tv_blk.mlp[0].weight)
        my_blk.mlp.fc1.bias.copy_(tv_blk.mlp[0].bias)

        my_blk.mlp.fc2.weight.copy_(tv_blk.mlp[3].weight)
        my_blk.mlp.fc2.bias.copy_(tv_blk.mlp[3].bias)

    my_model.norm.weight.copy_(tv_model.encoder.ln.weight)
    my_model.norm.bias.copy_(tv_model.encoder.ln.bias)

    my_model.head.weight.copy_(tv_model.heads.head.weight)
    my_model.head.bias.copy_(tv_model.heads.head.bias)

@torch.no_grad()
def test_bit_match_output_with_tv():
    for yml_file in yml_files:
        cfg = OmegaConf.load(yml_file)
        cfg.img_size = 224
        cfg.img_chls = 3
        cfg.n_class = 1000
        name = cfg.name
        kwargs = dict(cfg)
        if name not in model_map:
            continue
        tv_model = model_map[name]()
        my_model = ViT(ViTConfig(**kwargs))
        tv_model.eval()
        my_model.eval()
        load_weights_from_tv(my_model, tv_model)

        x = torch.randn(2, 3, 224, 224)

        y_tv = tv_model(x)
        y_my = my_model(x)

        assert torch.allclose(y_tv, y_my, atol=1e-5, rtol=1e-5), "NOT bit-matching!"