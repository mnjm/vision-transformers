import os
import logging
from contextlib import nullcontext
from pathlib import Path
from time import time
import hydra
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import ViTConfig, VisionTransformer
from utils import (
    get_dataset,
    torch_compile_ckpt_fix,
    torch_get_device,
    torch_set_seed,
    get_ist_time_now
)
OmegaConf.register_new_resolver("now_ist", get_ist_time_now)

@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg):
    logger = logging.getLogger("vit-train")
    device = torch_get_device(cfg.device_type)
    logger.info(f"Using {device}")
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = Path(hydra_cfg.runtime.output_dir)
    run_name = hydra_cfg.job.name
    torch_amp_dtype = {'f32': torch.float32, 'bf16': torch.bfloat16}[cfg.amp_dtype]

    torch_set_seed(cfg.rng_seed)

    model_config = ViTConfig(**cfg.model)

    train_ds, val_ds = get_dataset(cfg.dataset)
    logger.info(f"Loading {cfg.dataset.name} dataset")
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=cfg.dataloader.drop_last,
        pin_memory=cfg.dataloader.pin_memory, num_workers=cfg.dataloader.workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        drop_last=False, num_workers=cfg.dataloader.workers
    )
    # import matplotlib.pyplot as plt
    # from torchvision.utils import make_grid
    # imgs, lbls = next(iter(train_loader))
    # assert imgs.shape[0] > 16
    # grid_img = make_grid(imgs[:16, ...], 4, normalize=True).to("cpu").permute(1, 2, 0).numpy()
    # plt.title(",".join(str(x) for x in lbls.numpy().reshape(-1)[:16]))
    # plt.imshow(grid_img)
    # plt.show()
    # import sys; sys.exit(0)

    start_epoch = 1
    if cfg.init_from == 'scratch':
        model = VisionTransformer(model_config)
        model.to(device)
    else:
        ckpt = torch.load(cfg.init_from, map_location=device, weights_only=False)
        ckpt_cfg = ckpt['config']
        model_config = ViTConfig(**ckpt_cfg.model)
        assert cfg.dataset.name == ckpt_cfg.dataset.name, f"Different dataset: {ckpt_cfg.dataset.name}"
        model = VisionTransformer(model_config)
        model.to(device)
        model.load_state_dict(torch_compile_ckpt_fix(ckpt['model']))
        logger.info(f"Loaded checkpoint from {cfg.init_from}")
        start_epoch = ckpt['epoch']

    logger.info(f"Model type: {model_config.name} params: {sum(p.numel() for p in model.parameters()):,}")
    if cfg.torch_compile:
        model = torch.compile(model)

    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

    if cfg.init_from != 'scratch':
        optimizer.load_state_dict(ckpt['optimizer'])

    if cfg.enable_tf32:
        torch.set_float32_matmul_precision("high")
    amp_ctx = (
        torch.amp.autocast(device_type=device.type, dtype=torch_amp_dtype)
        if device.type == "cuda" and torch_amp_dtype == torch.bfloat16
        else nullcontext()
    )

    if cfg.logging.wandb.enable:
        load_dotenv()
        wnb_key = os.getenv('WANDB_API_KEY')
        assert wnb_key is not None, "WANDB_API_KEY not loaded in env"
        wandb.login(key=wnb_key)
        wandb.init(
          project=cfg.logging.wandb.project,
          name=run_name,
          config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        )
        metrics = 'train/loss', 'train/acc', 'val/loss', 'val/acc', 'train/time', 'val/time'
        wandb.define_metric("epoch")
        for metric in metrics:
            wandb.define_metric(metric, step_metric="epoch")

    norm = float('nan')
    for epoch in range(start_epoch, cfg.n_epochs + 1):
        last_epoch = epoch == cfg.n_epochs

        # Train
        t0 = time()
        loss_cum, correct, total = 0.0, 0, 0
        model.train()
        logger.info(f"Epoch: {epoch}/{cfg.n_epochs}")
        progress_bar = tqdm(train_loader, dynamic_ncols=True, desc="Train", leave=False)
        for step, batch in enumerate(progress_bar):
            imgs, lbls = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            with amp_ctx:
                pred = model(imgs)
                loss = F.cross_entropy(pred, lbls)
            loss.backward()
            if cfg.clip_grad_norm_1:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_cum += loss.item()
            correct += (pred.argmax(dim=1) == lbls).sum().item()
            total += lbls.size(0)
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time() - t0
        avg_loss = loss_cum / len(train_loader)
        acc = correct / total
        logger.info(f"Loss: {avg_loss:.4f} Acc: {acc:.2%} Norm: {norm: .4f} Time: {t:.2f}s")
        if cfg.logging.wandb.enable:
            wandb.log({
                'epoch': epoch,
                'train/loss': avg_loss,
                'train/acc': acc,
                'train/time': t,
            })

        # Val
        if last_epoch or epoch % cfg.val_every_epoch == 0:
            progress_bar = tqdm(val_loader, dynamic_ncols=True, desc="Val", leave=False)
            model.eval()
            t0 = time()
            loss_cum, correct, total = 0.0, 0, 0
            with torch.no_grad():
                correct, total = 0, 0
                for step, batch in enumerate(progress_bar):
                    imgs, lbls = batch[0].to(device), batch[1].to(device)
                    with amp_ctx:
                        pred = model(imgs)
                        loss = F.cross_entropy(pred, lbls)
                        correct += (pred.argmax(dim=1) == lbls).sum().item()
                        total += lbls.size(0)
                    loss_cum += loss.item()
                    progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            if device.type == "cuda":
                torch.cuda.synchronize()
            t = time() - t0
            avg_loss = loss_cum / len(val_loader)
            acc = correct / total
            logger.info(f"Val Loss: {avg_loss:.4f} Acc: {acc:.2%} Time: {t:.2f}s")
            if cfg.logging.wandb.enable:
                wandb.log({
                    'epoch': epoch,
                    'val/loss': avg_loss,
                    'val/acc': acc,
                    'val/time': t,
                })

        # Ckpt
        if last_epoch or epoch % cfg.save_every_epoch == 0:
            ckpt_path = log_dir / f"{cfg.model_name}.pt"
            torch.save({
                'model': model.state_dict(),
                'config': cfg,
                'epoch': epoch,
                'loss': avg_loss,
                'optimizer': optimizer.state_dict(),
            }, ckpt_path)
            logger.info(f"Saved checkpoint to {str(ckpt_path)}")

if __name__ == "__main__":
    main()