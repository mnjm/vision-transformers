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
from tqdm.auto import tqdm
from model import ViTConfig, VisionTransformer
from utils import (
    get_dataset,
    torch_compile_ckpt_fix,
    torch_get_device,
    torch_set_seed,
    get_ist_time_now,
    cosine_with_linear_warmup_lr_scheduler,
    AverageMetric,
    calc_accuracy,
)
OmegaConf.register_new_resolver("now_ist", get_ist_time_now)

def configure_optimizer(optim_cfg, model, device):
    optim_cfg.fused = getattr(optim_cfg, 'fused', False) and device.type == "cuda"
    params_dict = { pn: p for pn, p in model.named_parameters() }
    params_dict = { pn:p for pn, p in params_dict.items() if p.requires_grad } # filter params that requires grad
    # create optim groups of any params that is 2D or more. This group will be weight decayed ie weight tensors in Linar and embeddings
    decay_params = [ p for p in params_dict.values() if p.dim() >= 2]
    # create optim groups of any params that is 1D. All biases and layernorm params
    no_decay_params = [ p for p in params_dict.values() if p.dim() < 2]
    weight_decay = getattr(optim_cfg, 'weight_decay', 0.0)
    optim_cfg.weight_decay = 0.
    optim_groups = [
        { 'params': decay_params, 'weight_decay': weight_decay },
        { 'params': no_decay_params, 'weight_decay': 0.0 },
    ]
    optimizer = hydra.utils.instantiate(optim_cfg, params=optim_groups, _convert_="all")
    return optimizer

@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg):
    logger = logging.getLogger("vit")
    device = torch_get_device(cfg.device_type)
    logger.info(f"Using {device}")
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = Path(hydra_cfg.runtime.output_dir)
    run_name = hydra_cfg.job.name
    torch_autocast_dtype = {'f32': torch.float32, 'bf16': torch.bfloat16}[cfg.autocast_dtype]

    torch_set_seed(cfg.rng_seed)

    model_config = ViTConfig(**cfg.model)

    train_ds, val_ds = get_dataset(cfg)
    logger.info(f"Loading {cfg.dataset.name} dataset")
    kwargs = dict(cfg.dataloader)
    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    kwargs['drop_last'] = False
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    # import matplotlib.pyplot as plt
    # from torchvision.utils import make_grid
    # imgs, lbls = next(iter(train_loader))
    # assert imgs.shape[0] >= 16
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
        start_epoch = ckpt['epoch'] + 1

    logger.info(f"Model type: {model_config.name} params: {sum(p.numel() for p in model.parameters()):,}")
    if cfg.torch_compile:
        model = torch.compile(model)

    # optimizer
    optimizer = configure_optimizer(cfg.optimizer, model, device)
    lr_scheduler = None
    if cfg.lr_scheduler is not None:
        if cfg.lr_scheduler.name == "cosine-with-linear-warmup":
            kwargs = dict(cfg.lr_scheduler)
            del kwargs['name']
            lr_scheduler = cosine_with_linear_warmup_lr_scheduler(
                optimizer=optimizer, total_steps=cfg.n_epochs * len(train_loader), **kwargs
            )
        else:
            raise NotImplementedError(f"{cfg.lr_scheduler.name} is not implemented")

    if cfg.init_from != 'scratch':
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_schdlr_state = ckpt.get('lr_scheduler', None)
        if lr_schdlr_state and lr_scheduler:
            lr_scheduler.load_state_dict(lr_schdlr_state)

    if cfg.enable_tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    autocast_ctx = (
        torch.amp.autocast(device_type=device.type, dtype=torch_autocast_dtype)
        if device.type == "cuda" and torch_autocast_dtype == torch.bfloat16
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
        metrics = 'train/loss', 'train/acc@1', 'train/acc@5', 'val/loss', 'val/acc@1', 'val/acc@5', 'train/time', 'val/time'
        wandb.define_metric("epoch")
        for metric in metrics:
            wandb.define_metric(metric, step_metric="epoch")

    def train_epoch():
        model.train()
        t0 = time()
        loss, acc1, acc5 = AverageMetric(), AverageMetric(), AverageMetric()
        progress_bar = tqdm(train_loader, dynamic_ncols=True, desc="Train", leave=False)
        for step, batch in enumerate(progress_bar):
            imgs, lbls = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx:
                pred = model(imgs)
                loss_i = F.cross_entropy(pred, lbls)
            loss_i.backward()
            if cfg.clip_grad_norm_1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            acc1_i, acc5_i = calc_accuracy(pred, lbls, (1, 5))
            batch_size = lbls.size(0)
            loss.update(loss_i.item(), batch_size)
            acc1.update(acc1_i.item(), batch_size)
            acc5.update(acc5_i.item(), batch_size)
            progress_bar.set_postfix({
                'loss': f"{loss_i.item():.4f}",
                'acc@1': f"{acc1_i.item():.2%}",
                'acc@5': f"{acc5_i.item():.2%}",
            })

        progress_bar.close()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time() - t0
        return loss.avg, acc1.avg, acc5.avg, t

    @torch.no_grad()
    def val_epoch():
        progress_bar = tqdm(val_loader, dynamic_ncols=True, desc="Val", leave=False)
        model.eval()
        t0 = time()
        loss, acc1, acc5 = AverageMetric(), AverageMetric(), AverageMetric()
        for step, batch in enumerate(progress_bar):
            imgs, lbls = batch[0].to(device), batch[1].to(device)

            with autocast_ctx:
                pred = model(imgs)
                loss_i = F.cross_entropy(pred, lbls)

            acc1_i, acc5_i = calc_accuracy(pred, lbls, (1, 5))
            batch_size = lbls.size(0)
            loss.update(loss_i.item(), batch_size)
            acc1.update(acc1_i.item(), batch_size)
            acc5.update(acc5_i.item(), batch_size)
            progress_bar.set_postfix({
                'loss': f"{loss_i.item():.4f}",
                'acc@1': f"{acc1_i.item():.2%}",
                'acc@5': f"{acc5_i.item():.2%}",
            })

        progress_bar.close()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time() - t0
        return loss.avg, acc1.avg, acc5.avg, t

    loss, acc1, acc5, t = val_epoch()
    logger.info(f"Initial Val Loss: {loss:.4f} Acc@1: {acc1:.2%} Acc@5: {acc5:.2%} Time: {t:.2f}s")
    for epoch in range(start_epoch, cfg.n_epochs + 1):
        last_epoch = epoch == cfg.n_epochs
        logger.info(f"Epoch: {epoch}/{cfg.n_epochs}")

        # Train
        loss, acc1, acc5, t = train_epoch()
        logger.info(f"{'Train':<5} Loss: {loss:.4f} Acc@1: {acc1:.2%} Acc@5: {acc5:.2%} Time: {t:.2f}s")
        if cfg.logging.wandb.enable:
            wandb.log({
                'epoch': epoch,
                'train/loss': loss,
                'train/acc@1': acc1,
                'train/acc@5': acc5,
                'train/time': t,
            })

        # Val
        if last_epoch or epoch % cfg.val_every_epoch == 0:
            loss, acc1, acc5, t = val_epoch()
            logger.info(f"{'Val':<5} Loss: {loss:.4f} Acc@1: {acc1:.2%} Acc@5: {acc5:.2%} Time: {t:.2f}s")
            if cfg.logging.wandb.enable:
                wandb.log({
                    'epoch': epoch,
                    'val/loss': loss,
                    'val/acc@1': acc1,
                    'val/acc@5': acc5,
                    'val/time': t,
                })

        # Ckpt
        if last_epoch or epoch % cfg.save_every_epoch == 0:
            ckpt_path = log_dir / f"{cfg.model_name}.pt"
            torch.save({
                'model': model.state_dict(),
                'config': cfg,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
            }, ckpt_path)
            logger.info(f"Saved checkpoint to {str(ckpt_path)}")

if __name__ == "__main__":
    main()
