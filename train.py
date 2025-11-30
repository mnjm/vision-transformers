import logging
from contextlib import nullcontext
from pathlib import Path
import hydra
from omegaconf import OmegaConf
import torch
from tqdm.auto import tqdm
from model import init_model, init_deit
from data import init_dataloaders, show_batch
from utils import (
    torch_compile_ckpt_fix,
    torch_get_device,
    torch_set_seed,
    get_ist_time_now,
    cosine_with_linear_warmup_lr_scheduler,
    calc_accuracy,
    timer,
    AverageMetric,
    WandBLogger
)
OmegaConf.register_new_resolver("now_ist", get_ist_time_now)

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

    logger.info(f"Loading {cfg.dataset.name} dataset")
    train_dataloader, val_dataloader = init_dataloaders(cfg)
    if cfg.interactive:
        show_batch(train_dataloader, N=16)

    start_epoch = 1
    if cfg.init_from == 'scratch':
        model = init_model(cfg)
        model.to(device)
    else:
        ckpt = torch.load(cfg.init_from, map_location=device, weights_only=False)
        ckpt_cfg = ckpt['config']
        model = init_model(ckpt_cfg)
        assert cfg.dataset.name == ckpt_cfg.dataset.name, f"Different dataset: {ckpt_cfg.dataset.name}"
        model.to(device)
        model.load_state_dict(torch_compile_ckpt_fix(ckpt['model']))
        logger.info(f"Loaded checkpoint from {cfg.init_from}")
        start_epoch = ckpt['epoch'] + 1

    logger.info(f"Model type: {cfg.model.name} params: {sum(p.numel() for p in model.parameters()):,}")
    if cfg.torch_compile:
        model = torch.compile(model)

    # optimizer
    optimizer = model.configure_optimizer(cfg.optimizer, device)
    lr_scheduler = None
    if cfg.lr_scheduler is not None:
        if cfg.lr_scheduler.name == "cosine-with-linear-warmup":
            kwargs = dict(cfg.lr_scheduler)
            del kwargs['name']
            lr_scheduler = cosine_with_linear_warmup_lr_scheduler(
                optimizer=optimizer, total_steps=cfg.n_epochs * len(train_dataloader), **kwargs
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

    if hasattr(cfg, 'deit') and getattr(cfg.deit, 'enable', False): # init deit training if enabled
        init_deit(model, cfg, device, logger)

    wb_logger = WandBLogger(
        project=cfg.logging.wandb.project,
        run=run_name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        tags=("train", "val"),
        metrics=("loss", "acc@1", "acc@5", "time"),
        enable=cfg.logging.wandb.enable
    )

    @timer
    def train_epoch():
        model.train()
        loss, acc1, acc5 = AverageMetric(), AverageMetric(), AverageMetric()
        progress_bar = tqdm(train_dataloader, dynamic_ncols=True, desc="Train", leave=False, disable=(not cfg.interactive))
        for batch in progress_bar:
            imgs, lbls = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx:
                pred, loss_i = model(imgs, lbls)
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
        return loss.avg, acc1.avg, acc5.avg

    @timer
    @torch.no_grad()
    def val_epoch():
        progress_bar = tqdm(val_dataloader, dynamic_ncols=True, desc="Val", leave=False, disable=(not cfg.interactive))

        model.eval()
        loss, acc1, acc5 = AverageMetric(), AverageMetric(), AverageMetric()
        for batch in progress_bar:
            imgs, lbls = batch[0].to(device), batch[1].to(device)

            with autocast_ctx:
                pred, loss_i = model(imgs, lbls)

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
        return loss.avg, acc1.avg, acc5.avg

    t, (loss, acc1, acc5) = val_epoch()
    logger.info(f"Initial Val Loss: {loss:.4f} Acc@1: {acc1:.2%} Acc@5: {acc5:.2%} Time: {t:.2f}s")
    wb_logger.log("val", {'epoch': start_epoch-1, 'loss': loss, 'acc@1': acc1, 'acc@5':acc5, 'time':t})
    for epoch in range(start_epoch, cfg.n_epochs + 1):
        last_epoch = epoch == cfg.n_epochs
        logger.info(f"Epoch: {epoch}/{cfg.n_epochs}")

        # Train
        t, (loss, acc1, acc5) = train_epoch()
        logger.info(f"{'Train':<5} Loss: {loss:.4f} Acc@1: {acc1:.2%} Acc@5: {acc5:.2%} Time: {t:.2f}s")
        wb_logger.log("train", {'epoch': epoch, 'loss': loss, 'acc@1': acc1, 'acc@5':acc5, 'time':t})

        # Val
        if last_epoch or epoch % cfg.val_every_epoch == 0:
            t, (loss, acc1, acc5) = val_epoch()
            logger.info(f"{'Val':<5} Loss: {loss:.4f} Acc@1: {acc1:.2%} Acc@5: {acc5:.2%} Time: {t:.2f}s")
            wb_logger.log("val", {'epoch': epoch, 'loss': loss, 'acc@1': acc1, 'acc@5':acc5, 'time':t})

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