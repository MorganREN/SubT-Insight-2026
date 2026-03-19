from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import torch
from loguru import logger
from torch.amp import GradScaler, autocast

from criteria import SegEvaluator
from dataload import CLASS_NAMES, NUM_CLASSES, build_dataloaders
from models.segmentor import TunnelSegmentor
from utils.optimizer import build_optimizer
from utils.runtime import resolve_device, restore_training_checkpoint, setup_logger
from utils.scheduler import build_scheduler, log_lr

from .config import TrainConfig
from .loss_factory import build_loss


class SegmentationTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg

    @staticmethod
    def _set_seed(seed: int):
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _save_checkpoint(state: dict, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    @staticmethod
    def _format_metrics(metrics: dict) -> str:
        iou_str = " ".join(
            f"{CLASS_NAMES[i][:4]}={metrics['IoU'][i] * 100:.1f}"
            for i in range(NUM_CLASSES)
        )
        return (
            f"mIoU={metrics['mIoU'] * 100:.2f}%  "
            f"aAcc={metrics['aAcc'] * 100:.2f}%  "
            f"mDice={metrics['mDice'] * 100:.2f}%  "
            f"[{iou_str}]"
        )

    def _train_one_epoch(self, model, loader, criterion, optimizer, scaler, device, use_amp: bool, epoch: int):
        cfg = self.cfg
        model.train()
        total_loss = 0.0
        num_batches = len(loader)
        t0 = time.time()

        for step, (images, masks) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)

            if use_amp:
                scaler.scale(loss).backward()
                if cfg.clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_norm=cfg.clip_grad,
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_norm=cfg.clip_grad,
                    )
                optimizer.step()

            total_loss += loss.item()

            if step % 20 == 0 or step == num_batches:
                comps = getattr(criterion, "last_components", {})
                comp_str = "  ".join(f"{k}={v:.3f}" for k, v in comps.items())
                logger.info(
                    f"Epoch [{epoch:03d}/{cfg.epochs}]  "
                    f"step [{step:3d}/{num_batches}]  "
                    f"loss={loss.item():.4f}  ({comp_str})  "
                    f"time={time.time() - t0:.1f}s"
                )

        return total_loss / max(num_batches, 1)

    @staticmethod
    @torch.no_grad()
    def _validate(model, loader, criterion, evaluator, device, use_amp: bool):
        model.eval()
        evaluator.reset()
        total_loss = 0.0

        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)

            total_loss += loss.item()
            evaluator.update(logits, masks)

        metrics = evaluator.compute()
        avg_loss = total_loss / max(len(loader), 1)
        return avg_loss, metrics

    def run(self) -> float:
        cfg = self.cfg
        self._set_seed(cfg.seed)

        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        setup_logger(str(out_dir / "train.log"))

        logger.info("=" * 70)
        logger.info("语义分割训练启动")
        logger.info("=" * 70)
        logger.info(f"输出目录: {out_dir.resolve()}")
        logger.info(f"配置: {asdict(cfg)}")

        device = resolve_device(cfg.device)
        use_amp = cfg.use_amp and device.type == "cuda"
        if cfg.use_amp and not use_amp:
            logger.warning("AMP 当前仅在 CUDA 下启用，已自动关闭")

        loaders = build_dataloaders(
            data_roots=[cfg.data_root],
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            input_size=cfg.input_size,
            splits=["train", "val"],
        )
        train_loader = loaders["train"]
        val_loader = loaders["val"]
        logger.info(
            f"数据集: train={len(train_loader.dataset)} 张  "
            f"val={len(val_loader.dataset)} 张  "
            f"batch_size={cfg.batch_size}"
        )

        class_weights = None
        if cfg.use_class_weights:
            logger.info("正在统计训练集类别权重...")
            class_weights = train_loader.dataset.get_class_weights()
            logger.info(f"类别权重: {class_weights.tolist()}")

        model = TunnelSegmentor(
            num_classes=cfg.num_classes,
            backbone_weight_path=cfg.backbone_weight_path,
            head_type=cfg.head_type,
            head_channels=cfg.head_channels,
            frozen_stages=cfg.frozen_stages,
        ).to(device)

        criterion = build_loss(cfg, class_weights=class_weights, device=device)
        logger.info(f"损失函数: {cfg.loss_name}")

        optimizer = build_optimizer(
            model,
            optimizer_type=cfg.optimizer_type,
            base_lr=cfg.base_lr,
            backbone_lr_mult=cfg.backbone_lr_mult,
            weight_decay=cfg.weight_decay,
        )
        scheduler = build_scheduler(
            optimizer,
            scheduler_type=cfg.scheduler,
            total_epochs=cfg.epochs,
            warmup_epochs=cfg.warmup_epochs,
        )
        scaler = GradScaler("cuda", enabled=use_amp)
        evaluator = SegEvaluator(num_classes=cfg.num_classes, class_names=cfg.class_names)

        start_epoch = 1
        best_miou = -1.0
        if cfg.resume:
            start_epoch, best_miou = restore_training_checkpoint(
                cfg.resume,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
            )

        if cfg.dry_run:
            logger.success("Dry run 完成：数据、模型、损失、优化器、调度器均初始化成功")
            return best_miou

        logger.info(f"开始训练: epoch {start_epoch} → {cfg.epochs}")
        epoch_times = []

        for epoch in range(start_epoch, cfg.epochs + 1):
            epoch_start = time.time()

            train_loss = self._train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                use_amp=use_amp,
                epoch=epoch,
            )

            scheduler.step()

            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            avg_epoch_time = sum(epoch_times[-5:]) / len(epoch_times[-5:])
            eta_seconds = avg_epoch_time * (cfg.epochs - epoch)

            logger.info(
                f"【Train Epoch {epoch:03d}】  "
                f"loss={train_loss:.4f}  "
                f"time={epoch_time:.1f}s  "
                f"ETA≈{eta_seconds / 60:.1f}min"
            )
            log_lr(optimizer)

            if epoch % cfg.val_interval == 0 or epoch == cfg.epochs:
                val_loss, metrics = self._validate(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    evaluator=evaluator,
                    device=device,
                    use_amp=use_amp,
                )
                evaluator.print_table(metrics)
                logger.info(
                    f"【Val   Epoch {epoch:03d}】  "
                    f"loss={val_loss:.4f}  "
                    + self._format_metrics(metrics)
                )

                if metrics["mIoU"] > best_miou:
                    best_miou = metrics["mIoU"]
                    self._save_checkpoint(
                        {
                            "epoch": epoch,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "best_miou": best_miou,
                            "metrics": metrics,
                            "config": asdict(cfg),
                        },
                        str(out_dir / "best.pth"),
                    )
                    logger.success(
                        f"✅ 新最优模型！mIoU={best_miou * 100:.2f}%  → {out_dir / 'best.pth'}"
                    )

            self._save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_miou": best_miou,
                    "config": asdict(cfg),
                },
                str(out_dir / "last.pth"),
            )

        logger.success("=" * 70)
        logger.success(f"训练完成！最优验证集 mIoU = {best_miou * 100:.2f}%")
        logger.success(f"最优模型: {out_dir / 'best.pth'}")
        logger.success(f"日志文件: {out_dir / 'train.log'}")
        logger.success("=" * 70)

        return best_miou
