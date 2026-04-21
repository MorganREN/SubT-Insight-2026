from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from torch.amp import GradScaler, autocast

from criteria import SegEvaluator
from dataload import CLASS_NAMES, NUM_CLASSES, build_dataloaders
from models.segmentor import TunnelSegmentor
from models.segmentor_tmds import TMDSSegmentor
from utils.optimizer import build_optimizer
from utils.runtime import load_checkpoint_compat, resolve_device, restore_training_checkpoint, setup_logger
from utils.scheduler import build_scheduler, log_lr

from utils.feishu import send_eval_result, send_training_done

from .config import TrainConfig
from .loss_factory import TMDSCriterion, build_loss, build_stage_loss, build_tmds_criterion


class SegmentationTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg

    # ──────────────────────────────────────────────────────────────────────────
    # 静态工具
    # ──────────────────────────────────────────────────────────────────────────

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

    # ──────────────────────────────────────────────────────────────────────────
    # TMDS 三阶段训练辅助
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_stage(epoch: int, stage_epochs: tuple[int, int, int]) -> int:
        """返回 epoch 对应的阶段索引（0/1/2）。"""
        s0 = stage_epochs[0]
        s1 = s0 + stage_epochs[1]
        if epoch <= s0:
            return 0
        if epoch <= s1:
            return 1
        return 2

    def _enter_stage(
        self,
        model: nn.Module,
        stage: int,
        class_weights: Optional[torch.Tensor],
        device: torch.device,
    ) -> tuple:
        """
        进入新阶段：更新骨干冻结状态、重建 optimizer / scheduler / criterion。

        Returns
        -------
        (optimizer, scheduler, criterion)
        """
        cfg = self.cfg
        frozen = cfg.stage_frozen_stages[stage]
        stage_lr = cfg.stage_base_lrs[stage]
        stage_ep = cfg.stage_epochs[stage]

        model.set_frozen_stages(frozen)
        logger.info(
            f"━━━ 进入 Stage {stage + 1}/3 ━━━  "
            f"frozen_stages={frozen}  base_lr={stage_lr:.1e}  "
            f"epochs={stage_ep}  loss={cfg.stage_loss_names[stage]}"
            + ("+skel" if (stage == 2 and cfg.use_tmds and cfg.use_skeleton_loss) else "")
        )

        optimizer = build_optimizer(
            model,
            optimizer_type=cfg.optimizer_type,
            base_lr=stage_lr,
            backbone_lr_mult=cfg.backbone_lr_mult,
            weight_decay=cfg.weight_decay,
        )
        scheduler = build_scheduler(
            optimizer,
            scheduler_type=cfg.scheduler,
            total_epochs=stage_ep,
            warmup_epochs=min(2, stage_ep // 10),
        )
        if cfg.use_tmds:
            criterion = build_tmds_criterion(cfg, stage, class_weights, device)
        else:
            criterion = build_stage_loss(
                cfg.stage_loss_names[stage],
                num_classes=cfg.num_classes,
                class_weights=class_weights,
                device=device,
            )
        return optimizer, scheduler, criterion

    # ──────────────────────────────────────────────────────────────────────────
    # 训练 / 验证核心循环
    # ──────────────────────────────────────────────────────────────────────────

    def _train_one_epoch(
        self,
        model: torch.nn.Module,
        loader,
        criterion,
        optimizer: torch.optim.Optimizer,
        scaler: GradScaler,
        device: torch.device,
        use_amp: bool,
        epoch: int,
    ) -> tuple[float, dict[str, float]]:
        """返回 (avg_total_loss, avg_components)，后者供 TensorBoard 记录。"""
        cfg = self.cfg
        model.train()
        total_loss = 0.0
        components_sum: dict[str, float] = {}
        valid_batches  = 0
        num_batches    = len(loader)
        t0 = time.time()

        for step, batch in enumerate(loader, start=1):
            if cfg.max_steps > 0 and step > cfg.max_steps:
                break
            images    = batch[0].to(device, non_blocking=True)
            masks     = batch[1].to(device, non_blocking=True)
            skel_masks = (
                batch[2].to(device, non_blocking=True) if len(batch) > 2 else None
            )

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=use_amp):
                outputs = model(images)
                if isinstance(criterion, TMDSCriterion):
                    loss = criterion(outputs, masks, skel_masks)
                else:
                    loss = criterion(outputs, masks)

            if not torch.isfinite(loss):
                comps = getattr(criterion, "last_components", {})
                comp_str = "  ".join(f"{k}={v:.4f}" for k, v in comps.items())
                msg = (
                    f"NaN/inf loss 于 Epoch [{epoch:03d}] step [{step:3d}]: "
                    f"loss={loss.item()}  ({comp_str})"
                )
                logger.error(msg)
                raise ValueError(msg)

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

            total_loss  += loss.item()
            valid_batches += 1
            for k, v in getattr(criterion, "last_components", {}).items():
                components_sum[k] = components_sum.get(k, 0.0) + v

            if step % 20 == 0 or step == num_batches:
                comps = getattr(criterion, "last_components", {})
                # 若任意子损失为 NaN/inf，单独标注，帮助定位具体模块
                comp_str = "  ".join(
                    f"{k}={'NaN!' if not torch.isfinite(torch.tensor(v)) else f'{v:.3f}'}"
                    for k, v in comps.items()
                )
                logger.info(
                    f"Epoch [{epoch:03d}/{cfg.epochs}]  "
                    f"step [{step:3d}/{num_batches}]  "
                    f"loss={loss.item():.4f}  ({comp_str})  "
                    f"time={time.time() - t0:.1f}s"
                )

        n = max(valid_batches, 1)
        avg_components = {k: v / n for k, v in components_sum.items()}
        return total_loss / n, avg_components

    @staticmethod
    @torch.no_grad()
    def _validate(
        model: torch.nn.Module,
        loader,
        criterion,
        evaluator: SegEvaluator,
        device: torch.device,
        use_amp: bool,
    ) -> tuple[float, dict]:
        model.eval()
        evaluator.reset()
        total_loss = 0.0

        for batch in loader:
            images = batch[0].to(device, non_blocking=True)
            masks  = batch[1].to(device, non_blocking=True)

            with autocast("cuda", enabled=use_amp):
                # eval 模式下 TMDSSegmentor 返回单张量，criterion 兼容两种形式
                logits = model(images)
                loss   = criterion(logits, masks)

            total_loss += loss.item()
            evaluator.update(logits, masks)

        metrics  = evaluator.compute()
        avg_loss = total_loss / max(len(loader), 1)
        return avg_loss, metrics

    # ──────────────────────────────────────────────────────────────────────────
    # 主入口
    # ──────────────────────────────────────────────────────────────────────────

    def run(self) -> float:
        cfg = self.cfg
        self._set_seed(cfg.seed)

        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        setup_logger(str(out_dir / "train.log"))

        _stage_mode = "TMDS 三阶段" if cfg.use_tmds else ("三阶段" if cfg.use_stages else "")
        logger.info("=" * 70)
        logger.info("语义分割训练启动" + (f"（{_stage_mode}）" if _stage_mode else ""))
        logger.info("=" * 70)
        logger.info(f"输出目录: {out_dir.resolve()}")
        logger.info(f"配置: {asdict(cfg)}")

        device  = resolve_device(cfg.device)
        use_amp = cfg.use_amp and device.type == "cuda"
        if cfg.use_amp and not use_amp:
            logger.warning("AMP 当前仅在 CUDA 下启用，已自动关闭")

        # ── 三阶段模式：总 epoch = sum(stage_epochs) ──
        if cfg.use_tmds or cfg.use_stages:
            total_epochs = sum(cfg.stage_epochs)
            if total_epochs != cfg.epochs:
                logger.info(
                    f"三阶段模式：total epochs 由 stage_epochs {cfg.stage_epochs} "
                    f"决定，设为 {total_epochs}（原 cfg.epochs={cfg.epochs} 已忽略）"
                )
            cfg.epochs = total_epochs  # 就地更新，只影响本次 run

        # ── 数据加载 ──
        loaders = build_dataloaders(
            data_roots=[cfg.data_root],
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            input_size=cfg.input_size,
            splits=["train", "val"],
            use_skeleton=cfg.use_tmds and cfg.use_skeleton_loss,
        )
        train_loader = loaders["train"]
        val_loader   = loaders["val"]
        logger.info(
            f"数据集: train={len(train_loader.dataset)} 张  "
            f"val={len(val_loader.dataset)} 张  "
            f"batch_size={cfg.batch_size}"
        )

        # ── 类别权重 ──
        class_weights = None
        if cfg.use_class_weights:
            logger.info("正在统计训练集类别权重...")
            class_weights = train_loader.dataset.get_class_weights()
            logger.info(f"类别权重: {class_weights.tolist()}")

        # ── 模型 ──
        if cfg.use_tmds:
            model = TMDSSegmentor(
                num_classes=cfg.num_classes,
                backbone_type=cfg.backbone_type,
                backbone_weight_path=cfg.backbone_weight_path,
                frozen_stages=cfg.stage_frozen_stages[0],  # 初始阶段冻结状态
                head_channels=cfg.head_channels,
                dsa_num_heads=cfg.dsa_num_heads,
                dsa_num_strips=cfg.dsa_num_strips,
                dsa_points_per_strip=cfg.dsa_points_per_strip,
                mrm_stage_idx=cfg.mrm_stage_idx,
            ).to(device)
        else:
            model = TunnelSegmentor(
                num_classes=cfg.num_classes,
                backbone_weight_path=cfg.backbone_weight_path,
                head_type=cfg.head_type,
                head_channels=cfg.head_channels,
                frozen_stages=cfg.frozen_stages,
            ).to(device)

        # ── 损失 / 优化器 / 调度器 ──
        scaler    = GradScaler("cuda", enabled=use_amp)
        evaluator = SegEvaluator(num_classes=cfg.num_classes, class_names=CLASS_NAMES)

        if cfg.use_tmds or cfg.use_stages:
            # 三阶段模式：在 run 循环内按阶段动态初始化，此处先置 None
            optimizer  = None
            scheduler  = None
            criterion  = None
            current_stage = -1   # 尚未进入任何阶段
        else:
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

        # ── 断点恢复 ──
        start_epoch = 1
        best_miou   = -1.0
        # TMDS 模式下 optimizer/scheduler 在首次 _enter_stage 时才创建，
        # 此处只能恢复模型权重；optimizer/scheduler state_dict 暂存待后续注入。
        _resume_opt_sd  = None   # 待注入的 optimizer state_dict
        _resume_sch_sd  = None   # 待注入的 scheduler state_dict
        if cfg.resume:
            if cfg.use_tmds or cfg.use_stages:
                ckpt = load_checkpoint_compat(cfg.resume, map_location=device)
                state_dict = ckpt["model"] if "model" in ckpt else ckpt
                model.load_state_dict(state_dict)
                start_epoch     = ckpt.get("epoch", 0) + 1
                best_miou       = ckpt.get("best_miou", -1.0)
                _resume_opt_sd  = ckpt.get("optimizer")
                _resume_sch_sd  = ckpt.get("scheduler")
                logger.info(
                    f"三阶段断点续训自 epoch={start_epoch - 1}，"
                    f"best_mIoU={best_miou:.4f}；"
                    f"optimizer/scheduler 将在首次进入阶段后恢复"
                )
            else:
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

        # ── TensorBoard ──
        tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = out_dir / "tensorboard"
            tb_writer = SummaryWriter(log_dir=str(tb_dir))
            logger.info(f"TensorBoard 已启动，日志目录: {tb_dir}")
            logger.info(f"  查看命令: tensorboard --logdir {tb_dir}")
        except Exception as e:
            logger.warning(f"TensorBoard 不可用，跳过曲线记录: {e}")

        logger.info(f"开始训练: epoch {start_epoch} → {cfg.epochs}")
        epoch_times = []

        # ── 阶段内 epoch 计数（用于调度器 step）──
        stage_epoch_offset = 0   # 当前阶段开始时的全局 epoch

        for epoch in range(start_epoch, cfg.epochs + 1):

            # ── 三阶段切换检测 ──
            if cfg.use_tmds or cfg.use_stages:
                new_stage = self._get_stage(epoch, cfg.stage_epochs)
                if new_stage != current_stage:
                    current_stage    = new_stage
                    stage_epoch_offset = epoch - 1
                    optimizer, scheduler, criterion = self._enter_stage(
                        model, current_stage, class_weights, device
                    )
                    # 断点续训：首次进入阶段后恢复 optimizer/scheduler 状态
                    if _resume_opt_sd is not None:
                        optimizer.load_state_dict(_resume_opt_sd)
                        _resume_opt_sd = None
                        logger.info("断点续训：optimizer 状态已恢复")
                    if _resume_sch_sd is not None:
                        scheduler.load_state_dict(_resume_sch_sd)
                        _resume_sch_sd = None
                        logger.info("断点续训：scheduler 状态已恢复")

            epoch_start = time.time()

            train_loss, train_components = self._train_one_epoch(
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
            eta_seconds    = avg_epoch_time * (cfg.epochs - epoch)

            logger.info(
                f"【Train Epoch {epoch:03d}】  "
                f"loss={train_loss:.4f}  "
                f"time={epoch_time:.1f}s  "
                f"ETA≈{eta_seconds / 60:.1f}min"
            )
            log_lr(optimizer)

            # ── TensorBoard：训练曲线 ──
            if tb_writer is not None:
                tb_writer.add_scalar("Loss/train", train_loss, epoch)
                for k, v in train_components.items():
                    tb_writer.add_scalar(f"Loss/train_{k}", v, epoch)
                for pg in optimizer.param_groups:
                    tb_writer.add_scalar(
                        f"LR/{pg.get('name', 'group')}", pg["lr"], epoch
                    )

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

                # ── TensorBoard：验证曲线 ──
                if tb_writer is not None:
                    tb_writer.add_scalar("Loss/val",        val_loss,             epoch)
                    tb_writer.add_scalar("Metrics/mIoU",    metrics["mIoU"],      epoch)
                    tb_writer.add_scalar("Metrics/aAcc",    metrics["aAcc"],      epoch)
                    tb_writer.add_scalar("Metrics/mDice",   metrics["mDice"],     epoch)
                    for i, name in enumerate(CLASS_NAMES):
                        tb_writer.add_scalar(f"IoU/{name}", metrics["IoU"][i],    epoch)

                is_best = metrics["mIoU"] > best_miou
                if is_best:
                    best_miou = metrics["mIoU"]
                    self._save_checkpoint(
                        {
                            "epoch":      epoch,
                            "model":      model.state_dict(),
                            "optimizer":  optimizer.state_dict(),
                            "scheduler":  scheduler.state_dict(),
                            "best_miou":  best_miou,
                            "metrics":    metrics,
                            "config":     asdict(cfg),
                        },
                        str(out_dir / "best.pth"),
                    )
                    logger.success(
                        f"✅ 新最优模型！mIoU={best_miou * 100:.2f}%  → {out_dir / 'best.pth'}"
                    )

                send_eval_result(
                    epoch=epoch,
                    total_epochs=cfg.epochs,
                    metrics=metrics,
                    val_loss=val_loss,
                    is_best=is_best,
                    class_names=CLASS_NAMES,
                    run_name=Path(cfg.output_dir).name,
                )

            self._save_checkpoint(
                {
                    "epoch":     epoch,
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_miou": best_miou,
                    "config":    asdict(cfg),
                },
                str(out_dir / "last.pth"),
            )

        if tb_writer is not None:
            tb_writer.close()

        logger.success("=" * 70)
        logger.success(f"训练完成！最优验证集 mIoU = {best_miou * 100:.2f}%")
        logger.success(f"最优模型: {out_dir / 'best.pth'}")
        logger.success(f"日志文件: {out_dir / 'train.log'}")
        logger.success("=" * 70)

        send_training_done(
            best_miou=best_miou,
            output_dir=str(out_dir.resolve()),
            run_name=Path(cfg.output_dir).name,
        )

        return best_miou
