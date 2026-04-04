#!/usr/bin/env python3
"""MANIP-ULTRADEX training script — config-driven, checkpoint-safe, VRAM-monitored.

Usage:
    # Dry run (verify setup)
    CUDA_VISIBLE_DEVICES=2 uv run python scripts/train.py --config configs/debug.toml --dry-run

    # Smoke test (5 steps, verify checkpoint save/load)
    CUDA_VISIBLE_DEVICES=2 uv run python scripts/train.py --config configs/debug.toml --max-steps 5

    # Full training (nohup)
    CUDA_VISIBLE_DEVICES=2 nohup .venv/bin/python scripts/train.py --config configs/paper.toml \
        > /mnt/artifacts-datai/logs/project_manip_ultradex/train_$(date +%Y%m%d_%H%M).log 2>&1 &
    disown

    # Resume from checkpoint
    CUDA_VISIBLE_DEVICES=2 uv run python scripts/train.py --config configs/paper.toml \
        --resume /mnt/artifacts-datai/checkpoints/project_manip_ultradex/best.pth
"""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import time
import tomllib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from anima_manip_ultradex.config import load_module_config
from anima_manip_ultradex.policy.network import UltraDexPolicy


# ---------------------------------------------------------------------------
# Synthetic dataset (used until real UltraDexGrasp-20M is generated)
# ---------------------------------------------------------------------------


class SyntheticGraspDataset(Dataset):
    """Generates random (point_cloud, action) pairs for pipeline validation.

    Replace with real UltraDexGrasp-20M replay shards when data is available.
    """

    def __init__(self, num_samples: int, num_points: int = 2048, seed: int = 42) -> None:
        self.num_samples = num_samples
        self.num_points = num_points
        self.rng = np.random.default_rng(seed)
        # Pre-generate to ensure determinism across epochs
        self.points = self.rng.standard_normal((num_samples, num_points, 3)).astype(np.float32)
        self.actions = self.rng.uniform(-1, 1, (num_samples, 36)).astype(np.float32)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "scene_pc": torch.from_numpy(self.points[idx]),
            "action_target": torch.from_numpy(self.actions[idx]),
        }


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------


class CheckpointManager:
    def __init__(
        self, save_dir: str, keep_top_k: int = 2, metric: str = "val_loss", mode: str = "min"
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self.metric = metric
        self.mode = mode
        self.history: list[tuple[float, Path]] = []

    def save(self, state: dict, metric_value: float, step: int) -> Path:
        path = self.save_dir / f"checkpoint_step{step:06d}.pth"
        torch.save(state, path)
        self.history.append((metric_value, path))
        self.history.sort(key=lambda x: x[0], reverse=(self.mode == "max"))
        while len(self.history) > self.keep_top_k:
            _, old_path = self.history.pop()
            old_path.unlink(missing_ok=True)
        best_val, best_path = self.history[0]
        best_dest = self.save_dir / "best.pth"
        shutil.copy2(best_path, best_dest)
        return path


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            scale = self.current_step / max(self.warmup_steps, 1)
        else:
            progress = (self.current_step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(self.min_lr, base_lr * scale)

    def state_dict(self) -> dict:
        return {"current_step": self.current_step}

    def load_state_dict(self, state: dict):
        self.current_step = state["current_step"]


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 0.001, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.counter = 0

    def step(self, metric: float) -> bool:
        if self.mode == "min":
            improved = metric < self.best - self.min_delta
        else:
            improved = metric > self.best + self.min_delta
        if improved:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# VRAM monitoring
# ---------------------------------------------------------------------------


def check_gpu_memory(max_util: float = 0.80) -> dict:
    if not torch.cuda.is_available():
        return {"available": False}
    info = {}
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory
        used = torch.cuda.memory_allocated(i)
        util = used / total
        info[f"gpu_{i}"] = {"used_mb": used / 1e6, "total_mb": total / 1e6, "util": util}
        if util > max_util:
            raise RuntimeError(
                f"GPU {i} at {util * 100:.1f}% VRAM — exceeds {max_util * 100:.0f}% cap."
            )
    return info


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="MANIP-ULTRADEX training")
    parser.add_argument("--config", default="configs/paper.toml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_module_config(args.config)

    # Load training config from TOML
    raw = tomllib.loads(Path(args.config).read_text())
    train_cfg = raw.get("training", {})
    ckpt_cfg = raw.get("checkpoint", {})
    es_cfg = raw.get("early_stopping", {})
    log_cfg = raw.get("logging", {})

    # Resolve training params
    seed = train_cfg.get("seed", 42)
    epochs = train_cfg.get("epochs", 200)
    lr = train_cfg.get("learning_rate", 3e-4)
    wd = train_cfg.get("weight_decay", 0.01)
    warmup_ratio = train_cfg.get("warmup_ratio", 0.05)
    min_lr = train_cfg.get("min_lr", 1e-6)
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    precision = train_cfg.get("precision", "fp32")
    d_model = train_cfg.get("d_model", 128)
    num_heads = train_cfg.get("num_heads", 4)
    num_layers = train_cfg.get("num_layers", 4)
    batch_size_cfg = train_cfg.get("batch_size", "auto")

    ckpt_dir = ckpt_cfg.get("output_dir", "/mnt/artifacts-datai/checkpoints/project_manip_ultradex")
    save_every = ckpt_cfg.get("save_every_n_steps", 500)
    keep_top_k = ckpt_cfg.get("keep_top_k", 2)
    log_dir = log_cfg.get("log_dir", "/mnt/artifacts-datai/logs/project_manip_ultradex")

    set_seed(seed)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    policy = UltraDexPolicy(cfg, d_model=d_model, num_heads=num_heads, num_layers=num_layers).to(
        device
    )

    if args.dry_run:
        print(f"[CONFIG] {args.config}")
        print(f"[MODEL] {policy.parameter_count:,} parameters")
        print(f"[DEVICE] {device}")
        print(f"[BATCH] {batch_size_cfg}")
        print(f"[TRAIN] {epochs} epochs, lr={lr}, optimizer=AdamW")
        print(f"[CKPT] save every {save_every} steps, keep best {keep_top_k}")
        return 0

    # Batch size
    if batch_size_cfg == "auto":
        batch_size = 32  # Conservative default for L4
        print(f"[BATCH] Auto: using batch_size={batch_size} (run /gpu-batch-finder for optimal)")
    else:
        batch_size = int(batch_size_cfg)

    # Dataset — SYNTHETIC until real data available
    # TODO: Replace with real UltraDexGrasp-20M replay dataset loader
    total_samples = max(batch_size * 100, 1000)  # At least 1000 samples
    dataset = SyntheticGraspDataset(
        num_samples=total_samples,
        num_points=cfg.paper.policy_input_points,
        seed=seed,
    )

    # Train/val/test split (90/5/5)
    n_train = int(0.9 * len(dataset))
    n_val = int(0.05 * len(dataset))
    n_test = len(dataset) - n_train - n_val
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=wd)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps, min_lr)

    # Checkpoint manager
    ckpt_mgr = CheckpointManager(ckpt_dir, keep_top_k=keep_top_k, metric="val_loss", mode="min")

    # Early stopping
    early_stop = (
        EarlyStopping(
            patience=es_cfg.get("patience", 20),
            min_delta=es_cfg.get("min_delta", 0.001),
        )
        if es_cfg.get("enabled", True)
        else None
    )

    # Mixed precision
    use_amp = precision in ("fp16", "bf16") and device.type == "cuda"
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))

    # Resume
    start_epoch = 0
    global_step = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        policy.load_state_dict(ckpt.get("state_dict", ckpt.get("model", ckpt)), strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("step", 0)
        print(f"[RESUME] from epoch {start_epoch}, step {global_step}")

    # Print training header
    print(f"[CONFIG] {args.config}")
    print(f"[BATCH] batch_size={batch_size} ({'SYNTHETIC DATA' if True else 'real'})")
    print(
        f"[GPU] {torch.cuda.device_count()}x {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
    print(f"[DATA] train={len(train_ds)} val={len(val_ds)} test={n_test}")
    print(f"[MODEL] {policy.parameter_count:,} parameters")
    print(f"[TRAIN] {epochs} epochs, lr={lr}, optimizer=AdamW, precision={precision}")
    print(f"[CKPT] save every {save_every} steps, keep best {keep_top_k}")
    print()

    max_steps = args.max_steps
    policy.train()

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        t_epoch = time.monotonic()

        for batch in train_loader:
            scene_pc = batch["scene_pc"].to(device, non_blocking=True)
            action_target = batch["action_target"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                output = policy(scene_pc, action_targets=action_target)
                loss = output.nll_loss

            if torch.isnan(loss):
                print(f"[FATAL] Loss is NaN at step {global_step} — stopping")
                return 1

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            # Checkpoint
            if global_step % save_every == 0:
                val_loss = _validate(policy, val_loader, device, use_amp, amp_dtype)
                ckpt_mgr.save(
                    {
                        "state_dict": policy.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                        "step": global_step,
                        "val_loss": val_loss,
                        "config": train_cfg,
                    },
                    metric_value=val_loss,
                    step=global_step,
                )
                print(
                    f"  [CKPT] step={global_step} val_loss={val_loss:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )
                policy.train()

            if max_steps and global_step >= max_steps:
                break

        if max_steps and global_step >= max_steps:
            break

        # Epoch summary
        avg_train_loss = epoch_loss / max(epoch_steps, 1)
        val_loss = _validate(policy, val_loader, device, use_amp, amp_dtype)
        elapsed = time.monotonic() - t_epoch

        print(
            f"[Epoch {epoch + 1}/{epochs}] train_loss={avg_train_loss:.4f} "
            f"val_loss={val_loss:.4f} lr={optimizer.param_groups[0]['lr']:.2e} "
            f"time={elapsed:.1f}s"
        )

        # Save at end of epoch
        ckpt_mgr.save(
            {
                "state_dict": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
                "step": global_step,
                "val_loss": val_loss,
                "config": train_cfg,
            },
            metric_value=val_loss,
            step=global_step,
        )
        policy.train()

        # Early stopping
        if early_stop and early_stop.step(val_loss):
            print(f"[EARLY STOP] No improvement for {early_stop.patience} epochs. Stopping.")
            break

    # Final summary
    print(
        f"\n[DONE] Training complete. {global_step} steps, best checkpoint at {ckpt_dir}/best.pth"
    )

    # Check GPU memory utilization
    if torch.cuda.is_available():
        mem = check_gpu_memory(max_util=1.0)
        for k, v in mem.items():
            if isinstance(v, dict):
                print(f"  [{k}] {v['used_mb']:.0f}/{v['total_mb']:.0f} MB ({v['util'] * 100:.1f}%)")

    return 0


@torch.no_grad()
def _validate(
    policy: UltraDexPolicy,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> float:
    policy.eval()
    total_loss = 0.0
    count = 0
    for batch in val_loader:
        scene_pc = batch["scene_pc"].to(device, non_blocking=True)
        action_target = batch["action_target"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            output = policy(scene_pc, action_targets=action_target)
        total_loss += output.nll_loss.item()
        count += 1
    return total_loss / max(count, 1)


if __name__ == "__main__":
    raise SystemExit(main())
