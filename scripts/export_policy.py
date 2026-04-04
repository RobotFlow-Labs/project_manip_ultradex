#!/usr/bin/env python3
"""Full export pipeline: pth → safetensors → ONNX → TRT fp16 → TRT fp32.

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/export_policy.py \
        --checkpoint /mnt/artifacts-datai/checkpoints/project_manip_ultradex/best.pth \
        --output-dir /mnt/artifacts-datai/exports/project_manip_ultradex
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch import nn

from anima_manip_ultradex.config import load_module_config
from anima_manip_ultradex.policy.network import UltraDexPolicy


class ExportablePolicy(nn.Module):
    """Wrapper that exposes a single forward: scene_pc → action_mean."""

    def __init__(self, policy: UltraDexPolicy) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, scene_pc: torch.Tensor) -> torch.Tensor:
        return self.policy(scene_pc).action_mean


def export_pth(policy: UltraDexPolicy, output_dir: Path, cfg) -> Path:
    path = output_dir / "policy.pth"
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "config": {
                "d_model": 128,
                "num_heads": 4,
                "num_layers": 2,
                "paper_arxiv": cfg.project.paper_arxiv,
            },
        },
        path,
    )
    print(f"[EXPORT] pth: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def export_safetensors(policy: UltraDexPolicy, output_dir: Path) -> Path | None:
    try:
        from safetensors.torch import save_file
    except ImportError:
        print("[EXPORT] safetensors: SKIPPED (install safetensors package)")
        return None
    path = output_dir / "policy.safetensors"
    save_file(policy.state_dict(), str(path))
    print(f"[EXPORT] safetensors: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def export_onnx(exportable: ExportablePolicy, output_dir: Path, cfg) -> Path | None:
    try:
        import onnx  # noqa: F401
    except ImportError:
        print("[EXPORT] ONNX: SKIPPED (install onnx package)")
        return None
    path = output_dir / "policy.onnx"
    dummy = torch.randn(1, cfg.paper.policy_input_points, 3)
    torch.onnx.export(
        exportable,
        dummy,
        str(path),
        input_names=["scene_pc"],
        output_names=["action_mean"],
        dynamic_axes={"scene_pc": {0: "batch"}, "action_mean": {0: "batch"}},
        opset_version=17,
    )
    print(f"[EXPORT] ONNX: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def export_trt(onnx_path: Path, output_dir: Path, precision: str) -> Path | None:
    """Export ONNX → TensorRT using shared toolkit or trtexec."""
    trt_path = output_dir / f"policy_{precision}.engine"

    # Try shared TRT toolkit first
    toolkit = Path("/mnt/forge-data/shared_infra/trt_toolkit/export_to_trt.py")
    if toolkit.exists():
        cmd = [
            sys.executable,
            str(toolkit),
            "--onnx",
            str(onnx_path),
            "--output",
            str(trt_path),
            "--precision",
            precision,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0 and trt_path.exists():
            print(f"[EXPORT] TRT {precision}: {trt_path} ({trt_path.stat().st_size / 1e6:.1f} MB)")
            return trt_path

    # Fallback: try trtexec directly
    trtexec = "/usr/bin/trtexec"
    if not Path(trtexec).exists():
        trtexec = "trtexec"
    precision_flag = "--fp16" if precision == "fp16" else "--noTF32"
    cmd = [trtexec, f"--onnx={onnx_path}", f"--saveEngine={trt_path}", precision_flag]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0 and trt_path.exists():
            print(f"[EXPORT] TRT {precision}: {trt_path} ({trt_path.stat().st_size / 1e6:.1f} MB)")
            return trt_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    print(f"[EXPORT] TRT {precision}: SKIPPED (no trtexec or toolkit available)")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Export MANIP-ULTRADEX policy — full pipeline")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--output-dir", default="/mnt/artifacts-datai/exports/project_manip_ultradex"
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    args = parser.parse_args()

    cfg = load_module_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load policy
    policy = UltraDexPolicy(
        cfg, d_model=args.d_model, num_heads=args.num_heads, num_layers=args.num_layers
    )
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        policy.load_state_dict(state_dict, strict=False)
    policy.eval()

    print(f"[EXPORT] Module: {cfg.project.codename}")
    print(f"[EXPORT] Params: {policy.parameter_count:,}")
    print(f"[EXPORT] Output: {output_dir}")
    print()

    t0 = time.monotonic()
    results = {}

    # 1. PyTorch checkpoint
    results["pth"] = str(export_pth(policy, output_dir, cfg))

    # 2. Safetensors
    st_path = export_safetensors(policy, output_dir)
    results["safetensors"] = str(st_path) if st_path else None

    # 3. ONNX
    exportable = ExportablePolicy(policy)
    onnx_path = export_onnx(exportable, output_dir, cfg)
    results["onnx"] = str(onnx_path) if onnx_path else None

    # 4. TensorRT FP16 (MANDATORY)
    if onnx_path:
        trt16 = export_trt(onnx_path, output_dir, "fp16")
        results["trt_fp16"] = str(trt16) if trt16 else None
    else:
        results["trt_fp16"] = None

    # 5. TensorRT FP32 (MANDATORY)
    if onnx_path:
        trt32 = export_trt(onnx_path, output_dir, "fp32")
        results["trt_fp32"] = str(trt32) if trt32 else None
    else:
        results["trt_fp32"] = None

    elapsed = time.monotonic() - t0
    print(f"\n[EXPORT] Done in {elapsed:.1f}s")
    print(json.dumps(results, indent=2))

    # Write export manifest
    manifest = output_dir / "export_manifest.json"
    manifest.write_text(json.dumps(results, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
