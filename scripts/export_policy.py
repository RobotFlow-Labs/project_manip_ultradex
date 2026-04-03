#!/usr/bin/env python3
"""Export helper for Torch and ONNX policy artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from anima_manip_ultradex.config import load_module_config
from anima_manip_ultradex.policy.network import UltraDexPolicy


class ExportablePolicy(nn.Module):
    def __init__(self, policy: UltraDexPolicy) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, scene_pc: torch.Tensor) -> torch.Tensor:
        return self.policy(scene_pc).action_mean


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export MANIP-ULTRADEX policy artifacts")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="artifacts/export")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_module_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = UltraDexPolicy(cfg, d_model=64, num_heads=4, num_layers=2)
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu")
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        policy.load_state_dict(state_dict, strict=False)
    policy.eval()
    exportable = ExportablePolicy(policy)

    torch_path = output_dir / "policy_latest.pt"
    torch.save({"state_dict": policy.state_dict()}, torch_path)

    dummy_input = torch.randn(1, cfg.paper.policy_input_points, 3)
    onnx_path = output_dir / "policy_latest.onnx"
    try:
        import onnx  # noqa: F401

        torch.onnx.export(
            exportable,
            dummy_input,
            onnx_path,
            input_names=["scene_pc"],
            output_names=["action_mean"],
            opset_version=17,
        )
        onnx_exported = True
    except ModuleNotFoundError:
        onnx_exported = False

    mlx_stub_path = output_dir / "policy_latest_mlx_stub.json"
    mlx_stub_path.write_text(
        json.dumps(
            {
                "module": cfg.project.codename,
                "source_checkpoint": args.checkpoint,
                "note": "MLX export hook placeholder. Convert Torch weights after architecture lock.",
            },
            indent=2,
        )
    )

    print(
        json.dumps(
            {
                "torch": str(torch_path),
                "onnx": str(onnx_path) if onnx_exported else None,
                "mlx_stub": str(mlx_stub_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
