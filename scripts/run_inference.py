#!/usr/bin/env python3
"""Offline inference CLI for MANIP-ULTRADEX."""

from __future__ import annotations

import argparse
import json

import numpy as np

from anima_manip_ultradex.config import load_module_config
from anima_manip_ultradex.inference.runner import UltraDexRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MANIP-ULTRADEX offline inference")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--synthetic-points", type=int, default=4096)
    parser.add_argument("--use-sor", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_module_config(args.config)
    runner = UltraDexRunner(
        cfg, checkpoint_path=args.checkpoint, d_model=64, num_heads=4, num_layers=2
    )

    raw_pc = np.random.default_rng(7).normal(size=(args.synthetic_points, 3)).astype(np.float32)
    robot_pc = np.random.default_rng(17).normal(size=(256, 3)).astype(np.float32) * 0.1
    result = runner.predict(raw_pc, robot_pc=robot_pc, apply_sor=args.use_sor)

    summary = {
        "scene_input_shape": list(result.scene_input.shape),
        "action_vector_shape": list(result.action_vector.shape),
        "arm_actions_shape": list(result.arm_actions.shape),
        "hand_actions_shape": list(result.hand_actions.shape),
        "checkpoint_loaded": result.checkpoint_loaded,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
