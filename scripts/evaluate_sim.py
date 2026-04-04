#!/usr/bin/env python3
"""Simulation benchmark CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from anima_manip_ultradex.config import load_module_config
from anima_manip_ultradex.eval.benchmark import SimulationBenchmark
from anima_manip_ultradex.eval.metrics import PAPER_TARGETS, render_markdown_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MANIP-ULTRADEX on the sim benchmark")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_module_config(args.config)
    manifest_path = (
        Path(args.manifest) if args.manifest else (cfg.sim_benchmark_root / "manifest.json")
    )
    if not manifest_path.exists():
        print(f"Simulation manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    summary = SimulationBenchmark(manifest_path).run()
    if args.format == "markdown":
        print(
            render_markdown_report("Simulation", summary, paper_target=PAPER_TARGETS["sim_overall"])
        )
    else:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
