#!/usr/bin/env python3
"""Foundation training entrypoint scaffold."""

from __future__ import annotations

import argparse
import sys

from anima_manip_ultradex.config import load_module_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MANIP-ULTRADEX training scaffold")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_module_config(args.config)

    if args.dry_run:
        try:
            from anima_manip_ultradex.policy.network import UltraDexPolicy
        except ModuleNotFoundError as exc:
            print(
                f"{cfg.project.codename} foundation ready, but training extras are missing: {exc.name}",
                file=sys.stderr,
            )
            return 1

        policy = UltraDexPolicy(cfg)
        print(
            f"{cfg.project.codename} training scaffold ready | backend={cfg.compute.backend} "
            f"| policy_params={policy.parameter_count}"
        )
        return 0

    print(
        "Training is not implemented yet. Complete PRD-02 and PRD-03, and populate datasets "
        f"under {cfg.ultradexgrasp_dataset_root} before launching training.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
