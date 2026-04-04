"""Release bundling script for MANIP-ULTRADEX.

Usage:
    python scripts/package_release.py --export-dir /path/to/exports --output release_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_manip_ultradex.release.artifacts import (
    ReleaseManifest,
    scan_export_dir,
    write_manifest,
)
from anima_manip_ultradex.release.health import run_health_checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Package MANIP-ULTRADEX release")
    parser.add_argument(
        "--export-dir", type=str, default="/mnt/artifacts-datai/exports/project_manip_ultradex"
    )
    parser.add_argument("--benchmark", type=str, default=None, help="Path to benchmark JSON")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="release_manifest.json")
    args = parser.parse_args()

    # Scan exports
    artifacts = scan_export_dir(args.export_dir)
    print(f"[RELEASE] Found {len(artifacts)} export artifacts")
    for a in artifacts:
        print(f"  {a.format}: {a.path} ({a.size_bytes / 1e6:.1f} MB)")

    # Load benchmark
    benchmark = None
    if args.benchmark and Path(args.benchmark).exists():
        benchmark = json.loads(Path(args.benchmark).read_text())
        print(f"[RELEASE] Benchmark loaded: {benchmark.get('overall_success', 'N/A')}")

    # Health checks
    health = run_health_checks(args.checkpoint)
    print(f"[RELEASE] Health: critical={'PASS' if health.critical_passed else 'FAIL'}")

    # Build manifest
    manifest = ReleaseManifest(
        artifacts=artifacts,
        benchmark_summary=benchmark,
        ready=health.critical_passed and len(artifacts) > 0,
    )

    missing = manifest.check_completeness()
    if missing:
        print(f"[RELEASE] WARNING: {len(missing)} missing items:")
        for m in missing:
            print(f"  - {m}")

    out = write_manifest(manifest, args.output)
    print(f"[RELEASE] Manifest written to {out}")


if __name__ == "__main__":
    main()
