#!/usr/bin/env python3
"""UltraDexGrasp data generation pipeline.

Selects 1000 objects from DexGraspNet, generates synthetic grasp demonstrations
using mock sim (or BODex+cuRobo when available), and writes replay shards.

Usage:
    # Generate with mock sim (for pipeline validation)
    CUDA_VISIBLE_DEVICES=2 uv run python scripts/generate_data.py \
        --meshdata /mnt/forge-data/datasets/dexgraspnet/meshdata \
        --output /mnt/forge-data/datasets/manip-ultradex/ultradexgrasp_20m \
        --num-objects 1000 \
        --demos-per-object 20 \
        --points-per-scene 2048

    # Quick test (10 objects, 5 demos each)
    uv run python scripts/generate_data.py --num-objects 10 --demos-per-object 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from anima_manip_ultradex.config import load_module_config
from anima_manip_ultradex.data.demo_generator import DemoGenerator
from anima_manip_ultradex.data.replay_buffer import (
    ReplayRecord,
    ReplayShardManifest,
    write_manifest,
)
from anima_manip_ultradex.grasp.types import GraspCandidate, Pose7D


def select_objects(meshdata_root: Path, num_objects: int, seed: int = 42) -> list[Path]:
    """Select objects from DexGraspNet meshdata, stratified by category."""
    all_objects = sorted([d for d in meshdata_root.iterdir() if d.is_dir()])
    if len(all_objects) == 0:
        raise RuntimeError(f"No objects found in {meshdata_root}")

    rng = np.random.default_rng(seed)
    num_objects = min(num_objects, len(all_objects))

    # Group by category prefix (e.g., "core-bottle", "core-mug")
    categories: dict[str, list[Path]] = {}
    for obj in all_objects:
        parts = obj.name.rsplit("-", 1)
        cat = parts[0] if len(parts) > 1 else "unknown"
        categories.setdefault(cat, []).append(obj)

    # Stratified sampling: proportional to category size
    selected = []
    cat_items = list(categories.items())
    total = sum(len(v) for _, v in cat_items)
    for cat, objects in cat_items:
        n = max(1, round(num_objects * len(objects) / total))
        chosen = rng.choice(len(objects), min(n, len(objects)), replace=False)
        selected.extend(objects[i] for i in chosen)

    # Trim or pad to exact count
    rng.shuffle(selected)
    return selected[:num_objects]


def generate_synthetic_demo(
    obj_path: Path,
    cfg,
    demo_gen: DemoGenerator,
    rng: np.random.Generator,
    num_points: int = 2048,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate a synthetic (scene_pc, action) pair for one object.

    Uses random point cloud and mock grasp candidate until BODex is available.
    """
    # Generate a random scene point cloud centered on the object
    obj_center = rng.uniform(-0.3, 0.3, size=3).astype(np.float32)
    scene_pc = (rng.standard_normal((num_points, 3)) * 0.15 + obj_center).astype(np.float32)

    # Create a mock grasp candidate
    candidate = GraspCandidate(
        strategy="bimanual",
        object_id=obj_path.name,
        num_hands=2,
        wrist_pose=Pose7D(
            xyz=obj_center.tolist(),
            wxyz=[1.0, 0.0, 0.0, 0.0],
        ),
        hand_joints=[0.0] * 12,
        score=rng.uniform(0.5, 1.0),
    )

    # Generate 4-stage demo trajectory
    demo = demo_gen.generate(candidate)

    # Create action target (36-DoF: 2x6 arm + 2x12 hand)
    action = rng.uniform(-1, 1, size=36).astype(np.float32)

    metadata = {
        "object_id": obj_path.name,
        "strategy": candidate.strategy,
        "stages": demo.stage_names(),
        "score": candidate.score,
    }

    return scene_pc, action, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate UltraDexGrasp training data")
    parser.add_argument("--config", default="configs/paper.toml")
    parser.add_argument(
        "--meshdata",
        default="/mnt/forge-data/datasets/dexgraspnet/meshdata",
    )
    parser.add_argument(
        "--output",
        default="/mnt/forge-data/datasets/manip-ultradex/ultradexgrasp_20m",
    )
    parser.add_argument("--num-objects", type=int, default=1000)
    parser.add_argument("--demos-per-object", type=int, default=20)
    parser.add_argument("--points-per-scene", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-size", type=int, default=1000)
    args = parser.parse_args()

    cfg = load_module_config(args.config)
    meshdata_root = Path(args.meshdata)
    output_root = Path(args.output)

    if not meshdata_root.exists():
        print(f"[ERROR] Meshdata not found at {meshdata_root}")
        return 1

    # Select objects
    print(f"[DATA] Selecting {args.num_objects} objects from {meshdata_root}")
    objects = select_objects(meshdata_root, args.num_objects, seed=args.seed)
    print(f"[DATA] Selected {len(objects)} objects")

    # Create output directories
    points_dir = output_root / "points"
    actions_dir = output_root / "actions"
    manifests_dir = output_root / "manifests"
    for d in [points_dir, actions_dir, manifests_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Save selected object list
    obj_list_path = output_root / "selected_objects.json"
    obj_list_path.write_text(json.dumps([o.name for o in objects], indent=2))

    # Generate demos
    rng = np.random.default_rng(args.seed)
    demo_gen = DemoGenerator(cfg)
    total_demos = len(objects) * args.demos_per_object
    print(
        f"[DATA] Generating {total_demos} demos ({len(objects)} objects × {args.demos_per_object} demos)"
    )

    t0 = time.monotonic()
    records = []
    demo_idx = 0

    for obj_idx, obj_path in enumerate(objects):
        for d in range(args.demos_per_object):
            scene_pc, action, metadata = generate_synthetic_demo(
                obj_path,
                cfg,
                demo_gen,
                rng,
                args.points_per_scene,
            )

            # Save point cloud and action
            pc_path = f"points/{demo_idx:08d}.npy"
            np.save(points_dir / f"{demo_idx:08d}.npy", scene_pc)
            np.save(actions_dir / f"{demo_idx:08d}.npy", action)

            records.append(
                ReplayRecord(
                    scene_points_path=pc_path,
                    action=action.tolist(),
                    strategy=metadata["strategy"],
                    object_id=metadata["object_id"],
                    stage="full",
                    success=True,
                )
            )
            demo_idx += 1

        if (obj_idx + 1) % 100 == 0:
            elapsed = time.monotonic() - t0
            rate = (obj_idx + 1) / elapsed
            print(
                f"  [{obj_idx + 1}/{len(objects)}] {rate:.1f} objects/s, {demo_idx} demos generated"
            )

    # Write shard manifests
    shard_idx = 0
    for i in range(0, len(records), args.shard_size):
        shard_records = records[i : i + args.shard_size]
        manifest = ReplayShardManifest.from_records(f"shard-{shard_idx:04d}", shard_records)
        write_manifest(manifest, manifests_dir / f"shard-{shard_idx:04d}.json")
        shard_idx += 1

    elapsed = time.monotonic() - t0
    # Save dataset metadata
    meta = {
        "num_objects": len(objects),
        "demos_per_object": args.demos_per_object,
        "total_demos": total_demos,
        "num_shards": shard_idx,
        "shard_size": args.shard_size,
        "points_per_scene": args.points_per_scene,
        "seed": args.seed,
        "generation_time_s": elapsed,
        "note": "SYNTHETIC — mock grasp data. Replace with BODex+cuRobo pipeline for real training.",
    }
    (output_root / "metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"\n[DONE] {total_demos} demos in {shard_idx} shards ({elapsed:.1f}s)")
    print(f"  Output: {output_root}")
    print(f"  Objects: {obj_list_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
