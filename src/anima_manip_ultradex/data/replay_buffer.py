"""Dataset writer and reader for 20M-frame replay schema (§IV.B)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class ReplayRecord:
    scene_points_path: str
    action: list[float]
    strategy: str
    object_id: str
    stage: str
    success: bool


@dataclass
class ReplayShardManifest:
    shard_id: str
    num_frames: int
    records: list[ReplayRecord] = field(default_factory=list)

    @classmethod
    def from_records(cls, shard_id: str, records: Sequence[ReplayRecord]) -> ReplayShardManifest:
        return cls(shard_id=shard_id, num_frames=len(records), records=list(records))

    def supported_strategies(self) -> set[str]:
        return {r.strategy for r in self.records}


def write_manifest(manifest: ReplayShardManifest, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(manifest), indent=2))
    return out


def read_manifest(path: str | Path) -> ReplayShardManifest:
    payload = json.loads(Path(path).read_text())
    records = [ReplayRecord(**r) for r in payload.get("records", [])]
    return ReplayShardManifest(
        shard_id=payload["shard_id"],
        num_frames=payload["num_frames"],
        records=records,
    )
