"""Benchmark harness abstractions for simulation and real-world evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from anima_manip_ultradex.eval.metrics import EvalRecord, summarize_records


def load_manifest(path: str | Path) -> list[EvalRecord]:
    payload = json.loads(Path(path).read_text())
    records = payload["records"] if isinstance(payload, dict) else payload
    return [EvalRecord(**record) for record in records]


class BenchmarkHarness:
    def __init__(self, manifest_path: str | Path, benchmark_name: str) -> None:
        self.manifest_path = Path(manifest_path)
        self.benchmark_name = benchmark_name

    def run(self) -> dict[str, object]:
        return summarize_records(load_manifest(self.manifest_path))


class SimulationBenchmark(BenchmarkHarness):
    def __init__(self, manifest_path: str | Path) -> None:
        super().__init__(manifest_path=manifest_path, benchmark_name="simulation")


class RealWorldBenchmark(BenchmarkHarness):
    def __init__(self, manifest_path: str | Path) -> None:
        super().__init__(manifest_path=manifest_path, benchmark_name="real_world")
