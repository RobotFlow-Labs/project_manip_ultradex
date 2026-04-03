"""Evaluation metrics and report rendering for UltraDexGrasp."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import groupby
from operator import attrgetter
from typing import Iterable


PAPER_TARGETS = {
    "sim_overall": 0.84,
    "real_overall": 0.812,
}


@dataclass(frozen=True)
class EvalRecord:
    benchmark: str
    object_id: str
    success: bool
    split: str = "unknown"
    size_group: str = "unknown"
    ablation: str = "full"
    trial_id: str = "0"


def compute_success_rate(records: Iterable[EvalRecord]) -> float:
    rows = list(records)
    if not rows:
        return 0.0
    return sum(1.0 for record in rows if record.success) / len(rows)


def _group_by(records: list[EvalRecord], key_name: str) -> dict[str, float]:
    sorted_records = sorted(records, key=attrgetter(key_name))
    return {
        key: compute_success_rate(list(group))
        for key, group in groupby(sorted_records, key=attrgetter(key_name))
    }


def render_ablation_table(records: Iterable[EvalRecord]) -> list[dict[str, object]]:
    rows = list(records)
    return [
        {
            "ablation": ablation,
            "success_rate": compute_success_rate(list(group)),
            "num_trials": len(list(group_records)),
        }
        for ablation, group_records in groupby(
            sorted(rows, key=attrgetter("ablation")),
            key=attrgetter("ablation"),
        )
        for group in [list(group_records)]
    ]


def summarize_records(records: Iterable[EvalRecord]) -> dict[str, object]:
    rows = list(records)
    summary = {
        "overall_success": compute_success_rate(rows),
        "num_trials": len(rows),
        "by_split": _group_by(rows, "split"),
        "by_size_group": _group_by(rows, "size_group"),
        "ablation": render_ablation_table(rows),
        "records": [asdict(record) for record in rows],
    }
    return summary


def render_markdown_report(
    benchmark_name: str,
    summary: dict[str, object],
    *,
    paper_target: float | None = None,
) -> str:
    lines = [
        f"# {benchmark_name} Report",
        "",
        f"- Trials: {summary['num_trials']}",
        f"- Overall success: {summary['overall_success']:.3f}",
    ]
    if paper_target is not None:
        lines.append(f"- Paper target: {paper_target:.3f}")

    lines.extend(["", "## Split Breakdown"])
    for key, value in summary["by_split"].items():
        lines.append(f"- {key}: {value:.3f}")

    lines.extend(["", "## Size Breakdown"])
    for key, value in summary["by_size_group"].items():
        lines.append(f"- {key}: {value:.3f}")

    return "\n".join(lines)
