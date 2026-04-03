from anima_manip_ultradex.eval.benchmark import RealWorldBenchmark, SimulationBenchmark
from anima_manip_ultradex.eval.metrics import EvalRecord, render_markdown_report, summarize_records


def _records() -> list[EvalRecord]:
    return [
        EvalRecord(benchmark="sim", object_id="a", success=True, split="seen", size_group="small"),
        EvalRecord(benchmark="sim", object_id="b", success=False, split="seen", size_group="small"),
        EvalRecord(benchmark="sim", object_id="c", success=True, split="unseen", size_group="large"),
        EvalRecord(benchmark="sim", object_id="d", success=True, split="unseen", size_group="large"),
    ]


def test_metric_summary_keys() -> None:
    summary = summarize_records(_records())

    assert summary["num_trials"] == 4
    assert "overall_success" in summary
    assert summary["by_split"]["seen"] == 0.5
    assert summary["by_size_group"]["large"] == 1.0
    assert summary["ablation"][0]["ablation"] == "full"


def test_markdown_report_contains_group_breakdown() -> None:
    report = render_markdown_report("Simulation", summarize_records(_records()), paper_target=0.84)
    assert "# Simulation Report" in report
    assert "- seen: 0.500" in report
    assert "- large: 1.000" in report


def test_benchmark_harness_consumes_manifest(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        """
        {
          "records": [
            {"benchmark": "sim", "object_id": "a", "success": true, "split": "seen", "size_group": "small"},
            {"benchmark": "real", "object_id": "b", "success": false, "split": "unseen", "size_group": "medium"}
          ]
        }
        """.strip()
    )

    sim_summary = SimulationBenchmark(manifest_path).run()
    real_summary = RealWorldBenchmark(manifest_path).run()

    assert sim_summary["num_trials"] == 2
    assert real_summary["num_trials"] == 2
