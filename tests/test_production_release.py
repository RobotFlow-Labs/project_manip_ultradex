"""PRD-07 tests: production release validation."""

from anima_manip_ultradex.release.artifacts import (
    ExportArtifact,
    REQUIRED_FORMATS,
    ReleaseManifest,
    write_manifest,
    read_manifest,
)
from anima_manip_ultradex.release.health import run_health_checks


def test_release_manifest_roundtrip(tmp_path) -> None:
    manifest = ReleaseManifest(
        artifacts=[
            ExportArtifact(format="pth", path="/fake/model.pth", size_bytes=1000, exists=True),
            ExportArtifact(format="onnx", path="/fake/model.onnx", size_bytes=2000, exists=True),
        ],
        benchmark_summary={"overall_success": 0.84},
    )
    path = write_manifest(manifest, tmp_path / "manifest.json")
    loaded = read_manifest(path)
    assert loaded.module == "manip_ultradex"
    assert len(loaded.artifacts) == 2
    assert loaded.benchmark_summary["overall_success"] == 0.84


def test_completeness_check_flags_missing_formats() -> None:
    manifest = ReleaseManifest(
        artifacts=[
            ExportArtifact(format="pth", path="/fake", exists=True),
        ],
    )
    missing = manifest.check_completeness()
    assert len(missing) > 0
    assert any("safetensors" in m for m in missing)
    assert any("benchmark" in m.lower() for m in missing)


def test_complete_manifest_passes() -> None:
    artifacts = [
        ExportArtifact(format=fmt, path=f"/fake/model.{fmt}", exists=True)
        for fmt in REQUIRED_FORMATS
    ]
    manifest = ReleaseManifest(
        artifacts=artifacts,
        benchmark_summary={"overall_success": 0.82},
    )
    assert manifest.is_complete()
    assert len(manifest.check_completeness()) == 0


def test_health_checks_run_without_crash() -> None:
    health = run_health_checks()
    summary = health.summary()
    assert "all_passed" in summary
    assert "critical_passed" in summary
    assert len(summary["checks"]) >= 3


def test_health_checks_detect_torch() -> None:
    health = run_health_checks()
    torch_check = next(c for c in health.checks if c.name == "torch_available")
    assert torch_check.passed is True  # torch is installed in this venv
