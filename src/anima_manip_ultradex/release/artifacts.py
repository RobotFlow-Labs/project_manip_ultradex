"""Export manifest and artifact validation for production release."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


REQUIRED_FORMATS = ("pth", "safetensors", "onnx", "trt_fp16", "trt_fp32")


@dataclass
class ExportArtifact:
    format: str
    path: str
    size_bytes: int = 0
    exists: bool = False


@dataclass
class ReleaseManifest:
    module: str = "manip_ultradex"
    version: str = "0.1.0"
    paper: str = "arXiv:2603.05312"
    artifacts: list[ExportArtifact] = field(default_factory=list)
    benchmark_summary: dict | None = None
    ready: bool = False

    def check_completeness(self) -> list[str]:
        missing = []
        found_formats = {a.format for a in self.artifacts if a.exists}
        for fmt in REQUIRED_FORMATS:
            if fmt not in found_formats:
                missing.append(f"Missing export format: {fmt}")
        if self.benchmark_summary is None:
            missing.append("No benchmark summary attached")
        return missing

    def is_complete(self) -> bool:
        return len(self.check_completeness()) == 0


def scan_export_dir(export_dir: str | Path) -> list[ExportArtifact]:
    d = Path(export_dir)
    artifacts = []
    suffixes = {
        ".pth": "pth",
        ".safetensors": "safetensors",
        ".onnx": "onnx",
    }
    for f in d.iterdir() if d.exists() else []:
        if f.suffix in suffixes:
            artifacts.append(
                ExportArtifact(
                    format=suffixes[f.suffix],
                    path=str(f),
                    size_bytes=f.stat().st_size,
                    exists=True,
                )
            )
        elif f.suffix == ".engine" or f.suffix == ".plan":
            fmt = "trt_fp16" if "fp16" in f.stem else "trt_fp32"
            artifacts.append(
                ExportArtifact(
                    format=fmt,
                    path=str(f),
                    size_bytes=f.stat().st_size,
                    exists=True,
                )
            )
    return artifacts


def write_manifest(manifest: ReleaseManifest, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(manifest), indent=2))
    return out


def read_manifest(path: str | Path) -> ReleaseManifest:
    payload = json.loads(Path(path).read_text())
    artifacts = [ExportArtifact(**a) for a in payload.pop("artifacts", [])]
    return ReleaseManifest(**payload, artifacts=artifacts)
