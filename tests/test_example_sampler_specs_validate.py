"""Every engine sampler spec shipped with the repository builds its engine (the
adapter's own checks run, the server is not started), and no shipped spec or guide
snippet sets a --gpu-memory-utilization the ROCm K3.1 presets refuse."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from yeto.rl.adapters.sampler_engines.vllm_server import K31_ROCM_MAX_GPU_MEMORY_UTILIZATION, gpu_memory_utilization
from yeto.rl.engine import engine_sampler_entry as entry
from yeto.rl.engine.sampler_engine import SamplerEngineError

ROOT = Path(__file__).resolve().parents[1]
SPEC_DIRS = ("examples", "tutorial", "docs")


def _filled(doc, tmp_path: Path):
    """Angle-bracket placeholders replaced by scratch paths (as test_rl_k31_e2e_cpu)."""
    if isinstance(doc, dict):
        return {k: _filled(v, tmp_path) for k, v in doc.items() if k != "_comment"}
    if isinstance(doc, list):
        return [_filled(v, tmp_path) for v in doc]
    if isinstance(doc, str) and "<" in doc:
        return str(tmp_path / doc.replace("<", "").replace(">", "").replace("/", "_").replace(" ", "-"))
    return doc


def _engine_specs() -> list[Path]:
    out = []
    for name in SPEC_DIRS:
        for path in sorted((ROOT / name).rglob("*.json")):
            try:
                doc = json.loads(path.read_text())
            except (ValueError, UnicodeDecodeError):
                continue
            if isinstance(doc, dict) and isinstance(doc.get("engine"), dict) and doc["engine"].get("adapter"):
                out.append(path)
    return out


SPECS = _engine_specs()


def test_the_repository_ships_engine_sampler_specs():
    names = {p.name for p in SPECS}
    assert {"sampler-spec.nc-mi355x-vllm.json", "sampler-spec.nc-mi355x.json", "sampler-spec.cpu-fake.json"} <= names


@pytest.mark.parametrize("path", SPECS, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_shipped_engine_spec_builds_and_passes_the_engine_checks(path, tmp_path):
    spec = _filled(json.loads(path.read_text()), tmp_path)
    engine = entry.build_engine(spec)  # raises on a value the adapter refuses (e.g. a too high HBM budget)
    extra = ((spec["engine"].get("server") or {}).get("extra_args")) or []
    gmu = gpu_memory_utilization(extra)
    assert gmu is None or gmu <= K31_ROCM_MAX_GPU_MEMORY_UTILIZATION, path
    assert engine is not None


def test_a_shipped_rocm_spec_with_the_old_088_budget_is_refused(tmp_path):
    """The check above is not vacuous: the measured-to-fail 0.88 in a shipped ROCm
    template makes the engine refuse it."""
    path = ROOT / "examples/k31/sampler-spec.nc-mi355x-vllm.json"
    spec = _filled(json.loads(path.read_text()), tmp_path)
    spec["engine"]["server"]["extra_args"] = ["--gpu-memory-utilization", "0.88"]
    with pytest.raises(SamplerEngineError, match="above 0.86"):
        entry.build_engine(spec)


GUIDE_GMU = re.compile(r"--gpu-memory-utilization[\"']?\s*[,= ]\s*[\"']?(0\.\d+)")


def test_no_shipped_snippet_sets_a_refused_gpu_memory_utilization():
    """Specs, launch drafts and guide snippets that a customer copies (a value quoted
    as a measurement in prose, ``at 0.88 ...``, is not a flag and is not matched)."""
    bad = []
    for name in SPEC_DIRS:
        for path in sorted((ROOT / name).rglob("*")):
            if path.suffix not in (".json", ".md", ".sh", ".yaml", ".yml") or not path.is_file():
                continue
            for value in GUIDE_GMU.findall(path.read_text(errors="replace")):
                if float(value) > K31_ROCM_MAX_GPU_MEMORY_UTILIZATION:
                    bad.append(f"{path.relative_to(ROOT)}: {value}")
    assert bad == []
