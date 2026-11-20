"""Model store hit: every node resolves the snapshot from its local HF cache, never the Hub.

mfu16 a0 (2026-11-18): a 2-node Modal island served ``yeto-local/Kimi-K3.1-Pro-16layer``
(a truncated checkpoint that exists only on the model Volume) through the store env, but
the worker node's foreground snapshot_download asked the Hub and exited "Repository Not
Found"; the head's learner snapshot_download has the same Hub lookup."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from yeto import launcher
from yeto.gpu_spec import parse_gpu_spec
from yeto.models import MODEL_STORE_HIT_ENV, snapshot_path

REPO = Path(__file__).resolve().parents[1]
STORE_ONLY = "yeto-local/Kimi-store-only"
REV = "c6c0919e47f94c6041873f030148ef8c07f8e4d5"


def _recorder(result="/snap", raises=None):
    calls = []

    def download(**kw):
        calls.append(kw)
        if raises is not None and kw.get("local_files_only"):
            raise raises
        return result

    return calls, download


def test_store_hit_reads_the_local_cache_only():
    calls, download = _recorder()
    assert snapshot_path("org/m", REV, environ={MODEL_STORE_HIT_ENV: "1"}, download=download) == "/snap"
    assert calls == [{"repo_id": "org/m", "revision": REV, "local_files_only": True}]


def test_no_store_hit_keeps_the_hub_path():
    calls, download = _recorder()
    snapshot_path("org/m", REV, environ={}, download=download)
    snapshot_path("org/m", REV, environ={MODEL_STORE_HIT_ENV: ""}, download=download)
    assert calls == [{"repo_id": "org/m", "revision": REV}] * 2


def test_store_hit_without_the_snapshot_falls_back_to_the_hub(capsys):
    calls, download = _recorder(raises=FileNotFoundError("no snapshot"))
    assert snapshot_path("org/m", REV, environ={MODEL_STORE_HIT_ENV: "1"}, download=download) == "/snap"
    assert [c.get("local_files_only", False) for c in calls] == [True, False]
    assert "asking the Hub" in capsys.readouterr().err


def _store_cache(tmp_path: Path) -> Path:
    """The cache modal_volume_store_env builds: models--*/snapshots/<rev> -> Volume dir."""
    volume = tmp_path / "volume" / "hf" / "Kimi-store-only" / REV[:8]
    volume.mkdir(parents=True)
    (volume / "config.json").write_text("{}")
    cache = tmp_path / "hub"
    snaps = cache / f"models--{STORE_ONLY.replace('/', '--')}" / "snapshots"
    snaps.mkdir(parents=True)
    (snaps / REV).symlink_to(volume)
    return cache


@pytest.mark.skipif(importlib.util.find_spec("huggingface_hub") is None, reason="needs huggingface_hub")
def test_real_huggingface_hub_store_only_repo_needs_no_hub(tmp_path):
    cache = _store_cache(tmp_path)
    env = {**os.environ, "HF_HUB_CACHE": str(cache), MODEL_STORE_HIT_ENV: "1",
           "HF_ENDPOINT": "http://127.0.0.1:9",  # any Hub call would fail
           "PYTHONPATH": str(REPO)}
    env.pop("HF_HUB_OFFLINE", None)
    out = subprocess.run([sys.executable, "-c", "from yeto.models import snapshot_path; "
                          f"print(snapshot_path({STORE_ONLY!r}, {REV!r}))"],
                         env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert Path(out.stdout.strip(), "config.json").is_file()


def _two_node_task(monkeypatch):
    from test_rl_launcher import _Resources, _Storage, _StorageMode, _Task, _args, _prepare_rl_args

    monkeypatch.setitem(sys.modules, "sky", types.SimpleNamespace(
        Task=_Task, Resources=_Resources, Storage=_Storage, StorageMode=_StorageMode))
    args = _args(("--gpu", "nebius:2x4xl40s@eu-north1", "--rl-engine", "ports",
                  "--rollout-num-gpus-per-engine", "4", "--tensor-parallel", "1",
                  "--rollout-batch-size", "4", "--n-samples-per-prompt", "8"))
    args.model_revision, args.data_revision = REV, "b" * 40
    args.source_sha256, args.reward_sha256 = "c" * 64, "d" * 64
    _prepare_rl_args(args)
    args.model = STORE_ONLY
    return launcher.make_miles_island_task(args, parse_gpu_spec(args.gpu)[0], 0, 1, "127.0.0.1:29400")


def _worker_fetch_line(run: str) -> str:
    lines = [ln for ln in run.splitlines() if "worker could not fetch the model snapshot" in ln]
    assert len(lines) == 1, lines
    return lines[0]


def test_worker_fetch_uses_the_store_aware_lookup(monkeypatch):
    line = _worker_fetch_line(_two_node_task(monkeypatch).run)
    assert "from yeto.models import snapshot_path" in line and STORE_ONLY in line and REV in line
    assert "snapshot_download(repo_id=" not in line
    assert "PYTHONPATH=$HOME/sky_workdir" in line


@pytest.mark.skipif(importlib.util.find_spec("huggingface_hub") is None, reason="needs huggingface_hub")
def test_worker_fetch_on_a_store_hit_passes_for_a_store_only_repo(monkeypatch, tmp_path):
    line = _worker_fetch_line(_two_node_task(monkeypatch).run)
    home = tmp_path / "home"
    home.mkdir()
    (home / "sky_workdir").symlink_to(REPO)
    shim = tmp_path / "bin"
    shim.mkdir()
    (shim / "python3").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (shim / "python3").chmod(0o755)
    env = {**os.environ, "HOME": str(home), "PATH": f"{shim}:{os.environ['PATH']}",
           "HF_HUB_CACHE": str(_store_cache(tmp_path)), MODEL_STORE_HIT_ENV: "1",
           "HF_ENDPOINT": "http://127.0.0.1:9"}
    env.pop("PYTHONPATH", None)
    env.pop("HF_HUB_OFFLINE", None)
    proc = subprocess.run(["bash", "-c", f"set -e\ncd {tmp_path}\n{line}\necho JOINS"], env=env,
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0 and "JOINS" in proc.stdout, proc.stderr
