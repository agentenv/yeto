"""The examples of docs/NODE_RUNTIME.md run as written: the inventory loads as profiles,
the catalog example resolves and refuses, the manual join writes an assignment whose
dry run prints a CUDA container plan, and the sky pool config is a valid pool config."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from yeto.launch.nc import cli
from yeto.launch.node_inventory import load_profiles

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "NODE_RUNTIME.md"


def block(name: str) -> str:
    found = re.search(rf"<!-- doc-test: {re.escape(name)} -->\n```\w+\n(.*?)```", DOC.read_text(), re.S)
    assert found, name
    return found.group(1)


def test_the_inventory_example_loads_as_profiles(tmp_path):
    (tmp_path / "nodes.yaml").write_text(block("inventory"))
    h100, mi355x = load_profiles(tmp_path / "nodes.yaml")
    assert h100.family == "cuda" and h100.paths.work == "/data/yeto/work" and h100.model_source == "store"
    assert mi355x.family == "rocm" and "/dev/kfd" in mi355x.devices().args


def test_the_catalog_example_resolves_and_refuses(capsys):
    exec(compile(block("catalog"), "catalog", "exec"), {})  # noqa: S102 - the doc's own example
    out = capsys.readouterr().out
    assert "vllm-cuda" in out and "refused: no engine image" in out


def test_the_manual_join_example_dry_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMO", str(tmp_path))
    monkeypatch.chdir(REPO)
    exec(compile(block("manual"), "manual", "exec"), {})  # noqa: S102
    env = {**os.environ, "DEMO": str(tmp_path), "PYTHONPATH": str(REPO), "PATH": f"{Path(sys.executable).parent}:"
           + os.environ.get("PATH", "")}
    script = block("dry-run").replace("python -m", f"{sys.executable} -m")
    out = subprocess.run(["bash", "-c", script], cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    plan = json.loads(out.stdout)
    argv = plan["argv"]
    assert argv[argv.index("--gpus") + 1] == "all" and "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" in argv
    assert f"{tmp_path}/work:/work" in argv and not argv[argv.index("--entrypoint") + 2].startswith("docker:")


def test_the_sky_pool_example_is_a_valid_pool_config():
    cfg = json.loads(block("sky-pool"))
    cli.pool_config(cfg)
    assert cfg["node_runtime"] is True


def test_the_node_health_commands_run_as_dry_runs():
    env = {**os.environ, "PYTHONPATH": str(REPO), "PATH": f"{Path(sys.executable).parent}:" + os.environ.get("PATH", "")}
    script = block("node-health").replace("python -m", f"{sys.executable} -m")
    out = subprocess.run(["bash", "-c", script], cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "nvidia-smi --query-gpu" in out.stdout and "ibstat" in out.stdout and "NVRM: Xid" in out.stdout
    assert "device family npu: no health checker" in out.stdout
