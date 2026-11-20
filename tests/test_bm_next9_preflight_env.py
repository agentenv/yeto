"""The next9 overlay-safe image is pinned to its own Miles commit, and the cluster
preflight's host memory / disk limits follow the site overrides."""

from __future__ import annotations

import types

import pytest

from yeto.launch.preflight import cluster as cp
from yeto.launch.preflight import gate
from yeto.rl.adapters.miles import pins

NEXT9_DIGEST = "3a0c544558b67c587f460d1ab87c0c19fa1d578d5d77a9346a1729113a3b0225"


def test_next9_image_runs_its_own_miles_commit():
    ref = f"docker:registry.example/yeto-miles-k31@sha256:{NEXT9_DIGEST}"
    assert pins.image_miles_commit(ref) == pins.MILES_K31_NEXT9_COMMIT
    k31 = pins.MILES_LINEAGES["k31"]
    assert pins.lineage_of(pins.MILES_K31_NEXT9_COMMIT) is k31
    assert k31.commits[-2:] == (pins.MILES_K31_FULL_E_COMMIT, pins.MILES_K31_NEXT9_COMMIT)
    for feature in pins.K31_FEATURES:  # no new features: every k31 feature carries forward
        assert k31.has(feature, pins.MILES_K31_NEXT9_COMMIT), feature
    # already overlay-safe: nothing to repack
    assert pins.overlay_safe_image(ref) is None


def test_next9_is_the_k31_full_trainer_default():
    assert pins.MILES_K31_NEXT9_IMAGE == (
        f"docker:ghcr.io/agentenv/yeto-miles-k31@sha256:{NEXT9_DIGEST}")
    assert pins.default_rl_image(recipe="kimi_k31", tuning="full") == pins.MILES_K31_NEXT9_IMAGE
    assert pins.image_miles_commit(pins.MILES_K31_NEXT9_IMAGE) == pins.MILES_K31_NEXT9_COMMIT
    # next9 was measured on the full-parameter form only: LoRA and the other recipes keep theirs
    assert pins.default_rl_image(recipe="kimi_k31", tuning="lora") == pins.MILES_K31_IMAGE
    assert pins.default_rl_image() == pins.MILES_NEXT_IMAGE
    # the previous full default stays pinned for runs that name it
    assert pins.image_miles_commit(pins.MILES_K31_FULL_IMAGE) == pins.MILES_K31_FULL_E_COMMIT


def test_omitted_rl_image_resolves_to_next9_for_k31_full():
    from yeto.launch.rl_backend import resolve_default_rl_image

    args = types.SimpleNamespace(training_mode="rl", rl_backend=None, rl_image=None,
                                 rl_model_recipe="kimi_k31", tuning="full")
    resolve_default_rl_image(args)
    assert args.rl_image == pins.MILES_K31_NEXT9_IMAGE
    explicit = "docker:registry.example/yeto-miles-k31@sha256:" + "0" * 64
    args = types.SimpleNamespace(training_mode="rl", rl_backend=None, rl_image=explicit,
                                 rl_model_recipe="kimi_k31", tuning="full")
    resolve_default_rl_image(args)
    assert args.rl_image == explicit


def test_requirements_from_env():
    req = cp.requirements_from_env({})
    assert req == cp.Requirements()
    req = cp.requirements_from_env({"YETO_PREFLIGHT_MIN_HOST_MEMORY_GIB": "1400",
                                    "YETO_PREFLIGHT_MIN_DISK_FREE_GIB": " 700 "})
    assert (req.min_host_memory_gib, req.min_disk_free_gib) == (1400.0, 700.0)
    assert req.min_driver_major == cp.Requirements().min_driver_major
    for bad in ("lots", "-1", "nan"):
        with pytest.raises(ValueError, match="YETO_PREFLIGHT_MIN_DISK_FREE_GIB"):
            cp.requirements_from_env({"YETO_PREFLIGHT_MIN_DISK_FREE_GIB": bad})


def _capture(monkeypatch):
    seen = {}

    class Fake:
        def __init__(self, nodes, *, req=None, **kwargs):
            seen["req"] = req

        def run(self):
            return types.SimpleNamespace(ok=True, counts={"PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0}, checks=[],
                                         table=lambda: "", to_json=lambda: {})

    monkeypatch.setattr(cp, "ClusterPreflight", Fake)
    monkeypatch.setattr(cp, "resolve_nodes", lambda gpu, inventory: [])
    monkeypatch.setattr(cp, "trainer_image", lambda image, nodes: image)
    return seen


def test_preflight_cli_takes_the_env_limits_unless_a_flag_is_given(monkeypatch, tmp_path, capsys):
    seen = _capture(monkeypatch)
    monkeypatch.setenv("YETO_PREFLIGHT_MIN_HOST_MEMORY_GIB", "1500")
    monkeypatch.setenv("YETO_PREFLIGHT_MIN_DISK_FREE_GIB", "600")
    import argparse

    p = cp.add_parser(argparse.ArgumentParser().add_subparsers())
    args = p.parse_args(["--gpu", "ssh:6x8xb300", "--run-dir", str(tmp_path)])
    assert cp.main(args) == 0
    assert (seen["req"].min_host_memory_gib, seen["req"].min_disk_free_gib) == (1500.0, 600.0)
    args = p.parse_args(["--gpu", "ssh:6x8xb300", "--run-dir", str(tmp_path), "--min-disk-free-gib", "900"])
    assert cp.main(args) == 0
    assert (seen["req"].min_host_memory_gib, seen["req"].min_disk_free_gib) == (1500.0, 900.0)
    monkeypatch.setenv("YETO_PREFLIGHT_MIN_DISK_FREE_GIB", "x")
    assert cp.main(p.parse_args(["--gpu", "ssh:6x8xb300", "--run-dir", str(tmp_path)])) == 2


def test_launch_gate_uses_the_env_limits(monkeypatch, tmp_path):
    seen = {}

    class Fake:
        def __init__(self, nodes, *, req=None, **kwargs):
            seen["req"] = req

        def run(self):
            return types.SimpleNamespace(ok=True, counts={"PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0}, checks=[],
                                         table=lambda: "", to_json=lambda: {})

    monkeypatch.setattr(gate, "ClusterPreflight", Fake)
    monkeypatch.setattr(gate, "resolve_nodes", lambda entry, inventory: [])
    monkeypatch.setattr(gate, "trainer_image", lambda image, nodes: image)
    from yeto import runs

    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path)
    monkeypatch.setenv("YETO_PREFLIGHT_MIN_DISK_FREE_GIB", "650")
    args = types.SimpleNamespace(gpu="ssh:6x8xb300", rl_node_inventory="inv.yaml", cluster_prefix="t",
                                 rl_image=None, skip_preflight=False, dry_run=False)
    assert gate.launch_gate(args, out=open(tmp_path / "log", "w"))["reports"][0]["ok"]
    assert seen["req"].min_disk_free_gib == 650.0
    assert seen["req"].min_host_memory_gib == cp.Requirements().min_host_memory_gib
