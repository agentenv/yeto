"""Focused command and lifecycle tests for compare_diloco action probing."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "compare_diloco_action_probe", ROOT / "scripts" / "compare_diloco.py"
)
compare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compare
SPEC.loader.exec_module(compare)


def _args(**overrides):
    values = {
        "model": "qwen35-4b",
        "lora_r": 2,
        "lora_alpha": 4,
        "action_probe_gpus": [4, 5],
        "action_probe_anchor_manifest": Path("/tmp/anchor.manifest.json"),
        "action_probe_seq_len": 128,
        "action_probe_panels": 8,
        "action_probe_blocks_per_panel": 2,
        "action_probe_lora_targets": "auto",
        "action_probe_startup_timeout_s": 120.0,
        "action_probe_timeout_s": 30.0,
        "action_probe_run_uuid": "run-fixed",
        "action_probe_min_gain": None,
        "action_probe_lcb_z": None,
        "action_probe_min_win_rate": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _ready(endpoint="127.0.0.1:49000"):
    fragment_layout = {
        str(fragment): [
            f"base_model.model.layers.{fragment}.q_proj.lora_A.default.weight"
        ]
        for fragment in range(4)
    }
    return {
        "listen": endpoint,
        "protocol": "yeto-action-probe-v1",
        "backend": {
            "healthy": True,
            "workers": [{"lora_r": 2}, {"lora_r": 2}],
            "anchor_manifest_sha256": "a" * 64,
            "anchor_tensors_sha256": "d" * 64,
            "probe_config_sha256": "b" * 64,
            "layout_hash": "c" * 64,
            "fragment_layout": fragment_layout,
        },
    }


def test_probe_shadow_and_active_syncer_commands_carry_exact_contract(tmp_path):
    config = tmp_path / "expected.json"
    config.write_text("{}\n")
    for policy in (
        "probe_shadow",
        "probe_loo_v1",
        "probe_lr_shadow",
        "probe_lr_v1",
    ):
        arm = compare.PRESETS[policy]
        cmd = compare.syncer_command(
            arm,
            29400,
            tmp_path / policy,
            total_steps=16,
            action_probe_endpoint="127.0.0.1:49000",
            action_probe_timeout_ms=2500,
            action_probe_run_uuid=f"run-{policy}",
            action_probe_expected_config=config,
        )
        assert cmd[cmd.index("--commit-policy") + 1] == policy
        assert cmd[cmd.index("--action-probe-endpoint") + 1] == "127.0.0.1:49000"
        assert cmd[cmd.index("--action-probe-timeout-ms") + 1] == "2500"
        assert cmd[cmd.index("--action-probe-run-uuid") + 1] == f"run-{policy}"
        assert cmd[cmd.index("--action-probe-expected-config") + 1] == str(config)

    baseline = compare.syncer_command(
        compare.PRESETS["m4"], 29400, tmp_path / "m4", total_steps=16
    )
    assert baseline[baseline.index("--commit-policy") + 1] == "token_weighted"
    assert "--action-probe-endpoint" not in baseline


def test_sidecar_command_uses_probe_only_visible_gpu_indices():
    cmd = compare.action_probe_command(
        _args(action_probe_gpus=[6, 7, 9]),
        compare.PRESETS["probe_shadow"],
        "127.0.0.1:49000",
    )
    assert cmd[:3] == [sys.executable, "-m", "yeto.action_probe_server"]
    assert cmd[cmd.index("--gpus") + 1] == "0,1,2"
    assert cmd[cmd.index("--fragments") + 1] == "4"
    assert cmd[cmd.index("--fragment-pattern") + 1] == "binpack"


def test_scalar_presets_and_sidecar_command_use_predeclared_selector_defaults():
    assert compare.PRESETS["probe_lr_shadow"].commit_policy == "probe_lr_shadow"
    assert compare.PRESETS["probe_lr_v1"].commit_policy == "probe_lr_v1"

    cmd = compare.action_probe_command(
        _args(), compare.PRESETS["probe_lr_v1"], "127.0.0.1:49000"
    )
    assert cmd[cmd.index("--min-gain") + 1] == "0.00025"
    assert cmd[cmd.index("--lcb-z") + 1] == "2.365"
    assert cmd[cmd.index("--min-win-rate") + 1] == "0.75"

    overridden = compare.action_probe_command(
        _args(
            action_probe_min_gain=0.001,
            action_probe_lcb_z=1.25,
            action_probe_min_win_rate=0.8,
        ),
        compare.PRESETS["probe_lr_shadow"],
        "127.0.0.1:49000",
    )
    assert overridden[overridden.index("--min-gain") + 1] == "0.001"
    assert overridden[overridden.index("--lcb-z") + 1] == "1.25"
    assert overridden[overridden.index("--min-win-rate") + 1] == "0.8"


def test_expected_config_is_pinned_to_ready_sidecar_and_arm():
    config = compare.expected_probe_config(
        _ready(), compare.PRESETS["probe_loo_v1"], _args()
    )
    assert config == {
        "protocol": "yeto-action-probe-v1",
        "anchor_manifest_sha256": "a" * 64,
        "anchor_tensors_sha256": "d" * 64,
        "probe_config_sha256": "b" * 64,
        "layout_hash": "c" * 64,
        "fragment_pattern": "binpack",
        "lora_r": 2,
        "fragment_layout": _ready()["backend"]["fragment_layout"],
    }


def test_launch_readiness_and_clean_termination_lifecycle(tmp_path, monkeypatch):
    calls = {}

    class FakeProcess:
        def __init__(self):
            self.returncode = None
            self.terminated = 0
            self.killed = 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated += 1

        def kill(self):
            self.killed += 1
            self.returncode = -9

        def wait(self, timeout=None):
            calls.setdefault("wait_timeouts", []).append(timeout)
            self.returncode = 0
            return 0

    process = FakeProcess()

    def fake_popen(cmd, **kwargs):
        calls["cmd"] = cmd
        calls["env"] = kwargs["env"]
        return process

    monkeypatch.setattr(compare, "free_port", lambda: 49000)
    monkeypatch.setattr(compare.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        compare,
        "_read_action_probe_ready",
        lambda log_path, proc, timeout: _ready(),
    )
    monkeypatch.setattr(
        compare,
        "ping_action_probe",
        lambda endpoint, timeout: calls.setdefault("ping", (endpoint, timeout)),
    )

    sidecar = compare.launch_action_probe(
        _args(), compare.PRESETS["probe_shadow"], tmp_path
    )
    assert sidecar.endpoint == "127.0.0.1:49000"
    assert sidecar.run_uuid == "run-fixed"
    assert calls["env"]["CUDA_VISIBLE_DEVICES"] == "4,5"
    assert calls["cmd"][calls["cmd"].index("--gpus") + 1] == "0,1"
    assert calls["ping"][0] == "127.0.0.1:49000"
    expected = json.loads(sidecar.expected_config.read_text())
    assert expected["layout_hash"] == "c" * 64

    compare.stop_action_probe(sidecar)
    assert process.terminated == 1
    assert process.killed == 0
    assert sidecar.log_handle.closed


def test_launch_failure_terminates_sidecar_and_closes_log(tmp_path, monkeypatch):
    class FakeProcess:
        def __init__(self):
            self.returncode = None
            self.terminated = False

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

        def kill(self):
            self.returncode = -9

    process = FakeProcess()
    monkeypatch.setattr(compare, "free_port", lambda: 49000)
    monkeypatch.setattr(compare.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        compare,
        "_read_action_probe_ready",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("not ready")),
    )

    with pytest.raises(RuntimeError, match="not ready"):
        compare.launch_action_probe(_args(), compare.PRESETS["probe_shadow"], tmp_path)
    assert process.terminated is True
    # The parent descriptor is closed even though startup never completed.
    assert (tmp_path / "action_probe.log").exists()


def test_cli_rejects_overlapping_learner_and_probe_gpu_sets(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_diloco.py",
            "--data",
            "unused.jsonl",
            "--settings",
            "probe_shadow",
            "--device",
            "cuda",
            "--gpu-slots",
            "4",
            "--action-probe-gpus",
            "3,4",
            "--action-probe-anchor-manifest",
            "/tmp/anchor.manifest.json",
        ],
    )
    with pytest.raises(SystemExit, match="must be disjoint"):
        compare.main()


def test_cli_accepts_primary_scalar_probe_policies_in_dry_run(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_diloco.py",
            "--data",
            "unused.jsonl",
            "--settings",
            "probe_lr_shadow,probe_lr_v1",
            "--dry-run",
        ],
    )
    assert compare.main() == 0
