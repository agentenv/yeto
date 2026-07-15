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


_UNSET = object()


def _responder_objects(ids=(0, 1, 2, 3)):
    c_steps = 64
    c_tokens = 8192
    return [
        {
            "id": learner_id,
            "base_version": 0,
            "c_steps": c_steps,
            "c_tokens": c_tokens,
            "weight": c_tokens**2 / c_steps,
        }
        for learner_id in ids
    ]


def _probe_record(
    policy,
    *,
    step=1,
    selected_action="A1",
    selected_multiplier=_UNSET,
    committed_action=_UNSET,
    committed_multiplier=_UNSET,
    fallback=_UNSET,
    fallback_reason=_UNSET,
    latency_ms=10.0,
    digest_char="a",
    responders=None,
):
    if selected_multiplier is _UNSET:
        if policy in ("probe_lr_shadow", "probe_lr_v1"):
            selected_multiplier = compare.ACTION_PROBE_SCALAR_MULTIPLIERS[
                selected_action
            ]
        else:
            selected_multiplier = 1.0 if selected_action == "A0" else 1.1
    if committed_action is _UNSET:
        committed_action = (
            "A0"
            if policy in compare.ACTION_PROBE_SHADOW_POLICIES
            else selected_action
        )
    if committed_multiplier is _UNSET:
        committed_multiplier = (
            1.0
            if policy in compare.ACTION_PROBE_SHADOW_POLICIES
            else selected_multiplier
        )
    if fallback is _UNSET:
        fallback = selected_action == "A0"
    if fallback_reason is _UNSET:
        fallback_reason = "no_action_passed" if fallback else None
    return {
        "step": step,
        "fragment": (step - 1) % 4,
        "responders": _responder_objects() if responders is None else responders,
        "policy": policy,
        "selected_action": selected_action,
        "committed_action": committed_action,
        "selected_multiplier": selected_multiplier,
        "committed_multiplier": committed_multiplier,
        "fallback": fallback,
        "fallback_reason": fallback_reason,
        "probe_latency_ms": latency_ms,
        "request_digest": digest_char * 64,
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


def test_cttn_shadow_pseudo_optimizer_maps_to_sgd_commit_policy(tmp_path):
    config = tmp_path / "expected.json"
    config.write_text("{}\n")
    arm = compare.apply_arm_overrides(
        [compare.PRESETS["m4"]], outer_optimizer="cttn-shadow"
    )[0]
    assert arm.commit_policy == "cttn_shadow_v1"
    assert arm.outer_momentum == 0.0
    cmd = compare.syncer_command(
        arm,
        29400,
        tmp_path / "shadow",
        total_steps=320,
        cttn_shadow_samples=32,
        action_probe_endpoint="127.0.0.1:49000",
        action_probe_timeout_ms=600_000,
        action_probe_run_uuid="run-shadow",
        action_probe_expected_config=config,
    )
    assert cmd[cmd.index("--outer-optimizer") + 1] == "nesterov"
    assert cmd[cmd.index("--outer-momentum") + 1] == "0.0"
    assert cmd[cmd.index("--commit-policy") + 1] == "cttn_shadow_v1"
    assert cmd[cmd.index("--cttn-shadow-samples") + 1] == "32"


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


def test_probe_run_validation_writes_compact_summary_and_allows_abstention(
    tmp_path,
):
    arm = compare.PRESETS["probe_shadow"]
    records = [
        _probe_record(
            arm.commit_policy,
            step=1,
            selected_action="A2",
            selected_multiplier=1.2,
            latency_ms=10.0,
            digest_char="a",
        ),
        _probe_record(
            arm.commit_policy,
            step=2,
            selected_action="A0",
            latency_ms=30.0,
            digest_char="b",
        ),
    ]

    summary = compare.validate_action_probe_run(
        arm, records, tmp_path, expected_steps=2
    )
    summary_path = tmp_path / compare.ACTION_PROBE_SUMMARY_FILENAME
    assert json.loads(summary_path.read_text()) == summary
    assert len(summary_path.read_text().splitlines()) == 1
    assert summary["fallback_count"] == 1
    assert summary["fallback_reason_counts"] == {"no_action_passed": 1}
    assert summary["selected_action_counts"] == {"A0": 1, "A2": 1}
    assert summary["committed_action_counts"] == {"A0": 2}
    assert summary["probe_latency_ms"]["mean"] == pytest.approx(20.0)
    assert summary["probe_latency_ms"]["p95"] == pytest.approx(29.0)


@pytest.mark.parametrize(
    ("policy", "updates", "message"),
    [
        (
            "probe_shadow",
            {"committed_action": "A1", "committed_multiplier": 1.1},
            "shadow policy",
        ),
        (
            "probe_loo_v1",
            {"committed_action": "A0", "committed_multiplier": 1.0},
            "active policy",
        ),
        (
            "probe_loo_v1",
            {"committed_action": "A1", "committed_multiplier": 1.2},
            "active policy",
        ),
    ],
)
def test_probe_run_validation_enforces_commit_contract(
    tmp_path, policy, updates, message
):
    record = _probe_record(policy)
    record.update(updates)
    with pytest.raises(RuntimeError, match=message):
        compare.validate_action_probe_run(compare.PRESETS[policy], [record], tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("policy", "probe_lr_v1"),
        ("probe_latency_ms", 0.0),
        ("probe_latency_ms", float("inf")),
        ("request_digest", "g" * 64),
    ],
)
def test_probe_run_validation_rejects_malformed_records(tmp_path, field, value):
    arm = compare.PRESETS["probe_shadow"]
    record = _probe_record(arm.commit_policy)
    record[field] = value
    with pytest.raises(RuntimeError, match="malformed action-probe record"):
        compare.validate_action_probe_run(arm, [record], tmp_path)


def test_scalar_probe_record_must_match_predeclared_action_multiplier(tmp_path):
    arm = compare.PRESETS["probe_lr_v1"]
    record = _probe_record(
        arm.commit_policy,
        selected_action="A2",
        selected_multiplier=1.2,
        committed_multiplier=1.2,
    )
    with pytest.raises(RuntimeError, match="selected_multiplier must be exactly 1.125"):
        compare.validate_action_probe_run(arm, [record], tmp_path)


def test_probe_run_validation_enforces_full_responder_group(tmp_path):
    arm = compare.PRESETS["probe_shadow"]
    record = _probe_record(
        arm.commit_policy, responders=_responder_objects((0, 1, 2))
    )
    with pytest.raises(RuntimeError, match="invalid responders"):
        compare.validate_action_probe_run(arm, [record], tmp_path)


def test_event_tape_validation_normalizes_real_responder_object_shape():
    arm = compare.PRESETS["probe_shadow"]
    record = _probe_record(
        arm.commit_policy, responders=_responder_objects((2, 0, 3, 1))
    )
    compare.validate_event_tape_records(arm, [record], expected_steps=1)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda responders: responders[0].pop("weight"),
        lambda responders: responders[0].update(weight=-1.0),
        lambda responders: responders[0].update(c_steps=0),
    ],
)
def test_event_tape_validation_rejects_malformed_responder_objects(mutation):
    arm = compare.PRESETS["probe_shadow"]
    responders = _responder_objects()
    mutation(responders)
    record = _probe_record(arm.commit_policy, responders=responders)
    with pytest.raises(RuntimeError, match="malformed responder"):
        compare.validate_event_tape_records(arm, [record])


def test_probe_run_validation_rejects_all_transport_fallbacks(tmp_path):
    arm = compare.PRESETS["probe_loo_v1"]
    records = [
        _probe_record(
            arm.commit_policy,
            step=1,
            selected_action="A0",
            fallback_reason="probe_timeout",
            digest_char="a",
        ),
        _probe_record(
            arm.commit_policy,
            step=2,
            selected_action="A0",
            fallback_reason="probe_io_error",
            digest_char="b",
        ),
    ]
    with pytest.raises(RuntimeError, match="all 2 .*transport/config fallbacks"):
        compare.validate_action_probe_run(arm, records, tmp_path)
    summary = json.loads(
        (tmp_path / compare.ACTION_PROBE_SUMMARY_FILENAME).read_text()
    )
    assert summary["fallback_count"] == 2
    assert summary["transport_config_fallback_count"] == 2


def test_probe_run_validation_allows_all_no_action_passed_abstentions(tmp_path):
    arm = compare.PRESETS["probe_lr_v1"]
    records = [
        _probe_record(
            arm.commit_policy,
            step=1,
            selected_action="A0",
            digest_char="a",
        ),
        _probe_record(
            arm.commit_policy,
            step=2,
            selected_action="A0",
            digest_char="b",
        ),
    ]
    summary = compare.validate_action_probe_run(arm, records, tmp_path)
    assert summary["fallback_count"] == 2
    assert summary["fallback_reason_counts"] == {"no_action_passed": 2}
    assert summary["transport_config_fallback_count"] == 0


def test_reproducibility_metadata_records_command_commit_and_diff(
    tmp_path, monkeypatch
):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        stdout = "commit-123\n" if command[1] == "rev-parse" else "diff body\n"
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(compare.subprocess, "run", fake_run)
    report_dir = tmp_path / "run" / "report"
    argv = ["scripts/compare_diloco.py", "--data", "rows with spaces.jsonl"]
    run_root = compare.persist_reproducibility_metadata(report_dir, argv)

    assert run_root == report_dir.parent
    assert (run_root / "command.sh").read_text() == compare.shlex.join(argv) + "\n"
    assert (run_root / "git_commit.txt").read_text() == "commit-123\n"
    assert (run_root / "git_diff.patch").read_text() == "diff body\n"
    assert [call[0] for call in calls] == [
        ["git", "rev-parse", "HEAD"],
        ["git", "diff"],
    ]
    assert all(call[1]["cwd"] == compare.REPO_ROOT for call in calls)


def test_reproducibility_metadata_records_git_failure_without_failing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        compare.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(
            returncode=128, stdout="", stderr="not a git repository"
        ),
    )
    report_dir = tmp_path / "run" / "report"
    run_root = compare.persist_reproducibility_metadata(report_dir, ["compare"])
    assert "unavailable:" in (run_root / "git_commit.txt").read_text()
    assert "unavailable:" in (run_root / "git_diff.patch").read_text()


def test_reproducibility_metadata_write_failure_is_clear(tmp_path, monkeypatch):
    monkeypatch.setattr(
        compare.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(
            returncode=0, stdout="", stderr=""
        ),
    )
    blocked_root = tmp_path / "blocked"
    blocked_root.write_text("not a directory")
    with pytest.raises(RuntimeError, match="cannot write reproducibility metadata"):
        compare.persist_reproducibility_metadata(
            blocked_root / "report", ["compare"]
        )


def test_materialized_anchor_disjointness_writes_verified_summary(tmp_path):
    from yeto.action_probe import canonical_anchor_hash

    anchor_row = {"messages": [{"role": "user", "content": "anchor"}]}
    train_row = {"messages": [{"role": "user", "content": "train"}]}
    eval_row = {"messages": [{"role": "user", "content": "eval"}]}
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    train_path.write_text(json.dumps(train_row) + "\n")
    eval_path.write_text(json.dumps(eval_row) + "\n")
    summary_path = tmp_path / "anchor_overlap_check.json"

    summary = compare.validate_materialized_anchor_disjointness(
        anchor_hashes={canonical_anchor_hash(anchor_row)},
        data_files={"train": train_path, "eval": eval_path},
        summary_path=summary_path,
        manifest_sha256="a" * 64,
        anchor_data_sha256="b" * 64,
    )

    assert summary["verified_zero_overlap"] is True
    assert summary["overlap_count"] == 0
    assert summary["files"]["train"]["row_count"] == 1
    assert summary["files"]["eval"]["row_count"] == 1
    assert json.loads(summary_path.read_text()) == summary


def test_materialized_anchor_disjointness_fails_and_preserves_evidence(tmp_path):
    from yeto.action_probe import canonical_anchor_hash

    anchor_row = {"messages": [{"role": "user", "content": "shared"}]}
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    train_path.write_text(json.dumps(anchor_row) + "\n")
    eval_path.write_text(
        json.dumps({"messages": [{"role": "user", "content": "eval"}]})
        + "\n"
    )
    summary_path = tmp_path / "anchor_overlap_check.json"

    with pytest.raises(RuntimeError, match="overlaps the action-probe anchor"):
        compare.validate_materialized_anchor_disjointness(
            anchor_hashes={canonical_anchor_hash(anchor_row)},
            data_files={"train": train_path, "eval": eval_path},
            summary_path=summary_path,
            manifest_sha256="a" * 64,
            anchor_data_sha256="b" * 64,
        )

    summary = json.loads(summary_path.read_text())
    assert summary["verified_zero_overlap"] is False
    assert summary["overlap_count"] == 1
    assert summary["files"]["train"]["overlap_count"] == 1
