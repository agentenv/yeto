"""Fail-closed tests for the frozen CPLG stock-shadow runner."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_cplg_shadow_pair", ROOT / "scripts" / "run_cplg_shadow_pair.py"
)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)

COMPARE_SPEC = importlib.util.spec_from_file_location(
    "compare_diloco_cplg_test", ROOT / "scripts" / "compare_diloco.py"
)
COMPARE_MOD = importlib.util.module_from_spec(COMPARE_SPEC)
assert COMPARE_SPEC.loader is not None
sys.modules[COMPARE_SPEC.name] = COMPARE_MOD
COMPARE_SPEC.loader.exec_module(COMPARE_MOD)
FROZEN_CONFIG = (
    ROOT / "experiments" / "optimizer" / "cplg-sgd-shadow-direction-r2-config.json"
)


def _publish(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(f"{path}.sha256").write_text(f"{digest}  {path.name}\n")


def _writer(records: int = 32, vector_bytes: int = 256) -> dict:
    return {
        "state": "closed",
        "accepted_items": records,
        "completed_items": records,
        "accepted_bytes": vector_bytes,
        "completed_bytes": vector_bytes,
        "dropped_items": 0,
        "dropped_bytes": 0,
        "abandoned_items": 0,
        "abandoned_bytes": 0,
        "pending_items": 0,
        "pending_bytes": 0,
        "error": None,
    }


def _frozen_compare_argv(config: dict) -> list[str]:
    workload = config["workload"]
    runtime = config["runtime"]
    capture = config["capture"]
    values = {
        "--model": runtime["model"],
        "--data": runtime["data"],
        "--settings": ",".join(workload["arms_in_order"]),
        "--seq-len": str(workload["sequence_length"]),
        "--micro-batch-size": str(workload["micro_batch_size"]),
        "--inner-lr": "0.001",
        "--lora-r": str(workload["lora_rank"]),
        "--lora-alpha": str(workload["lora_alpha"]),
        "--eval-rows": str(workload["evaluation_rows"]),
        "--max-rows": str(workload["max_rows"]),
        "--shuffle-rows-seed": str(workload["shuffle_rows_seed"]),
        "--training-seed": str(workload["training_seed"]),
        "--device": "cuda",
        "--gpu-slots": "1",
        "--delta-correction": workload["delta_correction"],
        "--matrix-merge": workload["matrix_merge"],
        "--outer-momentum": "0",
        "--outer-lr": "0.28",
        "--token-budget": str(workload["compare_token_budget"]),
        "--syncer-total-steps": str(workload["outer_commits"]),
        "--learner-max-steps": str(workload["learner_max_steps_liveness_cap"]),
        "--fixed-window-microsteps": str(workload["fixed_window_microsteps"]),
        "--stock-shadow-capture-session": capture["capture_session_uuid"],
        "--arm-timeout-min": str(workload["arm_timeout_minutes"]),
        "--shard": workload["shard"],
        "--tuning": workload["tuning"],
    }
    argv = [item for pair in values.items() for item in pair]
    return argv + [
        "--strict-quorum",
        "--barrier-sync",
        "--deterministic-commit-order",
        "--skip-baseline",
    ]


def test_frozen_config_matches_executable_gate_contract() -> None:
    raw = FROZEN_CONFIG.read_bytes()
    config = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert Path(f"{FROZEN_CONFIG}.sha256").read_text() == (
        f"{digest}  {FROZEN_CONFIG.name}\n"
    )
    assert config["status"] == "frozen_before_direction_outcome"
    assert config["workload"]["arms_in_order"] == [MOD.OFF_ARM, MOD.ON_ARM]
    assert config["workload"]["result_rows_in_order"] == [
        MOD.BASE_ARM,
        MOD.OFF_ARM,
        MOD.ON_ARM,
    ]
    assert config["workload"]["raw_local_training_tokens"] == 4_352
    assert config["workload"]["compare_token_budget"] == 4_352
    assert config["workload"]["expected_terminal_local_steps"] == 34
    assert config["workload"]["outer_commits"] == 32
    assert config["workload"]["lora_targets"] == "all-linear"
    assert config["workload"]["gradient_checkpointing"] == "off"
    assert config["workload"]["weight_decay"] == 0.01
    assert config["workload"]["warmup_steps"] == 10
    assert config["workload"]["device"] == "cuda"
    assert config["workload"]["gpu_slots"] == 1
    assert config["workload"]["fragment_order"] == MOD.FRAGMENT_ORDER
    assert config["analysis"]["bootstrap_draws"] == 20_000
    assert config["analysis"]["bootstrap_seed"] == 0x43504C47
    assert config["analysis"]["bootstrap_lower_index"] == 1_000
    gates = config["gates"]
    assert gates["minimum_simulated_nonstock_actions"] == 8
    assert gates["minimum_mean_direction_gain"] == 0.001
    assert gates["minimum_positive_fragment_means"] == 3
    assert gates["minimum_one_sided_95_percent_bootstrap_lower_endpoint"] == 0.0
    assert gates["maximum_matched_capture_overhead_fraction"] == 0.02


def test_compare_argv_is_bound_to_every_frozen_scientific_input() -> None:
    config = json.loads(FROZEN_CONFIG.read_text())
    argv = _frozen_compare_argv(config)
    MOD._validate_frozen_config(config, argv)

    drifted = list(argv)
    drifted[drifted.index("--training-seed") + 1] = "999"
    with pytest.raises(MOD.RunnerError, match="training-seed differs"):
        MOD._validate_frozen_config(config, drifted)


def _arm_fixture(root: Path, name: str, *, capture: bool, duration: int) -> Path:
    arm = root / name
    arm.mkdir(parents=True)
    initial = {
        "schema": "cplg_stock_shadow_initial_state_v1",
        "capture_enabled": capture,
        "layout_sha256": "1" * 64,
        "initial_state_sha256": "4" * 64,
    }
    completion = {
        "schema": "cplg_stock_shadow_completion_v1",
        "capture_enabled": capture,
        "layout_sha256": "1" * 64,
        "initial_state_sha256": "4" * 64,
        "interval_scope": ("post_global_initialization_to_durable_vector_writer_close"),
        "interval_start_monotonic_ns": 0,
        "interval_end_monotonic_ns": duration,
        "commits": 32,
    }
    _publish(arm / "stock_shadow_initial_state.json", initial)
    _publish(arm / "stock_shadow_completion.json", completion)
    rows = [
        {
            "commit_seq": commit,
            "fragment": fragment,
            "commit_elapsed_ns": commit * 100,
            "ms": commit,
            "responders": [{"id": 0}],
            "gnorm": 1.0,
        }
        for commit, fragment in enumerate([0, 1, 2, 3] * 8, 1)
    ]
    (arm / "tape.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    (arm / "learner-0.log").write_text(
        "INFO inner loop done at local_step=34 global_step=34\n"
    )
    (arm / "state.ckpt").write_bytes(b"same-state")
    (arm / "export").mkdir()
    (arm / "export" / "adapter.safetensors").write_bytes(b"same-export")
    return arm


def _complete_pair(tmp_path: Path) -> tuple[Path, Path, Path]:
    work = tmp_path / "work"
    report = tmp_path / "report"
    _arm_fixture(work, MOD.OFF_ARM, capture=False, duration=10_000)
    on = _arm_fixture(work, MOD.ON_ARM, capture=True, duration=10_100)
    report.mkdir()
    (report / "results.jsonl").write_text(
        json.dumps({"arm": MOD.BASE_ARM, "eval_loss": 1.5})
        + "\n"
        + json.dumps({"arm": MOD.OFF_ARM, "eval_loss": 1.25})
        + "\n"
        + json.dumps({"arm": MOD.ON_ARM, "eval_loss": 1.25})
        + "\n"
    )
    vectors = on / "stock_vectors"
    vectors.mkdir()
    tape = vectors / "stock_tape.jsonl"
    tape.write_bytes(b"exact-vector-tape\n")
    tape_sha256 = hashlib.sha256(tape.read_bytes()).hexdigest()
    _publish(
        vectors / "stock_tape.manifest.json",
        {
            "status": "COMPLETE",
            "records": 32,
            "layout_sha256": "1" * 64,
            "initial_state_sha256": "4" * 64,
            "run_config_sha256": "2" * 64,
            "stock_tape_sha256": tape_sha256,
            "writer": _writer(),
        },
    )
    input_manifest = tmp_path / "input-provenance.sha256"
    input_manifest.write_text(f"{'3' * 64}  data\n")
    return work, report, input_manifest


def test_builds_strict_matched_overhead_evidence(tmp_path: Path) -> None:
    work, report, input_manifest = _complete_pair(tmp_path)
    output = tmp_path / "overhead.json"

    tape = MOD.build_overhead_evidence(
        work_dir=work,
        report_dir=report,
        input_manifest=input_manifest,
        run_config_sha256="2" * 64,
        output=output,
    )

    assert tape == work / MOD.ON_ARM / "stock_vectors" / "stock_tape.jsonl"
    evidence = json.loads(output.read_text())
    assert evidence["off"]["interval_end_monotonic_ns"] == 10_000
    assert evidence["on"]["interval_end_monotonic_ns"] == 10_100
    assert evidence["off"]["initial_state_sha256"] == "4" * 64
    assert evidence["on"]["writer"] == _writer()
    checksum = Path(f"{output}.sha256")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    assert checksum.read_text() == f"{digest}  {output.name}\n"


def test_initial_state_or_behavior_drift_is_fatal(tmp_path: Path) -> None:
    work, report, input_manifest = _complete_pair(tmp_path)
    on_initial = work / MOD.ON_ARM / "stock_shadow_initial_state.json"
    value = json.loads(on_initial.read_text())
    value["initial_state_sha256"] = "9" * 64
    _publish(on_initial, value)
    with pytest.raises(MOD.RunnerError, match="matched initial receipt differs"):
        MOD.build_overhead_evidence(
            work_dir=work,
            report_dir=report,
            input_manifest=input_manifest,
            run_config_sha256="2" * 64,
            output=tmp_path / "overhead.json",
        )


def test_wrapper_runs_compare_then_hash_pinned_analyzer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    tape = tmp_path / "work" / MOD.ON_ARM / "stock_vectors" / "stock_tape.jsonl"

    def fake_run(command: list[str], *, cwd: Path, check: bool):
        assert cwd == ROOT
        assert check is True
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(MOD.subprocess, "run", fake_run)
    monkeypatch.setattr(MOD, "build_overhead_evidence", lambda **_kwargs: tape)
    monkeypatch.setattr(MOD, "_validate_frozen_config", lambda *_args: None)
    monkeypatch.setattr(
        MOD, "_analysis_outcome", lambda *_args, **_kwargs: ("PASS", "f" * 64)
    )
    monkeypatch.setattr(
        MOD, "_verify_preflight_identity", lambda *_args, **_kwargs: None
    )
    helper = tmp_path / "oracle"
    helper.write_bytes(b"pinned helper")
    helper_sha256 = hashlib.sha256(helper.read_bytes()).hexdigest()
    monkeypatch.setattr(
        MOD,
        "_build_and_pin_helper",
        lambda *_args, **_kwargs: helper_sha256,
    )
    run_config = tmp_path / "frozen-config.json"
    _publish(run_config, {"schema": "frozen-test-config-v1"})
    run_config_sha256 = hashlib.sha256(run_config.read_bytes()).hexdigest()
    compare = [
        "--work-dir",
        str(tmp_path / "work"),
        "--report-dir",
        str(tmp_path / "report"),
        "--settings",
        f"{MOD.OFF_ARM},{MOD.ON_ARM}",
        "--stock-shadow-run-config-sha256",
        run_config_sha256,
    ]
    MOD.run_compare_then_analyze(
        compare_argv=compare,
        output=tmp_path / "verdict.json",
        overhead_output=tmp_path / "overhead.json",
        rust_libm_helper=helper,
        input_manifest=tmp_path / "input.sha256",
        run_config=run_config,
        preflight_output=tmp_path / "preflight.json",
        expected_source_commit="a" * 40,
        python="/venv/python",
    )

    assert calls[0] == ["/venv/python", str(MOD.COMPARE), *compare]
    assert calls[1][0:2] == ["/venv/python", str(MOD.ANALYZER)]
    assert "--enforce-shadow-gate" in calls[1]
    assert "--rust-libm-helper" in calls[1]


def test_wrapper_rejects_run_config_hash_not_bound_to_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        MOD.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("compare must not run"),
    )
    run_config = tmp_path / "frozen-config.json"
    _publish(run_config, {"schema": "frozen-test-config-v1"})
    compare = [
        "--work-dir",
        str(tmp_path / "work"),
        "--report-dir",
        str(tmp_path / "report"),
        "--settings",
        f"{MOD.OFF_ARM},{MOD.ON_ARM}",
        "--stock-shadow-run-config-sha256",
        "2" * 64,
    ]
    with pytest.raises(MOD.RunnerError, match="differs from the frozen"):
        MOD.run_compare_then_analyze(
            compare_argv=compare,
            output=tmp_path / "verdict.json",
            overhead_output=tmp_path / "overhead.json",
            rust_libm_helper=tmp_path / "oracle",
            input_manifest=tmp_path / "input.sha256",
            run_config=run_config,
            preflight_output=tmp_path / "preflight.json",
            expected_source_commit="a" * 40,
            python="/venv/python",
        )


def test_export_tree_rejects_symlinked_directory(tmp_path: Path) -> None:
    root = tmp_path / "export"
    root.mkdir()
    target = tmp_path / "outside"
    target.mkdir()
    (target / "adapter.safetensors").write_bytes(b"escape")
    (root / "linked-directory").symlink_to(target, target_is_directory=True)

    with pytest.raises(MOD.RunnerError, match="contains a symlink"):
        MOD._tree_sha256(root)


def test_duplicate_or_out_of_order_result_arms_are_fatal(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    results.write_text(
        json.dumps({"arm": MOD.ON_ARM, "eval_loss": 1.0})
        + "\n"
        + json.dumps({"arm": MOD.OFF_ARM, "eval_loss": 1.0})
        + "\n"
    )

    with pytest.raises(MOD.RunnerError, match="untrained base followed by"):
        MOD._evaluation_losses(results)


def test_real_compare_result_schema_accepts_base_then_frozen_pair(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results.jsonl"
    results.write_text(
        json.dumps({"arm": MOD.BASE_ARM, "m": 0, "wall_s": 0.0, "eval_loss": 1.5})
        + "\n"
        + json.dumps({"arm": MOD.OFF_ARM, "m": 1, "wall_s": 80.5, "eval_loss": 1.25})
        + "\n"
        + json.dumps({"arm": MOD.ON_ARM, "m": 1, "wall_s": 80.3, "eval_loss": 1.25})
        + "\n"
    )

    assert MOD._evaluation_losses(results) == (1.25, 1.25)


def test_helper_preflight_builds_locked_and_publishes_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    syncer = repo / "syncer"
    helper = syncer / "target" / "release" / "cplg_libm_oracle"
    helper.parent.mkdir(parents=True)
    helper.write_bytes(b"linux helper")
    (syncer / "Cargo.lock").write_bytes(b"locked dependencies")
    monkeypatch.setattr(MOD, "REPO_ROOT", repo)
    monkeypatch.setattr(MOD, "SYNCER_DIR", syncer)
    monkeypatch.setattr(MOD, "PINNED_HELPER", helper)
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs):
        calls.append(command)
        if command[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(command, 0, stdout="a" * 40 + "\n")
        if command[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(command, 0, stdout="")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(MOD.subprocess, "run", fake_run)
    output = tmp_path / "preflight.json"
    digest = MOD._build_and_pin_helper(
        helper,
        expected_source_commit="a" * 40,
        run_config_sha256="2" * 64,
        preflight_output=output,
    )

    assert calls[0][0:4] == ["cargo", "build", "--locked", "--release"]
    assert digest == hashlib.sha256(b"linux helper").hexdigest()
    receipt = json.loads(output.read_text())
    assert receipt["source_commit"] == "a" * 40
    assert receipt["rust_libm_helper_sha256"] == digest
    assert Path(f"{output}.sha256").exists()


def test_terminal_supervisor_publishes_inconclusive_on_evidence_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        MOD,
        "run_compare_then_analyze",
        lambda **_kwargs: (_ for _ in ()).throw(MOD.EvidenceError("broken tape")),
    )
    terminal = tmp_path / "terminal.json"
    code = MOD.main(
        [
            "--output",
            str(tmp_path / "analysis.json"),
            "--overhead-output",
            str(tmp_path / "overhead.json"),
            "--rust-libm-helper",
            str(tmp_path / "helper"),
            "--input-manifest",
            str(tmp_path / "input.sha256"),
            "--run-config",
            str(tmp_path / "config.json"),
            "--preflight-output",
            str(tmp_path / "preflight.json"),
            "--terminal-output",
            str(terminal),
            "--expected-source-commit",
            "a" * 40,
            "--",
            "unused",
        ]
    )

    assert code == 3
    verdict = json.loads(terminal.read_text())
    assert verdict["decision"] == "INCONCLUSIVE"
    assert verdict["stage"] == "post_acquisition_evidence_validation"
    assert verdict["error_class"] == "EvidenceError"
    assert Path(f"{terminal}.sha256").exists()


@pytest.mark.parametrize("decision", ["PASS", "FAIL"])
def test_terminal_supervisor_publishes_valid_scientific_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decision: str
) -> None:
    monkeypatch.setattr(
        MOD,
        "run_compare_then_analyze",
        lambda **_kwargs: (decision, "f" * 64),
    )
    terminal = tmp_path / f"terminal-{decision}.json"
    code = MOD.main(_minimal_main_argv(tmp_path, terminal))

    assert code == 0
    verdict = json.loads(terminal.read_text())
    assert verdict["decision"] == decision
    assert verdict["stage"] == "completed_analysis"
    assert verdict["analysis_sha256"] == "f" * 64


@pytest.mark.parametrize(
    "error",
    [
        MOD.RunnerError("configuration rejected"),
        MOD.RunnerError("comparison process failed"),
        subprocess.CalledProcessError(1, ["cargo", "build"]),
    ],
    ids=("configuration", "compare", "helper-build"),
)
def test_terminal_supervisor_publishes_infra_for_known_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    def fail(**_kwargs):
        raise error

    monkeypatch.setattr(MOD, "run_compare_then_analyze", fail)
    terminal = tmp_path / "terminal.json"
    code = MOD.main(_minimal_main_argv(tmp_path, terminal))

    assert code == 2
    verdict = json.loads(terminal.read_text())
    assert verdict["decision"] == "INFRA_FAILURE"
    assert verdict["stage"] == "configuration_or_runtime"
    assert verdict["error_class"] == type(error).__name__


def test_terminal_supervisor_catches_unexpected_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        MOD,
        "run_compare_then_analyze",
        lambda **_kwargs: (_ for _ in ()).throw(TypeError("unexpected shape")),
    )
    terminal = tmp_path / "terminal.json"
    code = MOD.main(_minimal_main_argv(tmp_path, terminal))

    assert code == 2
    verdict = json.loads(terminal.read_text())
    assert verdict["decision"] == "INFRA_FAILURE"
    assert verdict["error_class"] == "TypeError"
    assert verdict["error"] == "unexpected shape"


def test_nonfresh_terminal_prevents_attempt_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    terminal = tmp_path / "terminal.json"
    terminal.write_text("stale\n")
    monkeypatch.setattr(
        MOD,
        "run_compare_then_analyze",
        lambda **_kwargs: pytest.fail("a nonfresh attempt must not start"),
    )

    assert MOD.main(_minimal_main_argv(tmp_path, terminal)) == 2
    assert terminal.read_text() == "stale\n"


def test_dangling_symlink_is_nonfresh_and_cannot_be_replaced(tmp_path: Path) -> None:
    output = tmp_path / "analysis.json"
    output.symlink_to(tmp_path / "missing-target.json")

    with pytest.raises(MOD.RunnerError, match="not fresh"):
        MOD._require_fresh_checksummed_output(output, "analysis")
    with pytest.raises(MOD.RunnerError, match="not fresh"):
        MOD._atomic_checksummed_json(output, {"schema": "must-not-publish"})
    assert output.is_symlink()


def _minimal_main_argv(tmp_path: Path, terminal: Path) -> list[str]:
    return [
        "--output",
        str(tmp_path / "analysis.json"),
        "--overhead-output",
        str(tmp_path / "overhead.json"),
        "--rust-libm-helper",
        str(tmp_path / "helper"),
        "--input-manifest",
        str(tmp_path / "input.sha256"),
        "--run-config",
        str(tmp_path / "config.json"),
        "--preflight-output",
        str(tmp_path / "preflight.json"),
        "--terminal-output",
        str(terminal),
        "--expected-source-commit",
        "a" * 40,
        "--",
        "unused",
    ]


def test_compare_presets_differ_only_by_vector_capture() -> None:
    off = COMPARE_MOD.PRESETS[MOD.OFF_ARM]
    on = COMPARE_MOD.PRESETS[MOD.ON_ARM]
    from dataclasses import asdict

    off_fields = asdict(off)
    on_fields = asdict(on)
    assert off_fields.pop("name") == MOD.OFF_ARM
    assert on_fields.pop("name") == MOD.ON_ARM
    assert off_fields.pop("stock_vector_capture") is False
    assert on_fields.pop("stock_vector_capture") is True
    assert off_fields == on_fields

    command = COMPARE_MOD.syncer_command(
        on,
        12345,
        Path("/run/on"),
        32,
        deterministic_commit_order=True,
        stock_shadow_initial_state_manifest=Path("/run/on/initial.json"),
        stock_shadow_completion_manifest=Path("/run/on/completion.json"),
        stock_vector_capture_dir=Path("/run/on/vectors"),
        stock_vector_capture_session="session-1",
        stock_vector_capture_run_config_sha256="2" * 64,
    )
    assert command[command.index("--outer-optimizer") + 1] == "nesterov"
    assert command[command.index("--outer-lr") + 1] == "0.28"
    assert command[command.index("--outer-momentum") + 1] == "0.0"
    assert command[command.index("--stock-vector-capture-expected-records") + 1] == (
        "32"
    )
    assert "--stock-shadow-initial-state-manifest" in command

    learner = COMPARE_MOD.learner_command(
        SimpleNamespace(
            model="/model",
            lora_r=2,
            lora_alpha=4,
            seq_len=128,
            micro_batch_size=1,
            inner_lr=0.001,
            learner_gpus=0,
            training_seed=271271,
            tuning="lora",
            shard="ddp",
            device="cuda",
            bcmp_shadow_path=False,
            optimizer_state_capture=False,
        ),
        Path("/run/on"),
        learner_id=0,
        num_learners=1,
        syncer="127.0.0.1:12345",
        max_steps=96,
        arm=on,
    )
    for flag, expected in (
        ("--inner-optimizer", "adamw"),
        ("--weight-decay", "0.01"),
        ("--warmup-steps", "10"),
        ("--grad-clip", "1.0"),
        ("--gradient-checkpointing", "off"),
        ("--lora-targets", "all-linear"),
        ("--max-reconnects", "0"),
    ):
        assert learner.count(flag) == 1
        assert learner[learner.index(flag) + 1] == expected
