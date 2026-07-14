"""Launch-plumbing tests for exact optimizer-state captures."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parent.parent


def _load_compare():
    name = "compare_diloco_optimizer_state_capture"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / "compare_diloco.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


compare = _load_compare()


def _args(**overrides):
    values = {
        "model": "lfm25-230m",
        "lora_r": 2,
        "lora_alpha": 4,
        "seq_len": 128,
        "micro_batch_size": 1,
        "inner_lr": 1e-3,
        "device": "cpu",
        "shard": "ddp",
        "learner_gpus": 0,
        "training_seed": 223223,
        "tuning": "lora",
        "bcmp_shadow_path": False,
        "optimizer_state_capture": True,
        "optimizer_state_capture_profile": "full",
        "optimizer_state_capture_parity": False,
        "optimizer_state_capture_every": 2,
        "optimizer_state_capture_max_hmc_events": 7,
        "optimizer_state_capture_max_midpoint_windows": 9,
        "optimizer_state_capture_max_bytes": 123456,
        "optimizer_state_capture_background_writer": False,
        "optimizer_state_capture_writer_max_items": 4,
        "optimizer_state_capture_writer_max_bytes": 654321,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_capture_presets_freeze_unambiguous_geometry():
    smoke = compare.PRESETS["capture_m1"]
    qualification = compare.PRESETS["capture_m4"]
    assert (smoke.m, smoke.quorum, smoke.fixed_window_microsteps) == (1, 1, 4)
    assert (qualification.m, qualification.quorum) == (4, 4)
    for arm in (smoke, qualification):
        assert arm.optimizer_state_capture is True
        assert arm.strict_quorum is True
        assert arm.inner_optimizer == "adamw"
        assert arm.inner_control_variate == "none"
        assert arm.wire_dtype == "f32"
        assert arm.merge_alpha == 0.0
        assert arm.delta_correction == "none"
        assert arm.outer_lr == 0.28
        assert arm.outer_momentum == 0.0


def test_capture_flags_and_unique_directories_are_forwarded_per_learner():
    arm_dir = Path("/tmp/run/work/capture_m4")
    command = compare.learner_command(
        _args(),
        arm_dir,
        learner_id=3,
        num_learners=4,
        syncer="127.0.0.1:9000",
        max_steps=64,
        arm=compare.PRESETS["capture_m4"],
    )
    assert command[command.index("--optimizer-state-capture-dir") + 1] == str(
        arm_dir / "optimizer_state_capture_learner_3"
    )
    assert command[command.index("--optimizer-state-capture-every") + 1] == "2"
    assert command[command.index("--optimizer-state-capture-max-hmc-events") + 1] == "7"
    assert (
        command[command.index("--optimizer-state-capture-max-midpoint-windows") + 1]
        == "9"
    )
    assert command[command.index("--optimizer-state-capture-max-bytes") + 1] == "123456"
    assert command[command.index("--max-reconnects") + 1] == "0"
    assert "--optimizer-state-capture-profile" not in command


def test_directional_capture_profile_is_forwarded_only_when_opted_in():
    command = compare.learner_command(
        _args(
            optimizer_state_capture_profile="crp_pti_directional",
            optimizer_state_capture_max_hmc_events=0,
        ),
        Path("/tmp/run/work/capture_m4"),
        learner_id=2,
        num_learners=4,
        syncer="127.0.0.1:9000",
        max_steps=64,
        arm=compare.PRESETS["capture_m4"],
    )

    assert command[command.index("--optimizer-state-capture-profile") + 1] == (
        "crp_pti_directional"
    )
    assert command[command.index("--optimizer-state-capture-max-hmc-events") + 1] == (
        "0"
    )


def test_background_writer_flags_are_opt_in_and_forward_exact_caps():
    command = compare.learner_command(
        _args(
            optimizer_state_capture_background_writer=True,
            optimizer_state_capture_writer_max_items=3,
            optimizer_state_capture_writer_max_bytes=456789,
        ),
        Path("/tmp/run/work/capture_m4"),
        learner_id=1,
        num_learners=4,
        syncer="127.0.0.1:9000",
        max_steps=64,
        arm=compare.PRESETS["capture_m4"],
    )

    assert command.count("--optimizer-state-capture-background-writer") == 1
    assert (
        command[command.index("--optimizer-state-capture-writer-max-items") + 1] == "3"
    )
    assert (
        command[command.index("--optimizer-state-capture-writer-max-bytes") + 1]
        == "456789"
    )


def test_capture_flags_are_absent_for_baseline_or_when_disabled():
    baseline = compare.learner_command(
        _args(),
        Path("/tmp/run/work/baseline"),
        learner_id=0,
        num_learners=1,
        syncer="none",
        max_steps=8,
        arm=None,
    )
    disabled = compare.learner_command(
        _args(optimizer_state_capture=False),
        Path("/tmp/run/work/capture_m1"),
        learner_id=0,
        num_learners=1,
        syncer="127.0.0.1:9000",
        max_steps=8,
        arm=compare.PRESETS["capture_m1"],
    )
    matched_control = compare.learner_command(
        _args(),
        Path("/tmp/run/work/capture_m4_off"),
        learner_id=0,
        num_learners=4,
        syncer="127.0.0.1:9000",
        max_steps=8,
        arm=compare.PRESETS["capture_m4_off"],
    )
    assert "--optimizer-state-capture-dir" not in baseline
    assert "--optimizer-state-capture-dir" not in disabled
    assert "--optimizer-state-capture-dir" not in matched_control


@pytest.mark.parametrize("learner_count", [1, 4])
def test_matched_capture_presets_differ_only_by_treatment_and_name(learner_count):
    control = compare.PRESETS[f"capture_m{learner_count}_off"]
    treatment = compare.PRESETS[f"capture_m{learner_count}_on"]
    control_fields = vars(control).copy()
    treatment_fields = vars(treatment).copy()
    assert control_fields.pop("name") == f"capture_m{learner_count}_off"
    assert treatment_fields.pop("name") == f"capture_m{learner_count}_on"
    assert control_fields.pop("optimizer_state_capture") is False
    assert treatment_fields.pop("optimizer_state_capture") is True
    assert control_fields == treatment_fields


def test_m1_parity_presets_mirror_m4_except_learner_and_quorum_counts():
    for treatment in ("off", "on"):
        m1_fields = vars(compare.PRESETS[f"capture_m1_{treatment}"]).copy()
        m4_fields = vars(compare.PRESETS[f"capture_m4_{treatment}"]).copy()
        assert m1_fields.pop("name") == f"capture_m1_{treatment}"
        assert m4_fields.pop("name") == f"capture_m4_{treatment}"
        assert m1_fields.pop("m") == 1
        assert m4_fields.pop("m") == 4
        assert m1_fields.pop("quorum") == 1
        assert m4_fields.pop("quorum") == 4
        assert m1_fields == m4_fields


@pytest.mark.parametrize("learner_count", [1, 4])
def test_parity_learner_commands_differ_only_by_capture_artifact_options(
    learner_count,
):
    args = _args(optimizer_state_capture_parity=True)
    off_name = f"capture_m{learner_count}_off"
    on_name = f"capture_m{learner_count}_on"
    off_dir = Path(f"/tmp/run/work/{off_name}")
    on_dir = Path(f"/tmp/run/work/{on_name}")

    def command(arm_name: str, arm_dir: Path) -> list[str]:
        return compare.learner_command(
            args,
            arm_dir,
            learner_id=0,
            num_learners=learner_count,
            syncer="127.0.0.1:9000",
            max_steps=80,
            arm=compare.PRESETS[arm_name],
        )

    def normalized_options(
        argv: list[str], arm_dir: Path
    ) -> tuple[list[str], dict[str, str | None]]:
        prefix: list[str] = []
        options: dict[str, str | None] = {}
        index = 0
        while index < len(argv):
            token = argv[index]
            if not token.startswith("--"):
                prefix.append(token)
                index += 1
                continue
            assert token not in options, f"duplicate command option {token}"
            value = None
            if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                value = argv[index + 1].replace(str(arm_dir), "<ARM_DIR>")
                index += 1
            options[token] = value
            index += 1
        return prefix, options

    off_prefix, off_options = normalized_options(command(off_name, off_dir), off_dir)
    on_prefix, on_options = normalized_options(command(on_name, on_dir), on_dir)
    capture_only = {
        "--optimizer-state-capture-dir",
        "--optimizer-state-capture-every",
        "--optimizer-state-capture-max-hmc-events",
        "--optimizer-state-capture-max-midpoint-windows",
        "--optimizer-state-capture-max-bytes",
    }

    assert off_prefix == on_prefix
    assert set(on_options) - set(off_options) == capture_only
    assert set(off_options) - set(on_options) == set()
    for option in sorted(set(off_options) & set(on_options)):
        assert off_options[option] == on_options[option], option
    assert off_options["--inner-optimizer"] == "adamw"
    assert on_options["--inner-optimizer"] == "adamw"
    assert off_options["--max-reconnects"] == "0"
    assert on_options["--max-reconnects"] == "0"
    assert on_options["--optimizer-state-capture-dir"] == (
        "<ARM_DIR>/optimizer_state_capture_learner_0"
    )


@pytest.mark.parametrize("learner_count", [1, 4])
def test_parity_validator_command_uses_the_exact_matched_pair_paths(learner_count):
    off_name = f"capture_m{learner_count}_off"
    on_name = f"capture_m{learner_count}_on"
    args = SimpleNamespace(
        work_dir=Path("/tmp/run/work"),
        report_dir=Path("/tmp/run/report"),
        optimizer_state_capture_parity_overhead_limit=0.02,
        optimizer_state_capture_parity_require_barrier=True,
    )

    command = compare.optimizer_state_capture_parity_command(
        args, [compare.PRESETS[off_name], compare.PRESETS[on_name]]
    )

    assert command[command.index("--off-arm-dir") + 1] == str(args.work_dir / off_name)
    assert command[command.index("--on-arm-dir") + 1] == str(args.work_dir / on_name)
    assert command[command.index("--off-arm") + 1] == off_name
    assert command[command.index("--on-arm") + 1] == on_name
    assert command.count("--require-barrier-schedule") == 1


@pytest.mark.parametrize(
    "names",
    [
        {"capture_m1_off", "capture_m4_on"},
        {"capture_m1_off", "capture_m1_on", "capture_m4_off", "capture_m4_on"},
        {"capture_m1_off", "capture_m1_on", "m1"},
    ],
)
def test_parity_pair_selection_rejects_mixing_and_extra_arms(names):
    assert compare.capture_parity_pair_for_arm_names(names) is None


def test_parity_timing_uses_exact_producer_commit_interval():
    records = [
        {"commit_seq": 1, "commit_elapsed_ns": 900_000_000},
        {"commit_seq": 2, "commit_elapsed_ns": 1_000_000_000},
        {"commit_seq": 3, "commit_elapsed_ns": 1_800_000_000},
        {"commit_seq": 4, "commit_elapsed_ns": 2_900_000_000},
    ]

    assert compare.parity_commit_interval_seconds(records, expected_steps=4) == 2.0


@pytest.mark.parametrize(
    "records",
    [
        [
            {"commit_seq": 1, "commit_elapsed_ns": 10},
            {"commit_seq": 3, "commit_elapsed_ns": 20},
        ],
        [
            {"commit_seq": 1, "commit_elapsed_ns": 20},
            {"commit_seq": 2, "commit_elapsed_ns": 20},
        ],
    ],
)
def test_parity_timing_rejects_nonexact_commit_sequence(records):
    with pytest.raises(RuntimeError, match="commit"):
        compare.parity_commit_interval_seconds(records, expected_steps=2)


def test_syncer_response_transcript_requires_and_forwards_pair():
    arm = compare.PRESETS["capture_m4_on"]
    path = Path("/tmp/run/work/capture_m4_on/syncer_response_transcript.jsonl")
    session = "e116f6ac-8b8c-4f87-81da-8b9dc20d9741"
    command = compare.syncer_command(
        arm,
        9000,
        path.parent,
        16,
        response_transcript=path,
        response_transcript_session=session,
    )
    assert command[command.index("--response-transcript") + 1] == str(path)
    assert command[command.index("--response-transcript-session") + 1] == session

    with pytest.raises(ValueError, match="configured together"):
        compare.syncer_command(
            arm,
            9000,
            path.parent,
            16,
            response_transcript=path,
        )


def test_capture_rejects_ambiguous_arm_before_launch(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_diloco.py",
            "--data",
            "unused.jsonl",
            "--settings",
            "m4",
            "--optimizer-state-capture",
            "--dry-run",
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        compare.main()
    assert exc_info.value.code == 2


def test_background_writer_requires_capture_master_switch(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_diloco.py",
            "--data",
            "unused.jsonl",
            "--optimizer-state-capture-background-writer",
            "--dry-run",
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        compare.main()
    assert exc_info.value.code == 2


def test_capture_accepts_matched_off_on_campaign(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_diloco.py",
            "--data",
            "unused.jsonl",
            "--settings",
            "capture_m4_off,capture_m4_on",
            "--optimizer-state-capture",
            "--syncer-total-steps",
            "16",
            "--fixed-window-microsteps",
            "4",
            "--dry-run",
        ],
    )
    assert compare.main() == 0


def test_parity_campaign_requires_fully_sampled_probe(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_diloco.py",
            "--data",
            "unused.jsonl",
            "--settings",
            "capture_m4_off,capture_m4_on",
            "--optimizer-state-capture",
            "--optimizer-state-capture-parity",
            "--syncer-total-steps",
            "16",
            "--fixed-window-microsteps",
            "4",
            "--dry-run",
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        compare.main()
    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    "settings",
    ["capture_m1_off,capture_m1_on", "capture_m4_off,capture_m4_on"],
)
def test_parity_campaign_accepts_exact_matched_geometry(monkeypatch, settings):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_diloco.py",
            "--data",
            "unused.jsonl",
            "--settings",
            settings,
            "--optimizer-state-capture",
            "--optimizer-state-capture-parity",
            "--barrier-sync",
            "--optimizer-state-capture-parity-require-barrier",
            "--optimizer-state-capture-strict-writer",
            "--syncer-probe-capture",
            "--syncer-probe-capture-every",
            "1",
            "--syncer-total-steps",
            "16",
            "--fixed-window-microsteps",
            "4",
            "--dry-run",
        ],
    )
    assert compare.main() == 0


def test_parity_campaign_rejects_unproven_nonbarrier_geometry(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_diloco.py",
            "--data",
            "unused.jsonl",
            "--settings",
            "capture_m4_off,capture_m4_on",
            "--optimizer-state-capture",
            "--optimizer-state-capture-parity",
            "--optimizer-state-capture-strict-writer",
            "--syncer-probe-capture",
            "--syncer-probe-capture-every",
            "1",
            "--syncer-total-steps",
            "16",
            "--fixed-window-microsteps",
            "4",
            "--dry-run",
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        compare.main()
    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    "settings",
    [
        "capture_m1_off,capture_m4_on",
        "capture_m1_off,capture_m1_on,capture_m4_off,capture_m4_on",
        "capture_m1_off,capture_m1_on,m1",
    ],
)
def test_parity_campaign_rejects_mixed_or_extra_settings(monkeypatch, settings):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_diloco.py",
            "--data",
            "unused.jsonl",
            "--settings",
            settings,
            "--optimizer-state-capture",
            "--optimizer-state-capture-parity",
            "--barrier-sync",
            "--optimizer-state-capture-parity-require-barrier",
            "--optimizer-state-capture-strict-writer",
            "--syncer-probe-capture",
            "--syncer-probe-capture-every",
            "1",
            "--syncer-total-steps",
            "16",
            "--fixed-window-microsteps",
            "4",
            "--dry-run",
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        compare.main()
    assert exc_info.value.code == 2
