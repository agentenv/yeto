from __future__ import annotations

from pathlib import Path

from yeto.optimizer_harness import load_spec


ROOT = Path(__file__).parents[1]
SPECS = ROOT / "experiments" / "optimizer"
PINNED_COMMIT = "d2f66ab040dc539bcb06629513dfa9f8c3dc9692"


def _flag(command: tuple[str, ...], name: str) -> str:
    assert command.count(name) == 1
    index = command.index(name)
    assert index + 1 < len(command)
    return command[index + 1]


def _load(stage: str):
    return load_spec(SPECS / f"exp2-pti-online-{stage}-r1.json")


def test_e1_is_launchable_one_a100_exact_pti_online_pair() -> None:
    spec = _load("e1-m1-canary")
    assert spec.repo_commit == PINNED_COMMIT
    assert spec.cloud["adopt_only"] is False
    assert spec.cloud["machine_type"] == "a2-highgpu-1g"
    assert spec.cloud["accelerator_count"] == 1
    assert spec.cloud["max_total_accelerators"] == 8
    assert spec.cloud["max_run_duration_seconds"] == 3600
    assert spec.cloud["expected_source_image_id"] == "7290368630472593484"
    assert spec.command[:2] == (
        "/home/shou/venv/bin/python",
        "scripts/run_pti_online_pair.py",
    )
    assert _flag(spec.command, "--settings") == "pti_m1_stock,pti_m1_candidate"
    assert _flag(spec.command, "--gpu-slots") == "1"
    assert _flag(spec.command, "--fixed-window-microsteps") == "4"
    assert _flag(spec.command, "--syncer-total-steps") == "32"
    assert _flag(spec.command, "--learner-max-steps") == "96"
    assert _flag(spec.command, "--token-budget") == "32768"
    assert _flag(spec.command, "--eval-rows") == "8"
    assert _flag(spec.command, "--arm-timeout-min") == "20"


def test_e2_is_adopt_only_conditional_four_a100_exact_pti_online_pair() -> None:
    spec = _load("e2-m4-screen")
    assert spec.repo_commit == PINNED_COMMIT
    assert spec.cloud["adopt_only"] is True
    assert spec.cloud["labels"]["draft"] == "true"
    assert spec.cloud["labels"]["gate"] == "e1-pass-required"
    assert spec.cloud["machine_type"] == "a2-highgpu-4g"
    assert spec.cloud["accelerator_count"] == 4
    assert spec.cloud["max_total_accelerators"] == 8
    assert spec.cloud["max_run_duration_seconds"] == 3600
    assert spec.cloud["expected_source_image_id"] == "7290368630472593484"
    assert _flag(spec.command, "--settings") == "pti_m4_stock,pti_m4_candidate"
    assert _flag(spec.command, "--gpu-slots") == "4"
    assert _flag(spec.command, "--fixed-window-microsteps") == "16"
    assert _flag(spec.command, "--syncer-total-steps") == "32"
    assert _flag(spec.command, "--learner-max-steps") == "512"
    assert _flag(spec.command, "--token-budget") == "700000"
    assert _flag(spec.command, "--eval-rows") == "64"
    assert _flag(spec.command, "--arm-timeout-min") == "45"


def test_specs_freeze_treatment_and_complete_evidence_contracts() -> None:
    for stage, arms, headroom in (
        ("e1-m1-canary", {"pti_m1_stock", "pti_m1_candidate"}, 64),
        ("e2-m4-screen", {"pti_m4_stock", "pti_m4_candidate"}, 384),
    ):
        spec = _load(stage)
        assert "--outer-optimizer" not in spec.command
        assert _flag(spec.command, "--outer-lr") == "0.28"
        assert _flag(spec.command, "--outer-momentum") == "0"
        assert _flag(spec.command, "--delta-correction") == "none"
        assert _flag(spec.command, "--matrix-merge") == "rda"
        assert {name for name in spec.checks["expected_arms"]} == arms
        for flag in (
            "--strict-quorum",
            "--barrier-sync",
            "--deterministic-commit-order",
            "--skip-baseline",
        ):
            assert spec.command.count(flag) == 1
            assert spec.checks["expected_flags"][flag] == ""
        assert spec.checks["strict_quorum_step_budget"]["fragments"] == 4
        assert (
            spec.checks["strict_quorum_step_budget"]["min_headroom_steps"]
            == headroom
        )
        completions = set(spec.execution["completion_paths"])
        assert any(path.endswith("/report/results.jsonl") for path in completions)
        assert sum(path.endswith("/tape.jsonl") for path in completions) == 2
        assert sum(path.endswith("/state.ckpt") for path in completions) == 2
        assert any(
            path.endswith("/report/pti_online_validation.json")
            for path in completions
        )
        assert any(
            path.endswith("/report/pti_online_validation.json.sha256")
            for path in completions
        )
        assert any(path.endswith("/input-provenance.sha256") for path in completions)
        assert len(spec.execution["checksum_manifests"]) == 2
        assert spec.execution["input_checksum_manifests"] == [
            "/etc/yeto-model-files.sha256",
            "/etc/yeto-data.sha256",
        ]
