"""Immutable launch-contract tests for the CPLG stock-shadow acquisition."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from yeto.optimizer_harness import load_spec


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "experiments" / "optimizer" / "exp2-cplg-shadow-direction-r2.json"
CONFIG_PATH = (
    ROOT / "experiments" / "optimizer" / "cplg-sgd-shadow-direction-r2-config.json"
)
DOSSIER_PATH = (
    ROOT / "experiments" / "optimizer" / "cplg-sgd-shadow-direction-r2-prereg.md"
)
COMMIT = "eb6d21146011112ffe8df5cb518c985e8c0297bd"
CONFIG_SHA256 = "fb7d4c0539cc8760058e0f0b20101bde7fcbac9224c8b27ca69d9724180aaf96"
CAPTURE_SESSION = "667f5de8-6d6d-4ce0-9344-efc239583abf"
RUN = "/home/shou/runs/exp2-cplg-shadow-direction-r2"
REPO = "/home/shou/experiments/exp2-cplg-shadow-direction-r2/repo"


def _flag_value(command: tuple[str, ...], flag: str) -> str:
    assert command.count(flag) == 1
    position = command.index(flag)
    assert position + 1 < len(command)
    return command[position + 1]


def test_cloud_spec_is_pinned_to_frozen_implementation_and_scientific_files() -> None:
    raw = json.loads(SPEC_PATH.read_bytes())
    spec = load_spec(SPEC_PATH)
    bindings = raw["scientific_bindings"]
    spec_sha256 = hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest()

    assert Path(f"{SPEC_PATH}.sha256").read_text() == (
        f"{spec_sha256}  {SPEC_PATH.name}\n"
    )
    assert spec.run_id == "exp2-cplg-shadow-direction-r2"
    assert spec.repo_commit == COMMIT
    assert bindings["implementation_commit"] == COMMIT
    assert hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest() == CONFIG_SHA256
    assert bindings["scientific_config_sha256"] == CONFIG_SHA256
    assert (
        bindings["dossier_sha256"]
        == hashlib.sha256(DOSSIER_PATH.read_bytes()).hexdigest()
    )
    assert bindings["capture_session_uuid"] == CAPTURE_SESSION
    assert "not active E1" in bindings["claim_boundary"]
    assert spec.cloud["accelerator_count"] == 1
    assert spec.cloud["max_total_accelerators"] == 1
    assert spec.cloud["machine_type"] == "a2-highgpu-1g"
    assert spec.cloud["provisioning_model"] == "SPOT"
    assert spec.cloud["expected_source_image_id"] == "7290368630472593484"
    assert spec.execution["remote_repo_dir"] == REPO
    assert spec.execution["remote_run_dir"] == RUN
    assert spec.artifact_uri.endswith("/exp2-cplg-shadow-direction-r2")
    config = json.loads(CONFIG_PATH.read_bytes())
    assert config["workload"]["result_rows_in_order"] == [
        "base (untrained)",
        "cplg_shadow_off",
        "cplg_shadow_on",
    ]


def test_cloud_command_is_exact_stock_only_pair_with_truthful_work_accounting() -> None:
    spec = load_spec(SPEC_PATH)
    command = spec.command

    assert command[:2] == (
        "/home/shou/venv/bin/python",
        "scripts/run_cplg_shadow_pair.py",
    )
    expected = {
        "--expected-source-commit": COMMIT,
        "--run-config": (
            f"{REPO}/experiments/optimizer/cplg-sgd-shadow-direction-r2-config.json"
        ),
        "--stock-shadow-run-config-sha256": CONFIG_SHA256,
        "--stock-shadow-capture-session": CAPTURE_SESSION,
        "--settings": "cplg_shadow_off,cplg_shadow_on",
        "--model": "/home/shou/models/Qwen3.5-9B",
        "--data": "/home/shou/data/Capybara-local/train.parquet",
        "--seq-len": "128",
        "--micro-batch-size": "1",
        "--inner-lr": "0.001",
        "--tuning": "lora",
        "--shard": "ddp",
        "--lora-r": "2",
        "--lora-alpha": "4",
        "--eval-rows": "8",
        "--max-rows": "5000",
        "--shuffle-rows-seed": "271",
        "--training-seed": "271271",
        "--device": "cuda",
        "--gpu-slots": "1",
        "--delta-correction": "none",
        "--matrix-merge": "rda",
        "--outer-momentum": "0",
        "--outer-lr": "0.28",
        "--token-budget": "4352",
        "--syncer-total-steps": "32",
        "--learner-max-steps": "96",
        "--fixed-window-microsteps": "4",
        "--arm-timeout-min": "20",
        "--work-dir": f"{RUN}/work",
        "--report-dir": f"{RUN}/report",
    }
    for flag, value in expected.items():
        assert _flag_value(command, flag) == value
    for flag in (
        "--strict-quorum",
        "--barrier-sync",
        "--deterministic-commit-order",
        "--skip-baseline",
    ):
        assert command.count(flag) == 1
    assert "--baseline-loss" not in command
    assert "--outer-optimizer" not in command
    assert "--stock-shadow-expected-layout-sha256" not in command
    assert spec.checks["strict_quorum_step_budget"]["required_learner_steps"] == 96


def test_completion_contract_covers_every_vector_and_primary_receipt() -> None:
    spec = load_spec(SPEC_PATH)
    paths = list(spec.execution["completion_paths"])
    manifests = set(spec.execution["checksum_manifests"])

    assert len(paths) == len(set(paths))
    for basename in (
        "cplg_shadow_terminal.json",
        "cplg_shadow_analysis.json",
        "cplg_shadow_overhead.json",
        "cplg_shadow_preflight.json",
    ):
        report = f"{RUN}/report/{basename}"
        assert report in paths
        assert f"{report}.sha256" in paths
        assert f"{report}.sha256" in manifests
    for arm in ("cplg_shadow_off", "cplg_shadow_on"):
        root = f"{RUN}/work/{arm}"
        for basename in (
            "tape.jsonl",
            "state.ckpt",
            "learner-0.log",
            "syncer.log",
            "export.log",
            "export/adapter_config.json",
            "export/adapter_model.safetensors",
            "stock_shadow_initial_state.json",
            "stock_shadow_initial_state.json.sha256",
            "stock_shadow_completion.json",
            "stock_shadow_completion.json.sha256",
        ):
            assert f"{root}/{basename}" in paths
    vectors = [path for path in paths if path.endswith(".f32le")]
    assert len(vectors) == 32
    for commit, path in enumerate(vectors, 1):
        assert path.endswith(f"commit-{commit:08}-fragment-{(commit - 1) % 4:04}.f32le")
    tape_manifest = f"{RUN}/work/cplg_shadow_on/stock_vectors/stock_tape.manifest.json"
    assert tape_manifest in paths
    assert f"{tape_manifest}.sha256" in paths
    assert f"{tape_manifest}.sha256" in manifests
    assert f"{RUN}/spec.json" in paths
    assert f"{RUN}/command.sh" in paths
    assert f"{RUN}/input-provenance.sha256" in manifests
