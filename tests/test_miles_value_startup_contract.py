"""Shell-launcher contracts for Miles value training."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROD_LAUNCHER = ROOT / "scripts" / "run_miles_value_island_prod.sh"
SYNCER_SUPERVISOR = ROOT / "scripts" / "run_miles_value_syncer_supervisor.sh"
LAUNCHERS = (
    ROOT / "scripts" / "run_miles_value_island_smoke.sh",
    PROD_LAUNCHER,
)
SYNC_INTERVAL_ARGUMENT = {
    "run_miles_value_island_smoke": (
        '"${SYNCER_ADDR}" "${SYNC_INTERVAL_STEPS}" <<\'PY\''
    ),
    "run_miles_value_island_prod": (
        '"${LOCAL_BUDGET_STEPS}" "${SYNC_INTERVAL_STEPS}" <<\'PY\''
    ),
}


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda path: path.stem)
def test_launcher_wires_h12_to_required_learner_gate(launcher: Path):
    source = launcher.read_text(encoding="utf-8")

    assert "readonly SYNC_INTERVAL_STEPS=${SYNC_INTERVAL_STEPS:-12}" in source
    assert SYNC_INTERVAL_ARGUMENT[launcher.stem] in source
    assert "min_local_steps" in source
    assert '"YETO_VALUE_MIN_LOCAL_STEPS": min_local_steps' in source


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda path: path.stem)
def test_launcher_rejects_nonpositive_learner_h_before_startup(launcher: Path):
    env = {
        **os.environ,
        "SYNC_INTERVAL_STEPS": "0",
        "SYNCER_ADDR": "127.0.0.1:29400",
        "LEARNER_ID": "0",
        "ISLAND_DATA_TEMPLATE": "/tmp/data_{rollout_id}.pt",
        "OUTPUT_DIR": "/tmp/yeto-test-output",
    }
    result = subprocess.run(
        ["bash", str(launcher)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "SYNC_INTERVAL_STEPS must be a positive integer" in result.stderr


def test_production_budget_override_is_shared_and_rollout_cannot_diverge():
    learner = PROD_LAUNCHER.read_text(encoding="utf-8")
    syncer = SYNCER_SUPERVISOR.read_text(encoding="utf-8")
    override = "readonly LOCAL_BUDGET_STEPS=${LOCAL_BUDGET_STEPS:-364}"

    assert override in learner
    assert override in syncer
    assert learner.count("readonly NUM_ROLLOUT=") == 1
    assert "readonly NUM_ROLLOUT=${LOCAL_BUDGET_STEPS}" in learner
    assert '--num-rollout "${NUM_ROLLOUT}"' in learner
    assert '"YETO_VALUE_BUDGET_STEPS": budget_steps' in learner
    assert '--learner-budget-steps "${LOCAL_BUDGET_STEPS}"' in syncer


def test_production_lr_decay_override_keeps_existing_default_and_cli_wiring():
    learner = PROD_LAUNCHER.read_text(encoding="utf-8")

    assert "readonly LR_DECAY_ITERS=${LR_DECAY_ITERS:-105}" in learner
    assert '[[ "${LR_DECAY_ITERS}" =~ ^[1-9][0-9]*$ ]]' in learner
    assert '--lr-decay-iters "${LR_DECAY_ITERS}"' in learner


def test_production_phase2_rejoin_is_explicit_and_fail_closed():
    learner = PROD_LAUNCHER.read_text(encoding="utf-8")

    assert "readonly PHASE2_REJOIN=${PHASE2_REJOIN:-0}" in learner
    assert '((LOCAL_BUDGET_STEPS == 364)) || die "PHASE2_REJOIN requires LOCAL_BUDGET_STEPS=364"' in learner
    assert '((checkpoint_iteration + 1 == START_ROLLOUT_ID))' in learner
    assert '"YETO_VALUE_PHASE2_REJOIN": os.environ["PHASE2_REJOIN_VALUE"]' in learner
    assert '--critic-load "${CRITIC_LOAD_DIR}"' in learner
    assert '--start-rollout-id "${START_ROLLOUT_ID}"' in learner
    assert '--train-memory-margin-bytes "${TRAIN_MEMORY_MARGIN_BYTES}"' in learner
    assert 'RECOVERY_CLI_ARGS+=(--low-memory-resume)' in learner
    assert '"${RECOVERY_CLI_ARGS[@]}"' in learner


def _production_learner_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "LEARNER_ID": "0",
            "SYNCER_ADDR": "127.0.0.1:29400",
            "ISLAND_DATA_TEMPLATE": "/definitely-not-used/data_{rollout_id}.pt",
            "OUTPUT_DIR": "/definitely-not-used/output",
            "LOCAL_BUDGET_STEPS": "364",
            "LR_DECAY_ITERS": "105",
            "SAVE_INTERVAL": "15",
            "SYNC_INTERVAL_STEPS": "12",
        }
    )
    return env


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        (
            "LOCAL_BUDGET_STEPS",
            "0",
            "LOCAL_BUDGET_STEPS must be a positive integer",
        ),
        (
            "LOCAL_BUDGET_STEPS",
            "not-an-integer",
            "LOCAL_BUDGET_STEPS must be a positive integer",
        ),
        ("LR_DECAY_ITERS", "-1", "LR_DECAY_ITERS must be a positive integer"),
        ("LR_DECAY_ITERS", "1.5", "LR_DECAY_ITERS must be a positive integer"),
    ],
)
def test_production_learner_rejects_invalid_overrides_before_preflight(
    name: str, value: str, message: str
):
    env = _production_learner_env()
    env[name] = value

    result = subprocess.run(
        ["bash", str(PROD_LAUNCHER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert message in result.stderr
    assert "missing" not in result.stderr


@pytest.mark.parametrize("value", ["0", "-1", "not-an-integer", "1.5"])
def test_production_syncer_rejects_invalid_budget_before_preflight(value: str):
    env = dict(os.environ)
    env.pop("SMOKE_BUDGET_STEPS", None)
    env.update(
        {
            "YETO_MILES_VALUE_SMOKE": "0",
            "LOCAL_BUDGET_STEPS": value,
        }
    )

    result = subprocess.run(
        ["bash", str(SYNCER_SUPERVISOR)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "LOCAL_BUDGET_STEPS must be a positive integer" in result.stderr
    assert "set RUN_DIR" not in result.stderr


def test_smoke_syncer_ignores_production_budget_override():
    env = dict(os.environ)
    env.pop("RUN_DIR", None)
    env.update(
        {
            "YETO_MILES_VALUE_SMOKE": "1",
            "SMOKE_BUDGET_STEPS": "3",
            "LOCAL_BUDGET_STEPS": "not-a-production-budget",
        }
    )

    result = subprocess.run(
        ["bash", str(SYNCER_SUPERVISOR)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "set RUN_DIR" in result.stderr
    assert "LOCAL_BUDGET_STEPS must be a positive integer" not in result.stderr
