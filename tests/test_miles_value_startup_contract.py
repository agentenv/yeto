"""Shell-launcher contract for Miles value learner synchronization cadence."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = (
    ROOT / "scripts" / "run_miles_value_island_smoke.sh",
    ROOT / "scripts" / "run_miles_value_island_prod.sh",
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
