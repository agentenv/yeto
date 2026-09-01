"""Shell-launcher contracts for Miles value training."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

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
        '"${LOCAL_BUDGET_STEPS}" "${SYNC_INTERVAL_STEPS}" \\\n'
        '    "${TOPOLOGY_PROFILE}" "${ANCHOR_SPILL_DIR}" <<\'PY\''
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
    override = (
        "readonly LOCAL_BUDGET_STEPS="
        "${LOCAL_BUDGET_STEPS:-${DEFAULT_LOCAL_BUDGET_STEPS}}"
    )

    assert override in learner
    assert override in syncer
    assert "readonly DEFAULT_LOCAL_BUDGET_STEPS=240" in learner
    assert "readonly DEFAULT_LOCAL_BUDGET_STEPS=240" in syncer
    assert "readonly DEFAULT_LOCAL_BUDGET_STEPS=48" in learner
    assert "readonly DEFAULT_LOCAL_BUDGET_STEPS=48" in syncer
    assert learner.count("readonly NUM_ROLLOUT=") == 2
    assert "readonly NUM_ROLLOUT=${LOCAL_BUDGET_STEPS}" in learner
    assert "readonly NUM_ROLLOUT=$((CANARY_ROLLOUT_ID + 1))" in learner
    assert '--num-rollout "${NUM_ROLLOUT}"' in learner
    assert '"YETO_VALUE_BUDGET_STEPS": budget_steps' in learner
    assert '--learner-budget-steps "${LOCAL_BUDGET_STEPS}"' in syncer


def test_two_learner_diagnostic_contract_is_explicit_and_shared():
    learner = PROD_LAUNCHER.read_text(encoding="utf-8")
    syncer = SYNCER_SUPERVISOR.read_text(encoding="utf-8")
    flag = "YETO_MILES_VALUE_TWO_LEARNER_DIAGNOSTIC"

    assert flag in learner
    assert flag in syncer
    assert "readonly NUM_LEARNERS=2" in learner
    assert "readonly NUM_LEARNERS=2" in syncer
    assert "readonly NUM_LEARNERS=5" in learner
    assert "readonly NUM_LEARNERS=5" in syncer
    assert 'require(manifest.get("num_islands") == 5' in learner
    assert 'require(num_learners in (2, 5)' in learner
    assert '--learners "${NUM_LEARNERS}"' in syncer
    assert syncer.count('--quorum "${NUM_LEARNERS}"') == 2


def test_tp4_cp1_diagnostic_is_explicit_and_keeps_eight_gpu_default():
    learner = PROD_LAUNCHER.read_text(encoding="utf-8")
    syncer = SYNCER_SUPERVISOR.read_text(encoding="utf-8")

    assert "YETO_MILES_VALUE_TP4_CP1" in learner
    assert "YETO_MILES_VALUE_TP4_CP1" in syncer
    assert "readonly DEFAULT_LOCAL_BUDGET_STEPS=24" in learner
    assert "readonly DEFAULT_LOCAL_BUDGET_STEPS=24" in syncer
    assert "readonly ISLAND_GPU_COUNT=4" in learner
    assert "readonly ISLAND_GPU_COUNT=8" in learner
    assert "readonly CONTEXT_PARALLEL_SIZE=1" in learner
    assert "readonly CONTEXT_PARALLEL_SIZE=2" in learner
    assert "readonly TOPOLOGY_PROFILE=tp4-cp1" in learner
    assert "readonly TOPOLOGY_PROFILE=tp4-cp2" in learner
    assert '--num-gpus-per-node "${ISLAND_GPU_COUNT}"' in learner
    assert '--actor-num-gpus-per-node "${ISLAND_GPU_COUNT}"' in learner
    assert '--critic-num-gpus-per-node "${ISLAND_GPU_COUNT}"' in learner
    assert '--context-parallel-size "${CONTEXT_PARALLEL_SIZE}"' in learner
    assert "--optimizer-cpu-offload" in learner
    assert "--optimizer-offload-fraction 1.0" in learner
    assert "OPTIMIZER_RESIDENCY_ARGS=(--offload-optimizer-states)" in learner
    assert '"YETO_VALUE_TOPOLOGY_PROFILE": topology_profile' in learner
    assert '[[ "$(ulimit -l)" == unlimited ]]' in learner
    assert 'Path("/sys/fs/cgroup/memory.max")' in learner
    assert 'alive_nodes = [node for node in ray.nodes() if node.get("Alive")]' in learner
    assert 'cluster_gpus = float(ray.cluster_resources().get("GPU", 0.0))' in learner
    assert "minimum_hbm = 139 * 1024**3" in learner
    assert "the TP4/CP1 profile requires H200-class >=139 GiB GPUs" in learner
    assert 'die "the TP4/CP1 A/B diagnostic requires CONTEXT_LENGTH=131072"' in learner


def test_tp4_cp1_single_learner_canary_is_explicit_and_never_checkpoints():
    learner = PROD_LAUNCHER.read_text(encoding="utf-8")

    assert "YETO_MILES_VALUE_TP4_CP1_CANARY" in learner
    assert "readonly DEFAULT_LOCAL_BUDGET_STEPS=1" in learner
    assert "readonly CANARY_ROLLOUT_ID=${CANARY_ROLLOUT_ID:-167}" in learner
    assert 'die "TP4/CP1 canary requires VALUE_SYNC_MODE=none"' in learner
    assert 'die "TP4/CP1 canary requires LOCAL_BUDGET_STEPS=1"' in learner
    assert "TP4/CP1 canary forbids MILES_OFFLINE_VALIDATION_START_ROLLOUT" in learner
    assert 'if set(config) != {"gae_adaptive"}' in learner
    assert "if ((TP4_CP1_CANARY == 0)); then" in learner
    checkpoint_block = learner[
        learner.index("CHECKPOINT_ARGS=(") : learner.index(
            "# Decoder-only Megatron", learner.index("CHECKPOINT_ARGS=(")
        )
    ]
    assert '--save-interval "${SAVE_INTERVAL}"' in checkpoint_block
    assert '--save-retain-interval "${SAVE_RETAIN_INTERVAL}"' in checkpoint_block


def test_no_diloco_baseline_model_only_checkpoint_is_explicit_and_fail_closed():
    learner = PROD_LAUNCHER.read_text(encoding="utf-8")

    assert "YETO_MILES_VALUE_MODEL_ONLY_CHECKPOINT" in learner
    assert "model-only DiLoCo checkpoints are restricted to the 128K two-learner diagnostic" in learner
    assert 'die "model-only checkpointing cannot be used for an optimizer-resume run"' in learner
    assert 'die "model-only diagnostic must checkpoint exactly once at its terminal step"' in learner
    assert "CHECKPOINT_ARGS+=(--no-save-optim --no-save-rng)" in learner
    assert "CRITIC_PLACEMENT_ARGS=(--colocate-critic)" in learner
    assert "CRITIC_PLACEMENT_ARGS=()" in learner
    assert '"${CRITIC_PLACEMENT_ARGS[@]}"' in learner


def test_model_only_two_learner_checkpoint_owner_is_strict_and_nonconcurrent():
    learner = PROD_LAUNCHER.read_text(encoding="utf-8")

    assert (
        "readonly CHECKPOINT_OWNER=${YETO_MILES_VALUE_CHECKPOINT_OWNER:-0}"
        in learner
    )
    assert 'die "YETO_MILES_VALUE_CHECKPOINT_OWNER must be learner 0 or 1"' in learner
    assert "MODEL_ONLY_CHECKPOINT && TWO_LEARNER_DIAGNOSTIC && TP4_CP1_DIAGNOSTIC" in learner
    assert "readonly CHECKPOINT_WRITER=$((LEARNER_ID == CHECKPOINT_OWNER))" in learner
    assert "readonly CHECKPOINT_WRITER=1" in learner
    assert 'die "checkpoint-owner selection is restricted to the model-only 128K two-learner diagnostic"' in learner
    assert "if ((TP4_CP1_CANARY == 0 && CHECKPOINT_WRITER)); then" in learner
    assert "checkpoint_owner=%s checkpoint_writer=%d" in learner


def test_anchor_spill_is_strictly_gated_and_propagated_to_workers():
    learner = PROD_LAUNCHER.read_text(encoding="utf-8")

    assert "YETO_MILES_VALUE_ANCHOR_SPILL" in learner
    assert 'die "anchor spill is restricted to the TP4/CP1 two-learner diagnostic"' in learner
    assert 'die "anchor spill requires VALUE_SYNC_MODE=diloco"' in learner
    assert 'die "anchor spill requires the fresh 128K 24-step diagnostic contract"' in learner
    assert "${OUTPUT_DIR}/diloco_anchors" in learner
    assert "anchor spill directory already exists" in learner
    assert 'worker_env["YETO_VALUE_ANCHOR_SPILL_DIR"] = anchor_spill_dir' in learner
    assert "anchor_residency=%s anchor_spill_dir=%s" in learner


def test_anchor_spill_rejects_out_of_scope_launch_before_dataset_access():
    env = _production_learner_env()
    env["YETO_MILES_VALUE_ANCHOR_SPILL"] = "1"

    result = subprocess.run(
        ["bash", str(PROD_LAUNCHER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "anchor spill is restricted" in result.stderr
    assert "dataset manifest" not in result.stderr


def test_hdo_gradient_stream_is_strictly_gated_and_propagated_to_workers():
    learner = PROD_LAUNCHER.read_text(encoding="utf-8")

    assert "YETO_MILES_VALUE_HDO_GRADIENT_STREAM" in learner
    assert (
        'die "HDO gradient streaming is restricted to the TP4/CP1 '
        'two-learner diagnostic"' in learner
    )
    assert 'die "HDO gradient streaming requires VALUE_SYNC_MODE=diloco"' in learner
    assert (
        'die "HDO gradient streaming requires the fresh 128K 24-step '
        'diagnostic contract"' in learner
    )
    assert (
        'die "HDO gradient streaming requires '
        'TP4_CP1_OVERLAP_CPU_OPTIMIZER=0"' in learner
    )
    assert (
        '"YETO_VALUE_HDO_GRADIENT_STREAM": '
        'os.environ["HDO_GRADIENT_STREAM_VALUE"]' in learner
    )


def test_hdo_gradient_stream_rejects_out_of_scope_launch_before_dataset_access():
    env = _production_learner_env()
    env["YETO_MILES_VALUE_HDO_GRADIENT_STREAM"] = "1"

    result = subprocess.run(
        ["bash", str(PROD_LAUNCHER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "HDO gradient streaming is restricted" in result.stderr
    assert "dataset manifest" not in result.stderr


def test_hdo_gradient_stream_accepts_exact_diagnostic_shape_before_data_gate(
    tmp_path: Path,
):
    env = _production_learner_env()
    env.update(
        {
            "YETO_MILES_VALUE_HDO_GRADIENT_STREAM": "1",
            "YETO_MILES_VALUE_TWO_LEARNER_DIAGNOSTIC": "1",
            "YETO_MILES_VALUE_TP4_CP1": "1",
            "CONTEXT_LENGTH": "131072",
            "LOCAL_BUDGET_STEPS": "24",
            "SAVE_INTERVAL": "24",
            "SAVE_RETAIN_INTERVAL": "24",
            "LEARNER_ID": "0",
            "ISLAND_DATA_TEMPLATE": str(
                tmp_path / "bundle" / "island_0" / "data_{rollout_id}.pt"
            ),
            "OUTPUT_DIR": str(tmp_path / "output"),
        }
    )

    result = subprocess.run(
        ["bash", str(PROD_LAUNCHER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "HDO gradient streaming" not in result.stderr
    assert "missing dataset manifest" in result.stderr


@pytest.mark.parametrize("learner_id", ["0", "1"])
def test_model_only_two_learner_owner_zero_accepts_writer_and_nonwriter(
    tmp_path: Path, learner_id: str
):
    env = _production_learner_env()
    env.update(
        {
            "YETO_MILES_VALUE_TWO_LEARNER_DIAGNOSTIC": "1",
            "YETO_MILES_VALUE_TP4_CP1": "1",
            "YETO_MILES_VALUE_MODEL_ONLY_CHECKPOINT": "1",
            "YETO_MILES_VALUE_CHECKPOINT_OWNER": "0",
            "CONTEXT_LENGTH": "131072",
            "LOCAL_BUDGET_STEPS": "24",
            "SAVE_INTERVAL": "24",
            "SAVE_RETAIN_INTERVAL": "24",
            "LEARNER_ID": learner_id,
            "ISLAND_DATA_TEMPLATE": str(
                tmp_path / "bundle" / f"island_{learner_id}" / "data_{rollout_id}.pt"
            ),
            "OUTPUT_DIR": str(tmp_path / f"output-{learner_id}"),
        }
    )

    result = subprocess.run(
        ["bash", str(PROD_LAUNCHER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "checkpoint owner" not in result.stderr
    assert "checkpoint-owner selection" not in result.stderr
    assert "missing dataset manifest" in result.stderr


@pytest.mark.parametrize(
    ("owner", "overrides", "message"),
    [
        (
            "invalid",
            {},
            "YETO_MILES_VALUE_CHECKPOINT_OWNER must be learner 0 or 1",
        ),
        (
            "1",
            {},
            "checkpoint-owner selection is restricted to the model-only 128K two-learner diagnostic",
        ),
    ],
)
def test_checkpoint_owner_rejects_invalid_or_out_of_scope_use(
    owner: str, overrides: dict[str, str], message: str
):
    env = _production_learner_env()
    env["YETO_MILES_VALUE_CHECKPOINT_OWNER"] = owner
    env.update(overrides)

    result = subprocess.run(
        ["bash", str(PROD_LAUNCHER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert message in result.stderr
    assert "dataset manifest" not in result.stderr


def test_tp4_cp1_single_learner_canary_reaches_dataset_preflight(tmp_path: Path):
    env = _production_learner_env()
    env.pop("LOCAL_BUDGET_STEPS", None)
    env.update(
        {
            "YETO_MILES_VALUE_TP4_CP1_CANARY": "1",
            "VALUE_SYNC_MODE": "none",
            "SYNCER_ADDR": "none",
            "LEARNER_ID": "1",
            "CANARY_ROLLOUT_ID": "167",
            "ISLAND_DATA_TEMPLATE": str(
                tmp_path / "bundle" / "island_1" / "data_{rollout_id}.pt"
            ),
            "OUTPUT_DIR": str(tmp_path / "output"),
        }
    )

    result = subprocess.run(
        ["bash", str(PROD_LAUNCHER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "TP4/CP1 canary requires" not in result.stderr
    assert "missing dataset manifest" in result.stderr


def test_tp4_cp1_learner_accepts_24_step_shape_before_dataset_preflight(
    tmp_path: Path,
):
    env = _production_learner_env()
    env.update(
        {
            "YETO_MILES_VALUE_TWO_LEARNER_DIAGNOSTIC": "1",
            "YETO_MILES_VALUE_TP4_CP1": "1",
            "CONTEXT_LENGTH": "131072",
            "LOCAL_BUDGET_STEPS": "24",
            "SAVE_INTERVAL": "24",
            "SAVE_RETAIN_INTERVAL": "24",
            "ISLAND_DATA_TEMPLATE": str(
                tmp_path / "bundle" / "island_0" / "data_{rollout_id}.pt"
            ),
            "OUTPUT_DIR": str(tmp_path / "output"),
        }
    )

    result = subprocess.run(
        ["bash", str(PROD_LAUNCHER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "TP4/CP1 diagnostic requires" not in result.stderr
    assert "missing dataset manifest" in result.stderr


def test_tp4_cp1_syncer_accepts_24_step_shape_before_binary_preflight(
    tmp_path: Path,
):
    env = dict(os.environ)
    env.update(
        {
            "YETO_MILES_VALUE_SMOKE": "0",
            "YETO_MILES_VALUE_TWO_LEARNER_DIAGNOSTIC": "1",
            "YETO_MILES_VALUE_TP4_CP1": "1",
            "LOCAL_BUDGET_STEPS": "24",
            "RUN_DIR": str(tmp_path / "run"),
            "SYNCER_BIN": str(tmp_path / "missing-syncer"),
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
    assert "TP4/CP1 diagnostic requires" not in result.stderr
    assert "syncer binary is not executable" in result.stderr


def test_two_learner_diagnostic_accepts_learner_one_and_default_budget(
    tmp_path: Path,
):
    env = _production_learner_env()
    env.pop("LOCAL_BUDGET_STEPS")
    env.update(
        {
            "YETO_MILES_VALUE_TWO_LEARNER_DIAGNOSTIC": "1",
            "LEARNER_ID": "1",
            "ISLAND_DATA_TEMPLATE": str(
                tmp_path / "bundle" / "island_1" / "data_{rollout_id}.pt"
            ),
            "OUTPUT_DIR": str(tmp_path / "output"),
        }
    )

    result = subprocess.run(
        ["bash", str(PROD_LAUNCHER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "two-learner diagnostic requires" not in result.stderr
    assert "missing dataset manifest" in result.stderr


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"YETO_MILES_VALUE_TWO_LEARNER_DIAGNOSTIC": "maybe"},
            "YETO_MILES_VALUE_TWO_LEARNER_DIAGNOSTIC must be 0 or 1",
        ),
        (
            {
                "YETO_MILES_VALUE_TWO_LEARNER_DIAGNOSTIC": "1",
                "LOCAL_BUDGET_STEPS": "24",
            },
            "two-learner diagnostic requires LOCAL_BUDGET_STEPS=48",
        ),
        (
            {
                "YETO_MILES_VALUE_TWO_LEARNER_DIAGNOSTIC": "1",
                "YETO_MILES_VALUE_TP4_CP1": "1",
                "CONTEXT_LENGTH": "131072",
                "LOCAL_BUDGET_STEPS": "48",
            },
            "TP4/CP1 diagnostic requires LOCAL_BUDGET_STEPS=24",
        ),
        (
            {
                "YETO_MILES_VALUE_TP4_CP1": "1",
                "LOCAL_BUDGET_STEPS": "24",
            },
            "YETO_MILES_VALUE_TP4_CP1 requires a diagnostic or canary mode",
        ),
        (
            {
                "YETO_MILES_VALUE_TWO_LEARNER_DIAGNOSTIC": "1",
                "YETO_MILES_VALUE_TP4_CP1": "1",
                "CONTEXT_LENGTH": "131072",
                "LOCAL_BUDGET_STEPS": "24",
                "SAVE_INTERVAL": "24",
                "SAVE_RETAIN_INTERVAL": "24",
                "PHASE2_REJOIN": "1",
            },
            "PHASE2_REJOIN is not supported for TP4/CP1 CPU-offloaded checkpoints",
        ),
        (
            {
                "YETO_MILES_VALUE_TWO_LEARNER_DIAGNOSTIC": "1",
                "LEARNER_ID": "2",
                "LOCAL_BUDGET_STEPS": "48",
            },
            "LEARNER_ID must be in [0, 1]",
        ),
        (
            {
                "YETO_MILES_VALUE_TWO_LEARNER_DIAGNOSTIC": "1",
                "VALUE_SYNC_MODE": "none",
                "SYNCER_ADDR": "none",
                "LOCAL_BUDGET_STEPS": "48",
            },
            "two-learner diagnostic requires VALUE_SYNC_MODE=diloco",
        ),
    ],
)
def test_two_learner_diagnostic_learner_rejects_unsafe_shapes_before_preflight(
    overrides: dict[str, str], message: str
):
    env = _production_learner_env()
    env.update(overrides)

    result = subprocess.run(
        ["bash", str(PROD_LAUNCHER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert message in result.stderr
    assert "dataset manifest" not in result.stderr


def test_two_learner_diagnostic_syncer_reaches_binary_preflight(tmp_path: Path):
    env = dict(os.environ)
    env.pop("SMOKE_BUDGET_STEPS", None)
    env.update(
        {
            "YETO_MILES_VALUE_SMOKE": "0",
            "YETO_MILES_VALUE_TWO_LEARNER_DIAGNOSTIC": "1",
            "RUN_DIR": str(tmp_path / "run"),
            "SYNCER_BIN": str(tmp_path / "missing-syncer"),
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
    assert "two-learner diagnostic requires" not in result.stderr
    assert "syncer binary is not executable" in result.stderr


def test_two_learner_diagnostic_is_unavailable_in_smoke(tmp_path: Path):
    env = dict(os.environ)
    env.update(
        {
            "YETO_MILES_VALUE_SMOKE": "1",
            "YETO_MILES_VALUE_TWO_LEARNER_DIAGNOSTIC": "1",
            "SMOKE_BUDGET_STEPS": "2",
            "RUN_DIR": str(tmp_path / "run"),
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
    assert "two-learner diagnostic is unavailable in smoke mode" in result.stderr
    assert "syncer binary is not executable" not in result.stderr


def test_production_value_sync_mode_selects_exact_hook_contracts():
    learner = PROD_LAUNCHER.read_text(encoding="utf-8")

    assert "VALUE_SYNC_MODE=${VALUE_SYNC_MODE:-diloco}" in learner
    hook_start = learner.index(
        'if [[ "${VALUE_SYNC_MODE}" == diloco ]]', learner.index("Launching Miles")
    )
    hook_end = learner.index("# Decoder-only Megatron", hook_start)
    hook_block = learner[hook_start:hook_end]
    assert hook_block.count("--custom-megatron-") == 4
    assert (
        "--custom-megatron-after-model-init-hook-path "
        "yeto.megatron.miles_value_island.after_model_init"
    ) in hook_block
    assert (
        "--custom-megatron-before-train-step-hook-path "
        "yeto.megatron.miles_value_island.before_train_step"
    ) in hook_block
    assert (
        "--custom-megatron-after-train-step-hook-path "
        "yeto.megatron.miles_value_island.after_train_step"
    ) in hook_block
    assert (
        "--custom-megatron-after-model-init-hook-path "
        "yeto_value_validation_hook.after_model_init"
    ) in hook_block
    assert learner.count('"${VALUE_HOOK_ARGS[@]}"') == 1


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"VALUE_SYNC_MODE": "none", "SYNCER_ADDR": "127.0.0.1:29400"},
            "SYNCER_ADDR must be unset or 'none' in no-sync mode",
        ),
        (
            {
                "VALUE_SYNC_MODE": "none",
                "SYNCER_ADDR": "none",
                "PHASE2_REJOIN": "1",
            },
            "PHASE2_REJOIN is unavailable in no-sync mode",
        ),
        (
            {"VALUE_SYNC_MODE": "invalid"},
            "VALUE_SYNC_MODE must be diloco or none",
        ),
    ],
)
def test_production_no_sync_rejects_misuse_before_preflight(
    overrides: dict[str, str], message: str
):
    env = _production_learner_env()
    env.update(overrides)

    result = subprocess.run(
        ["bash", str(PROD_LAUNCHER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert message in result.stderr
    assert "dataset manifest" not in result.stderr
    assert "missing Miles" not in result.stderr


def test_value_syncer_defaults_to_equal_island_weight_with_legacy_override():
    syncer = SYNCER_SUPERVISOR.read_text(encoding="utf-8")

    assert "LEARNER_WEIGHT=${LEARNER_WEIGHT:-equal}" in syncer
    assert (
        '[[ "${LEARNER_WEIGHT}" == equal || '
        '"${LEARNER_WEIGHT}" == tokens2-over-steps ]]'
    ) in syncer
    assert 'die "LEARNER_WEIGHT must be equal or tokens2-over-steps"' in syncer
    # COMMON_ARGS is shared by the ordinary and terminal-consolidation phases.
    assert syncer.count('--learner-weight "${LEARNER_WEIGHT}"') == 1


@pytest.mark.parametrize("value", ["token", "weighted", "tokens-over-steps"])
def test_value_syncer_rejects_invalid_learner_weight_before_preflight(
    tmp_path: Path, value: str
):
    env = dict(os.environ)
    env.update(
        {
            "YETO_MILES_VALUE_SMOKE": "1",
            "SMOKE_BUDGET_STEPS": "2",
            "RUN_DIR": str(tmp_path / "run"),
            "LEARNER_WEIGHT": value,
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
    assert "LEARNER_WEIGHT must be equal or tokens2-over-steps" in result.stderr
    assert "syncer binary is not executable" not in result.stderr


@pytest.mark.parametrize("value", [None, "equal", "tokens2-over-steps"])
def test_value_syncer_accepts_default_equal_and_explicit_legacy_weight(
    tmp_path: Path, value: str | None
):
    env = dict(os.environ)
    env.pop("LEARNER_WEIGHT", None)
    env.update(
        {
            "YETO_MILES_VALUE_SMOKE": "1",
            "SMOKE_BUDGET_STEPS": "2",
            "RUN_DIR": str(tmp_path / "run"),
            "SYNCER_BIN": str(tmp_path / "missing-syncer"),
        }
    )
    if value is not None:
        env["LEARNER_WEIGHT"] = value

    result = subprocess.run(
        ["bash", str(SYNCER_SUPERVISOR)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "LEARNER_WEIGHT must be" not in result.stderr
    assert "syncer binary is not executable" in result.stderr


def test_production_lr_schedule_matches_the_full_contrastive_pack():
    learner = PROD_LAUNCHER.read_text(encoding="utf-8")

    assert "readonly LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-5}" in learner
    assert "readonly LR_DECAY_ITERS=${LR_DECAY_ITERS:-138}" in learner
    assert '[[ "${LR_WARMUP_ITERS}" =~ ^[0-9]+$ ]]' in learner
    assert '[[ "${LR_DECAY_ITERS}" =~ ^[1-9][0-9]*$ ]]' in learner
    assert '--critic-lr-warmup-iters "${LR_WARMUP_ITERS}"' in learner
    assert '--lr-decay-iters "${LR_DECAY_ITERS}"' in learner


def test_production_critic_is_bounded_and_requires_stratified_replay():
    learner = PROD_LAUNCHER.read_text(encoding="utf-8")

    assert "atomic-thread-reward-contrastive-window-balanced-v2" in learner
    assert 'verification.get("launch_ready") is True' in learner
    assert 'verification.get("step_label_failures") == 0' in learner
    assert 'verification.get("window_label_failures") == 0' in learner
    assert '"atomic-group-equal-within-step-v1"' in learner
    assert "--value-loss-type classification" in learner
    assert "--value-num-bins 51" in learner
    assert "--value-target-type hl_gauss" in learner
    assert "--hl-gauss-sigma-ratio 0.75" in learner
    assert "--value-loss-type mse" not in learner


def test_production_rejects_old_unstratified_bundle_before_gpu_preflight(
    tmp_path: Path,
):
    bundle = tmp_path / "old-bundle"
    island = bundle / "island_0"
    island.mkdir(parents=True)
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "dataset_version": "qwen38-value-five-islands-merged-20260826-v1",
                "strategy": "atomic-thread-compaction-group-lpt-by-sum-true-token-length-squared",
            }
        ),
        encoding="utf-8",
    )
    env = _production_learner_env()
    env["ISLAND_DATA_TEMPLATE"] = str(island / "data_{rollout_id}.pt")
    env["OUTPUT_DIR"] = str(tmp_path / "output")

    result = subprocess.run(
        ["bash", str(PROD_LAUNCHER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "dataset manifest failed launch gates" in result.stderr
    assert "missing Miles" not in result.stderr


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
            "LOCAL_BUDGET_STEPS": "240",
            "LR_WARMUP_ITERS": "5",
            "LR_DECAY_ITERS": "138",
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
        ("LR_WARMUP_ITERS", "-1", "LR_WARMUP_ITERS must be a nonnegative integer"),
        ("LR_WARMUP_ITERS", "1.5", "LR_WARMUP_ITERS must be a nonnegative integer"),
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
