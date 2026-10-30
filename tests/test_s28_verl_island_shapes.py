"""S28 G-MEG: verl islands of different parallel shapes in one syncer session.

A verl island's full-parameter layout is planned from the HF model's parameter
shapes (``full_parameter_plan``), so it is the same for every FSDP2 or Megatron
shape; ``--rl-island-placement ISLAND:tensor-parallel=N`` therefore needs no
``--rl-sync-layout shape-free`` on a verl run. A Miles run still does.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from yeto.rl.engine import island_placement as ip

# island 0: TP2 on 2 trainer cards; island 1: TP2 x PP2 on 4 trainer cards (2 rollout cards each).
# Not TP4 on 8 cards with 2 standby: the verl island ignores --rl-standby-gpus (G-MEG.md).
SHAPES = ("--rl-island-placement", "1:pipeline-parallel=2", "--rl-allow-pipeline-parallel-sync")


def _two_shape_launch(*extra):
    from test_s27_verl_full_parameter_launch import _prepared

    return _prepared(_launcher_args("ports", (
        "--rl-backend", "verl", "--rl-sync-preset", "full-parameter",
        "--rl-parameter-layout-sha256", "e" * 64, "--fragments", "4", "--reward-function", "gsm8k_reward:score",
        "--tensor-parallel", "2", "--rl-placement", "fixed-partition", "--rl-rollout-gpus", "2",
        "--rl-max-policy-age", "0", *SHAPES, *extra), gpu="modal:4xh200,modal:6xh200"))


def test_two_megatron_islands_of_different_shapes_share_one_session(tmp_path):
    from test_s27_verl_full_parameter_launch import MILES_ONLY, _flags, _island_argv

    from yeto import launcher
    from yeto.rl.adapters.verl import full_parameter_stream as vfs
    from yeto.rl.adapters.verl import island_entry

    args = _two_shape_launch()
    profile = launcher.rl_syncer_profile(args, 2).sha256
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "qwen3"}')
    seen = []
    for learner_id, (pp, cards) in enumerate([(1, 4), (2, 6)]):
        parsed, ignored = island_entry.parse_args(_island_argv(args, learner_id))
        # --rl-per-island-placement switches Miles' training contract to leave the
        # placement flags out; the verl contract never holds the shape
        # (full_parameter_stream.CONTRACT_RUN_FIELDS), so the verl island ignores it
        allowed = MILES_ONLY | {"--rl-per-island-placement"}
        assert _flags(ignored) <= allowed, _flags(ignored) - allowed
        assert (parsed.tensor_parallel, parsed.pipeline_parallel) == (2, pp)
        plan = island_entry.megatron_island_plan(
            parsed, model_path=str(model), train_file="/d/train.parquet", val_file="/d/test.parquet",
            out_dir="/o", algo={"correction": "tis", "tis_upper": 2.0, "tis_lower": 0.0}, chat={},
            ray_info={"after": {"GPU": cards}}, reward_path="/r.py", reward_name="gsm8k")
        shape = plan["shape"]
        assert (shape["tp"], shape["pp"], shape["trainer_gpus"], shape["rollout_gpus"]) == (2, pp, 2 * pp, 2)
        section = vfs.plan_section(parsed, plan["run"].to_dict(), model_path=str(model), sync="strict",
                                   algorithm_spec_sha256="a" * 64)["full_parameter"]
        seen.append((section["syncer_profile_sha256"], section["parameter_layout_sha256"],
                     section["training_contract_sha256"]))
    # one session: both islands bind the run's one syncer profile, one layout and
    # one training contract
    assert [s[:2] for s in seen] == [(profile, "e" * 64)] * 2
    assert seen[0][2] == seen[1][2]


def test_a_miles_run_with_per_island_shapes_still_needs_shape_free():
    args = SimpleNamespace(rl_backend="miles", rl_island_placement=["1:tensor-parallel=4"],
                           rl_sync_layout="shape-bound", rl_placement="fixed-partition", rl_standby_gpus=0,
                           rl_island_use_gpus_per_node=None, tensor_parallel=2, expert_parallel=None,
                           pipeline_parallel=1, rl_max_tokens_per_gpu=None)
    with pytest.raises(ValueError, match="only with --rl-sync-layout shape-free"):
        ip.check(args, 2, island_scheduling="legacy")
    args.rl_backend = "verl"
    placements = ip.check(args, 2, island_scheduling="legacy")
    assert dict(placements[1].parallel)["tensor_parallel"] == 4
    args.rl_backend, args.rl_sync_layout = "miles", "shape-free"
    ip.check(args, 2, island_scheduling="legacy")
