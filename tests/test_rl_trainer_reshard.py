"""rl-infra-spec 4.6 (CPU, protocol only): DP1<->2 resharding from a name-keyed cut.

The ranks are pure-torch stand-ins (``tests/rl_reshard_fakes.py``); these
tests prove the protocol and the refusals, not Megatron/fork-M5 behaviour on
GPU (that is A8 / X4 in evidence/infra-e3/plan.md).
"""

from __future__ import annotations

import importlib
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import pytest
import torch

from tests.rl_reshard_fakes import GBS, MBS, default_args, gathered, make_world, params, train_step
from yeto.rl.engine.cut import AlgorithmIdentity, CutError, CutProgress, RestoreExpectation
from yeto.rl.engine.miles_adapter import LoopRunner
from yeto.rl.engine.miles_adapter.reshard import (
    ReshardPlan,
    ReshardRefused,
    algorithm_problems,
    reshard_problems,
    rng_mapping,
    miles_microbatch_loss_scale,
    sample_mapping,
    sample_weight,
    scheduled_partitions,
    step_problems,
)
from yeto.rl.engine.miles_adapter.trainer import CutContext, MilesTrainerGroup

SPEC_SHA = "a" * 64
ALGO = AlgorithmIdentity(SPEC_SHA)


def _layout(dp):
    return {"world": dp, "tp": 1, "pp": 1, "cp": 1, "ep": 1, "dp": dp}


def _spec(**loss):
    return SimpleNamespace(
        loss=SimpleNamespace(aggregation="default", reducer=None, custom_loss=None, **loss),
        advantage=SimpleNamespace(estimator="grpo", whiten=False),
        sha256=lambda: SPEC_SHA,
    )


class RankGroup:
    def __init__(self, ranks):
        self.ranks = ranks

    async def run_plugin(self, fn_path, kwargs=None):
        module, name = fn_path.rsplit(".", 1)
        fn = getattr(importlib.import_module(module), name)
        return [fn(rank, **(kwargs or {})) for rank in self.ranks]


def _trainer(ranks, spec=None):
    return MilesTrainerGroup(args=ranks[0].args, actor_model=RankGroup(ranks), learner_id=0, learner_generation=0,
                             parameter_layout_hash=lambda: "L", runner=LoopRunner(), spec=spec or _spec())


def _context(tmp_path, cut_id, steps=2):
    return CutContext(
        root=str(tmp_path), cut_id=cut_id, backend_fingerprint="fp",
        progress=CutProgress(local_step=steps, scheduler_samples=steps * GBS, global_batch_size=GBS,
                             next_rollout_id=steps, policy_version=steps, policy_hash="h"),
        algorithm=ALGO,
        data={"sample_offset": 4, "epoch_id": 0, "sample_group_index": 4, "sample_index": 8},
        ledger={"carried_over": 0, "ready_unconsumed": 0}, outer={"settled": True},
    )


def _expect(dp, steps=2):
    return RestoreExpectation(algorithm=ALGO, layout=_layout(dp), backend_fingerprint="fp",
                              local_step=steps, policy_version=steps, epoch=1)


def _plan(src, dst):
    return ReshardPlan(_layout(src), _layout(dst), GBS, MBS)


def _batches():
    g = torch.Generator().manual_seed(7)
    return [torch.randn(GBS, 6, generator=g) for _ in range(3)]


def _trained(dp):
    ranks = make_world(dp)
    for b in _batches()[:2]:
        train_step(ranks, b)
    return ranks


@pytest.mark.parametrize("src,dst", [(1, 2), (2, 1)])
def test_reshard_restores_the_gathered_state_and_matches_the_next_step(tmp_path, src, dst):
    ranks = _trained(src)
    _trainer(ranks).save_cut(epoch=1, context=_context(tmp_path, "c"))
    new = make_world(dst, seed=99)  # fresh process: other init
    result = _trainer(new).restore_cut_resharded("c", epoch=1, root=str(tmp_path), expect=_expect(src),
                                                 plan=_plan(src, dst), certified=[SPEC_SHA])
    # X4a (protocol): master/moments/step/scheduler equal the source bitwise once gathered.
    before, after = gathered(ranks), gathered(new)
    for name in before:
        for key in ("param", "exp_avg", "exp_avg_sq"):
            assert torch.equal(before[name]["tensors"][key], after[name]["tensors"][key])
        assert torch.equal(before[name]["scalars"]["step"], after[name]["scalars"]["step"])
    for name, value in params(ranks).items():
        assert torch.equal(value, params(new)[name])
    assert all(r.opt_param_scheduler.num_steps == 2 * GBS for r in new)
    assert [m["source"] for m in result["rng_mapping"]] == ["fresh"] * dst
    # X4b (protocol): next step on the same frozen batch agrees up to summation order.
    third = _batches()[2]
    train_step(ranks, third)
    train_step(new, third)
    for name, value in params(ranks).items():
        torch.testing.assert_close(params(new)[name], value, rtol=1e-5, atol=1e-6)


def test_dp_roundtrip_1_2_1_is_lossless(tmp_path):
    ranks = _trained(1)
    _trainer(ranks).save_cut(epoch=1, context=_context(tmp_path, "c1"))
    two = make_world(2, seed=5)
    r2 = _trainer(two).restore_cut_resharded("c1", epoch=1, root=str(tmp_path), expect=_expect(1),
                                             plan=_plan(1, 2), certified=[SPEC_SHA])
    _trainer(two).save_cut(epoch=1, context=_context(tmp_path, "c2"))
    one = make_world(1, seed=6)
    r1 = _trainer(one).restore_cut_resharded("c2", epoch=1, root=str(tmp_path), expect=_expect(2),
                                             plan=_plan(2, 1), certified=[SPEC_SHA])
    assert r1["full_state_digests"] == r2["full_state_digests"]
    before, after = gathered(ranks), gathered(one)
    for name in before:
        for key in before[name]["tensors"]:
            assert torch.equal(before[name]["tensors"][key], after[name]["tensors"][key])


def test_unchanged_dp_restores_rng_exactly(tmp_path):
    ranks = _trained(2)
    _trainer(ranks).save_cut(epoch=1, context=_context(tmp_path, "c"))
    new = make_world(2, seed=3)
    result = _trainer(new).restore_cut_resharded("c", epoch=1, root=str(tmp_path), expect=_expect(2),
                                                 plan=replace(_plan(2, 2)), certified=[SPEC_SHA])
    assert {r["rng"] for r in result["ranks"]} == {"restored"}


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda s: SimpleNamespace(**{**vars(s), "loss": SimpleNamespace(aggregation="token", reducer=None,
                                                                          custom_loss=None)}), "aggregation"),
        (lambda s: SimpleNamespace(**{**vars(s), "advantage": SimpleNamespace(estimator="grpo", whiten=True)}),
         "whiten"),
        (lambda s: SimpleNamespace(**{**vars(s), "advantage": SimpleNamespace(estimator="gspo", whiten=False)}),
         "GRPO only"),
    ],
)
def test_uncertified_normalization_is_refused_before_any_write(tmp_path, mutate, match):
    ranks = _trained(1)
    _trainer(ranks).save_cut(epoch=1, context=_context(tmp_path, "c"))
    new = make_world(2, seed=9)
    snapshot = params(new)
    with pytest.raises(ReshardRefused, match=match):
        _trainer(new, spec=mutate(_spec())).restore_cut_resharded(
            "c", epoch=1, root=str(tmp_path), expect=_expect(1), plan=_plan(1, 2), certified=[SPEC_SHA])
    assert all(torch.equal(v, params(new)[n]) for n, v in snapshot.items())
    assert new[0].opt_param_scheduler.num_steps == 0


def test_uncertified_spec_hash_is_refused(tmp_path):
    ranks = _trained(1)
    _trainer(ranks).save_cut(epoch=1, context=_context(tmp_path, "c"))
    with pytest.raises(ReshardRefused, match="not certified"):
        _trainer(make_world(2)).restore_cut_resharded("c", epoch=1, root=str(tmp_path), expect=_expect(1),
                                                      plan=_plan(1, 2), certified=["b" * 64])


def test_layout_and_batch_refusals():
    p = ReshardPlan(_layout(1), {"world": 2, "tp": 2, "pp": 1, "cp": 1, "ep": 1, "dp": 1}, GBS, MBS)
    assert any("tp changes" in x for x in reshard_problems(p, spec=_spec()))
    p = ReshardPlan(_layout(1), _layout(3), 4, 1)
    assert any("not divisible" in x for x in reshard_problems(p, spec=_spec()))
    p = ReshardPlan(_layout(1), _layout(2), 4, 1, rng_policy="exact")
    assert any("exact" in x for x in reshard_problems(p, spec=_spec()))
    args = SimpleNamespace(**{**vars(default_args(2)), "calculate_per_token_loss": True, "fp16": True})
    problems = reshard_problems(_plan(1, 2), args=args, spec=_spec())
    assert any("per-token" in x for x in problems) and any("fp16" in x for x in problems)
    assert algorithm_problems(None)


def test_loss_weight_recomputed_with_the_fork_scale_is_dp_invariant():
    # fork loss_function: loss * num_microbatches / num_rollouts * dp (apply_megatron_loss_scaling)
    assert miles_microbatch_loss_scale(num_microbatches=8, num_rollouts=16, dp=2) == Fraction(1)
    for dp in (1, 2, 4):
        nmb = 16 // 2 // dp
        assert sample_weight(num_rollouts=16, num_microbatches=nmb, dp=dp) == Fraction(1, 16)


def test_scheduled_path_keeps_steps_and_micro_batches_and_changes_only_ranks():
    # multi-sample rollouts (repeated rollout id); 4 rollouts/step, mbs=1; rollout 8 is trailing -> dropped
    rids = [0, 0, 1, 2, 2, 3, 4, 5, 6, 6, 7, 7, 8]
    m = sample_mapping(rids, source_dp=1, target_dp=2, global_batch_size=4, micro_batch_size=1)
    assert m["problems"] == []
    assert m["source"]["num_rollouts"] == m["target"]["num_rollouts"] == [4, 4]
    assert m["source"]["num_microbatches"] == [6, 6] and m["target"]["num_microbatches"] == [3, 3]
    assert m["target"]["partitions"] == [[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]]
    assert m["source"]["partitions"] == [list(range(12))]


def test_step_problems_reports_odd_micro_batch_counts():
    rids = [0, 1, 1, 2, 3]  # step of 4 rollouts has 5 samples -> 5 micro-batches, not a multiple of dp 2
    assert step_problems(rids, _plan(1, 2))
    assert step_problems([0, 1, 2, 3], _plan(1, 2)) == []


def test_balance_data_and_dropout_are_refused_for_a_dp_change():
    args = SimpleNamespace(**{**vars(default_args(2)), "balance_data": True})
    assert any("balance-data" in x for x in reshard_problems(_plan(1, 2), args=args, spec=_spec()))
    for name, value in (("lora_dropout", 0.05), ("hidden_dropout", None), ("attention_dropout", 0.1)):
        a = SimpleNamespace(**{**vars(default_args(2)), name: value})
        assert any(name in x for x in reshard_problems(_plan(1, 2), args=a, spec=_spec())), name
    a = SimpleNamespace(**{**vars(default_args(2)), "lora_dropout": 0.05})
    assert not any("dropout" in x for x in reshard_problems(_plan(2, 2), args=a, spec=_spec()))


def test_rng_mapping_records_fresh_seed_derivation():
    args = SimpleNamespace(seed=1234, data_parallel_random_init=False)
    coords = [{"tp": 0, "pp": 0, "dp": d} for d in range(2)]
    mapping = rng_mapping(_plan(1, 2), coords, args)
    assert mapping[1] == {"coord": {"tp": 0, "pp": 0, "dp": 1}, "source": "fresh",
                          "seed": {"base_seed": 1234, "derived_seed": 1234, "cuda_tracker_seed": 3952}}


def test_missing_dp_shard_is_refused(tmp_path):
    ranks = _trained(2)
    _trainer(ranks).save_cut(epoch=1, context=_context(tmp_path, "c"))
    (tmp_path / "c" / "trainer_tp0_pp0_dp1.pt").unlink()
    with pytest.raises(CutError, match="missing shard"):
        _trainer(make_world(1)).restore_cut_resharded("c", epoch=1, root=str(tmp_path), expect=_expect(2),
                                                      plan=_plan(2, 1), certified=[SPEC_SHA])


def test_target_layout_is_read_back_from_the_ranks(tmp_path):
    ranks = _trained(1)
    _trainer(ranks).save_cut(epoch=1, context=_context(tmp_path, "c"))
    with pytest.raises(CutError, match="planned target"):
        _trainer(make_world(1)).restore_cut_resharded("c", epoch=1, root=str(tmp_path), expect=_expect(1),
                                                      plan=_plan(1, 2), certified=[SPEC_SHA])


FULL = {"dp_size": 2, "cp_size": 1, "vpp_size": 1, "microbatch_group_size_per_vp_stage": 1}


def test_indep_dp_and_multimodal_are_refused_before_any_write():
    for extra, match in (({"indep_dp": True}, "indep-dp"), ({"multimodal_keys": '{"image": "images"}'}, "multimodal")):
        args = SimpleNamespace(**{**vars(default_args(2)), **extra})
        assert any(match in x for x in reshard_problems(_plan(1, 2), args=args, spec=_spec()))


def test_batch_guard_refuses_what_the_fork_would_split_unscheduled():
    from yeto.rl.engine.miles_adapter.reshard import batch_guard_problems

    plan = _plan(1, 2)
    ok = batch_guard_problems(plan, rank_configs=[FULL, FULL], rollout_indices=list(range(GBS)), steps=1)
    assert ok == []
    assert any("lacks" in x for x in batch_guard_problems(plan, rank_configs=[{}, {}],
                                                          rollout_indices=list(range(GBS)), steps=1))
    assert any("dp_size 1" in x for x in batch_guard_problems(plan, rank_configs=[{**FULL, "dp_size": 1}],
                                                              rollout_indices=list(range(GBS)), steps=1))
    assert any("< GBS" in x for x in batch_guard_problems(plan, rank_configs=[FULL, FULL],
                                                          rollout_indices=list(range(GBS - 1)), steps=1))
    # multi-sample rollout -> odd micro-batch count on dp 2 (L1: static check assumed 1 sample/rollout)
    assert batch_guard_problems(plan, rank_configs=[FULL, FULL], rollout_indices=[0, 0, 1, 2, 3], steps=1)


def test_train_step_is_guarded_after_a_resharded_restore(tmp_path):
    from yeto.rl.engine.miles_adapter.trainer import TrainStepError

    ranks = _trained(1)
    _trainer(ranks).save_cut(epoch=1, context=_context(tmp_path, "c"))
    new = make_world(2, seed=9)
    trainer = _trainer(new)
    trainer.restore_cut_resharded("c", epoch=1, root=str(tmp_path), expect=_expect(1), plan=_plan(1, 2),
                                  certified=[SPEC_SHA])
    released = []
    trainer._release = lambda args, payload: released.append(payload)
    batch = SimpleNamespace(payload="refs", groups=[SimpleNamespace(sample_ids=(0, 1, 2, 3))],
                            rollout_id=3, policy_hash="h", policy_version=2)
    with pytest.raises(TrainStepError, match="train_parallel_config lacks"):
        trainer.train_step(batch)  # fake ranks advertise no schedule config
    assert released == ["refs"]


def test_batch_guard_is_dropped_when_the_trainer_goes_back_to_another_layout(tmp_path):
    """Review L-1: after REBUILD_OLD / restore_source the target-DP batch guard must not stay armed."""
    ranks = _trained(1)
    _trainer(ranks).save_cut(epoch=1, context=_context(tmp_path, "c"))
    new = make_world(2, seed=9)
    trainer = _trainer(new)
    trainer.restore_cut_resharded("c", epoch=1, root=str(tmp_path), expect=_expect(1), plan=_plan(1, 2),
                                  certified=[SPEC_SHA])
    assert trainer._reshard_plan is not None
    trainer.rebind_args(default_args(2))  # still the target layout: guard kept
    assert trainer._reshard_plan is not None
    trainer.rebind_args(default_args(1))  # back to the source layout (rollback)
    assert trainer._reshard_plan is None
    fresh = make_world(1, seed=3)
    back = _trainer(fresh)
    back._reshard_plan = _plan(1, 2)
    back.restore_cut("c", epoch=1, root=str(tmp_path), expect=_expect(1))  # exact same-shape restore
    assert back._reshard_plan is None


def test_restore_reports_missing_moment_keys():
    """GPU C1 diagnostic 2: a restore that leaves exp_avg/exp_avg_sq out must be refused."""
    from yeto.rl.engine.miles_adapter.cut_plugin import _slice_check

    full = {"w": {"tensors": {"param": torch.zeros(2), "exp_avg": torch.zeros(2), "exp_avg_sq": torch.zeros(2)},
                  "scalars": {"step": torch.tensor(2.0)}}}
    export = {"entries": {"w": {"start": 0, "end": 2, "tensors": {"param": torch.zeros(2)},
                                "scalars": {"step": torch.tensor(2.0)}}}}
    assert any("state keys" in p for p in _slice_check(export, full))


def _with_updaters(ranks, version):
    for r in ranks:
        r.weight_updater = SimpleNamespace(weight_version=version)
    return ranks


@pytest.mark.parametrize("src,dst", [(1, 2), (2, 1)])
def test_weight_version_continues_across_a_dp_change(tmp_path, src, dst):
    """Review L-2 / E2 f898516: a resharded trainer re-publishes from the saved weight version, not 0."""
    ranks = _with_updaters(_trained(src), 4)
    _trainer(ranks).save_cut(epoch=1, context=_context(tmp_path, "c"))
    new = _with_updaters(make_world(dst, seed=9), 0)
    _trainer(new).restore_cut_resharded("c", epoch=1, root=str(tmp_path), expect=_expect(src),
                                        plan=_plan(src, dst), certified=[SPEC_SHA])
    assert [r.weight_updater.weight_version for r in new] == [4] * dst
    # the next update_weights increments from there: no "weight version went backwards"
    new[0].weight_updater.weight_version += 1
    assert new[0].weight_updater.weight_version == 5 > 4


def test_dp_shards_must_agree_on_the_miles_counters(tmp_path):
    ranks = _trained(2)
    ranks[0].weight_updater = SimpleNamespace(weight_version=4)
    ranks[1].weight_updater = SimpleNamespace(weight_version=3)  # would diverge after the change
    _trainer(ranks).save_cut(epoch=1, context=_context(tmp_path, "c"))
    new = _with_updaters(make_world(1, seed=9), 0)
    with pytest.raises(Exception, match="miles_counters"):
        _trainer(new).restore_cut_resharded("c", epoch=1, root=str(tmp_path), expect=_expect(2),
                                            plan=_plan(2, 1), certified=[SPEC_SHA])
    assert new[0].weight_updater.weight_version == 0  # refused before any write
