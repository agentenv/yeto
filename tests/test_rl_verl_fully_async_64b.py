"""agentic-rollout-utilization 6.4b: verl fully_async adapter path (CPU, no verl/Ray).

The image-side runner is replaced by a fake trainer actor reached through the
same ``call(method, *args)`` the runner uses; the yeto driver, ports, version
map, discard and publication read-back are the real ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from yeto.rl.adapters.verl import fully_async_round as far
from yeto.rl.adapters.verl.fully_async_translate import VerlTranslationError, VersionMap
from yeto.rl.engine.algorithm import AlgorithmSpec, CorrectionSpec, ExecutionSpec
from yeto.rl.engine.policy_age import PolicyAgeError

TOLERANT = AlgorithmSpec(execution=ExecutionSpec(max_policy_staleness=1),
                         correction=CorrectionSpec(method="tis", tis_clip=2.0, tis_clip_low=0.0))


def _vmap(n: int) -> VersionMap:
    vm = VersionMap()
    for v in range(n):
        vm.record(v, v)
    return vm


def _meta(uid, mins, maxs, lengths, calls=None, rewards=None):
    nt = {"uid": [uid] * len(lengths), "min_global_steps": mins, "max_global_steps": maxs}
    if calls is not None:
        nt[far.RESUME_CALLS_KEY] = calls
    return far.sample_meta(nt, lengths, rewards)


# ------------------------------------------------------------------ overrides / support


def test_limit_drives_staleness_threshold_and_partial_rollout():
    import yeto.rl.adapters.verl.config as vconf

    run = vconf.VerlRunConfig(model_path="/m", train_file="/t", val_file="/v", out_dir="/o",
                              groups_per_round=8, samples_per_group=4)
    sync = vconf.build_overrides(run)
    out = far.fully_async_run_overrides(sync, 1, groups_per_round=8, rounds=5)
    assert "async_training.staleness_threshold=1.0" in out
    assert "async_training.partial_rollout=True" in out
    assert "async_training.trigger_parameter_sync_step=1" in out
    assert "async_training.require_batches=1" in out
    assert "rollout.n_gpus_per_node=1" in out and "trainer.n_gpus_per_node=1" in out
    # fork FullyAsyncTrainer asserts not hybrid_engine; the sync path never sets it
    assert "actor_rollout_ref.hybrid_engine=False" in out
    assert f"rollout.total_rollout_steps={8 * (2 * 5 + 1 + 1)}" in out
    assert not any(o.startswith(("trainer.use_v1=", "trainer.v1.trainer_mode=")) for o in out)
    assert sum(o.startswith("trainer.n_gpus_per_node=") for o in out) == 1
    assert "actor_rollout_ref.actor.ppo_mini_batch_size=8" in out  # require_batches * 8 = one round
    with pytest.raises(PolicyAgeError):
        far.fully_async_run_overrides(sync, 0, groups_per_round=8, rounds=5)
    # sync path at limit 0 unchanged (6.3)
    assert not any("staleness" in o for o in vconf.build_overrides(run))


def test_verl_declares_stage_two_up_to_limit_one():
    from yeto.rl.adapters.verl.entry import verl_capabilities
    from yeto.rl.adapters.verl.policy_age import SUPPORT, policy_age_overrides

    assert (SUPPORT.stage, SUPPORT.max_policy_age) == (2, 1)
    assert policy_age_overrides(0) == ()
    assert policy_age_overrides(1, groups_per_round=8) == (
        "async_training.staleness_threshold=1.0", "async_training.partial_rollout=True",
        "async_training.trigger_parameter_sync_step=1", "async_training.require_batches=1")
    with pytest.raises(PolicyAgeError, match="supports up to stage 2"):
        policy_age_overrides(2, groups_per_round=8)
    sync_caps = verl_capabilities("sha256:" + "0" * 64)
    assert sync_caps.execution.max_policy_staleness == 0 and sync_caps.placements == {"colocated"}
    fa = verl_capabilities("sha256:" + "0" * 64, fully_async_limit=1)
    assert fa.execution.max_policy_staleness == 1
    assert fa.placements == {"fixed-partition"} and fa.execution_modes == {"partitioned-serial"}


def test_island_entry_refuses_a_spec_that_does_not_tolerate_the_limit():
    from types import SimpleNamespace

    from yeto.rl.adapters.verl.island_entry import fully_async_plan

    args = SimpleNamespace(groups_per_round=4, samples_per_group=2, global_rounds=3)
    with pytest.raises(Exception, match="max_policy_staleness"):
        fully_async_plan(1, ["trainer.use_v1=True"], None, args, "none")
    overrides, keys = fully_async_plan(1, ["trainer.use_v1=True"], TOLERANT.to_dict(), args, "none")
    assert "async_training.staleness_threshold=1.0" in overrides
    assert "async_training.partial_rollout" in keys and "trainer.use_v1=True" not in overrides


def test_island_entry_parses_the_policy_age_flag():
    from yeto.rl.adapters.verl.island_entry import parse_args

    base = ["--model", "m", "--model-revision", "r", "--data", "d", "--learner-id", "0",
            "--reward-function", "gsm8k", "--global-rounds", "2", "--groups-per-round", "2",
            "--samples-per-group", "2", "--rollout-max-response-len", "8", "--event-tape", "t",
            "--inner-lr", "1e-5", "--seq-len", "8", "--seed", "1"]
    assert parse_args(base)[0].rl_max_policy_age == 0
    assert parse_args(base + ["--rl-max-policy-age", "1"])[0].rl_max_policy_age == 1


# ------------------------------------------------------------------ round collection


def test_round_keeps_within_limit_and_discards_over_age_and_unknown():
    vm = _vmap(3)
    rc = far.RoundCollector(vm, current_outer=2, limit=1, required=2)
    assert rc.offer(_meta("a", [2, 2], [2, 2], [4, 5], calls=[[(2, 4)], [(2, 5)]]))
    assert not rc.offer(_meta("old", [0, 1], [1, 1], [3, 3]))  # oldest v0: age 2 > 1
    assert not rc.offer(_meta("unk", [None, 2], [None, 2], [3, 3]))
    # resumed across a publication: 5 tokens at v1, 3 at v2
    assert rc.offer(_meta("b", [1, 2], [2, 2], [8, 2], calls=[[(1, 5), (2, 3)], [(2, 2)]],
                          rewards=[1.0, 0.0]))
    assert rc.full
    with pytest.raises(VerlTranslationError):
        rc.offer(_meta("c", [2], [2], [1]))
    fields = rc.carry_over_fields()
    assert fields["carried_in_groups"] == 1 and fields["carried_in_tokens"] == 5
    assert fields["cross_version_tokens"] == 5 and fields["trained_response_tokens"] == 19
    assert fields["over_age_discarded_groups"] == 1 and fields["over_age_discarded_tokens"] == 6
    assert fields["unknown_version_discarded_groups"] == 1
    assert fields["resumed_trajectories"] == 1 and fields["segment_token_mismatches"] == 0
    assert rc.discard_tally() == {"groups": 2, "samples": 4, "response_tokens": 12, "unknown_groups": 1}
    groups = far.group_metadata(rc, 2, {1: "h1", 2: "h2"})
    assert [g.policy_versions for g in groups] == [None, (1, 2)]
    assert [g.policy_token for g in groups] == ["yeto:2:h2", "yeto:1:h1"]
    assert groups[1].reward_mean == 0.5


def test_kept_groups_never_exceed_the_limit_when_verl_runs_one_round_ahead():
    """S19 10-10: staleness_threshold = N lets a queued sample wait N + 1 versions
    (queue_version_lag); the limit is then enforced only by yeto's per-group
    discard.  Every kept group's oldest token version is >= current - limit; every
    group older than that is discarded (F4/F5 rely on this)."""
    import random

    from yeto.rl.adapters.verl.fully_async_translate import queue_version_lag, staleness_threshold_for

    limit = 1
    lag = queue_version_lag(staleness_threshold_for(limit))
    assert lag == limit + 1
    rng = random.Random(17)
    vm = _vmap(8)
    for current in range(lag, 8):
        rc = far.RoundCollector(vm, current_outer=current, limit=limit, required=10 ** 6)
        for k in range(200):
            start = rng.randint(current - lag, current)
            if rng.random() < 0.5 and start < current:  # resumed across publications
                mid = rng.randint(start, current)
                calls = [(start, 3), (mid, 2)]
                lo, hi = start, mid
            else:
                calls, lo, hi = [(start, 4)], start, start
            n = sum(c for _, c in calls)
            rc.offer(_meta(f"g{current}-{k}", [lo, lo], [hi, hi], [n, n], calls=[calls, calls]))
        assert rc.kept and rc.discarded
        assert all(min(v.versions) >= current - limit for v in rc.kept)
        assert all(min(v.versions) < current - limit for v in rc.discarded if not v.unknown)
        assert rc.carry_over_fields()["resumed_trajectories"] > 0


def test_call_records_must_agree_with_the_span_and_length():
    vm = _vmap(3)
    mismatch = far.judge_group(_meta("x", [2], [2], [9], calls=[[(2, 4)]]), vm, 2, 1)
    assert mismatch.reason is None and mismatch.segment_mismatches == 1
    outside = far.judge_group(_meta("y", [2], [2], [4], calls=[[(1, 4)]]), vm, 2, 1)
    assert outside.unknown and "outside" in outside.reason
    unpublished = far.judge_group(_meta("z", [5], [5], [4]), vm, 2, 1)
    assert unpublished.unknown


def test_execution_profile_is_bounded_staleness_partitioned_serial():
    from yeto.rl.engine.execution_profile import check_algorithm_contract

    profile = far.execution_profile_for(TOLERANT, 1, groups_per_round=4, samples_per_group=2, sync="none")
    assert (profile.execution_mode, profile.algorithm_contract, profile.max_policy_age) == \
        ("partitioned-serial", "bounded-staleness", 1)
    check_algorithm_contract(profile, TOLERANT)


# ------------------------------------------------------------------ driver end to end

NAMES = ("base_model.model.model.layers.0.q_proj.lora_A.weight",
         "base_model.model.model.layers.0.q_proj.lora_B.weight")


class FakeTrainerActor:
    """Stands in for YetoFullyAsyncTrainer: a queue of prepared samples, a LoRA
    tensor dict, a param-version counter; the rollouter replica 'registers' what
    was pushed and writes the read-back file like vllm_readback does."""

    def __init__(self, readback_dir: Path, queue, *, tamper_at: int | None = None):
        self.push_timeouts = []
        self.readback_dir = readback_dir
        self.queue = list(queue)
        self.tensors = {NAMES[0]: torch.zeros(2, 4), NAMES[1]: torch.zeros(4, 2)}
        self.param_version = 0
        self.kept, self.last = [], None
        self.pushes, self.tamper_at = [], tamper_at
        self.calls = []

    def __call__(self, name, *args, timeout=None):
        self.calls.append(name)
        if name == "yeto_push_weights":
            self.push_timeouts.append(timeout)
            if getattr(self, "hang_push", False):
                raise TimeoutError("trainer.yeto_push_weights did not return within 600 s")
        return getattr(self, name)(*args)

    def yeto_configure(self, *_):
        return None

    def yeto_pull_sample(self):
        if not self.queue:
            return None
        self.last = self.queue.pop(0)
        return self.last

    def yeto_keep_last(self):
        self.kept.append(self.last)

    def yeto_drop_last(self):
        self.last = None

    def yeto_assemble(self):
        n, self.kept = len(self.kept), []
        return {"samples": n, "queue_size": len(self.queue)}

    def yeto_train_step(self):
        for t in self.tensors.values():
            t.add_(0.01)
        self.param_version += 1
        return {"metrics": {"actor/grad_norm": 0.5, "actor/pg_loss": 0.1, "actor/lr": 1e-5},
                "criteria": {"abs_diff": 0.01, "param_version": self.param_version},
                "param_version": self.param_version}

    def yeto_push_weights(self):
        from yeto.rl.adapters.verl import publish as pub

        version = int((self.readback_dir / "expected_version").read_text())
        seen = {k: v.clone() for k, v in self.tensors.items()}
        if self.tamper_at == version:
            seen[NAMES[0]] += 1.0
        (self.readback_dir / f"readback-v{version}.json").write_text(json.dumps(pub.lora_checksum(seen)))
        self.pushes.append((self.param_version, version))
        return {"param_version": self.param_version, "timing": {"idle_ratio": 0.0}}

    def yeto_export_lora(self):
        return {k: v.clone() for k, v in self.tensors.items()}, {"count": 2}

    def yeto_call_workers(self, fn, tensors, reset):
        for k in self.tensors:
            self.tensors[k] = tensors[k].clone().float()
        return [{"applied": len(tensors)}]


def _sample(uid, lo, hi, calls, n=2):
    lengths = [sum(c for _, c in calls)] * n
    return {"uid": uid, "min_steps": (lo,) * n, "max_steps": (hi,) * n,
            "calls": (tuple(calls),) * n, "lengths": tuple(lengths), "rewards": (1.0, 0.0)[:n]}


def _driver(tmp_path, actor, *, rounds=3, required=2):
    from yeto.rl.adapters.verl.entry import verl_capabilities
    from yeto.rl.adapters.verl.fully_async_ports import (FullyAsyncIsland, FullyAsyncPlacement,
                                                         FullyAsyncPublisher, FullyAsyncRolloutPool,
                                                         FullyAsyncTrainerGroup)
    from yeto.rl.adapters.verl.ports_impl import VerlPolicyState
    from yeto.rl.core import CanonicalTensorSpec, canonical_layout_hash
    from yeto.rl.engine.bridges import LocalOnlySync
    from yeto.rl.engine.driver import EventTape, IslandDriver

    specs = (CanonicalTensorSpec(NAMES[0], (2, 4), "float32", 8),
             CanonicalTensorSpec(NAMES[1], (4, 2), "float32", 8))
    started = []
    island = FullyAsyncIsland(actor, learner_id=0, limit=1, required=required, out_dir=str(tmp_path / "out"),
                              readback_dir=str(actor.readback_dir), start_rollouter=lambda: started.append(1),
                              base_model_revision="a" * 40, lora_config_hash="b" * 64,
                              layout_hash=canonical_layout_hash(specs), expected_specs=specs)
    tape = EventTape(tmp_path / "tape.jsonl", 0)
    island.emit = lambda event, **f: tape.append({"event": event, **f})
    profile = far.execution_profile_for(TOLERANT, 1, groups_per_round=required, samples_per_group=2,
                                        sync="none")
    driver = IslandDriver(
        learner_id=0, rollout=FullyAsyncRolloutPool(island), trainer=FullyAsyncTrainerGroup(island),
        policy_state=VerlPolicyState(island), publisher=FullyAsyncPublisher(island, timeout_s=1.0),
        placement=FullyAsyncPlacement(), capabilities=verl_capabilities("sha256:" + "0" * 64,
                                                                        fully_async_limit=1),
        algorithm=TOLERANT, sync=LocalOnlySync(rounds), events=tape, profile=profile)
    return driver, island, tape, started


def _events(tape_path):
    return [json.loads(line) for line in Path(tape_path).read_text().splitlines()]


def test_driver_runs_fully_async_rounds_with_carry_over_and_discard(tmp_path):
    rb = tmp_path / "rb"
    queue = [
        _sample("a", 0, 0, [(0, 4)]), _sample("b", 0, 0, [(0, 3)]),            # round 0
        _sample("c", 0, 1, [(0, 2), (1, 3)]), _sample("d", 1, 1, [(1, 4)]),    # round 1 (c resumed)
        _sample("e", 0, 0, [(0, 5)]),                                          # round 2: age 2, dropped
        _sample("f", 1, 2, [(1, 1), (2, 2)]), _sample("g", 2, 2, [(2, 6)]),
    ]
    actor = FakeTrainerActor(rb, queue)
    driver, island, tape, started = _driver(tmp_path, actor)
    state = driver.run()
    assert driver.rounds_completed == 3 and state.policy_version == 3
    assert started == [1]  # rollouter started once, after the first publication
    assert island.vmap.to_dict() == {"0": 0, "1": 1, "2": 2, "3": 3}
    assert actor.pushes == [(0, 0), (1, 1), (2, 2), (3, 3)]
    events = _events(tmp_path / "tape.jsonl")
    carry = [e for e in events if e["event"] == "rl_rollout_carry_over"]
    assert [e["rollout_id"] for e in carry] == [0, 1, 2]
    assert carry[1]["carried_in_groups"] == 1 and carry[1]["resumed_trajectories"] == 2
    assert carry[1]["cross_version_tokens"] == 4  # 2 tokens at v0 x 2 responses
    assert carry[2]["over_age_discarded_groups"] == 1 and carry[2]["over_age_discarded_tokens"] == 10
    cutoff = [e for e in events if e["event"] == "rl_rollout_cutoff"]
    assert [e["rollout_id"] for e in cutoff] == [2]
    trained = [e for e in events if e["event"] == "rl_round_trained"]
    assert [e["aborted_in_flight_groups"] for e in trained] == [0, 0, 1]  # never null (async10)
    checks = [e for e in events if e["event"] == "rl_publication_check"]
    assert all(e["status"] == "VERIFIED" for e in checks) and len(checks) == 4
    assert [e["param_version"] for e in events if e["event"] == "rl_verl_version_map"] == [0, 1, 2, 3]


def test_publication_mismatch_on_the_rollouter_replica_fails(tmp_path):
    from yeto.rl.engine.driver import PublicationError

    actor = FakeTrainerActor(tmp_path / "rb", [_sample("a", 0, 0, [(0, 4)])] * 2, tamper_at=1)
    driver, *_ = _driver(tmp_path, actor, rounds=2, required=1)
    with pytest.raises(PublicationError, match="v1"):
        driver.run()


def test_push_that_never_returns_ends_the_island_unverifiable(tmp_path):
    """S19 async5: the rollouter's vLLM died in the first push and the island hung
    33 min; the push now has a timeout and the publication is LORA_UNVERIFIABLE."""
    import json

    from yeto.rl.engine.driver import PublicationError

    actor = FakeTrainerActor(tmp_path / "rb", [_sample("a", 0, 0, [(0, 4)])] * 2)
    actor.hang_push = True
    driver, *_ = _driver(tmp_path, actor, rounds=2, required=1)
    with pytest.raises(PublicationError, match="UNVERIFIABLE.*push timeout"):
        driver.run()
    assert actor.push_timeouts == [600.0]
    rows = [json.loads(x) for x in (tmp_path / "out" / "verl-publish-0.jsonl").read_text().splitlines()]
    assert rows[-1]["status"].endswith("UNVERIFIABLE") and "push timeout" in rows[-1]["detail"]


def test_queue_end_and_too_many_stale_samples_stop_the_round(tmp_path):
    actor = FakeTrainerActor(tmp_path / "rb", [_sample("a", 0, 0, [(0, 4)])])
    driver, *_ = _driver(tmp_path, actor, rounds=2, required=2)
    with pytest.raises(Exception, match="ended the queue"):
        driver.run()


def test_patch_hooks_apply_to_the_pinned_fork(tmp_path):
    """Both anchors occur exactly once in the pinned fork (skipped without a checkout)."""
    import shutil

    from yeto.rl.adapters.verl import patch_verl

    fork = Path("/home/michael/work/verl-fork-acad9875")
    if not (fork / patch_verl.CALLS_TARGET).is_file():
        pytest.skip("verl fork checkout not present")
    for target, *_ in patch_verl.PATCHES:
        (tmp_path / target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fork / target, tmp_path / target)
    patch_verl.apply(tmp_path)
    text = (tmp_path / patch_verl.CALLS_TARGET).read_text()
    compile(text, "llm_server.py", "exec")
    assert text.count("yeto_resume_calls") == 1
    send = (tmp_path / patch_verl.SEND_TARGET).read_text()
    compile(send, "engine_workers.py", "exec")
    assert "_yeto_sender_stream(self)" in send and "per_tensor_param, _ = self.actor.engine" not in send
    recv = (tmp_path / patch_verl.RECV_TARGET).read_text()
    compile(recv, "base.py", "exec")
    assert recv.count("_yeto_receive_phases(weights)") == 1


def test_fully_async_hydra_entry_uses_an_absolute_config_dir(tmp_path, monkeypatch):
    """S19 GPU run s19-verl64b-async1-20261009a: verl's relative config_path failed
    when fully_async_main was imported (config/ has no __init__.py)."""
    import sys
    import types

    import importlib

    if importlib.util.find_spec("ray") is None:  # the safe test venv has no Ray; never start it
        fake_ray = types.SimpleNamespace(remote=lambda *a, **k: (lambda cls: cls))
        monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.delitem(sys.modules, "yeto.rl.adapters.verl.verl_main", raising=False)
    verl_main = importlib.import_module("yeto.rl.adapters.verl.verl_main")
    monkeypatch.delitem(sys.modules, "yeto.rl.adapters.verl.verl_main", raising=False)

    seen = {}

    def fake_main(config_path=None, config_name=None, version_base=None):
        seen.update(config_path=config_path, config_name=config_name, version_base=version_base)
        return lambda fn: fn

    monkeypatch.setitem(sys.modules, "hydra", types.SimpleNamespace(main=fake_main))

    def inner(config):
        return config

    def wrapped():
        return None

    wrapped.__wrapped__ = inner
    (tmp_path / "config").mkdir()
    fa_main = types.SimpleNamespace(__file__=str(tmp_path / "fully_async_main.py"), main=wrapped)
    assert verl_main.fully_async_hydra_entry(fa_main) is inner
    assert seen == {"config_path": str(tmp_path / "config"), "config_name": "fully_async_ppo_trainer",
                    "version_base": None}
    with pytest.raises(RuntimeError):
        verl_main.fully_async_hydra_entry(types.SimpleNamespace(__file__=fa_main.__file__, main=inner))


# Fork (acad9875) config defaults for the keys the fully_async startup asserts
# read: ppo_trainer.yaml + fully_async_ppo_trainer.yaml.
_FORK_DEFAULTS = {
    "actor_rollout_ref.hybrid_engine": True,
    "data.train_batch_size": 1024,
    "data.gen_batch_size": 1,
    "async_training": {},
    "async_training.staleness_threshold": 0.1,
    "async_training.trigger_parameter_sync_step": 4,
    "async_training.require_batches": 1,
    "reward.reward_model.enable": False,
    "reward.reward_model.enable_resource_pool": False,
    "actor_rollout_ref.rollout.calculate_log_probs": True,
    "actor_rollout_ref.rollout.mode": "async",
    "actor_rollout_ref.actor.ppo_mini_batch_size": 256,
}


def _composed(overrides):
    cfg = dict(_FORK_DEFAULTS)
    for o in overrides:
        key, _, value = o.lstrip("+").partition("=")
        cfg[key] = value
    return cfg.get


def test_fully_async_overrides_pass_every_fork_startup_assert():
    """s19-verl64b-async3-20261009a: FullyAsyncRollouter asserted
    train_batch_size == 0.  All startup asserts are checked together now."""
    import yeto.rl.adapters.verl.config as vconf

    run = vconf.VerlRunConfig(model_path="/m", train_file="/t", val_file="/v", out_dir="/o",
                              groups_per_round=32, samples_per_group=4, correction="tis")
    sync = vconf.build_overrides(run)
    out = far.fully_async_run_overrides(sync, 1, groups_per_round=32, rounds=5)
    assert far.fully_async_startup_problems(_composed(out)) == []
    assert sum(o.startswith("data.train_batch_size=") for o in out) == 1
    assert "data.train_batch_size=0" in out and "data.gen_batch_size=1" in out
    # the sync overrides alone trip the two asserts found on GPU (dbg1, async3)
    assert set(far.fully_async_startup_problems(_composed(sync))) == {
        "actor_rollout_ref.hybrid_engine is False", "data.train_batch_size == 0"}


def test_startup_problems_flag_each_broken_key():
    import yeto.rl.adapters.verl.config as vconf

    run = vconf.VerlRunConfig(model_path="/m", train_file="/t", val_file="/v", out_dir="/o",
                              groups_per_round=8, samples_per_group=4)
    good = far.fully_async_run_overrides(vconf.build_overrides(run), 1, groups_per_round=8, rounds=5)
    for bad, name in [("data.gen_batch_size=2", "data.gen_batch_size == 1"),
                      ("async_training.staleness_threshold=-1", "async_training.staleness_threshold >= 0"),
                      ("async_training.trigger_parameter_sync_step=0",
                       "async_training.trigger_parameter_sync_step >= 1"),
                      ("actor_rollout_ref.rollout.calculate_log_probs=False",
                       "actor_rollout_ref.rollout.calculate_log_probs"),
                      ("actor_rollout_ref.rollout.mode=sync", "actor_rollout_ref.rollout.mode == async"),
                      ("reward.reward_model.enable=True", "reward model off or enable_resource_pool")]:
        assert far.fully_async_startup_problems(_composed([*good, bad])) == [name], bad


_FORK = Path("/home/michael/work/verl-fork-acad9875/verl/experimental/fully_async_policy")


@pytest.mark.skipif(not _FORK.is_dir(), reason="local verl fork checkout not present")
def test_mirror_covers_every_config_assert_in_the_fork_startup_path():
    lines = []
    for name in ("fully_async_rollouter.py", "fully_async_trainer.py"):
        for line in (_FORK / name).read_text().splitlines():
            t = line.strip()
            if t.startswith("assert") and ("self.config" in t or "hybrid_engine" in t) \
                    and "resume_from_path" not in t:
                lines.append(t)
    uncovered = [t for t in lines if not t.startswith(far.FORK_STARTUP_ASSERTS)]
    assert lines and uncovered == []
