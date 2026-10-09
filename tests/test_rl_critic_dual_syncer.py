"""rl-algo-critic-family 4.2/4.3 (design D4 plan a): dual syncer, critic tensor
plugin, cross-channel atomic commit, round-cut critic checkpoint (CPU only)."""

from __future__ import annotations

import json

import pytest
import torch

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from yeto.rl.engine.algorithm import AlgorithmSpec

PPO = AlgorithmSpec(advantage_estimator="ppo")


# -- 4.2.1 launcher: second syncer, own port / checkpoint / tape --------------------


def _two_island_args(spec=None):
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", gpu="aws:1xa100@us-east-1,aws:1xa100@us-west-2")
    args.controller = "local"
    _prepare_rl_args(args)
    if spec is not None:
        # prepare refuses an undeclared critic without --rl-allow-unverified-mechanism execution:critic;
        # the launcher plumbing is exercised on the prepared args directly.
        args.rl_algorithm_spec_json = spec.canonical_json()
        args.rl_expected_algorithm_sha256 = spec.sha256()
    return args


def test_dry_run_ppo_has_two_syncers():
    from yeto import launcher

    args = _two_island_args(PPO)
    plan = launcher.dry_run_plan(args)
    critic = plan["critic_syncer"]
    assert critic["port"] == launcher.CRITIC_SYNCER_PORT != launcher.SYNCER_PORT
    assert critic["layout"] == "critic_layout_hash"
    command = critic["command"]
    assert f"--port {launcher.CRITIC_SYNCER_PORT}" in command
    assert f"--port {launcher.SYNCER_PORT}" not in command
    assert "yeto-critic-state.ckpt" in command and "yeto-critic-tape.jsonl" in command
    assert " --learners 2 " in command
    for island in plan["island_requests"]:
        assert "--syncer $SYNCER_ADDR --critic-syncer $CRITIC_SYNCER_ADDR" in island["learner_command"]
    full = launcher.syncer_command(args, 2)
    # critic syncer backgrounded first, actor syncer stays the foreground process
    assert full.index(f"--port {launcher.CRITIC_SYNCER_PORT}") < full.index(f"--port {launcher.SYNCER_PORT}")
    assert full.count("~/yeto-output/yeto-state.ckpt") == 2  # actor: path + resume test
    assert launcher.syncer_ports(args) == [launcher.SYNCER_PORT, launcher.CRITIC_SYNCER_PORT]
    snapshot = {k: plan[k] for k in ("islands", "syncer", "outer_sync", "critic_syncer")}
    assert json.loads(json.dumps(snapshot)) == snapshot


def test_critic_free_plan_and_syncer_unchanged():
    from yeto import launcher

    args = _two_island_args()
    plan = launcher.dry_run_plan(args)
    assert "critic_syncer" not in plan
    assert all("--critic-syncer" not in i["learner_command"] for i in plan["island_requests"])
    command = launcher.syncer_command(args, 2)
    assert str(launcher.CRITIC_SYNCER_PORT) not in command and "critic" not in command
    assert launcher.syncer_ports(args) == [launcher.SYNCER_PORT]


def test_island_env_carries_the_critic_syncer_address():
    from yeto import launcher
    from yeto.gpu_spec import parse_gpu_spec

    args = _two_island_args(PPO)
    task = launcher.make_miles_island_task(args, parse_gpu_spec(args.gpu)[0], 0, 2, "10.0.0.5:29400")
    assert task.envs["CRITIC_SYNCER_ADDR"] == f"10.0.0.5:{launcher.CRITIC_SYNCER_PORT}"
    assert launcher.critic_syncer_address("$SYNCER_ADDR") == "$CRITIC_SYNCER_ADDR"



# -- 4.2.2 critic full-parameter export / write-back plugin (CPU fake ranks) ---------


import asyncio  # noqa: E402
import importlib  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from yeto.rl import critic_state as cs  # noqa: E402
from yeto.rl.adapters.miles import state_plugin as sp  # noqa: E402
from yeto.rl.adapters.miles.trainer import MilesTrainerGroup  # noqa: E402

H = "a" * 64


class _Sched:
    def __init__(self):
        self.num_steps = 0

    def state_dict(self):
        return {"num_steps": self.num_steps}

    def load_state_dict(self, state):
        self.num_steps = state["num_steps"]


def _rank_actor(seed, dtype=torch.float32):
    torch.manual_seed(seed)
    body = torch.nn.Module()
    body.embedding = torch.nn.Linear(4, 3)
    body.output_layer = torch.nn.Linear(3, 1, bias=False)  # value head [1, hidden]
    body = body.to(dtype)
    body.embedding.bias.requires_grad_(False)  # frozen tensors are not critic state
    opt = torch.optim.Adam([p for p in body.parameters() if p.requires_grad], lr=1e-3)
    return SimpleNamespace(model=[body], optimizer=opt, opt_param_scheduler=_Sched())


class _CriticRanks:
    """A fake Miles critic TrainGroup: run_plugin calls the plugin on every rank."""

    def __init__(self, ranks):
        self.ranks = ranks

    async def run_plugin(self, fn_path, kwargs=None):
        module, _, name = fn_path.rpartition(".")
        fn = getattr(importlib.import_module(module), name)
        out = []
        saved = sp._rank
        try:
            for rank, actor in enumerate(self.ranks):
                sp._rank = lambda r=rank: r
                out.append(fn(actor, **(kwargs or {})))
        finally:
            sp._rank = saved
        return out


def _critic_trainer(ranks):
    return MilesTrainerGroup(
        args=SimpleNamespace(num_steps_per_rollout=1), actor_model=SimpleNamespace(),
        learner_id=0, learner_generation=0, parameter_layout_hash=lambda: H,
        algorithm="ppo", spec=PPO, critic_model=_CriticRanks(ranks))


def _step(actor):
    params = [p for p in actor.model[0].parameters() if p.requires_grad]
    loss = sum((p.float() ** 2).sum() for p in params)
    loss.backward()
    actor.optimizer.step()
    actor.optimizer.zero_grad()
    actor.opt_param_scheduler.num_steps += 8


def test_critic_tensors_round_trip_through_fragments():
    ranks = [_rank_actor(0), _rank_actor(1)]
    trainer = _critic_trainer(ranks)
    state = trainer.export_critic_state()
    assert sorted(state) == ["r0:0:embedding.weight", "r0:0:output_layer.weight",
                             "r1:0:embedding.weight", "r1:0:output_layer.weight"]
    layout = trainer.critic_layout()
    before = cs.critic_weights_sha256(state)
    fragments = cs.critic_fragments(state, layout_sha256=layout, max_fragment_bytes=48)
    assert len(fragments) > 1  # 12-float weights cut by the 48-byte budget
    specs = [(k, list(v.shape), str(v.dtype)) for k, v in state.items()]
    back = cs.assemble_critic_fragments(fragments, layout_sha256=layout, expected_specs=specs)
    assert cs.critic_weights_sha256(back) == before
    # write a different (averaged) state back, then re-export: same hash
    other = _critic_trainer([_rank_actor(5), _rank_actor(6)]).export_critic_state()
    avg = {k: (state[k] + other[k]) / 2 for k in state}
    assert trainer.import_critic_state(avg) == cs.critic_weights_sha256(avg)
    assert cs.critic_weights_sha256(trainer.export_critic_state()) == cs.critic_weights_sha256(avg)


def test_fragment_refusals():
    state = _critic_trainer([_rank_actor(0)]).export_critic_state()
    fragments = cs.critic_fragments(state, layout_sha256=H, max_fragment_bytes=1 << 20)
    with pytest.raises(cs.CriticLayoutMismatch):
        cs.assemble_critic_fragments(fragments, layout_sha256="b" * 64)
    bad = [cs.CriticFragment(f.index, f.layout_sha256, f.names,
                             {n: t + 1 for n, t in f.tensors.items()}, f.sha256) for f in fragments]
    with pytest.raises(cs.CriticStateError, match="content hash"):
        cs.assemble_critic_fragments(bad, layout_sha256=H)
    with pytest.raises(cs.CriticStateError, match="complete"):
        cs.assemble_critic_fragments(fragments + fragments, layout_sha256=H)
    with pytest.raises(cs.CriticLayoutMismatch, match="specs"):
        cs.assemble_critic_fragments(fragments, layout_sha256=H, expected_specs=[("x", [1], "torch.float32")])


def test_write_back_refusals():
    trainer = _critic_trainer([_rank_actor(0)])
    state = trainer.export_critic_state()
    with pytest.raises(cs.CriticStateError, match="names differ"):
        trainer.import_critic_state({"r0:0:extra": torch.zeros(1), **state})
    with pytest.raises(cs.CriticStateError, match="shape"):
        trainer.import_critic_state({k: torch.zeros(2, 2) for k in state})
    # the channel is over FP32 optimizer masters (test_rl_critic_fp32_masters.py); a bf16
    # critic whose optimizer holds no FP32 master is refused instead of written lossily
    bf16 = _critic_trainer([_rank_actor(0, torch.bfloat16)])
    with pytest.raises(sp.StatePluginError, match="no FP32 optimizer master"):
        bf16.export_critic_state()


# -- 4.3 round-cut critic checkpoint (weights + optimizer + scheduler) ------------------


def _opt_digest(actor):
    import hashlib

    state = actor.optimizer.state_dict()["state"]
    d = hashlib.sha256()
    for k in sorted(state):
        for name in sorted(state[k]):
            v = state[k][name]
            d.update(torch.as_tensor(v).float().numpy().tobytes())
    return d.hexdigest()


def _manifest(round_id, pointer):
    return SimpleNamespace(runtime={} if pointer is None else {"critic": pointer},
                           progress=SimpleNamespace(policy_version=round_id))


def test_critic_cut_save_restore_hash_equal(tmp_path):
    from yeto.rl.engine.cut import CutError

    ranks = [_rank_actor(0), _rank_actor(1)]
    for actor in ranks:
        _step(actor)
    trainer = _critic_trainer(ranks)
    saved_w = cs.critic_weights_sha256(trainer.export_critic_state())
    saved_o = [_opt_digest(a) for a in ranks]
    pointer = trainer._save_critic_cut(tmp_path / "critic", 3)
    assert pointer["round"] == 3 and [r["rank"] for r in pointer["ranks"]] == [0, 1]
    json.dumps(pointer)  # goes into the cut manifest runtime
    # a fresh trainer (new weights, one more step, scheduler elsewhere) restores the cut
    fresh = [_rank_actor(7), _rank_actor(8)]
    for actor in fresh:
        _step(actor)
        _step(actor)
    restored = _critic_trainer(fresh)
    restored._restore_critic_cut(tmp_path / "critic", _manifest(3, pointer))
    assert cs.critic_weights_sha256(restored.export_critic_state()) == saved_w
    assert [_opt_digest(a) for a in fresh] == saved_o
    assert [a.opt_param_scheduler.num_steps for a in fresh] == [8, 8]
    # round inconsistency: actor round 4 vs critic round 3 is refused
    with pytest.raises(CutError, match="actor round 4 != critic round 3"):
        restored._restore_critic_cut(tmp_path / "critic", _manifest(4, pointer))
    # the store itself also refuses a critic round != actor round (pointer tampered)
    with pytest.raises(CutError, match="critic restore refused"):
        restored._restore_critic_cut(tmp_path / "critic", _manifest(4, {**pointer, "round": 4}))
    with pytest.raises(CutError, match="carries no critic state"):
        restored._restore_critic_cut(tmp_path / "critic", _manifest(3, None))
    grpo = MilesTrainerGroup(args=SimpleNamespace(num_steps_per_rollout=1), actor_model=SimpleNamespace(),
                             learner_id=0, learner_generation=0, parameter_layout_hash=lambda: H)
    assert grpo._save_critic_cut(tmp_path / "x", 1) is None
    with pytest.raises(CutError, match="has no critic"):
        grpo._restore_critic_cut(tmp_path / "critic", _manifest(3, pointer))


def test_critic_cut_tampered_weights_refused(tmp_path):
    from yeto.rl.engine.cut import CutError

    trainer = _critic_trainer([_rank_actor(0)])
    pointer = trainer._save_critic_cut(tmp_path / "critic", 2)
    weights = tmp_path / "critic" / "rank-0" / "critic" / "round-2" / "weights.pt"
    data = torch.load(weights, weights_only=True)
    torch.save({k: v + 1 for k, v in data.items()}, weights)
    with pytest.raises(CutError, match="fails its manifest"):
        _critic_trainer([_rank_actor(1)])._restore_critic_cut(tmp_path / "critic", _manifest(2, pointer))


def test_critic_store_refuses_round_mismatch_on_the_rank(tmp_path, monkeypatch):
    actor = _rank_actor(0)
    _critic_trainer([actor])._save_critic_cut(tmp_path, 5)
    monkeypatch.setattr(sp, "_rank", lambda: 0)
    out = sp.restore_critic_cut(actor, directory=str(tmp_path), actor_round=6, critic_round=5)
    assert "actor round 6 != critic round 5" in out["refused"]


# -- 4.2.3 cross-channel atomic commit: fake two islands, two fake syncers -------------


from yeto.rl.core import build_avg_layout  # noqa: E402
from yeto.rl.engine.bridges import (  # noqa: E402
    CrossChannelCommitError,
    DualStrictAvgSync,
    _CriticDriverView,
)
from yeto.rl.engine.driver import EventTape, IslandDriver  # noqa: E402
from yeto.rl.engine.fake import FakeEngine, FakeStrictSyncer, fake_capabilities  # noqa: E402

ALLOW = ("advantage_estimators:ppo", "execution:critic")
NAME = "base_model.model.layer.lora_A.weight"


def _ppo_engine(actor, critic):
    return FakeEngine(tensors={NAME: torch.tensor([actor])}, critic=True,
                      critic_tensors={"backbone.weight": torch.tensor([[critic, 1.0]]),
                                      "output_layer.weight": torch.tensor([[critic, 2.0]])})


def _critic_syncer(engine, learners, rounds):
    view = _CriticDriverView(SimpleNamespace(trainer=engine.trainer),
                             ("0" * 40, engine.trainer.critic_layout()))
    return FakeStrictSyncer(build_avg_layout(view.critic_state(0).specs),
                            learners=learners, total_steps=rounds)


def _dual_islands(tmp_path, rounds, *, keep_committed=False):
    import test_rl_engine_driver as td

    engines = [_ppo_engine([1.0, 3.0], 1.0), _ppo_engine([3.0, 5.0], 5.0)]
    actor_syncer = td._strict_syncer(engines[0], learners=2, rounds=rounds)
    critic_syncer = _critic_syncer(engines[0], 2, rounds)
    drivers, syncs = [], []
    for island, engine in enumerate(engines):
        config = td._strict_config(tmp_path, engine, learner_id=island, rounds=rounds,
                                   tape=f"bridge-{island}.jsonl")
        sync = DualStrictAvgSync(
            config, critic_syncer_addr=("127.0.0.1", 29401), keep_committed=keep_committed,
            client_factory=lambda _b, i=island: actor_syncer.client(i),
            critic_client_factory=lambda _b, i=island: critic_syncer.client(i))
        syncs.append(sync)
        drivers.append(IslandDriver(
            learner_id=island, rollout=engine.rollout, trainer=engine.trainer,
            policy_state=engine.policy_state, publisher=engine.publisher,
            placement=engine.placement, algorithm=PPO, sync=sync,
            events=EventTape(tmp_path / f"i{island}.jsonl", island),
            capabilities=fake_capabilities().with_unverified(ALLOW)))
    return engines, drivers, syncs


def test_two_islands_actor_and_critic_hashes_agree(tmp_path):
    import test_rl_engine_driver as td

    engines, drivers, _ = _dual_islands(tmp_path, rounds=2)
    _, errors = td._run_threads(drivers)
    assert errors == {}
    actor = [cs.critic_weights_sha256(e.policy_state.export().to_lora().tensors) for e in engines]
    critic = [cs.critic_weights_sha256(e.critic_tensors) for e in engines]
    assert actor[0] == actor[1] and critic[0] == critic[1]
    # islands started apart (1.0 / 5.0); the critic syncer's initial policy is island 0's,
    # then each round trains the critic +0.5 before the average: 1.0 + 2 x 0.5
    assert engines[0].critic_tensors["output_layer.weight"][0, 0].item() == pytest.approx(2.0)
    for e in engines:
        applies = [c for c in e.calls if c[0] in ("apply", "critic_apply")]
        # initial + 2 committed rounds, critic applied before the actor each time
        assert [c[0] for c in applies] == ["critic_apply", "apply"] * 3
    # G3 evidence: per committed version, both islands record the same applied critic
    import json

    tapes = [[json.loads(line) for line in (tmp_path / f"i{i}.jsonl").read_text().splitlines()]
             for i in range(2)]
    per = [{e["policy_version"]: (e["sync/global_critic_hash"], e["rl/critic/applied_weights_sha256"])
            for e in t if e.get("event") == "rl_critic_apply"} for t in tapes]
    assert sorted(per[0]) == [0, 1, 2] and per[0] == per[1]
    assert per[0][2][1] == critic[0]


def test_modal_island_config_swaps_the_critic_syncer_address():
    from yeto import launcher
    from yeto.gpu_spec import parse_gpu_spec

    args = _two_island_args(PPO)
    spec = parse_gpu_spec(args.gpu)[0]
    task = launcher.make_miles_island_task(args, spec, 0, 2, "10.0.0.5:29400")
    task.envs = {**task.envs, "CRITIC_SYNCER_ADDR": launcher.critic_syncer_address("10.0.0.5:29400")}
    args.rl_image = "ghcr.io/x/miles@sha256:" + "a" * 64
    cfg = launcher.build_modal_island_config(args, spec, 0, task, "203.0.113.7:29400")
    assert cfg.envs["SYNCER_ADDR"] == "203.0.113.7:29400"
    assert cfg.envs["CRITIC_SYNCER_ADDR"] == f"203.0.113.7:{launcher.CRITIC_SYNCER_PORT}"


def test_critic_channel_failure_rolls_back_both_roles(tmp_path):
    import test_rl_engine_driver as td

    engines, drivers, syncs = _dual_islands(tmp_path, rounds=2, keep_committed=True)
    committed = {}
    for i, sync in enumerate(syncs):
        original = sync.start

        def start(driver, _orig=original, _i=i, _s=sync):
            out = _orig(driver)
            committed[_i] = (cs.critic_weights_sha256(_s.committed[1].to_lora().tensors),
                             cs.critic_weights_sha256(_s.committed[2]))

            def fail(*_a, **_k):
                raise TimeoutError("critic syncer did not deliver v+1")
            _s.critic._await = fail
            return out
        sync.start = start
    _, errors = td._run_threads(drivers)
    assert set(errors) == {0, 1}
    for i, e in enumerate(engines):
        error = errors[i]
        assert isinstance(error, CrossChannelCommitError) and isinstance(error.__cause__, TimeoutError)
        assert "neither actor nor critic applied" in str(error)
        # neither role applied round 1: both are back at the committed round 0 content
        assert cs.critic_weights_sha256(e.policy_state.export().to_lora().tensors) == committed[i][0]
        assert cs.critic_weights_sha256(e.critic_tensors) == committed[i][1]
        versions = [c[1] for c in e.calls if c[0] == "apply"]
        assert 1 not in versions  # the actor's v+1 from its own channel was never applied


def test_build_sync_selects_the_dual_channel_for_a_critic():
    from yeto.rl.adapters.miles.entry import build_sync

    base = dict(yeto_rl_bridge_config=SimpleNamespace(), yeto_rl_sync_preset="strict-avg",
                yeto_rl_completed_groups_path="/tmp/s13/none.pt")
    sync, _ = build_sync(SimpleNamespace(**base, use_critic=True,
                                         yeto_rl_critic_syncer_addr=("h", 29401)), yeto_policy_sync=True)
    assert isinstance(sync, DualStrictAvgSync) and sync.OUTER_SYNC_KIND == "strict"
    with pytest.raises(ValueError, match="--critic-syncer"):
        build_sync(SimpleNamespace(**base, use_critic=True), yeto_policy_sync=True)
