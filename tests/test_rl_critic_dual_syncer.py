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
        # prepare refuses an unverified critic on two islands before G3 (check_unverified_allowance);
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
from yeto.rl.engine.miles_adapter import state_plugin as sp  # noqa: E402
from yeto.rl.engine.miles_adapter.trainer import MilesTrainerGroup  # noqa: E402

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
    # bf16 parameters cannot hold an arbitrary fp32 average: the post-write hash check refuses
    bf16 = _critic_trainer([_rank_actor(0, torch.bfloat16)])
    odd = {k: v + 1e-4 for k, v in bf16.export_critic_state().items()}
    with pytest.raises(cs.CriticStateError, match="written critic hash"):
        bf16.import_critic_state(odd)
