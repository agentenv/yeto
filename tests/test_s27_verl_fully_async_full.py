"""S27 G-EW: full-parameter verl trainers on the fully_async ports (VERL-PARITY #14, #11).

CPU only. The fully_async trainer actor is a fake that answers ``yeto_call_workers``
and ``yeto_call_workers_shared`` over one in-process verl actor worker (the real
worker functions of ``yeto.rl.adapters.verl.full_parameter``). These tests check
the shared piece (:mod:`yeto.rl.adapters.verl.fully_async_full`):

* the identity, sync and policy state are the synchronous full-parameter island's
  (``full_parameter_stream.full_parameter_policy``), read through the trainer actor;
* the trainer port records each trained round on the full-parameter group;
* a whole-tensor apply goes through the actor's shared-argument call;
* what these paths do not wire is refused, and the island entry refuses the
  fully_async path and the buffer trainer on separate lines.
"""
from __future__ import annotations

import inspect

import pytest
import torch

from test_s27_verl_full_parameter_sync import SHAPES, _Island, _Model
from yeto.rl.adapters.verl import full_parameter as fp
from yeto.rl.adapters.verl import full_parameter_stream as vfs
from yeto.rl.adapters.verl import fully_async_full, island_entry
from yeto.rl.adapters.verl.fully_async_ports import FullyAsyncIsland, FullyAsyncTrainerGroup


class _TrainerActor:
    """The fully_async trainer actor's worker hooks over one worker (``call(name, *args)``)."""

    def __init__(self, model: _Model):
        self.model = model
        self.worker = _Island(model).worker
        self.calls: list[tuple[str, str]] = []
        self.param_version = 0

    def __call__(self, name, *args, timeout=None):
        if name in ("yeto_call_workers", "yeto_call_workers_shared"):
            fn, *rest = args
            self.calls.append((name, fn.__name__))
            return [fn(self.worker, *rest)]
        return getattr(self, name)(*args)

    def yeto_train_step(self):
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.add_(0.125)
        self.param_version += 1
        return {"metrics": {"actor/grad_norm": 0.5, "actor/pg_loss": 0.1, "actor/lr": 1e-6},
                "criteria": {}, "param_version": self.param_version}


def _plan(**extra) -> dict:
    return {"learner_id": 0, "sync": "none", "parameter_mode": "full", "model_revision": "A" * 40,
            "model_config_sha256": "b" * 64, "global_rounds": 3, **extra}


def _island(tmp_path, actor, plan):
    layout_hash, identity = fully_async_full.full_parameter_hashes(plan)
    return FullyAsyncIsland(actor, learner_id=0, limit=1, required=1, out_dir=str(tmp_path / "out"),
                            readback_dir=str(tmp_path / "rb"), base_model_revision=plan["model_revision"],
                            lora_config_hash=identity, layout_hash=layout_hash)


def test_the_identity_is_the_synchronous_full_parameter_islands():
    layout, identity = fully_async_full.full_parameter_hashes(_plan())
    assert layout is None  # no syncer: the local state plans the layout
    assert identity == vfs.full_parameter_identity("b" * 64)[1]
    synced = _plan(sync="strict", **{vfs.PLAN_KEY: {"parameter_layout_sha256": "e" * 64}})
    assert fully_async_full.full_parameter_hashes(synced) == ("e" * 64, identity)


def test_a_no_sync_full_parameter_trainer_reads_its_workers_through_the_trainer_actor(tmp_path):
    from yeto.rl.engine.bridges import LocalOnlySync

    actor = _TrainerActor(_Model(seed=0))
    plan = _plan()
    island = _island(tmp_path, actor, plan)
    assert island.full_group is None and island.layout_hash is None
    sync, state = fully_async_full.full_parameter_ports(island, plan, backend_identity_sha256="c" * 64)
    # the synchronous island's pieces, unchanged (trainer._full_parameter_ports)
    assert isinstance(sync, LocalOnlySync) and isinstance(state, vfs.VerlFullParameterLocalState)
    assert isinstance(island.full_group, vfs.VerlFullParameterGroup) and island.full_group.island is island
    assert state.max_fragment_bytes == vfs.DEFAULT_MAX_FRAGMENT_BYTES
    assert island.layout_hash == state.layout_hash and len(island.layout_hash) == 64
    assert ("yeto_call_workers", "remote_manifest") in actor.calls
    first = state.export(policy_version=0)
    assert first.digest.payload_bytes == 4 * sum(torch.Size(s).numel() for s in SHAPES.values())

    # the trainer port binds each trained round to the group the stream reads
    from yeto.rl.engine.ports import GroupMetadata, RolloutBatchHandle

    island.published = (0, first.policy_tensor_hash())
    group = GroupMetadata("g0", ("s0", "s1"), "token", 0.5, 0.5, 64)
    batch = RolloutBatchHandle(0, 0, first.policy_tensor_hash(), (group,), 1, 0, payload=None)
    receipt = FullyAsyncTrainerGroup(island).train_step(batch)
    assert island.full_group.trained_rounds == 1 and island.full_group.receipts[0] is receipt
    evidence = island.full_group.round_evidence(0)
    assert (evidence.trajectory_ids, evidence.trained_tokens) == (("s0", "s1"), 64)
    assert receipt.parameter_layout_hash == island.layout_hash
    assert state.export(policy_version=1).policy_tensor_hash() != first.policy_tensor_hash()


def test_a_whole_tensor_apply_goes_through_the_actors_shared_call(tmp_path):
    import asyncio

    actor = _TrainerActor(_Model(seed=0))
    island = _island(tmp_path, actor, _plan())
    group = vfs.VerlFullParameterGroup(island)
    target = {"model.norm.weight": torch.full(SHAPES["model.norm.weight"], 3.0)}
    asyncio.run(group.apply_parameters(target))
    assert ("yeto_call_workers_shared", "remote_apply_fragment") in actor.calls
    assert torch.equal(actor.model.model.norm.weight.detach(), target["model.norm.weight"])


@pytest.mark.parametrize("key,value,match", [
    ("resume", {"store": "/s"}, "round cuts"),
    ("hf_exports", {"rounds": "4", "store": "/x"}, "HF exports"),
    ("data_position_store", "/d", "--rl-data-position-store"),
    ("publish_selftest", True, "publish self-test"),
])
def test_what_the_fully_async_paths_do_not_wire_is_refused(tmp_path, key, value, match):
    actor = _TrainerActor(_Model(seed=0))
    plan = _plan(**{key: value})
    with pytest.raises(ValueError, match=match):
        fully_async_full.full_parameter_ports(_island(tmp_path, actor, plan), plan,
                                              backend_identity_sha256="c" * 64)
    assert actor.calls == []  # refused before any worker call


def test_a_synced_plan_needs_the_full_parameter_section(tmp_path):
    actor = _TrainerActor(_Model(seed=0))
    plan = _plan(sync="strict")
    with pytest.raises(ValueError, match="full-parameter sync plan"):
        fully_async_full.full_parameter_ports(_island(tmp_path, actor, plan), plan,
                                              backend_identity_sha256="c" * 64)


def test_the_island_entry_refuses_each_unwired_path_on_its_own_line():
    source = inspect.getsource(island_entry.main)
    # S27 G-EV (VERL-PARITY #11): the fully_async path is wired; only its colocated form is refused
    assert "全参数 fully_async 路径未接" not in source
    assert "if full and limit and buffer_section is None and pools is None:" in source
    assert "if full and args.rl_rollout_quant_tier:" in source
    # the buffer trainer's line is gone: VERL-PARITY #14 (test_s27_verl_sample_role.py)


def test_the_sync_path_takes_its_policy_from_the_shared_function():
    from pathlib import Path

    source = Path(fp.__file__).with_name("trainer.py").read_text()
    sync_part, async_part = source.split("def run_fully_async_island", 1)
    ports = sync_part.split("def _full_parameter_ports", 1)[1].split("\ndef ", 1)[0]
    assert "full_parameter_policy(island, plan, backend_identity_sha256=identity_sha256)" in ports
    assert "full_parameter_ports(island, plan, backend_identity_sha256=identity_sha)" in async_part
    assert "policy_state=policy_state" in async_part
