"""decoupling 2.7: backend behaviour the core branches on is declared by the adapter.

Covers the new declaration fields (BackendTraits.publish_while_offloaded), the
adapter-supplied ledger abort mechanism and the cut's ``backend_state``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from yeto.rl.engine.algorithm import AlgorithmSpec
from yeto.rl.engine.bridges import LocalOnlySync
from yeto.rl.engine.capabilities import BackendTraits
from yeto.rl.engine.cut import AlgorithmIdentity, CutFile, CutManifest, CutProgress
from yeto.rl.engine.driver import EventTape, IslandDriver
from yeto.rl.engine.fake import FakeEngine, fake_capabilities
from yeto.rl.engine.ledger import BatchLedger
from yeto.rl.engine.miles_adapter.entry import miles_capabilities
from yeto.rl.engine.miles_adapter.traits import MILES_ABORT_MECHANISM, MILES_TRAITS
from yeto.rl.engine.ports import GroupMetadata, RolloutBatchHandle

NAME = "base_model.model.layer.lora_A.weight"


def _run(tmp_path, capabilities):
    engine = FakeEngine(tensors={NAME: torch.zeros(1, 2)}, step_delta=1.0)
    engine.trainer.publish_offloaded = True  # this run would allow it (Miles: --offload-train)
    driver = IslandDriver(
        learner_id=0, rollout=engine.rollout, trainer=engine.trainer, policy_state=engine.policy_state,
        publisher=engine.publisher, placement=engine.placement, algorithm=AlgorithmSpec(),
        sync=LocalOnlySync(1), events=EventTape(tmp_path / "events.jsonl", 0), capabilities=capabilities)
    driver.run()
    return [c for c in engine.calls if c[0] != "export"]


def test_publish_while_offloaded_is_a_declaration():
    assert MILES_TRAITS.publish_while_offloaded is True
    assert miles_capabilities("sha256:" + "1" * 64).traits.publish_while_offloaded is True
    assert BackendTraits().publish_while_offloaded is False


def test_driver_offloads_before_publish_only_when_the_backend_declares_it(tmp_path):
    declared = _run(tmp_path / "a", fake_capabilities())
    assert declared[-2:] == [("offload",), ("publish", 1)]
    undeclared = _run(tmp_path / "b", fake_capabilities(traits=BackendTraits()))
    assert ("offload",) not in undeclared[undeclared.index(("train", 0)):]


def _batch(**kw):
    return RolloutBatchHandle(
        rollout_id=0, policy_version=0, policy_hash="h",
        groups=(GroupMetadata(group_id="g0", sample_ids=("s0",), policy_token="t", reward_mean=0.0,
                              reward_std=0.0, token_count=1),),
        completed=1, aborted=0, aborted_in_flight_groups=2, **kw)


def _discarded(tmp_path, batch):
    led = BatchLedger(tmp_path)
    led.prepare(batch, policy_token="t")
    from yeto.rl.engine.journal import read_journal

    [rec] = [r for r in read_journal(tmp_path / "ledger") if r["kind"] == "engine_discarded"]
    return rec["mechanism"]


def test_ledger_abort_mechanism_comes_from_the_adapter(tmp_path):
    assert _discarded(tmp_path / "m", _batch(abort_mechanism=MILES_ABORT_MECHANISM)) == \
        "miles generate_rollout abort"
    assert _discarded(tmp_path / "n", _batch()) == "engine abort (mechanism not reported)"


def _manifest(**kw):
    return CutManifest(
        cut_id="c1", epoch=1, runtime={"backend_fingerprint": "fp", "layout": {"dp": 1}, "rng_policy": "exact"},
        progress=CutProgress(local_step=1, scheduler_samples=4, global_batch_size=4, next_rollout_id=1,
                             policy_version=1, policy_hash="p"),
        algorithm=AlgorithmIdentity("a" * 64), data={}, ledger={}, outer={"settled": True},
        files=(CutFile(path="r0.pt", sha256="0" * 64, bytes=1),), **kw)


def test_cut_backend_state_is_written_only_when_present():
    plain = _manifest()
    assert "backend_state" not in json.loads(plain.to_json())
    assert CutManifest.from_json(plain.to_json()).to_json() == plain.to_json()
    private = _manifest(backend_state={"scheduler": {"kind": "cosine"}})
    raw = json.loads(private.to_json())
    assert raw["backend_state"] == {"scheduler": {"kind": "cosine"}}
    assert CutManifest.from_json(private.to_json()).backend_state == {"scheduler": {"kind": "cosine"}}
    assert private.digest() != plain.digest()
