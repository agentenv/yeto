"""CPU tests for miles_adapter.trainer (3.3) and miles_adapter.publish (3.5)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from yeto.rl.contracts import InferencePublicationManifest, LocalStepReceipt
from yeto.rl.engine.miles_adapter.publish import MilesPublisher, PublicationError
from yeto.rl.engine.miles_adapter.rollout import PolicyTokenMismatch, policy_token
from yeto.rl.engine.miles_adapter.state_plugin import APPLIED_LRS, GRAD_NORM
from yeto.rl.engine.miles_adapter.trainer import MilesTrainerGroup, TrainStepError, batch_hash
from yeto.rl.engine.ports import GroupMetadata, Publisher, RolloutBatchHandle, TrainerGroup

H = "a" * 64
L = "b" * 64
TOKEN = policy_token(3, H)


def handle(token=TOKEN, payload="PACK"):
    return RolloutBatchHandle(
        rollout_id=3, policy_version=3, policy_hash=H,
        groups=(
            GroupMetadata("g0", ("s0", "s1"), token, 0.5, 0.5, 10),
            GroupMetadata("g1", ("s10", "s11"), TOKEN, 1.0, 0.0, 7),
        ),
        completed=2, aborted=0, payload=payload,
    )


class FakeActorGroup:
    def __init__(self, outcome="NORMAL", norm=0.3, outputs=1, lrs=((1e-5,), (1e-5,))):
        self.outcome, self.norm, self.outputs = outcome, norm, outputs
        self.lrs = lrs
        self.calls = []

    async def train(self, rollout_id, pack):
        self.calls.append(("train", rollout_id, pack))
        return [SimpleNamespace(outcome=SimpleNamespace(name=self.outcome))] * self.outputs

    async def run_plugin(self, fn_path, kwargs=None):
        self.calls.append(("plugin", fn_path))
        if fn_path == APPLIED_LRS:
            return [list(v) for v in self.lrs]
        return [self.norm, self.norm]

    async def onload(self):
        self.calls.append(("onload",))

    async def offload(self):
        self.calls.append(("offload",))

    async def clear_memory(self):
        self.calls.append(("clear",))


def trainer(actor, released, offload=True):
    return MilesTrainerGroup(
        args=SimpleNamespace(num_steps_per_rollout=1, offload_train=offload),
        actor_model=actor, learner_id=0, learner_generation=0,
        parameter_layout_hash=lambda: L,
        release_refs=lambda args, pack: released.append(pack),
    )


def test_train_step_receipt_and_release():
    actor, released = FakeActorGroup(), []
    t = trainer(actor, released)
    assert isinstance(t, TrainerGroup)
    receipt = t.train_step(handle())
    assert isinstance(receipt, LocalStepReceipt)
    assert actor.calls[0] == ("train", 3, "PACK")  # opaque payload passed straight through
    assert actor.calls[1] == ("plugin", GRAD_NORM)
    assert actor.calls[2] == ("plugin", APPLIED_LRS)
    assert released == ["PACK"]
    assert receipt.optimizer_step_succeeded and receipt.optimizer_steps == 1
    assert receipt.trained_tokens == 17
    assert receipt.trajectory_ids == ("r3:g0:s0", "r3:g0:s1", "r3:g1:s10", "r3:g1:s11")
    assert receipt.base_policy_version == 3 and receipt.base_policy_hash == H
    assert receipt.input_batch_hash == batch_hash(handle()) and receipt.parameter_layout_hash == L
    assert t.last_grad_norm == 0.3


def test_token_mismatch_rejected_before_training_and_refs_released():
    actor, released = FakeActorGroup(), []
    with pytest.raises(PolicyTokenMismatch, match="g0"):
        trainer(actor, released).train_step(handle(token=policy_token(2, H)))
    assert actor.calls == [] and released == ["PACK"]


def test_discarded_step_is_not_a_successful_receipt():
    actor, released = FakeActorGroup(outcome="DISCARDED_SHOULD_RETRY"), []
    receipt = trainer(actor, released).train_step(handle())
    assert not receipt.optimizer_step_succeeded and receipt.optimizer_steps == 0
    assert released == ["PACK"]


def test_multi_cell_output_and_missing_payload_rejected():
    released = []
    with pytest.raises(TrainStepError, match="single-cell"):
        trainer(FakeActorGroup(outputs=2), released).train_step(handle())
    assert released == ["PACK"]
    with pytest.raises(TrainStepError, match="payload"):
        trainer(FakeActorGroup(), []).train_step(handle(payload=None))


def test_onload_offload():
    actor = FakeActorGroup()
    t = trainer(actor, [])
    t.onload(); t.offload()
    assert actor.calls == [("onload",), ("offload",)]
    actor2 = FakeActorGroup()
    t2 = trainer(actor2, [], offload=False)
    t2.onload(); t2.offload()
    assert actor2.calls == [("clear",)]


# ---------------------------------------------------------------- publish

torch = pytest.importorskip("torch")

from yeto.rl.core import canonical_state  # noqa: E402
from yeto.rl.engine.trainable_state import TrainableState  # noqa: E402

PREFIX = "base_model.model.model.layers.0.self_attn.q_proj"


def state(version=4, offset=0.0):
    tensors = {f"{PREFIX}.lora_A.weight": torch.ones(2, 4) + offset,
               f"{PREFIX}.lora_B.weight": torch.zeros(6, 2)}
    return TrainableState.from_lora(
        canonical_state(version, tensors, base_model_revision="0" * 40, lora_config_hash="1" * 64)
    )


class Engine:
    def __init__(self, url, fail=False, lie=False):
        self.server_url, self.fail, self.lie, self.version = url, fail, lie, None

    async def update_weight_version(self, v, abort_all_requests=False):
        if self.fail:
            raise RuntimeError("engine down")
        self.version = v

    async def get_weight_version(self):
        return "stale" if self.lie else self.version


class Controller:
    def __init__(self, engines, checksum_ok=True):
        self.engines, self.checksum_ok, self.log = engines, checksum_ok, []

    async def start_update_weights(self, model_id=None):
        self.log.append("start")
        return SimpleNamespace(
            rollout_engines=self.engines,
            snapshot_cell_id_to_hashes={f"c{i}": "h" for i in range(len(self.engines))},
        )

    async def get_cell_statuses(self):
        return {f"c{i}": SimpleNamespace(phase="Running") for i in range(len(self.engines))}

    async def end_update_weights(self, snapshot_cell_id_to_hashes):
        self.log.append("end")

    async def abort_update_weights(self):
        self.log.append("abort")

    async def check_weights(self, action, **kw):
        return [{"success": self.checksum_ok, "ranks": [{"checksums": {"w": "x"}}]} for _ in self.engines]


def flatten(raw):  # mirrors upstream flatten_inference_engine_checksums
    out = []
    for body in raw:
        assert body["success"], body
        out.append({f"rank0/{k}": v for r in body["ranks"] for k, v in r["checksums"].items()})
    return out


def publisher(controller, *, update_fails=False, export=None):
    calls = []

    async def update_weights(args, actor, executor, ctl, **kw):
        calls.append("update_weights")
        if update_fails:
            raise RuntimeError("broadcast failed")

    pub = MilesPublisher(
        args=SimpleNamespace(), actor_model=object(), rollout_executor=object(),
        inference_controller=controller, export_trainer_state=export,
        update_weights=update_weights, flatten_checksums=flatten,
    )
    return pub, calls


def test_publish_full_manifest_with_members():
    engines = [Engine("http://e0"), Engine("http://e1")]
    ctl = Controller(engines)
    pub, calls = publisher(ctl, export=lambda: state())
    assert isinstance(pub, Publisher)
    s = state()
    result = pub.publish(s)
    m = result.manifest
    assert isinstance(m, InferencePublicationManifest)
    assert m.publication_mode == "full" and m.base_policy_version is None and m.complete
    assert m.target_policy_version == 4 and m.target_policy_hash == s.policy_tensor_hash()
    assert m.payload_bytes == (8 + 12) * 4
    assert result.members == frozenset({"engine:c0", "engine:c1"})
    token = policy_token(4, s.policy_tensor_hash())
    assert all(e.version == token for e in engines)
    assert calls == ["update_weights"] and ctl.log == ["start", "end"]
    assert set(pub.last_engine_checksums) == {"http://e0", "http://e1"}
    # payload hash is content-addressed
    pub2, _ = publisher(Controller([Engine("http://e0")]))
    assert pub2.publish(state(offset=1.0)).manifest.payload_hash != m.payload_hash


@pytest.mark.parametrize("bad", [dict(fail=True), dict(lie=True)])
def test_single_engine_failure_is_an_error_not_a_manifest(bad):
    engines = [Engine("http://e0"), Engine("http://e1", **bad)]
    ctl = Controller(engines)
    pub, _ = publisher(ctl)
    with pytest.raises(PublicationError) as err:
        pub.publish(state())
    assert err.value.failed_members == frozenset({"http://e1"})
    assert ctl.log == ["start", "abort"]


def test_update_weights_failure_and_checksum_failure():
    pub, _ = publisher(Controller([Engine("http://e0")]), update_fails=True)
    with pytest.raises(PublicationError, match="update_weights"):
        pub.publish(state())
    pub, _ = publisher(Controller([Engine("http://e0")], checksum_ok=False))
    with pytest.raises(PublicationError, match="checksum"):
        pub.publish(state())


def test_trainer_state_must_match_published_state():
    pub, calls = publisher(Controller([Engine("http://e0")]), export=lambda: state(offset=1.0))
    with pytest.raises(PublicationError, match="differ"):
        pub.publish(state())
    assert calls == []


def test_no_engines_is_an_error():
    ctl = Controller([])
    pub, _ = publisher(ctl)
    with pytest.raises(PublicationError, match="no rollout engines"):
        pub.publish(state())
    assert ctl.log == ["start", "abort"]


def test_manifest_hash_is_the_token_hash():
    s = state(version=7)
    pub, _ = publisher(Controller([Engine("http://e0")]))
    m = pub.publish(s).manifest
    assert m.target_policy_hash == s.policy_tensor_hash() != s.policy_hash()


def test_publisher_members_share_the_pool_source():
    class Partial(Controller):
        async def get_cell_statuses(self):  # c1 is not Running
            return {"c0": SimpleNamespace(phase="Running"), "c1": SimpleNamespace(phase="Pending")}

    pub, _ = publisher(Partial([Engine("http://e0"), Engine("http://e1")]))
    assert pub.publish(state()).members == frozenset({"engine:c0"})


def test_step_metrics_reports_grad_norm_for_the_driver():
    from yeto.rl.engine.driver import TrainStepMetrics

    actor, released = FakeActorGroup(norm=0.7), []
    t = trainer(actor, released)
    t.train_step(handle())
    metrics = t.step_metrics()
    assert isinstance(metrics, TrainStepMetrics) and metrics.grad_norm == 0.7
    t2 = trainer(FakeActorGroup(outcome="DISCARDED"), [])
    t2.train_step(handle())
    import math

    assert math.isnan(t2.step_metrics().grad_norm)
    assert t2.step_metrics().applied_lrs is None


def test_step_metrics_reports_applied_lrs_per_optimizer_step():
    t = trainer(FakeActorGroup(lrs=((2e-5,), (2e-5,))), [])
    t.train_step(handle())
    assert t.step_metrics().applied_lrs == (2e-5,)
    # a step count mismatch or rank disagreement is a failed train step
    from yeto.rl.engine.miles_adapter.trainer import TrainStepError

    for lrs in (((),()), ((1e-5, 0.0), (1e-5, 0.0)), ((1e-5,), (0.0,))):
        with pytest.raises(TrainStepError):
            trainer(FakeActorGroup(lrs=lrs), []).train_step(handle())
