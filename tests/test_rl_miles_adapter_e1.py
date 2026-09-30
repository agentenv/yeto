"""rl-infra-spec 3.4/3.5 (CPU): MilesRolloutPool membership verbs and member publication.

The fake controller records the exact fork call sequence (yeto/ports 0af62f4d
signatures); a GPU run is still required for the 3.4 X2 and 3.5 acceptance.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from yeto.rl.core import canonical_state  # noqa: E402
from yeto.rl.engine.miles_adapter.publish import MilesPublisher, PublicationError  # noqa: E402
from yeto.rl.engine.miles_adapter.rollout import (  # noqa: E402
    MembershipPlanError,
    MilesRolloutPool,
    cell_of,
    member_id,
)
from yeto.rl.engine.ports import ElasticRolloutPool, MemberPublisher  # noqa: E402
from yeto.rl.engine.trainable_state import TrainableState  # noqa: E402

PREFIX = "base_model.model.model.layers.0.self_attn.q_proj"


def state(version=1):
    tensors = {f"{PREFIX}.lora_A.weight": torch.ones(2, 4),
               f"{PREFIX}.lora_B.weight": torch.zeros(6, 2)}
    return TrainableState.from_lora(
        canonical_state(version, tensors, base_model_revision="0" * 40, lora_config_hash="1" * 64))


class Engine:
    def __init__(self, url):
        self.server_url, self.version = url, None

    async def update_weight_version(self, v, abort_all_requests=False):
        self.version = v

    async def get_weight_version(self):
        return self.version


class ForkController:
    """Records the fork InferenceController calls (names/kwargs as in 0af62f4d)."""

    def __init__(self, running=("c0", "c1")):
        self.log = []
        self.running = list(running)
        self.engines = {c: Engine(f"http://{c}") for c in ("c0", "c1", "c2", "c3")}
        self.checksum = {c: "x" for c in self.engines}
        self.epoch = 0

    async def get_cell_statuses(self):
        return {c: SimpleNamespace(phase="Running") for c in self.running}

    async def start_cells(self, cell_ids, *, expected_epoch):
        self.log.append(("start_cells", tuple(cell_ids), expected_epoch))
        self.running += cell_ids
        self.epoch += 1
        return self.epoch

    async def wait_cells_tracked(self, cell_ids, *, timeout_seconds):
        self.log.append(("wait_cells_tracked", tuple(cell_ids)))

    async def stop_cells(self, cell_ids, *, expected_epoch):
        self.log.append(("stop_cells", tuple(cell_ids), expected_epoch))
        self.running = [c for c in self.running if c not in cell_ids]
        self.epoch += 1
        return self.epoch

    async def drain_cells(self, cell_ids, *, timeout_seconds):
        self.log.append(("drain_cells", tuple(cell_ids)))
        return True

    async def uncordon_cells(self, cell_ids):
        self.log.append(("uncordon_cells", tuple(cell_ids)))

    async def get_membership_status(self):
        return {"epoch": self.epoch, "incomplete": ["stop", ["c3"]], "incomplete_reason": "stop_failed"}

    async def restore_membership_state(self, *, epoch, incomplete, last_op, expected_current_epoch):
        self.log.append(("restore", epoch, incomplete, last_op, expected_current_epoch))
        return {"epoch": epoch}

    # update lock
    async def start_update_weights(self, model_id=None, members=None, expected_epoch=None,
                                   admit_cordoned=False):
        self.log.append(("start_update_weights", tuple(members or ()), expected_epoch, admit_cordoned))
        cells = members if members is not None else self.running
        return SimpleNamespace(rollout_engines=[self.engines[c] for c in cells],
                               snapshot_cell_id_to_hashes={c: "h" for c in cells})

    async def end_update_weights(self, snapshot_cell_id_to_hashes):
        self.log.append(("end_update_weights",))

    async def abort_update_weights(self):
        self.log.append(("abort_update_weights",))

    async def check_weights(self, action, **kw):
        self.log.append(("check_weights",))
        return [{"success": True, "ranks": [{"checksums": {"w": self.checksum[c]}}]}
                for c in self.running]

    async def admit_cells(self, cell_ids, *, expected_epoch, expected_weight_version):
        self.log.append(("admit_cells", tuple(cell_ids), expected_epoch, expected_weight_version))

    async def start_commit_weight_version(self, *, weight_version, expected_epoch, model_id=None):
        self.log.append(("start_commit_weight_version", weight_version, expected_epoch))

    async def end_commit_weight_version(self):
        self.log.append(("end_commit_weight_version",))


def flatten(raw):
    return [{f"rank0/{k}": v for r in body["ranks"] for k, v in r["checksums"].items()}
            for body in raw]


def make(controller):
    updates = []

    async def update_weights(args, actor, executor, ctl, **kw):
        updates.append(kw)
        ctl.log.append(("update_weights", tuple(kw.get("members") or ()),
                        kw.get("expected_epoch"), kw.get("admit_cordoned", False)))

    pub = MilesPublisher(args=SimpleNamespace(), actor_model=object(), rollout_executor=object(),
                         inference_controller=controller, update_weights=update_weights,
                         flatten_checksums=flatten)
    pool = MilesRolloutPool(inference_controller=controller, rollout_executor=object(),
                            metadata=None, expected_policy=lambda: (1, "h"),
                            declared_cells=["c0", "c1", "c2", "c3"])
    return pub, pool, updates


def test_member_ids_round_trip_and_protocols():
    assert cell_of(member_id("c7")) == "c7"
    with pytest.raises(ValueError):
        cell_of("c7")
    pub, pool, _ = make(ForkController())
    assert isinstance(pool, ElasticRolloutPool) and isinstance(pub, MemberPublisher)


def test_pool_membership_verbs_map_to_fork_calls():
    ctl = ForkController()
    _, pool, _ = make(ctl)
    assert pool.plan_add(2) == {"engine:c2", "engine:c3"}
    with pytest.raises(MembershipPlanError):
        pool.plan_add(3)
    assert pool.add_engines(2, epoch=0, members=frozenset({"engine:c3", "engine:c2"})) == {
        "engine:c2", "engine:c3"}
    assert pool.drain(frozenset({"engine:c3"}), deadline=0.0)
    pool.undrain(frozenset({"engine:c3"}))
    remaining = pool.remove_engines(frozenset({"engine:c3"}), epoch=1)
    assert remaining == {"engine:c0", "engine:c1", "engine:c2"}
    assert ctl.log == [
        ("start_cells", ("c2", "c3"), 0), ("wait_cells_tracked", ("c2", "c3")),
        ("drain_cells", ("c3",)), ("uncordon_cells", ("c3",)), ("stop_cells", ("c3",), 1)]
    assert pool.membership_status()["incomplete"] == ["stop", ["engine:c3"]]
    pool.restore_membership(epoch=3, incomplete=None, last_op=["start", ["engine:c2"]],
                            expected_current_epoch=0)
    assert ctl.log[-1] == ("restore", 3, None, ["start", ["c2"]], 0)
    with pytest.raises(MembershipPlanError, match="declared"):
        pool.add_engines(1, epoch=2, members=frozenset({"engine:c9"}))


def test_undeclared_pool_has_no_e1_verbs():
    pool = MilesRolloutPool(inference_controller=ForkController(), rollout_executor=object(),
                            metadata=None, expected_policy=lambda: (1, "h"))
    with pytest.raises(MembershipPlanError, match="off"):
        pool.plan_add(1)


def test_member_publish_admits_only_after_payload_readback():
    ctl = ForkController()
    pub, pool, updates = make(ctl)
    s = state(1)
    full = pub.publish(s)
    assert full.members == {"engine:c0", "engine:c1"}
    ctl.log.clear()
    pool.add_engines(2, epoch=0, members=pool.plan_add(2))
    ctl.log.clear()
    result = pub.publish_members(s, frozenset({"engine:c2", "engine:c3"}), epoch=1)
    token = f"yeto:1:{s.policy_tensor_hash()}"
    assert result.members == {"engine:c2", "engine:c3"}
    assert ctl.log == [
        ("wait_cells_tracked", ("c2", "c3")),
        ("update_weights", ("c2", "c3"), 1, True),
        ("start_update_weights", ("c2", "c3"), 1, False),
        ("end_update_weights",),
        ("check_weights",),
        ("admit_cells", ("c2", "c3"), 1, token),
        ("start_commit_weight_version", token, 1),
        ("end_commit_weight_version",),
    ]
    # non-members were never stamped by the member publish
    assert ctl.engines["c2"].version == token


def test_member_publish_with_wrong_payload_is_never_admitted():
    ctl = ForkController()
    pub, pool, _ = make(ctl)
    s = state(1)
    pub.publish(s)
    pool.add_engines(1, epoch=0, members=frozenset({"engine:c2"}))
    ctl.checksum["c2"] = "corrupt"
    with pytest.raises(PublicationError, match="read-back"):
        pub.publish_members(s, frozenset({"engine:c2"}), epoch=1)
    assert not [c for c in ctl.log if c[0] == "admit_cells"]


def test_member_publish_needs_a_verified_reference_of_the_same_policy():
    ctl = ForkController()
    pub, _, _ = make(ctl)
    with pytest.raises(PublicationError, match="reference"):
        pub.publish_members(state(1), frozenset({"engine:c2"}), epoch=0)
    pub.publish(state(1))
    with pytest.raises(PublicationError, match="reference"):  # another token
        pub.publish_members(state(2), frozenset({"engine:c2"}), epoch=0)
    pub._verify_checksums = False
    with pytest.raises(PublicationError, match="read-back"):
        pub.publish_members(state(1), frozenset({"engine:c2"}), epoch=0)
    pub._verify_checksums = True
    pub._args = SimpleNamespace(offload_rollout=True)
    with pytest.raises(PublicationError, match="partitioned"):
        pub.publish_members(state(1), frozenset({"engine:c2"}), epoch=0)
