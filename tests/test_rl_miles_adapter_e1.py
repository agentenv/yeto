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

    # fork context_lock semantics: @acquires_lock releases on its own failure;
    # @releases_lock asserts the lock is held.
    lock_held = False
    commit_fails = False

    async def start_commit_weight_version(self, *, weight_version, expected_epoch, model_id=None):
        self.log.append(("start_commit_weight_version", weight_version, expected_epoch))
        assert not self.lock_held
        if self.commit_fails:
            raise RuntimeError("serving engines report another version")
        self.lock_held = True

    async def end_commit_weight_version(self):
        self.log.append(("end_commit_weight_version",))
        assert self.lock_held, "releasing a lock that is not held"
        self.lock_held = False


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


# ---------------------------------------------------------------- data cursor / ledger / 4.4 / entry
def test_hook_reports_data_cursor_and_buffer_length():
    from yeto.rl.engine.miles_adapter.rollout import handle_from_metadata
    from yeto.rl.engine.miles_adapter.rollout_meta_hook import METADATA_SCHEMA, data_cursor

    src = SimpleNamespace(sample_offset=12, epoch_id=0, sample_group_index=3, sample_index=24,
                          buffer=[], get_buffer_length=lambda: 0)
    cursor, length = data_cursor(src)
    assert cursor == {"sample_offset": 12, "epoch_id": 0, "sample_group_index": 3,
                      "sample_index": 24} and length == 0
    assert data_cursor(SimpleNamespace(sample_offset=1)) == (None, None)
    assert data_cursor(None) == (None, None)
    payload = {"schema": METADATA_SCHEMA, "rollout_id": 1, "completed": 1, "aborted": 0,
               "groups": [{"group_id": "g", "sample_ids": ["s"], "policy_token": "t",
                           "reward_mean": 0.0, "reward_std": 0.0, "token_count": 1}],
               "data_cursor": cursor, "buffer_length": 0}
    h = handle_from_metadata(payload, rollout_id=1, policy_version=1, policy_hash="h",
                             data_pack=None)
    assert h.carried_over == 0 and h.data_cursor == cursor and h.buffer_length == 0
    del payload["buffer_length"]
    h = handle_from_metadata(payload, rollout_id=1, policy_version=1, policy_hash="h",
                             data_pack=None)
    assert h.carried_over is None


def test_ledger_engine_discarded_and_cut_summary(tmp_path):
    from yeto.rl.engine.journal import read_journal
    from yeto.rl.engine.ledger import BatchLedger

    g = SimpleNamespace(group_id="g0", sample_ids=("s0",), filtered_samples=None)
    batch = SimpleNamespace(rollout_id=0, groups=(g,), filtered=2, carried_over=0,
                            buffer_length=0, aborted_in_flight_groups=3)
    led = BatchLedger(tmp_path)
    led.prepare(batch, policy_token="t")
    summary = led.cut_summary()
    assert summary["ready_unconsumed"] == 1 and summary["ready_unconsumed_group_ids"] == ["g0"]
    assert summary["engine_carried_over"] == 0 and summary["carried_over"] == 0
    led.optimizer_applied(0)
    led.close()
    led = BatchLedger(tmp_path)  # replayed from disk
    assert led.cut_summary()["ready_unconsumed"] == 0
    assert led.cut_summary()["engine_buffer_length"] == 0
    kinds = {r["kind"]: r for r in read_journal(tmp_path / "ledger")}
    assert kinds["engine_discarded"]["groups"] == 3 and kinds["filtered"]["detail"]["groups"] == 2
    led.close()


def test_driver_rebuild_trainer_republishes_without_outer_effects(tmp_path):
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.driver import DriverError, EventTape, IslandDriver
    from yeto.rl.engine.fake import FakeEngine, fake_capabilities

    engine = FakeEngine(tensors={f"{PREFIX}.lora_A.weight": torch.zeros(1, 2)},
                        placement_kind="fixed-partition")
    calls = []

    class Sync:
        def start(self, d):
            from yeto.rl.engine.driver import SyncStart
            return SyncStart(d.export_local(), 0)

        def boundary(self, d, *, rollout_id, stats):
            from yeto.rl.engine.driver import SyncBoundary
            return SyncBoundary(d.export_local(), stop=rollout_id >= 1)

        def published(self, d, *, rollout_id, policy_hash):
            calls.append(("published", rollout_id))

        def finish(self, d):
            pass

        def close(self):
            pass

    driver = IslandDriver(
        learner_id=0, rollout=engine.rollout, trainer=engine.trainer,
        policy_state=engine.policy_state, publisher=engine.publisher,
        placement=engine.placement, capabilities=fake_capabilities(),
        algorithm=AlgorithmSpec(), sync=Sync(), events=EventTape(tmp_path / "e.jsonl", 0))
    rebuilt = []
    orig = driver.safe_point

    def safe_point(rollout_id):
        out = orig(rollout_id)
        if rollout_id == 1:
            h = driver.published_state.policy_tensor_hash()
            with pytest.raises(DriverError, match="not the published"):
                driver.rebuild_trainer(lambda: None, cut_policy_hash="0" * 64)
            driver.rebuild_trainer(lambda: rebuilt.append(1), cut_policy_hash=h)
        return out

    driver.safe_point = safe_point
    driver.handshake()
    driver.run()
    assert rebuilt == [1]
    assert calls == [("published", 0), ("published", 1), ("published", 2)]  # no extra outer call
    import json as _json
    ev = [_json.loads(x) for x in (tmp_path / "e.jsonl").read_text().splitlines()]
    assert [e["policy_version"] for e in ev if e["event"] == "rl_trainer_rebuilt"] == [1]
    driver.at_safe_point = False
    with pytest.raises(DriverError, match="safe point"):
        driver.rebuild_trainer(lambda: None, cut_policy_hash="x")


def test_compose_island_with_elastic_wiring(tmp_path, monkeypatch):
    from tests.test_rl_engine_selection import NAME as SEL_NAME, _Actor, _Controller
    from tests.test_rl_miles_adapter_rollout import Call, Sample, Span
    from yeto.rl.core import canonical_state
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.bridges import LocalOnlySync
    from yeto.rl.engine.driver import EventTape
    from yeto.rl.engine.execution_profile import ExecutionProfile
    from yeto.rl.engine.journal import read_journal
    from yeto.rl.engine.miles_adapter import LoopRunner
    from yeto.rl.engine.miles_adapter import rollout_meta_hook as hook
    from yeto.rl.engine.miles_adapter.elastic_placement import ElasticPlacement
    from yeto.rl.engine.miles_adapter.elastic_wiring import build_elastic
    from yeto.rl.engine.miles_adapter.entry import compose_island, miles_capabilities
    from yeto.rl.engine.miles_adapter.placement import MilesPlacement, PlacementRequest
    from yeto.rl.engine.miles_adapter.rollout import DirMetadataSource

    sink = tmp_path / "sink"
    monkeypatch.setenv(hook.META_SINK_ENV, f"dir:{sink}")

    class Ctl(_Controller):
        async def get_membership_status(self):
            return {"epoch": 0, "incomplete": None}

    controller, actor = Ctl(), _Actor()
    rollout_args = SimpleNamespace(n_samples_per_prompt=2, yeto_rl_elastic_metadata=True)
    source = SimpleNamespace(sample_offset=0, epoch_id=0, sample_group_index=0, sample_index=0,
                             buffer=[])

    class Executor:
        async def get(self, rollout_id):
            token = controller.engine.version
            # Miles group_index is the data source's monotonic sample_group_index
            groups = [[Sample(index=100 * rollout_id + 10 * g + i,
                              group_index=10 * rollout_id + g, rollout_id=rollout_id,
                              reward=float(i), weight_versions=[Call([Span(token)])])
                       for i in range(2)] for g in range(2)]
            source.sample_offset += 2
            hook.record_trained_groups(rollout_args, groups)
            hook.extract_rollout_metadata(rollout_args, groups, source)
            return SimpleNamespace(sample_indices=[s.index for g in groups for s in g])

    async def update_weights(*a, **k):
        pass

    fp = "sha256:" + "0" * 64
    profile = ExecutionProfile(name="p", execution_mode="partitioned-serial",
                               outer_protocol="none").bind_algorithm(AlgorithmSpec())
    elastic = build_elastic(
        state_dir=tmp_path / "state",
        resources={"configs": {"T1R1S1": {"trainer": 1, "rollout": 1, "standby": 1},
                               "T1R2S0": {"trainer": 1, "rollout": 2}},
                   "edges": [{"source": "T1R1S1", "target": "T1R2S0", "kind": "rollout-only"}]},
        attestation=None, profile=profile, initial_config="T1R1S1", runtime_fingerprint=fp,
        declared_cells=["c0", "c1"], pool_gpus=["g0", "g1", "g2"], on_watchdog=print)
    assert elastic.controller._on_watchdog is print
    runner = LoopRunner()
    driver = compose_island(
        miles_args=SimpleNamespace(num_steps_per_rollout=1, offload_train=True,
                                   offload_rollout=False),
        launch=SimpleNamespace(placement=PlacementRequest("colocated", 1, 1, 1)),
        algorithm=AlgorithmSpec(), inference_controller=controller, rollout_executor=Executor(),
        actor_model=actor, learner_id=0, base_model_revision="0" * 40,
        lora_config_hash="1" * 64,
        layout_hash=canonical_state(0, {SEL_NAME: torch.zeros(2, 4)},
                                    base_model_revision="0" * 40,
                                    lora_config_hash="1" * 64).layout_hash,
        sync=LocalOnlySync(2), progress=None, metadata=DirMetadataSource(sink),
        capabilities=miles_capabilities(fp), runner=runner,
        events=EventTape(tmp_path / "events.jsonl", 0), update_weights=update_weights,
        release_refs=lambda args, pack: None, flatten_checksums=lambda raw: [{"w": "x"} for _ in raw],
        placement=MilesPlacement(PlacementRequest("colocated", 1, 1, 1),
                                 {"actor": (None, [0], [0]), "rollout": (None, [0], [0])},
                                 logical=True),
        elastic=elastic,
    )
    assert isinstance(driver.placement, ElasticPlacement)
    assert driver.controller is elastic.controller and driver.ledger is elastic.ledger
    assert elastic.controller.inspect().members == ("engine:c0",)
    driver.run()
    runner.close()
    # metadata cursor of the last batch (the stub executor has no live data source)
    assert driver.rollout.last_batch_data_cursor()["sample_offset"] == 4
    assert driver.rollout.data_cursor() is None  # live position unknown: never the cache
    kinds = [r["kind"] for r in read_journal(tmp_path / "state/ledger")]
    assert kinds.count("outer_recorded") == 2
    # no attestation: every transition is refused (fail closed)
    with pytest.raises(Exception, match="attestation"):
        elastic.controller.plan("T1R2S0", 0, deadline_s=60)
    elastic.controller.close()
    elastic.ledger.close()


def test_failed_commit_check_does_not_release_a_lock_it_does_not_hold():
    ctl = ForkController()
    pub, _, _ = make(ctl)
    ctl.commit_fails = True
    with pytest.raises(PublicationError, match="do not all report"):
        pub.verify_serving_policy(epoch=0, token_rollout_id=1, state=state(1))
    assert ("end_commit_weight_version",) not in ctl.log and not ctl.lock_held
    ctl.commit_fails = False
    pub.verify_serving_policy(epoch=0, token_rollout_id=1, state=state(1))
    assert ctl.log[-1] == ("end_commit_weight_version",) and not ctl.lock_held


def test_default_metadata_is_unchanged_without_elastic(tmp_path, monkeypatch):
    """M1: data cursor / buffer length only when E1/E2 is on; default carried_over stays None."""
    import json as _json

    from tests.test_rl_miles_adapter_rollout import Call, Sample, Span
    from yeto.rl.engine.miles_adapter import rollout_meta_hook as hook
    from yeto.rl.engine.miles_adapter.rollout import DirMetadataSource, handle_from_metadata

    monkeypatch.delenv(hook.ELASTIC_METADATA_ENV, raising=False)
    source = SimpleNamespace(sample_offset=4, epoch_id=0, sample_group_index=2, sample_index=4,
                             buffer=[])
    outs = {}
    for enabled in (False, True):
        sink = tmp_path / f"s{enabled}"
        monkeypatch.setenv(hook.META_SINK_ENV, f"dir:{sink}")
        args = SimpleNamespace(n_samples_per_prompt=2, yeto_rl_elastic_metadata=enabled)
        groups = [[Sample(index=i, group_index=0, rollout_id=0, reward=float(i),
                          weight_versions=[Call([Span("yeto:0:" + "a" * 64)])]) for i in range(2)]]
        hook.record_trained_groups(args, groups)
        hook.extract_rollout_metadata(args, groups, source)
        outs[enabled] = DirMetadataSource(sink).take(0)
    assert "data_cursor" not in outs[False] and "buffer_length" not in outs[False]
    extra = {k: v for k, v in outs[True].items() if k not in outs[False]}
    assert set(extra) == {"data_cursor", "buffer_length"}
    assert _json.dumps({k: v for k, v in outs[True].items() if k not in extra}, sort_keys=True) \
        == _json.dumps(outs[False], sort_keys=True)
    h = handle_from_metadata(outs[False], rollout_id=0, policy_version=0, policy_hash="a",
                             data_pack=None)
    assert h.carried_over is None and h.data_cursor is None and h.buffer_length is None
