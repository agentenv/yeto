"""Peer recovery on the Miles full-parameter role stream (CPU, no Miles, no Ray).

The Megatron trainer group, the chunked adapter and the syncer connection are
stubs; the role stream, the cut-plugin call sequence, the snapshot store and
the recovery exchange are the real code. What stays GPU-only: the cut-plugin
shard save/restore inside Megatron ranks and the chunked apply of the resent
fragments (the stubs record what they were asked to do).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto import recovery_protocol as rp
from yeto.protocol import BcastFragment
from yeto.rl.adapters.miles import cut_plugin
from yeto.rl.adapters.miles.full_parameter import role_stream as role_stream_module
from yeto.rl.adapters.miles.full_parameter.peer_recovery import (
    RANKS,
    MilesRankShards,
    PeerRecoveryOptions,
    layout_tag,
    parallel_layout,
)
from yeto.rl.adapters.miles.full_parameter.role_stream import (
    MilesFullParameterRoleStream,
    RoleStreamConfig,
)
from yeto.rl.adapters.miles.models.chunked_full_parameter import (
    MilesChunkedFullParameterAdapter,
    StoredAuthoritativeFragment,
)
from yeto.rl.local_learner import ComponentIdentity
from yeto.rl.peer_recovery import DirectorySnapshotStore, SnapshotRefused

P = 2
LAYOUT_HASH = "ab" * 32


class FakeGroup:
    """Megatron ranks: each holds an optimizer state dict; shards are files."""

    def __init__(self, *, tp=2, dp=1, adam_step=0, missing_rank=None):
        self.coords = [{"tp": r % tp, "pp": 0, "dp": r // tp, "tp_size": tp, "pp_size": 1, "cp_size": 1,
                        "ep_size": 1, "dp_size": dp} for r in range(tp * dp)]
        self.states = [{"adam_step": adam_step, "scheduler_steps": adam_step, "m": [0.25 * r] * 3}
                       for r in range(tp * dp)]
        # Miles' full-parameter tracking on the trainer (one actor stands for every rank).
        self.actor = SimpleNamespace(
            opt_param_scheduler=SimpleNamespace(num_steps=adam_step),
            _external_full_policy_version=0, _external_full_local_step_generation=0,
            _external_full_exported_local_step_generation=0, _external_full_scheduler_anchor_steps=adam_step,
            _external_full_last_local_rollout_id=None, _external_full_last_local_step_receipt=None)
        self.missing_rank = missing_rank
        self.installed = 0
        self.generation = 0
        self.applied = []
        self.calls = []
        # Background writes (S27 G-EL): polls until every rank reports done.
        self.write_polls = 1
        self.pending_shards = None
        self.polls_until_written = 0

    async def run_plugin(self, name, kwargs):
        self.calls.append(name)
        if name == cut_plugin.RANK_COORDS:
            return [dict(c) for c in self.coords]
        if name == cut_plugin.SAVE_CUT_SHARD:
            out = []
            for coord, state in zip(self.coords, self.states, strict=True):
                path = cut_plugin.shard_name(coord)
                data = json.dumps(state).encode()
                if coord != self.missing_rank:  # a rank on another node writes elsewhere
                    (Path(kwargs["directory"]) / path).write_bytes(data)
                out.append({"path": path, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                            "coord": coord})
            if kwargs.get("background") == cut_plugin.HOST_BACKGROUND:
                # S27 G-EL: the ranks write in the background; FINISH returns the summaries
                self.pending_shards, self.polls_until_written = out, self.write_polls
                return [{"pending": kwargs["cut_id"], "coord": r["coord"]} for r in out]
            return out
        if name == cut_plugin.POLL_CUT_SHARD:
            self.polls_until_written -= 1
            return [{"done": self.polls_until_written <= 0} for _ in self.coords]
        if name == cut_plugin.FINISH_CUT_SHARD:
            out, self.pending_shards = self.pending_shards, None
            return out
        if name == cut_plugin.RESTORE_CUT_SHARD:
            by_path = {f["path"]: f for f in kwargs["files"]}
            for rank, coord in enumerate(self.coords):
                entry = by_path[cut_plugin.shard_name(coord)]
                data = (Path(kwargs["directory"]) / entry["path"]).read_bytes()
                assert hashlib.sha256(data).hexdigest() == entry["sha256"]
                self.states[rank] = json.loads(data)
            # The real restore loads the donor's LR scheduler, then rebases Miles' anchor.
            self.actor.opt_param_scheduler.num_steps = self.states[0]["scheduler_steps"]
            cut_plugin.rebase_full_parameter_tracking(self.actor)
            return [{"ok": True} for _ in self.coords]
        raise AssertionError(name)

    async def full_parameter_optimizer_states(self):
        return tuple(SimpleNamespace(role="actor", installed_policy_version=self.installed,
                                     local_step_generation=self.generation, adam_step=s["adam_step"])
                     for s in self.states)


class FreshGroup(FakeGroup):
    """Like Megatron in a fresh process: no optimizer-state proof before the
    island's first local step, even after a snapshot restore (s27-gd-rec-a)."""

    async def full_parameter_optimizer_states(self):
        raise RuntimeError("full-parameter last rollout ID is unavailable")


class FakeAdapter:
    def __init__(self):
        self.layout = SimpleNamespace(layout_hash=LAYOUT_HASH, fragments=SimpleNamespace(num_fragments=P))
        self.released = []

    def fragment_sink(self, initial, *, ray_module=None):
        return SimpleNamespace(release_all=lambda: None)

    def assemble_mixed_target(self, current, *, target_policy_version, authoritative_fragments):
        return SimpleNamespace(policy_version=target_policy_version, policy_hash=f"p{target_policy_version}",
                               fragments={f.fragment_id: f.refs[0] for f in authoritative_fragments})

    async def apply(self, group, target, *, commit_token):
        assert commit_token
        actor = group.actor  # Miles prepare_full_parameter_chunked_shard checks scheduler congruence
        if actor.opt_param_scheduler.num_steps != actor._external_full_scheduler_anchor_steps:
            raise RuntimeError("full-parameter scheduler progress has no local-step receipt")
        group.applied.append(dict(target.fragments))
        group.installed = target.policy_version
        group.generation = 0

    def release(self, cut):
        self.released.append(cut)


def stored(fragment_id, version, value):
    return StoredAuthoritativeFragment(fragment_id, version, LAYOUT_HASH, None, "", None, [value],
                                       hashlib.sha256(repr((fragment_id, version, value)).encode()).hexdigest())


class FakeClient:
    """SyncerClient surface: scripted recovery events, snapshot requests, broadcasts."""

    def __init__(self, *, events=(), requests=(), updates=()):
        self.events = list(events)
        self.requests = list(requests)
        self.updates = list(updates)
        self.sent = []
        self.finalizing = threading.Event()
        self.started = self.closed = self.marked = False

    def install_inbound_chunk_sink(self, sink):
        pass

    def start(self):
        self.started = True

    def check_health(self):
        pass

    def heartbeat(self, step):
        pass

    def next_recovery_event(self, timeout):
        return self.events.pop(0) if self.events else None

    def drain_updates(self):
        out, self.updates = self.updates, []
        return out

    def drain_snapshot_requests(self):
        out, self.requests = self.requests, []
        return out

    def send_snapshot_ready(self, request, versions, sha256, locator):
        self.sent.append(("ready", tuple(versions), sha256, locator))
        return True

    def send_recovery_restored(self, versions):
        self.sent.append(("restored", versions))
        return True

    def send_recovery_failed(self, reason):
        self.sent.append(("failed", reason))
        return True

    def send_initial_fragments(self, parts):
        raise AssertionError("a recovering island must not send INIT")

    def mark_initial_cut_received(self):
        self.marked = True

    def close(self):
        self.closed = True


def config(store, *, recover, timeout_s=10.0, island_lease_s=None, donor_write="background"):
    return RoleStreamConfig(
        island_lease_s=island_lease_s,
        algorithm="grpo", role="actor",
        component=ComponentIdentity(role="actor", model_revision="a" * 40, config_hash="c" * 64),
        syncer_addr=("127.0.0.1", 30100), learner_id=0, learner_generation=0, learner_generations=(0, 0),
        total_fragment_steps=8 * P, expected_fragments=P, expected_layout_hash=LAYOUT_HASH,
        local_horizon=1, optimizer_steps_per_round=1, training_contract_hash="2" * 64,
        syncer_profile_hash="3" * 64, pipeline_depth=P, algorithm_spec_sha256="a1" * 32,
        partial_quorum=True, syncer_epoch=3,
        peer_recovery=PeerRecoveryOptions(store=str(store), recover=recover, timeout_s=timeout_s,
                                          donor_write=donor_write))


def _install_miles_rebase_stub(monkeypatch):
    """Without Miles, stand in for its scheduler-anchor rebase (same rule, no checks)."""
    try:
        import miles.backends.megatron_utils.full_parameter_state  # noqa: F401
        return
    except ImportError:
        pass

    def rebase(actor, *, reason):
        assert reason
        actor._external_full_scheduler_anchor_steps = actor.opt_param_scheduler.num_steps
        return actor._external_full_scheduler_anchor_steps

    for name in ("miles", "miles.backends", "miles.backends.megatron_utils"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    module = types.ModuleType("miles.backends.megatron_utils.full_parameter_state")
    module.rebase_full_parameter_scheduler_anchor = rebase
    monkeypatch.setitem(sys.modules, module.__name__, module)


@pytest.fixture
def stubs(monkeypatch):
    clients = []
    seen_kwargs = []

    def make_client(*args, **kwargs):
        seen_kwargs.append(kwargs)
        return clients.pop(0)

    async def initialize(cls, group, **_kwargs):
        return FakeAdapter(), SimpleNamespace(policy_version=0, policy_hash="p0")

    monkeypatch.setattr(MilesChunkedFullParameterAdapter, "initialize", classmethod(initialize))
    _install_miles_rebase_stub(monkeypatch)
    monkeypatch.setattr("yeto.rl.engine.full_parameter_stream.SyncerClient", make_client)
    monkeypatch.setattr(role_stream_module, "MilesRayInboundChunkSink", lambda sink: sink)
    monkeypatch.setattr(role_stream_module, "role_stream_session_contract_hash", lambda *_a: b"\0" * 32)
    return SimpleNamespace(clients=clients, kwargs=seen_kwargs)


def serve(stream, *, max_polls=10) -> int:
    """Donor safe points until the snapshot is served: the copy (S27 G-EL capture),
    the safe points that find the shards written, and, once the background hash +
    commit finished, the one that reports it; returns the snapshots served."""
    async def run() -> int:
        served = await stream.serve_peer_snapshots()
        for _ in range(max_polls):
            donor = stream._donor
            if donor is None or served or (not donor.busy and not donor._waiting):
                break
            assert donor.background
            donor.wait(10)  # the hash + commit thread, once the shards are written
            served += await stream.serve_peer_snapshots()
        return served

    return asyncio.run(run())


def donor_stream(store_root, group, client, *, donor_write="background"):
    """A running donor: initialized normally, then at fragment versions (3, 4)."""
    stream = MilesFullParameterRoleStream(config(store_root, recover=False, donor_write=donor_write))
    stream.group, stream.adapter, stream.client = group, FakeAdapter(), client
    stream._fragment_versions = [3, 4]
    return stream


@pytest.mark.parametrize("group_type", [FakeGroup, FreshGroup])
def test_donor_snapshot_and_recovering_restore_through_the_role_stream(tmp_path, stubs, group_type):
    store_root = tmp_path / "store"
    donor_group = FakeGroup(adam_step=17)
    request = rp.SnapshotRequest(1, 9, 5, (3, 4))
    donor_client = FakeClient(requests=[request])
    donor = donor_stream(store_root, donor_group, donor_client)
    assert serve(donor) == 1
    (_, versions, sha256, locator), = donor_client.sent
    assert versions == (3, 4)
    # S27 G-EL: the ranks copy at the safe point and write in the background; a later
    # safe point finds them written and collects them.
    assert donor_group.calls == [cut_plugin.SAVE_CUT_SHARD, cut_plugin.RANK_COORDS,
                                 cut_plugin.POLL_CUT_SHARD, cut_plugin.FINISH_CUT_SHARD]

    resent = [BcastFragment(0, 5, stored(0, 5, "g0"), stored=True), BcastFragment(1, 6, stored(1, 6, "g1"),
                                                                                   stored=True)]
    client = FakeClient(events=[rp.RecoveryClock(5, (3, 4), 1),
                                rp.RecoverySource(rp.SOURCE_PEER, 0, versions, sha256, locator),
                                rp.RecoveryDone(5, 6, 1, (5, 6))], updates=resent)
    stubs.clients.append(client)
    group = group_type(adam_step=0)
    stream = MilesFullParameterRoleStream(config(store_root, recover=True))
    asyncio.run(stream.initialize(group))

    assert stubs.kwargs[0]["recover"] is True and stubs.kwargs[0]["syncer_epoch"] == 3
    assert client.sent == [("restored", None)], "the island asks for every fragment"
    assert group.states == donor_group.states, "inner optimizer state of every rank is the donor's"
    assert group.applied == [{0: "g0", 1: "g1"}], "the resent fragments are applied once, as anchors"
    assert stream.fragment_versions == (5, 6) and stream.transport_version == 1
    assert [a.version for a in stream._anchors] == [5, 6]
    assert client.marked and stream.recovery_outcome.path == "peer"
    assert DirectorySnapshotStore(store_root).locators() == [], "a restored snapshot is removed"


def test_island_pool_member_recovers_from_a_peer_while_it_heartbeats(tmp_path, stubs):
    """An island pool fills a slot; the island's state still comes from a peer."""
    store_root = tmp_path / "store"
    donor_client = FakeClient(requests=[rp.SnapshotRequest(1, 9, 5, (3, 4))])
    donor = donor_stream(store_root, FakeGroup(adam_step=17), donor_client)
    assert serve(donor) == 1
    (_, versions, sha256, locator), = donor_client.sent
    client = FakeClient(events=[rp.RecoveryClock(5, (3, 4), 1),
                                rp.RecoverySource(rp.SOURCE_PEER, 0, versions, sha256, locator),
                                rp.RecoveryDone(5, 6, 1, (5, 6))],
                        updates=[BcastFragment(0, 5, stored(0, 5, "g0"), stored=True),
                                 BcastFragment(1, 6, stored(1, 6, "g1"), stored=True)])
    stubs.clients.append(client)
    stream = MilesFullParameterRoleStream(config(store_root, recover=True, island_lease_s=30.0))
    asyncio.run(stream.initialize(FakeGroup(adam_step=0)))
    try:
        assert stubs.kwargs[0]["recover"] is True
        assert stream.heartbeat is not None, "the lease is renewed during and after the restore"
        assert stream.recovery_outcome.path == "peer"
        assert stream.fragment_versions == (5, 6)
    finally:
        stream.heartbeat.stop()


def test_other_tensor_parallel_layout_is_refused(tmp_path, stubs):
    store_root = tmp_path / "store"
    donor_client = FakeClient(requests=[rp.SnapshotRequest(1, 9, 5, (3, 4))])
    serve(donor_stream(store_root, FakeGroup(tp=2), donor_client))
    (_, versions, sha256, locator), = donor_client.sent

    client = FakeClient(events=[rp.RecoverySource(rp.SOURCE_PEER, 0, versions, sha256, locator)])
    stubs.clients.append(client)
    group = FakeGroup(tp=1, dp=2)
    stream = MilesFullParameterRoleStream(config(store_root, recover=True, timeout_s=0.5))
    with pytest.raises(TimeoutError):
        asyncio.run(stream.initialize(group))
    assert client.sent[0][0] == "failed" and "layout" in client.sent[0][1]
    assert cut_plugin.RESTORE_CUT_SHARD not in group.calls and client.closed


class DigestGroup(FakeGroup):
    """FakeGroup whose save and restore report the cut summary digests, as the
    real cut_plugin does; ``tamper`` makes one rank restore other state."""

    def __init__(self, *, tamper=False, **kw):
        super().__init__(**kw)
        self.tamper = tamper

    def _digests(self, state):
        body = json.dumps(state, sort_keys=True).encode()
        return {"state_digest": hashlib.sha256(body).hexdigest(), "rng_digest": "rng",
                "scheduler_samples": state["scheduler_steps"]}

    async def run_plugin(self, name, kwargs):
        out = await super().run_plugin(name, kwargs)
        if name == cut_plugin.SAVE_CUT_SHARD:
            return [{**r, **self._digests(state)} for r, state in zip(out, self.states, strict=True)]
        if name == cut_plugin.RESTORE_CUT_SHARD:
            if self.tamper:
                self.states[-1] = {**self.states[-1], "m": [9.0] * 3}
            return [{"path": cut_plugin.shard_name(c), **self._digests(state)}
                    for c, state in zip(self.coords, self.states, strict=True)]
        return out


@pytest.mark.parametrize("tamper", [False, True])
def test_restore_checks_every_rank_digest_against_the_donor(tmp_path, capsys, monkeypatch, tamper):
    _install_miles_rebase_stub(monkeypatch)
    donor = DigestGroup(adam_step=17)
    asyncio.run(MilesRankShards(donor, LAYOUT_HASH).save(tmp_path))
    ranks = json.loads((tmp_path / RANKS).read_text())
    assert all({"state_digest", "rng_digest", "scheduler_samples"} <= set(f) for f in ranks["files"])
    donor_line = capsys.readouterr().err
    assert "donor snapshot: 2 rank shard(s)" in donor_line
    fresh = DigestGroup(adam_step=0, tamper=tamper)
    shards = MilesRankShards(fresh, LAYOUT_HASH)
    if tamper:
        with pytest.raises(RuntimeError, match="not the donor's state.*state_digest"):
            asyncio.run(shards.restore(tmp_path))
        return
    asyncio.run(shards.restore(tmp_path))
    restored_line = capsys.readouterr().err
    assert "restored 2/2 rank shard(s)" in restored_line
    assert donor_line.split("digests ")[-1].strip() == restored_line.split("digests ")[-1].strip()


def test_rank_shard_on_another_node_refuses_the_snapshot(tmp_path):
    group = FakeGroup(tp=2, missing_rank=None)
    group.missing_rank = group.coords[1]
    shards = MilesRankShards(group, LAYOUT_HASH)
    with pytest.raises(SnapshotRefused, match="node-shared"):
        asyncio.run(shards.save(tmp_path))
    assert not (tmp_path / RANKS).exists()


def test_layout_tag_names_the_parallel_layout():
    group = FakeGroup(tp=2, dp=2)
    layout = parallel_layout(group.coords)
    assert layout == {"world": 4, "tp": 2, "pp": 1, "cp": 1, "ep": 1, "dp": 2}
    assert layout_tag(LAYOUT_HASH, layout) != layout_tag(LAYOUT_HASH, {**layout, "tp": 4, "dp": 1})
    assert layout_tag(LAYOUT_HASH, layout) != layout_tag("cd" * 32, layout)


def test_peer_recovery_needs_asynchronous_rounds(tmp_path):
    base = config(tmp_path, recover=False).__dict__
    with pytest.raises(ValueError, match="asynchronous rounds"):
        RoleStreamConfig(**{**base, "partial_quorum": False, "pipeline_depth": 1})
    with pytest.raises(ValueError, match="asynchronous rounds"):
        RoleStreamConfig(**{**base, "bcast_delta": True})


def test_streams_without_peer_recovery_serve_nothing(tmp_path, stubs):
    stream = MilesFullParameterRoleStream(RoleStreamConfig(**{**config(tmp_path, recover=False).__dict__,
                                                              "peer_recovery": None}))
    assert asyncio.run(stream.serve_peer_snapshots()) == 0


def test_restore_without_scheduler_rebase_fails_the_first_apply(tmp_path, stubs, monkeypatch):
    """G-D2 rec-c: restore OK and digests equal, but Miles refused the first apply."""
    store_root = tmp_path / "store"
    donor_client = FakeClient(requests=[rp.SnapshotRequest(1, 9, 5, (3, 4))])
    serve(donor_stream(store_root, FakeGroup(adam_step=17), donor_client))
    (_, versions, sha256, locator), = donor_client.sent
    resent = [BcastFragment(0, 5, stored(0, 5, "g0"), stored=True),
              BcastFragment(1, 6, stored(1, 6, "g1"), stored=True)]
    stubs.clients.append(FakeClient(events=[rp.RecoveryClock(5, (3, 4), 1),
                                            rp.RecoverySource(rp.SOURCE_PEER, 0, versions, sha256, locator),
                                            rp.RecoveryDone(5, 6, 1, (5, 6))], updates=resent))
    monkeypatch.setattr(cut_plugin, "rebase_full_parameter_tracking", lambda actor: None)
    with pytest.raises(Exception, match="no local-step receipt"):
        asyncio.run(MilesFullParameterRoleStream(config(store_root, recover=True)).initialize(FakeGroup()))


# -- S27 G-EL: the donor copies at the safe point and writes while it trains ----


@pytest.mark.parametrize("donor_write", ["background", "inline"])
def test_donor_copies_at_the_safe_point_and_writes_while_training(tmp_path, stubs, donor_write):
    """Background (default): the safe point only starts the ranks' host-copy
    writes; later safe points ask (never wait) and the snapshot is answered once
    every shard is written. Inline (the behavior before): one SAVE at the safe
    point writes the shards."""
    store_root = tmp_path / "store"
    group = FakeGroup(adam_step=17)
    group.write_polls = 3  # the shards take three safe points to write
    client = FakeClient(requests=[rp.SnapshotRequest(1, 9, 5, (3, 4))])
    donor = donor_stream(store_root, group, client, donor_write=donor_write)

    if donor_write == "inline":
        assert serve(donor) == 1
        assert group.calls == [cut_plugin.SAVE_CUT_SHARD, cut_plugin.RANK_COORDS]
        assert "background_write_s" not in donor._donor.timings[0]
        return

    async def safe_points():
        early = [await donor.serve_peer_snapshots() for _ in range(3)]
        assert early == [0, 0, 0] and client.sent == [], "not answered before every shard is written"
        assert group.calls == [cut_plugin.SAVE_CUT_SHARD, cut_plugin.RANK_COORDS,
                               cut_plugin.POLL_CUT_SHARD, cut_plugin.POLL_CUT_SHARD]
        assert donor._donor.busy and donor._donor._capturing is not None
        served = 0
        for _ in range(5):
            served += await donor.serve_peer_snapshots()
            donor._donor.wait(10)
        return served

    assert asyncio.run(safe_points()) == 1
    assert group.calls[-2:] == [cut_plugin.POLL_CUT_SHARD, cut_plugin.FINISH_CUT_SHARD]
    (_, versions, sha256, locator), = client.sent
    assert versions == (3, 4)
    timing, = donor._donor.timings
    assert timing["result"] == "served" and timing["background_write_s"] >= 0
    ranks = json.loads((store_root / "peer-snapshots" / locator / "files" / RANKS).read_text())
    assert [f["path"] for f in ranks["files"]] == [cut_plugin.shard_name(c) for c in group.coords]


def test_a_rank_refusing_the_background_copy_refuses_the_snapshot(tmp_path, stubs):
    class RefusingGroup(FakeGroup):
        async def run_plugin(self, name, kwargs):
            if name == cut_plugin.SAVE_CUT_SHARD and kwargs.get("background"):
                self.calls.append(name)
                return [{"pending": kwargs["cut_id"], "coord": self.coords[0]},
                        {cut_plugin.REFUSED: "not at a step boundary", cut_plugin.REFUSAL_KIND: "refused"}]
            if name == cut_plugin.FINISH_CUT_SHARD:
                self.calls.append(name)
                return [{"path": "x"}]
            return await super().run_plugin(name, kwargs)

    group = RefusingGroup()
    client = FakeClient(requests=[rp.SnapshotRequest(1, 9, 5, (3, 4))])
    donor = donor_stream(tmp_path / "store", group, client)
    assert serve(donor) == 0 and client.sent == []
    (_, reason), = donor._donor.refused
    assert "not at a step boundary" in reason
    # The rank that did start writing was drained before the refusal.
    assert group.calls == [cut_plugin.SAVE_CUT_SHARD, cut_plugin.FINISH_CUT_SHARD]
