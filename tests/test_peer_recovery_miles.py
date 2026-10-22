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
import threading
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
        self.states = [{"adam_step": adam_step, "m": [0.25 * r] * 3} for r in range(tp * dp)]
        self.missing_rank = missing_rank
        self.installed = 0
        self.generation = 0
        self.applied = []
        self.calls = []

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
            return out
        if name == cut_plugin.RESTORE_CUT_SHARD:
            by_path = {f["path"]: f for f in kwargs["files"]}
            for rank, coord in enumerate(self.coords):
                entry = by_path[cut_plugin.shard_name(coord)]
                data = (Path(kwargs["directory"]) / entry["path"]).read_bytes()
                assert hashlib.sha256(data).hexdigest() == entry["sha256"]
                self.states[rank] = json.loads(data)
            return [{"ok": True} for _ in self.coords]
        raise AssertionError(name)

    async def full_parameter_optimizer_states(self):
        return tuple(SimpleNamespace(role="actor", installed_policy_version=self.installed,
                                     local_step_generation=self.generation, adam_step=s["adam_step"])
                     for s in self.states)


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


def config(store, *, recover, timeout_s=10.0):
    return RoleStreamConfig(
        algorithm="grpo", role="actor",
        component=ComponentIdentity(role="actor", model_revision="a" * 40, config_hash="c" * 64),
        syncer_addr=("127.0.0.1", 30100), learner_id=0, learner_generation=0, learner_generations=(0, 0),
        total_fragment_steps=8 * P, expected_fragments=P, expected_layout_hash=LAYOUT_HASH,
        local_horizon=1, optimizer_steps_per_round=1, training_contract_hash="2" * 64,
        syncer_profile_hash="3" * 64, pipeline_depth=P, algorithm_spec_sha256="a1" * 32,
        partial_quorum=True, syncer_epoch=3,
        peer_recovery=PeerRecoveryOptions(store=str(store), recover=recover, timeout_s=timeout_s))


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
    monkeypatch.setattr(role_stream_module, "SyncerClient", make_client)
    monkeypatch.setattr(role_stream_module, "MilesRayInboundChunkSink", lambda sink: sink)
    monkeypatch.setattr(role_stream_module, "role_stream_session_contract_hash", lambda *_a: b"\0" * 32)
    return SimpleNamespace(clients=clients, kwargs=seen_kwargs)


def donor_stream(store_root, group, client):
    """A running donor: initialized normally, then at fragment versions (3, 4)."""
    stream = MilesFullParameterRoleStream(config(store_root, recover=False))
    stream.group, stream.adapter, stream.client = group, FakeAdapter(), client
    stream._fragment_versions = [3, 4]
    return stream


def test_donor_snapshot_and_recovering_restore_through_the_role_stream(tmp_path, stubs):
    store_root = tmp_path / "store"
    donor_group = FakeGroup(adam_step=17)
    request = rp.SnapshotRequest(1, 9, 5, (3, 4))
    donor_client = FakeClient(requests=[request])
    donor = donor_stream(store_root, donor_group, donor_client)
    assert asyncio.run(donor.serve_peer_snapshots()) == 1
    (_, versions, sha256, locator), = donor_client.sent
    assert versions == (3, 4)
    assert donor_group.calls == [cut_plugin.SAVE_CUT_SHARD, cut_plugin.RANK_COORDS]

    resent = [BcastFragment(0, 5, stored(0, 5, "g0"), stored=True), BcastFragment(1, 6, stored(1, 6, "g1"),
                                                                                   stored=True)]
    client = FakeClient(events=[rp.RecoveryClock(5, (3, 4), 1),
                                rp.RecoverySource(rp.SOURCE_PEER, 0, versions, sha256, locator),
                                rp.RecoveryDone(5, 6, 1, (5, 6))], updates=resent)
    stubs.clients.append(client)
    group = FakeGroup(adam_step=0)
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


def test_other_tensor_parallel_layout_is_refused(tmp_path, stubs):
    store_root = tmp_path / "store"
    donor_client = FakeClient(requests=[rp.SnapshotRequest(1, 9, 5, (3, 4))])
    asyncio.run(donor_stream(store_root, FakeGroup(tp=2), donor_client).serve_peer_snapshots())
    (_, versions, sha256, locator), = donor_client.sent

    client = FakeClient(events=[rp.RecoverySource(rp.SOURCE_PEER, 0, versions, sha256, locator)])
    stubs.clients.append(client)
    group = FakeGroup(tp=1, dp=2)
    stream = MilesFullParameterRoleStream(config(store_root, recover=True, timeout_s=0.5))
    with pytest.raises(TimeoutError):
        asyncio.run(stream.initialize(group))
    assert client.sent[0][0] == "failed" and "layout" in client.sent[0][1]
    assert cut_plugin.RESTORE_CUT_SHARD not in group.calls and client.closed


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
