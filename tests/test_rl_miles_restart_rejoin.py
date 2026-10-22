"""S27 E-4c: a restarted full-parameter island joins a running pool through
the real Miles role stream. Real Rust syncer (asynchronous rounds, island
lease); the Megatron trainer group and Ray are CPU fakes.

Learner 1 trains a few rounds as a raw client, then its process dies (the
sockets close, no LEAVE). A new process for learner 1 starts the real
``MilesFullParameterRoleStream`` from the checkpoint weights. It must not
send INIT, must take the syncer's catch-up (every fragment at its current
version) as its trainer weights and anchors, push deltas measured from those
anchors that the syncer accepts, and acknowledge the final cut.

Needs Miles (``importorskip``) and cargo.
"""

from __future__ import annotations

# ruff: noqa: I001 -- the Miles test helpers install a bridge stub before import.

import asyncio
import json
import threading
import time
from dataclasses import replace

import numpy as np
import pytest
import torch

pytest.importorskip("miles")

from test_e3_overlap_miles_role_stream import SIZES, _Driver, _Trainer, _adapter  # noqa: E402
from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: E402,F401
from test_rl_full_parameter_async import (  # noqa: E402,F401
    P,
    _hmac_key,
    _start_async_syncer,
    _wait,
    needs_cargo,
    syncer_binary,
)
from test_rl_full_parameter_sync import _role_stream_config  # noqa: E402
from test_rl_miles_chunked_full_parameter import FakeRay, _evidence, component  # noqa: E402

from yeto.protocol import DTYPE_F32, SyncerClient  # noqa: E402
from yeto.rl.adapters.miles.full_parameter import role_stream as role_stream_module  # noqa: E402
from yeto.rl.adapters.miles.full_parameter.role_stream import (  # noqa: E402
    MilesFullParameterRoleStream,
    role_stream_session_contract_hash,
)
from yeto.rl.adapters.miles.models.chunked_full_parameter import (  # noqa: E402
    MilesChunkedFullParameterAdapter,
)
from yeto.rl.engine.full_parameter_sync import (  # noqa: E402
    FullParameterSync,
    FullParameterSyncProfile,
    RolePairing,
)

LEASE_S = 3.0
POOL = ("--rl-sync-quorum", "1", "--rl-sync-island-lease-s", str(LEASE_S))
ROUNDS = 8  # outer versions per fragment: total steps


def _initial() -> dict[int, torch.Tensor]:
    return {f: torch.arange(n, dtype=torch.float32) + float(f) for f, n in enumerate(SIZES)}


class _FreshTrainer(_Trainer):
    """Like Megatron: no optimizer-state proof before the first local step
    (s27-rj-a: "full-parameter last rollout ID is unavailable")."""

    async def full_parameter_optimizer_states(self):
        if not self.trained_from:
            raise RuntimeError("full-parameter last rollout ID is unavailable")
        return await super().full_parameter_optimizer_states()


def _raw_delta(learner: int, step: int, fid: int) -> np.ndarray:
    return np.full(SIZES[fid], 0.01 * (learner + 1) + 0.001 * step, dtype=np.float32)


class _RawIsland(threading.Thread):
    """A raw-client island: answers every PULL against its newest version."""

    def __init__(self, client: SyncerClient, learner: int) -> None:
        super().__init__(daemon=True)
        self.client, self.learner = client, learner
        self.versions = [0] * P
        self.seen: set = set()
        self.halt = threading.Event()
        self.error: BaseException | None = None

    def step(self) -> None:
        self.client.check_health()
        for update in self.client.drain_updates():
            self.versions[update.fragment_id] = max(self.versions[update.fragment_id], update.version)
        for permit in self.client.drain_pulls():
            key = (permit.global_step, permit.round_attempt)
            if key in self.seen:
                continue
            self.seen.add(key)
            self.client.push_fragment(
                permit.fragment_id, permit.global_step, permit.round_attempt,
                self.versions[permit.fragment_id], permit.global_step, 1, 7,
                _raw_delta(self.learner, permit.global_step, permit.fragment_id).tobytes())

    def run(self) -> None:
        try:
            while not self.halt.is_set() and not self.client.finalizing.is_set():
                self.step()
                self.halt.wait(0.005)
        except BaseException as error:  # noqa: BLE001 - reported by the test
            self.error = error


def _stream_config(port: int, profile_hash: str, adapter):
    return replace(
        _role_stream_config("actor", port),
        component=component(), learner_id=1, learner_generation=0, learner_generations=(0, 0),
        total_fragment_steps=ROUNDS * P, expected_fragments=P,
        expected_layout_hash=adapter.layout.layout_hash, syncer_profile_hash=profile_hash,
        pipeline_depth=2, wait_timeout=60.0, poll_seconds=0.005,
        max_fragment_bytes=64, max_chunk_bytes=8, partial_quorum=True, island_lease_s=LEASE_S,
    )


def _raw_client(learner, port, profile_hash, adapter, contract) -> SyncerClient:
    return SyncerClient(("127.0.0.1", port), learner, adapter.layout.fragments, dtype=DTYPE_F32,
                        num_streams=1, connect_timeout=10, max_reconnects=0, finalization_timeout=60,
                        session_contract_hash=contract, syncer_profile_hash=bytes.fromhex(profile_hash))


@needs_cargo
@pytest.mark.usefixtures("_hmac_key")
@pytest.mark.parametrize("death", ["closed", "silent"])
def test_restarted_miles_island_catches_up_trains_and_acknowledges(syncer_binary, tmp_path, monkeypatch,
                                                                    death):
    """``closed``: the old sockets close (the syncer sees the departure).
    ``silent``: the old connection stays open and mute (a lost host); the new
    generation supersedes it before its lease runs out."""
    process, port, profile_hash = _start_async_syncer(syncer_binary, tmp_path, *POOL,
                                                      total_steps=ROUNDS)
    adapter = _adapter()
    config = _stream_config(port, profile_hash, adapter)
    contract = role_stream_session_contract_hash(adapter.layout.layout_hash, config)
    initial = _initial()
    islands = {learner: _RawIsland(_raw_client(learner, port, profile_hash, adapter, contract), learner)
               for learner in (0, 1)}
    stream = dead = None
    try:
        for island in islands.values():
            island.client.start()
        for fid in range(P):
            islands[0].client.send_init(fid, initial[fid].numpy().tobytes())
        for island in islands.values():
            island.start()
        _wait(lambda: min(islands[0].versions) >= 2, "the pool never reached version 2", timeout=30)
        # Learner 1's process dies without a LEAVE.
        islands[1].halt.set()
        islands[1].join(timeout=5)
        dead = islands[1].client
        if death == "closed":
            dead.close()
        died_at = list(islands[0].versions)

        # The new learner-1 process starts the Miles role stream from the
        # checkpoint weights (the run's initial weights).
        ray = FakeRay()
        trainer = _FreshTrainer(adapter, ray, initial, lambda _k: {f: torch.full((n,), 0.05)
                                                              for f, n in enumerate(SIZES)})

        async def initialize(cls, group, *, policy_version, **_kwargs):
            cut = await group.export_full_parameter_chunked_cut(policy_version, 0,
                                                                max_chunk_bytes=adapter.max_chunk_bytes)
            return adapter, adapter.validate_cut(cut, policy_version, 0)

        monkeypatch.setattr(MilesChunkedFullParameterAdapter, "initialize", classmethod(initialize))
        sent_init = []
        real_client = role_stream_module.SyncerClient

        def client_factory(*args, **kwargs):
            client = real_client(*args, **kwargs)
            original = client.send_initial_fragments
            client.send_initial_fragments = lambda parts: sent_init.append(1) or original(parts)
            return client

        monkeypatch.setattr(role_stream_module, "SyncerClient", client_factory)
        stream = MilesFullParameterRoleStream(config, ray_module=ray)
        profile = FullParameterSyncProfile.from_configs({"actor": config})
        session = FullParameterSync(
            RolePairing({"actor": stream}, profile), groups={"actor": trainer}, run=asyncio.run,
            trajectories=lambda rollout_id: _evidence(rollout_id=rollout_id,
                                                      behavior_policy_hash=stream.current.policy_hash))
        driver = _Driver()
        session.start(driver)

        caught = stream.fragment_versions
        assert not sent_init, "a non-seed island must not send INIT"
        assert all(v >= d for v, d in zip(caught, died_at, strict=True)) and min(caught) >= 2
        for fid in range(P):
            # The trainer holds the pool's cut, not the checkpoint weights.
            assert not torch.equal(trainer.values[fid], initial[fid])
            assert stream._anchors[fid].version == caught[fid]

        stop = False
        deadline = time.monotonic() + 60
        rollout_id = -1
        while time.monotonic() < deadline:
            rollout_id += 1
            boundary = session.boundary(driver, rollout_id=rollout_id, stats=None)
            if boundary.stop:
                stop = True
                session.published(driver, rollout_id=rollout_id,
                                  policy_hash=boundary.state.policy_tensor_hash())
                break
        assert stop, "the restarted island never reached the final cut"
        session.finish(driver)
        assert stream.final_applied and stream.final_acknowledged
        islands[0].join(timeout=30)
        assert islands[0].error is None, islands[0].error
        manifest, finals = islands[0].client.wait_for_final_fragments(timeout=30)
        islands[0].client.acknowledge_finalization(manifest)
        for fragment in finals:
            final = np.frombuffer(bytes(fragment.data), dtype=np.float32)[-SIZES[fragment.fragment_id]:]
            np.testing.assert_array_equal(trainer.values[fragment.fragment_id].numpy(), final)
        assert process.wait(timeout=20) == 0
    finally:
        for island in islands.values():
            island.halt.set()
            island.client.close()
        if stream is not None:
            stream.close()
        if dead is not None:
            dead.close()
        if process.poll() is None:
            process.terminate()
        log, _ = process.communicate(timeout=10)
    records = [json.loads(line) for line in (tmp_path / "tape.jsonl").read_text().splitlines() if line]
    island_1 = [r for r in records if r.get("island_id") == 1]
    rejoin = next(i for i, r in enumerate(island_1) if r.get("kind") == "pool_join" and r.get("catch_up"))
    after = island_1[rejoin + 1:]
    assert any(r["kind"] == "delta_accepted" for r in after), "the restarted island never contributed"
    # Quorum 1 closes some rounds before the island answers (out_of_round);
    # nothing is refused for its base after the catch-up.
    assert {r.get("reason") for r in after if r["kind"] == "delta_rejected"} <= {"out_of_round"}, after
    assert "predates the re-JOIN catch-up" not in log


def test_initial_wait_stops_when_the_pool_finished_first(monkeypatch):
    """s27-rj-a: a restarted island that dials after the pool's final cut
    must not wait for broadcasts until the 900 s timeout."""
    from test_rl_full_parameter_sync import _role_stream_config as config_of
    from types import SimpleNamespace

    from yeto.rl.adapters.miles.full_parameter.role_stream import PoolFinishedBeforeJoin

    finalizing = threading.Event()
    finalizing.set()
    client = SimpleNamespace(check_health=lambda: None, drain_updates=lambda: [],
                             finalizing=finalizing, send_progress=lambda: None)
    config = replace(config_of("actor", 30100), partial_quorum=True, island_lease_s=60.0)
    adapter = SimpleNamespace(layout=SimpleNamespace(fragments=SimpleNamespace(num_fragments=2)))
    with pytest.raises(PoolFinishedBeforeJoin):
        MilesFullParameterRoleStream(config)._wait_for_initial_fragments(client, adapter)
