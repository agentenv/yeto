"""Full-parameter islands join and leave a running island pool. Real Rust syncer, no Ray.

The launcher accepts ``--rl-sync-max-learners`` with the full-parameter preset:
the syncer admits ids past the launched islands (``--max-learners``) and every
island binds the roster ``0..N-1`` into its session contract
(``--sync-max-learners``), so an island launched after the start offers the
contract of the launched ones. A two-island run admits a third island midway:
it catches up to the newest outer weights before it merges anything, then
contributes; one launched island then leaves (LEAVE) and the rounds merge over
the two remaining islands. The final cut, and the catch-up values, equal an
f32 numpy recomputation of every merged round (Nesterov), within 1e-6.

The launched islands are plain clients (no heartbeats, no syncer epoch, no
membership messages): an island that predates dynamic membership keeps
training in the pool as before.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from test_rl_algorithm_provenance import _fake_sky, _no_modal_listing  # noqa: F401
from test_rl_full_parameter_async import (  # noqa: F401
    NUMEL,
    P,
    _delta,
    _f32,
    _hmac_key,
    _initial,
    _layout,
    _start_async_syncer,
    _wait,
    needs_cargo,
    syncer_binary,
)

LAUNCHED = 2
MAX_LEARNERS = 3
# The capacity quorum θ = 1.0 follows the islands in the pool: a round needs every
# island it expects (2, then 3 after the join, then 2 after the leave), at most
# --rl-sync-max-learners of them. A base lag of 1 is admitted (pipelined rounds);
# the soft deadline only bounds a hang of a broken run.
POOL = ("--rl-sync-quorum-theta", "1.0", "--rl-sync-max-base-lag", "1", "--rl-sync-island-lease-s", "5",
        "--rl-sync-soft-deadline-s", "30", "--rl-sync-max-learners", str(MAX_LEARNERS))
LR, MU = 0.7, 0.9  # the launcher's outer Nesterov step of the async rounds


def _stream_config(learner_id: int, max_learners: int | None):
    from yeto.rl.adapters.miles.full_parameter.role_stream import RoleStreamConfig
    from yeto.rl.engine.full_parameter_stream import roster_generations
    from yeto.rl.local_learner import ComponentIdentity

    generations = roster_generations(learner_id, LAUNCHED, 0, max_learners)
    return RoleStreamConfig(
        algorithm="grpo", role="actor", component=ComponentIdentity("actor", "a" * 40, "c" * 64),
        syncer_addr=("127.0.0.1", 30100), learner_id=learner_id, learner_generation=0,
        learner_generations=generations, total_fragment_steps=4, expected_fragments=2,
        expected_layout_hash="1" * 64, local_horizon=1, optimizer_steps_per_round=1,
        training_contract_hash="2" * 64, syncer_profile_hash="3" * 64, pipeline_depth=2,
        algorithm_spec_sha256="a1" * 32, partial_quorum=True)


def _contract(learner_id: int, max_learners: int | None = MAX_LEARNERS) -> bytes:
    from yeto.rl.engine.full_parameter_stream import session_contract_hash

    return session_contract_hash("1" * 64, _stream_config(learner_id, max_learners),
                                 backend_identity_sha256="b" * 64)


def test_without_max_learners_the_contract_binds_the_launched_roster_as_before():
    from yeto.rl.engine.full_parameter_stream import roster_generations

    assert roster_generations(1, 2) == (0, 0) and roster_generations(1, 2, 3) == (0, 3)
    assert _stream_config(1, None).learner_generations == (0, 0)  # every existing run
    with pytest.raises(ValueError, match="outside the island roster 0..1"):
        roster_generations(2, 2)
    with pytest.raises(ValueError, match="below --num-learners"):
        roster_generations(0, 3, 0, 2)


def test_a_pool_binds_every_admissible_id_so_a_later_island_offers_the_same_contract():
    contracts = {_contract(learner) for learner in range(MAX_LEARNERS)}
    assert len(contracts) == 1  # the launched islands and island 2 share one session
    assert _contract(0) != _contract(0, None)  # the pool roster is part of the contract
    with pytest.raises(ValueError, match="outside the island roster 0..2"):
        _stream_config(3, MAX_LEARNERS)


def test_the_islands_get_the_pool_roster_and_refuse_it_without_a_pool():
    from types import SimpleNamespace

    from yeto.rl.adapters.miles.island_entry import _roster_generations

    pooled = SimpleNamespace(learner_id=2, num_learners=2, learner_generation=0,
                             sync_max_learners=3, rl_island_pool=True)
    assert _roster_generations(pooled) == (0, 0, 0)
    with pytest.raises(ValueError, match="needs an island pool"):
        _roster_generations(SimpleNamespace(**{**vars(pooled), "rl_island_pool": False}))
    launched = SimpleNamespace(learner_id=1, num_learners=2, learner_generation=0)
    assert _roster_generations(launched) == (0, 0)


def test_the_verl_island_binds_the_same_pool_roster():
    from yeto.rl.adapters.verl.full_parameter_stream import role_stream_config

    section = {"async_rounds": True, "num_learners": 2, "learner_generation": 0, "max_learners": 3,
               "total_fragment_steps": 4, "fragments": 2, "parameter_layout_sha256": "1" * 64,
               "local_horizon": 1, "training_contract_sha256": "2" * 64,
               "syncer_profile_sha256": "3" * 64, "pipeline": 2, "algorithm_spec_sha256": "a1" * 32,
               "wait_timeout_s": 900.0, "wan_streams": 1, "model_config_sha256": "c" * 64,
               "island_pool": {"lease_s": 5.0, "syncer_epoch": 0}, "push_dtype": "f32",
               "max_fragment_bytes": 2 << 30, "final_cut": "download", "local_init": True,
               "matrix_merge": "avg"}
    config = role_stream_config(section, syncer="127.0.0.1:30100", learner_id=2, model_revision="a" * 40)
    assert config.learner_generations == (0, 0, 0)
    launched = role_stream_config({**section, "max_learners": None}, syncer="127.0.0.1:30100",
                                  learner_id=1, model_revision="a" * 40)
    assert launched.learner_generations == (0, 0)


def _client(learner_id: int, port: int, profile_hash: str):
    from yeto.protocol import DTYPE_F32, SyncerClient

    return SyncerClient(("127.0.0.1", port), learner_id, _layout(), dtype=DTYPE_F32,
                        num_streams=1, connect_timeout=10, max_reconnects=0,
                        finalization_timeout=30, syncer_profile_hash=bytes.fromhex(profile_hash),
                        session_contract_hash=_contract(learner_id))


def _values(data) -> np.ndarray:
    return np.frombuffer(bytes(data), dtype=np.float32)[-NUMEL:].copy()


def _outer_steps(tape: str) -> list[dict]:
    """The merged rounds; each weight entry gains the step its delta was pushed for
    (a late delta carried into a later round of its fragment keeps its own step)."""
    records = [json.loads(line) for line in tape.splitlines() if line.strip()]
    carried = {(r["island_id"], r["step"]): r["missed_step"]
               for r in records if r.get("kind") == "delta_carried_over"}
    steps = sorted((r for r in records if r.get("kind") == "outer_step"), key=lambda r: r["step"])
    for record in steps:
        # a carried delta is merged against its own (older) base, next to the island's push
        record["pushed_for"] = [
            carried[(int(learner), record["step"])] if int(base) < record["base_version"] else record["step"]
            for learner, base, _ in record["raw_weights"]]
    return steps


def _nesterov(steps: list[dict], *, upto: int | None = None) -> dict[int, np.ndarray]:
    """The syncer's outer step in f32 over the merged rounds of the tape (in its
    merge order, with its raw weights), up to global step ``upto``."""

    lr32, mu32 = np.float32(LR), np.float32(MU)
    params = {fid: _initial(fid) for fid in range(P)}
    buffers = {fid: np.zeros(NUMEL, dtype=np.float32) for fid in range(P)}
    for record in steps:
        if upto is not None and record["step"] > upto:
            break
        fid = record["fragment"]
        contributions = [(int(learner), float(weight), step) for (learner, _, weight), step
                         in zip(record["raw_weights"], record["pushed_for"], strict=True)]
        total = sum(weight for _, weight, _ in contributions)
        merged = np.zeros(NUMEL, dtype=np.float32)
        for learner, weight, step in contributions:  # the syncer's order: pushes, then carried deltas
            merged += np.float32(weight / total) * -_delta(learner, step)
        buffers[fid] = mu32 * buffers[fid] + merged
        params[fid] = params[fid] - lr32 * (merged + mu32 * buffers[fid])
    return params


@needs_cargo
@pytest.mark.usefixtures("_hmac_key")
def test_an_island_joins_after_catching_up_and_another_leaves_mid_run(syncer_binary, tmp_path):
    total_steps, join_at = 10, 3
    log_file = tmp_path / "syncer.log"
    process, port, profile_hash = _start_async_syncer(syncer_binary, tmp_path, *POOL,
                                                      total_steps=total_steps, log_file=log_file)
    tape_path = tmp_path / "tape.jsonl"
    command = " ".join(process.args)
    assert f"--learners {LAUNCHED} " in command and "--quorum-theta 1.0 " in command
    assert f"--max-learners {MAX_LEARNERS} " in command
    clients = {learner: _client(learner, port, profile_hash) for learner in range(LAUNCHED)}
    versions = {learner: [0] * P for learner in range(MAX_LEARNERS)}
    seen: dict[int, set] = {learner: set() for learner in range(MAX_LEARNERS)}
    # learner 0's broadcast values by (fragment, version): the newest outer weights
    observed: dict[tuple[int, int], np.ndarray] = {}
    held: dict[int, list] = {}  # PULLs of island 2 that arrived during its catch-up
    state = {"joined": None, "catch_up": {}, "first_base": None, "left": False}

    def drain(learner):
        client = clients[learner]
        client.check_health()
        for update in client.drain_updates():
            fid = update.fragment_id
            versions[learner][fid] = max(versions[learner][fid], update.version)
            if learner == 0:
                observed[(fid, update.version)] = _values(update.data)
        for permit in held.pop(learner, []) + client.drain_pulls():
            key = (permit.global_step, permit.round_attempt)
            if key in seen[learner]:
                continue
            seen[learner].add(key)
            base = versions[learner][permit.fragment_id]
            if learner == 2 and state["first_base"] is None:
                state["first_base"] = (permit.fragment_id, base)
            client.push_fragment(permit.fragment_id, permit.global_step, permit.round_attempt,
                                 base, permit.global_step, 1, 7, _f32(_delta(learner, permit.global_step)))

    def join():
        """Island 2 (an id past the launch) joins: no INIT, the syncer's catch-up first."""
        from yeto.launch.learner_ids import allocate_learner_id
        from yeto.rl.engine.object_store import DirectoryObjectStore

        newest = list(versions[0])
        # its id comes from the run store's island table, past the launched ones
        assert allocate_learner_id(DirectoryObjectStore(tmp_path / "store"), launched=LAUNCHED,
                                   max_learners=MAX_LEARNERS, island="joiner") == 2
        fresh = _client(2, port, profile_hash)
        fresh.start()
        caught: dict[int, tuple[int, np.ndarray]] = {}

        def catch_up():
            fresh.check_health()
            for update in fresh.drain_updates():
                if update.fragment_id not in caught:
                    assert update.catch_up or update.version >= newest[update.fragment_id]
                    caught[update.fragment_id] = (update.version, _values(update.data))
            # A PULL travels on the control link and may overtake the catch-up
            # bytes; the island applies its broadcasts before it answers (the
            # role stream's boundary), so it holds the PULL until then.
            held.setdefault(2, []).extend(fresh.drain_pulls())
            return len(caught) == P

        _wait(catch_up, "the joining island got no catch-up", timeout=30)
        clients[2] = fresh
        versions[2] = [caught[fid][0] for fid in range(P)]
        state.update(joined=newest, catch_up=caught)

    def merged_with_three() -> bool:  # three islands online for several rounds
        return sum(record["arrived"] == 3 for record in _outer_steps(tape_path.read_text())) >= 4

    def poll():
        for learner in list(clients):
            drain(learner)
        if state["joined"] is None and min(versions[0]) >= join_at:
            join()
        if state["joined"] is not None and not state["left"] and merged_with_three():
            # island 1 leaves once a round merged all three: the rounds then need 0 and 2
            clients[1].send_leave(2)  # stopped: island 1 leaves the pool on purpose
            clients.pop(1).close()
            state["left"] = True
        return state["left"] and all(client.finalizing.is_set() for client in clients.values())

    finals = {}
    try:
        for client in clients.values():
            client.start()
        for fid in range(P):
            clients[0].send_init(fid, _f32(_initial(fid)))
        _wait(poll, "syncer never finalized", timeout=120)
        for learner, client in clients.items():
            manifest, fragments = client.wait_for_final_fragments(timeout=30)
            finals[learner] = {f.fragment_id: _values(f.data) for f in fragments}
            client.acknowledge_finalization(manifest)
        assert process.wait(timeout=15) == 0
    finally:
        for client in clients.values():
            client.close()
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=10)
    tape = tape_path.read_text()
    steps = _outer_steps(tape)
    assert [record["step"] for record in steps] == list(range(1, total_steps * P + 1))

    # The joiner started from the newest outer weights: every fragment at least at
    # the version island 0 held when island 2 dialed, equal to island 0's
    # broadcast of that version and to the Nesterov recomputation.
    for fid, (version, values) in state["catch_up"].items():
        assert version >= max(state["joined"][fid], join_at)
        if (fid, version) in observed:
            assert np.array_equal(values, observed[(fid, version)])
        np.testing.assert_allclose(values, _nesterov(steps, upto=version)[fid], rtol=0, atol=1e-6)
    fid, base = state["first_base"]
    assert base == state["catch_up"][fid][0] > 0  # its first push builds on the catch-up
    # island 2 merged nothing older than its catch-up
    for record in steps:
        for learner, base, _ in record["raw_weights"]:
            if int(learner) == 2:
                assert int(base) >= state["catch_up"][record["fragment"]][0]
    assert '"kind":"pool_join"' in tape and '"island_id":2' in tape and '"catch_up":true' in tape

    # Island 2 merged; island 1 is in no round after its LEAVE; the weights renormalize.
    joined_rounds = [r for r in steps if any(int(e[0]) == 2 for e in r["raw_weights"])]
    assert joined_rounds
    leave = next(json.loads(line) for line in tape.splitlines()
                 if '"pool_leave"' in line and '"island_id":1' in line)
    assert leave["reason"] == "stopped"
    lines = tape.splitlines()
    at_leave = next(i for i, line in enumerate(lines) if '"pool_leave"' in line and '"island_id":1' in line)
    # rounds merged after the LEAVE, past the two rounds per fragment that were in flight
    after = [json.loads(line) for line in lines[at_leave:] if '"kind":"outer_step"' in line][2 * P:]
    assert after and all(int(e[0]) in (0, 2) for r in after for e in r["raw_weights"])
    assert any({int(e[0]) for e in r["raw_weights"]} == {0, 2} for r in after)
    for record in steps:
        assert abs(sum(float(e[2]) for e in record["weights"]) - 1.0) < 1e-9
        assert not record["timed_out"]  # every island of the pool answered every round
    assert all(r["arrived"] == 2 for r in steps[:join_at])  # θ = 1 over the two launched islands
    assert sum(r["arrived"] == 3 for r in steps) >= 4  # all three, several rounds in a row

    # One global cut, equal to the recomputation of every merged round.
    expected = _nesterov(steps)
    assert finals.keys() == {0, 2}
    for fid in range(P):
        assert np.array_equal(finals[0][fid], finals[2][fid])
        np.testing.assert_allclose(finals[0][fid], expected[fid], rtol=0, atol=1e-6)
    assert "learner id" not in log_file.read_text()  # no id was refused


@needs_cargo
@pytest.mark.usefixtures("_hmac_key")
@pytest.mark.parametrize("leave", [True, False], ids=["leave", "gone-without-leave"])
def test_an_island_that_is_gone_releases_its_open_round(syncer_binary, tmp_path, leave):
    """Island 1 holds a PULL it will never answer and closes. With a LEAVE the
    round closes on island 0 at once (the lease is 30 s here); without one (its
    machines torn down) once island 1 stayed away for one lease (6 s), not at
    the soft deadline (60 s)."""
    lease = 30 if leave else 6
    pool = ("--rl-sync-quorum-theta", "1.0", "--rl-sync-max-base-lag", "1",
            "--rl-sync-island-lease-s", str(lease), "--rl-sync-soft-deadline-s", "60")
    process, port, profile_hash = _start_async_syncer(syncer_binary, tmp_path, *pool, total_steps=4,
                                                      log_file=tmp_path / "syncer.log")
    clients = {learner: _client(learner, port, profile_hash) for learner in range(LAUNCHED)}
    versions = [0] * P
    seen: dict[int, set] = {0: set(), 1: set()}
    held: dict = {}
    beat = [0.0]

    def poll():
        if time.monotonic() - beat[0] > 1:  # island 0 renews its (short) lease while it waits
            beat[0] = time.monotonic()
            clients[0].heartbeat(0)
        for learner, client in list(clients.items()):
            client.check_health()
            for update in client.drain_updates():
                if learner == 0:
                    versions[update.fragment_id] = max(versions[update.fragment_id], update.version)
            for permit in client.drain_pulls():
                if (permit.global_step, permit.round_attempt) in seen[learner]:
                    continue
                seen[learner].add((permit.global_step, permit.round_attempt))
                if learner == 1 and permit.global_step >= 3:
                    held["step"] = permit.global_step  # answered never: island 1 goes instead
                    held["generation"] = client.connection_generation
                    if leave:
                        client.send_leave(2)
                    clients.pop(1).close()
                    break
                client.push_fragment(permit.fragment_id, permit.global_step, permit.round_attempt,
                                     versions[permit.fragment_id], permit.global_step, 1, 7,
                                     _f32(_delta(learner, permit.global_step)))
        return 1 not in clients and clients[0].finalizing.is_set()

    try:
        for client in clients.values():
            client.start()
        for fid in range(P):
            clients[0].send_init(fid, _f32(_initial(fid)))
        started = time.monotonic()
        _wait(poll, "syncer never finalized", timeout=50)
        assert time.monotonic() - started < 25  # no round waited for the soft deadline (or the 30 s lease)
        manifest, _ = clients[0].wait_for_final_fragments(timeout=30)
        clients[0].acknowledge_finalization(manifest)
        assert process.wait(timeout=15) == 0
    finally:
        for client in clients.values():
            client.close()
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=10)
    steps = {r["step"]: r for r in _outer_steps((tmp_path / "tape.jsonl").read_text())}
    left = steps[held["step"]]
    assert [int(e[0]) for e in left["raw_weights"]] == [0] and not left["timed_out"]
    if leave:
        assert left["step_seconds"] < 10
    else:  # a bare disconnect may be a short outage: its slot goes after one lease
        assert lease - 1 <= left["step_seconds"] < lease + 10
    tape = (tmp_path / "tape.jsonl").read_text()
    assert f'"island_id":1,"generation":{held["generation"]},"reason":"{"stopped" if leave else "disconnected"}"' in tape
    assert '"lease_expired"' not in tape
    released = [json.loads(line) for line in tape.splitlines() if '"pool_slots_released"' in line]
    assert [(r["island_id"], r["reason"], r["steps"]) for r in released] == [(1, "left" if leave else "lease",
                                                                             [held["step"]])]


@needs_cargo
@pytest.mark.usefixtures("_hmac_key")
def test_a_leave_releases_the_slot_of_a_dropped_generation_it_had_not_rebound(syncer_binary, tmp_path):
    """Generation A of island 1 holds a PULL and drops without LEAVE; generation B
    connects and sends LEAVE at once, before its catch-up could rebind A's slot.
    The round closes on island 0 at once: not at the soft deadline (60 s), nor at
    the lease (30 s) after A's disconnect. (Whether B's catch-up rebinds A's slot
    before the LEAVE is up to timing here; both orders release it. The scheduler
    test in server.rs pins the order without the rebind.)"""
    pool = ("--rl-sync-quorum-theta", "1.0", "--rl-sync-max-base-lag", "1",
            "--rl-sync-island-lease-s", "30", "--rl-sync-soft-deadline-s", "60")
    process, port, profile_hash = _start_async_syncer(syncer_binary, tmp_path, *pool, total_steps=4,
                                                      log_file=tmp_path / "syncer.log")
    clients = {learner: _client(learner, port, profile_hash) for learner in range(LAUNCHED)}
    versions = [0] * P
    seen: dict[int, set] = {0: set(), 1: set()}
    held: dict = {}

    def poll():
        if "dropped" in held and "left" not in held and time.monotonic() - held["dropped"] > 1:
            again = _client(1, port, profile_hash)  # generation B of the same island
            again.start()
            held["left"] = again.send_leave(2)
            again.close()
        for learner, client in list(clients.items()):
            client.check_health()
            for update in client.drain_updates():
                if learner == 0:
                    versions[update.fragment_id] = max(versions[update.fragment_id], update.version)
            for permit in client.drain_pulls():
                if (permit.global_step, permit.round_attempt) in seen[learner]:
                    continue
                seen[learner].add((permit.global_step, permit.round_attempt))
                if learner == 1 and permit.global_step >= 3:
                    held["step"] = permit.global_step
                    held["dropped"] = time.monotonic()
                    clients.pop(1).close()  # generation A drops, no LEAVE
                    break
                client.push_fragment(permit.fragment_id, permit.global_step, permit.round_attempt,
                                     versions[permit.fragment_id], permit.global_step, 1, 7,
                                     _f32(_delta(learner, permit.global_step)))
        return held.get("left") is not None and clients[0].finalizing.is_set()

    try:
        for client in clients.values():
            client.start()
        for fid in range(P):
            clients[0].send_init(fid, _f32(_initial(fid)))
        started = time.monotonic()
        _wait(poll, "syncer never finalized", timeout=50)
        assert time.monotonic() - started < 25  # neither the soft deadline nor the lease
        manifest, _ = clients[0].wait_for_final_fragments(timeout=30)
        clients[0].acknowledge_finalization(manifest)
        assert process.wait(timeout=15) == 0
    finally:
        for client in clients.values():
            client.close()
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=10)
    assert held["left"] is True
    tape = (tmp_path / "tape.jsonl").read_text()
    left = {r["step"]: r for r in _outer_steps(tape)}[held["step"]]
    assert [int(e[0]) for e in left["raw_weights"]] == [0] and not left["timed_out"]
    assert left["step_seconds"] < 10
    released = [json.loads(line) for line in tape.splitlines() if '"pool_slots_released"' in line]
    assert [(r["island_id"], r["reason"]) for r in released] == [(1, "left")]
    assert held["step"] in released[0]["steps"]


@needs_cargo
@pytest.mark.usefixtures("_hmac_key")
def test_an_island_back_within_its_lease_keeps_its_slot(syncer_binary, tmp_path):
    """Island 1 holds a PULL and its connection drops (no LEAVE); it is back
    within the lease (4 s) as a new generation, gets the round's PULL again and
    answers it only after the lease has passed: the round merges both islands,
    nothing is released."""
    lease = 4
    pool = ("--rl-sync-quorum-theta", "1.0", "--rl-sync-max-base-lag", "1",
            "--rl-sync-island-lease-s", str(lease), "--rl-sync-soft-deadline-s", "60")
    process, port, profile_hash = _start_async_syncer(syncer_binary, tmp_path, *pool, total_steps=4,
                                                      log_file=tmp_path / "syncer.log")
    clients = {learner: _client(learner, port, profile_hash) for learner in range(LAUNCHED)}
    versions = {learner: [0] * P for learner in range(LAUNCHED)}
    seen: dict[int, set] = {0: set(), 1: set()}
    held: dict = {}
    beat = [0.0]

    def poll():
        if time.monotonic() - beat[0] > 1:  # both islands renew their (short) leases while they wait
            beat[0] = time.monotonic()
            for client in clients.values():
                client.heartbeat(0)
        if "dropped" in held and "back" not in held and time.monotonic() - held["dropped"] > 0.5:
            held["back"] = time.monotonic()  # the same island, a new connection generation
            clients[1] = _client(1, port, profile_hash)
            clients[1].start()
        for learner, client in list(clients.items()):
            client.check_health()
            for update in client.drain_updates():
                versions[learner][update.fragment_id] = max(versions[learner][update.fragment_id], update.version)
            for permit in client.drain_pulls():
                key = (permit.global_step, permit.round_attempt)
                if learner == 1 and permit.global_step == held.get("step"):
                    if "back" not in held:
                        continue  # the dropped generation's copy
                    held["pending"] = permit  # the new generation's: answered after the lease
                    continue
                if key in seen[learner]:
                    continue
                seen[learner].add(key)
                if learner == 1 and "step" not in held and permit.global_step >= 3:
                    held["step"] = permit.global_step
                    held["dropped"] = time.monotonic()
                    clients.pop(1).close()  # the connection drops; the island is not gone
                    break
                client.push_fragment(permit.fragment_id, permit.global_step, permit.round_attempt,
                                     versions[learner][permit.fragment_id], permit.global_step, 1, 7,
                                     _f32(_delta(learner, permit.global_step)))
        pending = held.get("pending")
        if pending is not None and time.monotonic() - held["dropped"] > lease + 2:
            held["pending"] = None
            held["answered"] = True
            clients[1].push_fragment(pending.fragment_id, pending.global_step, pending.round_attempt,
                                     versions[1][pending.fragment_id], pending.global_step, 1, 7,
                                     _f32(_delta(1, pending.global_step)))
        return all(c.finalizing.is_set() for c in clients.values()) and held.get("answered", False)

    try:
        for client in clients.values():
            client.start()
        for fid in range(P):
            clients[0].send_init(fid, _f32(_initial(fid)))
        _wait(poll, "syncer never finalized", timeout=50)
        for client in clients.values():
            manifest, _ = client.wait_for_final_fragments(timeout=30)
            client.acknowledge_finalization(manifest)
        assert process.wait(timeout=15) == 0
    finally:
        for client in clients.values():
            client.close()
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=10)
    tape = (tmp_path / "tape.jsonl").read_text()
    steps = {r["step"]: r for r in _outer_steps(tape)}
    kept = steps[held["step"]]
    assert sorted(int(e[0]) for e in kept["raw_weights"]) == [0, 1] and not kept["timed_out"]
    assert kept["step_seconds"] > lease  # it waited past the lease for the island that came back
    assert '"pool_slots_released"' not in tape and '"lease_expired"' not in tape


@needs_cargo
@pytest.mark.usefixtures("_hmac_key")
def test_a_round_whose_islands_all_left_reopens_for_a_new_island_at_once(syncer_binary, tmp_path):
    """Islands 0 and 1 both hold the PULL of a round and leave (0 with LEAVE, 1
    torn down without one, released after its 4 s lease); island 2 joins later.
    Under θ = 1.0 the round is relaunched for island 2 once it caught up, not at
    the quorum timeout (900 s), and the run ends on island 2 alone."""
    lease, quorum_timeout = 4, 900  # the syncer's default quorum timeout
    pool = ("--rl-sync-quorum-theta", "1.0", "--rl-sync-max-base-lag", "1",
            "--rl-sync-island-lease-s", str(lease), "--rl-sync-max-learners", str(MAX_LEARNERS))
    tape_path = tmp_path / "tape.jsonl"
    process, port, profile_hash = _start_async_syncer(syncer_binary, tmp_path, *pool, total_steps=4,
                                                      log_file=tmp_path / "syncer.log")
    clients = {learner: _client(learner, port, profile_hash) for learner in range(LAUNCHED)}
    versions = {learner: [0] * P for learner in range(MAX_LEARNERS)}
    seen: dict[int, set] = {learner: set() for learner in range(MAX_LEARNERS)}
    held: dict[int, list] = {}
    state: dict = {"step": None, "gone": set(), "left_at": None, "joined": False, "beat": 0.0}

    def poll():
        if time.monotonic() - state["beat"] > 1:  # the islands still here renew their leases
            state["beat"] = time.monotonic()
            for client in clients.values():
                client.heartbeat(0)
        for learner, client in list(clients.items()):
            client.check_health()
            for update in client.drain_updates():
                fid = update.fragment_id
                versions[learner][fid] = max(versions[learner][fid], update.version)
            if learner == 2 and not state["joined"]:  # a joiner answers only after its catch-up
                held.setdefault(2, []).extend(client.drain_pulls())
                if min(versions[2]) == 0 and any(versions[0]):
                    continue
                state["joined"] = True
            for permit in held.pop(learner, []) + client.drain_pulls():
                key = (permit.global_step, permit.round_attempt)
                if key in seen[learner]:
                    continue
                seen[learner].add(key)
                if learner < LAUNCHED and permit.global_step >= 3:
                    state["step"] = permit.global_step  # never answered: the island goes
                    if learner == 0:
                        client.send_leave(2)
                    clients.pop(learner).close()
                    state["gone"].add(learner)
                    if state["gone"] == {0, 1}:
                        state["left_at"] = time.monotonic()
                    break
                client.push_fragment(permit.fragment_id, permit.global_step, permit.round_attempt,
                                     versions[learner][permit.fragment_id], permit.global_step, 1, 7,
                                     _f32(_delta(learner, permit.global_step)))
        if state["left_at"] is not None and 2 not in clients:
            versions[2] = [0] * P
            clients[2] = _client(2, port, profile_hash)
            clients[2].start()
        return 2 in clients and clients[2].finalizing.is_set()

    try:
        for client in clients.values():
            client.start()
        for fid in range(P):
            clients[0].send_init(fid, _f32(_initial(fid)))
        _wait(poll, "syncer never finalized", timeout=60)
        manifest, _ = clients[2].wait_for_final_fragments(timeout=30)
        clients[2].acknowledge_finalization(manifest)
        assert process.wait(timeout=15) == 0
    finally:
        for client in clients.values():
            client.close()
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=10)
    tape = tape_path.read_text()
    records = [json.loads(line) for line in tape.splitlines() if line.startswith("{")]
    deserted = [r for r in records if r.get("kind") == "round_attempt_timeout" and r.get("global_step") == state["step"]]
    assert deserted and {r["reason"] for r in deserted} == {"members_left"}, deserted
    assert deserted[-1]["next"] == "new_attempt"
    merged = {r["step"]: r for r in _outer_steps(tape)}[state["step"]]
    assert [int(e[0]) for e in merged["raw_weights"]] == [2] and not merged["timed_out"]
    # the round waited for island 1's lease and island 2's arrival, never for the quorum timeout
    assert merged["step_seconds"] < 30 < quorum_timeout, merged["step_seconds"]
    assert not [r for r in records if r.get("kind") == "round_attempt_timeout" and r.get("reason") == "quorum_timeout"]


@needs_cargo
@pytest.mark.usefixtures("_hmac_key")
def test_an_id_past_max_learners_is_still_refused(syncer_binary, tmp_path):
    from yeto.protocol import DTYPE_F32, SyncerClient

    log_file = tmp_path / "syncer.log"
    process, port, profile_hash = _start_async_syncer(syncer_binary, tmp_path, *POOL,
                                                      total_steps=2, log_file=log_file)
    outsider = SyncerClient(("127.0.0.1", port), MAX_LEARNERS, _layout(), dtype=DTYPE_F32,
                            num_streams=1, connect_timeout=10, max_reconnects=0,
                            syncer_profile_hash=bytes.fromhex(profile_hash),
                            session_contract_hash=_contract(0))
    try:
        outsider.start()
        _wait(lambda: "outside configured range 0..3" in log_file.read_text(),
              "the syncer admitted an id past --max-learners", timeout=20)
    finally:
        outsider.close()
        process.terminate()
        process.wait(timeout=10)


# -- learner ids of joining islands (yeto.launch.learner_ids) -----------------------


def test_joining_islands_get_unique_ids_that_are_never_reused(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from yeto.launch.island_lifecycle import IslandStateStore
    from yeto.launch.learner_ids import LearnerIdsExhausted, allocate_learner_id, claim_key
    from yeto.rl.engine.object_store import DirectoryObjectStore

    objects = DirectoryObjectStore(tmp_path / "store")
    IslandStateStore(objects, log=lambda m: None).write(0, "running")  # a launched island's record
    # eight islands join at once: every one gets its own id, none of the launched ones
    with ThreadPoolExecutor(8) as pool:
        ids = list(pool.map(lambda n: allocate_learner_id(objects, launched=2, max_learners=10,
                                                          island=f"join{n}"), range(8)))
    assert sorted(ids) == list(range(2, 10))
    claims = {json.loads(objects.read(claim_key(k)))["island"] for k in ids}
    assert claims == {f"join{n}" for n in range(8)}
    with pytest.raises(LearnerIdsExhausted, match="never reused"):
        allocate_learner_id(objects, launched=2, max_learners=10, island="late")
    # an island that left keeps its claim: the next run of the table skips its id
    assert allocate_learner_id(objects, launched=2, max_learners=11, island="later") == 10


def test_a_stale_listing_still_cannot_hand_out_one_id_twice(tmp_path):
    from yeto.launch.learner_ids import allocate_learner_id
    from yeto.rl.engine.object_store import DirectoryObjectStore

    class StaleListing(DirectoryObjectStore):
        def list(self, prefix, *, start_after=""):
            return []  # both joiners listed before either claimed

    objects = StaleListing(tmp_path / "store")
    first = allocate_learner_id(objects, launched=2, max_learners=4, island="a")
    second = allocate_learner_id(objects, launched=2, max_learners=4, island="b")
    assert (first, second) == (2, 3)  # the exclusive create decides, not the listing


def test_learner_ids_refuse_a_store_without_an_exclusive_create(tmp_path):
    from yeto.launch.learner_ids import allocate_learner_id
    from yeto.rl.engine.object_store import DirectoryObjectStore

    objects = DirectoryObjectStore(tmp_path / "store")
    objects.atomic_create = False  # e.g. a check-then-write relay
    with pytest.raises(ValueError, match="exclusive create"):
        allocate_learner_id(objects, launched=2, max_learners=4, island="a")
    with pytest.raises(ValueError, match="launched"):
        allocate_learner_id(DirectoryObjectStore(tmp_path / "s2"), launched=5, max_learners=4, island="a")


def test_the_verl_island_reads_the_pool_roster_flag():
    from test_verl_island_flags import REQUIRED

    from yeto.rl.adapters.verl import island_entry

    args, unread = island_entry.parse_args([*REQUIRED, "--sync-max-learners", "3"])
    assert args.sync_max_learners == 3 and "--sync-max-learners" not in unread
