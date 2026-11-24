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
# A count quorum of 2 (rounds may merge without an island: a base lag of 1 is
# admitted); the soft deadline only bounds a hang of a broken run.
POOL = ("--rl-sync-quorum", "2", "--rl-sync-max-base-lag", "1", "--rl-sync-island-lease-s", "5",
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
    records = [json.loads(line) for line in tape.splitlines() if line.strip()]
    return sorted((r for r in records if r.get("kind") == "outer_step"), key=lambda r: r["step"])


def _nesterov(steps: list[dict], *, upto: int | None = None) -> dict[int, np.ndarray]:
    """The syncer's outer step in f32 over the merged rounds of the tape (in
    learner order, its raw weights), up to global step ``upto``."""

    lr32, mu32 = np.float32(LR), np.float32(MU)
    params = {fid: _initial(fid) for fid in range(P)}
    buffers = {fid: np.zeros(NUMEL, dtype=np.float32) for fid in range(P)}
    for record in steps:
        if upto is not None and record["step"] > upto:
            break
        fid = record["fragment"]
        contributions = sorted((int(learner), float(weight)) for learner, _, weight in record["raw_weights"])
        total = sum(weight for _, weight in contributions)
        merged = np.zeros(NUMEL, dtype=np.float32)
        for learner, weight in contributions:
            merged += np.float32(weight / total) * -_delta(learner, record["step"])
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
    assert f"--learners {LAUNCHED} " in command and f"--max-learners {MAX_LEARNERS} " in command
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
        newest = list(versions[0])
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

    def poll():
        for learner in list(clients):
            drain(learner)
        if state["joined"] is None and min(versions[0]) >= join_at:
            join()
        if state["joined"] is not None and not state["left"]:
            # island 1 leaves once island 2 caught up: the rounds then need both 0 and 2
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
        assert not record["timed_out"]

    # One global cut, equal to the recomputation of every merged round.
    expected = _nesterov(steps)
    assert finals.keys() == {0, 2}
    for fid in range(P):
        assert np.array_equal(finals[0][fid], finals[2][fid])
        np.testing.assert_allclose(finals[0][fid], expected[fid], rtol=0, atol=1e-6)
    assert "learner id" not in log_file.read_text()  # no id was refused


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
