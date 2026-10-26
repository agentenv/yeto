"""Late-push carry against the real Rust syncer (CPU, no Ray).

With quorum K below the island count, a round may close before a slightly
slower island's push arrives. In asynchronous rounds (a positive base-lag
bound) the syncer merges that late delta into the next round of the same
fragment, against its own base and with its own token weight, instead of
dropping it. Each island anchor yields one delta: a push from an anchor
whose delta was already taken is refused, so nothing is counted twice.

The rig is the launcher-built syncer of ``test_rl_outer_merge_real_syncer``
(P=2 fragments, pipeline 1): round ``t`` merges fragment ``(t - 1) % 2``.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from test_rl_outer_merge_real_syncer import (DEFAULT_LR, DEFAULT_MU, NUMEL, Reference,  # noqa: F401
                                             _check, _deltas, _f32, _from, _initial, make_rig,
                                             syncer_binary)
from test_rl_algorithm_provenance import _fake_sky, _no_modal_listing  # noqa: F401

ASYNC_K1 = ("--rl-sync-quorum", "1", "--rl-sync-max-base-lag", "1",
            "--rl-sync-learner-weight", "tokens2-over-steps")


@pytest.fixture(autouse=True)
def _hmac_key(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "late-carry-test-key")


def _tape(rig):
    path = Path(rig.command.split("--event-tape ")[1].split()[0])
    if not path.exists():
        return []
    with path.open() as tape:
        return [json.loads(line) for line in tape if line.strip()]


def _merge_records(rig):
    return {r["step"]: r for r in _tape(rig) if "responders" in r and "step" in r}


def _weight(c_steps, c_tokens):
    return c_tokens * c_tokens / c_steps


def test_late_push_is_merged_once_into_the_next_round_of_its_fragment(make_rig):
    rig = make_rig(2, *ASYNC_K1, "--rl-sync-grace-ms", "0", total_steps=4)
    assert "--quorum 1" in rig.command and "--max-base-lag 2" in rig.command
    initial = _initial(rig.layout)
    rig.start(initial)
    ref = Reference(rig.layout, initial, DEFAULT_LR, DEFAULT_MU)
    d = _deltas(rig.layout, 0, 7, 4)

    # Round 1 (fragment 0, base 0) closes on island 0 alone.
    rig.push(0, 1, d[0], base=0, c_steps=1, c_tokens=4)
    _check(rig.bcast(0, 0, 1), ref.step(0, [d[0]], [1.0]))
    # Island 1's delta for round 1 arrives late; fragment 1 has round 2 open.
    rig.push(1, 1, d[1], base=0, c_steps=2, c_tokens=6)
    d1 = _deltas(rig.layout, 1, 8, 1)[0]
    rig.push(0, 2, d1, base=0)
    _check(rig.bcast(0, 1, 2), ref.step(1, [d1], [1.0]))

    # Round 3 (fragment 0, base 1). Island 1 answers from its old anchor 0
    # again: that anchor already gave the carried delta, so it is refused.
    rig.push(1, 3, d[2], base=0, c_steps=3, c_tokens=9)
    with pytest.raises(TimeoutError):
        rig.bcast(0, 0, 3, timeout=1.0)
    rig.push(0, 3, d[3], base=1, c_steps=1, c_tokens=4)
    want = ref.step(0, [d[3], d[1]], [_weight(1, 4), _weight(2, 6)])
    _check(rig.bcast(0, 0, 3), want)

    records = _merge_records(rig)
    assert [r["id"] for r in records[3]["responders"]] == [0]
    (carried,) = records[3]["carried"]
    assert (carried["id"], carried["missed_step"], carried["base_version"], carried["lag"]) == (1, 1, 0, 1)
    assert "carried" not in records[1] and "carried" not in records[2]


def test_late_push_beyond_the_base_lag_bound_is_dropped(make_rig):
    rig = make_rig(2, *ASYNC_K1, "--rl-sync-grace-ms", "0", total_steps=6)
    initial = _initial(rig.layout)
    rig.start(initial)
    ref = Reference(rig.layout, initial, DEFAULT_LR, DEFAULT_MU)
    for step in (1, 2, 3):
        fid = (step - 1) % 2
        delta = _deltas(rig.layout, fid, step, 1)[0]
        rig.push(0, step, delta, base=max(step - 2, 0))
        _check(rig.bcast(0, fid, step), ref.step(fid, [delta], [1.0]))
    # Fragment 0 is at version 3: a delta from base 0 lags 3 > 2 versions.
    rig.push(1, 1, np.ones(NUMEL), base=0)
    delta = _deltas(rig.layout, 1, 4, 1)[0]
    rig.push(0, 4, delta, base=2)
    _check(rig.bcast(0, 1, 4), ref.step(1, [delta], [1.0]))
    delta = _deltas(rig.layout, 0, 5, 1)[0]
    rig.push(0, 5, delta, base=3)
    _check(rig.bcast(0, 0, 5), ref.step(0, [delta], [1.0]))
    assert all("carried" not in r for r in _merge_records(rig).values())
    assert rig.process.poll() is None


def test_strict_control_still_drops_a_late_push(make_rig):
    rig = make_rig(2, "--rl-sync-quorum", "1", "--rl-sync-grace-ms", "0", total_steps=4)
    assert "--max-base-lag 0" in rig.command
    initial = _initial(rig.layout)
    rig.start(initial)
    ref = Reference(rig.layout, initial, DEFAULT_LR, DEFAULT_MU)
    d = _deltas(rig.layout, 0, 3, 2)
    rig.push(0, 1, d[0], base=0)
    _check(rig.bcast(0, 0, 1), ref.step(0, [d[0]], [1.0]))
    rig.push(1, 1, d[1], base=0)  # late: dropped
    d1 = _deltas(rig.layout, 1, 4, 1)[0]
    rig.push(0, 2, d1, base=0)
    _check(rig.bcast(0, 1, 2), ref.step(1, [d1], [1.0]))
    d3 = _deltas(rig.layout, 0, 5, 1)[0]
    rig.push(0, 3, d3, base=1)
    _check(rig.bcast(0, 0, 3), ref.step(0, [d3], [1.0]))
    assert all("carried" not in r for r in _merge_records(rig).values())


# ---- many rounds with free-running islands -----------------------------------------

class Island(threading.Thread):
    """A fake island that trains in fixed-length local steps.

    Each step adds a fixed per-island increment to every fragment. At a step
    end the island answers every PULL it holds with ``local - anchor`` of
    that fragment, then keeps training; a broadcast becomes the fragment's
    anchor at a step end and the progress not yet pushed is kept on top of
    it (``theta <- G + unpushed progress``). A broadcast older than an open
    PULL is applied before answering it; one that closes the PULL's own
    round is applied after the push, so that push may arrive late. Like the
    pool client, it heartbeats several times per step with its step count.
    Every delta sent is recorded by (step, fragment, base)."""

    def __init__(self, rig, index, step_s, stop):
        super().__init__(daemon=True)
        self.rig, self.index, self.step_s, self.stop = rig, index, step_s, stop
        self.client = rig.clients[index]
        fragments = rig.layout.fragments
        rng = np.random.default_rng(100 + index)
        self.increment = [rng.normal(size=f.numel).astype(np.float32).astype(np.float64) * 0.01
                          for f in fragments]
        self.version = [0] * len(fragments)
        self.unpushed = [0] * len(fragments)
        self.staged = {}
        self.pending = {}
        self.sent = {}
        self.updates = {}
        self.steps = 0
        self.error = None

    def _apply(self, fid, below):
        for (f, version) in sorted(self.staged):
            if f == fid and version < below:
                self.staged.pop((f, version))
                self.version[fid] = max(self.version[fid], version)

    def run(self):
        try:
            while not self.stop.is_set():
                for _ in range(4):  # heartbeats repeat the current step count
                    time.sleep(self.step_s / 4)
                    self.client.heartbeat(self.steps)
                self.steps += 1
                self.client.check_health()
                for fid in range(len(self.unpushed)):
                    self.unpushed[fid] += 1
                for update in self.client.drain_updates():
                    self.updates[(update.fragment_id, update.version)] = _from(update.data)
                    self.staged[(update.fragment_id, update.version)] = True
                for pull in self.client.drain_pulls():
                    self.pending[pull.fragment_id] = pull
                for fid, pull in sorted(self.pending.items()):
                    self._apply(fid, pull.global_step)
                    steps = self.unpushed[fid]
                    delta = (self.increment[fid] * steps).astype(np.float32)
                    base = self.version[fid]
                    self.sent[(pull.global_step, fid, base)] = (delta.astype(np.float64), steps)
                    self.client.push_fragment(fid, pull.global_step, pull.round_attempt, base,
                                              self.steps, steps, 10 * steps, delta.tobytes())
                    self.unpushed[fid] = 0
                self.pending.clear()
                for fid in range(len(self.unpushed)):
                    self._apply(fid, float("inf"))
        except Exception as error:  # noqa: BLE001 - surfaced by the test
            if not self.stop.is_set():
                self.error = error


def _run_free(make_rig, step_s, rounds, grace_ms):
    rig = make_rig(2, *ASYNC_K1, "--rl-sync-grace-ms", str(grace_ms), total_steps=rounds + 2)
    initial = _initial(rig.layout)
    rig.start(initial)
    stop = threading.Event()
    islands = [Island(rig, i, s, stop) for i, s in enumerate(step_s)]
    for island in islands:
        island.start()
    deadline = time.monotonic() + 60 + rounds * max(step_s) * 4
    try:
        while time.monotonic() < deadline:
            records = _merge_records(rig)
            if all(t in records for t in range(1, rounds + 1)):
                break
            for island in islands:
                assert island.error is None, island.error
            time.sleep(0.1)
        else:
            raise TimeoutError(f"syncer merged only {sorted(_merge_records(rig))}")
        time.sleep(2 * max(step_s))  # let island 0 drain the last broadcasts
    finally:
        stop.set()
        for island in islands:
            island.join(timeout=5)
    records = _merge_records(rig)
    return rig, initial, islands, [records[t] for t in range(1, rounds + 1)]


def check_free_run(rig, initial, islands, records):
    """Every merge equals the numpy reference over the deltas its tape record
    names (carried ones included); no delta is taken twice. Returns the
    number of rounds each island contributed to, and its late deltas."""
    ref = Reference(rig.layout, initial, DEFAULT_LR, DEFAULT_MU)
    taken = set()
    participated = [0] * len(islands)
    carried = [0] * len(islands)
    for record in records:
        fid, step = record["fragment"], record["step"]
        entries = ([(r, step) for r in record["responders"]]
                   + [(c, c["missed_step"]) for c in record.get("carried", [])])
        deltas, weights, seen = [], [], set()
        for entry, sent_step in entries:
            island = entry["id"]
            key = (island, sent_step, fid, entry["base_version"])
            assert key not in taken, f"delta {key} merged twice"
            taken.add(key)
            seen.add(island)
            delta, steps = islands[island].sent[key[1:]]
            assert entry["c_steps"] == steps
            deltas.append(delta)
            weights.append(_weight(entry["c_steps"], entry["c_tokens"]))
        for island in seen:
            participated[island] += 1
        for entry in record.get("carried", []):
            carried[entry["id"]] += 1
        want = ref.step(fid, deltas, weights)
        got = islands[0].updates.get((fid, step))
        if got is not None:
            _check(got, want)
    return taken, participated, carried


def _summary(rounds, records, participated, carried):
    direct = [sum(any(r["id"] == i for r in rec["responders"]) for rec in records) for i in (0, 1)]
    return f"rounds {rounds}: merged-in {participated}, direct {direct}, carried {carried}"


@pytest.mark.parametrize("step_s", [(0.25, 0.25), (0.25, 0.28)], ids=["equal", "slower"])
def test_free_running_islands_lose_no_delta_and_count_none_twice(make_rig, step_s):
    rounds = 24
    rig, initial, islands, records = _run_free(make_rig, step_s, rounds, grace_ms=0)
    taken, participated, carried = check_free_run(rig, initial, islands, records)
    print(_summary(rounds, records, participated, carried))
    # Every delta sent for an early enough round was merged exactly once.
    for index, island in enumerate(islands):
        lost = [key for key in island.sent
                if key[0] <= rounds - 2 and (index, *key) not in taken]
        assert not lost, f"island {index} lost deltas {lost}"
    # Both islands contribute to many rounds. The count depends on load: an
    # island cannot answer a round it never held a PULL for (a later PULL of
    # the same fragment in the same local step replaces it). The no-loss
    # check above is the exact one.
    assert min(participated) >= rounds // 2, participated


def test_adaptive_grace_waits_for_a_slightly_slower_island(make_rig):
    rounds = 24
    rig, initial, islands, records = _run_free(make_rig, (0.25, 0.28), rounds, grace_ms=2000)
    taken, participated, carried = check_free_run(rig, initial, islands, records)
    print(_summary(rounds, records, participated, carried))
    direct = sum(any(r["id"] == 1 for r in rec["responders"]) for rec in records)
    assert direct >= rounds * 3 // 4, _summary(rounds, records, participated, carried)


class ReactiveIsland(threading.Thread):
    """An island that starts a local round on each PULL and answers it
    ``work_s`` later from the newest broadcast, heartbeating four times per
    round with its round count (like the pool client). Two of them with
    different ``work_s`` keep a fixed lateness between their pushes."""

    def __init__(self, rig, index, work_s, stop):
        super().__init__(daemon=True)
        self.client, self.index, self.work_s, self.stop = rig.clients[index], index, work_s, stop
        self.version = [0] * rig.layout.num_fragments
        self.rounds = 0
        self.error = None

    def run(self):
        try:
            while not self.stop.is_set():
                pulls = self.client.drain_pulls()
                if not pulls:
                    time.sleep(0.005)
                    continue
                for _ in range(4):
                    time.sleep(self.work_s / 4)
                    self.client.heartbeat(self.rounds)
                for update in self.client.drain_updates():
                    fid = update.fragment_id
                    self.version[fid] = max(self.version[fid], update.version)
                self.rounds += 1
                for pull in pulls:
                    fid = pull.fragment_id
                    self.client.push_fragment(fid, pull.global_step, pull.round_attempt,
                                              self.version[fid], self.rounds, 1, 10,
                                              _f32(np.full(NUMEL, 0.01 * (self.index + 1))))
        except Exception as error:  # noqa: BLE001 - surfaced by the test
            if not self.stop.is_set():
                self.error = error


def test_grace_window_waits_for_a_peer_that_is_late_by_less_than_the_cap(make_rig):
    """GPU run G-A scaled 1:100: a round of ~100 s, island 1 always ~20 s
    behind island 0, grace cap 30 s. The grace window must wait for it, so
    island 1 answers most rounds directly (it used to answer 1 of 24)."""
    rounds = 24
    rig = make_rig(2, *ASYNC_K1, "--rl-sync-grace-ms", "300", total_steps=rounds + 2)
    rig.start(_initial(rig.layout))
    stop = threading.Event()
    islands = [ReactiveIsland(rig, 0, 0.8, stop), ReactiveIsland(rig, 1, 1.0, stop)]
    for island in islands:
        island.start()
    deadline = time.monotonic() + 300
    try:
        while time.monotonic() < deadline:
            records = _merge_records(rig)
            if all(t in records for t in range(1, rounds + 1)):
                break
            for island in islands:
                assert island.error is None, island.error
            time.sleep(0.1)
        else:
            raise TimeoutError(f"syncer merged only {sorted(_merge_records(rig))}")
    finally:
        stop.set()
        for island in islands:
            island.join(timeout=5)
    records = [_merge_records(rig)[t] for t in range(1, rounds + 1)]
    direct = sum(any(r["id"] == 1 for r in rec["responders"]) for rec in records)
    grace = [rec["grace_ms"] for rec in records]
    print(f"island 1 direct in {direct}/{rounds} rounds; grace_ms {grace}")
    assert direct >= rounds - 3, (direct, grace)
