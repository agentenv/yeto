"""Request budget of the store dispenser (``yeto.rl.engine.store_dispense``): a scan
lists the unsettled slots only, with the same result as a listing of the whole
namespace; a lease renewal is one write per held slot."""

from __future__ import annotations

import bisect
import json
import random
from collections import Counter

import pytest

from test_store_dispense import Clock, FakeObjects, _open, _rows, _slots, _write
from yeto.rl.engine import store_dispense as sd

NS = "run/dispense"
PAGE = 1000  # keys of one ListObjectsV2 answer


class PagedObjects(FakeObjects):
    """The fake store with S3's listing: keys in byte order, 1000 a request, a
    range listing stopped at the first page that reaches ``end``. ``ops`` counts
    the requests by kind."""

    def __init__(self) -> None:
        super().__init__()
        self.ops: Counter[str] = Counter()
        self._sorted: list[str] | None = None

    def read(self, key):
        self.ops["get"] += 1
        return super().read(key)

    def write(self, key, data):
        self.ops["put"] += 1
        self._sorted = None
        super().write(key, data)

    def create(self, key, data):
        self.ops["put"] += 1
        self._sorted = None
        return super().create(key, data)

    def exists(self, key):
        self.ops["head"] += 1
        return super().exists(key)

    def put_raw(self, key, stamp=None):
        self.data[key] = (b"{}", self.now if stamp is None else stamp)
        self._sorted = None

    def list_meta(self, prefix, *, start_after="", end=""):
        self.requests += 1
        with self.lock:
            if self._sorted is None:
                self._sorted = sorted(self.data)
            keys = self._sorted
            i = bisect.bisect_right(keys, max(prefix, start_after)) if start_after else bisect.bisect_left(keys, prefix)
            out, pages = [], 0
            while True:
                pages += 1
                page = [k for k in keys[i:i + PAGE] if k.startswith(prefix)]
                out += [k for k in page if not end or k < end]
                i += PAGE
                if len(page) < PAGE or (end and page[-1] >= end):
                    break
            self.ops["list"] += pages
            return [(k, self.data[k][1]) for k in out], self.now


def _fill(store, *, settled=(), running=(), failed=(), unsettled_claimed=()):
    """Ledger keys written directly: ``settled`` slots (claim, beat, done), ``running``
    ones (claim, beat), ``failed`` ones (claim, beat, one ``infra`` failure)."""
    for p in settled:
        store.put_raw(f"{NS}/claims/{p:012d}/0000")
        store.put_raw(f"{NS}/beats/{p:012d}/0000")
        store.put_raw(f"{NS}/done/{p:012d}.trained")
    for p in running:
        store.put_raw(f"{NS}/claims/{p:012d}/0000")
        store.put_raw(f"{NS}/beats/{p:012d}/0000")
    for p in failed:
        store.put_raw(f"{NS}/claims/{p:012d}/0000")
        store.put_raw(f"{NS}/beats/{p:012d}/0000")
        store.put_raw(f"{NS}/fail/{p:012d}/0000.infra")


def _ledger(store, rows, *, holder="s0", full_scan=False, **kw):
    return sd.StoreLedger(store, NS, rows=rows, fingerprint=b"f", holder=holder, incarnation="i0",
                          lease_s=10**6, full_scan=full_scan, **kw)


def _scan_cost(ledger, store):
    store.ops.clear()
    states, now = ledger.scan()
    return dict(store.ops), ledger.candidates(states, now)


# ------------------------------------------------------------------ ranges
def test_scan_ranges_cover_every_unsettled_slot_and_merge_close_ones():
    assert sd.scan_ranges([], 0) == [(0, None)]
    assert sd.scan_ranges([], 500) == [(500, None)]
    assert sd.scan_ranges([3, 4, 90], 100, merge=10) == [(3, 5), (90, None)]
    assert sd.scan_ranges([3, 40], 100, merge=10) == [(3, 4), (40, 41), (100, None)]
    assert sd.scan_ranges([3, 40], 100, merge=50) == [(3, 41), (100, None)]
    rng = random.Random(0)
    for _ in range(200):
        top = rng.randrange(0, 400)
        open_slots = sorted(rng.sample(range(top), rng.randrange(0, min(top, 30) + 1))) if top else []
        ranges = sd.scan_ranges(open_slots, top, merge=rng.choice([1, 8, 64]))
        assert ranges[-1][1] is None and ranges[-1][0] <= top
        assert all(lo < hi for lo, hi in ranges[:-1]) and all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:]))
        for p in open_slots:
            assert any(lo <= p and (hi is None or p < hi) for lo, hi in ranges)


def test_settled_slots_keep_a_floor_instead_of_every_slot():
    seen = sd.SettledSlots(limit=10)
    for p in (2, 0, 1, 5, 11):
        seen.add(p, "trained")
    assert seen.floor == 3 and seen.above == {5: "trained"} and len(seen) == 4 and seen.top == 5
    assert 1 in seen and 5 in seen and 3 not in seen and 11 not in seen
    assert seen.open_below_top() == [3, 4] and seen.ranges() == [(3, None)]


# ------------------------------------------------------------ differential
def _same_view(inc, full, accept, rng):
    """One scan of ``inc`` (range listings) and of ``full`` (the whole namespace)
    agree on every unsettled slot, on which slots are settled and on the candidates."""
    inc_states, inc_now = inc.scan()
    full_states, full_now = full.scan()
    assert inc_now == full_now
    for p, state in full_states.items():
        if state.done is not None:
            assert inc.is_settled(p, inc_states) or p >= inc.limit, p
        else:
            assert inc_states.get(p) == state, p
    for p, state in inc_states.items():
        if state.done is None:
            assert p in full_states and full_states[p].done is None, p
    assert inc.retired == full.retired and inc.drain_seq == full.drain_seq
    order = [(inc, inc_states, inc_now), (full, full_states, full_now)]
    if rng.random() < 0.5:
        order.reverse()  # whose scan ends the expired attempts first does not matter
    got = [ledger.candidates(states, now, accept=accept) for ledger, states, now in order]
    assert got[0] == got[1]


@pytest.mark.parametrize("seed", range(12))
def test_a_range_scan_sees_what_a_full_listing_sees_on_random_ledgers(tmp_path, monkeypatch, seed):
    """Samplers claim, train, fail, crash, restart, get retired and drained in a
    random order (half of them scanning by ranges, half listing everything); after
    every step an observer of each kind must agree."""
    rng = random.Random(seed)
    merge = [1, 4, 32, 1024][seed % 4]
    monkeypatch.setattr(sd, "SCAN_MERGE_SLOTS", merge)
    most_ranges, list_slots = [1], sd.list_slots

    def counted(objects, namespace, directory, ranges):
        most_ranges[0] = max(most_ranges[0], len(ranges))
        return list_slots(objects, namespace, directory, ranges)

    monkeypatch.setattr(sd, "list_slots", counted)
    mixed = seed % 3 == 1
    rows = _rows(60, source=(lambda i: i % 2) if mixed else None)
    path = _write(tmp_path, rows)
    store = PagedObjects()
    clock = Clock(store)
    config = {"lease_s": 90, "max_redo": 1, "passes": 2 if seed % 3 == 2 else 1,
              **({"mix": ["0.7", "0.3"]} if mixed else {})}
    incarnations = Counter()

    def start(k):
        incarnations[k] += 1
        prompts = _open(path, store, holder=f"s{k}", inc=f"i{incarnations[k]}", clock=clock,
                        groups=rng.choice([1, 2, 5]), config={**config, "full_scan": k % 2 == 1})
        assert prompts.ledger.incremental == (k % 2 == 0)
        prompts.recover([])
        return prompts

    samplers = {k: start(k) for k in range(4)}
    inc = sd.StoreLedger(store, NS, rows=60, fingerprint=samplers[0].ledger.fingerprint, holder="watch",
                         incarnation="w", lease_s=90, max_redo=1, passes=config["passes"],
                         slots=samplers[0].ledger.slots, full_scan=False)
    full = sd.StoreLedger(store, NS, rows=60, fingerprint=samplers[0].ledger.fingerprint, holder="watch",
                          incarnation="w", lease_s=90, max_redo=1, passes=config["passes"],
                          slots=samplers[0].ledger.slots, full_scan=True)
    accept = (lambda slot: inc.slots.stream(slot) == 0) if mixed and seed % 2 else None
    limit, ended = inc.limit, False
    for _step in range(400):
        k = rng.randrange(4)
        prompts = samplers[k]
        roll = rng.random()
        if roll < 0.45:
            state, got = prompts.poll(prompts.groups_per_round)
            ended = state == "end"
        elif roll < 0.75:
            held = sorted(prompts._held)
            done = [p for p in held if rng.random() < 0.6]
            again = [p for p in held if p not in done and rng.random() < 0.5]
            prompts.written(0, done, requeue=again, unwritten=rng.choice(["release", "keep"]))
        elif roll < 0.82:
            samplers[k] = start(k)  # a crash: the old process's claims are released by the new one
        elif roll < 0.87:
            clock.sleep(rng.choice([10, 40, 100]))  # past the lease now and then
        elif roll < 0.92:
            prompts.renew()
        elif roll < 0.95:
            sd.retire(store, NS, f"s{k}", reason="test")
            samplers[k] = start(k)
        elif roll < 0.98:
            sd.request_drain(store, NS, f"s{k}", on=rng.random() < 0.5)
        else:  # a sandbox gate settles a task that was never claimed
            slot = rng.randrange(limit)
            if not store.exists(f"{NS}/claims/{slot:012d}/0000"):
                prompts.ledger.settle(slot, "dropped", reason="test")
        _same_view(inc, full, accept, rng)
        if ended:
            break
    # run it to the end: nothing is lost, and both kinds of sampler see the end
    for k in range(4):
        sd.request_drain(store, NS, f"s{k}", on=False)
    for _round in range(400):
        states = []
        for k, prompts in samplers.items():
            state, got = prompts.poll(prompts.groups_per_round)
            states.append(state)
            prompts.written(0, sorted(prompts._held))
        _same_view(inc, full, accept, rng)
        if all(s == "end" for s in states):
            break
        clock.sleep(1)
    else:
        raise AssertionError("the ledger never ended")
    assert most_ranges[0] >= 2 or merge > 4  # the ledger had old unsettled slots: several ranges a scan
    report = sd.status(store, NS)
    assert report["totals"]["ended"] and len(inc.settled) == limit == inc.settled.floor
    brief = sd.status(store, NS, settled=False)
    assert brief["totals"] == report["totals"] and brief["tasks"] == []


def test_a_new_process_recovers_the_same_slots_with_range_listings(tmp_path):
    path = _write(tmp_path, _rows(40))
    reports = []
    for full_scan in (False, True):
        store = PagedObjects()
        first = _open(path, store, holder="s0", inc="i1", groups=6, config={"full_scan": full_scan})
        slots = _slots(first(0))
        first.written(0, slots[:2], keep=slots[2:], unwritten="keep")

        class Entry:
            def __init__(self, slot):
                self.sampler_round = {"sampler": "s0", "dispense_position": slot}

        again = _open(path, store, holder="s0", inc="i2", groups=6, config={"full_scan": full_scan})
        # slots[2] and slots[3] reached the buffer before the first process died
        reports.append((again.recover([Entry(p) for p in slots[:4]]), sd.status(store, NS)["totals"]))
    assert reports[0] == reports[1]
    assert reports[0][0] == {"recovered": 2, "released": 2}


# ----------------------------------------------------------- request counts
SLOTS = 20_000


def test_a_scan_of_a_long_run_costs_a_handful_of_requests_not_one_per_thousand_keys():
    store = PagedObjects()
    _fill(store, settled=range(SLOTS - 300), running=range(SLOTS - 300, SLOTS - 100))
    full_cost, full_out = _scan_cost(_ledger(store, SLOTS + 1000, full_scan=True), store)
    assert full_cost["list"] >= 3 * (SLOTS - 300) // PAGE  # every key, every scan
    ledger = _ledger(store, SLOTS + 1000)
    first, out = _scan_cost(ledger, store)
    assert out == full_out
    # a new process: done/ whole, then the unsettled window and retired/
    assert first["list"] <= SLOTS // PAGE + 5 and first["list"] < full_cost["list"] / 2
    steady, out = _scan_cost(ledger, store)
    assert out == full_out
    assert steady == {"list": 5, "get": 1}  # done, beats, claims, fail, retired + the drain request
    # the run goes on: the window moves, the cost stays
    _fill(store, settled=range(SLOTS - 300, SLOTS - 50), running=range(SLOTS - 50, SLOTS + 200))
    moved, out = _scan_cost(ledger, store)
    assert moved == {"list": 5, "get": 1}
    assert out == _scan_cost(_ledger(store, SLOTS + 1000, full_scan=True), store)[1]
    assert ledger.settled.floor == SLOTS - 50 and not ledger.settled.above


def test_old_unsettled_slots_cost_a_range_each_and_never_much_more_than_a_full_listing():
    def cost(stragglers):
        store = PagedObjects()
        behind = set(stragglers)
        _fill(store, settled=[p for p in range(SLOTS - 200) if p not in behind],
              running=[*behind, *range(SLOTS - 200, SLOTS)])
        ledger = _ledger(store, SLOTS + 1000)
        ledger.scan()
        steady, out = _scan_cost(ledger, store)
        full, full_out = _scan_cost(_ledger(store, SLOTS + 1000, full_scan=True), store)
        assert out == full_out
        return steady["list"], full["list"]

    assert cost([]) == (5, 60)
    assert cost([10]) == (9, 60)  # one trial that has run since the start: one more range
    assert cost([10, 500, 600]) == (9, 60)  # close ones share a range
    assert cost(range(0, SLOTS - 200, 4000)) == (5 + 4 * 5, 60)
    every_other, full = cost(range(0, SLOTS - 200, 2))  # a stream no sampler serves: no settled run to skip
    assert full - 5 <= every_other <= full + 5


def test_a_scan_reads_the_drain_request_and_retire_markers_without_the_full_listing(tmp_path):
    path = _write(tmp_path, _rows(8))
    store = PagedObjects()
    a = _open(path, store, holder="s0", groups=2)
    b = _open(path, store, holder="s1", groups=2)
    held = _slots(a(0))
    seq = sd.request_drain(store, NS, "s0")
    assert a.poll(2) == ("wait", []) and a.draining() and a.ledger.drain_seq == seq
    assert sd.retire(store, NS, "s0")["released"] == 2
    assert a.poll(2) == ("fenced", [])
    assert sorted(b.poll(2)[1]) == sorted(held)  # the retired sampler's slots, released at once


def test_a_slot_settled_without_a_claim_past_the_frontier_is_not_given_out():
    """A sandbox gate settles ``dropped`` a task nobody claimed. On main the frontier
    was the highest claimed slot, and every slot past it was offered unseen."""
    store = PagedObjects()
    ledger = _ledger(store, 8)
    assert ledger.claim(0, 0)
    assert ledger.settle(5, "dropped", reason="sandbox validation failed")
    for full_scan in (False, True):
        states, now = _ledger(store, 8, holder="s1", full_scan=full_scan).scan()
        candidates, ended = _ledger(store, 8, holder="s1", full_scan=full_scan).candidates(states, now)
        assert [p for p, _a in candidates] == [1, 2, 3, 4, 6, 7] and not ended
    # the lowest slot unclaimed (its claim raised), the next one settled
    store = PagedObjects()
    ledger = _ledger(store, 8)
    assert ledger.claim(1, 0) and ledger.settle(1, "trained") and ledger.settle(5, "dropped")
    for full_scan in (False, True):
        watch = _ledger(store, 8, holder="s1", full_scan=full_scan)
        assert [p for p, _a in watch.candidates(*watch.scan())[0]] == [0, 2, 3, 4, 6, 7]


def test_retire_and_open_status_list_done_and_the_unsettled_slots_only(tmp_path):
    store = PagedObjects()
    _fill(store, settled=range(SLOTS), running=range(SLOTS, SLOTS + 40), failed=range(SLOTS + 40, SLOTS + 50))
    store.data[f"{NS}/plan.json"] = (json.dumps({"rows": SLOTS + 100, "passes": 1, "lease_s": 10**6}).encode(), store.now)
    for p in range(SLOTS, SLOTS + 40):
        store.data[f"{NS}/claims/{p:012d}/0000"] = (
            json.dumps({"holder": f"s{p % 2}", "incarnation": "i0", "task_id": f"t{p}"}).encode(), store.now)
    store._sorted = None
    store.ops.clear()
    brief = sd.status(store, NS, settled=False)
    cost = dict(store.ops)
    assert cost["list"] <= SLOTS // PAGE + 4 and cost["get"] <= 1 + 50 + 10  # plan, claim bodies, failure bodies
    assert [t["slot"] for t in brief["tasks"]] == list(range(SLOTS, SLOTS + 50))
    assert brief["totals"]["trained"] == SLOTS and brief["totals"]["running"] == 40
    assert brief["totals"]["queued"] == 10 and brief["totals"]["unclaimed"] == 50
    store.ops.clear()
    full = sd.status(store, NS, details=False)
    assert store.ops["list"] >= 3 * SLOTS // PAGE and store.ops["get"] >= SLOTS
    assert full["totals"] == brief["totals"]
    assert [t for t in sd.status(store, NS)["tasks"] if t["slot"] >= SLOTS] == brief["tasks"]
    store.ops.clear()
    out = sd.retire(store, NS, "s1", reason="test")
    assert out["released"] == 20 and out["slots"] == list(range(SLOTS + 1, SLOTS + 40, 2))
    assert store.ops["list"] <= SLOTS // PAGE + 3
    assert sd.retire(store, NS, "s1", reason="test")["released"] == 0


# ----------------------------------------------------------------- renewals
def test_a_lease_renewal_is_one_write_per_held_slot_and_checks_takeover_every_third_time(tmp_path):
    path = _write(tmp_path, _rows(8))
    store = PagedObjects()
    worker = _open(path, store, groups=4)
    held = _slots(worker(0))
    store.ops.clear()
    assert [worker.renew() for _ in range(sd.SUPERSEDED_CHECK_EVERY)] == [4] * sd.SUPERSEDED_CHECK_EVERY
    # main: 3 existence checks and a write per slot per renewal (36 + 12 here)
    assert dict(store.ops) == {"put": 4 * sd.SUPERSEDED_CHECK_EVERY, "head": 4 * 3}
    # another sampler ends one attempt and takes the slot over
    other = _ledger(store, 8, holder="s1")
    assert other.fail(held[0], 0, "expired") and other.claim(held[0], 1)
    assert worker.renew() == 4 and worker.renew() == 4  # unseen: the beats change nothing
    states, now = other.scan()
    assert states[held[0]].latest == 1 and states[held[0]].last_seen(1) == store.data[f"{NS}/claims/{held[0]:012d}/0001"][1]
    assert worker.renew() == 3 and worker.stats["lost"] == 1  # the third renewal checks
    assert not worker.still_holds(held[0]) and worker.still_holds(held[1])


def test_a_scan_marks_a_taken_over_slot_lost_without_asking_the_store(tmp_path):
    path = _write(tmp_path, _rows(8))
    store = PagedObjects()
    events = []
    worker = _open(path, store, groups=2, emit=lambda event, **fields: events.append((event, fields)))
    held = _slots(worker(0))
    other = _ledger(store, 8, holder="s1")
    assert other.fail(held[0], 0, "release", reason="fenced")
    store.ops.clear()
    worker.poll(0)
    assert store.ops["head"] == 0 and worker.stats["lost"] == 1
    assert [f for e, f in events if e == sd.LOST_EVENT] == [
        {"backend": "store", "namespace": NS, "slot": held[0], "attempt": 0, "reason": "superseded"}]
    store.ops.clear()
    assert not worker.still_holds(held[0]) and store.ops["head"] == 0  # already known
    assert worker.renew() == 1  # only the slot it still holds


def test_full_scan_switch_lists_the_whole_namespace(tmp_path, monkeypatch):
    store = PagedObjects()
    _fill(store, settled=range(3000))
    monkeypatch.setenv(sd.FULL_SCAN_ENV, "1")
    ledger = _ledger(store, 4000, full_scan=None)
    assert not ledger.incremental
    assert _scan_cost(ledger, store)[0] == {"list": 10}
    monkeypatch.delenv(sd.FULL_SCAN_ENV)
    assert _ledger(store, 4000, full_scan=None).incremental

    class Plain(FakeObjects):
        def list_meta(self, prefix):  # an object API without range listings
            return super().list_meta(prefix)

    assert not _ledger(Plain(), 10).incremental
