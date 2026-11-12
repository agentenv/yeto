"""Store-ledger task dispenser (``yeto.rl.engine.store_dispense``) on a fake
object store: exclusive create (``If-None-Match: *``) and server-side time."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from yeto.rl.engine import store_dispense as sd
from yeto.rl.engine.task_dispense import DISPENSE_POSITION_KEY, DispenseEnded, DispenseError, DispenseMismatch


class FakeObjects:
    """An object store: atomic exclusive create, LastModified and the listing's
    Date from one server clock (``now``), never the caller's."""

    def __init__(self) -> None:
        self.now = 1_000_000.0
        self.data: dict[str, tuple[bytes, float]] = {}
        self.lock = threading.Lock()
        self.requests = 0

    def read(self, key):
        self.requests += 1
        item = self.data.get(key)
        return None if item is None else item[0]

    def write(self, key, data):
        self.requests += 1
        with self.lock:
            self.data[key] = (bytes(data), self.now)

    def create(self, key, data):
        self.requests += 1
        with self.lock:
            if key in self.data:
                return False  # 412 PreconditionFailed
            self.data[key] = (bytes(data), self.now)
            return True

    def exists(self, key):
        self.requests += 1
        return key in self.data

    def list_meta(self, prefix, *, start_after="", end=""):
        self.requests += 1
        with self.lock:
            return [(k, t) for k, (_, t) in self.data.items()
                    if k.startswith(prefix) and k > start_after and (not end or k < end)], self.now


def _rows(n, *, label=None, source=None):
    rows = []
    for i in range(n):
        meta = {"task_id": f"t{i}"}
        if source is not None:
            meta["yeto_data_source"] = source(i)
        row = {"text": f"q{i}", "label": "", "metadata": meta}
        if label is not None:
            row["task_label"] = label(i)
        rows.append(row)
    return rows


def _write(tmp_path: Path, rows) -> Path:
    path = tmp_path / "prompts.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


class Clock:
    def __init__(self, store):
        self.t = 0.0
        self.store = store

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s
        self.store.now += s


def _open(path, store, holder="s0", inc="i0", config=None, groups=2, lease_of_holder=None, **kw):
    cfg = {"backend": "store", "namespace": "run/dispense", "lease_s": 90, "max_redo": 1, "scan_s": 0.0,
           "max_scan_s": 0.0, **(config or {})}
    clock = kw.pop("clock", None) or Clock(store)
    return sd.StoreDispensedPrompts.open(path, cfg, store=store, holder=holder, incarnation=inc,
                                         input_key="text", label_key="label", groups_per_round=groups,
                                         task_key="metadata", renew=False, sleep=clock.sleep, clock=clock,
                                         lease_of_holder=lease_of_holder, **kw)


def _slots(items):
    return [it.metadata[DISPENSE_POSITION_KEY] for it in items]


def _status(store):
    return sd.status(store, "run/dispense")


def test_concurrent_samplers_never_share_a_task(tmp_path):
    store = FakeObjects()
    path = _write(tmp_path, _rows(200))
    samplers = [_open(path, store, holder=f"s{k}", inc="i", groups=3) for k in range(8)]
    got: dict[int, list[int]] = {k: [] for k in range(8)}
    barrier = threading.Barrier(8)

    def run(k):
        barrier.wait()
        try:
            while True:
                items = samplers[k](0)
                spots = _slots(items)
                got[k] += spots
                samplers[k].written(0, spots)
        except DispenseEnded:
            pass

    threads = [threading.Thread(target=run, args=(k,)) for k in range(8)]
    [t.start() for t in threads]
    [t.join(30) for t in threads]
    every = [p for spots in got.values() for p in spots]
    assert sorted(every) == list(range(200))  # each task once, none lost
    assert _status(store)["totals"]["trained"] == 200
    assert _status(store)["totals"]["ended"]


def test_crashed_sampler_task_is_reclaimed_after_the_lease(tmp_path):
    store = FakeObjects()
    path = _write(tmp_path, _rows(2))
    dead = _open(path, store, holder="dead", groups=2)
    assert _slots(dead(0)) == [0, 1]  # then the process dies: no renewal, no report
    live = _open(path, store, holder="live", groups=2)
    store.now += 89
    assert live._claim(2)[0] == []  # still alive at 89 s (server time): nothing to take
    items = live(0)  # live waits (its sleep advances the server clock) until the lease runs out
    assert sorted(_slots(items)) == [0, 1]
    assert all(it.metadata["dispense_attempt"] == 1 for it in items)
    assert not dead.still_holds(0)  # the dead holder may not write
    live.written(0, _slots(items))
    tasks = {t["slot"]: t for t in _status(store)["tasks"]}
    assert tasks[0]["state"] == "trained" and tasks[0]["fails"] == {"0": "expired"}


def test_long_task_is_renewed_not_expired(tmp_path):
    store = FakeObjects()
    path = _write(tmp_path, _rows(1))
    worker = _open(path, store, holder="w", groups=1)
    other = _open(path, store, holder="o", groups=1)
    assert _slots(worker(0)) == [0]
    for _ in range(40):  # 40 x 30 s = 20 minutes, lease 90 s, renewed every lease/3
        store.now += 30
        assert worker.renew() == 1
        got, _ended = other._claim(1)
        assert got == []
    assert worker.renew_period() == pytest.approx(30.0)
    assert worker.still_holds(0)
    worker.written(0, [0])
    assert _status(store)["tasks"][0]["fails"] == {}


def test_row_lease_overrides_the_default(tmp_path):
    store = FakeObjects()
    rows = _rows(1)
    rows[0]["metadata"]["dispense_lease_s"] = 3600
    path = _write(tmp_path, rows)
    worker = _open(path, store, holder="w", groups=1)
    worker(0)
    assert worker.renew_period() == pytest.approx(1200.0)
    store.now += 1000  # past the default 90 s, inside the row's hour
    other = _open(path, store, holder="o", groups=1)
    assert other._claim(1)[0] == []


def test_infra_failure_requeues_up_to_max_redo(tmp_path):
    store = FakeObjects()
    path = _write(tmp_path, _rows(1))
    s = _open(path, store, groups=1, config={"max_redo": 2})
    for attempt in range(3):
        items = s(0)
        assert _slots(items) == [0] and items[0].metadata["dispense_attempt"] == attempt
        out = s.written(0, [], requeue=[0], outcome={0: {"reason": "below_min_samples"}})
        assert out["requeued"] == 1
    with pytest.raises(DispenseEnded):
        s(0)  # 3 infra endings > max_redo 2: exhausted, the data ends
    task = _status(store)["tasks"][0]
    assert task["state"] == "exhausted" and task["attempts"] == 3
    assert task["failure_reasons"] == ["0:infra:below_min_samples", "1:infra:below_min_samples",
                                       "2:infra:below_min_samples"]


def test_zero_reward_group_is_trained_not_redispensed(tmp_path):
    store = FakeObjects()
    path = _write(tmp_path, _rows(2))
    s = _open(path, store, groups=2)
    spots = _slots(s(0))
    s.written(0, spots, outcome={p: {"reason": None, "reward_mean": 0.0} for p in spots})
    with pytest.raises(DispenseEnded):
        s(1)
    assert _status(store)["totals"]["trained"] == 2


def test_unwritten_release_does_not_count(tmp_path):
    store = FakeObjects()
    path = _write(tmp_path, _rows(1))
    s = _open(path, store, groups=1, config={"max_redo": 0})
    for _ in range(4):  # released (fenced/reclaimed/unwritten) attempts never exhaust a task
        s(0)
        s.written(0, [])
    s(0)
    s.written(0, [0])
    assert _status(store)["tasks"][0]["state"] == "trained"


def test_fenced_sampler_cannot_write_and_its_claims_are_freed(tmp_path):
    store = FakeObjects()
    path = _write(tmp_path, _rows(2))
    leases = {"old": "i-old"}
    fenced = {"v": False}
    old = _open(path, store, holder="old", inc="i-old", groups=2, lease_of_holder=leases.get,
                fenced=lambda: fenced["v"])
    assert _slots(old(0)) == [0, 1]
    # the NC pool fences the shard (fence_shard: lease -> "pool-fence:<by>")
    leases["old"] = "pool-fence:pool"
    fenced["v"] = True
    assert not old.still_holds(0)  # its own check: no group write
    assert old.renew() == 0  # no renewal
    new = _open(path, store, holder="nc-2", inc="i2", groups=2, lease_of_holder=leases.get)
    items = new(0)  # at once, without waiting for the lease
    assert sorted(_slots(items)) == [0, 1]
    out = old.written(0, [0, 1])  # a late write report of the fenced one settles nothing
    assert out["trained"] == 0
    new.written(0, _slots(items))
    tasks = {t["slot"]: t for t in _status(store)["tasks"]}
    assert tasks[0]["state"] == "trained" and tasks[0]["fails"] == {"0": "release"}
    assert tasks[0]["failure_reasons"] == ["0:release:fenced"]
    assert tasks[0]["holder"] == "nc-2"
    with pytest.raises(DispenseEnded):
        old(1)  # a fenced incarnation takes nothing new


def test_restart_continues_without_loss_or_duplicates(tmp_path):
    store = FakeObjects()
    path = _write(tmp_path, _rows(6))
    first = _open(path, store, holder="s0", inc="a", groups=3)
    spots = _slots(first(0))
    assert spots == [0, 1, 2]
    # slot 0 was written to the buffer, then the process died before reporting it

    class Entry:
        sampler_round = {"sampler": "s0", "incarnation": "a", "dispense_position": 0}

    second = _open(path, store, holder="s0", inc="b", groups=3)
    assert second.recover([Entry()]) == {"recovered": 1, "released": 2}
    seen = []
    try:
        while True:
            got = _slots(second(0))
            seen += got
            second.written(0, got)
    except DispenseEnded:
        pass
    assert sorted(seen) == [1, 2, 3, 4, 5]  # 0 not again, 1 and 2 not lost
    totals = _status(store)["totals"]
    assert totals["trained"] == 6 and totals["ended"]


def test_holdout_leak_is_refused(tmp_path):
    store = FakeObjects()
    path = _write(tmp_path, _rows(3))
    with pytest.raises(DispenseMismatch, match="hold-out"):
        _open(path, store, config={"holdout_ids": ["t2", "zz"]})
    with pytest.raises(DispenseMismatch, match="require_holdout"):
        _open(path, store, config={"require_holdout": True})
    _open(path, store, holder="a", config={"holdout_ids": ["zz"]})
    with pytest.raises(DispenseMismatch, match="holdout_sha256"):  # another list than the plan's
        _open(path, store, holder="b", config={"holdout_ids": ["yy"]})


def test_plan_mismatch_is_refused(tmp_path):
    store = FakeObjects()
    _open(_write(tmp_path, _rows(3)), store)
    other = tmp_path / "o"
    other.mkdir()
    with pytest.raises(DispenseMismatch, match="rows"):
        _open(_write(other, _rows(4)), store)


def test_data_ends_after_every_pass(tmp_path):
    store = FakeObjects()
    path = _write(tmp_path, _rows(4))
    s = _open(path, store, groups=3, config={"passes": 2, "seed": 7})
    rows = []
    with pytest.raises(DispenseEnded):
        while True:
            items = s(0)
            rows += [it.metadata["row"] for it in items]
            s.written(0, _slots(items))
    assert rows[:4] == [0, 1, 2, 3] and sorted(rows[4:]) == [0, 1, 2, 3]
    assert _status(store)["totals"] == {**_status(store)["totals"], "trained": 8, "slots": 8, "ended": True,
                                        "unclaimed": 0}
    assert s.close()["released"] == 0


def test_gap_below_the_frontier_is_given_out(tmp_path):
    store = FakeObjects()
    path = _write(tmp_path, _rows(3))
    s = _open(path, store, groups=1)
    s.ledger.claim(2, 0)  # a claim of slot 2 landed, 0 and 1 never did
    s.ledger.settle(2, "trained")
    got = []
    with pytest.raises(DispenseEnded):
        while True:
            items = s(0)
            got += _slots(items)
            s.written(0, _slots(items))
    assert sorted(got) == [0, 1]


def test_mix_weights_hold_across_samplers(tmp_path):
    store = FakeObjects()
    rows = _rows(400, source=lambda i: 0 if i < 100 else 1)  # 100 rows of source 0, 300 of source 1
    path = _write(tmp_path, rows)
    a = _open(path, store, holder="a", groups=5, config={"mix": ["0.5", "0.5"], "passes": 4})
    b = _open(path, store, holder="b", groups=7, config={"mix": ["0.5", "0.5"], "passes": 4})
    sources = []
    for _ in range(20):
        for s in (a, b):
            items = s(0)
            sources += [rows[it.metadata["row"]]["metadata"]["yeto_data_source"] for it in items]
            s.written(0, _slots(items))
    assert abs(sources.count(0) - sources.count(1)) <= 1  # 50/50 despite 1:3 rows


def test_labels_take_only_their_streams(tmp_path):
    store = FakeObjects()
    rows = _rows(10, label=lambda i: "swe" if i % 2 else "term")
    path = _write(tmp_path, rows)
    swe = _open(path, store, holder="swe", groups=10, config={"labels": ["swe"]})
    term = _open(path, store, holder="term", groups=10, config={"labels": ["term"]})
    a, b = swe(0), term(0)
    assert {rows[it.metadata["row"]]["task_label"] for it in a} == {"swe"} and len(a) == 5
    assert {rows[it.metadata["row"]]["task_label"] for it in b} == {"term"} and len(b) == 5
    swe.written(0, _slots(a))
    term.written(0, _slots(b))
    with pytest.raises(DispenseEnded):
        swe(1)


def test_interleave_matches_the_weights():
    order = sd.interleave([30, 30, 30], [1, 2, 3])
    for n in range(1, 31):  # within one slot of the mix while no stream is exhausted
        prefix = order[:n]
        for s, w in enumerate((1, 2, 3)):
            assert abs(prefix.count(s) - n * w / 6) < 1 + 1e-9
    assert sorted(order) == [0] * 30 + [1] * 30 + [2] * 30


def test_store_without_conditional_writes_is_refused(tmp_path):
    store = FakeObjects()
    store.conditional_writes = False
    with pytest.raises(DispenseError, match="exclusive create"):
        _open(_write(tmp_path, _rows(1)), store)


def test_bucket_mount_is_refused(tmp_path):
    from yeto.rl.engine.resume import BucketMountStore

    with pytest.raises(DispenseError, match="bucket mount"):
        sd.ledger_objects(BucketMountStore(tmp_path))


def test_dir_objects_ledger_and_status_cli(tmp_path, capsys):
    store = sd.DirObjects(tmp_path / "store")
    path = _write(tmp_path, _rows(2))
    s = _open(path, store, groups=2)
    s.written(0, _slots(s(0))[:1])
    assert sd.main(["status", "--store", str(tmp_path / "store"), "--namespace", "run/dispense",
                    "--log", str(tmp_path / "log.jsonl")]) == 0
    out = capsys.readouterr().out
    assert "trained" in out and "slot" in out
    lines = [json.loads(x) for x in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [x["event"] for x in lines] == ["rl_dispense_task"] * 2
    assert {x["state"] for x in lines} == {"trained", "queued"}


def test_events_reach_the_tape(tmp_path):
    store = FakeObjects()
    path = _write(tmp_path, _rows(2))
    events = []
    s = _open(path, store, groups=2, emit=lambda e, **f: events.append((e, f)))
    spots = _slots(s(0))
    s.written(0, spots[:1], requeue=spots[1:])
    names = [e for e, _ in events]
    assert names == ["rl_dispense_claim", "rl_dispense_claim", "rl_dispense_settle", "rl_dispense_requeue"]
    assert events[2][1]["state"] == "trained" and events[2][1]["backend"] == "store"


def test_build_prompts_wires_the_store_backend(tmp_path):
    from yeto.rl.engine import engine_sampler_entry as entry
    from yeto.rl.engine.task_dispense import SAMPLER_INCARNATION_ENV

    path = _write(tmp_path, _rows(2))
    spec = {"sampler_id": "nc-0", "store": str(tmp_path / "store"),
            "prompts": {"path": str(path), "input_key": "text", "label_key": "label", "groups_per_round": 2,
                        "dispense": {"backend": "store", "namespace": "task-dispense"}}}
    prompts = entry.build_prompts(spec, {SAMPLER_INCARNATION_ENV: "inc-1"})
    try:
        assert isinstance(prompts, sd.StoreDispensedPrompts)
        assert prompts.ledger.holder == "nc-0" and prompts.ledger.incarnation == "inc-1"
        assert (tmp_path / "store" / "task-dispense" / "plan.json").exists()
    finally:
        prompts.close()
    with pytest.raises(entry.SamplerSpecError, match="incarnation"):
        entry.build_prompts(spec, {})


def test_nc_launcher_dispense_overrides(tmp_path):
    from yeto.launch.nc import tasks

    holdout = tmp_path / "h.json"
    holdout.write_text(json.dumps(["x1", "x2"]))
    cfg = tasks.dispense_config("k31/dispense", lease_s=1800, max_redo=3, holdout_file=str(holdout),
                                mix="0.7,0.3")
    assert cfg == {"backend": "store", "namespace": "k31/dispense", "lease_s": 1800, "max_redo": 3,
                   "holdout_ids": ["x1", "x2"], "require_holdout": True, "mix": ["0.7", "0.3"]}
    spec = tasks.apply_sampler_overrides({"prompts": {"path": "p"}}, dispense=cfg)
    assert spec["prompts"]["dispense"] == cfg
    with pytest.raises(ValueError, match="data slot"):
        tasks.apply_sampler_overrides({"prompts": {}}, dispense=cfg, data_slot="0/2")
    with pytest.raises(ValueError, match="relative"):
        tasks.dispense_config("../x")


def test_agentic_round_checks_ownership_before_the_group_write(tmp_path):
    from tests.test_agentic_round_streaming import Harness, Prompts, _entries, _round
    from yeto.rl.engine.agentic_round import GROUP_EVENT

    class Held(Prompts):
        def __init__(self, *a, **kw):
            super().__init__(*a, dispensed=True, **kw)
            self.lost: set[int] = set()
            self.reports = []

        def still_holds(self, position):
            return position not in self.lost

        def _written(self, rollout_id, positions, *, unwritten="drop", keep=(), requeue=(), outcome=None):
            self.reports.append((list(positions), sorted(requeue), dict(outcome or {})))
            return {"rollout_id": rollout_id, "trained": len(positions)}

    prompts = Held({0: ["a"], 1: ["b"]})
    rnd, _engine, events = _round(tmp_path, Harness({"a": 0.0, "b": 0.0}), prompts, incarnation="aaaa")
    try:
        rnd(0)
        rnd.flush()
        prompts.lost.add(101)  # "b": another sampler took the task over
        rnd(1)
        rnd.flush()
    finally:
        rnd.close()
    groups = [f for n, f in events if n == GROUP_EVENT]
    assert [(g["prompt_id"], g["written"], g["dropped_reason"]) for g in groups] == [
        ("a", True, None), ("b", False, "dispense_lost")]
    assert len(_entries(tmp_path)[0]) == 1
    assert [r[0] for r in prompts.reports] == [[100], []]
    assert prompts.reports[1][2][101]["reason"] == "dispense_lost"


def test_brief_status_table_shows_task_id_and_holder(tmp_path):
    store = FakeObjects()
    path = _write(tmp_path, _rows(2))
    s = _open(path, store, holder="sampler-a", groups=2)
    spots = _slots(s(0))
    s.written(0, spots[:1], outcome={spots[0]: {"reason": None, "reward_mean": 1.0}})
    full = {t["slot"]: t for t in sd.status(store, "run/dispense")["tasks"]}
    brief = {t["slot"]: t for t in sd.status(store, "run/dispense", details=False)["tasks"]}
    for slot in spots:
        assert brief[slot]["task_id"] == full[slot]["task_id"] == f"t{slot}"
        assert brief[slot]["holder"] == full[slot]["holder"] == "sampler-a"
    lines = sd.format_table(sd.status(store, "run/dispense", details=False)).splitlines()[2:]
    for line in lines:
        cells = line.split()
        assert cells[3] == f"t{cells[0]}" and cells[4] == "sampler-a"


def test_a_draining_sampler_claims_nothing_new_finishes_its_claims_and_answers(tmp_path):
    """Scale down in "drain" mode (the NC pool's ``request_drain``): the drained sampler
    claims no new task, its claims in flight are written and trained, and it answers
    ``drained`` once it holds none; the other sampler takes every other task, so no task
    is given out twice and none is lost. A lifted request lets it claim again."""
    store = FakeObjects()
    path = _write(tmp_path, _rows(6))
    clock = Clock(store)
    old = _open(path, store, holder="nc-0", groups=2, clock=clock)
    other = _open(path, store, holder="nc-1", groups=2)
    assert _slots(old(0)) == [0, 1]
    seq = sd.request_drain(store, "run/dispense", "nc-0")
    assert old(0) == []  # holding: nothing new
    assert not sd.drain_answered(store, "run/dispense", "nc-0", seq)
    taken = []
    while len(taken) < 4:
        spots = _slots(other(0))
        taken += spots
        other.written(0, spots)
    assert sorted(taken) == [2, 3, 4, 5]
    assert old.written(0, [0, 1])["trained"] == 2  # its trials in flight still count
    answered = {}

    def sleep(s):  # the pool reads the answer, then lifts the request (a scale up again)
        answered["v"] = sd.drain_answered(store, "run/dispense", "nc-0", seq)
        sd.request_drain(store, "run/dispense", "nc-0", on=False)
        clock.sleep(s)

    old._sleep = sleep
    with pytest.raises(DispenseEnded):
        old(0)
    assert answered["v"] is True
    tasks = {t["slot"]: t for t in _status(store)["tasks"]}
    assert len(tasks) == 6 and all(t["state"] == "trained" for t in tasks.values())
    assert tasks[0]["holder"] == tasks[1]["holder"] == "nc-0"
    assert {tasks[i]["holder"] for i in range(2, 6)} == {"nc-1"}


# ------------------------------------------------- retire racing a claim pass
def test_retire_writes_its_marker_before_it_lists_the_claims(tmp_path):
    store = FakeObjects()
    path = _write(tmp_path, _rows(2))
    worker = _open(path, store, groups=1)
    assert _slots(worker(0)) == [0]
    marker = "run/dispense/retired/s0/i0"
    seen = []
    listing = store.list_meta

    def list_meta(prefix, **kw):
        seen.append(store.exists(marker))
        return listing(prefix, **kw)

    store.list_meta = list_meta
    assert sd.retire(store, "run/dispense", "s0", "i0")["slots"] == [0]
    assert seen and all(seen)


def test_a_claim_taken_while_retire_runs_is_released_by_its_claimant(tmp_path):
    """The sampler scans (no marker yet), ``retire`` lists the claims and writes its
    marker, then the sampler claims: retire released nothing, the claim pass that won
    sees the marker, gives the claim back and the process writes nothing."""
    store = FakeObjects()
    path = _write(tmp_path, _rows(2))
    worker = _open(path, store, groups=1)
    claim = worker.ledger.claim
    retired = {}

    def claim_after_retire(slot, attempt, **detail):
        if not retired:
            retired.update(sd.retire(store, "run/dispense", "s0", "i0"))
        return claim(slot, attempt, **detail)

    worker.ledger.claim = claim_after_retire
    assert worker.poll(1) == ("fenced", [])
    assert retired["released"] == 0  # its listing came before the claim
    assert not worker.still_holds(0)
    other = _open(path, store, holder="s1", groups=2)
    assert sorted(_slots(other(0))) == [0, 1]  # released at once, not after the lease


def test_a_process_does_not_claim_again_a_slot_whose_lost_attempt_it_still_runs(tmp_path):
    store = FakeObjects()
    path = _write(tmp_path, _rows(2))
    worker = _open(path, store, groups=1)
    assert _slots(worker(0)) == [0]
    store.now += 91  # no renewal: attempt 0 runs out while its group is still generating here
    got, _ended = worker._claim(1)
    assert got == [1]  # slot 0 goes to another sampler, not to a second attempt here
    worker.written(0, [])  # the lost group settles: slot 0 is no longer held here
    store.now += 91
    assert worker._claim(1)[0] == [0]


def test_a_retire_check_the_store_does_not_answer_keeps_the_claims_and_the_sampler(tmp_path):
    """``retired_now`` after a won claim pass is one existence check: a 5xx there is no
    verdict. The pass keeps what it claimed (the next scan reads the marker, and
    ``still_holds`` gates the write) instead of raising out of the claim with the
    claims taken but handed to nobody."""
    store = FakeObjects()
    path = _write(tmp_path, _rows(2))
    worker = _open(path, store, groups=1)
    exists = store.exists

    def flaky(key):
        if "/retired/" in key:
            raise OSError("503 Service Unavailable")
        return exists(key)

    store.exists = flaky
    assert worker.poll(1) == ("items", [0])
    assert worker.still_holds(0)


def test_a_retire_that_fails_after_its_marker_still_frees_the_claims(tmp_path):
    """The marker is written, the release is not (the controller died, a 5xx): the
    retired process fences itself at its next scan and another sampler's scan
    releases the claim it still finds, long before the lease runs out."""
    store = FakeObjects()
    path = _write(tmp_path, _rows(2))
    worker = _open(path, store, groups=1)
    assert _slots(worker(0)) == [0]
    create = store.create

    def create_marker_only(key, body):
        if "/retired/" not in key:
            raise OSError("503 Service Unavailable")
        return create(key, body)

    store.create = create_marker_only
    try:
        sd.retire(store, "run/dispense", "s0", "i0")
    except OSError:
        pass
    store.create = create
    assert worker.poll(1) == ("fenced", [])
    other = _open(path, store, holder="s1", groups=2)
    assert sorted(_slots(other(0))) == [0, 1]


def test_a_new_incarnation_of_a_retired_holder_is_not_fenced_by_the_old_marker(tmp_path):
    store = FakeObjects()
    path = _write(tmp_path, _rows(2))
    old = _open(path, store, groups=1)
    assert _slots(old(0)) == [0]
    sd.retire(store, "run/dispense", "s0")  # no incarnation given: the ones holding claims
    new = _open(path, store, inc="i1", groups=2)
    assert new.poll(2) == ("items", [0, 1])
