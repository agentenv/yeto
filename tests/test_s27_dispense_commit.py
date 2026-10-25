"""S27 P2 task dispenser, G-DH: rows are reported trained when their round
passed the sync boundary (commit), the end of the data ends an island
FINISHED, and the launcher keeps the dispenser log with the run.

Real Rust syncer for the dispenser (no learner), stand-in Miles data sources,
the fake engine for the island driver. No Ray, no GPU."""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest

from test_s27_dispense import FP, Syncer, crash, syncer_binary  # noqa: F401
from test_s27_dispense_wiring import FakeMilesSource
from tools.eval.dispense_check import check, check_tapes, tape_rows
from yeto.rl.engine import task_dispense as td
from yeto.rl.engine.buffer_payload import SampleBufferExhausted, data_end_kind, is_buffer_exhausted

KEY = b"dispense-test-key"


def _island(port: int, learner_id: int, rows: int, *, block: int, prefetch: int = 1,
            executor_round: bool = False):
    """One stand-in Miles island: (fake data source, executor, dispensed source)."""
    from yeto.rl.adapters.miles import task_dispense as hook

    fake = FakeMilesSource(rows)
    executor = SimpleNamespace(data_source=fake)
    if executor_round:
        executor.rollout_id = 0
    config = {"address": f"127.0.0.1:{port}", "learner_id": learner_id, "rows": rows,
              "fingerprint": FP.hex(), "seed": 3, "block": block, "prefetch_blocks": prefetch}
    hook._CURRENT = None
    hook.install(executor, config, KEY)
    return fake, executor, hook.current()


def _positions(groups):
    return [g[0].metadata[td.DISPENSE_POSITION_KEY] for g in groups]


@pytest.fixture(autouse=True)
def _no_current_source():
    from yeto.rl.adapters.miles import task_dispense as hook

    hook._CURRENT = None
    yield
    hook._CURRENT = None


# ------------------------------------------------------------ commit after the boundary


def test_kept_rows_wait_for_the_commit_and_a_death_before_it_gives_them_out_again(
        syncer_binary, tmp_path):
    """Island 1 generates and keeps a round, then dies before the round's commit
    (e.g. during the train step): its rows go back first, island 0 trains them,
    and no row is reported trained twice."""
    syncer = Syncer(syncer_binary, tmp_path, lease_s=30)
    try:
        rows = 40
        a_src, a_exec, a = _island(syncer.port, 0, rows, block=4, executor_round=True)
        b_src, b_exec, b = _island(syncer.port, 1, rows, block=4, executor_round=True)
        # round 0 on both; the executor names the round (Miles RolloutExecutor.get)
        a_groups = a_src.get_samples(4)
        assert a.on_trained(a_groups)["rollout_id"] == 0
        b_groups = b_src.get_samples(4)
        b.on_trained(b_groups)
        assert b.pending_positions() == set(_positions(b_groups))
        assert b.tasks.stats["trained"] == 0, "nothing is reported before the commit"
        assert a.commit(0) == {"rounds": [0], "trained": 4, "acked": True, "lost": 0}
        # rollout ahead: round 1 is generated and kept before round 0 commits
        a_exec.rollout_id = 1
        a_next = a_src.get_samples(4)
        a.on_trained(a_next)
        assert a.commit(0)["trained"] == 0 and a.commit(1)["trained"] == 4
        # island 1 dies before its round-0 commit; a new process supersedes it
        crash(b.tasks)
        b2_src, _, b2 = _island(syncer.port, 1, rows, block=4)
        lost = set(_positions(b_groups))
        deadline = time.monotonic() + 10
        while b2.tasks.ready_count() < 4 and time.monotonic() < deadline:
            time.sleep(0.05)
        regiven = set(_positions(b2_src.get_samples(4)))
        assert regiven == lost, "the dead process's kept rows come back first (to the next claim)"
        records = syncer.records()
        reclaim = next(i for i, r in enumerate(records) if r.get("kind") == "dispense_reclaim")
        assert records[reclaim]["reason"] == "superseded"
        a.close()
        b2.close()
        summary = check(syncer.records())
        assert summary["ok"], summary["problems"]
        assert summary["trained"] == 8 and summary["held_at_end"] == 0
    finally:
        syncer.stop()


def test_commit_of_rounds_up_to_an_id_and_close_releases_the_rest(syncer_binary, tmp_path):
    from yeto.rl.adapters.miles import task_dispense as hook

    syncer = Syncer(syncer_binary, tmp_path, lease_s=30)
    try:
        fake, executor, wrapped = _island(syncer.port, 0, 30, block=3, prefetch=2)
        for round_id in range(3):
            wrapped.on_trained(fake.get_samples(3), rollout_id=round_id)
        assert hook.commit_round(executor, 1) == {"rounds": [0, 1], "trained": 6, "acked": True,
                                                  "lost": 0}
        closed = hook.close_source(executor)
        assert closed["uncommitted"] == 3 and closed["released"] >= 3
        assert hook.current() is None and hook.commit_round(executor, 2) is None
        summary = check(syncer.records())
        assert summary["ok"], summary["problems"]
        assert summary["trained"] == 6 and summary["held_at_end"] == 0
        assert summary["queued_at_end"] == summary["given_out"] - 6
    finally:
        syncer.stop()


# --------------------------------------------------------------------- end of the data


def test_islands_that_cannot_fill_a_batch_at_the_end_give_up_and_never_wait_forever(
        syncer_binary, tmp_path):
    """--rl-data-passes 1 over 10 rows, batches of 4, two islands: the last rows
    cannot fill a batch. Before G-DH an island holding a partial batch waited
    for itself (WAIT for ever). Now each island ends with DispenseEnded, the
    partial rows go back to the queue, and no row trains twice."""
    syncer = Syncer(syncer_binary, tmp_path, "--dispense-passes", "1", lease_s=30)
    try:
        rows = 10
        islands = [_island(syncer.port, i, rows, block=4) for i in (0, 1)]
        trained: list[int] = []
        ended: dict[int, str] = {}
        lock = threading.Lock()

        def run(index):
            fake, _executor, wrapped = islands[index]
            while True:
                try:
                    groups = fake.get_samples(4)
                except td.DispenseEnded as error:
                    ended[index] = str(error)
                    return
                wrapped.on_trained(groups)
                wrapped.commit(wrapped._hooks)
                with lock:
                    trained.extend(_positions(groups))
                time.sleep(0.02 * (index + 1))

        threads = [threading.Thread(target=run, args=(i,), daemon=True) for i in (0, 1)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert not any(t.is_alive() for t in threads), "an island waited for ever at the end"
        assert sorted(ended) == [0, 1]
        assert len(trained) == len(set(trained)) == 8, "two full batches, each row once"
        for _fake, _executor, wrapped in islands:
            wrapped.close()
        summary = check(syncer.records())
        assert summary["ok"], summary["problems"]
        assert summary["trained"] == 8 and summary["held_at_end"] == 0
        assert summary["queued_at_end"] == 2, "the rows of the incomplete batch, not trained"
    finally:
        syncer.stop()


def test_data_end_kind_follows_wrappers():
    inner = td.DispenseEnded("ended")
    try:
        try:
            raise inner
        except td.DispenseEnded as e:
            raise RuntimeError("ray task failed") from e
    except RuntimeError as outer:
        assert data_end_kind(outer) == "dispense_ended"
        assert not is_buffer_exhausted(outer)
    dual = type("DispenseEnded", (RuntimeError,), {})("dual class of another process")
    assert data_end_kind(dual) == "dispense_ended"
    assert data_end_kind(SampleBufferExhausted("x")) == "buffer_exhausted"
    assert is_buffer_exhausted(SampleBufferExhausted("x"))
    assert data_end_kind(RuntimeError("other")) is None and data_end_kind(None) is None


# --------------------------------------------------------------------- island driver


def _events(tmp_path):
    return [json.loads(line) for line in (tmp_path / "e.jsonl").read_text().splitlines()]


def test_driver_commits_every_round_and_ends_finished_on_the_data_end(tmp_path):
    from test_s26_e9_data_slots import _fake_driver

    from yeto.rl.engine.bridges import LocalOnlySync

    driver = _fake_driver(tmp_path, LocalOnlySync(6), exhaust_at=99)
    commits, closes = [], []
    generate = driver.rollout.generate

    def gen(rollout_id, **kw):
        if rollout_id == 3:
            raise RuntimeError("ray task failed") from td.DispenseEnded("no rows left")
        return generate(rollout_id, **kw)

    driver.rollout.generate = gen
    driver.rollout.commit_dispensed = lambda rid: commits.append(rid) or {"trained": 1, "acked": True}
    driver.rollout.close_dispensed = lambda: closes.append(True) or {"released": 0}
    final = driver.run()
    assert final.policy_version == 3
    assert commits == [0, 1, 2] and closes == [True]
    events = _events(tmp_path)
    names = [e["event"] for e in events]
    assert "island_leave" not in names and "rl_buffer_exhausted" not in names
    ended = next(e for e in events if e["event"] == "rl_data_ended")
    assert ended["rollout_id"] == 3 and ended["source"] == "dispense_ended"
    assert [e["rollout_id"] for e in events if e["event"] == "rl_dispense_commit"] == [0, 1, 2]


def test_driver_with_rollout_ahead_trains_the_round_then_ends_on_the_data_end(tmp_path):
    from test_s27_rollout_ahead import _driver

    driver, engine = _driver(tmp_path, rounds=6, ahead=1)
    commits, closes = [], []
    driver.rollout.commit_dispensed = lambda rid: commits.append(rid) or {"trained": 1, "acked": True}
    driver.rollout.close_dispensed = lambda: closes.append(True) or {"released": 3}
    finish = driver.rollout.finish_generate

    def finish_generate(pending):
        if pending.rollout_id == 3:  # the executor call failed: the engines are idle
            engine.generating.discard(3)
            raise RuntimeError("ray task failed") from td.DispenseEnded("no rows left")
        return finish(pending)

    driver.rollout.finish_generate = finish_generate
    final = driver.run()
    assert final.policy_version == 3, "rounds 0, 1, 2 trained and published"
    assert commits == [0, 1, 2] and closes == [True]
    events = _events(tmp_path)
    ended = next(e for e in events if e["event"] == "rl_data_ended")
    assert ended["rollout_id"] == 3
    ahead = [e for e in events if e["event"] == "rl_rollout_ahead" and e.get("rollout_id") == 3]
    assert ahead and ahead[0].get("discarded") is True and "ray task failed" in ahead[0]["error"]
    assert "island_leave" not in [e["event"] for e in events]


def test_driver_on_a_strict_roster_still_fails_on_the_data_end(tmp_path):
    from test_s26_e9_data_slots import _fake_driver

    from yeto.rl.engine.bridges import LocalOnlySync

    sync = LocalOnlySync(5)
    sync.ends_on_data_end = False
    driver = _fake_driver(tmp_path, sync, exhaust_at=99)
    generate = driver.rollout.generate

    def gen(rollout_id, **kw):
        if rollout_id == 2:
            raise RuntimeError("ray task failed") from td.DispenseEnded("no rows left")
        return generate(rollout_id, **kw)

    driver.rollout.generate = gen
    with pytest.raises(RuntimeError, match="ray task failed"):
        driver.run()


def test_full_parameter_island_of_async_rounds_may_end_on_the_data_end():
    from yeto.rl.engine.full_parameter_sync import FullParameterSync

    def sync(partial_quorum):
        pairing = SimpleNamespace(streams={"actor": object()},
                                  profile=SimpleNamespace(partial_quorum=partial_quorum),
                                  fragment_versions=lambda: (3, 4))
        return FullParameterSync(pairing, groups={"actor": object()}, run=None, trajectories=None)

    events = []
    driver = SimpleNamespace(emit=lambda name, **f: events.append((name, f)))
    strict = sync(False)
    assert strict.ends_on_data_end is False
    with pytest.raises(RuntimeError, match="final cut"):
        strict.finish(driver)
    island = sync(True)
    assert island.ends_on_data_end is True
    island.end_on_data(driver)
    island.finish(driver)  # no final cut needed any more
    assert [name for name, _ in events] == ["rl_full_parameter_data_end"]
    assert "outer_version" in events[0][1]


def test_data_passes_are_refused_on_a_strict_roster():
    from yeto.launch.island_tasks import _strict_roster

    def args(**kw):
        return SimpleNamespace(**kw)

    assert _strict_roster(args(rl_sync_preset="full-parameter", rl_sync_full_parameter_rounds="sweep"))
    assert not _strict_roster(args(rl_sync_preset="full-parameter", rl_sync_full_parameter_rounds="async"))
    assert _strict_roster(args(rl_sync_preset="strict-avg"))
    assert not _strict_roster(args(rl_sync_preset="strict-avg", rl_sync_island_lease_s=60))
    assert not _strict_roster(args(rl_sync_preset="decoupled"))


# --------------------------------------------------------------------- the checker


def test_tape_rows_of_an_uncommitted_round_are_counted_not_a_problem():
    def rows(rollout_id, start, end, island=1):
        return {"island_id": island, "event": "rl_round_data_rows", "rollout_id": rollout_id,
                "rows": [[start, end - 1, 1]]}

    def commit(rollout_id, island=1):
        return {"island_id": island, "event": "rl_dispense_commit", "rollout_id": rollout_id,
                "rounds": [rollout_id], "trained": 2, "acked": True}

    def start():
        return {"island_id": 1, "event": "rl_driver_start"}

    tapes = [start(), rows(0, 0, 2), commit(0), rows(1, 2, 4),  # dies before the commit of round 1
             start(), rows(0, 4, 6), commit(0)]
    for t, record in enumerate(tapes):  # real records differ at least in their time
        record["time_unix"] = 1000.0 + t
    committed, uncommitted = tape_rows(tapes)
    assert sum(committed[1].values()) == 4 and sorted(uncommitted[1]) == [2, 3]
    # a relaunched island's tape repeats the earlier records: each counts once
    committed_twice, _ = tape_rows(tapes + tapes[:3])
    assert committed_twice[1] == committed[1]
    # a tape from before the commit existed: every round counts as committed
    committed, uncommitted = tape_rows([rows(0, 0, 2, island=0)])
    assert sum(committed[0].values()) == 2 and not uncommitted
    # rows 0..1 and 4..5 reported trained by island 1 (rows = positions in pass 0)
    summary = {"rows": 100, "trained_positions_by_island": {"1": [0, 1, 4, 5]}}
    info: dict = {}
    assert check_tapes(summary, tapes, seed=1, info=info) == []
    assert info["tape_rows_uncommitted"] == {"1": 2}
    summary = {"rows": 100, "trained_positions_by_island": {"1": [0, 1]}}
    assert check_tapes(summary, tapes, seed=1)  # rows 4, 5 committed but never reported


# --------------------------------------------------------------------- launcher evidence


def _dispense_args(**kw):
    return SimpleNamespace(_data_split_plan=SimpleNamespace(dispense=True), **kw)


def test_launcher_keeps_the_log_of_this_run_and_its_check(tmp_path, monkeypatch, capsys):
    from yeto.launch import dispense_evidence as ev

    home = tmp_path / "home"
    (home / "yeto-output").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    now = time.time()
    old = {"kind": "dispense_open", "island_id": 0, "incarnation": "aa", "rows": 4, "time_unix": now - 7200}
    records = [
        {"kind": "dispense_open", "island_id": 0, "incarnation": "bb", "rows": 4, "time_unix": now},
        {"kind": "dispense_grant", "island_id": 0, "incarnation": "bb", "ranges": [[0, 2]], "time_unix": now},
        {"kind": "dispense_report", "island_id": 0, "incarnation": "bb", "report": "trained",
         "ranges": [[0, 2]], "time_unix": now},
    ]
    (home / "yeto-output/task-dispense.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in [old, *records]))
    run_dir = tmp_path / "run"
    out = ev.collect_dispense_log(_dispense_args(), run_dir, since_unix=now - 10)
    assert out["copied"] and out["records"] == 3 and out["ok"] is True
    assert len((run_dir / "task-dispense.jsonl").read_text().splitlines()) == 3
    check_json = json.loads((run_dir / "dispense-check.json").read_text())
    assert check_json["trained"] == 2 and check_json["islands"]["0"]["incarnations"] == 1
    assert (home / "yeto-output/dispense-check.json").exists()
    assert "[launcher] task dispenser: ok=True" in capsys.readouterr().out
    # not a dispenser run: nothing; a missing log: reported, never raised
    assert ev.collect_dispense_log(SimpleNamespace(_data_split_plan=SimpleNamespace(dispense=False)),
                                   run_dir) is None
    (home / "yeto-output/task-dispense.jsonl").unlink()
    assert ev.collect_dispense_log(_dispense_args(), run_dir) == {"copied": False}


def test_head_run_pulls_the_log_and_check_from_the_head(tmp_path):
    from yeto.launch import dispense_evidence as ev

    calls = []

    def run(argv, **kw):
        calls.append(argv)
        return SimpleNamespace(returncode=0)

    out = ev.pull_from_head(_dispense_args(), "s27-x-head", tmp_path / "run", run=run)
    assert out == {"task-dispense.jsonl": "ok", "dispense-check.json": "ok"}
    assert [c[-2] for c in calls] == ["s27-x-head:~/yeto-output/task-dispense.jsonl",
                                      "s27-x-head:~/yeto-output/dispense-check.json"]
    assert ev.pull_from_head(SimpleNamespace(_data_split_plan=SimpleNamespace(dispense=False)),
                             "h", tmp_path, run=run) is None
