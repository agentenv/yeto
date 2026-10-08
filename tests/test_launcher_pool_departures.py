"""S16 §6.6: an island the elastic syncer removed from the pool owes no
rl_learner_finalized; only islands neither departed nor finalized keep exit 3."""
import json

from yeto import launcher


def _tape(tmp_path, records, prefix=""):
    p = tmp_path / "yeto-tape.jsonl"
    p.write_text(prefix + "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return p


def test_departures_last_pool_record_wins(tmp_path):
    tape = _tape(tmp_path, [
        {"kind": "pool_join", "island_id": 0}, {"kind": "pool_join", "island_id": 1},
        {"kind": "pool_leave", "island_id": 0, "reason": "requested"},
        {"kind": "pool_join", "island_id": 0, "catch_up": True},
        {"kind": "outer_step", "outer_version": 2},
        {"kind": "pool_leave", "island_id": 1, "reason": "lease_expired"},
    ])
    assert launcher.syncer_pool_departures(tape) == {1: "lease_expired"}


def test_departures_rejoined_island_not_departed(tmp_path):
    tape = _tape(tmp_path, [
        {"kind": "pool_leave", "island_id": 1, "reason": "lease_expired"},
        {"kind": "pool_join", "island_id": 1, "catch_up": True},
    ])
    assert launcher.syncer_pool_departures(tape) == {}


def test_departures_offset_skips_previous_run(tmp_path):
    old = json.dumps({"kind": "pool_leave", "island_id": 0, "reason": "requested"}) + "\n"
    tape = _tape(tmp_path, [{"kind": "pool_join", "island_id": 1}], prefix=old)
    assert launcher.syncer_pool_departures(tape, offset=len(old)) == {}
    assert launcher.syncer_pool_departures(tape) == {0: "requested"}


def test_departures_missing_or_garbled_tape(tmp_path):
    assert launcher.syncer_pool_departures(tmp_path / "nope.jsonl") == {}
    p = tmp_path / "t.jsonl"
    p.write_text("not json\n[1]\n" + json.dumps(
        {"kind": "pool_leave", "island_id": 2, "reason": "requested"}) + "\n")
    assert launcher.syncer_pool_departures(p) == {2: "requested"}


def test_excuse_departed_keeps_exit3_for_others():
    still, left = launcher.excuse_departed_islands(
        ["run-l0-modal", "run-l1-modal", "odd-name"], {1: "lease_expired"})
    assert still == ["run-l0-modal", "odd-name"]
    assert left == [{"island": 1, "name": "run-l1-modal", "reason": "lease_expired"}]
    assert launcher.NO_SYNC_INCOMPLETE_EXIT == 3
