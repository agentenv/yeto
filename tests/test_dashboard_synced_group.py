"""A synced group's sampler islands on the dashboard: identity from the
per-cluster tape, per-round reward/grad_norm points, and a sampler parked at
its backlog cap is waiting, not stale."""

import json

from yeto.dashboard.reducer import Reducer
from yeto.dashboard.sources import TapeSource, cluster_tape_island

T0 = 1_700_000_000.0


def _tape(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_cluster_tape_island_needs_events_dir(tmp_path):
    assert cluster_tape_island(tmp_path / "events" / "k31-dlrl-l2-modal.jsonl") == "2"
    assert cluster_tape_island(tmp_path / "other" / "k31-dlrl-l2-modal.jsonl") is None
    assert cluster_tape_island(tmp_path / "events" / "fleet.jsonl") is None


def test_samplers_reporting_learner_zero_stay_separate(tmp_path):
    ev = tmp_path / "events"
    sampler = [{"event": "rl_sampler_start", "island_id": 0, "time_unix": T0},
               {"event": "rl_sampler_round", "island_id": 0, "time_unix": T0 + 5}]
    _tape(ev / "run-l2-modal.jsonl", sampler)
    _tape(ev / "run-l3-modal.jsonl", sampler)
    r = Reducer()
    for p in ("run-l2-modal.jsonl", "run-l3-modal.jsonl"):
        TapeSource(ev / p).pump(r)
    ids = {c["id"] for c in r.overview(now=T0 + 10)["islands"]}
    assert {"2", "3"} <= ids
    assert "0" not in ids


def test_round_trained_reward_and_grad_norm_points():
    r = Reducer()
    for k in (1, 2, 3):
        r.feed({"event": "rl_round_trained", "island_id": 0, "time_unix": T0 + k,
                "local_round_id": k, "rollout_id": k, "reward_mean": 0.1 * k, "grad_norm": 1.0 + k})
    isl = r.islands["0"]
    s = r.series(isl)
    assert [v for _, v in s["reward"]] == [0.1, 0.2, 0.30000000000000004]
    assert [v for _, v in s["grad_norm"]] == [2.0, 3.0, 4.0]


def test_sampler_waiting_on_backlog_is_not_stale():
    r = Reducer()
    r.feed({"event": "rl_sampler_start", "island_id": 2, "time_unix": T0})
    r.feed({"event": "rl_sampler_backlog_full", "island_id": 2, "time_unix": T0 + 1})
    ov = r.overview(now=T0 + 3600)
    card = next(c for c in ov["islands"] if c["id"] == "2")
    assert card["status"] == "waiting"
    assert card["waiting_backlog"] is True
    assert [a for a in ov["alerts"] if a["island"] == "2" and a["rule"] in ("heartbeat", "startup")] == []
    r.feed({"event": "rl_sampler_round", "island_id": 2, "time_unix": T0 + 3601})
    card = next(c for c in r.overview(now=T0 + 3602)["islands"] if c["id"] == "2")
    assert card["status"] == "ok"
