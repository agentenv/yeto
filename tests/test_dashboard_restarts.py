"""Trainer relaunches (store tape launch dirs), trainer start stages and sampler
in-place restarts, fed through StoreTapes from a fake local store."""

import json

from yeto.dashboard.reducer import Reducer
from yeto.dashboard.store_tapes import StoreTapes
from yeto.rl.engine.object_store import DirectoryObjectStore

T0 = 1_794_000_000.0
RUN = "k31-test"


def _put(root, island, inc, seq, recs):
    d = root / "tapes" / island / inc
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{seq:08d}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))


def _phase(phase, t):
    return {"event": "trainer_start_phase", "island_id": 0, "phase": phase, "time_unix": t}


def _pump(root):
    r = Reducer()
    StoreTapes(DirectoryObjectStore(str(root))).pump(r)
    return r


def _card(r, iid, now):
    return next(c for c in r.overview(now=now)["islands"] if c["id"] == iid)


def test_relaunched_trainer_starts_over_and_old_launch_is_superseded(tmp_path):
    old, new = "l0-modal-20261110T151448Z", "l0-modal-20261110T170000Z"
    _put(tmp_path, "trainer", old, 0, [_phase("submitted", T0), {"event": "rl_driver_start", "island_id": 0,
                                                                  "time_unix": T0 + 600}])
    t1 = T0 + 3600
    _put(tmp_path, "trainer", new, 0, [_phase("submitted", t1), _phase("container_ready", t1 + 60),
                                       _phase("setup_done", t1 + 120), _phase("ray_ready", t1 + 150),
                                       _phase("actor_import_done", t1 + 200)])
    r = _pump(tmp_path)
    now = t1 + 200 + 1500  # 25 min of silent weight load
    c = _card(r, "0", now)
    assert c["status"] == "starting" and not c["driver_started"]
    assert c["launch"] == new
    assert [(x["launch"], x["state"]) for x in c["launches"]] == [(old, "superseded"), (new, "current")]
    st = c["startup_stage"]
    assert st["init"] and "weights" in st["label"] and st["age_s"] == 1500 and st["warn_s"] == 2700
    assert c["startup_s"] == 1700
    al = r.overview(now=now)["alerts"]
    assert not [a for a in al if a["rule"] in ("startup", "startup_step", "heartbeat")]
    late = [a for a in r.overview(now=t1 + 200 + 2800)["alerts"] if a["rule"] == "startup"]
    assert len(late) == 1 and late[0]["sev"] == 1 and "actor init" in late[0]["detail"]


def test_late_record_of_superseded_launch_does_not_move_current_start(tmp_path):
    old, new = "l0-modal-20261110T151448Z", "l0-modal-20261110T170000Z"
    _put(tmp_path, "trainer", old, 0, [_phase("submitted", T0)])
    _put(tmp_path, "trainer", new, 0, [_phase("submitted", T0 + 3600), _phase("actor_import_done", T0 + 3700)])
    r = _pump(tmp_path)
    r.feed({"event": "rl_driver_start", "island_id": 0, "time_unix": T0 + 50},
           source=f"store:trainer/{old}", offset=1 << 32)
    c = _card(r, "0", T0 + 3800)
    assert c["starting"] and c["startup_stage"]["phase"] == "actor_import_done"


def test_startup_outside_init_keeps_default_threshold(tmp_path):
    _put(tmp_path, "trainer", "l0-modal-20261110T151448Z", 0, [_phase("submitted", T0)])
    r = _pump(tmp_path)
    c = _card(r, "0", T0 + 1300)
    assert not c["startup_stage"]["init"] and c["startup_stage"]["warn_s"] == 1200
    assert [a for a in r.overview(now=T0 + 1300)["alerts"] if a["rule"] == "startup"]


def _sampler(inc, rid, t, *, groups=4, reward=None):
    sid = f"{RUN}-0"
    out = [{"event": "rl_sampler_round", "sampler_id": sid, "rollout_id": rid, "groups": groups,
            "generate_s": 30.0, "time_unix": t}]
    if reward is not None:
        out.append({"event": "rl_sampler_engine_round", "sampler_id": sid, "rollout_id": rid,
                    "samples": 32, "reward_mean": reward, "time_unix": t})
    return out


def test_sampler_restart_keeps_one_continuous_series(tmp_path):
    sid = f"{RUN}-0"
    recs = [{"event": "rl_sampler_join", "sampler_id": sid, "sampler": 0, "incarnation": "aaaa", "time_unix": T0}]
    for i in range(3):
        recs += _sampler("aaaa", i, T0 + 60 * (i + 1), reward=0.1 * (i + 1))
    t1 = T0 + 1000
    recs.append({"event": "rl_sampler_join", "sampler_id": sid, "sampler": 0, "incarnation": "bbbb", "time_unix": t1})
    for i in range(2):  # the restarted sampler counts its rounds from 0 again
        recs += _sampler("bbbb", i, t1 + 60 * (i + 1), reward=0.5 + 0.1 * i)
    for i in range(2, 800):  # then idles: empty rounds (backlog full)
        recs += _sampler("bbbb", i, t1 + 200 + i, groups=0)
    _put(tmp_path, sid, "5ce0ed95", 0, recs)
    r = _pump(tmp_path)
    s = r.overview(now=t1 + 1100) and r.async_.view()["samplers"][sid]
    rewards = [x["reward_mean"] for x in s["rounds"] if x.get("reward_mean") is not None]
    assert rewards == [0.1, 0.2, 0.30000000000000004, 0.5, 0.6]  # both incarnations, time order
    assert s["n_rounds"] == 803 and s["empty_rounds"] == 798 and s["last_round"]["groups"] == 0
    assert s["reward_mean_last"] == 0.6 and s["reward_last_rollout_id"] == 1
    assert [x["incarnation"] for x in s["restarts"]] == ["bbbb"] and s["restarts"][0]["ts"] == t1
    assert s["incarnation"] == "bbbb"
    assert len(r.island_ids()) == 1  # one sampler island across incarnations


def test_old_incarnation_points_kept_when_new_one_is_in_an_earlier_sorted_folder(tmp_path):
    sid = f"{RUN}-0"
    old = [{"event": "rl_sampler_join", "sampler_id": sid, "sampler": 0, "incarnation": "5ce0ed95", "time_unix": T0}]
    for i in range(3):
        old += _sampler("5ce0ed95", i, T0 + 60 * (i + 1), reward=0.375)
    t1 = T0 + 1200
    new = [{"event": "rl_sampler_join", "sampler_id": sid, "sampler": 0, "incarnation": "a963a865", "time_unix": t1}]
    for i in range(3, 900):
        new += _sampler("a963a865", i, t1 + i, groups=0)
    new += _sampler("a963a865", 900, t1 + 1000, reward=0.5)
    _put(tmp_path, sid, "5ce0ed95", 0, old)
    _put(tmp_path, sid, "0000aaaa", 0, new)  # listed (and read) before the old folder
    r = _pump(tmp_path)
    s = r.async_.view()["samplers"][sid]
    pts = [(x["incarnation"], x["reward_mean"]) for x in s["rounds"] if x.get("reward_mean") is not None]
    assert pts == [("5ce0ed95", 0.375)] * 3 + [("a963a865", 0.5)]
    assert [x["incarnation"] for x in s["incarnations"]] == ["5ce0ed95", "a963a865"]
    assert [x["incarnation"] for x in s["restarts"]] == ["a963a865"]
    assert s["incarnation"] == "a963a865"


def test_store_only_run_infers_hardware_cost_and_read_position(tmp_path):
    sid = f"{RUN}-0"
    gpus = [{"index": i, "mem_total_mb": 275040.0, "mem_used_mb": 900.0, "util_pct": 0} for i in range(8)]
    _put(tmp_path, "trainer", "l0-modal-20261110T151448Z", 0, [
        _phase("submitted", T0),
        {"event": "remote_code_prewarm", "island_id": 0, "nodes": [{"node": "a"}, {"node": "b"}], "time_unix": T0 + 10},
        {"event": "rl_resource_sample", "island_id": 0, "available": True, "gpus": gpus, "time_unix": T0 + 20},
        {"event": "rl_driver_start", "island_id": 0, "time_unix": T0 + 30},
        {"event": "rl_buffer_stale_skipped", "island_id": 0, "rollout_id": 0, "stale_groups": 0, "time_unix": T0 + 3600},
        {"event": "rl_group_policies", "island_id": 0, "rollout_id": 0, "groups": [["g0", "p", [0]], ["g1", "p", [0]]],
         "time_unix": T0 + 3600}])
    _put(tmp_path, sid, "5ce0ed95", 0, [
        {"event": "rl_sampler_start", "sampler_id": sid, "provenance": {"card_model": "MI355X", "topology": "tp8"},
         "time_unix": T0}] + _sampler("x", 0, T0 + 1800, groups=5, reward=0.5))
    _put(tmp_path, "pool", "controller", 0, [{"event": "nc_elastic_member_ready", "shard": sid, "t": T0 - 10}])
    r = _pump(tmp_path)
    ov = r.overview(now=T0 + 3600)
    cards = {c["id"]: c for c in ov["islands"]}
    assert (cards["0"]["cloud"], cards["0"]["gpu"], cards["0"]["gpus"]) == ("modal", "B300", 16)
    assert (cards[sid]["cloud"], cards[sid]["gpu"], cards[sid]["gpus"]) == ("nc", "MI355X", 8)
    rows = {x["id"]: x for x in ov["cost"]["islands"]}
    assert abs(rows["0"]["rate_usd_h"] - 136.0) < 1e-6 and abs(rows["0"]["cost_usd"] - 136.0) < 1e-6
    assert abs(rows[sid]["rate_usd_h"] - 24.0) < 1e-6 and abs(rows[sid]["cost_usd"] - 12.0) < 1e-6
    b = r.async_.view()["backlog"]
    assert b["has_reads"] and b["read"] == 2 and b["read_source"] == "trainer groups consumed"


def test_failed_trainer_does_not_end_samplers_and_relaunch_is_starting(tmp_path):
    sid = f"{RUN}-0"
    old, new = "l0-modal-20261110T151448Z", "l0-modal-20261110T164111Z"
    _put(tmp_path, "trainer", old, 0, [
        _phase("submitted", T0), {"event": "rl_driver_start", "island_id": 0, "time_unix": T0 + 30},
        {"event": "island_leave", "island_id": 0, "reason_name": "INFRA_FAILED", "time_unix": T0 + 3000}])
    _put(tmp_path, sid, "a963a865", 0, [
        {"event": "rl_sampler_join", "sampler_id": sid, "sampler": 0, "incarnation": "a963a865", "time_unix": T0}]
        + _sampler("a963a865", 0, T0 + 3100, reward=0.5))
    r = _pump(tmp_path)
    cards = {c["id"]: c for c in r.overview(now=T0 + 3130)["islands"]}
    assert cards["0"]["status"] == "left"  # no new launch yet
    assert cards[sid]["status"] not in ("done", "left", "lost")  # a failed trainer does not end the samplers
    _put(tmp_path, "trainer", new, 0, [_phase("submitted", T0 + 3200), _phase("actor_import_done", T0 + 3300)])
    r = _pump(tmp_path)
    cards = {c["id"]: c for c in r.overview(now=T0 + 3400)["islands"]}
    assert cards["0"]["status"] == "starting" and cards["0"]["left"] is None
    assert cards[sid]["status"] not in ("done", "left", "lost")


def test_cleanly_stopped_trainer_ends_samplers(tmp_path):
    sid = f"{RUN}-0"
    _put(tmp_path, "trainer", "l0-modal-20261110T151448Z", 0, [
        {"event": "rl_driver_start", "island_id": 0, "time_unix": T0},
        {"event": "island_leave", "island_id": 0, "reason_name": "STOPPED", "time_unix": T0 + 100}])
    _put(tmp_path, sid, "a963a865", 0, _sampler("a", 0, T0 + 50, reward=0.5))
    r = _pump(tmp_path)
    assert _card(r, sid, T0 + 200)["status"] == "done"
