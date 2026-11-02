"""The spot run patrol reads tapes / usage offline and plans the stop-loss."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yeto.rl.spot import patrol

NOW = 1_800_000_000.0


def _write(path: Path, records, *, log: bool = False) -> str:
    lines = [("YETO_RL_EVENT " if log else "") + json.dumps(r) for r in records]
    path.write_text("noise line\n" + "\n".join(lines) + "\n{torn")
    return str(path)


def _round(i, *, t=NOW - 60, train_s=150.0, grad=0.4, zero_var=0.25, rollout_s=100.0):
    return {"event": "rl_local_round", "local_round_id": i, "time_unix": t, "train_seconds": train_s,
            "grad_norm": grad, "zero_variance_group_ratio": zero_var, "rollout_seconds": rollout_s}


def _args(**over):
    args = patrol.parser().parse_args([])
    for k, v in over.items():
        setattr(args, k, v)
    return args


def test_quiet_run_has_no_alerts(tmp_path):
    tape = _write(tmp_path / "t.log", [_round(i) for i in range(1, 6)], log=True)
    report = patrol.patrol(_args(tape=tape), now=NOW)
    assert report["alerts"] == [] and report["steps"] == 5 and patrol.exit_code(report) == 0


def test_trainer_thresholds(tmp_path):
    th = patrol.Thresholds()
    events = [_round(1, train_s=900.0), _round(2, train_s=300.0, t=NOW - 3600, rollout_s=10),
              _round(3, grad=float("nan"), t=NOW - 3600)]
    events += [{"event": "rl_verl_fa_round", "pulled": 10, "kept": 8, "discarded": 3, "uniform_reward_groups": 7}]
    keys = {a.key: a for a in patrol.trainer_alerts(events, NOW, th)}
    assert keys["train_s_high"].value == 300.0  # step 1 excluded
    assert keys["grad_norm_bad"].severity == "stop"
    assert keys["waiting_for_samples"].value == 3600
    assert keys["drop_rate_high"].value == 0.3
    assert keys["nonzero_adv_groups_low"].value == 0.125


def test_memory_floor_and_rss_slope():
    th = patrol.Thresholds()
    records = [{"event": "rl_host_memory", "phase": "after_step", "rollout_id": i,
                "nodes": {"n1": {"mem_available_mib": 300 * 1024, "workers_rssanon_mib": 1024 * (100 + 3 * i)},
                          "n2": {"mem_available_mib": (150 if i == 4 else 300) * 1024,
                                 "workers_rssanon_mib": 1024 * 100}}} for i in range(5)]
    keys = {a.key: a for a in patrol.memory_alerts(records, th)}
    assert keys["mem_available_low"].value == {"n2": 150.0}
    assert keys["rssanon_slope"].value == {"n1": 3.0}


def test_consumed_and_sampler_and_boat_and_vm(tmp_path):
    th = patrol.Thresholds()
    consumed = [{"kept": True}] * 6 + [{"kept": False, "dropped": "short_group"}] * 3 \
        + [{"kept": False, "dropped": "stale_incarnation"}]
    keys = {a.key for a in patrol.consumed_alerts(consumed, th)}
    assert keys == {"feed_drop_rate_high", "stale_incarnation_groups"}

    sampler = {"events.aaaa.jsonl": [
        {"event": "rl_sampler_engine_round", "time_unix": NOW - 60, "policy_version": 2, "groups": 0},
        {"event": "rl_agentic_group", "time_unix": NOW - 3000, "written": True, "version_lag": 3}],
        "events.old.jsonl": [{"event": "rl_sampler_leave", "time_unix": NOW - 99999}]}
    trainer = [_round(i, t=NOW - 5000 + i) for i in range(1, 5)]
    alerts = {a.key: a for a in patrol.sampler_alerts(sampler, trainer, NOW, th)}
    assert set(alerts) == {"island_no_output", "policy_lag", "release_delay"}
    assert alerts["release_delay"].target == "events.aaaa.jsonl"

    boat = {"starts": {"day": {"used": 170, "limit": 200}}, "activeSandboxes": 10, "maxActiveSandboxes": 100,
            "keepGoing": {"usedDollars": 10, "capDollars": 80}}
    assert [a.key for a in patrol.boat_alerts(boat, th)] == ["boat_quota"]

    doc = {"items": [
        {"metadata": {"id": "vm-1", "name": "spot-a", "created_at": "2027-01-15T07:00:00Z"},
         "status": {"state": "STARTING"}},
        {"metadata": {"id": "vm-2", "name": "other", "created_at": "2027-01-15T07:00:00Z"},
         "status": {"state": "RUNNING"}}]}
    vms = patrol.nebius_instances(doc, "spot")
    assert [v["id"] for v in vms] == ["vm-1"]
    stuck = patrol.vm_alerts(vms, vms[0]["since"] + 11 * 60, th)
    assert stuck[0].key == "vm_starting_stuck" and stuck[0].target == "vm-1"


def test_read_only_by_default_and_act_runs_the_planned_stop(tmp_path):
    tape = _write(tmp_path / "t.jsonl", [_round(1), _round(2, grad=0.0)])
    boat = tmp_path / "boat.json"
    boat.write_text(json.dumps({"starts": {"day": {"used": 10, "limit": 200}}}))
    calls = []

    def runner(argv, stdin):
        calls.append((list(argv), stdin))
        if argv[:3] == ["nebius", "compute", "instance"] and argv[3] == "list":
            return 0, json.dumps({"items": [{"metadata": {"id": "vm-9", "name": "spot-runx"},
                                             "status": {"state": "RUNNING"}}]})
        return 0, "ok"

    common = dict(tape=tape, boat_json=str(boat), nebius_cmd="nebius compute instance list --format json",
                  run="runx", kill_script="/tmp/kill.sh", boat_run="runx", trainer_host="trainer-host")
    report = patrol.patrol(_args(**common), now=NOW, runner=runner)
    assert patrol.exit_code(report) == 2
    assert [c[0][3] for c in calls] == ["list"]  # read-only: only the listing ran
    assert report["planned_actions"][0] == "ssh trainer-host bash -s -- runx < /tmp/kill.sh"
    assert "nebius compute instance delete --id vm-9" in report["planned_actions"]

    calls.clear()
    report = patrol.patrol(_args(act=True, **common), now=NOW, runner=runner)
    ran = [c[0] for c in calls[1:]]
    assert ran[0] == ["ssh", "trainer-host", "bash", "-s", "--", "runx"] and calls[1][1] == "/tmp/kill.sh"
    assert ran[1] == ["nebius", "compute", "instance", "delete", "--id", "vm-9"]
    assert ran[2][1:] == ["-m", "yeto.rl.harness.harbor.boat_labels", "sweep", "--run", "runx"]
    assert all(a["exit"] == 0 for a in report["actions"])


def test_stop_on_narrows_the_stop_alerts(tmp_path):
    tape = _write(tmp_path / "t.jsonl", [_round(1), _round(2, grad=0.0)])
    report = patrol.patrol(_args(tape=tape, stop_on=frozenset()), now=NOW)
    assert patrol.exit_code(report) == 1 and report["planned_actions"] == []


@pytest.mark.parametrize("spent, alerts", [(150.0, ["spend_budget"]), (100.0, [])])
def test_spend(spent, alerts):
    assert [a.key for a in patrol.spend_alerts(spent, 180.0, patrol.Thresholds())] == alerts


def test_without_a_trainer_host_the_trainer_is_not_stopped(tmp_path, monkeypatch):
    tape = _write(tmp_path / "t.jsonl", [_round(1), _round(2, grad=0.0)])
    monkeypatch.delenv("TRAINER_HOST", raising=False)
    assert patrol.parser().parse_args([]).trainer_host is None
    report = patrol.patrol(_args(tape=tape, run="r", kill_script="/tmp/kill.sh", trainer_host=None), now=NOW)
    assert not any(a.startswith("ssh ") for a in report["planned_actions"])
    monkeypatch.setenv("TRAINER_HOST", "from-env")
    assert patrol.parser().parse_args([]).trainer_host == "from-env"


def test_act_with_a_nebius_source_requires_a_vm_filter(tmp_path):
    doc = tmp_path / "vms.json"
    doc.write_text("{}")
    with pytest.raises(SystemExit) as exc:
        patrol.main(["--act", "--nebius-json", str(doc)])
    assert exc.value.code == 2
