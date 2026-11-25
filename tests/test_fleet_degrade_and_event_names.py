"""Run up starts with the pools that pass (a pool that is not ``required`` and fails its
preflight is degraded, a required one refuses the run), and the neutral pool event names
(``rl_pool_*``) next to the legacy ``nc_elastic_*`` ones (fakes; no cloud)."""

from __future__ import annotations

import json

import pytest

from tests.test_rollout_entry import _two_pools
from yeto.breakers.registry import default_registry
from yeto.dashboard import alerts, cost as dash_cost
from yeto.dashboard.reducer import Reducer, pool_record
from yeto.launch.pool.status import EventLog, legacy_name, neutral_name
from yeto.rl import up


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("YETO_NC_ACCOUNTS", "1,2,3,4")
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    monkeypatch.setattr(up.Run, "budget_problem", lambda self: None)
    monkeypatch.setattr(up.Run, "stall_timeout_s", lambda self: 900.0)


_CFG: dict = {}


def _run(tmp_path, *, required_modal: bool | None, failing: set[str]):
    if tmp_path not in _CFG:
        _CFG[tmp_path] = _two_pools(tmp_path)[0]
    run_cfg = json.loads(json.dumps(_CFG[tmp_path]))
    if required_modal is not None:
        run_cfg["samplers"][1]["required"] = required_modal
    run = up.Run(run_cfg)
    configs = {ref.config: ref.name for ref in run.pools}

    def cmd(argv, *, log, timeout=None):
        if "pool" in argv and "--dry-run" in argv:
            return 1 if configs[argv[argv.index("--config") + 1]] in failing else 0
        return 0

    run._cmd = cmd
    return run


def test_a_pool_that_is_not_required_and_fails_its_preflight_is_degraded(tmp_path, capsys):
    run = _run(tmp_path, required_modal=False, failing={"modal"})
    assert run.preflight() is True
    assert [r.name for r in run.pools] == ["nc"]
    assert "pool dry-run failed" in run.state["degraded_pools"]["modal"]
    out = capsys.readouterr().out
    assert "pool modal is not required and its preflight failed: DEGRADED" in out
    events = [json.loads(line) for line in (run.dir / "events.jsonl").read_text().splitlines()]
    assert [e["pool"] for e in events if e["event"] == "rl_pool_degraded"] == ["modal"]


def test_a_required_pool_refuses_the_run_as_before(tmp_path):
    assert _run(tmp_path, required_modal=None, failing={"modal"}).preflight() is False  # required by default
    assert _run(tmp_path, required_modal=True, failing={"modal"}).preflight() is False


def test_a_config_error_refuses_the_run_even_for_a_pool_that_is_not_required(tmp_path, monkeypatch):
    """REV #757: only what the environment decides (the pool dry-run) degrades a pool; a shard,
    field or consistency problem is the run config's and still refuses the start."""
    run = _run(tmp_path, required_modal=False, failing=set())
    monkeypatch.setattr(up.Run, "consistency_problem",
                        lambda self, i: "inconsistent" if self.pools[i].name == "modal" else None)
    assert run.preflight() is False and not run.state.get("degraded_pools")


def test_every_pool_degraded_refuses_the_run(tmp_path, monkeypatch):
    run = _run(tmp_path, required_modal=False, failing={"modal", "nc"})
    run.pools = [run.pools[0].__class__(**{**run.pools[0].__dict__, "required": False}), run.pools[1]]
    assert run.preflight() is False


def test_a_later_clean_preflight_clears_the_degraded_pools(tmp_path):
    run = _run(tmp_path, required_modal=False, failing={"modal"})
    assert run.preflight() and run.state["degraded_pools"]
    again = _run(tmp_path, required_modal=False, failing=set())
    assert again.preflight() and again.state["degraded_pools"] == {}


def test_required_must_be_a_bool(tmp_path):
    run_cfg, _, _ = _two_pools(tmp_path)
    run_cfg["samplers"][1]["required"] = "no"
    with pytest.raises(up.PoolConfigError, match="required must be true or false"):
        up.Run(run_cfg)


def test_the_dashboard_warns_about_a_degraded_pool(tmp_path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"degraded_pools": {"daytona": "no DAYTONA_API_KEY"}}))
    view = dash_cost.run_up_view(str(state), 1.0)
    assert view["degraded_pools"] == {"daytona": "no DAYTONA_API_KEY"}
    [alert] = alerts.degraded_pools({"run_up": view})
    assert alert["title"] == "pool daytona degraded" and "DAYTONA_API_KEY" in alert["detail"]


# -- event names ----------------------------------------------------------------------

def test_dual_names_carry_the_neutral_name_next_to_the_legacy_one(tmp_path):
    log = EventLog(path=tmp_path / "e.jsonl", log=lambda _m: None, clock=lambda: 1.0)
    log("ordered", ticket="t1")
    assert "neutral_event" not in log.events[-1]  # off by default: the records as before
    log.dual_names = True
    log("ordered", ticket="t2")
    doc = json.loads((tmp_path / "e.jsonl").read_text().splitlines()[-1])
    assert doc["event"] == "nc_elastic_ordered" and doc["neutral_event"] == "rl_pool_ordered"
    assert neutral_name("nc_elastic_released") == "rl_pool_released" == neutral_name("rl_pool_released")
    assert legacy_name("rl_pool_released") == "nc_elastic_released" == legacy_name("nc_elastic_released")


def test_the_dashboard_folds_either_name_alike():
    assert pool_record({"event": "rl_pool_member_ready", "shard": "s-0"})["event"] == "nc_elastic_member_ready"
    assert pool_record({"event": "nc_elastic_member_ready"})["event"] == "nc_elastic_member_ready"
    folded = []
    for name in ("nc_elastic_ordered", "rl_pool_ordered"):
        r = Reducer()
        r.feed({"event": name, "t": 10.0, "ticket": "t1", "shard": "s-0", "account": 1})
        folded.append([e["type"] for e in r.events])
    assert folded[0] == folded[1] == ["nc_elastic_ordered"]


def test_a_dual_named_tape_reads_like_a_legacy_one_and_a_neutral_only_one(tmp_path):
    """The compatibility window end to end: an EventLog with dual_names writes records the
    dashboard folds exactly like the legacy records, and like records that carry only the
    neutral name (after the window)."""
    tapes = {}
    for mode in ("legacy", "dual", "neutral"):
        log = EventLog(path=tmp_path / f"{mode}.jsonl", log=lambda _m: None, clock=lambda: 10.0)
        log.dual_names = mode == "dual"
        log("ordered", ticket="t1", shard="s-0", account=1)
        log("member_ready", ticket="t1", shard="s-0", account=1)
        log("released", ticket="t1", shard="s-0", account=1)
        records = [json.loads(line) for line in (tmp_path / f"{mode}.jsonl").read_text().splitlines()]
        if mode == "neutral":
            records = [{**r, "event": neutral_name(r["event"])} for r in records]
        r = Reducer()
        for record in records:
            r.feed(record)
        tapes[mode] = [(e["type"], e.get("shard")) for e in r.events]
    assert tapes["legacy"] == tapes["dual"] == tapes["neutral"] and len(tapes["legacy"]) == 3


def test_breakers_know_both_names_of_every_legacy_pool_event():
    for breaker in default_registry():
        if breaker.layer != "pool":
            continue
        legacy = [e for e in breaker.legacy_events if e.startswith("nc_elastic_")]
        assert all(neutral_name(e) in breaker.legacy_events for e in legacy), breaker.name
