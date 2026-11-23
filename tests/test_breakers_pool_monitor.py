"""The pool and monitor layers of the breaker registry: declarations match the code they
describe, the pool reports its trips as ``rl_breaker`` next to its legacy events, a run's
``breakers:`` section reaches the pool's elastic knobs (budget inherited from ``run.budget``),
and the dashboard thresholds come from the monitor layer unchanged."""

from __future__ import annotations

import ast
import importlib.util
import json
from dataclasses import fields
from pathlib import Path

import pytest
from test_nc_elastic import Clock, Cloud, _names, _pool, _run

from yeto.breakers import events as breaker_events
from yeto.breakers.layers import monitor, pool
from yeto.breakers.registry import Registry, load_layers
from yeto.breakers.resolve import Sources, resolve
from yeto.breakers.spec import Breaker, BreakerConfigError, Param
from yeto.dashboard import alerts
from yeto.launch.nc import cli as nc_cli
from yeto.launch.pool import status as el
from yeto.launch.pool.island_pool import GONE, PoolConfig
from yeto.launch.rollout import pools as rollout_pools
from yeto.rl import breaker as health

H = 3600.0


def _registry() -> Registry:
    """Every declared layer; while the run layer is not on this branch, a stand-in
    ``run.budget`` (the parent ``pool.spend_cap.usd`` inherits from)."""
    if importlib.util.find_spec("yeto.breakers.layers.run") is not None:
        return load_layers()
    stub = Breaker(name="run.budget", layer="run", kind="breaker",
                   params=(Param("usd", float, None, unit="usd", lo=0, nullable=True),),
                   actions=("stop",), default_action="stop", default_enabled=False, enabled_by=("usd",),
                   aliases=(),)
    return Registry(pool.BREAKERS + monitor.BREAKERS + (stub,))


REG = _registry()


def _resolve(**surfaces):
    return resolve(Sources(**surfaces), registry=REG, layers=("pool", "monitor"))


# -- declarations -------------------------------------------------------------

def test_layers_load_and_stay_in_their_layer():
    assert all(b.layer == "pool" and b.name.startswith("pool.") for b in pool.BREAKERS)
    assert all(b.layer == "monitor" and b.kind in ("alert", "semantic") for b in monitor.BREAKERS)
    assert {b.name for b in pool.BREAKERS} <= {b.name for b in REG}


def test_every_elastic_alias_is_an_elastic_field_with_the_registry_default():
    elastic = {f.name for f in fields(el.ElasticConfig)}
    defaults = el.ElasticConfig()
    for b in pool.BREAKERS:
        for a in b.aliases:
            if a.surface != "pool_elastic":
                continue
            assert a.path in elastic, f"{b.name}: {a.path} is no ElasticConfig field"
            want = getattr(defaults, a.path)
            got = b.default_action if a.param == "action" else b.param(a.param).default
            assert got == want, f"{b.name}: {a.path} default {want!r}, registry {got!r}"


def test_lease_aliases_are_pool_config_keys_with_the_pool_defaults():
    lease = REG.get("pool.lease_end")
    by_name = {f.name: f.default for f in fields(PoolConfig)}
    for a in lease.aliases:
        assert a.surface == "pool_config" and a.path in nc_cli.POOL_KEYS
        assert lease.param(a.param).default == by_name[a.path]


def test_entries_without_an_off_today_cannot_be_disabled():
    """Only knobs that have an off spelling today (null) can be turned off."""
    off = {b.name for b in pool.BREAKERS if any(a.off_values for a in b.aliases) or b.enabled_by}
    assert off == {"pool.engine_stall", "pool.sampler_stall", "pool.fence_hold", "pool.spend_cap"}
    assert all(not b.can_disable for b in pool.BREAKERS if b.name not in off)


def test_defaults_resolve_to_todays_elastic_config():
    rb = _resolve(pool_elastic={})
    e = el.ElasticConfig()
    assert rb.get("pool.provision_breaker.failures_n") == e.breaker_failures
    assert rb.get("pool.sampler_restarts.max_n") == 2 and rb.get("pool.engine_stall.limit_s") == 600.0
    assert rb.enabled("pool.engine_stall") and not rb.enabled("pool.sampler_stall")
    assert not rb.enabled("pool.spend_cap") and rb.get("pool.spend_cap.usd") is None
    assert rb.entry("pool.crash_loop").action == "replace"
    assert rb.get("pool.node_cap.max_nodes_n") is None  # null: 2 x target, applied at run time


def test_legacy_elastic_values_and_off_spellings():
    rb = _resolve(pool_elastic={"engine_stall_s": None, "stall_s": 900, "max_sampler_restarts": 0,
                                "crash_loop_before_ready": "halt", "max_run_spend_usd": 45})
    assert not rb.enabled("pool.engine_stall")
    assert rb.enabled("pool.sampler_stall") and rb.get("pool.sampler_stall.limit_s") == 900
    assert rb.enabled("pool.sampler_restarts") and rb.get("pool.sampler_restarts.max_n") == 0  # 0 is not off
    assert rb.entry("pool.crash_loop").action == "halt"
    assert rb.enabled("pool.spend_cap") and rb.get("pool.spend_cap.usd") == 45


def test_spend_cap_inherits_the_run_budget_only_from_the_section():
    legacy = _resolve(run_config={"budget": {"usd": 450}}, pool_elastic={})
    assert not legacy.enabled("pool.spend_cap")  # budget.usd alone: today's pool has no cap
    section = _resolve(run_config={"breakers": {"run": {"budget": {"usd": 450}}}}, pool_elastic={})
    assert section.enabled("pool.spend_cap") and section.get("pool.spend_cap.usd") == 450
    assert section.entry("pool.spend_cap").sources["usd"].kind == "inherit"
    own = _resolve(run_config={"breakers": {"run": {"budget": {"usd": 450}}}},
                   pool_elastic={"max_run_spend_usd": 100})
    assert own.get("pool.spend_cap.usd") == 100  # the pool's own cap wins (D11 checks it is not larger)


def test_section_and_elastic_disagreeing_is_a_conflict():
    with pytest.raises(BreakerConfigError, match="pool.start_timeout"):
        _resolve(run_config={"breakers": {"pool": {"start_timeout": {"limit_s": 1800}}}},
                 pool_elastic={"start_timeout_s": 3600})


def test_monitor_health_limits_are_the_check_constants():
    for key, (value, direction) in health.LIMITS.items():
        if key.startswith("nan_skip"):
            continue
        b = REG.get(f"monitor.health_{key}")
        assert b.direction == direction and b.params[0].default == value
    nan = REG.get("monitor.health_nan_skip")
    assert nan.param("max_consecutive_n").default == health.LIMITS["nan_skip_consecutive"][0]
    assert nan.param("max_fraction").default == health.LIMITS["nan_skip_fraction"][0]
    assert nan.param("min_rounds_n").default == health.NAN_SKIP_MIN_ROUNDS
    logprob = REG.get("monitor.health_logprob")
    for hw, (warn, act) in health.LOGPROB_LIMITS.items():
        assert (logprob.param(f"{hw}_warn_abs_diff").default, logprob.param(f"{hw}_act_abs_diff").default) == (
            warn, act)


# -- dashboard thresholds -----------------------------------------------------

# today's table, written out once more: the derivation must keep every key, value and type
TODAY = {
    "heartbeat_warn_s": 60.0, "heartbeat_severe_s": 300.0, "startup_warn_s": 1200.0, "startup_severe_s": 1800.0,
    "startup_init_warn_s": 2700.0, "startup_init_severe_s": 3600.0, "startup_step_warn_s": 900.0,
    "missed_consecutive_warn": 2, "missed_consecutive_severe": 5, "quorum_recent_rounds": 7,
    "quorum_min_rounds": 10, "quorum_factor": 3.0, "grad_norm_window": 8, "grad_norm_spike_factor": 5.0,
    "clipfrac_info": 0.2, "kl_warn": 0.1, "budget_warn_pct": 60.0, "budget_severe_pct": 80.0,
    "sft_ref_step_s": 0.0, "sft_hbm_limit_gib": 0.0, "sft_syncer_rss_limit_bytes": 0.0, "sft_slow_factor": 1.1,
    "sft_stop_factor": 1.5, "sft_stop_slack_s": 300.0, "compaction_window_rounds": 5, "compaction_min_trials": 5,
    "compaction_summary_ok_warn": 0.9, "compaction_summary_ok_severe": 0.5, "compaction_truncated_warn": 0.2,
    "compaction_frac_info": 0.5, "compaction_long_loss_gap": 0.2, "compaction_fork_lost_warn": 0.05,
    "trial_fail_window_rounds": 10, "trial_fail_min_trials": 8, "trial_fail_warn": 0.5,
    "trainer_missing_warn_s": 1200.0, "trainer_missing_severe_s": 1800.0, "trainer_fail_window_s": 3600.0,
    "trainer_fail_loop": 2, "stale_read_recent_s": 300.0, "release_stall_warn_s": 600.0,
}


def test_dashboard_thresholds_are_derived_unchanged():
    assert list(alerts.DEFAULT_THRESHOLDS) == list(TODAY)
    for k, v in TODAY.items():
        got = alerts.DEFAULT_THRESHOLDS[k]
        assert got == v and type(got) is type(v), k
    assert alerts.merged_thresholds({"heartbeat_warn_s": 5})["heartbeat_warn_s"] == 5.0
    with pytest.raises(ValueError, match="unknown alert threshold"):
        alerts.merged_thresholds({"nope": 1})


def _alert_functions() -> list[tuple[set[str], set[str]]]:
    """(alert rules, threshold keys read) of each rule function of alerts.py, from its source."""
    tree = ast.parse(Path(alerts.__file__).read_text())
    out = []
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        rules = {n.args[1].value for n in ast.walk(fn) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", None) == "_alert" and len(n.args) > 1
                 and isinstance(n.args[1], ast.Constant)}
        keys = {n.slice.value for n in ast.walk(fn) if isinstance(n, ast.Subscript)
                and isinstance(n.slice, ast.Constant) and n.slice.value in TODAY}
        if rules and keys:
            out.append((rules, keys))
    return out


def test_every_alert_rule_with_a_threshold_names_the_entry_that_holds_it():
    functions = _alert_functions()
    assert functions
    for rules, keys in functions:
        missing = sorted(r for r in rules if r not in monitor.DASHBOARD_RULES)
        assert not missing, f"alert rules {missing} read {sorted(keys)} but name no registry entry"
        params = {p.name for r in rules for p in REG.get(monitor.DASHBOARD_RULES[r]).params}
        assert keys <= params, f"{sorted(rules)}: {sorted(keys - params)} are not params of their entries"
    assert set(monitor.DASHBOARD_RULES.values()) <= {b.name for b in REG}


def test_an_alert_carries_its_breaker():
    assert alerts._alert(1, "release_stalled", "t", "d")["breaker"] == "monitor.dashboard_release_stalled"
    assert alerts._alert(1, "store_unreadable", "t", "d")["breaker"] is None


def test_monitor_entries_switch_off_in_the_section():
    rb = _resolve(run_config={"breakers": {"monitor": {"dashboard_heartbeat": {"enabled": False,
                                                                                 "heartbeat_warn_s": 120}}}})
    assert not rb.enabled("monitor.dashboard_heartbeat")
    rb = _resolve(run_config={"breakers": {"monitor": {"dashboard_heartbeat": {"heartbeat_warn_s": 120}}}})
    assert rb.get("monitor.dashboard_heartbeat.heartbeat_warn_s") == 120


# -- pool trips ---------------------------------------------------------------

def _breaker_events(p, name=None):
    return [e for e in p.events.events if e["event"] == "rl_breaker" and (name is None or e["name"] == name)]


@pytest.fixture(autouse=True)
def _no_stop_reason():
    breaker_events.clear_stop_reason()
    yield
    breaker_events.clear_stop_reason()


def test_provision_breaker_warns_on_open_and_halts_with_a_stop_reason():
    clock = Clock()
    cloud = Cloud(clock)
    p = _pool(clock, cloud, accounts=(1,), breaker_failures=2, breaker_cooldown_s=600, breaker_max_trips=2)
    cloud.lose_before_grant = 100
    _run(p, clock, 3 * H)
    names = _names(p)
    assert "breaker_open" in names and "breaker_tripped" in names  # legacy events kept
    assert p.stopped == "breaker tripped and no member left"
    opened, tripped = _breaker_events(p, "pool.provision_breaker")
    assert (opened["state"], opened["action"], opened["value"], opened["limit"]) == ("warn", "halt", 1, 2)
    assert "stop_reason" not in opened
    assert (tripped["state"], tripped["action"], tripped["value"]) == ("trip", "halt", 2)
    assert tripped["stop_reason"] == "breaker:pool.provision_breaker"
    assert tripped["legacy_event"] == "nc_elastic_breaker_tripped" and tripped["run"] == "run-a"
    assert tripped["scope"]["pool"] == "nc" and tripped["layer"] == "pool"
    assert breaker_events.current_stop_reason() == "breaker:pool.provision_breaker"
    assert not breaker_events.validate_event(tripped)


@pytest.mark.parametrize("mode", ["replace", "halt"])
def test_crash_loop_reports_its_action(mode):
    clock = Clock()
    cloud = Cloud(clock, ready_after=10**9)  # never joins
    p = _pool(clock, cloud, accounts=(1, 4), max_sampler_restarts=1, crash_loop_before_ready=mode)
    while not any(k == "start" for k, *_ in cloud.log):
        p.tick()
        clock.t += 60
    m = p.members[0]
    for _ in range(2):
        cloud.tickets[m.ticket].update(sampler="exited", exit=1)
        p.tick()
        clock.t += 60
    (event,) = _breaker_events(p, "pool.crash_loop")
    assert event["action"] == mode and event["value"] == m.restarts and event["limit"] == 1
    assert event["scope"]["slot"] == m.account
    assert ("stop_reason" in event) is (mode == "halt")
    p.tick()
    assert (m.state == GONE) is (mode == "replace")  # what the pool does is unchanged


def test_a_spend_cap_refusal_is_reported_once():
    clock = Clock()
    cloud = Cloud(clock)
    p = _pool(clock, cloud, accounts=(1, 4), max_run_spend_ucr=1)
    _run(p, clock, 600)
    refused = [e for e in p.events.events if e["event"] == "nc_elastic_order_refused"]
    assert refused and not p.members
    (event,) = _breaker_events(p, "pool.spend_cap")
    assert (event["state"], event["action"], event["limit"], event["unit"]) == ("trip", "pause", 1, "ucr")
    assert event["detail"] == refused[0]["why"] and "stop_reason" not in event


def test_budget_check_keeps_the_refusal_text():
    b = el.Budget(spent_ucr=10.0, orders=3)
    for cfg, kw, breaker in (
            (el.ElasticConfig(max_nodes=1), {"billed_nodes": 1}, "pool.node_cap"),
            (el.ElasticConfig(max_orders=3), {"billed_nodes": 0}, "pool.node_cap"),
            (el.ElasticConfig(max_run_spend_ucr=5), {"billed_nodes": 0}, "pool.spend_cap"),
            (el.ElasticConfig(max_run_spend_usd=0.0), {"billed_nodes": 0}, "pool.spend_cap")):
        got = b.check(cfg, target=1, hourly_ucr=1, **kw)
        assert got is not None and got.breaker == breaker
        assert b.refusal(cfg, target=1, hourly_ucr=1, **kw) == got.why
    assert b.check(el.ElasticConfig(), billed_nodes=0, target=1, hourly_ucr=1) is None


def test_a_failing_report_never_reaches_the_pool(monkeypatch):
    clock = Clock()
    cloud = Cloud(clock)
    p = _pool(clock, cloud, accounts=(1,), breaker_failures=1, breaker_max_trips=5)

    def broken(*a, **k):
        raise RuntimeError("registry broken")

    monkeypatch.setattr(breaker_events, "trip", broken)
    p._fail(0.0, "test")
    assert "breaker_open" in _names(p) and p.breaker.trips == 1


# -- the run's section reaches the pool config --------------------------------

def _pool_file(tmp_path, **extra):
    path = tmp_path / "pool.json"
    path.write_text(json.dumps({"target": 1, "shard_base": "k31", "elastic": {"start_timeout_s": 3600}, **extra}))
    return str(path)


def test_without_a_section_the_pool_config_is_used_as_it_is(tmp_path):
    path = _pool_file(tmp_path)
    (ref,) = rollout_pools.resolve({"pool": {"config": path}, "budget": {"usd": 450}}, tmp_path / "state")
    assert ref.config == path and not (tmp_path / "state").exists()


def test_the_section_is_written_into_a_copy_of_the_pool_config(tmp_path, monkeypatch):
    monkeypatch.setattr("yeto.breakers.registry._DEFAULT", REG)
    path = _pool_file(tmp_path)
    cfg = {"pool": {"config": path},
           "breakers": {"run": {"budget": {"usd": 450}}, "pool": {"crash_loop": {"action": "halt"}},
                        "pools": {"pool": {"lease_end": {"replace_lead_s": 1800}}}}}
    (ref,) = rollout_pools.resolve(cfg, tmp_path / "state")
    assert ref.config == str(tmp_path / "state" / "pools" / "pool.json")
    doc = json.loads(Path(ref.config).read_text())
    assert doc["elastic"] == {"start_timeout_s": 3600, "max_run_spend_usd": 450,
                              "crash_loop_before_ready": "halt"}
    assert doc["replace_lead_s"] == 1800 and doc["shard_base"] == "k31"
    assert json.loads(Path(path).read_text())["elastic"] == {"start_timeout_s": 3600}  # the user's file untouched
    assert el.ElasticConfig.from_dict(doc["elastic"]).max_run_spend_usd == 450


def test_a_section_that_changes_nothing_keeps_the_file(tmp_path, monkeypatch):
    monkeypatch.setattr("yeto.breakers.registry._DEFAULT", REG)
    path = _pool_file(tmp_path)
    cfg = {"pool": {"config": path}, "breakers": {"pool": {"start_timeout": {"limit_s": 3600}}}}
    (ref,) = rollout_pools.resolve(cfg, tmp_path / "state")
    assert ref.config == path


def test_a_section_against_the_pool_config_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr("yeto.breakers.registry._DEFAULT", REG)
    path = _pool_file(tmp_path)
    cfg = {"pool": {"config": path}, "breakers": {"pool": {"start_timeout": {"limit_s": 1800}}}}
    with pytest.raises(ValueError, match="pool 'pool'.*start_timeout"):
        rollout_pools.resolve(cfg, tmp_path / "state")
