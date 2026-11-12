"""Cost protection: resource ledger, billing, budget breaker, idle shutdown, cleanup, report.

All CPU: fake Modal billing, fake NC / Modal cleanup ops, fake clocks."""

from __future__ import annotations

import json

import pytest

from yeto import cli, runs
from yeto.cost import cleanup as cleanup_mod
from yeto.cost import reaper
from yeto.cost.billing import BillingRow, ModalBilling, Rates, resource_costs
from yeto.cost.guard import CostGuard, GuardConfig, TapeTail, guard_config_from_args
from yeto.cost.idle import IdleConfig, IdleDetector
from yeto.cost.ledger import ResourceLedger
from yeto.cost.report import build_report, productive_seconds

H = 3600.0
T0 = 1_800_000_000.0 - (1_800_000_000.0 % H)  # an hour boundary
MIN = 60.0


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    return tmp_path / "runs"


class Clock:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class FakeOps:
    def __init__(self, apps_named=None, fail=()):
        self.calls: list[tuple] = []
        self.apps_named = apps_named or {}
        self.fail = set(fail)

    def _do(self, *a):
        self.calls.append(a)
        if a[1] in self.fail:
            raise RuntimeError(f"boom {a[1]}")

    def modal_cancel_call(self, call_id):
        self._do("cancel", call_id)

    def modal_stop_app(self, app_id):
        self._do("stop_app", app_id)

    def modal_app_ids_named(self, name, created_after):
        self.calls.append(("list", name))
        return list(self.apps_named.get(name, []))

    def nc_release(self, account, ticket):
        self._do("nc_down", ticket, account)
        return {"ticket": ticket, "state": "ended"}

    def sky_down(self, cluster):
        self._do("sky_down", cluster)
        return True


def ledger_at(path, t):
    return ResourceLedger(path, clock=Clock(t))


# -- ledger -----------------------------------------------------------------------------

def test_ledger_folds_create_release_and_survives_a_torn_line(tmp_path):
    p = tmp_path / "resources.jsonl"
    ledger_at(p, T0).intent("modal_app", "yeto-r1")
    ledger_at(p, T0 + 5).created("modal_app", "ap-1", "yeto-r1", gpu="B300", gpus=8)
    ledger_at(p, T0 + 6).created("modal_call", "fc-1", "island", app_id="ap-1")
    ledger_at(p, T0 + 7).created("nc_lease", "t-9", "nc", account=4)
    ledger_at(p, T0 + 8).released("nc_lease", "t-9")
    with open(p, "a") as fh:
        fh.write('{"op": "create", "kind": "modal_ca')  # SIGKILL mid-write
    res = {r.key: r for r in ResourceLedger(p).resources()}
    assert set(res) == {"modal_app:ap-1", "modal_call:fc-1", "nc_lease:t-9"}
    assert res["modal_app:ap-1"].created_at == T0  # from the intent
    assert res["modal_app:ap-1"].attrs["gpus"] == 8
    assert not res["nc_lease:t-9"].open
    assert {r.key for r in ResourceLedger(p).open_resources()} == {"modal_app:ap-1", "modal_call:fc-1"}


def test_stopping_an_app_ends_its_calls_in_the_ledger(tmp_path):
    p = tmp_path / "r.jsonl"
    ledger_at(p, T0).created("modal_app", "ap-1", "yeto-r1")
    ledger_at(p, T0 + 1).created("modal_call", "fc-1", "i0", app_id="ap-1")
    ledger_at(p, T0 + 1).created("modal_call", "fc-2", "i1", app_id="ap-OTHER")
    ledger_at(p, T0 + 9).released("modal_app", "ap-1", "yeto-r1")
    assert [r.key for r in ResourceLedger(p).open_resources()] == ["modal_call:fc-2"]


def test_unknown_kind_is_refused(tmp_path):
    with pytest.raises(ValueError):
        ResourceLedger(tmp_path / "x").created("vm", "i-1")


# -- billing ----------------------------------------------------------------------------

def test_modal_cost_is_matched_by_exact_app_id_and_the_unbilled_tail_estimated(tmp_path):
    p = tmp_path / "r.jsonl"
    ledger_at(p, T0 + 0.5 * H).created("modal_app", "ap-mine", "yeto-k31-sampler-b300", usd_per_hour=60.0)
    rows = [
        BillingRow("ap-mine", "yeto-k31-sampler-b300", T0, 30.0),
        BillingRow("ap-mine", "yeto-k31-sampler-b300", T0 + H, 60.0),
        # same NAME, another run's app: never counted
        BillingRow("ap-theirs", "yeto-k31-sampler-b300", T0 + H, 999.0),
        # this app, but an hour before it was this run's
        BillingRow("ap-mine", "yeto-k31-sampler-b300", T0 - H, 500.0),
    ]
    billing = ModalBilling(lambda s, e: rows)
    s = resource_costs(ResourceLedger(p).resources(), T0 + 2.5 * H, billing=billing)
    (item,) = s.items
    assert item.billed_usd == 90.0
    assert item.estimated_usd == pytest.approx(30.0)  # 0.5 h after the newest billed hour x $60
    assert s.total_usd == pytest.approx(120.0)
    assert set(item.hours) and sum(item.hours.values()) == pytest.approx(120.0)


def test_rate_falls_back_to_the_newest_complete_billed_hour_and_billing_errors_keep_estimating(tmp_path):
    p = tmp_path / "r.jsonl"
    ledger_at(p, T0).created("modal_app", "ap-1", "a")
    good = [BillingRow("ap-1", "a", T0, 50.0)]
    state = {"fail": False}

    def fetch(s, e):
        if state["fail"]:
            raise ConnectionError("billing down")
        return good

    billing = ModalBilling(fetch, refresh_s=0)
    s = resource_costs(ResourceLedger(p).resources(), T0 + 1.5 * H, billing=billing)
    assert s.total_usd == pytest.approx(75.0)
    state["fail"] = True
    s = resource_costs(ResourceLedger(p).resources(), T0 + 2 * H, billing=billing)
    assert s.billing_error and s.total_usd == pytest.approx(100.0)


def test_nc_lease_keeps_billing_past_lease_until(tmp_path):
    p = tmp_path / "r.jsonl"
    ledger_at(p, T0).created("nc_lease", "t-1", "nc", account=4, granted_at=T0, lease_until=T0 + 4 * H,
                             gpus=8, rate_ucr_per_gpu_hour=3_000_000)
    rates = Rates(nc_usd_per_ucr=1e-6)  # $3 / GPU-hour
    s = resource_costs(ResourceLedger(p).resources(), T0 + 6 * H, rates=rates)
    (item,) = s.items
    assert item.usd == pytest.approx(6 * 8 * 3.0)
    assert "past lease_until" in item.note


def test_a_resource_without_a_rate_is_unpriced_not_zero(tmp_path):
    p = tmp_path / "r.jsonl"
    ledger_at(p, T0).created("nc_lease", "t-1", "nc", account=4, rate_ucr_per_gpu_hour=3_000_000)
    s = resource_costs(ResourceLedger(p).resources(), T0 + H,
                       rates=Rates(nc_usd_per_ucr=None, nc_ucr_per_gpu_hour=None))
    assert [i.key for i in s.unpriced] == ["nc_lease:t-1"] and s.items[0].usd is None


def test_an_nc_lease_is_priced_by_default_at_about_24_usd_an_hour(tmp_path):
    """F: an NC lease with no rate in the ledger still counts against the budget."""
    p = tmp_path / "r.jsonl"
    ledger_at(p, T0).created("nc_lease", "t-1", "nc", account=4)
    s = resource_costs(ResourceLedger(p).resources(), T0 + 2 * H, rates=Rates())
    assert not s.unpriced and s.items[0].usd == pytest.approx(48.0)
    env = {"YETO_NC_USD_PER_UCR": "2e-6", "YETO_NC_UCR_PER_GPU_HOUR": "1000000"}
    s = resource_costs(ResourceLedger(p).resources(), T0 + H, rates=Rates.from_env(environ=env))
    assert s.items[0].usd == pytest.approx(16.0)  # 8 GPUs x 1e6 ucr x 2e-6 USD
    assert Rates.from_env(environ={}).nc_usd_per_ucr == 1e-6


def test_the_guard_budget_counts_the_nc_lease(tmp_path):
    clock = Clock(T0 + 3 * H)
    g, led, calls = make_guard(tmp_path, clock, budget=50.0)
    ledger_at(led.path, T0).created("nc_lease", "t-1", "nc", account=4)
    g.tick()
    assert "cost_budget_exceeded" in events(tmp_path) and calls["graceful"]


# -- idle detector ------------------------------------------------------------------------

def idle(start=T0, **kw):
    cfg = IdleConfig(**{"idle_after_s": 15 * MIN, "stop_after_s": 15 * MIN, "load_grace_s": 45 * MIN,
                        "max_load_s": 120 * MIN, **kw})
    return IdleDetector(cfg, start)


def test_weight_loading_with_zero_gpu_and_no_progress_is_not_idle():
    d = idle()
    d.observe({"event": "rl_driver_start"}, T0 + MIN)
    for m in range(0, 25):
        d.observe({"event": "modal_host_sample", "gpu_util_pct": [0, 0]}, T0 + m * MIN)
    assert d.verdict(T0 + 22 * MIN).state == "loading"
    assert d.verdict(T0 + 40 * MIN).state == "loading"
    # loaded, training
    d.observe({"event": "rl_round_trained"}, T0 + 41 * MIN)
    d.observe({"event": "modal_host_sample", "gpu_util_pct": [97, 95]}, T0 + 42 * MIN)
    assert d.verdict(T0 + 50 * MIN).state == "ok"


def test_no_progress_after_loading_warns_then_stops_and_progress_resets():
    d = idle()
    d.observe({"event": "rl_round_trained"}, T0 + 50 * MIN)
    d.observe({"event": "rl_heartbeat"}, T0 + 60 * MIN)  # heartbeats are no progress
    assert d.verdict(T0 + 60 * MIN).state == "ok"
    v = d.verdict(T0 + 66 * MIN)
    assert v.state == "warn" and "no progress" in v.reason
    assert d.verdict(T0 + 81 * MIN).state == "stop"
    d.observe({"event": "rl_sampler_round"}, T0 + 81 * MIN)
    assert d.verdict(T0 + 82 * MIN).state == "ok"


def test_all_gpus_at_zero_is_idle_even_with_events_but_one_busy_gpu_is_not():
    d = idle()
    d.observe({"event": "rl_round_trained"}, T0 + 46 * MIN)
    for m in range(46, 80):
        d.observe({"event": "rl_sampler_round"}, T0 + m * MIN)
        d.observe({"event": "modal_host_sample", "island": "s0", "gpu_util_pct": [0]}, T0 + m * MIN)
        d.observe({"event": "modal_host_sample", "island": "t0", "gpu_util_pct": [80]}, T0 + m * MIN)
    assert d.verdict(T0 + 80 * MIN).state == "ok"  # the trainer GPU is busy
    for m in range(80, 100):
        d.observe({"event": "rl_sampler_round"}, T0 + m * MIN)
        d.observe({"event": "modal_host_sample", "island": "t0", "gpu_util_pct": [0]}, T0 + m * MIN)
    v = d.verdict(T0 + 99 * MIN)
    assert v.state == "warn" and "GPU utilization 0" in v.reason


def test_a_new_loading_phase_mid_run_is_exempt_but_bounded():
    d = idle()
    d.observe({"event": "rl_round_trained"}, T0 + 50 * MIN)
    # trainer incarnation restarts and reloads weights for 30 min
    d.observe({"event": "rl_trainer_lifecycle", "phase": "submitted"}, T0 + 55 * MIN)
    assert d.verdict(T0 + 85 * MIN).state == "loading"
    # a loader stuck in a restart loop keeps emitting loading events: bounded by max_load_s
    for m in range(55, 300, 10):
        d.observe({"event": "rl_startup_step"}, T0 + m * MIN)
    assert d.verdict(T0 + 160 * MIN).state == "loading"
    v = d.verdict(T0 + 50 * MIN + 120 * MIN + 16 * MIN)
    assert v.state == "warn"


def test_idle_detection_disabled_with_zero_minutes():
    d = idle(idle_after_s=0)
    assert d.verdict(T0 + 100 * H).state == "ok"


# -- guard: budget ------------------------------------------------------------------------

def make_guard(tmp_path, clock, *, budget=100.0, warn=None, ended=lambda: False, idle_cfg=None,
               rows=None, tapes=()):
    led = ResourceLedger(tmp_path / "resources.jsonl", clock=clock)
    calls = {"graceful": [], "force": [], "release": []}
    g = CostGuard("r1", GuardConfig(budget_usd=budget, budget_warn_usd=warn, stop_grace_s=20 * MIN,
                                    idle=idle_cfg or IdleConfig(idle_after_s=0)),
                  ledger=led, tapes=TapeTail(list(tapes)),
                  billing=ModalBilling(lambda s, e: list(rows or []), refresh_s=0),
                  graceful_stop=lambda why: calls["graceful"].append(why),
                  force_stop=lambda why: calls["force"].append(why),
                  release_resource=lambda r: calls["release"].append(r.key),
                  run_ended=ended, clock=clock, log=lambda m: None, start=T0)
    return g, led, calls


def events(tmp_path):
    p = tmp_path / "cost-events.jsonl"
    return [json.loads(x)["event"] for x in p.read_text().splitlines()] if p.exists() else []


def test_budget_warns_once_then_stops_gracefully_then_forces(tmp_path):
    clock = Clock(T0)
    g, led, calls = make_guard(tmp_path, clock, budget=100.0)
    led.created("modal_app", "ap-1", "yeto-r1", usd_per_hour=60.0)
    clock.t = T0 + 1.0 * H
    g.tick()
    assert "cost_budget_warn" not in events(tmp_path)
    clock.t = T0 + 1.4 * H  # $84 >= 80 % of $100
    g.tick()
    g.tick()
    assert events(tmp_path).count("cost_budget_warn") == 1 and not calls["graceful"]
    clock.t = T0 + 1.7 * H  # $102
    g.tick()
    assert len(calls["graceful"]) == 1 and "budget" in calls["graceful"][0]
    assert "cost_budget_exceeded" in events(tmp_path)
    clock.t += 10 * MIN
    g.tick()
    assert not calls["force"] and len(calls["graceful"]) == 1
    clock.t += 11 * MIN  # 21 min after the graceful stop, still up
    g.tick()
    g.tick()
    assert len(calls["force"]) == 1


def test_a_reused_run_name_does_not_count_the_earlier_launchs_released_spend(tmp_path):
    """Relaunching a run name after a budget stop must not stop the new launch at its
    first tick: resources released before this guard started are not its spend; one
    still open (left by the earlier launch) bills now and counts."""
    clock = Clock(T0 + H)
    g, led, calls = make_guard(tmp_path, clock, budget=50.0)
    early = ledger_at(led.path, T0 - 3 * H)
    early.created("modal_app", "ap-old", "yeto-r1", usd_per_hour=60.0)
    ledger_at(led.path, T0 - H).released("modal_app", "ap-old", "yeto-r1")  # $120, before T0
    s = g.tick()
    assert s.total_usd == pytest.approx(0.0) and not calls["graceful"]
    ledger_at(led.path, T0 - H).created("modal_app", "ap-left", "yeto-r1", usd_per_hour=30.0)
    s = g.tick()  # still open: 2 h at $30
    assert s.total_usd == pytest.approx(60.0) and calls["graceful"]


def test_no_forced_stop_once_the_run_has_ended(tmp_path):
    clock = Clock(T0)
    ended = {"v": False}
    g, led, calls = make_guard(tmp_path, clock, budget=10.0, ended=lambda: ended["v"])
    led.created("modal_app", "ap-1", "yeto-r1", usd_per_hour=60.0)
    clock.t = T0 + H
    g.tick()
    ended["v"] = True
    clock.t += H
    g.tick()
    assert calls["graceful"] and not calls["force"]


def test_budget_counts_billing_rows_and_nc_leases_together(tmp_path):
    clock = Clock(T0 + 2 * H)
    rows = [BillingRow("ap-1", "x", T0, 40.0), BillingRow("ap-1", "x", T0 + H, 40.0)]
    g, led, calls = make_guard(tmp_path, clock, budget=100.0, rows=rows)
    ledger_at(led.path, T0).created("modal_app", "ap-1", "x", usd_per_hour=40.0)
    ledger_at(led.path, T0).created("nc_lease", "t-1", "nc", account=4, granted_at=T0, usd_per_hour=12.0)
    s = g.tick()
    assert s.total_usd == pytest.approx(80 + 24) and calls["graceful"]


def test_graceful_stop_failure_forces_at_once(tmp_path):
    clock = Clock(T0 + H)
    g, led, calls = make_guard(tmp_path, clock, budget=1.0)

    def bad(why):
        raise OSError("no run dir")

    g.graceful_stop = bad
    ledger_at(led.path, T0).created("modal_app", "ap-1", "x", usd_per_hour=10.0)
    g.tick()
    assert calls["force"]


def test_unpriced_resource_is_reported_once(tmp_path):
    clock = Clock(T0 + H)
    g, led, _ = make_guard(tmp_path, clock, budget=None)
    ledger_at(led.path, T0).created("sky_cluster", "c-1", "nc")  # no rate anywhere
    g.tick()
    g.tick()
    assert events(tmp_path).count("cost_unpriced") == 1


def test_overdue_nc_lease_is_reported_not_released_by_default(tmp_path):
    """The NC pool owns its leases: a trainer's cost guard only alerts on an overdue one."""
    clock = Clock(T0 + 4 * H + 6 * MIN)
    g, led, calls = make_guard(tmp_path, clock, budget=None)
    ledger_at(led.path, T0).created("nc_lease", "t-1", "nc", account=4, lease_until=T0 + 4 * H)
    g.tick()
    g.tick()
    assert calls["release"] == [] and calls["force"] == []
    assert events(tmp_path).count("nc_lease_overdue") == 1


def test_overdue_nc_lease_is_released_once_when_opted_in(tmp_path):
    clock = Clock(T0 + 4 * H + 2 * MIN)
    g, led, calls = make_guard(tmp_path, clock, budget=None)
    g.config.release_overdue_nc = True
    ledger_at(led.path, T0).created("nc_lease", "t-1", "nc", account=4, lease_until=T0 + 4 * H)
    ledger_at(led.path, T0).created("nc_lease", "t-2", "nc", account=5, lease_until=T0 + 8 * H)
    g.tick()
    assert not calls["release"]  # within the 5 min grace
    clock.t = T0 + 4 * H + 6 * MIN
    g.tick()
    g.tick()
    assert calls["release"] == ["nc_lease:t-1"]
    assert "nc_lease_overdue" in events(tmp_path)


# -- guard: idle via tapes -----------------------------------------------------------------

def test_guard_reads_tapes_and_stops_an_idle_run_but_not_a_loading_one(tmp_path):
    tape = tmp_path / "events" / "island-0.jsonl"
    tape.parent.mkdir()
    clock = Clock(T0)
    g, led, calls = make_guard(tmp_path, clock, budget=None, tapes=[str(tmp_path / "events" / "*.jsonl")],
                               idle_cfg=IdleConfig(idle_after_s=15 * MIN, stop_after_s=15 * MIN,
                                                   load_grace_s=45 * MIN))

    def write(rec):
        with open(tape, "a") as fh:
            fh.write("YETO_RL_EVENT " + json.dumps(rec) + "\n")

    write({"event": "rl_driver_start", "time_unix": T0 + MIN})
    for m in range(0, 44, 2):  # 22-min weight load, silent; then waiting
        clock.t = T0 + m * MIN
        g.tick()
    assert not calls["graceful"] and "idle_warn" not in events(tmp_path)
    write({"event": "rl_round_trained", "time_unix": T0 + 44 * MIN})
    clock.t = T0 + 50 * MIN
    g.tick()
    clock.t = T0 + 60 * MIN  # 16 min without progress
    g.tick()
    assert "idle_warn" in events(tmp_path) and not calls["graceful"]
    clock.t = T0 + 75 * MIN  # 31 min
    g.tick()
    assert "idle_shutdown" in events(tmp_path) and len(calls["graceful"]) == 1


def test_tape_tail_skips_a_partial_last_line_until_it_completes(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text('{"event": "a"}\n{"event": "b"')
    t = TapeTail([str(p)])
    assert [r["event"] for _s, r in t.read()] == ["a"]
    with open(p, "a") as fh:
        fh.write("}\n")
    assert [r["event"] for _s, r in t.read()] == ["b"]
    assert t.read() == []


# -- cleanup ------------------------------------------------------------------------------

def test_cleanup_releases_exactly_the_ledger_resources_of_this_run(runs_dir):
    mine = ResourceLedger.for_run("run-a")
    other = ResourceLedger.for_run("run-b")
    mine.created("modal_app", "ap-a", "yeto-k31-sampler-b300")
    mine.created("modal_call", "fc-a", "sampler", app_id="ap-a")
    mine.created("nc_lease", "t-a", "nc", account=4)
    mine.created("sky_cluster", "run-a-syncer")
    mine.created("modal_call", "fc-done", "x", app_id="ap-a")
    mine.released("modal_call", "fc-done")
    other.created("modal_app", "ap-b", "yeto-k31-sampler-b300")  # same app name, other run
    other.created("nc_lease", "t-b", "nc", account=5)
    ops = FakeOps()
    assert cleanup_mod.cleanup_run("run-a", ops, include_nc=True, log=lambda m: None)
    assert ops.calls == [("cancel", "fc-a"), ("stop_app", "ap-a"), ("nc_down", "t-a", 4),
                         ("sky_down", "run-a-syncer")]
    assert not mine.open_resources()
    assert len(other.open_resources()) == 2
    # rerun: nothing left to do
    ops2 = FakeOps()
    assert cleanup_mod.cleanup_run("run-a", ops2, include_nc=True, log=lambda m: None) and ops2.calls == []


def test_cleanup_keeps_every_pool_providers_machines_unless_include_pools(runs_dir):
    """Modal sampler calls and Sky sampler clusters share their kind with the trainer's;
    the pool's rows (marked, or with a pool ticket / slot) are kept like NC leases and
    Daytona sandboxes until --include-pools (--include-nc: the same, deprecated)."""
    led = ResourceLedger.for_run("rp")
    led.created("modal_call", "fc-trainer", "trainer", app_id="ap-t")
    led.created("modal_call", "fc-pool", "rollout_sampler", ticket="yeto-smp-rp-1", run="rp", pool=True)
    led.created("modal_call", "fc-old", "rollout_sampler", ticket="yeto-smp-rp-2", run="rp")  # before the mark
    led.created("sky_cluster", "rp-syncer", "rp-syncer", role="syncer")
    led.created("sky_cluster", "yeto-smp-rp-3", "yeto-smp-rp-3", slot=1, run="rp")
    led.created("daytona_sandbox", "sb-1", "yeto-smp-rp-4")
    led.created("nc_lease", "t-1", "nc", account=4)
    ops = FakeOps()
    logs = []
    assert cleanup_mod.cleanup_run("rp", ops, log=logs.append)  # nothing it may release is left
    assert ops.calls == [("cancel", "fc-trainer"), ("sky_down", "rp-syncer")]
    assert sum("a sampler pool machine" in m and "--include-pools" in m for m in logs) == 5
    assert sorted(r.key for r in led.open_resources()) == [
        "daytona_sandbox:sb-1", "modal_call:fc-old", "modal_call:fc-pool", "nc_lease:t-1",
        "sky_cluster:yeto-smp-rp-3"]
    from yeto.cost import report  # noqa: PLC0415
    import argparse  # noqa: PLC0415

    sub = argparse.ArgumentParser().add_subparsers(dest="cmd")
    report.add_parsers(sub)
    parser = sub.choices["cleanup"]
    assert parser.parse_args(["--run", "rp", "--include-nc"]).include_pools
    assert parser.parse_args(["--run", "rp", "--include-pools"]).include_pools
    assert not parser.parse_args(["--run", "rp"]).include_pools
    assert "deprecated" in parser.format_help()


def test_cleanup_failure_is_recorded_retried_and_does_not_block_the_rest(runs_dir):
    led = ResourceLedger.for_run("r")
    led.created("modal_call", "fc-1", "i")
    led.created("nc_lease", "t-1", "nc", account=4)
    ops = FakeOps(fail={"fc-1"})
    assert not cleanup_mod.cleanup_run("r", ops, include_nc=True, log=lambda m: None)
    assert ("nc_down", "t-1", 4) in ops.calls
    (left,) = led.open_resources()
    assert left.key == "modal_call:fc-1" and "boom" in left.release_error
    assert cleanup_mod.cleanup_run("r", FakeOps(), include_nc=True, log=lambda m: None)


def test_cleanup_dry_run_changes_nothing(runs_dir):
    led = ResourceLedger.for_run("r")
    led.created("modal_app", "ap-1", "a")
    ops = FakeOps()
    assert cleanup_mod.cleanup_run("r", ops, include_nc=True, dry_run=True, log=lambda m: None)
    assert ops.calls == [] and len(led.open_resources()) == 1


def test_intent_is_resolved_only_by_one_exact_name_match(runs_dir):
    led = ResourceLedger.for_run("r")
    led.intent("modal_app", "yeto-r")  # crashed during deploy
    ops = FakeOps(apps_named={"yeto-r": ["ap-9"], "yeto-r-2": ["ap-x"]})
    assert cleanup_mod.cleanup_run("r", ops, include_nc=True, log=lambda m: None)
    assert ("stop_app", "ap-9") in ops.calls and ("stop_app", "ap-x") not in ops.calls
    led2 = ResourceLedger.for_run("r2")
    led2.intent("modal_app", "app-x")
    ops = FakeOps(apps_named={"app-x": ["ap-1", "ap-2"]})
    assert not cleanup_mod.cleanup_run("r2", ops, include_nc=True, log=lambda m: None)
    assert not [c for c in ops.calls if c[0] == "stop_app"]
    led3 = ResourceLedger.for_run("r3")
    led3.intent("modal_app", "yeto-r3")  # deploy never created anything
    assert cleanup_mod.cleanup_run("r3", FakeOps(), include_nc=True, log=lambda m: None)
    assert not led3.open_resources()


def test_shared_app_is_not_stopped_only_its_calls_cancelled(runs_dir):
    led = ResourceLedger.for_run("r")
    led.created("modal_app", "ap-s", "shared", shared=True)
    led.created("modal_call", "fc-1", "i", app_id="ap-s")
    ops = FakeOps()
    assert cleanup_mod.cleanup_run("r", ops, include_nc=True, log=lambda m: None)
    assert ops.calls == [("cancel", "fc-1")]


def test_production_ops_refuse_a_non_id_app():
    with pytest.raises(ValueError):
        cleanup_mod.ProductionCleanupOps().modal_stop_app("yeto-*")


# -- launcher crash: reaper ------------------------------------------------------------------

def test_reaper_cleans_up_after_a_killed_launcher(runs_dir):
    runs.create_run("r", {"cluster_prefix": "r"})
    runs.update_run("r", pid=4242, state=runs.RUNNING)  # SIGKILLed: state never updated
    led = ResourceLedger.for_run("r")
    led.created("modal_app", "ap-1", "yeto-r")
    led.created("modal_call", "fc-1", "i", app_id="ap-1")
    ResourceLedger.for_run("bystander").created("modal_app", "ap-2", "yeto-r")
    alive = iter([True, True, False])
    ops = FakeOps()
    code = reaper.reap("r", 4242, alive=lambda pid: next(alive, False), sleep=lambda s: None,
                       cleanup=lambda run: cleanup_mod.cleanup_run(run, ops, log=lambda m: None),
                       log=lambda m: None)
    assert code == 0
    assert ops.calls == [("cancel", "fc-1"), ("stop_app", "ap-1")]
    assert len(ResourceLedger.for_run("bystander").open_resources()) == 1


def test_reaper_leaves_keep_runs_and_clean_runs_alone(runs_dir):
    runs.create_run("k", {"cluster_prefix": "k", "keep": True})
    ResourceLedger.for_run("k").created("modal_app", "ap-1", "yeto-k")
    called = []
    reaper.reap("k", 1, alive=lambda p: False, cleanup=lambda r: called.append(r), log=lambda m: None)
    runs.create_run("c", {"cluster_prefix": "c"})
    led = ResourceLedger.for_run("c")
    led.created("modal_app", "ap-2", "yeto-c")
    led.released("modal_app", "ap-2")
    reaper.reap("c", 1, alive=lambda p: False, cleanup=lambda r: called.append(r), log=lambda m: None)
    assert called == []


def test_worker_backstop_cleans_ledger_left_overs_after_a_launcher_exception(runs_dir, monkeypatch):
    runs.create_run("w", {"cluster_prefix": "w"})
    ResourceLedger.for_run("w").created("modal_call", "fc-1", "i")
    import yeto.launcher as launcher

    def boom(args, **kw):
        raise RuntimeError("launcher crashed")

    monkeypatch.setattr(launcher, "run", boom)
    seen = []
    monkeypatch.setattr(cleanup_mod, "cleanup_run", lambda run, *a, **k: seen.append(run) or True)
    assert cli.cmd_worker("w") == 1
    assert seen == ["w"]


# -- report & CLI -----------------------------------------------------------------------------

def test_effective_utilization_counts_only_close_progress_gaps():
    recs = [("t", {"event": "rl_round_trained", "time_unix": T0 + m * MIN}) for m in (0, 5, 10, 40, 45)]
    prod, _ = productive_seconds(recs, gap_s=10 * MIN)
    assert prod["t"] == pytest.approx(15 * MIN)  # 0-5-10 and 40-45; the 30-min gap is idle


def test_cost_report_per_resource_per_hour_and_utilization(tmp_path):
    p = tmp_path / "r.jsonl"
    ledger_at(p, T0).created("modal_app", "ap-1", "a", usd_per_hour=10.0)
    ResourceLedger(p, clock=Clock(T0 + 2 * H)).released("modal_app", "ap-1")
    tape = tmp_path / "t.jsonl"
    tape.write_text("".join(json.dumps({"event": "rl_round_trained", "time_unix": T0 + m * MIN}) + "\n"
                            for m in range(0, 61, 5)))
    s = resource_costs(ResourceLedger(p).resources(), T0 + 3 * H)
    rep = build_report("r", summary=s, tape_globs=[str(tape)])
    assert rep["total_usd"] == 20.0 and len(rep["by_hour"]) == 2
    assert rep["effective_utilization"] == pytest.approx(0.5)


def test_cli_parses_cost_cleanup_and_launch_guard_flags(runs_dir, capsys, monkeypatch):
    p = cli.build_parser()
    a = p.parse_args(["cost", "--run", "r", "--json", "--no-billing"])
    assert a.command == "cost" and a.no_billing
    a = p.parse_args(["cleanup", "--run", "r", "--dry-run"])
    assert a.command == "cleanup" and a.dry_run
    a = p.parse_args(["launch", "--model", "m", "--data", "d", "--gpu", "H100:8", "--budget-usd", "500"])
    cfg = guard_config_from_args(a)
    assert cfg.budget_usd == 500 and cfg.warn_usd == 400
    assert cfg.idle.idle_after_s == 15 * MIN and cfg.idle.stop_after_s == 15 * MIN
    assert cfg.idle.load_grace_s == 45 * MIN and cfg.stop_grace_s == 20 * MIN
    led = ResourceLedger.for_run("r")
    ResourceLedger(led.path, clock=Clock(T0)).created("modal_app", "ap-1", "a", usd_per_hour=6.0)
    monkeypatch.setattr("time.time", lambda: T0 + H)
    assert cli.main(["cost", "--run", "r", "--no-billing", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["total_usd"] == 6.0
    ops = FakeOps()
    monkeypatch.setattr(cleanup_mod, "ProductionCleanupOps", lambda: ops)
    assert cli.main(["cleanup", "--run", "r"]) == 0 and ops.calls == [("stop_app", "ap-1")]


def test_modal_ops_record_deploy_spawn_cancel_and_stop(tmp_path, monkeypatch):
    from yeto import modal_runner

    led = ResourceLedger(tmp_path / "r.jsonl")

    class Call:
        object_id = "fc-7"

    class Fn:
        def spawn(self, payload):
            return Call()

        def cancel(self, **kw):
            pass

    class App:
        app_id = "ap-7"

        def deploy(self, name):
            pass

    class Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class FakeModal:
        class Function:
            @staticmethod
            def from_name(app, fn):
                return Fn()

        class FunctionCall:
            @staticmethod
            def from_id(i):
                return Fn()

        @staticmethod
        def enable_output():
            return Ctx()

    ops = modal_runner.ModalOps("yeto-r")
    ops.ledger = led
    ops._app = App()
    ops._functions = {"island": object()}
    monkeypatch.setattr(ops, "_modal", lambda: FakeModal)

    class Cfg:
        function_name = "island"
        learner_id = 0

        def to_json(self):
            return {}

    ops.deploy()
    assert ops.spawn(Cfg()) == "fc-7"
    assert {r.key for r in led.open_resources()} == {"modal_app:ap-7", "modal_call:fc-7"}
    ops.cancel("fc-7")
    assert {r.key for r in led.open_resources()} == {"modal_app:ap-7"}

    class Proc:
        returncode = 0
        stdout = "stopped"
        stderr = ""

    monkeypatch.setattr(modal_runner.subprocess, "run", lambda *a, **k: Proc())
    ops.stop_app()
    assert led.open_resources() == []


def test_engine_sampler_launch_records_its_ephemeral_app_and_call(tmp_path, monkeypatch):
    from yeto.cloud import modal_engine_sampler as mes
    from yeto.cost.ledger import LEDGER_ENV

    led_path = tmp_path / "resources.jsonl"
    monkeypatch.setenv(LEDGER_ENV, str(led_path))

    class Call:
        object_id = "fc-s"

    class Fn:
        def spawn(self, rec):
            return Call()

    class App:
        app_id = "ap-s"

        def run(self, detach):
            import contextlib

            return contextlib.nullcontext()

    monkeypatch.setattr(mes, "build_app", lambda c: (App(), Fn()))

    class C:
        app_name, gpu, tp, shard, trainer_buffer_shards = "yeto-k31-sampler-b300", "B300", 8, "s0", "x"

        def record(self):
            return {}

    assert mes.launch(C(), wait=False, log=lambda m: None) == "fc-s"
    res = {r.key: r for r in ResourceLedger(led_path).open_resources()}
    assert set(res) == {"modal_app:ap-s", "modal_call:fc-s"}
    assert res["modal_app:ap-s"].attrs["gpus"] == 8


def test_automatic_cleanup_keeps_the_nc_pool_lease(tmp_path, monkeypatch):
    """A trainer launch's backstop / guard cleanup must not release the NC pool's node."""
    monkeypatch.setenv("HOME", str(tmp_path))
    from yeto.cost import cleanup as cleanup_mod
    from yeto.cost.ledger import ResourceLedger

    led = ResourceLedger.for_run("r-nc")
    led.created("nc_lease", "t1", "r-nc", account=5)
    calls = []

    class Ops:
        def release_nc(self, r):
            calls.append(r.id)
            return {"released": True}

    assert cleanup_mod.cleanup_run("r-nc", Ops(), log=lambda m: None)
    assert calls == [] and [r.kind for r in led.open_resources()] == ["nc_lease"]


def test_a_modal_sampler_pools_calls_count_before_the_billing_report_shows_them(tmp_path):
    # the pool's app is recorded without a rate; each call carries the pool's price_usd_h
    p = tmp_path / "r.jsonl"
    ledger_at(p, T0).created("modal_app", "ap-roll", "yeto-rollout-rp", role="rollout", run="rp")
    ledger_at(p, T0).created("modal_call", "fc-1", "rollout_sampler", ticket="yeto-smp-1", run="rp",
                             pool=True, usd_per_hour=68.0)
    ledger_at(p, T0 + H).created("modal_call", "fc-2", "rollout_sampler", ticket="yeto-smp-2", run="rp",
                                 pool=True, usd_per_hour=68.0)
    s = resource_costs(ResourceLedger(p).resources(), T0 + 2 * H, billing=ModalBilling(lambda s, e: []))
    assert s.total_usd == pytest.approx(3 * 68.0)  # 2 h + 1 h of sampler calls, nothing billed yet
    assert not s.unpriced
    # once the report covers the first hour, the calls estimate only the time after it
    rows = [BillingRow("ap-roll", "yeto-rollout-rp", T0, 70.0)]
    s = resource_costs(ResourceLedger(p).resources(), T0 + 2 * H, billing=ModalBilling(lambda s, e: rows))
    assert s.total_usd == pytest.approx(70.0 + 68.0 + 68.0)


def test_the_guard_stops_on_a_modal_sampler_pool_before_its_billing_arrives(tmp_path, runs_dir):
    p = tmp_path / "r.jsonl"
    ledger_at(p, T0).created("modal_app", "ap-roll", "yeto-rollout-rp", role="rollout", run="rp")
    ledger_at(p, T0).created("modal_call", "fc-1", "rollout_sampler", ticket="t", run="rp", pool=True,
                             usd_per_hour=68.0)
    clock, stops = Clock(T0 + 2 * H), []
    g = CostGuard("rp", GuardConfig(budget_usd=100.0), ledger=ResourceLedger(p),
                  billing=ModalBilling(lambda s, e: []), graceful_stop=stops.append,
                  force_stop=lambda why: None, clock=clock, log=lambda m: None, start=T0)
    g.tick()
    assert stops and "budget" in stops[0]


def test_a_respawned_pool_call_stops_counting_when_its_ticket_starts_the_next(tmp_path):
    # a restart on the same ticket cancels the old call; it must not keep accruing next to the new one
    p = tmp_path / "r.jsonl"
    ledger_at(p, T0).created("modal_app", "ap-roll", "yeto-rollout-rp", role="rollout", run="rp")
    ledger_at(p, T0).created("modal_call", "fc-1", "rollout_sampler", ticket="t1", run="rp",
                             pool=True, usd_per_hour=68.0)
    ledger_at(p, T0 + H).created("modal_call", "fc-2", "rollout_sampler", ticket="t1", run="rp",
                                 pool=True, usd_per_hour=68.0)
    s = resource_costs(ResourceLedger(p).resources(), T0 + 3 * H, billing=ModalBilling(lambda s, e: []))
    assert s.total_usd == pytest.approx(3 * 68.0)


def test_pool_calls_of_an_earlier_app_count_from_their_own_apps_billing(tmp_path):
    # the run redeployed its rollout app; the old app's calls after its billed hour still count
    p = tmp_path / "r.jsonl"
    ledger_at(p, T0).created("modal_app", "ap-old", "yeto-rollout-rp", role="rollout", run="rp")
    ledger_at(p, T0).created("modal_call", "fc-1", "rollout_sampler", ticket="t1", run="rp",
                             pool=True, usd_per_hour=68.0)
    ledger_at(p, T0 + 3 * H).released("modal_call", "fc-1", "rollout_sampler")
    ledger_at(p, T0 + 4 * H).created("modal_app", "ap-new", "yeto-rollout-rp", role="rollout", run="rp")
    ledger_at(p, T0 + 4 * H).created("modal_call", "fc-2", "rollout_sampler", ticket="t2", run="rp",
                                     pool=True, usd_per_hour=68.0)
    rows = [BillingRow("ap-old", "yeto-rollout-rp", T0, 68.0)]
    s = resource_costs(ResourceLedger(p).resources(), T0 + 5 * H, billing=ModalBilling(lambda s, e: rows))
    assert s.total_usd == pytest.approx(68.0 + 2 * 68.0 + 68.0)
