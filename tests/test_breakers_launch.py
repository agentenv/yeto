"""Launch-layer breakers (``yeto.breakers.layers.launch``): declarations, legacy aliases,
budget inheritance, write-back, and the ``rl_breaker`` lines of the cost guard and the
launcher (stall, trainer failures, teardown). All CPU: fake clocks, fake billing."""

from __future__ import annotations

import argparse
import json

import pytest

from yeto import breakers, cli, runs
from yeto.breakers.layers import launch as launch_layer
from yeto.cost.billing import ModalBilling
from yeto.cost.guard import (
    DEFAULT_LOAD_GRACE_MINUTES, CostGuard, GuardConfig, TapeTail, guard_config_from_args, guard_enabled,
)
from yeto.cost.idle import IdleConfig
from yeto.cost.ledger import ResourceLedger
from yeto.cost.report import usd_or_off
from yeto.launch import breakers_file, trips
from yeto.launch.fleet import (
    DEFAULT_RL_STALL_TIMEOUT_S, RUN_STALLED_EXIT, TEARDOWN_UNVERIFIED_EXIT, TRAINER_GAVE_UP_EXIT,
    RunStalled, effective_stall_timeout,
)

H = 3600.0
MIN = 60.0
T0 = 1_800_000_000.0 - (1_800_000_000.0 % H)
BASE = ["--model", "m", "--data", "d", "--gpu", "H100:8"]
LAUNCH_ENTRIES = ("launch.budget", "launch.idle", "launch.nc_lease_overdue", "launch.stall",
                  "launch.trainer_failures", "launch.deterministic_stop", "launch.island_relaunches",
                  "launch.provider_timeout", "launch.teardown_verify")


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    return tmp_path / "runs"


@pytest.fixture(autouse=True)
def no_stop_reason():
    """The process-wide stop reason starts empty in every test."""
    breakers.clear_stop_reason()
    yield
    breakers.clear_stop_reason()


class Clock:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def parse(*flags: str) -> argparse.Namespace:
    return cli.build_parser().parse_args(["launch", *BASE, *flags])


def resolve(*flags: str, section=None):
    return breakers.resolve(breakers.Sources(argv=[*BASE, *flags], section=section))


def launch_parser_actions() -> dict[str, argparse.Action]:
    parser = cli.build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return {opt: action for action in sub.choices["launch"]._actions for opt in action.option_strings}


# -- declarations ------------------------------------------------------------------

def test_the_registry_loads_every_launch_entry_and_the_cost_guard_switch():
    reg = breakers.default_registry()
    assert all(name in reg for name in LAUNCH_ENTRIES)
    assert {b.name for b in launch_layer.BREAKERS} == set(LAUNCH_ENTRIES)
    (group,) = launch_layer.SWITCH_GROUPS
    assert group.name == "cost_guard"
    assert group.members == ("launch.budget", "launch.idle", "launch.nc_lease_overdue")
    assert [a.path for a in group.aliases] == ["--no-cost-guard"]
    teardown = next(b for b in launch_layer.BREAKERS if b.name == "launch.teardown_verify")
    assert teardown.kind == "invariant" and not teardown.can_disable


def test_every_alias_is_a_yeto_launch_flag():
    actions = launch_parser_actions()
    for entry in launch_layer.BREAKERS:
        for alias in entry.aliases:
            assert alias.surface == "launch_flag" and alias.path in actions, (entry.name, alias.path)
    for group in launch_layer.SWITCH_GROUPS:
        for alias in group.aliases:
            assert alias.path in actions


def test_defaults_equal_the_code_constants_and_argparse_defaults():
    from yeto.modal_runner import DEFAULT_RETRIES, DEFAULT_TIMEOUT_S

    actions = launch_parser_actions()
    entries = {b.name: {p.name: p.default for p in b.params} for b in launch_layer.BREAKERS}
    guard, idle = GuardConfig(), IdleConfig()
    assert entries["launch.budget"] == {"usd": guard.budget_usd, "warn_usd": guard.budget_warn_usd,
                                        "stop_grace_s": guard.stop_grace_s}
    assert entries["launch.budget"]["stop_grace_s"] == actions["--budget-stop-grace-minutes"].default * MIN
    assert actions["--budget-usd"].default is None and actions["--budget-warn-usd"].default is None
    assert entries["launch.idle"]["warn_s"] == idle.idle_after_s == actions["--idle-warn-minutes"].default * MIN
    assert entries["launch.idle"]["stop_after_warn_s"] == idle.stop_after_s \
        == actions["--idle-stop-minutes"].default * MIN
    assert entries["launch.idle"]["max_load_s"] == idle.max_load_s == actions["--max-load-minutes"].default * MIN
    # null: the recipe default (guard.default_load_grace_minutes), as the flag's None default
    assert entries["launch.idle"]["load_grace_s"] is None and actions["--load-grace-minutes"].default is None
    assert entries["launch.stall"]["timeout_s"] == DEFAULT_RL_STALL_TIMEOUT_S
    assert actions["--rl-stall-timeout"].default is None  # unset: DEFAULT_RL_STALL_TIMEOUT_S
    assert entries["launch.trainer_failures"]["max_n"] == actions["--rl-trainer-max-failures"].default
    assert entries["launch.provider_timeout"] == {"timeout_s": DEFAULT_TIMEOUT_S, "retries_n": DEFAULT_RETRIES}
    assert entries["launch.nc_lease_overdue"]["grace_s"] == guard.nc_overdue_grace_s \
        == actions["--nc-lease-overdue-minutes"].default * MIN


def test_the_trainer_failures_default_is_the_trainer_loop_constant():
    from yeto.rl.engine.trainer_loop import DEFAULT_MAX_CONSECUTIVE_FAILURES

    entry = next(b for b in launch_layer.BREAKERS if b.name == "launch.trainer_failures")
    assert entry.params[0].default == DEFAULT_MAX_CONSECUTIVE_FAILURES


# -- resolution without a section: today's values ---------------------------------------

def test_no_section_resolves_to_what_the_launch_reads_today():
    rb = resolve()
    cfg = guard_config_from_args(parse())
    assert not rb.has_section
    assert not rb.enabled("launch.budget") and rb.get("launch.budget.usd") is None and cfg.budget_usd is None
    assert rb.enabled("launch.idle") and rb.get("launch.idle.warn_s") == cfg.idle.idle_after_s
    assert rb.get("launch.idle.stop_after_warn_s") == cfg.idle.stop_after_s
    assert rb.get("launch.idle.max_load_s") == cfg.idle.max_load_s
    assert rb.get("launch.budget.stop_grace_s") == cfg.stop_grace_s
    assert cfg.idle.load_grace_s == DEFAULT_LOAD_GRACE_MINUTES * MIN
    assert rb.enabled("launch.stall") and rb.get("launch.stall.timeout_s") == effective_stall_timeout(parse())
    assert rb.switches()["cost_guard"] is True


def test_legacy_flags_resolve_with_their_source():
    rb = resolve("--budget-usd", "500", "--budget-stop-grace-minutes", "5", "--idle-warn-minutes", "30",
                 "--rl-stall-timeout", "3600", "--rl-trainer-max-failures", "5", "--modal-timeout-s", "7200",
                 "--nc-lease-overdue-minutes", "10")
    assert rb.get("launch.nc_lease_overdue.grace_s") == 10 * MIN
    assert rb.enabled("launch.budget") and rb.get("launch.budget.usd") == 500
    assert rb.entry("launch.budget").sources["usd"].kind == "legacy"
    assert rb.get("launch.budget.stop_grace_s") == 5 * MIN
    assert rb.get("launch.idle.warn_s") == 30 * MIN
    assert rb.get("launch.stall.timeout_s") == 3600
    assert rb.get("launch.trainer_failures.max_n") == 5
    assert rb.get("launch.provider_timeout.timeout_s") == 7200
    cfg = guard_config_from_args(parse("--budget-usd", "500", "--budget-stop-grace-minutes", "5",
                                       "--idle-warn-minutes", "30", "--nc-lease-overdue-minutes", "10"))
    assert (cfg.budget_usd, cfg.stop_grace_s, cfg.idle.idle_after_s) == (500, 5 * MIN, 30 * MIN)
    assert cfg.nc_overdue_grace_s == 10 * MIN


@pytest.mark.parametrize("flags, off", [
    (("--no-cost-guard",), {"launch.budget", "launch.idle", "launch.nc_lease_overdue"}),
    (("--no-cost-guard", "--budget-usd", "100"), {"launch.budget", "launch.idle", "launch.nc_lease_overdue"}),
    (("--idle-warn-minutes", "0"), {"launch.budget", "launch.idle"}),  # budget: off by default
    (("--rl-stall-timeout", "0"), {"launch.budget", "launch.stall"}),
    (("--budget-usd", "off"), {"launch.budget"}),
])
def test_legacy_off_spellings(flags, off):
    rb = resolve(*flags)
    assert {name for name in ("launch.budget", "launch.idle", "launch.nc_lease_overdue", "launch.stall")
            if not rb.enabled(name)} == off


def test_no_cost_guard_is_the_switch_group():
    rb = resolve("--no-cost-guard")
    assert rb.switches()["cost_guard"] is False
    assert rb.entry("launch.idle").sources["enabled"].kind == "switch"
    assert "cost_guard" in rb.entry("launch.budget").switch_groups
    assert not guard_enabled(parse("--no-cost-guard"))


def test_budget_usd_off_parses_as_no_budget():
    assert usd_or_off("off") is None and usd_or_off("None") is None and usd_or_off("12.5") == 12.5
    assert parse("--budget-usd", "off").budget_usd is None
    with pytest.raises(ValueError):
        usd_or_off("lots")


# -- budget inheritance (decision 3) -----------------------------------------------------

def test_launch_budget_inherits_the_section_run_budget_and_is_written_back():
    section = {"run": {"budget": {"usd": 450}}}
    sources = breakers.Sources(argv=list(BASE), section=section)
    rb = breakers.resolve(sources)
    assert rb.enabled("launch.budget") and rb.get("launch.budget.usd") == 450
    assert rb.entry("launch.budget").sources["usd"].kind == "inherit"
    out = breakers.materialize(rb, sources, surfaces=("launch_flag",))
    cfg = guard_config_from_args(cli.build_parser().parse_args(["launch", *out.argv]))
    assert cfg.budget_usd == 450 and cfg.warn_usd == 0.8 * 450


def test_a_legacy_run_budget_alone_is_not_inherited():
    """budget.usd without a breakers: section: the launcher keeps no budget, as today."""
    rb = breakers.resolve(breakers.Sources(argv=list(BASE), run_config={"budget": {"usd": 45}}))
    assert not rb.enabled("launch.budget") and rb.get("launch.budget.usd") is None


def test_an_explicit_launch_budget_wins_over_the_inherited_one():
    rb = resolve("--budget-usd", "600", section={"run": {"budget": {"usd": 450}}})
    assert rb.get("launch.budget.usd") == 600
    assert rb.entry("launch.budget").sources["usd"].kind == "legacy"  # D11 warns on this one
    rb = resolve(section={"run": {"budget": {"usd": 450}}, "launch": {"budget": {"usd": 300}}})
    assert rb.get("launch.budget.usd") == 300 and rb.entry("launch.budget").sources["usd"].kind == "breakers"


def test_a_disabled_run_budget_leaves_the_launch_without_one():
    rb = resolve(section={"run": {"budget": {"enabled": False}}})
    assert not rb.enabled("launch.budget") and rb.get("launch.budget.usd") is None


def test_a_launch_budget_above_the_run_budget_is_a_d11_warning():
    rules = pytest.importorskip("yeto.breakers.rules")
    sources = breakers.Sources(argv=[*BASE, "--budget-usd", "600"], section={"run": {"budget": {"usd": 450}}})
    findings = rules.check(breakers.resolve(sources), sources=sources)
    assert any(f.rule == "D11" and f.level == "warn" for f in findings)
    inherited = breakers.Sources(argv=list(BASE), section={"run": {"budget": {"usd": 450}}})
    assert not any(f.rule == "D11" for f in rules.check(breakers.resolve(inherited), sources=inherited))


# -- write-back from a section ----------------------------------------------------------

def written(section, *flags: str) -> list[str]:
    sources = breakers.Sources(argv=[*BASE, *flags], section=section)
    return breakers.materialize(breakers.resolve(sources), sources, surfaces=("launch_flag",)).argv


def test_section_values_become_the_launch_flags():
    argv = written({"launch": {"stall": {"timeout_s": 3600}, "idle": {"warn_s": 1800, "load_grace_s": 5400},
                               "trainer_failures": {"max_n": 5}, "budget": {"usd": 80}}})
    a = cli.build_parser().parse_args(["launch", *argv])
    assert effective_stall_timeout(a) == 3600 and a.rl_trainer_max_failures == 5
    cfg = guard_config_from_args(a)
    assert cfg.budget_usd == 80 and cfg.idle.idle_after_s == 1800 and cfg.idle.load_grace_s == 5400


def test_disabled_entries_become_the_legacy_off_spellings():
    a = cli.build_parser().parse_args(["launch", *written({"switches": {"cost_guard": False}})])
    assert a.no_cost_guard and not guard_enabled(a)
    a = cli.build_parser().parse_args(["launch", *written({"launch": {"idle": {"enabled": False},
                                                                      "stall": {"enabled": False}}})])
    assert a.idle_warn_minutes == 0 and not guard_config_from_args(a).idle.enabled
    assert effective_stall_timeout(a) == 0.0
    a = cli.build_parser().parse_args(["launch", *written({"launch": {"budget": {"enabled": False}}},
                                                          )])
    assert a.budget_usd is None


def test_no_section_writes_the_argv_back_unchanged():
    flags = ["--budget-usd", "35", "--idle-warn-minutes", "0", "--rl-stall-timeout=3600", "--no-cost-guard"]
    assert written(None, *flags) == [*BASE, *flags]


@pytest.mark.parametrize("section, flags", [
    ({"launch": {"stall": {"timeout_s": 3600}}}, ("--rl-stall-timeout", "900")),
    ({"launch": {"budget": {"enabled": False}}}, ("--budget-usd", "500")),
    ({"switches": {"cost_guard": False}, "launch": {"idle": {"enabled": True}}}, ()),
    ({"launch": {"idle": {"enabled": False}}}, ("--idle-warn-minutes", "20")),
])
def test_a_section_that_disagrees_with_a_flag_is_refused(section, flags):
    with pytest.raises(breakers.BreakerConfigError):
        resolve(*flags, section=section)


@pytest.mark.parametrize("section", [
    {"launch": {"teardown_verify": {"enabled": False}}},
    {"launch": {"trainer_failures": {"max_n": 0}}},
    {"launch": {"idle": {"warn_s": 0}}},
])
def test_bad_launch_sections_are_refused(section):
    with pytest.raises(breakers.BreakerConfigError):
        breakers.parse_section(section)


# -- yeto launch --breakers FILE --------------------------------------------------------

def test_breakers_file_applies_a_whole_run_config_to_the_launch_argv(tmp_path):
    cfg = tmp_path / "run.json"
    cfg.write_text(json.dumps({"run": "r", "breakers": {
        "run": {"budget": {"usd": 450}}, "launch": {"stall": {"timeout_s": 3600}}}}))
    logs: list[str] = []
    argv, rb = breakers_file.apply(cfg, [*BASE, "--idle-warn-minutes", "0"], log=logs.append)
    a = cli.build_parser().parse_args(["launch", *argv])
    assert guard_config_from_args(a).budget_usd == 450 and effective_stall_timeout(a) == 3600
    assert a.idle_warn_minutes == 0 and rb.get("launch.budget.usd") == 450
    assert any("not applied in this process: run" in m for m in logs)
    assert any("--budget-usd 450" in m for m in logs)  # the write-back is logged


def test_breakers_file_conflict_refuses_the_launch(tmp_path, capsys):
    f = tmp_path / "b.json"
    f.write_text(json.dumps({"launch": {"stall": {"timeout_s": 3600}}}))
    assert cli.main(["launch", *BASE, "--rl-stall-timeout", "900", "--breakers", str(f)]) == 2
    assert "--rl-stall-timeout" in capsys.readouterr().err


def test_breakers_file_saves_the_resolved_view(tmp_path, runs_dir):
    f = tmp_path / "b.json"
    f.write_text(json.dumps({"run": {"budget": {"usd": 450}}}))
    _argv, rb = breakers_file.apply(f, list(BASE), log=lambda m: None)
    path = breakers_file.save_resolved("r", rb)
    doc = json.loads(path.read_text())
    entry = next(e for e in doc["entries"] if e["name"] == "launch.budget")
    assert entry["params"]["usd"] == 450 and doc["has_section"]


# -- trips: the cost guard ---------------------------------------------------------------

def make_guard(tmp_path, clock, *, budget=100.0, idle_cfg=None, tapes=()):
    led = ResourceLedger(tmp_path / "resources.jsonl", clock=clock)
    calls: list[str] = []
    g = CostGuard("r1", GuardConfig(budget_usd=budget, stop_grace_s=20 * MIN,
                                    idle=idle_cfg or IdleConfig(idle_after_s=0)),
                  ledger=led, tapes=TapeTail(list(tapes)), billing=ModalBilling(lambda s, e: [], refresh_s=0),
                  graceful_stop=calls.append, force_stop=calls.append, release_resource=lambda r: None,
                  clock=clock, log=lambda m: None, start=T0)
    return g, led, calls


def tape(tmp_path) -> list[dict]:
    return [json.loads(x) for x in (tmp_path / "cost-events.jsonl").read_text().splitlines()]


def breaker_lines(tmp_path, name=None) -> list[dict]:
    return [r for r in tape(tmp_path) if r["event"] == "rl_breaker" and name in (None, r["name"])]


def test_a_budget_stop_writes_rl_breaker_and_the_stop_reason(tmp_path):
    clock = Clock(T0)
    g, led, calls = make_guard(tmp_path, clock, budget=100.0)
    led.created("modal_app", "ap-1", "yeto-r1", usd_per_hour=60.0)
    clock.t = T0 + 1.4 * H  # $84: the warn line
    g.tick()
    clock.t = T0 + 1.7 * H  # $102
    g.tick()
    warn, trip = breaker_lines(tmp_path, "launch.budget")
    assert warn["state"] == "warn" and warn["limit"] == 80.0 and "stop_reason" not in warn
    assert trip["state"] == "trip" and trip["value"] == 102.0 and trip["limit"] == 100.0
    assert trip["layer"] == "launch" and trip["action"] == "stop" and trip["unit"] == "usd"
    assert trip["stop_reason"] == "breaker:launch.budget" and trip["run"] == "r1"
    assert breakers.current_stop_reason() == "breaker:launch.budget"
    legacy = [r["event"] for r in tape(tmp_path) if r["event"] != "cost_tick"]
    assert legacy == ["cost_budget_warn", "rl_breaker", "cost_budget_exceeded", "rl_breaker",
                      "cost_guard_stop"]
    assert len(calls) == 1 and "budget" in calls[0]  # the graceful stop, as before


def test_an_idle_shutdown_writes_rl_breaker(tmp_path):
    clock = Clock(T0)
    t = tmp_path / "events" / "i.jsonl"
    t.parent.mkdir()
    t.write_text(json.dumps({"event": "rl_round_trained", "time_unix": T0 + 50 * MIN}) + "\n")
    g, _led, calls = make_guard(tmp_path, clock, budget=None, tapes=[str(t)],
                                idle_cfg=IdleConfig(idle_after_s=15 * MIN, stop_after_s=15 * MIN,
                                                    load_grace_s=45 * MIN))
    clock.t = T0 + 66 * MIN
    g.tick()
    clock.t = T0 + 81 * MIN
    g.tick()
    warn, trip = breaker_lines(tmp_path, "launch.idle")
    assert warn["state"] == "warn" and warn["limit"] == 15 * MIN
    assert trip["state"] == "trip" and trip["limit"] == 30 * MIN and trip["value"] >= 30 * MIN
    assert trip["stop_reason"] == "breaker:launch.idle" and len(calls) == 1


def test_an_overdue_nc_lease_is_an_alert_without_a_stop_reason(tmp_path):
    clock = Clock(T0)
    g, led, calls = make_guard(tmp_path, clock, budget=None)
    led.created("nc_lease", "t-1", "n", usd_per_hour=1.0, lease_until=T0 + H)
    clock.t = T0 + H + 6 * MIN
    g.tick()
    (rec,) = breaker_lines(tmp_path, "launch.nc_lease_overdue")
    assert rec["action"] == "warn" and rec["scope"]["resource"].endswith("t-1")
    assert "stop_reason" not in rec and breakers.current_stop_reason() is None and not calls


def test_a_failing_rl_breaker_writer_does_not_change_the_guard(tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("no registry")

    monkeypatch.setattr(breakers, "trip", boom)
    clock = Clock(T0)
    g, led, calls = make_guard(tmp_path, clock, budget=10.0)
    led.created("modal_app", "ap-1", "yeto-r1", usd_per_hour=60.0)
    clock.t = T0 + H
    g.tick()
    assert "cost_budget_exceeded" in [r["event"] for r in tape(tmp_path)] and len(calls) == 1


# -- trips: the launcher ---------------------------------------------------------------

def test_launcher_trips_go_to_the_launch_tape_with_their_exit_codes_unchanged(runs_dir):
    assert (RUN_STALLED_EXIT, TRAINER_GAVE_UP_EXIT, TEARDOWN_UNVERIFIED_EXIT) == (6, 8, 5)
    trips.trip("r", "launch.stall", 901.0, 900.0, legacy_event="RunStalled", detail="no island event")
    lines = [json.loads(x) for x in (runs.run_dir("r") / "cost-events.jsonl").read_text().splitlines()]
    (rec,) = lines
    assert rec["event"] == "rl_breaker" and rec["name"] == "launch.stall" and rec["unit"] == "s"
    assert rec["stop_reason"] == "breaker:launch.stall" and rec["legacy_event"] == "RunStalled"
    trips.trip("r", "launch.trainer_failures", 3, 3)
    assert breakers.current_stop_reason() == "breaker:launch.stall"  # the first stop wins


def test_a_launcher_trip_never_raises(runs_dir, monkeypatch):
    monkeypatch.setattr(breakers, "trip", lambda *a, **k: 1 / 0)
    assert trips.trip("r", "launch.stall", 1.0, 1.0) is None


def test_run_stalled_carries_the_silence_and_the_limit():
    class Ops:
        t = 0.0

        def now(self):
            return self.t

    from yeto.launch.fleet import FleetController

    ops, count = Ops(), [5]
    fc = FleetController({}, None, ops, poll_interval=1, recover_timeout=0,
                         progress_probe=lambda: count[0], stall_timeout=900.0)
    fc._check_stall()
    ops.t = 901.0
    with pytest.raises(RunStalled) as info:
        fc._check_stall()
    assert info.value.silent_s == 901.0 and info.value.timeout_s == 900.0


@pytest.mark.parametrize("value, expected", [(None, DEFAULT_RL_STALL_TIMEOUT_S), (0, 0.0), (0.0, 0.0),
                                             (3600, 3600.0)])
def test_effective_stall_timeout(value, expected):
    assert effective_stall_timeout(argparse.Namespace(rl_stall_timeout=value)) == expected
    assert effective_stall_timeout(argparse.Namespace()) == DEFAULT_RL_STALL_TIMEOUT_S


def test_the_worker_records_the_stop_reason(runs_dir, monkeypatch):
    runs.create_run("r", {"cluster_prefix": "r"})

    def launcher_run(args, on_clusters=None):
        trips.trip("r", "launch.stall", 901.0, 900.0)
        return RUN_STALLED_EXIT

    monkeypatch.setattr("yeto.launcher.run", launcher_run)
    monkeypatch.setattr(cli, "_backstop_cleanup", lambda name, args: None)
    assert cli.cmd_worker("r") == RUN_STALLED_EXIT
    meta = runs.load_run("r")
    assert meta["exit_code"] == 6 and meta["stop_reason"] == "breaker:launch.stall"


def test_the_launch_cli_prints_each_deprecation_once_and_never_fails():
    class Resolved:
        def deprecations(self):
            return ["launch_flag --old is deprecated: write breakers.launch.x.y"]

    lines: list[str] = []
    breakers_file.print_deprecations(BASE, Resolved(), log=lines.append)
    assert lines == ["[yeto] DEPRECATED: launch_flag --old is deprecated: write breakers.launch.x.y"]
    lines.clear()
    breakers_file.print_deprecations([*BASE, "--rl-trainer-max-failures", "0"], log=lines.append)
    assert lines == []  # no launch alias is deprecated yet; an out-of-bounds flag is the launch's own check
