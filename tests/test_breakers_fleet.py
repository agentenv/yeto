"""Breakers of an island fleet (``islands`` in a run config): scope, not semantics.

Each island gets its own run and trainer entries, rules and stops; run.budget and
run.wall_clock span the fleet, whose spend is every pool, every island's trainer and the
syncer. A run config without ``islands`` reads exactly as before.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from yeto import breakers
from yeto.breakers import cli as bcli
from yeto.breakers import fleet, rules, rules_facts
from yeto.breakers.resolve import Sources, resolve

SPEC_AGE12 = str(Path(__file__).resolve().parents[1] / "examples" / "k31" / "algorithm-spec.async-age12-tis.json")
FLEET_ARGV = ["--training-mode", "rl", "--model", "Qwen/Qwen3-4B", "--data", "prompts.jsonl",
              "--rl-algorithm-spec", SPEC_AGE12, "--rl-stall-timeout", "3600", "--rl-sample-wait-s", "2400",
              "--rollout-batch-size", "8", "--n-samples-per-prompt", "8", "--budget-usd", "400"]
PLAIN_SPEC = {"sampling": {"n": 8}}
AGENTIC_SPEC = {"sampling": {"n": 8},
                "agentic": {"weight_switch": "live", "task_defaults": {"agent_timeout_s": 1800}}}


@pytest.fixture(autouse=True)
def _no_stop_reason():
    breakers.clear_stop_reason()
    yield
    breakers.clear_stop_reason()


def _pool(tmp_path, name, spec):
    spec_path = tmp_path / f"{name}-spec.json"
    spec_path.write_text(json.dumps(spec))
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"spec": str(spec_path), "target": 2}))
    return str(path)


def _fleet(tmp_path, **extra):
    return {
        "run": "r", "store": "s3://b/r", "budget": {"usd": 300, "wall_h": 5}, "syncer": {"usd_per_h": 1.5},
        "fleet": {"trainer_argv": list(FLEET_ARGV), "dispense": "fleet"},
        "islands": [
            {"name": "east-b300", "trainer": {"gpu": "modal:1x8xb300", "usd_per_h": 48.0,
                                              "argv": ["--rl-release-every", "2"]},
             "samplers": [{"name": "nc-a", "config": _pool(tmp_path, "plain", PLAIN_SPEC)}]},
            {"name": "west-h100", "trainer": {"gpu": "modal:2xh100", "usd_per_h": 8.0,
                                              "argv": ["--rl-max-policy-age", "4"]},
             "samplers": [{"name": "nc-a", "config": _pool(tmp_path, "agentic", AGENTIC_SPEC)}]},
        ], **extra}


def _write(tmp_path, cfg):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(cfg))
    return str(path)


def _run(argv):
    p = argparse.ArgumentParser()
    bcli.add_parser(p.add_subparsers(dest="command"))
    return bcli.main(p.parse_args(bcli.normalize_argv(argv)))


# --------------------------------------------------------------------------- scope


def test_only_budget_and_wall_clock_span_the_fleet():
    assert fleet.breaker_scope("run.budget") == fleet.breaker_scope("run.wall_clock") == "fleet"
    assert fleet.breaker_scope("launch.stall") == "fleet"  # the fleet's one yeto launch
    assert fleet.breaker_scope("run.starvation") == fleet.breaker_scope("trainer.sample_wait") == "island"
    assert fleet.breaker_scope("pool.engine_stall") == fleet.breaker_scope("sampler.drain") == "component"
    names = [b.name for b in breakers.default_registry()]
    spanning = [n for n in names if fleet.breaker_scope(n) == "fleet" and not n.startswith("launch.")]
    assert spanning == ["run.budget", "run.wall_clock"]


def test_each_island_reads_as_one_trainers_run_config(tmp_path):
    cfg = _fleet(tmp_path)
    assert fleet.islands(cfg) == (fleet.Island(0, "east-b300"), fleet.Island(1, "west-h100"))
    view = fleet.island_run_config(cfg, 1)
    assert "islands" not in view and "fleet" not in view and view["budget"] == cfg["budget"]
    assert view["trainer"] == {"argv": [*FLEET_ARGV, "--gpu", "modal:2xh100", "--rl-sample-role", "trainer",
                                        "--rl-sampler-follow", "trainer", "--rl-max-policy-age", "4"],
                               "usd_per_h": 8.0}
    assert [s["name"] for s in view["samplers"]] == ["i1-nc-a"]
    whole = fleet.fleet_run_config(cfg)
    assert whole["trainer"] == {"argv": FLEET_ARGV, "usd_per_h": 56.0}
    assert [n for n, _c, _p in rules_facts.run_pools(cfg)] == ["i0-nc-a", "i1-nc-a"]


def test_facts_hold_one_trainer_per_island(tmp_path):
    cfg = _fleet(tmp_path)
    rb = resolve(Sources.from_run_config(fleet.fleet_run_config(cfg)), strict=False)
    facts = rules_facts.collect(rb, run_config=cfg)
    assert facts.trainer is None and [p.name for p in facts.pools] == ["i0-nc-a", "i1-nc-a"]
    east, west = facts.islands
    assert (east.island.name, west.island.name) == ("east-b300", "west-h100")
    assert east.facts.trainer.modal and west.facts.trainer.modal and east.facts.trainer.buffer_trainer
    assert [p.name for p in east.facts.pools] == ["i0-nc-a"] and [p.name for p in west.facts.pools] == ["i1-nc-a"]
    # each island resolves its own trainer knobs: only west sets a policy age
    assert not east.resolved.enabled("trainer.policy_age")
    assert west.resolved.get("trainer.policy_age.max_versions") == 4
    assert [(i.index, r.get("trainer.sample_wait.timeout_s")) for i, r in fleet.resolve_islands(cfg)] == \
        [(0, 2400), (1, 2400)]


def test_run_up_can_hand_each_islands_pools_and_resolve(tmp_path):
    """``collect_fleet`` takes the pools and resolves run up already holds, island by island."""
    cfg = _fleet(tmp_path)
    resolved = dict(fleet.resolve_islands(cfg))
    west = fleet.Island(1, "west-h100")
    pools = {1: [("i1-nc-a", json.loads(Path(_pool(tmp_path, "plain2", PLAIN_SPEC)).read_text()), "r2")]}
    rb = resolve(Sources.from_run_config(fleet.fleet_run_config(cfg)), strict=False)
    facts = rules_facts.collect_fleet(rb, cfg, island_pools=pools, island_resolved={1: resolved[west]})
    assert facts.islands[1].resolved is resolved[west]
    assert not facts.islands[1].facts.pools[0].agentic  # the handed pool, not the run config's agentic one
    assert not [f for f in rules.evaluate(rb, facts) if f.rule == "D4"]


def test_rules_run_per_island_and_budget_once(tmp_path):
    cfg = _fleet(tmp_path)
    rb = resolve(Sources.from_run_config(fleet.fleet_run_config(cfg)), strict=False)
    found = rules.check(rb, run_config=cfg)
    sample_wait = [f for f in found if f.rule == "D4"]  # the agentic pool's longest group outlasts 2400 s
    assert len(sample_wait) == 1 and sample_wait[0].scope == {"island": "1", "island_name": "west-h100"}
    assert sample_wait[0].message.startswith("island 1 (west-h100): trainer.sample_wait 2400 s")
    assert sample_wait[0].to_json()["scope"] == {"island": "1", "island_name": "west-h100"}
    budget = [f for f in found if f.rule == "D11"]  # launch.budget 400 > run.budget 300: once for the fleet
    assert len(budget) == 1 and budget[0].scope == {} and "scope" not in budget[0].to_json()


def test_an_islands_own_knob_fails_only_that_island(tmp_path, capsys):
    """Island 1's --rl-max-policy-age 4 without a spec that tolerates it: D15 refuses island 1 only."""
    cfg = _fleet(tmp_path)
    at = cfg["fleet"]["trainer_argv"].index("--rl-algorithm-spec")
    del cfg["fleet"]["trainer_argv"][at:at + 2]
    rb = resolve(Sources.from_run_config(fleet.fleet_run_config(cfg)), strict=False)
    refused = rules.errors(rules.check(rb, run_config=cfg))
    assert [(f.rule, f.scope["island"]) for f in refused] == [("D15", "1")]
    assert _run(["breaker", "show", "--json", "--config", _write(tmp_path, cfg)]) == 1
    data = json.loads(capsys.readouterr().out)
    assert [f["rule"] for f in data["islands"][1]["findings"] if f["level"] == "error"] == ["D15"]
    assert not [f for f in data["islands"][0]["findings"] if f["level"] == "error"]


# --------------------------------------------------------------------------- stops


def _islands():
    return [fleet.Island(0, "east-b300"), fleet.Island(1, "west-h100")]


def test_a_sample_wait_trip_stops_only_its_island():
    from yeto.rl.engine import trainer_breakers

    stops = fleet.FleetStops(_islands())
    events = []
    trainer_breakers.trip(trainer_breakers.SAMPLE_WAIT, lambda event, **f: events.append({"event": event, **f}),
                          value=2401.0, limit=2400.0, scope={"island": "1"})
    assert events[0]["stop_reason"] == "breaker:trainer.sample_wait"
    assert stops.record(events[0]) == (1,)
    assert stops.stop_reason(1) == "breaker:trainer.sample_wait"
    assert stops.stop_reason(0) is None and stops.running() == (0,)
    assert stops.fleet_stop_reason is None  # island 0 trains on
    # the island whose tape it came from, when the event names none
    other = fleet.FleetStops(_islands())
    unscoped = {**events[0], "scope": {}}
    assert other.record(unscoped, island=0) == (0,) and other.running() == (1,)


def test_component_trips_and_warnings_stop_no_island():
    stops = fleet.FleetStops(_islands())
    halt = breakers.trip("pool.provision_breaker", 3, 3, scope={"pool": "i1-nc-a"})
    assert halt["stop_reason"] and stops.record(halt) == ()
    warn = breakers.trip("trainer.sample_wait", 2000, 2400, state="warn", scope={"island": "0"})
    assert stops.record(warn) == ()
    assert stops.running() == (0, 1)


def test_a_launch_trip_stops_every_island():
    stops = fleet.FleetStops(_islands())
    assert stops.record(breakers.trip("launch.stall", 901, 900)) == (0, 1)
    assert stops.fleet_stop_reason == "breaker:launch.stall"


# --------------------------------------------------------------------------- spend and budget


def test_fleet_spend_counts_pools_every_island_and_the_syncer(tmp_path):
    cfg = _fleet(tmp_path)
    assert fleet.island_prices(cfg) == (48.0, 8.0) and fleet.syncer_price(cfg) == 1.5
    spend = fleet.fleet_spend(pool_node_hours=[(10.0, 24.0), (2.0, 5.0)], island_hours=[3.0, 1.5],
                              island_usd_per_h=fleet.island_prices(cfg), syncer_hours=3.0,
                              syncer_usd_per_h=fleet.syncer_price(cfg))
    assert spend.pools_usd == 250.0 and spend.islands_usd == (144.0, 12.0) and spend.syncer_usd == 4.5
    assert spend.trainers_usd == 156.0 and spend.total_usd == 410.5
    assert spend.to_json() == {"usd": 410.5, "pools_usd": 250.0, "islands_usd": [144.0, 12.0], "syncer_usd": 4.5}
    # nothing priced bills nothing; a negative or unreadable amount counts 0
    assert fleet.fleet_spend(pool_node_hours=[(-1, 24), (5, None)], island_hours=[2.0],
                             island_usd_per_h=[float("nan")]).total_usd == 0.0
    assert fleet.island_prices({"islands": [{}]}) == (0.0,)
    # unset: the machine the syncer runs on (the launch default: the controller head VM on AWS)
    assert fleet.syncer_price({"syncer": {}}) == fleet.syncer_price({}) == 0.384
    with pytest.raises(ValueError, match="2 island hours for 1 island prices"):
        fleet.fleet_spend(pool_node_hours=[], island_hours=[1.0, 2.0], island_usd_per_h=[1.0])


def test_the_fleet_budget_stops_every_island_with_stop_reason_budget(tmp_path):
    cfg = _fleet(tmp_path)
    cap = cfg["budget"]["usd"]
    spend = fleet.fleet_spend(pool_node_hours=[(5.0, 24.0)], island_hours=[3.0, 3.0],
                              island_usd_per_h=fleet.island_prices(cfg))
    assert spend.total_usd == 288.0 and fleet.budget_trip(spend, cap) is None  # under the cap: no trip
    assert breakers.current_stop_reason() is None
    spend = fleet.fleet_spend(pool_node_hours=[(5.0, 24.0)], island_hours=[3.5, 3.0],
                              island_usd_per_h=fleet.island_prices(cfg))
    events = []
    event = fleet.budget_trip(spend, cap, emit=events.append, run="r")
    assert event["name"] == "run.budget" and event["value"] == 312.0 and event["limit"] == 300
    assert event["stop_reason"] == "breaker:run.budget" and events == [event]
    assert breakers.current_stop_reason() == "breaker:run.budget"
    stops = fleet.FleetStops(fleet.islands(cfg))
    assert stops.record(event) == (0, 1) and stops.running() == ()
    assert stops.fleet_stop_reason == "breaker:run.budget"
    assert stops.to_json() == {"stop_reason": "breaker:run.budget",
                               "islands": [{"index": 0, "name": "east-b300", "stop_reason": "breaker:run.budget"},
                                           {"index": 1, "name": "west-h100", "stop_reason": "breaker:run.budget"}]}
    assert not fleet.budget_tripped(5e8, None)  # budget.usd null: no cap, as for one trainer


# --------------------------------------------------------------------------- show


FLEET_SHOW_RUN_LAYER = """\
FLEET  run.budget and run.wall_clock span every island; launch, pool, sampler, harness and monitor rows as for one trainer
NAME            LAYER  KIND     ON  ACTION  EFFECTIVE                                                                 SOURCE
run.budget      run    breaker  on  stop    usd=300 warn_ratio=0.8 force_pools_ratio=1.2 release_pools_on_trip=false  run config budget.usd (legacy)     !D11
run.wall_clock  run    breaker  on  stop    hours=5                                                                   run config budget.wall_h (legacy)
SWITCHES
  cost_guard           on   launch.budget, launch.idle, launch.nc_lease_overdue
DEPENDENCIES
  D11  warn   launch.budget 400 USD > run.budget 300 USD: the trainer's cost guard (--budget-usd) would let the run spend past its budget (the run monitor and the dashboard read run.budget); remove the override to inherit run.budget, or lower it

ISLAND 0  east-b300  pools: i0-nc-a
NAME                     LAYER  KIND       ON   ACTION  EFFECTIVE                                                                                       SOURCE
run.trainer_silence      run    breaker    on   stop    limit_s=null floor_s=1800 stall_factor_n=4                                                      default
run.first_group          run    breaker    on   stop    timeout_s=5400                                                                                  default
run.starvation           run    breaker    on   stop    alert_s=1800 stop_s=7200 unhealthy_s=600 trainer_wait_max_s=7200 stop_grace_s=1200 check_s=300  default           !D7
run.rollout_stall_alert  run    alert      on   warn    limit_s=1800                                                                                    default
run.sandbox_containers   run    alert      off  warn    max_run_n=null max_per_sampler_n=null                                                           default
run.release_stall        run    breaker    on   warn    limit_s=null check_s=300 stop_grace_s=1200                                                      default
run.trainer_exit         run    invariant  on   stop    -                                                                                               (cannot disable)
run.controller_exit      run    invariant  on   stop    -                                                                                               (cannot disable)
DEPENDENCIES
  D7   info   island 0 (east-b300): with no group arriving, the trainer's sample wait (SampleBufferRolloutError: the trainer exits) fires first: run.starvation.stop_s 7200 s (2 h), trainer.sample_wait 2400 s (40 min)

ISLAND 1  west-h100  pools: i1-nc-a
NAME                     LAYER  KIND       ON   ACTION  EFFECTIVE                                                                                       SOURCE
run.trainer_silence      run    breaker    on   stop    limit_s=null floor_s=1800 stall_factor_n=4                                                      default
run.first_group          run    breaker    on   stop    timeout_s=5400                                                                                  default
run.starvation           run    breaker    on   stop    alert_s=1800 stop_s=7200 unhealthy_s=600 trainer_wait_max_s=7200 stop_grace_s=1200 check_s=300  default           !D7
run.rollout_stall_alert  run    alert      on   warn    limit_s=1800                                                                                    default
run.sandbox_containers   run    alert      off  warn    max_run_n=null max_per_sampler_n=null                                                           default
run.release_stall        run    breaker    on   warn    limit_s=null check_s=300 stop_grace_s=1200                                                      default
run.trainer_exit         run    invariant  on   stop    -                                                                                               (cannot disable)
run.controller_exit      run    invariant  on   stop    -                                                                                               (cannot disable)
DEPENDENCIES
  D4   warn   island 1 (west-h100): trainer.sample_wait 2400 s (40 min) < 7240 s (2.01 h): the longest group takes about 7240 s (2.01 h) (trial wall clock + infra retries + drain cut); a round that sees no new group that long ends the trainer (SampleBufferRolloutError); raise --rl-sample-wait-s
  D7   info   island 1 (west-h100): with no group arriving, the trainer's sample wait (SampleBufferRolloutError: the trainer exits) fires first: run.starvation.stop_s 7200 s (2 h), trainer.sample_wait 2400 s (40 min)
pool / sampler / harness rows show defaults: 2 pools (i0-nc-a, i1-nc-a), pass --pool NAME
"""


def test_show_groups_a_fleet_by_island(tmp_path, capsys):
    path = _write(tmp_path, _fleet(tmp_path))
    rc = _run(["breaker", "show", "--config", path, "--layer", "run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out == FLEET_SHOW_RUN_LAYER


def test_show_json_of_a_fleet(tmp_path, capsys):
    path = _write(tmp_path, _fleet(tmp_path))
    rc = _run(["breaker", "show", "--json", "--config", path])
    data = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert {e["name"] for e in data["entries"]} >= {"run.budget", "run.wall_clock", "launch.budget", "pool.unreachable"}
    assert not [e for e in data["entries"] if fleet.breaker_scope(e["name"]) == "island"]
    assert [f["rule"] for f in data["findings"]] == ["D11"]
    assert [(i["index"], i["name"], i["pools"]) for i in data["islands"]] == \
        [(0, "east-b300", ["i0-nc-a"]), (1, "west-h100", ["i1-nc-a"])]
    west = data["islands"][1]
    policy = next(e for e in west["entries"] if e["name"] == "trainer.policy_age")
    assert policy["enabled"] and policy["params"]["max_versions"] == 4
    assert all(e["layer"] in ("run", "trainer") and e["name"] not in fleet.FLEET_ENTRIES for e in west["entries"])
    assert {f["scope"]["island"] for f in west["findings"]} == {"1"}


def test_show_of_a_fleet_names_its_pools_by_island(tmp_path, capsys):
    path = _write(tmp_path, _fleet(tmp_path))
    assert _run(["breaker", "show", "--json", "--config", path, "--pool", "i1-nc-a"]) == 0
    assert json.loads(capsys.readouterr().out)["pool"] == "i1-nc-a"
    assert _run(["breaker", "show", "--config", path, "--pool", "nc-a"]) == 2
    assert "i0-nc-a, i1-nc-a" in capsys.readouterr().err


def test_check_reads_each_island(tmp_path, capsys):
    path = _write(tmp_path, _fleet(tmp_path))
    assert _run(["breaker", "check", "--config", path]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith("D11 warn launch.budget 400 USD") for line in lines)
    assert any(line.startswith("D4 warn island 1 (west-h100): trainer.sample_wait") for line in lines)
    assert not any("island 0 (east-b300): trainer.sample_wait 2400 s (40 min) <" in line for line in lines)


# --------------------------------------------------------------------------- one trainer: unchanged


def test_one_trainer_reads_as_before(tmp_path, capsys):
    """No ``islands``: no fleet branch; findings carry no scope, ``show`` prints one table."""
    cfg = {"run": "r", "store": "s3://b/r", "budget": {"usd": 300},
           "trainer": {"argv": [*FLEET_ARGV, "--gpu", "modal:2xh100"], "usd_per_h": 8.0},
           "samplers": [{"name": "nc-a", "config": _pool(tmp_path, "agentic", AGENTIC_SPEC)}]}
    path = _write(tmp_path, cfg)
    rb = resolve(Sources.from_run_config(cfg), strict=False)
    facts = rules_facts.collect(rb, run_config=cfg)
    assert facts.islands == () and facts.trainer is not None
    found = rules.check(rb, run_config=cfg)
    assert found and all(f.scope == {} and "scope" not in f.to_json() for f in found)
    assert not any(f.message.startswith("island ") for f in found)
    assert _run(["breaker", "show", "--json", "--config", path]) == 0
    data = json.loads(capsys.readouterr().out)
    assert "islands" not in data and data["pool"] == "nc-a"
    assert _run(["breaker", "show", "--config", path]) == 0
    out = capsys.readouterr().out
    assert out.startswith("NAME ") and "FLEET" not in out and "ISLAND" not in out


def _syncer_cfg(*argv: str, usd: object = None) -> dict:
    cfg = {"fleet": {"trainer_argv": ["--training-mode", "rl", *argv]}}
    return cfg if usd is None else {**cfg, "syncer": {"usd_per_h": usd}}


@pytest.mark.parametrize("argv, provider", [
    ((), "aws"), (("--controller", "head"), "aws"), (("--controller=head",), "aws"),
    (("--controller", "head", "--syncer-region", "modal/us-east"), "modal"),
])
def test_a_syncer_on_the_controller_head_vm_bills_that_vm(argv, provider):
    """--controller head (the launch default) runs the syncer on the controller head VM: one
    on-demand VM on the --syncer-region cloud, in no other fleet spend line, priced by its row."""
    from yeto.cost import prices

    cfg = _syncer_cfg(*argv)
    placement = fleet.run_syncer_placement(cfg)
    assert placement == fleet.SyncerPlacement("head", provider)
    rate, source = fleet.syncer_rate(None, placement)
    assert rate > 0 and rate == fleet.syncer_price(cfg) == prices.syncer_usd_h(provider)
    assert source.startswith(f"the {fleet.HEAD_VM} on {provider} (it hosts the syncer and the fleet controller): "
                             f"price table syncer@{provider}")
    # the compiled trainer.argv wins over fleet.trainer_argv
    compiled = {**_syncer_cfg("--controller", "head"), "trainer": {"argv": ["--controller", "local"]}}
    assert fleet.run_syncer_placement(compiled) == fleet.SyncerPlacement("vm", "aws")


@pytest.mark.parametrize("argv, provider, usd", [
    (("--controller", "local"), "aws", 0.384),  # the default --syncer-region: an AWS region
    (("--controller", "local", "--syncer-region", "us-east-1"), "aws", 0.384),
    (("--controller", "local", "--syncer-region", "modal/us-east"), "modal", 2.4),  # a Modal CPU syncer
    (("--controller", "local", "--syncer-region=Modal/us-east"), "modal", 2.4),
])
def test_a_syncer_on_its_own_vm_reads_the_price_row_of_that_provider(argv, provider, usd):
    cfg = _syncer_cfg(*argv)
    placement = fleet.run_syncer_placement(cfg)
    assert placement == fleet.SyncerPlacement("vm", provider)
    rate, source = fleet.syncer_rate(None, placement)
    assert rate == fleet.syncer_price(cfg) == usd
    assert source.startswith(f"its own syncer VM on {provider}: price table syncer@{provider}")


def test_a_syncer_vm_without_a_price_row_or_with_a_set_price():
    cfg = _syncer_cfg("--controller", "local", "--syncer-region", "nebius/eu-north1")
    rate, source = fleet.syncer_rate(None, fleet.run_syncer_placement(cfg))
    assert rate is None and source == "its own syncer VM on nebius: no price table row for it"
    assert fleet.syncer_price(cfg) == 0.0  # unpriced counts $0 (the dry-run says so)
    # syncer.usd_per_h set wins wherever it runs; 0 bills nothing
    assert fleet.syncer_price(_syncer_cfg("--controller", "local", usd=1.5)) == 1.5
    assert fleet.syncer_price(_syncer_cfg("--controller", "local", usd=0)) == 0.0
    assert fleet.syncer_rate(None, None) == (None, "the syncer's placement is unknown")
    assert fleet.syncer_placement(["--rl-single-island-no-sync"]) == fleet.SyncerPlacement("none")
    assert fleet.syncer_rate(None, fleet.SyncerPlacement("none"))[0] == 0.0
    assert fleet.syncer_placement(["--syncer-memory", "96"]).memory_gib == 96.0


def test_the_syncer_placement_defaults_are_the_launch_parser_defaults():
    from yeto.cli import build_parser
    from yeto.launch import syncer

    args = build_parser().parse_args(["launch", "--training-mode", "rl", "--model", "m", "--data", "d.jsonl"])
    assert args.controller == fleet.DEFAULT_CONTROLLER
    assert syncer.syncer_cloud(args) == fleet.DEFAULT_SYNCER_CLOUD
    assert args.syncer_memory == fleet.DEFAULT_SYNCER_MEMORY_GIB
    assert fleet.syncer_placement([]).memory_gib == fleet.DEFAULT_SYNCER_MEMORY_GIB


def test_the_dry_run_says_when_the_syncer_memory_outgrows_the_price_row():
    import types

    from yeto.rl import up_fleet

    base = ["--training-mode", "rl", "--model", "m", "--data", "d.jsonl"]
    plan = types.SimpleNamespace(islands=())
    lines = up_fleet.syncer_lines(plan, base, None)
    assert any(line.startswith("fleet: syncer.usd_per_h is unset: the fleet spend counts the syncer at $0.384/h, "
                               "the controller head VM on aws") for line in lines)
    assert not any("--syncer-memory" in line for line in lines)
    lines = up_fleet.syncer_lines(plan, [*base, "--syncer-memory", "96"], None)
    assert any(line.startswith("WARNING: fleet: --syncer-memory 96 GiB orders a larger machine") for line in lines)
    assert not any("--syncer-memory" in line for line in up_fleet.syncer_lines(plan, [*base, "--syncer-memory", "96"],
                                                                                {"usd_per_h": 1.0}))
