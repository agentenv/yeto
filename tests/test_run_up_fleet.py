"""``yeto run up`` with an island fleet: the ``islands`` / ``fleet`` sections, their
compile into one synced-group trainer launch and the islands' pools
(:mod:`yeto.rl.fleet`), the training-form check, the dry-run lines and
``<store>/fleet/plan.json``."""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from yeto import launch_preflight
from yeto.launch.nc import cli as nc_cli
from yeto.launch.nc import tasks
from yeto.launch.pool.island_pool import Member
from yeto.rl import fleet, up, up_consistency
from yeto.rl.engine.object_store import DirectoryObjectStore

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "fleet" / "run.two-islands.json"
STORE = "s3://b/runs/r"
TRAINER_ARGV = ["--training-mode", "rl", "--model", "Qwen/Qwen3-4B", "--data", "openai/gsm8k", "--tuning", "full",
                "--rl-sync-preset", "decoupled", "--local-rl-rounds-per-sync", "4", "--fragments", "4",
                "--n-samples-per-prompt", "8",
                "--rl-algorithm-spec", str(ROOT / "examples" / "k31" / "algorithm-spec.async-age12-tis.json"),
                "--rl-checkpoint-store", STORE]


@pytest.fixture
def files(tmp_path, monkeypatch):
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    spec = {"sampler_id": "x", "store": "/work/stage/b/runs/__RUN_PREFIX__", "engine": {"adapter": "vllm-server"},
            "event_tape": "/work/sampler/events-__RUN_PREFIX__.jsonl", "sampling": {"n": 8},
            "prompts": {"path": "p.jsonl", "input_key": "q", "groups_per_round": 1,
                        "dispense": {"backend": "store", "namespace": "task-dispense"}}}
    (tmp_path / "vllm.json").write_text(json.dumps(spec))
    sglang = dict(spec, engine={"adapter": "sglang-server"}, rollout_quant={"tier": 1, "apply": "x:y"})
    (tmp_path / "sglang.json").write_text(json.dumps(sglang))
    (tmp_path / "store.env").write_text("R2_BUCKET=b\n")
    (tmp_path / "store.env").chmod(0o600)
    # a pool config file reused by both islands (the compile renames its pools and shards per island)
    (tmp_path / "pool.sglang.json").write_text(json.dumps({
        "provider": {"kind": "modal", "image": "i", "secret": "s", "bucket": "b", "gpu": "L40S"},
        "spec": str(tmp_path / "sglang.json"), "run_prefix": "r", "r2_env_file": str(tmp_path / "store.env"),
        "shard_base": "sg", "shard_limit": 2, "target": 1, "slots": [1, 2], "run": "r"}))
    return tmp_path


def _modal_vllm(tmp_path, **extra):
    return {"name": "vllm", "provider": {"kind": "modal", "image": "i", "secret": "s", "bucket": "b", "gpu": "H100"},
            "spec": str(tmp_path / "vllm.json"), "run_prefix": "r", "r2_env_file": str(tmp_path / "store.env"),
            "shard_base": "vl", "shard_limit": 3, "target": 1, "slots": [1, 2, 3], "run": "r",
            "usd_per_h": 4.0, **extra}


def _cfg(tmp_path, **extra):
    cfg = {"run": "r", "store": STORE, "budget": {"usd": 40, "wall_h": 4, "nc_usd_per_h": 0},
           "fleet": {"trainer_argv": list(TRAINER_ARGV)},
           "islands": [
               {"name": "east", "trainer": {"gpu": "modal:1x2xh100", "usd_per_h": 8.0,
                                            "argv": ["--rl-release-every", "2"]},
                "samplers": [_modal_vllm(tmp_path), {"name": "sglang", "config": str(tmp_path / "pool.sglang.json"),
                                                     "usd_per_h": 2.0}]},
               {"name": "west", "trainer": {"gpu": "modal:1x2xh100@us-east", "usd_per_h": 8.0,
                                            "argv": ["--rl-max-policy-age=3", "--rl-release-every", "1"]},
                "samplers": [{"name": "sglang", "config": str(tmp_path / "pool.sglang.json"), "usd_per_h": 2.0}]}]}
    cfg.update(extra)
    return cfg


def _write(tmp_path, cfg) -> str:
    path = tmp_path / "run.json"
    path.write_text(json.dumps(cfg))
    return str(path)


def _main_args(path, **kw):
    base = dict(config=path, rl_command="up", skip_preflight=False, dry_run=True, no_monitor=True,
                trainer_now=True, allow_after_forced_stop=False)
    return types.SimpleNamespace(**{**base, **kw})


@pytest.fixture
def no_subprocess(monkeypatch):
    """The preflight's subprocesses (pool / launch dry runs) pass without running."""
    calls = []
    monkeypatch.setattr(up.Run, "_cmd", lambda self, argv, **kw: calls.append(list(argv)) or 0)
    monkeypatch.setattr(launch_preflight, "modal_sdk_problem", lambda env: None)  # no Modal client here
    return calls


# -- the sections (spec: fleet config section) -------------------------------------------

def test_a_two_island_fleet_dry_run_prints_each_island(files, no_subprocess, capsys):
    assert up.main(_main_args(_write(files, _cfg(files)))) == 0
    out = capsys.readouterr().out
    assert "fleet: 2 island(s)" in out and "sync decoupled -> full-parameter (async rounds, full parameters)" in out
    assert ("fleet: island 0 east: learner 0, store trainer0/ (s3://b/runs/r/trainer0), trainer modal:1x2xh100 at "
            "$8/h, release_every 2, 5 buffer shard(s), pools i0-vllm, i0-sglang") in out
    assert ("fleet: island 1 west: learner 1, store trainer1/ (s3://b/runs/r/trainer1), trainer "
            "modal:1x2xh100@us-east at $8/h, release_every 1, max_policy_age 3, 2 buffer shard(s), "
            "pools i1-sglang") in out
    assert "pool i0-sglang (from sglang): modal, sglang, rollout_quant 1, shards sg-i0-0..1" in out
    assert "dry run: preflight ok, nothing ordered; the fleet plan is not written" in out
    launches = [a for a in no_subprocess if "launch" in a]
    assert launches and all(a[a.index("--gpu") + 1] == "modal:1x2xh100,modal:1x2xh100@us-east" for a in launches)


def test_islands_and_trainer_are_mutually_exclusive(files):
    for key, value in (("trainer", {"argv": []}), ("samplers", [{}]), ("pool", {"config": "p.json"})):
        with pytest.raises(ValueError, match=f"'islands' and '{key}' are mutually exclusive"):
            up.load_config(_write(files, _cfg(files, **{key: value})))
    with pytest.raises(ValueError, match="'fleet'.*needs an 'islands' section"):
        up.load_config(_write(files, {"run": "r", "store": STORE, "trainer": {"argv": []},
                                      "pool": {"config": "p.json"}, "fleet": {"trainer_argv": []}}))


def test_an_island_without_a_trainer_is_named(files):
    cfg = _cfg(files)
    del cfg["islands"][1]["trainer"]
    with pytest.raises(ValueError, match=r"islands\[1\] \(west\): missing 'trainer'"):
        up.load_config(_write(files, cfg))


@pytest.mark.parametrize("edit, match", [
    (lambda c: c["fleet"].update(dispense="pool"), r"fleet.dispense must be one of \['fleet', 'island'\]"),
    (lambda c: c["fleet"].update(trainer_argvs=[]), "unknown fleet keys"),
    (lambda c: c.pop("fleet"), "needs a 'fleet' object"),
    (lambda c: c["fleet"]["trainer_argv"].extend(["--gpu", "modal:8xh100"]), "--gpu is written by the fleet"),
    (lambda c: c["fleet"]["trainer_argv"].append("--rl-buffer-shards=a-0"), "--rl-buffer-shards is written"),
    (lambda c: c["fleet"]["trainer_argv"].extend(["--rl-island-release-every", "0:2"]), "--rl-island-release-every"),
    (lambda c: c["fleet"]["trainer_argv"].append("--rl-single-island-no-sync"), "single buffer trainer"),
    (lambda c: c["fleet"].update(trainer_argv=TRAINER_ARGV[:-2]), "needs --rl-checkpoint-store"),
    (lambda c: c["fleet"]["trainer_argv"].extend(["--rl-checkpoint-store", "s3://b/runs/other"]),
     "!= the run config's store"),
    (lambda c: c.update(islands=[]), "non-empty list"),
    (lambda c: c["islands"][1].update(name="east"), "island names repeat"),
    (lambda c: c["islands"][0].update(name="east side"), "name must be a word"),
    (lambda c: c["islands"][0].update(pools=[]), r"unknown islands\[0\] \(east\) keys \['pools'\]"),
    (lambda c: c["islands"][0]["trainer"].update(argv=["--inner-lr", "1e-5"]), "not an island trainer flag"),
    (lambda c: c["islands"][0]["trainer"].update(argv=["--rl-release-every", "0"]), "whole number >= 1"),
    (lambda c: c["islands"][0]["trainer"].update(gpu="modal:8xh100,modal:8xh100"), "one --gpu cluster"),
    (lambda c: c["islands"][0]["trainer"].update(usd_per_h=-1), "usd_per_h must be a number >= 0"),
    (lambda c: c["islands"][0].update(samplers=[]), r"islands\[0\] \(east\): samplers must be a non-empty list"),
    (lambda c: c["islands"][0]["samplers"][0].pop("shard_limit"), "needs shard_limit"),
    (lambda c: c["islands"][0]["samplers"][0].pop("run_prefix"), "needs run_prefix"),
    (lambda c: c["islands"][0]["samplers"][0].update(data_path="syncer"), "data_path 'syncer'"),
    (lambda c: c["islands"][0]["samplers"][0].update(sampler={"island": 3}), "sampler.island is written"),
    (lambda c: c["islands"][0]["samplers"][0].update(
        sampler={"dispense": {"backend": "store", "store": "s3://b/x"}}), "sampler.dispense.store"),
    (lambda c: c["islands"][0]["samplers"][1].update(name="vllm"), "pool names repeat"),
])
def test_a_bad_fleet_section_is_refused_at_load(files, edit, match):
    cfg = _cfg(files)
    edit(cfg)
    with pytest.raises(ValueError, match=match):
        up.load_config(_write(files, cfg))


def test_a_single_trainer_config_loads_unchanged():
    """No islands section: the run config comes back as it is (the customer K3.1 plan golden
    pins the launch side of it)."""
    for path in (ROOT / "examples" / "k31" / "run.agentic-262k.json",
                 ROOT / "tests" / "fixtures" / "k31_customer_kit" / "run.json"):
        assert up.load_config(str(path)) == json.loads(path.read_text())


# -- compile ------------------------------------------------------------------------------

def test_the_compiled_trainer_argv(files):
    cfg = up.load_config(_write(files, _cfg(files)))
    assert cfg["trainer"] == {"usd_per_h": 16.0, "argv": [
        *TRAINER_ARGV, "--gpu", "modal:1x2xh100,modal:1x2xh100@us-east",
        "--rl-sample-role", "trainer", "--rl-sampler-follow", "trainer",
        "--rl-sync-island-lease-s", "150", "--rl-sync-quorum", "1",  # no fixed roster (default)
        "--rl-island-buffer-shards", "0:vl-i0-0,vl-i0-1,vl-i0-2,sg-i0-0,sg-i0-1",
        "--rl-island-buffer-shards", "1:sg-i1-0,sg-i1-1",
        "--rl-island-release-every", "0:2",
        "--rl-island-release-every", "1:1", "--rl-island-max-policy-age", "1:3"]}
    assert cfg["islands"] == _cfg(files)["islands"] and cfg["fleet"] == _cfg(files)["fleet"]
    assert fleet.plan_of(cfg) is cfg[fleet.PLAN_KEY]


def _island_waits(cfg):
    """{island: trainer.sample_wait.timeout_s} as the run's preflight resolves each island."""
    from yeto.breakers.fleet import resolve_islands

    return {i.index: r.entry("trainer.sample_wait").params["timeout_s"] for i, r in resolve_islands(cfg)}


def _launch_waits(argv):
    """{learner id: --rl-sample-wait-s} each trainer island of the compiled launch runs with."""
    from yeto.rl.engine import sample_roles

    from yeto import cli

    args = cli.build_parser().parse_args(["launch", *argv])
    shared = getattr(args, "rl_sample_wait_s", None)
    overrides = sample_roles.island_trainer_overrides(args, 2)
    return {k: overrides.get(k, {}).get("rl_sample_wait_s", shared) for k in range(2)}


def test_the_run_configs_sample_wait_reaches_every_trainer_island(files):
    """The preflight resolves breakers.trainer.sample_wait per island; every trainer reads
    --rl-sample-wait-s, so the compile passes the section's value as the flag."""
    raw = _cfg(files, breakers={"trainer": {"sample_wait": {"timeout_s": 1200}}})
    argv = up.load_config(_write(files, raw))["trainer"]["argv"]
    follow = argv.index("--rl-sampler-follow")
    assert argv[follow + 2:follow + 4] == ["--rl-sample-wait-s", "1200"]
    assert "--rl-island-sample-wait-s" not in argv
    assert _island_waits(raw) == {0: 1200, 1: 1200} and _launch_waits(argv) == {0: 1200.0, 1: 1200.0}


def test_an_islands_own_sample_wait_wins_over_the_run_configs(files):
    raw = _cfg(files, breakers={"trainer": {"sample_wait": {"timeout_s": 1200}}})
    raw["islands"][1]["trainer"]["argv"] += ["--rl-sample-wait-s", "900"]
    cfg = up.load_config(_write(files, raw))
    argv = cfg["trainer"]["argv"]
    assert argv.count("--rl-sample-wait-s") == 1 and argv[argv.index("--rl-sample-wait-s") + 1] == "1200"
    assert argv[-2:] == ["--rl-island-sample-wait-s", "1:900"]
    assert _island_waits(raw) == {0: 1200, 1: 900.0} and _launch_waits(argv) == {0: 1200.0, 1: 900.0}
    west = fleet.plan_of(cfg).islands[1]
    assert west.sample_wait_s == 900.0 and fleet.plan_of(cfg).islands[0].sample_wait_s is None
    assert fleet.FleetPlan.from_json(fleet.plan_of(cfg).to_json()).islands[1].sample_wait_s == 900.0
    assert any("west" in line and "sample_wait_s 900" in line for line in fleet.plan_of(cfg).lines())
    # an island's own value also wins over the fleet's flag
    raw2 = _cfg(files)
    raw2["fleet"]["trainer_argv"] += ["--rl-sample-wait-s", "1800"]
    raw2["islands"][0]["trainer"]["argv"] += ["--rl-sample-wait-s=600"]
    argv2 = up.load_config(_write(files, raw2))["trainer"]["argv"]
    assert _island_waits(raw2) == {0: 600.0, 1: 1800.0} and _launch_waits(argv2) == {0: 600.0, 1: 1800.0}


def test_without_a_sample_wait_the_compiled_argv_is_unchanged(files):
    for extra in ({}, {"breakers": {"run": {"starvation": {"stop_s": 900}}}}):
        argv = up.load_config(_write(files, _cfg(files, **extra)))["trainer"]["argv"]
        assert "--rl-sample-wait-s" not in argv and "--rl-island-sample-wait-s" not in argv


@pytest.mark.parametrize("edit, match", [
    (lambda c: c["fleet"]["trainer_argv"].extend(["--rl-sample-wait-s", "900"]),
     r"breakers.trainer.sample_wait.timeout_s = 1200 but fleet.trainer_argv --rl-sample-wait-s = 900"),
    (lambda c: c["islands"][0]["trainer"]["argv"].extend(["--rl-sample-wait-s", "0"]), "seconds > 0"),
    (lambda c: c["islands"][0]["trainer"]["argv"].extend(["--rl-sample-wait-s", "1", "--rl-sample-wait-s", "2"]),
     "repeats"),
])
def test_a_conflicting_or_bad_sample_wait_is_refused(files, edit, match):
    cfg = _cfg(files, breakers={"trainer": {"sample_wait": {"timeout_s": 1200}}})
    edit(cfg)
    with pytest.raises(ValueError, match=match):
        up.load_config(_write(files, cfg))


def test_the_compiled_pools(files):
    cfg = up.load_config(_write(files, _cfg(files)))
    run = up.Run(cfg)
    assert [r.name for r in run.pools] == ["i0-vllm", "i0-sglang", "i1-sglang"]
    assert [r.usd_per_h for r in run.pools] == [4.0, 2.0, 2.0]
    docs = [r.load() for r in run.pools]
    assert [d["run_prefix"] for d in docs] == ["r/trainer0", "r/trainer0", "r/trainer1"]
    assert [d["shard_base"] for d in docs] == ["vl-i0", "sg-i0", "sg-i1"]
    assert [d["sampler"] for d in docs] == [
        {"island": k, "until_trainers_end": 1, "dispense_store": "s3://b/runs/r/fleet/dispense"} for k in (0, 0, 1)]
    assert docs[1]["provider"]["gpu"] == "L40S" and docs[1]["spec"] == str(files / "sglang.json")
    # the samplers' store (s3://<bucket>/runs/<run_prefix>) is the island's sub-directory, which the
    # run's store consistency check accepts for a trainer reading <store>/trainer<k>
    args = types.SimpleNamespace(rl_checkpoint_store=STORE)
    for doc, k in zip(docs, (0, 0, 1)):
        assert up_consistency.pool_store(doc) == f"{STORE}/trainer{k}"
        assert up_consistency.store_problems(STORE, args, doc) == []
    assert up_consistency.store_problems(STORE, types.SimpleNamespace(rl_checkpoint_store=None), docs[0])
    # a node's rendered spec: the island, the trainer-end stop, the fleet ledger, the island's store
    spec = nc_cli.render_spec(docs[2], Member(account=1, ticket="t", shard="sg-i1-0"), ["sg-i1-0", "sg-i1-1"])
    assert (spec["island"], spec["until_trainers_end"], spec["dispense_store"]) == (
        1, 1, "s3://b/runs/r/fleet/dispense")
    assert spec["store"] == "/work/stage/b/runs/r/trainer1"


def test_dispense_island_keeps_each_islands_own_ledger(files):
    cfg = _cfg(files)
    cfg["fleet"]["dispense"] = "island"
    compiled = up.load_config(_write(files, cfg))
    assert all("dispense_store" not in s["sampler"] for s in compiled["samplers"])
    plan = fleet.plan_of(compiled)
    assert plan.dispense == "island" and plan.dispense_store is None
    # run up (status, the data-end settlement) and the pool (retire / drain) read each island's ledger
    run = up.Run(compiled)
    assert [run.dispense_ledger(i)[0] for i in range(3)] == [f"{STORE}/trainer0", f"{STORE}/trainer0",
                                                            f"{STORE}/trainer1"]
    assert all(nc_cli.pool_dispense_store(r.load()) is None for r in run.pools)


def test_dispense_fleet_shares_one_ledger_on_both_sides(files):
    run = up.Run(up.load_config(_write(files, _cfg(files))))
    shared = "s3://b/runs/r/fleet/dispense"
    assert {nc_cli.pool_dispense_store(r.load()) for r in run.pools} == {shared}
    assert run.dispense_ledgers() == [(shared, "task-dispense", ["i0-vllm", "i0-sglang", "i1-sglang"])]


def test_the_sampler_knobs_change_nothing_unset():
    spec = {"prompts": {}, "sampling": {}}
    assert tasks.apply_sampler_overrides(spec, **nc_cli._overrides({})) == spec
    with pytest.raises(ValueError, match="island must be a whole number"):
        nc_cli._overrides({"island": -1})
    with pytest.raises(ValueError, match="dispense_store must be a store URI"):
        nc_cli._overrides({"dispense_store": ""})


# -- training form (spec: fleet training form and backend) ---------------------------------

def _form(files, argv):
    cfg = _cfg(files)
    cfg["fleet"]["trainer_argv"] = argv
    return up.Run(up.load_config(_write(files, cfg)))


def test_a_lora_fleet_is_refused_in_the_preflight(files, no_subprocess):
    lora = [a for a in TRAINER_ARGV if a not in ("--tuning", "full")] + ["--tuning", "lora"]
    run = _form(files, lora)
    assert "LoRA fleet" in run.fleet_problem() and "external" in run.fleet_problem()
    assert not run.preflight() and no_subprocess == []
    run = _form(files, TRAINER_ARGV + ["--expert-full-count", "8"])
    assert "LoRA fleet" in run.fleet_problem()


def test_a_verl_or_strict_fleet_is_refused(files):
    assert "runs on Miles" in _form(files, TRAINER_ARGV + ["--rl-backend", "verl"]).fleet_problem()
    i = TRAINER_ARGV.index("decoupled")
    strict = TRAINER_ARGV[:i] + ["full-parameter", "--rl-sync-full-parameter-rounds", "sweep"] + TRAINER_ARGV[i + 1:]
    assert "asynchronous full-parameter rounds" in _form(files, strict).fleet_problem()


def test_a_partial_experts_fleet_passes_as_full_parameter_async(files, tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text("{}")
    run = _form(files, TRAINER_ARGV + ["--rl-expert-selection", str(selection)])
    assert run.fleet_problem() is None
    sync = run.fleet_plan().sync
    assert (sync["requested"], sync["preset"], sync["rounds"], sync["form"]) == (
        "decoupled", "full-parameter", "async", "partial experts")
    assert "sync decoupled -> full-parameter (async rounds, partial experts)" in run.fleet_plan().lines()[0]


# -- plan.json -----------------------------------------------------------------------------

def test_run_up_writes_the_plan_the_dry_run_printed(files, no_subprocess, monkeypatch, capsys):
    path = _write(files, _cfg(files))
    bucket = DirectoryObjectStore(files / "bucket")
    monkeypatch.setattr(up.Run, "_objects", lambda self: bucket)
    assert up.main(_main_args(path)) == 0
    printed = [line.split(" [rl up] ", 1)[1] for line in capsys.readouterr().out.splitlines()
               if " [rl up] fleet:" in line]
    assert bucket.read(fleet.PLAN_OBJECT) is None  # a dry run writes nothing
    monkeypatch.setattr(up.Run, "sandbox_prepare", lambda self: False)  # stop right after the plan
    assert up.main(_main_args(path, dry_run=False)) == 2
    doc = json.loads(bucket.read(fleet.PLAN_OBJECT))
    plan = fleet.FleetPlan.from_json(doc)
    assert plan.lines() == printed
    assert fleet.FleetPlan.from_json(doc).to_json() == doc
    assert doc["version"] == 1 and doc["run"] == "r" and doc["store"] == STORE and doc["dispense"] == "fleet"
    assert doc["dispense_store"] == "s3://b/runs/r/fleet/dispense"
    east, west = doc["islands"]
    assert {k: east[k] for k in ("index", "name", "learner", "substore", "store", "gpu", "usd_per_h",
                                 "release_every", "max_policy_age")} == {
        "index": 0, "name": "east", "learner": 0, "substore": "trainer0", "store": f"{STORE}/trainer0",
        "gpu": "modal:1x2xh100", "usd_per_h": 8.0, "release_every": 2, "max_policy_age": None}
    assert east["buffer_shards"] == ["vl-i0-0", "vl-i0-1", "vl-i0-2", "sg-i0-0", "sg-i0-1"]
    assert east["pools"][0] == {"name": "i0-vllm", "source_name": "vllm", "source": "external",
                                "provider": "modal", "shard_base": "vl-i0",
                                "shard_limit": 3, "run_prefix": "r/trainer0", "target": 1, "engine": "vllm",
                                "rollout_quant": None, "usd_per_h": 4.0}
    assert east["pools"][1]["engine"] == "sglang" and east["pools"][1]["rollout_quant"] == 1
    assert east["pools"][1]["usd_per_h"] == 2.0 and doc["syncer"] == {"usd_per_h": None}
    assert (west["learner"], west["max_policy_age"], [p["name"] for p in west["pools"]]) == (1, 3, ["i1-sglang"])
    state = json.loads((up.STATE_ROOT / "r" / "state.json").read_text())
    assert state["steps"]["fleet_plan"]["object"] == "s3://b/runs/r/fleet/plan.json"


def test_a_plan_the_store_refuses_orders_nothing(files, no_subprocess, monkeypatch):
    class Refuses:
        def write(self, key, data):
            raise OSError("AccessDenied")

    monkeypatch.setattr(up.Run, "_objects", lambda self: Refuses())
    ordered = []
    monkeypatch.setattr(up.Run, "sandbox_prepare", lambda self: ordered.append("sandbox") or True)
    assert up.main(_main_args(_write(files, _cfg(files)), dry_run=False)) == 2
    assert ordered == []


def test_a_single_trainer_run_writes_no_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    (tmp_path / "pool.json").write_text(json.dumps({"shard_base": "s"}))
    run = up.Run({"run": "r1", "store": STORE, "pool": {"config": str(tmp_path / "pool.json")},
                  "trainer": {"argv": []}})
    monkeypatch.setattr(up.Run, "_objects", lambda self: pytest.fail("no store write"))
    assert run.fleet_plan() is None and run.fleet_problem() is None and run.write_fleet_plan()


# -- the documented example ----------------------------------------------------------------

def test_the_example_fleet_config_loads_compiles_and_passes_the_form_check(tmp_path, monkeypatch, no_subprocess,
                                                                         capsys):
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    text = EXAMPLE.read_text()
    for placeholder, value in (("<bucket>", "b"), ("<run>", "r"), ("<store.env>", str(tmp_path / "store.env")),
                               ("<sampler-spec.vllm.json>", str(tmp_path / "vllm.json")),
                               ("<sampler-spec.sglang.json>", str(tmp_path / "sglang.json")),
                               ("<sampler-image>", "registry/sampler@sha256:" + "0" * 64),
                               ("<modal-secret>", "store-secret"), ("<volume>", "weights")):
        text = text.replace(placeholder, value)
    assert "<" not in text, "an unfilled placeholder in the example"
    (tmp_path / "store.env").write_text("R2_BUCKET=b\n")
    for name, adapter in (("vllm", "vllm-server"), ("sglang", "sglang-server")):
        (tmp_path / f"{name}.json").write_text(json.dumps({
            "sampler_id": "x", "engine": {"adapter": adapter}, "sampling": {"n": 8},
            "store": "/tmp/stage/b/runs/__RUN_PREFIX__",
            "prompts": {"path": "p.jsonl", "input_key": "q", "groups_per_round": 1}}))
    path = tmp_path / "run.json"
    path.write_text(text)
    assert up.main(_main_args(str(path))) == 0
    out = capsys.readouterr().out
    assert "fleet: island 0" in out and "fleet: island 1" in out and "dry run: preflight ok" in out
    run = up.open_run(str(path))
    plan = run.fleet_plan()
    assert run.fleet_problem() is None and len(plan.islands) == 2
    assert [p.engine for p in plan.islands[0].pools] == ["vllm", "sglang"]
    assert {r.load()["run_prefix"] for r in run.pools} == {"r/trainer0", "r/trainer1"}
