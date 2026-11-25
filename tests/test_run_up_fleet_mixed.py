"""An island fleet whose islands mix Miles sampler islands (``"kind": "miles"`` in an
island's ``samplers``) with external pools (:mod:`yeto.rl.fleet`): the compiled
launch argv (the trainers first, then the Miles samplers, each bound to its own
island's trainer), the trainers' buffer shards (the union of both kinds), the
plan's ``source`` field, the island prices and the launcher's own dry-run plan."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest

from test_rl_algorithm_provenance import _fake_sky, _no_modal_listing  # noqa: F401
from test_run_up_fleet import STORE, TRAINER_ARGV, _cfg, _main_args, _modal_vllm, _write, files, no_subprocess  # noqa: F401
from yeto.breakers import fleet as breakers_fleet
from yeto.rl import fleet, up
from yeto.rl.adapters.miles import island_entry
from yeto.rl.engine import sample_roles

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "fleet" / "run.mixed-miles.json"
MILES = {"kind": "miles", "name": "miles", "gpu": "modal:1x1xl40s", "usd_per_h": 1.5}


@pytest.fixture(autouse=True)
def _island_env(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "fleet-mixed-test-key")
    monkeypatch.setenv("YETO_RL_STORE_S3_ACCESS_KEY_ID", "test-key-id")
    monkeypatch.setenv("YETO_RL_STORE_S3_SECRET_ACCESS_KEY", "test-secret")


LEDGER = f"{STORE}/fleet/dispense"
STORE_DISPENSE = '{"backend":"store","namespace":"task-dispense"}'


def _prompts(tmp_path):
    """The pools' prompt file, which a Miles sampler on their ledger reads as --data."""
    path = tmp_path / "prompts.jsonl"
    path.write_text("".join(json.dumps({"q": f"question {n}", "a": str(n)}) + "\n" for n in range(6)))
    return str(path)


def _with_data(cfg, tmp_path):
    """fleet.trainer_argv --data: the pools' prompt file (a Miles sampler shares their ledger)."""
    argv = cfg["fleet"]["trainer_argv"]
    argv[argv.index("--data") + 1] = _prompts(tmp_path)
    return cfg


def _argv(tmp_path):
    argv = list(TRAINER_ARGV)
    argv[argv.index("--data") + 1] = str(tmp_path / "prompts.jsonl")
    return argv


def _mixed(tmp_path):
    """Island 0 (east): vLLM + SGLang pools; island 1 (west): a Miles sampler and an SGLang pool."""
    cfg = _with_data(_cfg(tmp_path), tmp_path)
    cfg["islands"][1]["samplers"].insert(0, dict(MILES))
    return cfg


def _launchable(cfg):
    """The launch also needs a reward and a policy-age limit (island 1 overrides its own)."""
    cfg["fleet"]["trainer_argv"] += ["--rl-max-policy-age", "2", "--reward-function", "yeto.rl.gsm8k_reward:score",
                                     "--trust-remote-code", "--rl-parameter-layout-sha256", "e" * 64]
    return cfg


# -- compile ------------------------------------------------------------------------------

def test_the_compiled_argv_binds_the_miles_sampler_to_its_islands_trainer(files):
    cfg = up.load_config(_write(files, _mixed(files)))
    assert cfg["trainer"] == {"usd_per_h": 17.5, "argv": [
        *_argv(files), "--gpu", "modal:1x2xh100,modal:1x2xh100@us-east,modal:1x1xl40s",
        "--rl-sample-role", "trainer", "--rl-sampler-follow", "trainer",
        "--rl-sync-island-lease-s", "150",  # no fixed roster (default): the async rounds keep quorum 1
        "--rl-island-buffer-shards", "0:vl-i0-0,vl-i0-1,vl-i0-2,sg-i0-0,sg-i0-1",
        "--rl-island-buffer-shards", "1:sg-i1-0,sg-i1-1",
        "--rl-island-sample-role", "2:sampler", "--rl-island-sampler-follows", "2:1",
        # the Miles sampler claims from the fleet's ledger, as the pools do (their settings)
        "--rl-island-dispense-store", f"2:{LEDGER}", "--rl-store-dispense", STORE_DISPENSE,
        "--rl-island-release-every", "0:2",
        "--rl-island-release-every", "1:1", "--rl-island-max-policy-age", "1:3"]}
    # the Miles sampler is no pool: run up drives the three external pools only
    assert [s["name"] for s in cfg["samplers"]] == ["i0-vllm", "i0-sglang", "i1-sglang"]
    assert [r.name for r in up.Run(cfg).pools] == ["i0-vllm", "i0-sglang", "i1-sglang"]


def test_the_plan_marks_each_samplers_source(files):
    plan = fleet.plan_of(up.load_config(_write(files, _mixed(files))))
    east, west = plan.islands
    assert (east.learner, west.learner) == (0, 1) and west.substore == "trainer1"
    assert west.buffer_shards() == ["island2", "sg-i1-0", "sg-i1-1"]  # Miles sampler shard, then the pool's
    assert east.buffer_shards() == east.pool_shards()
    doc = plan.to_json()
    w = doc["islands"][1]
    assert w["miles_samplers"] == [{"name": "i1-miles", "source_name": "miles", "source": "miles",
                                    "gpu": "modal:1x1xl40s", "usd_per_h": 1.5, "launch_island": 2,
                                    "shard": "island2", "engine": "sglang", "dispense_store": LEDGER,
                                    "dispense_namespace": "task-dispense"}]
    assert [p["source"] for i in doc["islands"] for p in i["pools"]] == ["external"] * 3
    assert doc["islands"][0]["miles_samplers"] == []
    # the Miles sampler's price counts to its island; the trainer's own price stays readable
    assert (w["usd_per_h"], w["trainer_usd_per_h"]) == (9.5, 8.0)
    assert (doc["islands"][0]["usd_per_h"], doc["islands"][0]["trainer_usd_per_h"]) == (8.0, 8.0)
    assert fleet.FleetPlan.from_json(doc).to_json() == doc
    lines = plan.lines()
    assert any("pools i1-sglang, Miles samplers i1-miles" in line for line in lines)
    assert ("fleet:   Miles sampler i1-miles (from miles): launch island 2, modal:1x1xl40s at $1.5/h, sglang, "
            f"shard island2, follows learner 1, rows from the task ledger {LEDGER}/task-dispense") in lines


def test_a_plan_written_before_miles_samplers_still_reads():
    old = {"version": 1, "run": "r", "store": STORE, "dispense": "fleet", "islands": [
        {"index": 0, "name": "east", "learner": 0, "substore": "trainer0", "store": f"{STORE}/trainer0",
         "gpu": "modal:1x2xh100", "usd_per_h": 8.0, "release_every": None, "max_policy_age": None,
         "buffer_shards": ["a-i0-0"],
         "pools": [{"name": "i0-a", "source_name": "a", "provider": "modal",
                    "shard_base": "a-i0", "shard_limit": 1, "run_prefix": "r/trainer0"}]}]}
    island = fleet.FleetPlan.from_json(old).islands[0]
    assert island.miles_samplers == () and island.trainer_usd_per_h == 8.0
    assert island.buffer_shards() == ["a-i0-0"]


def test_a_pure_pool_fleet_compiles_as_before(files):
    """No Miles sampler: the argv and the island price are those of a pure-pool fleet."""
    cfg = up.load_config(_write(files, _cfg(files)))
    assert "--rl-island-sample-role" not in cfg["trainer"]["argv"]
    assert "--rl-island-sampler-follows" not in cfg["trainer"]["argv"]
    assert cfg["trainer"]["usd_per_h"] == 16.0
    assert all(i.miles_samplers == () and i.usd_per_h == i.trainer_usd_per_h for i in fleet.plan_of(cfg).islands)


def test_an_island_with_miles_samplers_only_names_no_pool_shard(files):
    cfg = _with_data(_cfg(files), files)
    cfg["islands"][1]["samplers"] = [dict(MILES), dict(MILES, name="miles-b", usd_per_h=None)]
    compiled = up.load_config(_write(files, cfg))
    argv = compiled["trainer"]["argv"]
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "--rl-island-buffer-shards"] == [
        "0:vl-i0-0,vl-i0-1,vl-i0-2,sg-i0-0,sg-i0-1"]
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "--rl-island-sampler-follows"] == ["2:1", "3:1"]
    # both claim from the fleet's ledger, with island 0's pools
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "--rl-island-dispense-store"] == [
        f"2:{LEDGER}", f"3:{LEDGER}"]
    west = fleet.plan_of(compiled).islands[1]
    assert west.buffer_shards() == ["island2", "island3"] and west.usd_per_h == 9.5


# -- the task ledger -------------------------------------------------------------------------

def _flag_values(argv, flag):
    return [argv[i + 1] for i, a in enumerate(argv) if a == flag]


def test_dispense_island_points_each_miles_sampler_at_its_islands_ledger(files):
    cfg = _mixed(files)
    cfg["fleet"]["dispense"] = "island"
    compiled = up.load_config(_write(files, cfg))
    argv = compiled["trainer"]["argv"]
    assert _flag_values(argv, "--rl-island-dispense-store") == [f"2:{STORE}/trainer1"]
    assert _flag_values(argv, "--rl-store-dispense") == [STORE_DISPENSE]
    (miles,) = fleet.plan_of(compiled).islands[1].miles_samplers
    assert (miles.dispense_store, miles.dispense_namespace) == (f"{STORE}/trainer1", "task-dispense")


def test_pools_without_a_store_ledger_leave_the_miles_sampler_on_its_slot(files):
    """No pool claims from a store ledger: there is no ledger to share, the Miles sampler
    keeps its data slot of --data (and --data may be any dataset)."""
    spec = json.loads((files / "vllm.json").read_text())
    spec["prompts"].pop("dispense")
    for name in ("vllm.json", "sglang.json"):
        (files / name).write_text(json.dumps(dict(json.loads((files / name).read_text()), prompts=spec["prompts"])))
    cfg = _cfg(files)
    cfg["islands"][1]["samplers"].insert(0, dict(MILES))
    compiled = up.load_config(_write(files, cfg))
    assert "--rl-island-dispense-store" not in compiled["trainer"]["argv"]
    assert "--rl-store-dispense" not in compiled["trainer"]["argv"]
    assert fleet.plan_of(compiled).islands[1].miles_samplers[0].dispense_store is None


def test_a_miles_sampler_on_a_ledger_needs_the_pools_prompt_file_as_data(files):
    cfg = _cfg(files)  # --data openai/gsm8k: a Hub dataset, not the pools' prompt file
    cfg["islands"][1]["samplers"].insert(0, dict(MILES))
    with pytest.raises(ValueError, match="--data 'openai/gsm8k' is not a local prompt file"):
        up.load_config(_write(files, cfg))


def test_a_pool_prompts_file_other_than_data_is_refused(files):
    cfg = _mixed(files)
    other = files / "other.jsonl"
    other.write_text('{"q": "x", "a": "1"}\n')
    cfg["islands"][1]["samplers"][1]["prompts_file"] = str(other)
    with pytest.raises(ValueError, match="prompts_file .* is not the file fleet.trainer_argv --data"):
        up.load_config(_write(files, cfg))
    cfg["islands"][1]["samplers"][1]["prompts_file"] = str(files / "prompts.jsonl")  # the same file: fine
    assert "--rl-store-dispense" in up.load_config(_write(files, cfg))["trainer"]["argv"]


def test_pools_with_different_ledger_settings_are_refused(files):
    cfg = _mixed(files)
    cfg["islands"][1]["samplers"][1]["sampler"] = {"dispense": {"backend": "store", "namespace": "task-dispense",
                                                                "passes": 2}}
    with pytest.raises(ValueError, match="different settings"):
        up.load_config(_write(files, cfg))


def test_a_hold_out_file_of_the_pools_is_refused(files):
    cfg = _mixed(files)
    for island in cfg["islands"]:
        for entry in island["samplers"]:
            if "kind" not in entry:
                entry["sampler"] = {"dispense": {"backend": "store", "namespace": "task-dispense",
                                                 "holdout": "/pool/node/holdout.txt"}}
    with pytest.raises(ValueError, match="holdout_ids"):
        up.load_config(_write(files, cfg))


def test_the_breaker_budget_counts_the_miles_sampler_to_its_island(files):
    cfg = _mixed(files)
    assert breakers_fleet.island_prices(cfg) == (8.0, 9.5)
    assert [p["name"] for p in breakers_fleet.island_run_config(cfg, 1)["samplers"]] == ["i1-sglang"]
    assert breakers_fleet.fleet_run_config(cfg)["trainer"]["usd_per_h"] == 17.5


@pytest.mark.parametrize("edit, match", [
    (lambda s: s.update(gpu="modal:1xl40s,modal:1xl40s"), r"samplers\[0\].gpu must be one --gpu cluster"),
    (lambda s: s.pop("gpu"), r"samplers\[0\].gpu must be one --gpu cluster"),
    (lambda s: s.update(usd_per_h=-2), r"samplers\[0\].usd_per_h must be a number >= 0"),
    (lambda s: s.update(name="a b"), "a Miles sampler needs a name"),
    (lambda s: s.pop("name"), "a Miles sampler needs a name"),
    (lambda s: s.update(argv=["--rl-sampler-max-backlog-rounds", "4"]), r"unknown .*samplers\[0\] keys \['argv'\]"),
    (lambda s: s.update(name="sglang"), "pool names repeat"),
])
def test_a_bad_miles_entry_is_refused_at_load(files, edit, match):
    cfg = _mixed(files)
    edit(cfg["islands"][1]["samplers"][0])
    with pytest.raises(ValueError, match=match):
        up.load_config(_write(files, cfg))


def test_a_fleet_without_any_external_pool_is_refused(files):
    cfg = _cfg(files)
    for island in cfg["islands"]:
        island["samplers"] = [dict(MILES)]
    with pytest.raises(ValueError, match="at least one external pool"):
        up.load_config(_write(files, cfg))


def test_a_mixed_fleet_dry_run_prints_the_miles_sampler(files, no_subprocess, capsys):
    assert up.main(_main_args(_write(files, _mixed(files)))) == 0
    out = capsys.readouterr().out
    assert "Miles sampler i1-miles (from miles): launch island 2" in out
    launches = [a for a in no_subprocess if "launch" in a]
    assert launches and all(a[a.index("--gpu") + 1].endswith(",modal:1x1xl40s") for a in launches)


# -- the launcher's dry-run plan -------------------------------------------------------------

def _launch_plan(cfg):
    """The launcher dry-run plan of the compiled trainer argv and each island's parsed
    island args (a trainer's ``$LEARNER_ID`` is its learner id)."""
    from yeto.cli import parse_args
    from yeto.launcher import _prepare_rl_args, dry_run_plan

    args = parse_args(cfg["trainer"]["argv"])
    args.model_revision, args.data_revision = "a" * 40, "b" * 40
    args.source_sha256, args.reward_sha256 = "c" * 64, "d" * 64
    args.controller = "local"
    _prepare_rl_args(args)
    launch = dry_run_plan(args)
    trainers = [i["learner_id"] for i in launch["island_requests"] if i.get("sample_role") == "trainer"]
    parsed = []
    for request in launch["island_requests"]:
        learner = trainers.index(request["learner_id"]) if request["learner_id"] in trainers else 0
        tail = request["learner_command"].split("-m yeto.rl.adapters.miles.island_entry", 1)[1]
        tokens = [{"$LEARNER_ID": str(learner), "$SYNCER_ADDR": "10.0.0.1:29400"}.get(t, t) for t in shlex.split(tail)]
        parsed.append((request, island_entry.parse_args(tokens)))
    return args, launch, parsed


def _check_launch(cfg):
    """Every fleet island's trainer reads the union of its Miles samplers' and its pools'
    shards in trainer<k>/; every Miles sampler writes its own shard there and stops after
    its trainer."""
    plan = fleet.plan_of(cfg)
    args, launch, parsed = _launch_plan(cfg)
    assert sample_roles.is_mixed_group(args)
    assert f"--learners {len(plan.islands)} " in launch["role_syncers"][0]["command"]
    for island in plan.islands:
        request, trainer = parsed[island.learner]
        assert request["sample_role"] == "trainer" and request["sample_substore"] == island.substore
        assert request["buffer_shards"] == island.buffer_shards()
        assert str(trainer.rl_sample_store).endswith(f"/{island.substore}") and trainer.rl_sample_lanes == 1
        shards = trainer.rl_buffer_shards
        assert (shards.split(",") if isinstance(shards, str) else list(shards)) == island.buffer_shards()
        assert int(trainer.rl_release_every or 1) == (island.release_every or 1)
        for miles in island.miles_samplers:
            request, sampler = parsed[miles.launch_island]
            assert request["sample_role"] == "sampler" and request["feeds_trainer"] == island.learner
            assert request["sample_substore"] == island.substore and request["sampler_until_trainers_end"] == 1
            assert str(sampler.rl_resume_store).endswith(f"/{island.substore}")
            assert sampler.rl_sampler_node == miles.shard
            if miles.dispense_store:  # its rows from the pools' ledger, not a data slot
                assert request["dispense_store"] == miles.dispense_store
                # by its URI: the island's store hook reaches no path outside trainer<k>
                assert sampler.rl_dispense_store == miles.dispense_store
                assert json.loads(sampler.rl_store_dispense) == json.loads(STORE_DISPENSE)
                assert not sampler.rl_sampler_data_slot and not sampler.rl_data_dispense
    return plan, parsed


def test_the_launcher_plan_of_a_mixed_fleet(files):
    _check_launch(up.load_config(_write(files, _launchable(_mixed(files)))))


def test_one_island_of_a_trainer_a_miles_sampler_and_a_pool(files):
    """The fleet island of the change: trainer + one Miles sampler + one external pool."""
    cfg = _with_data(_cfg(files), files)
    cfg["islands"] = [{"name": "solo", "trainer": {"gpu": "modal:1x2xh100", "usd_per_h": 8.0},
                       "samplers": [dict(MILES), _modal_vllm(files)]}]
    plan, parsed = _check_launch(up.load_config(_write(files, _launchable(cfg))))
    (island,) = plan.islands
    assert island.buffer_shards() == ["island1", "vl-i0-0", "vl-i0-1", "vl-i0-2"]
    (_req_t, trainer), (_req_s, sampler) = parsed
    assert str(trainer.rl_sample_store).endswith("/trainer0") and str(sampler.rl_resume_store).endswith("/trainer0")


# -- the documented example ----------------------------------------------------------------

def _example(tmp_path):
    text = EXAMPLE.read_text()
    for placeholder, value in (("<bucket>", "b"), ("<run>", "r"), ("<store.env>", str(tmp_path / "store.env")),
                               ("<sampler-spec.vllm.json>", str(tmp_path / "vllm.json")),
                               ("<sampler-spec.sglang.json>", str(tmp_path / "sglang.json")),
                               ("<sampler-image>", "registry/sampler@sha256:" + "0" * 64),
                               ("<modal-secret>", "store-secret"), ("<volume>", "weights"),
                               ("<prompts.jsonl>", _prompts(tmp_path))):
        text = text.replace(placeholder, value)
    assert "<" not in text, "an unfilled placeholder in the example"
    (tmp_path / "store.env").write_text("R2_BUCKET=b\n")
    for name, adapter in (("vllm", "vllm-server"), ("sglang", "sglang-server")):
        (tmp_path / f"{name}.json").write_text(json.dumps({
            "sampler_id": "x", "engine": {"adapter": adapter}, "sampling": {"n": 8},
            "store": "/tmp/stage/b/runs/__RUN_PREFIX__",
            "prompts": {"path": "/work/prompts.jsonl", "input_key": "q", "label_key": "a",
                        "groups_per_round": 1}}))
    path = tmp_path / "run.json"
    path.write_text(text)
    return str(path)


def test_the_mixed_example_loads_compiles_and_dry_runs(tmp_path, monkeypatch, no_subprocess, capsys):
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    path = _example(tmp_path)
    assert up.main(_main_args(path)) == 0
    out = capsys.readouterr().out
    assert "Miles sampler i1-miles-l40s" in out and "dry run: preflight ok" in out
    run = up.open_run(path)
    assert run.fleet_problem() is None
    plan, _parsed = _check_launch(run.cfg)
    mixed = plan.islands[1]
    assert mixed.name == "mixed" and [m.launch_island for m in mixed.miles_samplers] == [2]
    assert mixed.buffer_shards() == ["island2"] + [f"r-sglang-i1-{n}" for n in range(4)]
    assert [p.engine for p in mixed.pools] == ["sglang"] and plan.islands[0].miles_samplers == ()


def test_the_island_scope_of_the_mixed_example_has_no_miles_pool():
    cfg = json.loads(EXAMPLE.read_text())
    assert breakers_fleet.island_prices(cfg) == (8.0, 10.0)
    assert [p["name"] for p in breakers_fleet.island_run_config(cfg, 1)["samplers"]] == ["i1-sglang-l40s"]
