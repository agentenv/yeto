"""``yeto run up`` preflight of an island fleet, island by island (:mod:`yeto.rl.up_fleet`):
the pool store and shards of each island, the islands' ``rollout_fields``, the release
form the engines load, a vLLM island's release cadence, the syncer and the budget
estimate."""

from __future__ import annotations

import json
import types

import pytest

from tests.test_run_up_fleet import TRAINER_ARGV, _cfg, _main_args, _write, files, no_subprocess  # noqa: F401
from yeto.rl import fleet, up, up_fleet


def _run(files, cfg=None):
    return up.Run(up.load_config(_write(files, cfg or _cfg(files))))


def _spec(files, name, **edit):
    doc = json.loads((files / f"{name}.json").read_text())
    doc.update(edit)
    (files / f"{name}.json").write_text(json.dumps(doc))


def _lines(capsys):
    return [line.split(" [rl up] ", 1)[1] for line in capsys.readouterr().out.splitlines() if " [rl up] " in line]


# -- rollout_fields (spec: per-island preflight, rollout_fields mismatch) ------------------

def test_a_pool_whose_top_logprobs_differ_from_the_trainer_names_its_island_and_pool(files):
    _spec(files, "sglang", rollout_fields={"top_logprobs_num": 64})
    _spec(files, "vllm", rollout_fields={"top_logprobs_num": 128})
    plan = fleet.plan_of(up.load_config(_write(files, _cfg(files))))
    argv = TRAINER_ARGV + ["--rl-loss", "score_centering", "--rl-rollout-top-logprobs-num", "128"]
    specs = [("i0-vllm", json.loads((files / "vllm.json").read_text())),
             ("i0-sglang", json.loads((files / "sglang.json").read_text())),
             ("i1-sglang", json.loads((files / "sglang.json").read_text()))]
    found = up_fleet.rollout_fields_problems(plan, argv, specs)
    assert any(p.startswith("island 1 (west), pool i1-sglang: rollout_fields.top_logprobs_num 64 != the trainer's "
                            "--rl-rollout-top-logprobs-num 128") for p in found)
    assert any(p.startswith("island 0 (east), pool i0-sglang:") for p in found)
    assert not any("pool i0-vllm" in p for p in found)


def test_the_run_preflight_refuses_a_score_centering_pool_and_names_island_and_pool(files, no_subprocess, capsys):
    _spec(files, "sglang", rollout_fields={"top_logprobs_num": 64})
    cfg = _cfg(files)
    cfg["islands"][0]["samplers"] = cfg["islands"][0]["samplers"][:1]  # island 0: the vLLM pool only
    _spec(files, "vllm", rollout_fields={"top_logprobs_num": 128})
    cfg["fleet"]["trainer_argv"] += ["--rl-loss", "score_centering"]
    run = _run(files, cfg)
    assert not run.preflight()
    refused = [line for line in _lines(capsys) if "top_logprobs_num" in line]
    assert refused and "(island 1 (west), pool i1-sglang)" in refused[0] and "64" in refused[0]
    assert not any(a for a in no_subprocess if "--dry-run" in a)  # refused before any dry run


def test_pools_of_one_island_recording_different_fields_are_refused(files, no_subprocess, capsys):
    _spec(files, "vllm", rollout_fields={"routed_experts": True, "top_logprobs_num": 0})
    run = _run(files)
    problem = run.fleet_preflight()
    assert problem.startswith("island 0 (east): its pools record different rollout_fields (i0-vllm: "
                              "fp32_lm_head false, routed_experts true, top_logprobs_num 0; i0-sglang: "
                              "fp32_lm_head false, routed_experts false, top_logprobs_num 0)")
    assert "island 1" not in problem  # one pool there: nothing to disagree with
    assert not run.preflight()


def test_an_r3_trainer_needs_routed_experts_in_every_pool(files):
    plan = fleet.plan_of(up.load_config(_write(files, _cfg(files))))
    argv = TRAINER_ARGV + ["--use-rollout-routing-replay"]
    r3 = {"rollout_fields": {"routed_experts": True}}
    found = up_fleet.rollout_fields_problems(plan, argv, [("i0-vllm", r3), ("i0-sglang", r3), ("i1-sglang", {})])
    assert found == ["island 1 (west), pool i1-sglang: the trainer replays the rollout routing (R3) but the pool's "
                     "sampler spec has no rollout_fields.routed_experts: set it true"]


# -- store and shards ----------------------------------------------------------------------

def test_a_pool_writing_another_islands_substore_is_refused(files):
    run = _run(files)
    plan = run.fleet_plan()
    doc = run.pools[2].load()
    assert up_fleet.store_problem(plan, "i1-sglang", doc) is None
    wrong = dict(doc, run_prefix="r/trainer0")  # island 1's pool writing island 0's sub-directory
    problem = up_fleet.store_problem(plan, "i1-sglang", wrong)
    assert "s3://b/runs/r/trainer0" in problem and "island 1's trainer reads s3://b/runs/r/trainer1" in problem
    # the single-trainer store check accepts any trainer<k>; the fleet check is what refuses it
    from yeto.rl import up_consistency

    assert up_consistency.store_problems("s3://b/runs/r", types.SimpleNamespace(rl_checkpoint_store="s3://b/runs/r"),
                                         wrong) == []
    assert up_fleet.pool_where(plan, "i1-sglang") == "island 1 (west), pool i1-sglang"


def test_the_shard_check_reads_the_islands_own_shards(files, capsys):
    run = _run(files)
    assert [run.shard_problem(i) for i in range(3)] == [None, None, None]
    argv = run.cfg["trainer"]["argv"]
    k = argv.index("1:sg-i1-0,sg-i1-1")
    argv[k] = "1:sg-i1-0"
    assert run.shard_problem(2) == ("pool shard_limit 2 but island 1's --rl-island-buffer-shards lacks ['sg-i1-1']: "
                                    "those nodes' groups would never be trained on")
    assert run.shard_problem(0) is None  # island 0's list is untouched
    del argv[k - 1:k + 1]
    assert "no --rl-island-buffer-shards 1:" in run.shard_problem(2)
    assert run._pool_suffix(2) == " (island 1 (west), pool i1-sglang)"


def test_island_shards_parse_like_the_launcher():
    argv = ["--rl-island-buffer-shards", "0: a-0, a-1,", "--rl-island-buffer-shards=1:b-0"]
    assert up_fleet.island_shards(argv) == {0: ["a-0", "a-1"], 1: ["b-0"]}


# -- release form and cadence (spec: release frequency vs engine reload cost) --------------

def test_a_vllm_island_releasing_every_version_is_warned_with_a_cadence(files, no_subprocess, capsys):
    cfg = _cfg(files)
    cfg["islands"][0]["trainer"].pop("argv")  # island 0 (vLLM pool): releases every local version
    run = _run(files, cfg)
    assert run.fleet_preflight() is None
    warned = [line for line in _lines(capsys) if "drains" in line]
    assert warned == ["preflight: WARNING: island 0 (east): pool(s) i0-vllm run vLLM, whose in-place reload drains the "
                      "engine (about 9-12 min of sampling per release measured on MI355X), and the island's trainer "
                      "releases every local version (--rl-release-every 1): set islands[0].trainer.argv "
                      "[\"--rl-release-every\", \"4\"]"]


def test_the_suggested_cadence_stays_within_the_islands_policy_age(files):
    cfg = _cfg(files)
    cfg["islands"][0]["trainer"]["argv"] = ["--rl-max-policy-age", "3"]
    plan = fleet.plan_of(up.load_config(_write(files, cfg)))
    specs = [("i0-vllm", {"engine": {"adapter": "vllm-server"}})]
    (line,) = up_fleet.release_every_warnings(plan, TRAINER_ARGV, specs)
    assert '["--rl-release-every", "3"]' in line and "raise --rl-max-policy-age" not in line
    cfg["islands"][0]["trainer"]["argv"] = ["--rl-max-policy-age", "1"]
    plan = fleet.plan_of(up.load_config(_write(files, cfg)))
    (line,) = up_fleet.release_every_warnings(plan, TRAINER_ARGV, specs)
    assert '["--rl-release-every", "2"]' in line and "raise --rl-max-policy-age to at least 2 (it is 1" in line
    # releasing every other version, or no vLLM pool: no warning
    assert up_fleet.release_every_warnings(plan, TRAINER_ARGV + ["--rl-release-every", "2"], specs) == []
    assert up_fleet.release_every_warnings(plan, TRAINER_ARGV, [("i0-vllm", {"engine": {"adapter": "sglang-server"}})]) == []


def test_an_engine_that_loads_no_release_is_refused(files):
    plan = fleet.plan_of(up.load_config(_write(files, _cfg(files))))
    reload = {"server": {"in_place_reload": True}}
    problems, notes = up_fleet.release_form_lines(plan, [
        ("i0-vllm", {"engine": {"adapter": "vllm-server", **reload}}),
        ("i0-sglang", {"engine": {"adapter": "sglang-server"}}),
        ("i1-sglang", {"engine": {"adapter": "miles-rollout"}})])
    assert problems == ["island 1 (west), pool i1-sglang: engine adapter 'miles-rollout' loads no full-parameter "
                        "release from the store (a fleet's pools are external vllm-server / sglang-server engines)"]
    assert notes == ["WARNING: island 0 (east), pool i0-sglang: engine.server.in_place_reload is off: every "
                     "full-parameter release restarts the sglang server (a cold start per release); turn it on once "
                     "its on-card reload test passed"]


def test_partial_experts_on_vllm_note_the_whole_reload(files, tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text("{}")
    cfg = _cfg(files)
    cfg["fleet"]["trainer_argv"] += ["--rl-expert-selection", str(selection)]
    plan = fleet.plan_of(up.load_config(_write(files, cfg)))
    _problems, notes = up_fleet.release_form_lines(plan, [
        ("i0-vllm", {"engine": {"adapter": "vllm-server", "server": {"in_place_reload": True}}})])
    assert notes == ["island 0 (east), pool i0-vllm: partial experts on vLLM: each release is laid over "
                     "engine.server.base_dir and reloaded whole (only the Kimi-K3.1 expert overlay reloads the "
                     "selected experts alone)"]


# -- syncer and budget (spec: fleet-level syncer region and budget) ------------------------

def test_the_dry_run_prints_the_fleet_budget_estimate_and_the_syncer(files, no_subprocess, capsys):
    cfg = _cfg(files, syncer={"usd_per_h": 0.5})
    assert up.main(_main_args(_write(files, cfg))) == 0
    out = _lines(capsys)
    assert ("preflight: fleet: budget estimate $24.50/h = pools $8.00/h (i0-vllm 1 x $4, i0-sglang 1 x $2, "
            "i1-sglang 1 x $2) + trainers $16.00/h (east $8, west $8) + syncer $0.50/h; over the 4 h wall clock "
            "$98.00 (budget.usd $40)") in out
    assert ("preflight: WARNING: fleet: budget.usd $40 lasts about 1.6 h at $24.50/h, before the 4 h wall clock: the "
            "budget breaker stops the whole fleet then") in out
    assert "preflight: fleet: syncer region us-west-2 (the default: the islands are not all pinned to one area)" in out
    assert any(line.startswith("preflight: WARNING: syncer: Modal island(s) 1@us-east run in another region") and
               "(launch islands: 0 = east, 1 = west)" in line for line in out)
    assert not any("syncer.usd_per_h is unset" in line for line in out)


def test_the_syncer_follows_islands_pinned_to_one_area(files, no_subprocess, capsys):
    cfg = _cfg(files)
    cfg["islands"][0]["trainer"]["gpu"] = "modal:1x2xh100@us-east"
    run = _run(files, cfg)
    assert run.fleet_preflight() is None
    out = _lines(capsys)
    assert "preflight: fleet: syncer region us-east-1 (follows the islands' @region)" in out
    assert not any("WARNING: syncer" in line for line in out)
    assert "preflight: fleet: syncer.usd_per_h is unset: the fleet spend counts the syncer at $0/h (set it when the " \
           "syncer runs on a billed VM)" in out


def test_an_unpriced_island_trainer_is_warned():
    plan = fleet.FleetPlan(run="r", store="s3://b/runs/r", dispense="fleet", islands=(
        fleet.FleetIsland(index=0, name="a", gpu="modal:1xh100", usd_per_h=None, store="s3://b/runs/r/trainer0"),))
    lines = up_fleet.budget_lines(plan, pools=[("i0-p", 2, 1.5)], syncer_usd_per_h=0.0, usd=100.0, wall_h=None)
    assert lines[0] == ("fleet: budget estimate $3.00/h = pools $3.00/h (i0-p 2 x $1.5) + trainers $0.00/h "
                        "(a $0 (unset)) + syncer $0.00/h (budget.usd $100)")
    assert lines[1].startswith("WARNING: island 0 (a): trainer.usd_per_h is unset")
    assert lines[2] == ("fleet: budget.usd $100 lasts about 33.3 h at $3.00/h (no wall clock): the budget breaker "
                        "stops the whole fleet then")


def test_an_island_with_a_miles_sampler_checks_its_pools_and_bills_its_sampler(files, no_subprocess, capsys):
    cfg = _cfg(files)
    cfg["islands"][1]["samplers"].insert(0, {"kind": "miles", "name": "miles", "gpu": "modal:1x1xl40s",
                                             "usd_per_h": 1.5})
    # a Miles sampler claiming from the pools' store ledger reads --data as their local prompt file
    (files / "p.jsonl").write_text('{"q": "1 + 1"}\n')
    argv = cfg["fleet"]["trainer_argv"]
    argv[argv.index("--data") + 1] = str(files / "p.jsonl")
    run = _run(files, cfg)
    assert [run.shard_problem(i) for i in range(len(run.pools))] == [None, None, None]
    assert run.fleet_preflight() is None
    (line,) = [x for x in _lines(capsys) if "budget estimate" in x]
    assert "trainers $17.50/h (east $8, west $9.5 with 1 Miles sampler(s))" in line


# -- single trainer ------------------------------------------------------------------------

def test_a_single_trainer_run_takes_no_fleet_check(tmp_path, monkeypatch):
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    (tmp_path / "pool.json").write_text(json.dumps({"shard_base": "s"}))
    run = up.Run({"run": "r1", "store": "s3://b/runs/r1", "pool": {"config": str(tmp_path / "pool.json")},
                  "trainer": {"argv": ["--rl-buffer-shards", "s-0"]}})
    assert run.fleet_preflight() is None and run._pool_suffix(0) == ""
    assert run.shard_problem(0).startswith("the trainer reads --rl-buffer-shards (1 shards)")
    run.pools.append(run.pools[0])
    assert run._pool_suffix(1) == f" (pool {run.pools[1].name})"


@pytest.mark.parametrize("flag, value", [("--rl-release-every", "3"), ("--rl-max-policy-age", "5")])
def test_island_cadence_falls_back_to_the_shared_argv(flag, value):
    island = fleet.FleetIsland(index=0, name="a", gpu="g", usd_per_h=None, store="s")
    argv = [flag, value]
    if flag == "--rl-release-every":
        assert up_fleet.island_release_every(island, argv) == 3 and up_fleet.island_release_every(island, []) == 1
    else:
        assert up_fleet.island_max_policy_age(island, argv) == 5 and up_fleet.island_max_policy_age(island, []) is None
