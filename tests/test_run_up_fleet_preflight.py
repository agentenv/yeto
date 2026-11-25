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

# a local --model config (Qwen3-8B's shape): 8.19B parameters, a 16.4 GB bf16 release,
# fetched in 81.9 s (200 MB/s) and loaded in 16.4 s (1 GB/s)
MODEL_CONFIG = {"num_hidden_layers": 36, "hidden_size": 4096, "intermediate_size": 12288, "vocab_size": 151936,
                "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "tie_word_embeddings": False}


def _local_model(files, cfg):
    model = files / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text(json.dumps(MODEL_CONFIG))
    argv = cfg["fleet"]["trainer_argv"]
    argv[argv.index("--model") + 1] = str(model)
    return cfg


def _argv(files):
    return _local_model(files, {"fleet": {"trainer_argv": list(TRAINER_ARGV)}})["fleet"]["trainer_argv"]


def test_the_release_size_is_the_model_configs_parameters_in_bf16(files):
    size, why = up_fleet.release_bytes(_argv(files))
    assert size == 8190722048 * 2 and why == f"8.2B parameters of {files / 'model'} x 2 B"
    assert up_fleet.release_bytes(["--model", str(files / "absent")]) == (
        None, f"no local config.json of --model {files / 'absent'}")


def test_a_vllm_island_without_a_step_time_reports_the_release_cost_only(files, no_subprocess, capsys):
    cfg = _local_model(files, _cfg(files))
    cfg["islands"][0]["trainer"].pop("argv")  # island 0 (vLLM pool): releases every local version
    run = _run(files, cfg)
    assert run.fleet_preflight() is None
    (warned,) = [line for line in _lines(capsys) if "a release costs" in line and "island 0" in line]
    assert warned == ("preflight: WARNING: island 0 (east): a release costs 11.6 min of sampling on pool i0-vllm "
                      "(vllm): fetch 82 s + load 16 s of a 16 GB release (8.2B parameters of "
                      f"{files / 'model'} x 2 B) + drain up to 10.0 min (min of drain_timeout_s, reload_max_defer_s); "
                      "the island releases every local version (--rl-release-every 1); set "
                      "islands[0].trainer.step_s (or fleet.step_s), the seconds of one local version, for a "
                      "suggested --rl-release-every")
    assert "release_every" not in warned.split("; ")[0] and "9-12 min" not in warned


def test_a_known_step_time_gives_the_cadence_and_the_policy_age(files, no_subprocess, capsys):
    cfg = _local_model(files, _cfg(files))
    cfg["fleet"]["step_s"] = 120
    cfg["islands"][1]["trainer"]["step_s"] = 600  # island 1's own step time
    run = _run(files, cfg)
    assert run.fleet_preflight() is None
    out = [line for line in _lines(capsys) if "a release costs" in line]
    # island 0 releases every 2 versions: ceil(698 s / 120 s) = 6, its policy age the default
    assert out[0].startswith("preflight: WARNING: island 0 (east): a release costs 11.6 min of sampling on pool "
                             "i0-vllm (vllm)")
    assert out[0].endswith("at a step of about 2.0 min the island's --rl-release-every 2 releases faster than its pools "
                           "take a release in: set islands[0].trainer.argv [\"--rl-release-every\", \"6\"] and "
                           "[\"--rl-max-policy-age\", \"8\"] (at least N + 2; it is the default) (N = max(2, "
                           "ceil(698 s / 120 s)))")
    # island 1 (SGLang, no live_update: it drains as well): ceil(698 / 600) = 2 > its release_every 1
    assert out[1].startswith("preflight: WARNING: island 1 (west): a release costs 11.6 min of sampling on pool "
                             "i1-sglang (sglang)")
    assert '["--rl-release-every", "2"] and ["--rl-max-policy-age", "4"] (at least N + 2; it is 3)' in out[1]
    assert len(out) == 2


def test_the_cadence_follows_the_engine_and_its_drain_cut(files):
    cfg = _local_model(files, _cfg(files))
    plan = fleet.plan_of(up.load_config(_write(files, cfg)))
    argv = _argv(files)
    vllm = {"engine": {"adapter": "vllm-server"}, "reload_max_defer_s": 900, "agentic": {"drain_timeout_s": 300}}
    (line,) = up_fleet.release_every_warnings(plan, argv, [("i0-vllm", vllm)], step_s={0: 60})
    assert "drain up to 5.0 min (min of drain_timeout_s, reload_max_defer_s)" in line
    assert '["--rl-release-every", "7"]' in line and "ceil(398 s / 60 s)" in line  # 82 + 16 + 300
    # SGLang with live_update: no drain, fetch + load only: ceil(98 / 60) = 2 = island 0's release_every 2
    live = {"engine": {"adapter": "sglang-server", "options": {"live_update": True}}}
    assert up_fleet.release_every_warnings(plan, argv, [("i0-sglang", live)], step_s={0: 60}) == []
    (line,) = up_fleet.release_every_warnings(plan, argv, [("i0-sglang", live)], step_s={0: 30})
    assert "+ live_update: no drain" in line and '["--rl-release-every", "4"]' in line
    # vLLM is drain-only: live_update in its options changes nothing
    (line,) = up_fleet.release_every_warnings(
        plan, argv, [("i0-vllm", {"engine": {"adapter": "vllm-server", "options": {"live_update": True}}})],
        step_s={0: 60})
    assert "drain up to 10.0 min" in line
    # a release often enough, or no external engine: no line
    assert up_fleet.release_every_warnings(plan, argv, [("i0-vllm", vllm)], step_s={0: 600}) == []  # N = 2
    assert up_fleet.release_every_warnings(plan, argv, [("i0-vllm", {"engine": {"adapter": "fake"}})]) == []


def test_an_unknown_release_size_bounds_the_cost_by_the_drain(files):
    plan = fleet.plan_of(up.load_config(_write(files, _cfg(files))))
    argv = list(TRAINER_ARGV)
    argv[argv.index("--model") + 1] = str(files / "no-model")
    (line,) = up_fleet.release_every_warnings(plan, argv, [("i0-vllm", {"engine": {"adapter": "vllm-server"}})],
                                              step_s={0: 100})
    assert line.startswith("WARNING: island 0 (east): a release costs at least 10.0 min of sampling on pool i0-vllm "
                           f"(vllm): fetch + load not estimated (no local config.json of --model {files / 'no-model'})")
    assert '["--rl-release-every", "6"]' in line


def test_island_step_times_are_validated_and_read(files):
    cfg = _cfg(files)
    cfg["fleet"]["step_s"] = 90
    cfg["islands"][1]["trainer"]["step_s"] = 45.5
    assert fleet.island_step_s(up.load_config(_write(files, cfg))) == {0: 90.0, 1: 45.5}
    for where, edit in (("fleet.step_s", lambda c: c["fleet"].update(step_s=0)),
                        (r"islands\[0\] \(east\).trainer.step_s", lambda c: c["islands"][0]["trainer"].update(
                            step_s="2m"))):
        bad = _cfg(files)
        edit(bad)
        with pytest.raises(ValueError, match=f"{where} must be a number of seconds > 0"):
            up.load_config(_write(files, bad))


def test_an_engine_that_loads_no_release_is_refused(files):
    plan = fleet.plan_of(up.load_config(_write(files, _cfg(files))))
    reload = {"server": {"in_place_reload": True}}
    problems, notes = up_fleet.release_form_lines(plan, [
        ("i0-vllm", {"engine": {"adapter": "vllm-server", **reload}}),
        ("i0-sglang", {"engine": {"adapter": "sglang-server"}}),
        ("i1-sglang", {"engine": {"adapter": "miles-rollout"}})], TRAINER_ARGV)
    assert problems == ["island 1 (west), pool i1-sglang: engine adapter 'miles-rollout' loads no full-parameter "
                        "release from the store (a fleet's pools are external vllm-server / sglang-server engines)"]
    assert notes == ["WARNING: island 0 (east), pool i0-sglang: engine.server.in_place_reload is off: every "
                     "full-parameter release restarts the sglang server (a cold start per release); turn it on once "
                     "its on-card reload test passed"]


def test_generic_partial_experts_are_a_whole_checkpoint_reloaded_whole(files, tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text("{}")
    cfg = _cfg(files)
    cfg["fleet"]["trainer_argv"] += ["--rl-expert-selection", str(selection)]
    plan = fleet.plan_of(up.load_config(_write(files, cfg)))
    argv = cfg["fleet"]["trainer_argv"]
    server = {"server": {"in_place_reload": True}}
    _problems, notes = up_fleet.release_form_lines(plan, [
        ("i0-vllm", {"engine": {"adapter": "vllm-server", **server}}),
        ("i0-sglang", {"engine": {"adapter": "sglang-server", **server}})], argv)
    assert notes == [f"island 0 (east), pool i0-{e}: partial experts (generic recipe): each release is a whole HF "
                     "safetensors checkpoint with no release manifest and no expert overlay, and the "
                     f"{e} server reloads it whole; --rl-release-delta-full-every applies to it"
                     for e in ("vllm", "sglang")]
    assert not any("base_dir" in n for n in notes)
    # only the kimi_k31 recipe releases an expert overlay with a manifest
    _problems, notes = up_fleet.release_form_lines(plan, [
        ("i0-vllm", {"engine": {"adapter": "vllm-server", **server}})], argv + ["--rl-model-recipe", "kimi_k31"])
    assert notes == ["island 0 (east), pool i0-vllm: partial experts (kimi_k31 recipe): each release carries the "
                     "trained experts and a release manifest, laid over engine.server.base_dir (the expert overlay) "
                     "and reloaded expert by expert with engine.options.expert_reload"]


def test_a_generic_partial_experts_fleet_refuses_a_pool_bound_to_a_selection(files, no_subprocess, capsys, tmp_path):
    """The sampler of a spec with expert_selection_sha256 refuses every generic release
    (no manifest): the fleet preflight refuses it and names the island and the pool."""
    selection = tmp_path / "selection.json"
    selection.write_text("{}")
    _spec(files, "sglang", expert_selection_sha256="a" * 64)
    cfg = _cfg(files)
    cfg["fleet"]["trainer_argv"] += ["--rl-expert-selection", str(selection)]
    run = _run(files, cfg)
    assert not run.preflight()
    (refused,) = [line for line in _lines(capsys) if "expert_selection_sha256" in line]
    assert "carry no expert selection" in refused and "remove expert_selection_sha256 from the sampler spec" in refused
    assert refused.endswith("(island 0 (east), pool i0-sglang)")
    assert not any(a for a in no_subprocess if "--dry-run" in a)


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
    assert "preflight: fleet: syncer.usd_per_h is unset: the fleet spend counts the syncer at $0/h, on the head " \
           "node (--controller head runs it next to the controller, on no machine of its own); set syncer.usd_per_h " \
           "to price its own machine" in out


def test_an_unpriced_island_trainer_is_warned():
    plan = fleet.FleetPlan(run="r", store="s3://b/runs/r", dispense="fleet", islands=(
        fleet.FleetIsland(index=0, name="a", gpu="modal:1xh100", usd_per_h=None, store="s3://b/runs/r/trainer0"),))
    lines = up_fleet.budget_lines(plan, pools=[("i0-p", 2, 1.5)], syncer_usd_per_h=0.0, usd=100.0, wall_h=None)
    assert lines[0] == ("fleet: budget estimate $3.00/h = pools $3.00/h (i0-p 2 x $1.5) + trainers $0.00/h "
                        "(a $0 (unset)) + syncer $0.00/h (budget.usd $100)")
    assert lines[1].startswith("WARNING: island 0 (a): trainer.usd_per_h is unset")
    assert lines[2] == ("fleet: budget.usd $100 lasts about 33.3 h at $3.00/h (no wall clock): the budget breaker "
                        "stops the whole fleet then")


def _shared_prompts(files, cfg):
    """A Miles sampler claims from the pools' task ledger, which pins their prompt file:
    ``--data`` is a local JSONL with the same bytes as every pool's ``prompts_file``."""
    rows = "".join(json.dumps({"q": f"question {n}", "a": str(n)}) + "\n" for n in range(6))
    (files / "prompts.jsonl").write_text(rows)
    (files / "pool-prompts.jsonl").write_text(rows)
    pool = json.loads((files / "pool.sglang.json").read_text())
    (files / "pool.sglang.json").write_text(json.dumps(dict(pool, prompts_file=str(files / "pool-prompts.jsonl"))))
    cfg["islands"][0]["samplers"][0]["prompts_file"] = str(files / "pool-prompts.jsonl")
    argv = cfg["fleet"]["trainer_argv"]
    argv[argv.index("--data") + 1] = str(files / "prompts.jsonl")
    return cfg


def test_an_island_with_a_miles_sampler_checks_its_pools_and_bills_its_sampler(files, no_subprocess, capsys):
    cfg = _shared_prompts(files, _cfg(files))
    cfg["islands"][1]["samplers"].insert(0, {"kind": "miles", "name": "miles", "gpu": "modal:1x1xl40s",
                                             "usd_per_h": 1.5})
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
