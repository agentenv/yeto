"""CPU end-to-end simulation of async GRPO with context compaction (scripts/sim_compaction_grpo.py).

Runs on main: the no-compaction scenario through ``AgenticRound`` -> buffer ->
trainer read -> Miles' GRPO normalization, and the compaction scenarios at the
records level (the records proxy compacts; each segment becomes a buffer
sample in the compaction-1118 contract form; Miles' normalization runs on the
segment samples). The steps that need the compaction-1118 parts skip until
those parts are in the tree (``capabilities()``):

* ``sampler_segments`` (C1): ``AgenticRound`` writes one sample per segment;
* ``spec_compaction`` (C2): the spec's ``agentic.compaction`` reaches the proxy;
* ``trainer_segments`` (C3): the trainer reads a group of segment samples with
  its trial identity (siblings share an advantage).
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import sim_compaction_grpo as sim  # noqa: E402

pytest.importorskip("aiohttp")

CAPS = sim.capabilities()
needs_c1 = pytest.mark.skipif(not CAPS["sampler_segments"], reason="pending integration: C1 sampler segment samples")
needs_c2 = pytest.mark.skipif(not CAPS["spec_compaction"], reason="pending integration: C2 spec agentic.compaction")
needs_c3 = pytest.mark.skipif(not CAPS["trainer_segments"], reason="pending integration: C3 trainer segment groups")
needs_c1_c3 = pytest.mark.skipif(not (CAPS["sampler_segments"] and CAPS["trainer_segments"]),
                                 reason="pending integration: C1 sampler segments + C3 trainer segment groups")

# the buffer sample of a trial without compaction on main (agentic_round.agentic_sample)
MAIN_KEYS = {"group_index", "index", "label", "loss_mask", "metadata", "prompt", "response", "response_length",
             "reward", "rollout_log_probs", "rollout_routed_experts", "rollout_top_logprobs",
             "rollout_top_logprobs_ids", "status", "tokens"}
MAIN_META = {"agent", "agent_turns", "engine", "finish_reason", "prompt_id", "prompt_tokens_known", "provenance",
             "task_id", "trajectory_id", "weight_version"}


def scenario(name: str, **kw) -> sim.Scenario:
    return dataclasses.replace(sim.SCENARIOS[name], **{"tasks": 2, **kw})


def run(name: str, tmp_path: Path, *, path: str = "auto", **kw) -> sim.SimRun:
    return sim.SimRun(scenario(name, **kw), path=path, workdir=tmp_path / name).run()


def trials_of(samples):
    by = {}
    for x in samples:
        by.setdefault(sim.trial_key(x), []).append(x)
    return by


def check_trial_samples(r: sim.SimRun) -> None:
    """Contract checks on every kept trial's samples (sampler side)."""
    cfg = r.scenario.compaction_config()
    for t in r.trials:
        if not t.samples:
            continue
        rewards = {x["reward"] for x in t.samples}
        assert len(rewards) == 1, f"{t.trajectory_id}: segments disagree on the reward {rewards}"
        assert len({(x["group_index"], x["index"]) for x in t.samples}) == 1
        for x in t.samples:
            assert sim.check_sample_alignment(x, r.scenario) == [], t.trajectory_id
            assert len(x["tokens"]) <= r.scenario.max_model_len
        if len(t.samples) == 1:
            assert "compaction" not in t.samples[0]["metadata"]
            continue
        metas = [x["metadata"]["compaction"] for x in t.samples]
        k = len(t.samples)
        assert [m["segment_index"] for m in metas] == list(range(k))
        assert {m["num_segments"] for m in metas} == {k} and {m["schema"] for m in metas} == {1}
        assert {m["trajectory_id"] for m in metas} == {t.trajectory_id}
        assert {m["context_budget"] for m in metas} == {cfg.context_budget}
        trained = [sum(x["loss_mask"]) for x in t.samples]
        assert [m["segment_tokens"] for m in metas] == trained
        assert [m["tokens_after"] for m in metas] == [sum(trained[i + 1:]) for i in range(k)]
        assert all(x["status"] == "completed" for x in t.samples[:-1])
        for x in t.samples:
            meta = x["metadata"]
            assert int(meta.get("final_weight_version", meta["weight_version"])) >= int(meta["weight_version"])


def check_advantages(verdict) -> None:
    assert verdict["samples"] > 0
    assert verdict["segments_share_advantage"], verdict
    assert verdict["matches_trial_grpo"], verdict


# ---------------------------------------------------------------------- the pieces
def test_sim_template_records_without_render_drift(tmp_path):
    r = run("compact_once", tmp_path, path="records")
    counters = Counter()
    for t in r.trials:
        counters.update(t.records["counters"])
    assert counters["render_drift_turns"] == 0 and counters["stop_added"] == 0
    assert counters["compactions"] == sum(len(t.samples) - 1 for t in r.trials)


def test_engine_fakes_every_rollout_array_and_reloads_in_place(tmp_path):
    r = run("mixed_reload", tmp_path, path="records", reload_after_calls=(40, 80))
    assert [v for _, v in r.engine.reloads] == [1, 2] and r.engine.served_version() == 2
    m = r.metrics()
    assert len(m["versions"]) >= 2
    assert m["multi_version_samples"] > 0  # a segment whose turns span a reload
    kinds = Counter(c.kind for c in r.engine.ledger)
    assert kinds["action"] and kinds["final"] and kinds["summary"] == m["compactions"]


def test_timing_is_simulated_from_tokens_and_tool_draws(tmp_path):
    r = run("no_trigger", tmp_path, path="records")
    s = r.scenario
    for t in r.trials:
        calls = [c for c in r.engine.ledger if c.trajectory == t.trajectory_id]
        expected = sum(s.call_overhead_s + s.prefill_s_per_token * (c.prompt_tokens - c.cached_tokens)
                       + s.decode_s_per_token * c.completion_tokens for c in calls)
        assert math.isclose(t.llm_s, expected) and t.llm_s > 0
        env = r.agent.trials[t.trajectory_id]["env"]
        assert len(env.tool_s) == env.done == env.steps and t.env_s == pytest.approx(sum(env.tool_s)) and t.env_s > 0
        assert all(c.cached_tokens > 0 for c in calls[1:])  # one segment: every later call extends the stream
    m = r.metrics()
    assert m["time_env_plus_llm_s"]["max"] >= m["time_env_s"]["max"] > 0
    assert m["round_time_env_s"] > m["wall_sampler_s"]  # nothing sleeps


def test_normalization_port_matches_miles_helpers_and_trial_grpo():
    miles = sim.miles_imports()
    rows = [dict(group_index=g, index=i, rollout_id=None, reward=float(r)) for g, i, r in
            [(0, 0, 1), (0, 0, 1), (0, 1, 0), (0, 2, 1), (1, 3, 0), (1, 3, 0), (1, 3, 0), (1, 4, 0), (1, 5, 1)]]
    samples = sim.to_miles_samples(sim.SimpleNamespace(Sample=None), rows)
    args = sim.miles_args(scenario("no_compaction"))
    rewards = [x["reward"] for x in rows]
    port = sim.normalize_rewards_by_rollout(sim.SimpleNamespace(normalize=None, reward_group_segments=None,
                                                                rollout_segments=None), args, samples, rewards,
                                            unshaped_rewards=rewards)
    trial = {0: sim.grpo_reference([1.0, 0.0, 1.0]), 1: sim.grpo_reference([0.0, 0.0, 1.0])}
    expected = [trial[0][0], trial[0][0], trial[0][1], trial[0][2], trial[1][0], trial[1][0], trial[1][0],
                trial[1][1], trial[1][2]]
    assert port == pytest.approx(expected, abs=1e-6)
    if miles.rollout_segments is not None:  # Miles on the path: its own helpers give the same rows
        assert miles.rollout_segments(samples, list(range(9))) == sim._rollout_segments(samples, list(range(9)))
        assert miles.reward_group_segments(args, samples) == sim._reward_group_segments(args, samples)
        assert sim.normalize_rewards_by_rollout(miles, args, samples, rewards, unshaped_rewards=rewards) == \
            pytest.approx(expected, abs=1e-6)
    with pytest.raises(ValueError, match="share one reward"):
        sim.normalize_rewards_by_rollout(miles, args, samples, [1.0, 0.0, *rewards[2:]], unshaped_rewards=rewards)


# ---------------------------------------------------------------------- on main
def test_no_compaction_runs_end_to_end_in_the_main_format(tmp_path):
    r = run("no_compaction", tmp_path)
    assert r.path == "agentic"
    m = r.metrics()
    assert m["trials_per_task"] == {0: 8, 1: 8} and m["outcomes"] == {"written": 16}
    samples = [x for t in r.trials for x in t.samples]
    for x in samples:
        assert set(x) == MAIN_KEYS and set(x["metadata"]) == MAIN_META
    check_trial_samples(r)
    assert m["segments_per_trial"] == {1: 16} and m["alignment_problem_count"] == 0
    (step,) = m["trainer_steps"]
    assert step["samples"] == step["trials"] == 16 and step["groups"] == 2
    check_advantages(m["advantage_check"]["steps"][0])


@pytest.mark.parametrize("name", ["no_trigger", "compact_once", "compact3_truncated", "partial_abort"])
def test_records_level_compaction_segments_and_shared_advantage(tmp_path, name):
    r = run(name, tmp_path, path="records")
    m = r.metrics()
    assert m["trials_per_task"] == {0: 8, 1: 8}
    check_trial_samples(r)
    check_advantages(m["advantage_check"]["direct"])
    segments = m["segments_per_trial"]
    if name == "no_trigger":
        assert segments == {1: 16} and m["compactions"] == 0
        assert all("compaction" not in x["metadata"] for t in r.trials for x in t.samples)
    elif name == "compact_once":
        assert segments == {2: 16} and m["compaction_attempts"] == m["compactions"] == 16
        assert m["segment_tokens_by_index"][1]["max"] < m["segment_tokens_by_index"][0]["min"]
    elif name == "compact3_truncated":
        cfg = r.scenario.compaction_config()
        assert segments == {cfg.max_compactions + 1: 16}
        for t in r.trials:  # compactions used up: the context fills and the last segment is truncated
            assert t.reward == 0.0 and [x["status"] for x in t.samples][-1] == "truncated"
        assert m["status"]["truncated"] == 16
    else:
        assert m["outcomes"]["aborted"] > 0
        for t in r.trials:
            assert (t.outcome == "aborted") == (not t.samples)
        assert sum(m["kept_trials_per_task"].values()) == m["outcomes"].get("ok", 0) + \
            m["outcomes"].get("policy_limit", 0)


def test_tcomp_below_one_step_overflows_at_the_summary_request(tmp_path):
    """t_comp 512 with steps of up to ~700 tokens: the trigger fires after the step that
    crossed it, and the summary request no longer fits; the trial ends as a policy
    limit (reward 0), not an infrastructure abort."""
    r = run("tcomp_overflow", tmp_path, path="records", tasks=4)
    m = r.metrics()
    assert m["compaction_attempts"] > m["compactions"] and m["context_full_exits"] > 0
    assert "aborted" not in m["outcomes"] and m["outcomes"]["policy_limit"] == m["context_full_exits"]
    check_advantages(m["advantage_check"]["direct"])


def test_cli_prints_both_tables_and_writes_json(tmp_path, capsys):
    out = tmp_path / "m.json"
    assert sim.main(["--scenario", "no_trigger", "--path", "records", "--json", str(out),
                     "--workdir", str(tmp_path)]) == 0
    text = capsys.readouterr().out
    assert "| no_trigger | records |" in text and "trainer steps" in text
    (metrics,) = json.loads(out.read_text())
    assert metrics["scenario"] == "no_trigger" and metrics["samples"] == 32


# ---------------------------------------------------------------------- pending integration
@needs_c1
@pytest.mark.parametrize("name", ["no_trigger", "compact_once", "compact3_truncated", "mixed_reload",
                                  "partial_abort"])
def test_agentic_round_writes_one_sample_per_segment(tmp_path, name):
    r = run(name, tmp_path, path="agentic")
    m = r.metrics()
    check_trial_samples(r)
    assert m["alignment_problem_count"] == 0
    if name == "no_trigger":  # compaction on but never triggered: byte-for-byte the main sample
        for x in (x for t in r.trials for x in t.samples):
            assert set(x) == MAIN_KEYS and set(x["metadata"]) == MAIN_META
    if name in ("compact_once", "compact3_truncated"):
        assert min(m["segments_per_trial"]) >= 2
    if name == "mixed_reload":
        assert m["multi_version_samples"] > 0 and len(m["versions"]) >= 2


@needs_c3
def test_trainer_reads_segment_groups_with_one_advantage_per_trial(tmp_path):
    r = run("compact_once", tmp_path, path="records")
    m = r.metrics()
    assert m["trainer_steps"] and all("error" not in step for step in m["trainer_steps"])
    trials = sum(step["trials"] for step in m["trainer_steps"])
    assert trials == sum(m["kept_trials_per_task"].values())
    assert sum(step["samples"] for step in m["trainer_steps"]) == m["samples"] > trials
    for verdict in m["advantage_check"]["steps"]:
        check_advantages(verdict)


@needs_c1_c3
@pytest.mark.parametrize("name", ["compact_once", "compact3_truncated", "mixed_reload", "partial_abort"])
def test_end_to_end_agentic_round_to_trainer_advantages(tmp_path, name):
    r = run(name, tmp_path, path="agentic")
    m = r.metrics()
    check_trial_samples(r)
    steps = m["trainer_steps"]
    assert steps and all("error" not in step for step in steps)
    assert sum(step["trials"] for step in steps) == sum(m["kept_trials_per_task"].values())
    for verdict in m["advantage_check"]["steps"]:
        check_advantages(verdict)


@needs_c2
def test_spec_compaction_section_builds_the_proxy_config():
    from yeto.rl.compaction import CompactionConfig
    from yeto.rl.engine import agentic_round

    parsers = [getattr(agentic_round, n) for n in dir(agentic_round)
               if "compaction" in n.lower() and callable(getattr(agentic_round, n))]
    assert parsers, "C2: no spec compaction parser in agentic_round"
    spec = {"agentic": {"compaction": dict(sim.SMALL)}}
    assert any(_try(p, spec) == CompactionConfig(**sim.SMALL) for p in parsers)
    assert all(_try(p, {"agentic": {}}) in (None, "error") for p in parsers)


def _try(parser, spec):
    try:
        return parser(spec)
    except Exception:  # noqa: BLE001 - probing an interface C2 names
        return "error"
