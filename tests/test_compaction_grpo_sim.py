"""CPU end-to-end simulation of async GRPO with context compaction (scripts/sim_compaction_grpo.py).

The spec's ``agentic.compaction`` (``spec_compaction`` / ``config_from_spec``) -> the
records proxy behind ``ProxyTokenRecorder(allow_compaction=True)`` -> ``AgenticRound``
(one sample per segment, ``agentic_samples``) -> local sample buffer -> the trainer's
``BatchReader`` and ``_BufferRound`` (trial numbering, padding by trial, the
``compaction_batch`` line) -> Miles' GRPO normalization, with a fake engine, a pi-like
agent and a simulated environment.
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

# the buffer sample of a trial without compaction on main (agentic_round.agentic_sample)
MAIN_KEYS = {"group_index", "index", "label", "loss_mask", "metadata", "prompt", "response", "response_length",
             "reward", "rollout_log_probs", "rollout_routed_experts", "rollout_top_logprobs",
             "rollout_top_logprobs_ids", "status", "tokens"}
MAIN_META = {"agent", "agent_turns", "engine", "finish_reason", "prompt_id", "prompt_tokens_known", "provenance",
             "task_id", "trajectory_id", "weight_version"}
COMPACTION_KEYS = {"schema", "trajectory_id", "segment_index", "num_segments", "context_budget", "summary_ok",
                   "kept_steps", "segment_tokens", "tokens_after", "summary_tokens", "retry_replays"}


def scenario(name: str, **kw) -> sim.Scenario:
    return dataclasses.replace(sim.SCENARIOS[name], **{"tasks": 2, **kw})


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Each scenario run once (two tasks of n=8; batch_count keeps its six tasks)."""
    cache: dict[str, sim.SimRun] = {}

    def get(name: str) -> sim.SimRun:
        if name not in cache:
            kw = {} if name == "batch_count" else {"tasks": 2}
            cache[name] = sim.SimRun(dataclasses.replace(sim.SCENARIOS[name], **kw),
                                     workdir=tmp_path_factory.mktemp(name)).run()
        return cache[name]

    return get


def check_trial_samples(r: sim.SimRun) -> None:
    """Contract checks on every kept trial's samples (sampler side)."""
    cfg = r.cfg
    for t in r.trials:
        if not t.samples:
            continue
        assert len({x["reward"] for x in t.samples}) == 1, t.trajectory_id
        assert len({(x["group_index"], x["index"]) for x in t.samples}) == 1, t.trajectory_id
        for x in t.samples:
            assert sim.check_sample_alignment(x, r.scenario) == [], t.trajectory_id
            assert len(x["tokens"]) <= r.scenario.max_model_len
        if len(t.samples) == 1:
            assert "compaction" not in t.samples[0]["metadata"]
            continue
        metas = [x["metadata"]["compaction"] for x in t.samples]
        k = len(t.samples)
        assert all(set(m) == COMPACTION_KEYS for m in metas)
        assert [m["segment_index"] for m in metas] == list(range(k))
        assert {m["num_segments"] for m in metas} == {k} and {m["schema"] for m in metas} == {1}
        assert {m["trajectory_id"] for m in metas} == {t.trajectory_id}
        assert {m["context_budget"] for m in metas} == {cfg.context_budget}
        trained = [sum(x["loss_mask"]) for x in t.samples]
        assert [m["segment_tokens"] for m in metas] == trained
        assert [m["tokens_after"] for m in metas] == [sum(trained[i + 1:]) for i in range(k)]
        assert all(x["status"] == "completed" for x in t.samples[:-1])
        assert metas[0]["kept_steps"] is None and all(m["kept_steps"] >= 0 for m in metas[1:])
        assert metas[-1]["summary_ok"] is None and all(isinstance(m["summary_ok"], bool) for m in metas[:-1])
        for x in t.samples:
            meta = x["metadata"]
            assert int(meta.get("final_weight_version", meta["weight_version"])) >= int(meta["weight_version"])


def check_trainer(r: sim.SimRun) -> None:
    """Every trainer step: no error, the global batch counts trials (rollout_batch_size x n
    distinct rollouts, padding one rollout per missing trial), the ``compaction_batch``
    line names the step's trials and segments, and every trial's segments share the
    advantage of trial-level GRPO."""
    m = r.metrics()
    steps = m["trainer_steps"]
    assert steps and all("error" not in step for step in steps), steps
    assert sum(step["trials"] for step in steps) == sum(m["kept_trials_per_task"].values())
    assert sum(step["samples"] for step in steps) == m["samples"]
    for step, verdict in zip(steps, m["advantage_check"]):
        assert step["distinct_rollouts"] == step["expected_rollouts"] == r.scenario.rollout_batch_size * r.scenario.n
        assert step["trials"] + step["padded_samples"] == step["expected_rollouts"]
        line = step["compaction_batch"]
        if step["samples"] > step["trials"]:  # a compacted trial in the round
            assert line == {"rollout": step["step"], "trials": step["trials"], "segments": step["samples"],
                            "per_trial": line["per_trial"]}
            assert sum(line["per_trial"].values()) == step["trials"]
            assert sum(k * v for k, v in line["per_trial"].items()) == step["samples"]
            meta = step["round_metadata"]
            assert meta["buffer_compacted_samples"] == sum(k * v for k, v in line["per_trial"].items() if k > 1)
        assert verdict["segments_share_advantage"] and verdict["one_rollout_per_trial"], verdict
        assert verdict["matches_trial_grpo"], verdict


# ---------------------------------------------------------------------- the pieces
def test_sim_template_records_without_render_drift(runs):
    r = runs("compact_once")
    counters = Counter()
    for t in r.trials:
        counters.update(t.counters)
    assert counters["render_drift_turns"] == 0 and counters["stop_added"] == 0
    assert counters["compactions"] == sum(len(t.samples) - 1 for t in r.trials) == r.metrics()["compaction_attempts"]


def test_spec_section_builds_the_proxy_config_and_refuses_what_launch_refuses():
    engine = sim.SimEngine(scenario("compact_once"), sim.sim_template())
    assert scenario("compact_once").compaction_config(engine) == sim.CompactionConfig(**sim.SMALL)
    assert scenario("no_compaction").compaction_config(engine) is None
    assert "cannot hold the summary turn" in scenario("tcomp_overflow").spec_error()
    assert "max_response_len 3250 < agentic.compaction.context_budget" in scenario("r3_segment_window").spec_error()
    from yeto.rl.agentic import HarnessConfigError
    from yeto.rl.engine.agentic_round import spec_compaction

    with pytest.raises(HarnessConfigError):
        spec_compaction(scenario("tcomp_overflow").spec(), engine)


def test_engine_fakes_every_rollout_array_and_reloads_in_place(tmp_path):
    r = sim.SimRun(scenario("mixed_reload", reload_after_calls=(40, 80)), workdir=tmp_path).run()
    assert [v for _, v in r.engine.reloads] == [1, 2] and r.engine.served_version() == 2
    m = r.metrics()
    assert len(m["versions"]) >= 2 and m["multi_version_samples"] > 0  # a segment whose turns span a reload
    kinds = Counter(c.kind for c in r.engine.ledger)
    assert kinds["action"] and kinds["summary"] == m["compactions"]
    check_trial_samples(r)


def test_timing_is_simulated_from_tokens_and_tool_draws(runs):
    r = runs("no_trigger")
    s = r.scenario
    for t in r.trials:
        calls = [c for c in r.engine.ledger if c.trajectory == t.trajectory_id]
        expected = sum(s.call_overhead_s + s.prefill_s_per_token * (c.prompt_tokens - c.cached_tokens)
                       + s.decode_s_per_token * c.completion_tokens for c in calls)
        assert math.isclose(t.llm_s, expected) and t.llm_s > 0
        env = r.agent.trials[t.trajectory_id]["env"]
        assert len(env.tool_s) == env.done == env.steps and t.env_s == pytest.approx(sum(env.tool_s))
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


# ---------------------------------------------------------------------- sampler side
def test_no_compaction_runs_end_to_end_in_the_main_format(runs):
    r = runs("no_compaction")
    m = r.metrics()
    assert m["trials_per_task"] == {0: 8, 1: 8} and m["outcomes"] == {"written": 16}
    for x in (x for t in r.trials for x in t.samples):
        assert set(x) == MAIN_KEYS and set(x["metadata"]) == MAIN_META
    check_trial_samples(r)
    check_trainer(r)
    assert all(step["compaction_batch"] is None for step in m["trainer_steps"])  # no line without compaction


@pytest.mark.parametrize("name", ["no_trigger", "compact_once", "compact3_truncated", "mixed_reload",
                                  "partial_abort", "tcomp_overflow", "r3_segment_window", "batch_count"])
def test_agentic_round_writes_one_sample_per_segment(runs, name):
    r = runs(name)
    m = r.metrics()
    check_trial_samples(r)
    assert m["alignment_problem_count"] == 0
    segments = m["segments_per_trial"]
    cfg = r.cfg
    if name == "no_trigger":  # compaction on but never triggered: the main sample
        assert segments == {1: 16} and m["compactions"] == 0
        for x in (x for t in r.trials for x in t.samples):
            assert set(x) == MAIN_KEYS and set(x["metadata"]) == MAIN_META
    elif name == "compact_once":
        assert segments == {2: 16}
        assert m["segment_tokens_by_index"][1]["max"] < m["segment_tokens_by_index"][0]["min"]
    elif name == "compact3_truncated":  # compactions used up: the context fills, the last segment is truncated
        assert segments == {cfg.max_compactions + 1: 16}
        for t in r.trials:
            assert t.reward == 0.0 and [x["status"] for x in t.samples] == ["completed"] * 3 + ["truncated"]
    elif name == "mixed_reload":
        assert m["multi_version_samples"] > 0 and len(m["versions"]) >= 2
    elif name in ("partial_abort", "batch_count"):
        assert m["lost"]["SandboxLostError"] > 0
        assert sum(m["kept_trials_per_task"].values()) == sum(m["trials_per_task"].values()) - m["lost"][
            "SandboxLostError"]
    elif name == "tcomp_overflow":  # the spec refuses t_comp 512; at run time the proxy omits the newest output
        assert m["spec_error"] and m["proxy_counters"]["summary_newest_omitted"] > 0
        assert m["compaction_attempts"] == m["compactions"] and m["compaction_no_room"] == 0
    elif name == "r3_segment_window":  # each segment's R3 window starts at its own first prompt
        assert m["r3_cut_turns"] > 0 and m["response_at_max_response_len"] > 0
        for t in r.trials:
            if t.samples:
                assert t.samples[-1]["status"] == "truncated"
                assert all(x["response_length"] < r.scenario.max_response_len for x in t.samples[:-1])
        assert m["lost"].get("compaction_segment_truncated", 0) <= 2  # a segment 0 longer than the window


def test_tcomp_overflow_runs_without_the_spec_checks_and_loses_no_trial(runs):
    m = runs("tcomp_overflow").metrics()
    assert "aborted" not in json.dumps(m["outcomes"]) and m["context_full_exits"] == 0


# ---------------------------------------------------------------------- trainer side
@pytest.mark.parametrize("name", ["compact_once", "compact3_truncated", "mixed_reload", "partial_abort",
                                  "tcomp_overflow", "r3_segment_window", "batch_count"])
def test_trainer_trains_each_trial_as_one_rollout(runs, name):
    check_trainer(runs(name))


def test_cli_prints_both_tables_and_writes_json(tmp_path, capsys):
    out = tmp_path / "m.json"
    assert sim.main(["--scenario", "no_trigger", "--json", str(out), "--workdir", str(tmp_path)]) == 0
    text = capsys.readouterr().out
    assert "| no_trigger |" in text and "compaction_batch lines" in text
    (metrics,) = json.loads(out.read_text())
    assert metrics["scenario"] == "no_trigger" and metrics["samples"] == 32
