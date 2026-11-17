"""CompactionRL metrics: sampler round summary, trainer step summary, W&B, dashboard.

The buffer samples are built by hand with the contract's ``metadata.compaction``
(schema 1), until the segment writer lands; a sample without that key is a whole
trial. Without compaction data every reader keeps its old field set.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import threading

from test_agentic_round_streaming import Harness, Prompts, Recorder, _round

from yeto.dashboard.alerts import merged_thresholds
from yeto.dashboard.export import export_view, render_page
from yeto.dashboard.reducer import Reducer
from yeto.rl.engine import agentic_round
from yeto.rl.engine.agentic_round import DRAIN_EVENT, LOST_DRAIN
from yeto.rl.engine.engine_sampler import ROUND_EVENT, PromptItem
from yeto.rl.engine.timeline import (COMPACTION_COUNT_KEYS, CompactionTally, compaction_log_line,
                                     compaction_stats, drained_fields, live_segments, segment_bucket,
                                     trial_segments)
from yeto.rl.wandb_rl import event_metrics


def seg(tid, i, n, *, ok=None, mask=(1, 1, 0, 1), status="completed", summary_tokens=None, gi=0, index=0, **extra):
    comp = {"schema": 1, "trajectory_id": tid, "segment_index": i, "num_segments": n, "context_budget": 4096,
            "summary_ok": ok, "kept_steps": 2, "segment_tokens": sum(mask), "tokens_after": 0, **extra}
    if summary_tokens is not None:
        comp["summary_tokens"] = summary_tokens
    return {"group_index": gi, "index": index, "reward": 1.0, "status": status, "loss_mask": list(mask),
            "metadata": {"trajectory_id": tid, "compaction": comp}}


def whole(tid, *, mask=(1, 1), status="completed", gi=0, index=0):
    return {"group_index": gi, "index": index, "reward": 0.0, "status": status, "loss_mask": list(mask),
            "metadata": {"trajectory_id": tid}}


def round_samples():
    """Trial a: 3 segments (summaries ok, failed; ends truncated); b: 2 segments; c: not compacted."""
    return [seg("a", 0, 3, ok=True, summary_tokens=1, index=0), seg("a", 1, 3, ok=False, summary_tokens=1, index=0),
            seg("a", 2, 3, status="truncated", index=0),
            seg("b", 0, 2, ok=True, summary_tokens=2, index=1), seg("b", 1, 2, index=1),
            whole("c", index=2)]


def test_round_summary_counts_trials_segments_summaries_and_truncation():
    s = compaction_stats(round_samples(), max_compactions=2)
    assert (s["trials"], s["samples"], s["compacted_trials"], s["compactions"]) == (3, 6, 2, 3)
    assert s["compacted_frac"] == pytest.approx(0.6667) and s["compactions_mean"] == 1.0
    assert s["segments_hist"] == {"1": 1, "2": 1, "3": 1} and s["compactions_max"] == 2
    assert s["samples_per_trial"] == 2.0 and s["segment_samples"] == 5
    assert s["segment_sample_frac"] == pytest.approx(0.8333)
    assert (s["summaries"], s["summary_ok"], s["summary_ok_frac"]) == (3, 2, pytest.approx(0.6667))
    assert (s["truncated_trials"], s["truncated_compacted_trials"], s["truncated_compacted_frac"]) == (1, 1, 0.5)
    assert (s["exhausted_trials"], s["exhausted_truncated_trials"], s["max_compactions"]) == (1, 1, 2)
    # 5 segments x 3 mask tokens + 2 = 17 trainable tokens, 4 of them summary tokens
    assert (s["trainable_tokens"], s["summary_tokens"]) == (17, 4)
    assert s["summary_token_frac"] == pytest.approx(4 / 17, abs=1e-4)
    assert s["context_full_trials"] is None and s["schema"] == 1
    assert "summary_ok=66.7%" in compaction_log_line(s)


def test_segments_arriving_out_of_order_take_the_last_segments_status():
    samples = round_samples()
    s = compaction_stats([samples[2], samples[0], samples[1]])
    assert s["truncated_compacted_trials"] == 1 and s["trials"] == 1
    assert s["exhausted_trials"] is None and s["max_compactions"] is None  # limit unknown


def test_no_compaction_data_gives_nothing_unless_compaction_is_on():
    plain = [whole("x"), whole("y", status="truncated")]
    assert compaction_stats(plain) is None
    assert compaction_stats([]) is None and compaction_stats([], enabled=True) is None
    on = compaction_stats(plain, enabled=True)
    assert (on["trials"], on["compacted_trials"], on["compacted_frac"]) == (2, 0, 0.0)
    assert on["summary_ok_frac"] is None and on["truncated_compacted_frac"] is None
    assert on["summary_tokens"] is None and on["summary_token_frac"] is None


def test_records_counters_add_compactions_and_full_contexts():
    # a compaction whose next segment never sampled: one segment sample, one summary turn
    samples = [seg("a", 0, 1, ok=True)]
    s = compaction_stats(samples, counters={"a": {"compactions": 1, "context_full": 1}})
    assert (s["compacted_trials"], s["compactions"], s["context_full_trials"]) == (1, 1, 1)
    # counters alone mark compaction data as seen
    assert compaction_stats([whole("z")], counters={"z": {"compactions": 2}})["compactions"] == 2


def test_trainer_side_reads_miles_samples():
    status = SimpleNamespace(value="truncated")
    miles = [SimpleNamespace(group_index=0, index=0, status=status, loss_mask=[1, 1],
                             metadata={"compaction": {"trajectory_id": "t0", "segment_index": i,
                                                      "num_segments": 2, "summary_ok": i == 0 or None}})
             for i in range(2)]
    miles += [SimpleNamespace(group_index=0, index=j, status=SimpleNamespace(value="completed"), loss_mask=[1],
                              metadata={}) for j in (1, 2)]  # no trajectory id: (group_index, index)
    s = compaction_stats(miles)
    assert (s["samples"], s["trials"], s["samples_per_trial"]) == (4, 3, pytest.approx(1.3333))
    assert (s["segment_samples"], s["segment_sample_frac"]) == (2, 0.5)
    assert s["truncated_compacted_trials"] == 1 and s["summary_ok_frac"] == 1.0


def test_tally_accumulates_across_groups():
    tally = CompactionTally()
    samples = round_samples()
    tally.add(samples[:3])
    tally.add(samples[3:])
    assert tally.summary() == compaction_stats(samples)
    assert set(COMPACTION_COUNT_KEYS) <= set(tally.summary())


def test_trials_are_bucketed_by_segment_count_written_lost_and_dropped():
    tally = CompactionTally()
    tally.add(round_samples())  # written: a (3 segments), b (2), c (1)
    tally.lose(SimpleNamespace(num_segments=4), "drain_timeout")
    tally.lose(SimpleNamespace(num_segments=5), "drain_timeout")
    tally.lose(SimpleNamespace(segments=[1, 2]), "served_by_other_version")
    tally.lose(SimpleNamespace(reason="drain_timeout"), "drain_timeout")  # cancelled: count unknown
    tally.drop([SimpleNamespace(segments=object())], "below_min_samples")  # one joined stream
    by = tally.summary()["by_segments"]
    assert by["1"] == {"written": 1, "lost": {}, "dropped": {"below_min_samples": 1}, "not_written_frac": 0.5}
    assert by["2"]["lost"] == {"served_by_other_version": 1} and by["2"]["not_written_frac"] == 0.5
    assert by["3"] == {"written": 1, "lost": {}, "dropped": {}, "not_written_frac": 0.0}
    assert by["4+"] == {"written": 0, "lost": {"drain_timeout": 2}, "dropped": {}, "not_written_frac": 1.0}
    assert by["unknown"]["lost"] == {"drain_timeout": 1}
    assert [segment_bucket(n) for n in (None, -1, 0, 1, 3, 4, 9)] == ["unknown", "unknown", "0", "1", "3", "4+", "4+"]
    assert trial_segments(SimpleNamespace(segments=None)) is None


def test_lost_trials_alone_report_when_compaction_is_on_and_only_then():
    tally = CompactionTally()
    tally.lose(SimpleNamespace(segments=object()), "drain_timeout")  # an uncompacted trial
    assert tally.summary() is None
    on = tally.summary(enabled=True)
    assert on["trials"] == 0 and on["by_segments"]["1"]["lost"] == {"drain_timeout": 1}
    tally.lose(SimpleNamespace(num_segments=3), "drain_timeout")  # a lost compacted trial is data
    assert tally.summary()["by_segments"]["3"]["not_written_frac"] == 1.0


def fork_lost():
    """A trial the gateway invalidated for a fork of a compacted session, as finish_sample aborts it."""
    why = ("records proxy invalidated the trajectory: compaction_fork_lost: retry_fork at message 7 "
           "of chain c1, whose session has compacted")
    return SimpleNamespace(trajectory=SimpleNamespace(reason=f"token_alignment: {why}"), segments=None,
                           loss={"stage": "recorder_close", "error": f"TokenAlignmentError: {why}"})


def test_gateway_fork_losses_and_retry_replays_are_counted():
    samples = round_samples()  # 2 compacted trials written
    samples[3]["metadata"]["compaction"]["retry_replays"] = 2  # trial b, carried by both its segments
    samples[4]["metadata"]["compaction"]["retry_replays"] = 2
    tally = CompactionTally()
    tally.add(samples, counters={"a": {"retry_replays": 1}})
    tally.lose(fork_lost(), "token_alignment")
    tally.lose(SimpleNamespace(trajectory=SimpleNamespace(reason="token_alignment: other"), segments=None,
                               loss=None), "token_alignment")
    s = tally.summary()
    assert (s["fork_lost_trials"], s["compaction_exposed_trials"]) == (1, 3)
    assert s["fork_lost_frac"] == pytest.approx(0.3333)
    assert s["by_segments"]["unknown"]["lost"] == {"compaction_fork_lost": 1, "token_alignment": 1}
    assert (s["retry_replays"], s["retry_replay_trials"]) == (3, 2)
    plain = compaction_stats(round_samples())
    assert plain["fork_lost_trials"] == 0 and plain["retry_replays"] is None and plain["fork_lost_frac"] == 0.0
    alone = CompactionTally()
    alone.lose(fork_lost(), "token_alignment")  # a fork loss alone is compaction data
    assert alone.summary()["fork_lost_frac"] == 1.0


def test_reward_and_tokens_per_trial_differ_from_per_segment():
    samples = [dict(x, reward=1.0) for x in round_samples()[:3]] + [whole("c", index=2)]  # a: 3 segments, c: 1
    s = compaction_stats(samples)
    assert s["reward_mean_trial"] == 0.5 and s["reward_mean_segment"] == 0.75
    assert s["policy_tokens_mean_trial"] == 5.5 and s["policy_tokens_mean_segment"] == 2.75


def test_xhw_downweight_moves_the_trial_mean_of_a_segmented_group():
    def down(x, before, after):
        return dict(x, reward=after, metadata={**x["metadata"], "yeto_xhw_downweight": {"weight": 0.5,
                                                                                       "reward": before}})
    # trial a (2 segments) reward 1, trial c (1 segment) reward 0: sample mean 2/3, w = 0.5
    mean = 2 / 3
    after = [mean + 0.5 * (r - mean) for r in (1.0, 1.0, 0.0)]
    group = [down(seg("a", 0, 2, ok=True), 1.0, after[0]), down(seg("a", 1, 2), 1.0, after[1]),
             down(whole("c", index=1), 0.0, after[2])]
    s = compaction_stats(group)
    assert (s["xhw_downweighted_groups"], s["xhw_downweighted_trials"], s["xhw_downweighted_samples"]) == (1, 2, 3)
    # trial mean before 0.5, after (5/6 + 1/3) / 2 = 7/12
    assert s["xhw_trial_mean_shift"] == pytest.approx(1 / 12, abs=1e-6)
    assert "xhw_trial_mean_shift" not in compaction_stats(round_samples())


# ---------------------------------------------------------------- the sampler round event
class CompactingRecorder(Recorder):
    """A recorder whose records proxy compacts (``proxy.backend.compaction``); while
    a trial runs its backend session holds 3 segments for slot 0, 1 for the others."""

    def __init__(self, engine) -> None:
        super().__init__(engine)
        self.proxy = SimpleNamespace(backend=SimpleNamespace(compaction=SimpleNamespace(max_compactions=3),
                                                             sessions={}))

    async def open(self, trajectory_id, sampling_params):
        segments = [0, 1, 2] if trajectory_id.endswith("-0") else [0]
        self.proxy.backend.sessions[f"rec-{trajectory_id}"] = SimpleNamespace(trajectory_id=trajectory_id,
                                                                              segments=segments)
        return await super().open(trajectory_id, sampling_params)

    async def close(self, trajectory_id):
        self.proxy.backend.sessions.pop(f"rec-{trajectory_id}", None)  # forgotten at close
        return await super().close(trajectory_id)


def test_live_segments_reads_the_open_records_sessions(tmp_path):
    rec = CompactingRecorder(None)
    rec.proxy.backend.sessions["x"] = SimpleNamespace(trajectory_id="t", segments=[0, 1])
    rec.proxy.backend.sessions["y"] = SimpleNamespace(trajectory_id="t", segments=[0])  # a fork
    assert live_segments(rec, "t") == 2 and live_segments(rec, "u") is None
    assert live_segments(Recorder(None), "t") is None
    assert drained_fields({"1": 2, "3": 1, "4+": 1, "0": 2}) == {
        "drained_by_segments": {"0": 2, "1": 2, "3": 1, "4+": 1}, "drained_compacted_trials": 2,
        "drained_uncompacted_trials": 2}


def _drained(tmp_path, recorder=None):
    harness = Harness({"b": 0.0})
    harness.gates["a"] = threading.Event()  # never set: only the drain timeout ends it
    pending = {"seq": None}
    rnd, engine, events = _round(tmp_path, harness, Prompts({0: ["a"], 1: ["b"]}), drain_timeout_s=0.05)
    if recorder is not None:
        rnd.recorder = recorder(engine)
    rnd.release_pending = lambda: pending["seq"]
    try:
        rnd(0)
        pending["seq"] = 3
        rnd(1)
        rnd.flush()
    finally:
        rnd.close()
    return events


def test_a_drain_counts_the_trials_it_cut_by_segment_count(tmp_path):
    events = _drained(tmp_path, CompactingRecorder)
    (drain,) = [f for n, f in events if n == DRAIN_EVENT]
    assert drain["abandoned_trials"] == 2 and drain["drained_by_segments"] == {"1": 1, "3": 1}
    assert (drain["drained_compacted_trials"], drain["drained_uncompacted_trials"]) == (1, 1)
    fate = [f["agentic"]["compaction"]["by_segments"] for n, f in events
            if n == ROUND_EVENT and "compaction" in f["agentic"]]
    lost = {b: v["lost"] for by in fate for b, v in by.items() if v["lost"]}
    assert lost == {"1": {LOST_DRAIN: 1}, "3": {LOST_DRAIN: 1}}


def test_a_drain_event_keeps_its_fields_without_compaction(tmp_path):
    (drain,) = [f for n, f in _drained(tmp_path) if n == DRAIN_EVENT]
    assert not any(k.startswith("drained_") for k in drain)


def _run(tmp_path, **kw):
    rnd, engine, events = _round(tmp_path, Harness({"p": 0.0}), Prompts({0: ["p"]}), **kw)
    try:
        rnd(0)
        rnd.flush()
    finally:
        rnd.close()
    return [f["agentic"] for n, f in events if n == ROUND_EVENT]


def test_the_round_event_has_no_compaction_key_by_default(tmp_path):
    rounds = _run(tmp_path)
    assert rounds and all("compaction" not in a for a in rounds)


def test_the_round_event_carries_the_written_segments(tmp_path, monkeypatch):
    real = agentic_round.agentic_sample

    def segmented(item, finished, **kw):  # the second trial is the last of two segments
        sample = real(item, finished, **kw)
        if kw["index"] % 2:
            sample["metadata"]["compaction"] = {"schema": 1, "trajectory_id": sample["metadata"]["trajectory_id"],
                                                "segment_index": 1, "num_segments": 2, "summary_ok": None}
        return sample

    monkeypatch.setattr(agentic_round, "agentic_sample", segmented)
    rounds = [a for a in _run(tmp_path) if "compaction" in a]
    comp = [a["compaction"] for a in rounds]
    assert len(comp) == 1 and (comp[0]["trials"], comp[0]["compacted_trials"]) == (2, 1)
    assert comp[0]["segments_hist"] == {"1": 1, "2": 1} and comp[0]["max_compactions"] is None
    assert comp[0]["by_segments"]["2"]["written"] == 1
    assert rounds[0]["reward_mean"] == comp[0]["reward_mean_trial"] == 1.0
    assert rounds[0]["policy_tokens_mean"] == comp[0]["policy_tokens_mean_trial"]


def test_a_compacting_proxy_reports_rounds_where_no_trial_compacted(tmp_path):
    rnd, engine, events = _round(tmp_path, Harness({"p": 0.0}), Prompts({0: ["p"]}))
    rnd.recorder = CompactingRecorder(engine)
    try:
        rnd(0)
        rnd.flush()
    finally:
        rnd.close()
    comp = [f["agentic"]["compaction"] for n, f in events if n == ROUND_EVENT and "compaction" in f["agentic"]]
    assert len(comp) == 1 and (comp[0]["trials"], comp[0]["compacted_trials"]) == (2, 0)
    assert comp[0]["max_compactions"] == 3 and comp[0]["exhausted_trials"] == 0
    assert comp[0]["by_segments"] == {"1": {"written": 2, "lost": {}, "dropped": {}, "not_written_frac": 0.0}}


# ---------------------------------------------------------------- W&B
def test_wandb_projects_the_trainer_step_compaction_summary():
    summary = compaction_stats(round_samples())
    m = event_metrics({"event": "rl_round_trained", "rollout_id": 4, "train_step": 5, "compaction": summary})
    assert m["train/step"] == 5
    assert m["train/compaction/samples_per_trial"] == 2.0 and m["train/compaction/trials"] == 3
    assert m["train/compaction/summary_ok_frac"] == pytest.approx(0.6667)
    assert not any("segments_hist" in k or "schema" in k for k in m)
    plain = event_metrics({"event": "rl_round_trained", "rollout_id": 4, "train_step": 5,
                           "train_metrics": {"loss": 1.0}})
    assert not any(k.startswith("train/compaction/") for k in plain)


# ---------------------------------------------------------------- dashboard
T0 = 1_790_000_000.0


def _samples(rid, *, ok=True, truncated=False):
    """Four trials of two segments (summary in the first) and one whole trial."""
    return [seg(f"{rid}-{j}", i, 2, ok=ok if i == 0 else None, index=j,
                status="truncated" if truncated and i == 1 else "completed")
            for j in range(4) for i in range(2)] + [whole(f"{rid}-w", index=9)]


def _tapes(rounds, *, compaction=True, ok=True, truncated=False, lost_long=0, forks=0):
    recs = [{"event": "rl_sampler_start", "sampler_id": "nc-c1", "max_backlog_groups": 8, "time_unix": T0}]
    for rid in range(rounds):
        tally = CompactionTally()
        tally.add(_samples(rid, ok=ok, truncated=truncated))
        for _ in range(lost_long):
            tally.lose(SimpleNamespace(num_segments=3), "drain_timeout")
        for _ in range(forks):
            tally.lose(fork_lost(), "token_alignment")
        stats = tally.summary(max_compactions=1)
        agentic = {"retried_trials": 0, "lost_trials": 0, **({"compaction": stats} if compaction else {})}
        t = T0 + 10 + 30 * rid
        recs += [{"event": "rl_sampler_engine_round", "sampler_id": "nc-c1", "rollout_id": rid, "groups": 1,
                  "samples": 9, "response_tokens": 90, "finish": {"stop": 9}, "time_unix": t, "agentic": agentic},
                 {"event": "rl_sampler_round", "sampler_id": "nc-c1", "rollout_id": rid, "groups": 1,
                  "generate_s": 20.0, "time_unix": t}]
    if compaction:
        recs.append({"event": "rl_agentic_drain", "sampler_id": "nc-c1", "seq": 1, "time_unix": T0 + 5,
                     **drained_fields({"1": 2, "3": 1})})
    recs += [{"event": "rl_driver_start", "island_id": 0, "time_unix": T0},
             {"event": "rl_round_trained", "island_id": 0, "rollout_id": 0, "trained_groups": 1,
              "trained_samples": 9, "train_step": 1, "time_unix": T0 + 50,
              **({"compaction": compaction_stats(_samples(0))} if compaction else {})},
             {"event": "rl_buffer_stale_skipped", "island_id": 0, "rollout_id": 0, "stale_groups": 0,
              "time_unix": T0 + 40, **({"stale_by_segments": {"2": 4}} if compaction else {})}]
    r = Reducer(run="cmp-t")
    for rec in sorted(recs, key=lambda x: x["time_unix"]):
        r.feed(rec)
    return export_view(r, generated_at=T0 + 500)


def test_dashboard_sums_sampler_and_trainer_compaction():
    view = _tapes(3)
    s = view["page"]["async"]["samplers"]["nc-c1"]["compaction"]
    assert (s["rounds"], s["trials"], s["samples"], s["compacted_trials"]) == (3, 15, 27, 12)
    assert s["compacted_frac"] == 0.8 and s["summary_ok_frac"] == 1.0
    assert s["segments_hist"] == {"1": 3, "2": 12} and s["exhausted_trials"] == 12
    rows = view["page"]["async"]["samplers"]["nc-c1"]["rounds"]
    assert all(r["compaction"]["trials"] == 5 for r in rows)
    t = view["page"]["async"]["trainers"]["0"]["compaction"]
    assert (t["samples"], t["trials"], t["segment_samples"]) == (9, 5, 8)
    assert t["samples_per_trial"] == 1.8
    assert s["by_segments"]["2"] == {"written": 12.0, "lost": {}, "dropped": {}, "stale": None,
                                     "not_written_frac": 0.0}
    assert s["reward_mean_trial"] == 0.8 and s["reward_mean_segment"] == pytest.approx(0.8889, abs=1e-4)
    assert view["page"]["async"]["samplers"]["nc-c1"]["compaction"]["drained_by_segments"] == {"1": 2, "3": 1}
    assert t["by_segments"]["2"]["stale"] == 4 and t["by_segments"]["2"]["written"] == 4
    assert 'id="aCompBox"' in render_page(view)
    rules = {a["rule"]: a["sev"] for a in view["overview"]["alerts"]}
    assert rules.get("compaction_frac") == 2 and "compaction_summary" not in rules
    assert "compaction_truncated" not in rules and "compaction_long_lost" not in rules
    assert "compaction_fork_lost" not in rules


def test_dashboard_alerts_on_trials_lost_to_forks_after_compaction():
    view = _tapes(3, forks=1)
    c = view["page"]["async"]["samplers"]["nc-c1"]["compaction"]
    assert (c["fork_lost_trials"], c["compaction_exposed_trials"]) == (3, 15) and c["fork_lost_frac"] == 0.2
    rules = {a["rule"]: a for a in view["overview"]["alerts"]}
    assert rules["compaction_fork_lost"]["sev"] == 1 and "20%" in rules["compaction_fork_lost"]["detail"]


def test_dashboard_alerts_when_long_trials_are_lost_more_often():
    view = _tapes(3, lost_long=4)
    fate = view["page"]["async"]["samplers"]["nc-c1"]["compaction"]["by_segments"]
    assert fate["3"]["lost"] == {"drain_timeout": 12} and fate["3"]["not_written_frac"] == 1.0
    rules = {a["rule"]: a for a in view["overview"]["alerts"]}
    assert rules["compaction_long_lost"]["sev"] == 1 and "drain_timeout_s" in rules["compaction_long_lost"]["detail"]


def test_dashboard_alerts_on_failing_summaries_and_truncation():
    rules = {a["rule"]: a for a in _tapes(3, ok=False, truncated=True)["overview"]["alerts"]}
    assert rules["compaction_summary"]["sev"] == 0 and rules["compaction_summary"]["island"] == "nc-c1"
    assert rules["compaction_truncated"]["sev"] == 1 and "100%" in rules["compaction_truncated"]["detail"]


def test_dashboard_without_compaction_keeps_its_keys():
    view = _tapes(2, compaction=False)
    async_view = view["page"]["async"]
    assert "compaction" not in async_view["samplers"]["nc-c1"]
    assert all("compaction" not in r for r in async_view["samplers"]["nc-c1"]["rounds"])
    assert "compaction" not in async_view["trainers"]["0"]
    assert not any(a["rule"].startswith("compaction") for a in view["overview"]["alerts"])


def test_compaction_thresholds_are_overridable():
    th = merged_thresholds({"compaction_summary_ok_warn": 0.5})
    assert th["compaction_summary_ok_warn"] == 0.5 and th["compaction_truncated_warn"] == 0.2


# ---------------------------------------------------------------- driver: trials, not segment samples
def _main_pooled_reward(batch):  # the per-sample formula, the reference without compaction
    import math

    total = sum(len(g.sample_ids) for g in batch.groups)
    if total == 0:
        return 0.0, 0.0
    mean = sum(g.reward_mean * len(g.sample_ids) for g in batch.groups) / total
    second = sum((g.reward_std ** 2 + g.reward_mean ** 2) * len(g.sample_ids) for g in batch.groups) / total
    return mean, math.sqrt(max(0.0, second - mean * mean))


def _group(ids, mean, std):
    return SimpleNamespace(sample_ids=tuple(ids), reward_mean=mean, reward_std=std)


def test_driver_counts_without_compaction_equal_main():
    from yeto.rl.engine.driver import _group_trials, _pooled_reward

    batch = SimpleNamespace(groups=[_group(["s0", "s1", "s2", "s3"], 0.25, 0.433), _group(["s4", "s5"], 1.0, 0.0),
                                    _group(["s6", "s7", "s8"], 0.1234567, 0.3)])
    assert _pooled_reward(batch) == _main_pooled_reward(batch)  # bit for bit
    assert sum(_group_trials(g) for g in batch.groups) == sum(len(g.sample_ids) for g in batch.groups) == 9
    empty = SimpleNamespace(groups=[])
    assert _pooled_reward(empty) == _main_pooled_reward(empty) == (0.0, 0.0)


def test_driver_counts_segments_of_one_trial_once():
    from yeto.rl.engine.driver import _group_trials, _pooled_reward

    # group 0: trial s0 has 3 segments, s1 one; group 1: two whole trials
    batch = SimpleNamespace(groups=[_group(["s0", "s0", "s0", "s1"], 0.5, 0.5), _group(["s2", "s3"], 1.0, 0.0)])
    assert [_group_trials(g) for g in batch.groups] == [2, 2]
    mean, std = _pooled_reward(batch)
    # per trial: second moment ((0.25 + 0.25) * 2 + 1 * 2) / 4 = 0.75, variance 0.75 - 0.75 ** 2
    assert mean == 0.75 and std == pytest.approx(0.1875 ** 0.5, abs=1e-12)
    assert _main_pooled_reward(batch)[0] == pytest.approx(2 / 3)  # the per-sample formula weights segments


# ---------------------------------------------------------------- the segment writer's FinishedSample
ITEM = PromptItem("p0", "task", metadata={"agent_task": {"instance_id": "fix-git", "agent_name": "codex"}})


def _compacted_finished(parts_spec, *, session="t-0", reward=1.0):
    """A compacted trial as ``finish_sample`` returns it: one ``CompactionSegment`` per
    (trained tokens, summary_ok, summary_tokens, retry_replays)."""
    from yeto.rl.agentic import AgentTask, AgentTrajectory, Outcome, TrialVerdict
    from yeto.rl.agentic.finish import CompactionSegment, FinishedSample
    from yeto.rl.agentic.types import TokenSegments

    task = AgentTask(task_id="fix-git", agent="codex", provider="docker", trajectory_id=session)
    trajectory = AgentTrajectory(task=task, outcome=Outcome.OK, reward=reward, reason="verified",
                                 verdict=TrialVerdict(rewards={"reward": reward}))
    parts = tuple(CompactionSegment(turns=(), summary_ok=ok, context_budget=4096, summary_tokens=summary,
                                    retry_replays=replays,
                                    tokens=TokenSegments(prompt_ids=(1, 2), response_ids=tuple(range(n)),
                                                         response_mask=(1,) * n, response_logprobs=(0.0,) * n))
                  for n, ok, summary, replays in parts_spec)
    return FinishedSample(trajectory=trajectory, segments=parts[-1].tokens, turns=(), parts=parts,
                          compacting=True)


def test_trial_segments_counts_the_parts_of_a_compacted_trial():
    # a FinishedSample has no num_segments and its ``segments`` is the last part's
    # TokenSegments, so the count must come from ``parts`` (``segments`` alone reads 1)
    from yeto.rl.agentic.finish import FinishedSample

    finished = _compacted_finished([(4, True, 2, None), (4, True, 3, None), (3, None, 0, None)])
    assert trial_segments(finished) == 3
    whole = FinishedSample(trajectory=finished.trajectory, segments=finished.segments, turns=())
    assert trial_segments(whole) == 1  # one stream: parts empty
    tally = CompactionTally()
    tally.drop([finished, whole], "below_min_samples")
    tally.lose(finished, "served_by_other_version")
    by = tally.summary()["by_segments"]
    assert by["3"]["dropped"] == {"below_min_samples": 1} and by["3"]["lost"] == {"served_by_other_version": 1}
    assert by["1"]["dropped"] == {"below_min_samples": 1}


def test_the_segment_writers_samples_give_summary_tokens_and_retry_replays():
    # agentic_samples writes summary_tokens / retry_replays per segment
    finished = _compacted_finished([(4, True, 2, 1), (5, False, 3, 1), (3, None, 0, 1)])
    samples = agentic_round.agentic_samples(ITEM, finished, version=0, provenance={}, group_index=0, index=0)
    assert [s["metadata"]["compaction"]["summary_tokens"] for s in samples] == [2, 3, 0]
    s = compaction_stats(samples)
    assert (s["trials"], s["samples"], s["compacted_trials"], s["compactions"]) == (1, 3, 1, 2)
    assert (s["trainable_tokens"], s["summary_tokens"]) == (12, 5)
    assert s["summary_token_frac"] == pytest.approx(5 / 12, abs=1e-4)
    assert (s["retry_replays"], s["retry_replay_trials"]) == (1, 1)  # one trial's count, not one per segment
    assert (s["summaries"], s["summary_ok_frac"]) == (2, 0.5)
    off = _compacted_finished([(4, True, 2, None), (3, None, 0, None)], session="t-1")  # replay off
    s = compaction_stats(agentic_round.agentic_samples(ITEM, off, version=0, provenance={}, group_index=0, index=1))
    assert s["retry_replays"] is None and s["summary_tokens"] == 2
