"""Per-source train/rollout mismatch guard (yeto.rl.engine.xhw_guard) and its wiring:
buffer reader quarantine, downweight, round metadata, driver event."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.adapters.miles.rollout_meta_hook import record_round_metadata
from yeto.rl.engine import xhw_guard as xg
from yeto.rl.engine.driver import IslandDriver
from yeto.rl.core import StrictRlInvariantError
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer
from yeto.rl.engine.trainer_lifecycle import SampleIndexEntry

A = {"sampler_id": "nc-a4", "engine": "vllm", "engine_version": "vllm-kimi-k31@x", "card_model": "MI355X"}
B = {"sampler_id": "modal-s4", "engine": "sglang", "engine_version": "sglang@35708f76", "card_model": "B300"}
KA, KB = xg.source_key(A), xg.source_key(B)


def _metrics(abs_diff, kl, clip=0.004):
    return {"train_rollout_logprob_abs_diff": abs_diff, "train/train_rollout_kl": kl, "sc_is_clipped": clip}


# ---------------------------------------------------------------- keys / config
def test_source_key_names_sampler_engine_version_card():
    assert KA == "nc-a4|vllm|vllm-kimi-k31@x|MI355X"
    assert xg.source_key(None, "shard-1") == "shard-1|unknown|unknown|unknown"
    assert xg.source_key({"sampler_id": "unknown", "engine": "vllm"}, "s9").startswith("s9|vllm|")
    assert xg.source_fields(KA)["card_model"] == "MI355X"


def test_env_parsing():
    assert xg.XhwGuard.from_env({}) is None and xg.XhwGuard.from_env({xg.ENV: "off"}) is None
    g = xg.XhwGuard.from_env({xg.ENV: "on"})
    assert g.warn.abs_diff == 0.10 and g.act.kl == 0.06 and g.action == "quarantine"
    g = xg.XhwGuard.from_env({xg.ENV: "act_abs_diff=0.2,window=5,action=downweight,weight=0.25"})
    assert g.act.abs_diff == 0.2 and g.warn.abs_diff == 0.10 and g.window == 5 and g.weight == 0.25
    for bad in ("nope=1", "window", "action=recompute", "warn_kl=0.1,act_kl=0.05", "weight=0", "window=x"):
        with pytest.raises(xg.XhwGuardConfigError):
            xg.XhwGuard.from_env({xg.ENV: bad})


# ---------------------------------------------------------------- attribution
def test_single_source_step_is_exact():
    att = xg.SourceAttribution(window=3)
    att.add({KA: 1.0}, 0.077)
    e = att.estimate()[KA]
    assert e["value"] == pytest.approx(0.077) and e["exact"] == 1.0


def test_mixed_steps_are_solved_by_least_squares():
    att = xg.SourceAttribution(window=4)
    truth = {KA: 0.08, KB: 0.025}
    for share in (0.2, 0.5, 0.8):
        shares = {KA: share, KB: 1 - share}
        att.add(shares, sum(truth[s] * w for s, w in shares.items()))
    e = att.estimate()
    assert e[KA]["value"] == pytest.approx(0.08) and e[KB]["value"] == pytest.approx(0.025)
    assert e[KA]["exact"] == 0.0


def test_inseparable_mix_falls_back_to_the_mean_of_its_rows():
    att = xg.SourceAttribution(window=3)
    att.add({KA: 0.5, KB: 0.5}, 0.05)
    att.add({KA: 0.5, KB: 0.5}, 0.07)
    e = att.estimate()
    assert e[KA]["value"] == pytest.approx(0.06) and e[KB]["value"] == pytest.approx(0.06)


def test_window_drops_old_rows():
    att = xg.SourceAttribution(window=2)
    for v in (1.0, 0.1, 0.2):
        att.add({KA: 1.0}, v)
    assert att.estimate()[KA]["value"] == pytest.approx(0.15)


# ---------------------------------------------------------------- policy
def test_measured_mi355x_band_is_ok():
    g = xg.XhwGuard()
    for r, (d, k, c) in enumerate([(0.0773, 0.0153, 0.0040), (0.0772, 0.0166, 0.0047),
                                   (0.0989, 0.0205, 0.0055), (0.0756, 0.0161, 0.0043)]):
        ev = g.observe(r, {KA: 22000}, _metrics(d, k, c))
        assert ev["sources"][KA]["level"] == "ok" and ev["actions"] == {}


def test_warn_then_quarantine_then_probe():
    g = xg.XhwGuard(min_rounds=2, quarantine_rounds=3)
    ev = g.observe(0, {KA: 100, KB: 0}, _metrics(0.2, 0.01))
    assert ev["sources"][KA]["level"] == "warn"  # over act but only one round seen
    assert ev["actions"] == {}
    ev = g.observe(1, {KA: 100}, _metrics(0.2, 0.01))
    row = ev["sources"][KA]
    assert row["level"] == "act" and row["over"] == ["abs_diff"] and row["action"] == "quarantine"
    assert ev["actions"] == {KA: {"action": "quarantine", "until_round": 5}}
    ev = g.observe(2, {KB: 100}, _metrics(0.02, 0.001))  # A passed over; B healthy
    assert ev["sources"][KB]["level"] == "ok" and KA in ev["actions"]
    ev = g.observe(5, {KB: 100}, _metrics(0.02, 0.001))
    assert ev["actions"] == {}  # quarantine over: the next round probes A


def test_downweight_and_recovery():
    g = xg.XhwGuard(action="downweight", min_rounds=1, weight=0.25)
    ev = g.observe(0, {KA: 1}, _metrics(0.05, 0.1))
    assert ev["actions"] == {KA: {"action": "downweight", "weight": 0.25}}
    for r in (1, 2, 3):
        ev = g.observe(r, {KA: 1}, _metrics(0.03, 0.002))
    assert ev["actions"] == {}


def test_stop_raises():
    g = xg.XhwGuard(action="stop", min_rounds=1)
    with pytest.raises(xg.XhwGuardStop, match="clip"):
        g.observe(0, {KA: 1}, _metrics(0.05, 0.01, clip=0.2))


def test_mixed_round_blames_the_bad_source_only():
    g = xg.XhwGuard(min_rounds=2, window=4)
    truth = {KA: 0.30, KB: 0.025}
    for r, share in enumerate((0.3, 0.7)):
        shares = {KA: share * 1000, KB: (1 - share) * 1000}
        ev = g.observe(r, shares, _metrics(truth[KA] * share + truth[KB] * (1 - share), 0.001, 0.001))
    assert ev["sources"][KB]["level"] == "ok" and ev["sources"][KB]["abs_diff"] == pytest.approx(0.025)
    assert ev["actions"] == {KA: {"action": "quarantine", "until_round": 10}}
    ev = g.observe(2, {KB: 1000}, _metrics(0.025, 0.001, 0.001))  # the reader passes A over
    assert ev["sources"][KB]["level"] == "ok" and ev["sources"][KB]["abs_diff_exact"]


def test_missing_metrics_are_ignored():
    g = xg.XhwGuard()
    ev = g.observe(0, {KA: 1}, {"loss": 1.0})
    assert ev["sources"][KA]["level"] == "ok" and ev["step"]["abs_diff"] is None


# ---------------------------------------------------------------- rollout side
def test_rollout_stats():
    s = {"rollout_log_probs": [-0.5, -1.5, float("nan")],
         "rollout_top_logprobs": [[math.log(0.6), math.log(0.39), float("-inf")],
                                  [math.log(0.5), math.log(0.2), float("-inf")]]}
    st = xg.rollout_stats([s])
    assert st["tokens"] == 2 and st["nonfinite"] == 1 and st["sampled_lp_mean"] == pytest.approx(-1.0)
    assert st["head_mass_mean"] == pytest.approx((0.99 + 0.7) / 2) and st["low_head_mass_frac"] == 0.5
    assert math.isnan(xg.rollout_stats([])["head_mass_mean"])


def test_downweight_scales_centered_advantages_exactly():
    group = [{"reward": r, "metadata": {}} for r in (1.0, 0.0, 0.0, 1.0)]
    out = xg.downweight_group(group, 0.5)
    mean = 0.5
    assert [s["reward"] - mean for s in out] == [0.25, -0.25, -0.25, 0.25]
    assert out[0]["metadata"]["yeto_xhw_downweight"] == {"weight": 0.5, "reward": 1.0}
    assert group[0]["reward"] == 1.0  # input untouched
    assert xg.downweight_group([{"reward": None}], 0.5)[0]["reward"] is None


# ---------------------------------------------------------------- reader wiring
def _buffer(tmp_path, sources):
    buf = FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1)
    for i, prov in enumerate(sources):
        ref = buf.write_payload(f"s{i}", br.encode_group([{"tokens": [i], "reward": float(i % 2),
                                                          "rollout_log_probs": [-0.1]}]))
        buf.append(SampleIndexEntry(f"s{i}", 0, ref, tokens=10 + i, provenance=prov))
    return buf


def test_reader_reports_sources_and_passes_quarantined_groups_over(tmp_path):
    reader = br.BatchReader(_buffer(tmp_path, [A, B, A, B, B]), max_policy_age=4, prefetch=False,
                            read_workers=1)
    out, rep = reader.take(0, 2, current_version=0)
    assert rep.source_groups == {KB: 1, KA: 1} and rep.source_tokens == {KA: 10, KB: 11}
    assert rep.position_sources == {0: KA, 1: KB} and rep.quarantined_groups == 0
    reader.set_xhw_actions({KA: {"action": "quarantine", "until_round": 5}})
    out, rep = reader.take(1, 2, current_version=0)
    assert [p for p, _ in out] == [3, 4] and rep.quarantined_groups == 1
    assert rep.metrics()["sample_buffer/xhw_quarantined_groups"] == 1.0


def test_quarantine_ends_at_its_round(tmp_path):
    reader = br.BatchReader(_buffer(tmp_path, [A, A]), max_policy_age=4, prefetch=False, read_workers=1)
    reader.set_xhw_actions({KA: {"action": "quarantine", "until_round": 1}})
    out, rep = reader.take(1, 1, current_version=0)
    assert [p for p, _ in out] == [0] and rep.quarantined_groups == 0


def test_xhw_round_downweights_and_reports(tmp_path):
    reader = br.BatchReader(_buffer(tmp_path, [A, B]), max_policy_age=4, prefetch=False, read_workers=1)
    groups, rep = reader.take(0, 2, current_version=0)
    groups = [(0, [{"reward": 1.0}, {"reward": 0.0}]), (1, [{"reward": 1.0}, {"reward": 0.0}])]
    out, meta = br.xhw_round(groups, rep, {KA: {"action": "downweight", "weight": 0.5}}, std_normalized=False)
    assert [s["reward"] for s in out[0][1]] == [0.75, 0.25] and [s["reward"] for s in out[1][1]] == [1.0, 0.0]
    assert meta["buffer_source_tokens"] == {KA: 10, KB: 11} and meta["buffer_xhw_downweighted"] == 1
    assert meta["buffer_source_stats"][KA]["groups"] == 1
    with pytest.raises(br.SampleBufferRolloutError, match="std normalization"):
        br.xhw_round(groups, rep, {KA: {"action": "downweight", "weight": 0.5}}, std_normalized=True)
    assert br.xhw_round(groups, rep, {}, std_normalized=True, enabled=False) == (groups, {})


def test_round_metadata_accepts_the_source_maps(tmp_path):
    sink = f"dir:{tmp_path}"
    record_round_metadata(None, 3, sink=sink, buffer_source_tokens={KA: 10}, buffer_xhw_quarantined=1,
                          buffer_source_stats={KA: {"tokens": 1.0}}, buffer_xhw_downweighted=0)
    with pytest.raises(RuntimeError, match="mapping"):
        record_round_metadata(None, 3, sink=sink, buffer_source_tokens=[1])


def test_source_fields_reach_the_handle(tmp_path):
    import json
    import os

    from tests.test_rl_miles_adapter_rollout import H, group
    from yeto.rl.adapters.miles import rollout_meta_hook as hook
    from yeto.rl.adapters.miles.rollout import merge_round_metadata, handle_from_metadata

    kept = [group(0, [1.0, 0.0])]
    args = SimpleNamespace()
    hook.record_trained_groups(args, kept)
    os.environ[hook.META_SINK_ENV] = f"dir:{tmp_path}"
    try:
        hook.extract_rollout_metadata(args, kept, SimpleNamespace(sample_group_index=6))
    finally:
        os.environ.pop(hook.META_SINK_ENV)
    payload = json.loads((tmp_path / "rollout-3.json").read_text())
    extra = {"schema": hook.ROUND_META_SCHEMA, "rollout_id": 3, "buffer_stale_groups": 0,
             "buffer_source_tokens": {KA: 7}, "buffer_source_stats": {KA: {"tokens": 7}},
             "buffer_xhw_quarantined": 2, "buffer_xhw_downweighted": 1}
    h = handle_from_metadata(merge_round_metadata(payload, [extra], 3), rollout_id=3, policy_version=3,
                             policy_hash=H, data_pack=None)
    assert h.buffer_source_tokens == {KA: 7} and h.buffer_source_stats == {KA: {"tokens": 7.0}}
    assert h.buffer_xhw_quarantined == 2 and h.buffer_xhw_downweighted == 1
    plain = handle_from_metadata(payload, rollout_id=3, policy_version=3, policy_hash=H, data_pack=None)
    assert plain.buffer_source_tokens is None and plain.buffer_xhw_quarantined is None


# ---------------------------------------------------------------- driver
def _driver(guard, sink=None):
    events = []
    d = SimpleNamespace(xhw_guard=guard, xhw_actions_sink=sink,
                        emit=lambda name, **f: events.append((name, f)))
    return d, events


def test_driver_emits_and_hands_actions_to_the_reader():
    got = []
    d, events = _driver(xg.XhwGuard(min_rounds=1), got.append)
    batch = SimpleNamespace(buffer_source_tokens={KA: 5}, buffer_source_stats={KA: {"tokens": 5.0}},
                            buffer_xhw_quarantined=0, buffer_xhw_downweighted=0)
    IslandDriver._guard_sources(d, 0, batch, _metrics(0.5, 0.01))
    name, fields = events[0]
    assert name == "rl_xhw_mismatch" and fields["sources"][KA]["level"] == "act"
    assert fields["rollout_stats"] == {KA: {"tokens": 5.0}}
    assert got == [{KA: {"action": "quarantine", "until_round": 9}}]


def test_driver_without_guard_or_sources_does_nothing():
    d, events = _driver(None)
    IslandDriver._guard_sources(d, 0, SimpleNamespace(buffer_source_tokens={KA: 1}), {})
    d, events2 = _driver(xg.XhwGuard())
    IslandDriver._guard_sources(d, 0, SimpleNamespace(), {})
    assert events == events2 == []


def test_driver_stop_is_a_strict_invariant():
    d, events = _driver(xg.XhwGuard(action="stop", min_rounds=1))
    with pytest.raises(StrictRlInvariantError):
        IslandDriver._guard_sources(d, 0, SimpleNamespace(buffer_source_tokens={KA: 1}), _metrics(0.5, 0.5))
    assert events[0][0] == "rl_xhw_mismatch" and "stop" in events[0][1]
