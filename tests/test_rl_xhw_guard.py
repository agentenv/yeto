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
    g.observe(0, {KB: 100}, _metrics(0.02, 0.001))
    ev = g.observe(1, {KA: 100, KB: 0}, _metrics(0.2, 0.01))
    assert ev["sources"][KA]["level"] == "warn"  # over act but only one round seen
    assert ev["actions"] == {}
    # B feeds this round too (A's estimate stays exact from its single-source round)
    ev = g.observe(2, {KA: 500, KB: 500}, _metrics(0.11, 0.0055))
    row = ev["sources"][KA]
    assert row["level"] == "act" and row["over"] == ["abs_diff"] and row["action"] == "quarantine"
    assert ev["actions"] == {KA: {"action": "quarantine", "until_round": 6}}
    ev = g.observe(3, {KB: 100}, _metrics(0.02, 0.001))  # A passed over; B healthy
    assert ev["sources"][KB]["level"] == "ok" and KA in ev["actions"]
    ev = g.observe(6, {KB: 100}, _metrics(0.02, 0.001))
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
    for r, share in enumerate((0.3, 0.7, 0.5, 0.4)):
        shares = {KA: share * 1000, KB: (1 - share) * 1000}
        ev = g.observe(r, shares, _metrics(truth[KA] * share + truth[KB] * (1 - share), 0.001, 0.001))
    assert ev["sources"][KB]["level"] == "ok" and ev["sources"][KB]["abs_diff"] == pytest.approx(0.025)
    assert ev["actions"] == {KA: {"action": "quarantine", "until_round": 12}}
    ev = g.observe(4, {KB: 1000}, _metrics(0.025, 0.001, 0.001))  # the reader passes A over
    assert ev["sources"][KB]["level"] == "ok" and ev["sources"][KB]["abs_diff_exact"]


# ---------------------------------------------------------------- identifiability (fix1115 F02)
def test_two_sources_at_a_stable_50_50_mix_only_warn():
    """Review repro: two sources, token shares a stable 50/50, healthy 0.02 and bad 0.3.
    The share matrix is rank deficient however many rows: main quarantined both (the
    trainer read nothing); now both warn with the reason and nothing is acted on."""
    g = xg.XhwGuard()
    for r in range(6):
        ev = g.observe(r, {KA: 500, KB: 500}, _metrics(0.16, 0.001, 0.001))
        assert ev["actions"] == {}
        for s in (KA, KB):
            row = ev["sources"][s]
            assert row["level"] == "warn" and row["action"] is None
            assert {h["reason"] for h in row["held"]} == {"rank_deficient"}
            assert row["held"][0]["rank"] == 1 and row["held"][0]["sources"] == 2


def test_two_sources_at_a_stable_70_30_mix_are_not_attributed():
    g = xg.XhwGuard()
    for r in range(6):
        ev = g.observe(r, {KA: 700, KB: 300}, _metrics(0.30 * 0.3 + 0.02 * 0.7, 0.001, 0.001))
        assert ev["actions"] == {}


def test_nearly_collinear_shares_only_warn():
    g = xg.XhwGuard()
    for r, share in enumerate((0.50, 0.501, 0.499, 0.5005, 0.4995, 0.502)):
        ev = g.observe(r, {KA: share * 1000, KB: (1 - share) * 1000}, _metrics(0.3 * share + 0.02 * (1 - share), 0.001, 0.001))
        assert ev["actions"] == {}
    held = ev["sources"][KA]["held"][0]
    assert held["reason"] == "collinear" and held["cond"] > 20


def test_mixed_estimate_needs_two_residual_degrees_of_freedom():
    """rows == sources + 1 is an exact fit plus one residual: not enough to act on."""
    g = xg.XhwGuard(min_rounds=1, window=3)
    for r, share in enumerate((0.3, 0.7, 0.5)):
        ev = g.observe(r, {KA: share * 1000, KB: (1 - share) * 1000}, _metrics(0.3 * share + 0.025 * (1 - share), 0.001, 0.001))
    assert ev["actions"] == {}
    assert ev["sources"][KA]["held"][0]["reason"] == "no_residual_dof"


def test_noisy_mixed_estimate_is_low_confidence():
    g = xg.XhwGuard(min_rounds=1)
    values = (0.20, 0.05, 0.16, 0.09, 0.03)  # a source near the limit, noisy steps
    for r, (share, v) in enumerate(zip((0.6, 0.3, 0.5, 0.4, 0.2), values)):
        ev = g.observe(r, {KA: share * 1000, KB: (1 - share) * 1000}, _metrics(v, 0.001, 0.001))
    row = ev["sources"][KA]
    assert ev["actions"] == {} and row["abs_diff"] > 0.15
    assert row["held"][0]["reason"] == "low_confidence"


def test_positive_bias_clip_residual_does_not_quarantine_a_clean_source():
    """A source whose log-probs read high saturates at 0: its clip fraction moves per
    round, and a mixed solve hands part of it to a clean source. A mixed clip-only
    estimate (abs_diff and KL clean) is held at warn."""
    g = xg.XhwGuard(min_rounds=1)
    bad_clip = (0.70, 0.78, 0.66, 0.75, 0.72)
    for r, (share, c) in enumerate(zip((0.5, 0.3, 0.6, 0.4, 0.2), bad_clip)):
        leak = 0.06 if r % 2 else 0.0  # the clean source's step looks clipped too
        ev = g.observe(r, {KA: share * 1000, KB: (1 - share) * 1000},
                       _metrics(0.21 * share + 0.01 * (1 - share), 0.001, c * share + leak * (1 - share)))
    assert KB not in ev["actions"]
    row = ev["sources"][KB]
    assert row["clip"] > 0.03 and row["level"] == "warn" and row["action"] is None  # leaked clip, held
    assert {h["reason"] for h in row["held"]} & {"clip_mixed_uncorroborated", "low_confidence"}


def test_quarantine_never_empties_the_batch():
    """The only live source over the act limit: quarantining it would leave the
    trainer reading nothing (main timed out), so it warns with would_empty_batch."""
    g = xg.XhwGuard(min_rounds=1)
    ev = g.observe(0, {KA: 100}, _metrics(0.3, 0.001))
    row = ev["sources"][KA]
    assert ev["actions"] == {} and row["level"] == "warn" and row["over"] == ["abs_diff"]
    assert row["held"] == [{"metric": None, "reason": "would_empty_batch"}]
    # two sources both over: the second is kept so the batch is not empty
    # both live sources over at once (each exact from its own round): one is kept
    g = xg.XhwGuard(min_rounds=1)
    g.observe(0, {KA: 100}, _metrics(0.3, 0.001))
    ev = g.observe(1, {KB: 100}, _metrics(0.3, 0.001))  # A not live this round: B kept
    assert ev["actions"] == {}
    ev = g.observe(2, {KA: 500, KB: 500}, _metrics(0.3, 0.001))
    assert len(ev["actions"]) == 1
    kept = ({KA, KB} - set(ev["actions"])).pop()
    assert ev["sources"][kept]["held"][-1]["reason"] == "would_empty_batch"


def test_a_dropped_source_in_the_window_is_not_live():
    """F10 review repro: s0 and s1 feed six rounds, s0 drops out, s1 alone goes over.
    s0 is still in the window rows but feeds nothing: quarantining s1 would empty the
    batch (and a trainer that does not train never calls observe again)."""
    g = xg.XhwGuard(min_rounds=1)
    for r, share in enumerate((0.3, 0.7, 0.5, 0.4, 0.6, 0.35)):
        ev = g.observe(r, {KA: share * 1000, KB: (1 - share) * 1000}, _metrics(0.02, 0.001, 0.001))
        assert ev["actions"] == {}
    for r in (6, 7, 8):
        ev = g.observe(r, {KB: 1000}, _metrics(0.3, 0.001))
        row = ev["sources"][KB]
        assert ev["actions"] == {} and row["action"] is None and row["level"] == "warn"
        assert {"metric": None, "reason": "would_empty_batch"} in row["held"]


# single-source decisions as on origin/main e14b5f74 (recorded from that module):
# (level, over, action, abs_diff estimate) per round
_SINGLE = dict(
    ad=[0.077, 0.099, 0.21, 0.23, 0.08, 0.076, 0.31, 0.09, 0.12, 0.085, 0.2, 0.2, 0.2, 0.079],
    kl=[0.015, 0.02, 0.02, 0.018, 0.07, 0.016, 0.02, 0.04, 0.016, 0.015, 0.016, 0.016, 0.016, 0.015],
    clip=[0.004, 0.005, 0.004, 0.004, 0.004, 0.05, 0.004, 0.004, 0.004, 0.02, 0.004, 0.004, 0.004, 0.004])
_A = ("act", ("abs_diff",))
_MAIN_DOWNWEIGHT = [("ok", (), None, 0.077), ("ok", (), None, 0.088), ("warn", (), None, 0.128667),
                    (*_A, "downweight", 0.179667), (*_A, "downweight", 0.173333), ("warn", (), "downweight", 0.128667),
                    (*_A, "downweight", 0.155333), (*_A, "downweight", 0.158667), (*_A, "downweight", 0.173333),
                    ("ok", (), None, 0.098333), ("warn", (), None, 0.135), (*_A, "downweight", 0.161667),
                    (*_A, "downweight", 0.2), (*_A, "downweight", 0.159667)]


def _single(action):
    g, out = xg.XhwGuard(action=action), []
    for r, (a, k, c) in enumerate(zip(_SINGLE["ad"], _SINGLE["kl"], _SINGLE["clip"])):
        try:
            ev = g.observe(r, {KA: 20000 + r}, _metrics(a, k, c))
        except xg.XhwGuardStop:
            out.append("STOP")
            break
        row = ev["sources"][KA]
        out.append((row["level"], tuple(row["over"]), row["action"], round(row["abs_diff"], 6)))
    return out


def test_single_source_decisions_match_main():
    assert _single("downweight") == _MAIN_DOWNWEIGHT
    assert _single("warn") == [(lvl, over, None, v) for lvl, over, _, v in _MAIN_DOWNWEIGHT]
    assert _single("stop") == _MAIN_DOWNWEIGHT[:3] + ["STOP"]
    # quarantine: same estimates and limits crossed; the sole source is never
    # quarantined (main left the trainer reading nothing), it warns instead
    got = _single("quarantine")
    assert [(o, v) for _, o, _, v in got[:4]] == [(o, v) for _, o, _, v in _MAIN_DOWNWEIGHT[:4]]
    assert all(a is None for _, _, a, _ in got) and got[3][0] == "warn"


def test_k31_delivery_form_is_unchanged():
    """K3.1 as delivered: the guard is off (env unset), nothing is built or emitted.
    With it on, healthy MI355X vLLM samplers in the measured band never act."""
    assert xg.XhwGuard.from_env({}) is None
    import random
    for seed in range(40):
        rnd, g = random.Random(seed), xg.XhwGuard()
        for r in range(30):
            toks = {f"nc-a{i}|vllm|vllm-kimi-k31@x|MI355X": rnd.randint(5000, 30000)
                    for i in (1, 4, 5) if rnd.random() > 0.2} or {KA: 1000}
            ev = g.observe(r, toks, _metrics(rnd.uniform(0.075, 0.099), rnd.uniform(0.015, 0.021), rnd.uniform(0.004, 0.0055)))
            assert ev["actions"] == {}, (seed, r, ev["sources"])


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
    d, events = _driver(xg.XhwGuard(min_rounds=1, action="downweight"), got.append)
    batch = SimpleNamespace(buffer_source_tokens={KA: 5}, buffer_source_stats={KA: {"tokens": 5.0}},
                            buffer_xhw_quarantined=0, buffer_xhw_downweighted=0)
    IslandDriver._guard_sources(d, 0, batch, _metrics(0.5, 0.01))
    name, fields = events[0]
    assert name == "rl_xhw_mismatch" and fields["sources"][KA]["level"] == "act"
    assert fields["rollout_stats"] == {KA: {"tokens": 5.0}}
    assert got == [{KA: {"action": "downweight", "weight": 0.5}}]


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
