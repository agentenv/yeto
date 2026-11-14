"""verl buffer trainer: the per-source xhw guard (YETO_RL_XHW_GUARD, default off),
the multi-family preflight advice and the verl refusal of OPSM / custom corrections."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from yeto.rl.adapters.verl import buffer_feed as bf
from yeto.rl.adapters.verl import fully_async_ports as fap
from yeto.rl.adapters.verl import fully_async_round as far
from yeto.rl.adapters.verl.fully_async_translate import VersionMap
from yeto.rl.algos import mismatch_correction as mc
from yeto.rl.engine import offpolicy_correction as oc
from yeto.rl.engine.algorithm import AlgorithmSpec, CorrectionSpec
from yeto.rl.engine.buffer_payload import GroupWriter
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer
from yeto.rl.engine.xhw_guard import XhwGuard, source_key

SHAPE = bf.RowShape(prompt_length=6, response_length=8, pad_token_id=0)
MI = {"sampler_id": "nc0", "engine": "vllm", "engine_version": "0.11", "card_model": "MI355X"}
B3 = {"sampler_id": "m0", "engine": "sglang", "engine_version": "0.5", "card_model": "B300"}


def _sample(index, response=(21, 22, 23)):
    return {"index": index, "prompt": "p", "tokens": [11, 12, *response], "response_length": len(response),
            "rollout_log_probs": [-0.5] * len(response), "reward": float(index),
            "metadata": {"weight_version": "0", "prompt_tokens_known": True, "finish_reason": "stop"}}


def _feed(tmp_path, **kw):
    buffer = FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1)
    writer = GroupWriter(buffer, "s")
    for key, prov in (("a", MI), ("b", B3), ("c", MI), ("d", B3)):
        writer.write([_sample(0), _sample(1)], group_key=key, provenance=prov)
    vmap = VersionMap()
    vmap.record(0, 0)
    records = []
    feed = bf.BufferFeed(FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1), vmap=lambda: vmap,
                         shape=SHAPE, add_group=lambda rows: None, record=records.append,
                         sleep=lambda s: None, incarnation_gate=None, **kw)
    return feed, vmap, records


def _round(feed, vmap, groups):
    collector = far.RoundCollector(vmap, 0, 1, groups)
    while not collector.full:
        meta = feed.pull()
        feed("yeto_keep_last" if collector.offer(far.SampleMeta(**meta)) else "yeto_drop_last")
    feed.close_round(0)


def test_feed_default_off_reports_nothing_and_quarantines_nothing(tmp_path):
    feed, vmap, records = _feed(tmp_path)
    feed.set_xhw_actions({source_key(MI): {"action": "quarantine", "until_round": 99}})
    _round(feed, vmap, 4)
    assert feed.xhw_round() == {}
    assert all(r["kept"] for r in records) and "xhw_quarantine" not in feed.dropped


def test_feed_counts_tokens_per_source_and_drops_a_quarantined_source(tmp_path):
    rid = {"r": 0}
    feed, vmap, records = _feed(tmp_path, xhw=True, round_of=lambda: rid["r"])
    _round(feed, vmap, 2)
    assert feed.xhw_round() == {"buffer_source_tokens": {source_key(MI): 6, source_key(B3): 6},
                                "buffer_xhw_quarantined": 0, "buffer_xhw_downweighted": 0}
    feed.set_xhw_actions({source_key(MI): {"action": "quarantine", "until_round": 5}})
    rid["r"] = 1
    _round(feed, vmap, 1)  # c (MI) is passed over, d (B3) trained
    assert feed.xhw_round()["buffer_source_tokens"] == {source_key(B3): 6}
    assert feed.xhw_round()["buffer_xhw_quarantined"] == 1
    assert [r.get("dropped") for r in records if not r["kept"]] == ["xhw_quarantine"]


def test_quarantine_expires_at_its_until_round(tmp_path):
    feed, _, _ = _feed(tmp_path, xhw=True, round_of=lambda: 5)
    feed.set_xhw_actions({source_key(MI): {"action": "quarantine", "until_round": 5}})
    assert not feed._quarantined(source_key(MI))


def test_rollout_handle_and_round_metrics_carry_the_guard_inputs_only_when_on():
    island = SimpleNamespace(xhw_feed=None, last_metrics={}, last_criteria={"abs_diff": 0.2, "k3": 0.01})
    pool = SimpleNamespace(island=island)
    assert fap._xhw_round_fields(island) == {}
    assert fap.FullyAsyncTrainerGroup.round_metrics(pool) == {}
    island.xhw_feed = SimpleNamespace(xhw_round=lambda: {"buffer_source_tokens": {"x": 1}})
    assert fap._xhw_round_fields(island) == {"buffer_source_tokens": {"x": 1}}
    assert fap.FullyAsyncTrainerGroup.round_metrics(pool) == {
        "train_rollout_logprob_abs_diff": 0.2, "train_rollout_kl": 0.01}
    # the driver's guard reads the step values under the Miles names
    guard = XhwGuard()
    metrics = {name: island.last_criteria[c] for name, c in fap.XHW_CRITERIA.items() if c in island.last_criteria}
    event = guard.observe(0, {"x": 1}, metrics)
    assert event["step"]["abs_diff"] == 0.2 and event["step"]["kl"] == 0.01


def test_verl_buffer_trainer_refuses_downweight_and_is_off_by_default():
    from yeto.rl.adapters.verl.buffer_feed import xhw_guard_enabled

    assert xhw_guard_enabled({}) is False
    assert xhw_guard_enabled({"YETO_RL_XHW_GUARD": "1"}) is True
    with pytest.raises(ValueError, match="downweight"):
        xhw_guard_enabled({"YETO_RL_XHW_GUARD": "action=downweight"})


# ------------------------------------------------------------------ preflight (T25)
def _args(**kw):
    return SimpleNamespace(training_mode="rl", **kw)


def test_preflight_warns_on_several_device_families_and_never_turns_the_guard_on(capsys):
    from yeto.gpu_spec import parse_gpu_spec
    from yeto.launch_preflight import xhw_family_advice

    one = parse_gpu_spec("modal:8xH100,modal:8xH100")
    assert xhw_family_advice(_args(), one, environ={}) is None
    mixed = parse_gpu_spec("modal:8xH100,sky:8x910B4")
    record = xhw_family_advice(_args(), mixed, environ={})
    assert record == {"device_families": {"cuda": [0], "npu": [1]}, "xhw_guard": False}
    assert "YETO_RL_XHW_GUARD" in capsys.readouterr().out
    assert xhw_family_advice(_args(), mixed, environ={"YETO_RL_XHW_GUARD": "1"})["xhw_guard"] is True
    assert "WARNING" not in capsys.readouterr().out
    assert xhw_family_advice(SimpleNamespace(training_mode="sft"), mixed, environ={}) is None


# ------------------------------------------------------------------ T24
@pytest.mark.parametrize("method", ["opsm", "custom"])
def test_verl_refuses_opsm_and_custom_at_launch_and_at_the_island(method, tmp_path):
    from yeto.rl.adapters.verl.island_entry import append_correction_record

    if method == "opsm":
        spec = AlgorithmSpec(correction=CorrectionSpec(method="opsm", opsm_delta=1e-4))
    else:
        spec = AlgorithmSpec(correction={"method": "custom", "function": mc.icepop_ref().to_dict(),
                                         "tis_clip": 5.0, "tis_clip_low": 0.5})
    manifest = oc.run_correction_manifest(spec, {"0": "verl"})
    with pytest.raises(oc.CorrectionConfigError, match="on backend 'verl'"):
        oc.refuse_correction_warnings(manifest["warnings"])
    args = SimpleNamespace(event_tape=str(tmp_path / "tape.jsonl"), learner_id=0)
    with pytest.raises(oc.CorrectionConfigError, match="on backend 'verl'"):
        append_correction_record(args, spec.to_dict())
    # Miles maps both: no refusal there
    assert not [w for w in oc.run_correction_manifest(spec, {"0": "miles"})["warnings"]
                if w.get("backend") == "miles"]
