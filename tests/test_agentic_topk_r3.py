"""Agentic samples carry the rollout top-K log-probs and R3 routed experts.

Multi-turn records-proxy turns -> ``turns_from_records`` -> ``join_turns`` ->
``agentic_sample`` -> ``miles_sample_dict`` -> Miles ``Sample`` ->
``convert_samples_to_train_data`` with score_centering and routing replay on.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy
import pytest

from yeto.rl.agentic import AgentTask, AgentTrajectory, Outcome, TokenAlignmentError, TokenTurn, join_turns
from yeto.rl.agentic.finish import FinishedSample
from yeto.rl.engine.agentic_round import agentic_sample
from yeto.rl.engine.buffer_payload import (
    MILES_TOPK_LOG_PROBS,
    MILES_TOPK_TOKEN_IDS,
    ROLLOUT_ROUTED_EXPERTS,
    ROLLOUT_TOP_LOGPROBS,
    ROLLOUT_TOP_LOGPROBS_IDS,
    decode_group,
    encode_group,
    miles_sample_dict,
)
from yeto.rl.engine.engine_sampler import PromptItem
from yeto.rl.harness.gateway.proxy import _records_dumps
from yeto.rl.harness.gateway.recorder import turns_from_records

K, LAYERS, TOPK, EXPERTS = 4, 3, 2, 16


def _top(completion, seed):
    ids = numpy.array([[t] + [(t + c) % 1000 for c in range(1, K)] for t in completion], dtype=numpy.int32)
    vals = -numpy.random.default_rng(seed).random((len(completion), K)).astype(numpy.float32) - 0.01
    return ids.reshape(len(completion), K), vals


def _routed(rows, seed):
    return numpy.random.default_rng(seed).integers(0, EXPERTS, size=(rows, LAYERS, TOPK), dtype=numpy.uint16)


def _turn(prompt, completion, seed, *, r3=True):
    ids, vals = _top(completion, seed)
    return TokenTurn(prompt_ids=tuple(prompt), completion_ids=tuple(completion),
                     completion_logprobs=tuple(float(v) for v in vals[:, 0]),
                     completion_top_ids=ids, completion_top_logprobs=vals,
                     routed_experts=_routed(len(prompt) + len(completion) - 1, seed) if r3 else None,
                     finish_reason="stop")


def _three_turns():
    p0 = [1, 2, 3]
    c0 = [10, 11]
    p1 = p0 + c0 + [20, 21, 22]  # tool output
    c1 = [12, 13, 14]
    p2 = p1 + c1 + [23]
    c2 = [15]
    return [_turn(p0, c0, 0), _turn(p1, c1, 1), _turn(p2, c2, 2)]


def test_join_aligns_topk_with_the_mask_and_keeps_the_last_turn_r3():
    turns = _three_turns()
    seg = join_turns(turns)
    n = len(seg.response_ids)
    assert seg.response_top_ids.shape == seg.response_top_logprobs.shape == (n, K)
    mask = numpy.array(seg.response_mask, dtype=bool)
    # mask-0 rows: Miles' fill between turns (id -1, log-prob -inf)
    assert (seg.response_top_ids[~mask] == -1).all() and numpy.isneginf(seg.response_top_logprobs[~mask]).all()
    # mask-1 rows: each turn's own candidates, the sampled token first with its log-prob
    assert (seg.response_top_ids[mask, 0] == numpy.array(seg.response_ids)[mask]).all()
    assert numpy.allclose(seg.response_top_logprobs[mask, 0], numpy.array(seg.response_logprobs)[mask])
    assert numpy.array_equal(seg.response_top_ids[mask][:2], turns[0].completion_top_ids)
    # R3: whole stream minus one row, the last turn's (as a single-turn sample covers its prompt)
    assert seg.routed_experts.shape == (len(seg.prompt_ids) + n - 1, LAYERS, TOPK)
    assert seg.routed_experts is turns[-1].routed_experts or numpy.array_equal(seg.routed_experts,
                                                                               turns[-1].routed_experts)


def test_join_truncation_cuts_every_array_to_the_kept_tokens():
    seg = join_turns(_three_turns(), max_response_len=6)
    assert seg.truncated and len(seg.response_ids) == 6
    assert seg.response_top_ids.shape == (6, K)
    assert seg.routed_experts.shape[0] == len(seg.prompt_ids) + 6 - 1


def test_join_without_arrays_is_unchanged_and_mixed_topk_is_refused():
    plain = TokenTurn(prompt_ids=(1, 2), completion_ids=(3,), completion_logprobs=(-0.1,))
    seg = join_turns([plain])
    assert seg.response_top_ids is None and seg.routed_experts is None
    with pytest.raises(TokenAlignmentError, match="top-K"):
        join_turns([_turn([1, 2], [3], 0), TokenTurn(prompt_ids=(1, 2, 3, 4), completion_ids=(5,),
                                                    completion_logprobs=(-0.1,))])


def test_turn_refuses_r3_rows_that_miss_its_positions():
    with pytest.raises(TokenAlignmentError, match="routed experts"):
        TokenTurn(prompt_ids=(1, 2), completion_ids=(3,), completion_logprobs=(-0.1,),
                  routed_experts=_routed(5, 0))


def _records(turns):
    """Proxy records of one segment, the way RecordingBackend writes them
    (only the newest turn keeps its R3)."""
    last = turns[-1]
    ids = list(last.prompt_ids) + list(last.completion_ids)
    mask = [0] * len(ids)
    out, start = [], 0
    for number, turn in enumerate(turns):
        added = turn.prompt_ids[start:]
        for i in range(len(turn.prompt_ids), len(turn.prompt_ids) + len(turn.completion_ids)):
            mask[i] = 1
        out.append({"turn": number, "prompt_added": list(added), "completion": list(turn.completion_ids),
                    "logprobs": list(turn.completion_logprobs), "finish_reason": "stop", "stop_added": [],
                    "top_ids": turn.completion_top_ids, "top_logprobs": turn.completion_top_logprobs,
                    "routed_experts": turn.routed_experts if number == len(turns) - 1 else None})
        start = len(turn.prompt_ids) + len(turn.completion_ids)
    return {"sessions": [{"samples": [{"segment": 0, "token_ids": ids, "loss_mask": mask, "turns": out}]}]}


def test_recorded_turns_carry_the_arrays_through_the_join():
    turns = _three_turns()
    seg = join_turns(turns_from_records(_records(turns)))
    ref = join_turns(turns)
    assert numpy.array_equal(seg.response_top_ids, ref.response_top_ids)
    assert numpy.array_equal(seg.routed_experts, ref.routed_experts)
    assert json.loads(_records_dumps(_records(turns)))  # the HTTP route serializes the arrays


def test_records_backend_records_engine_arrays_and_keeps_only_the_newest_r3():
    import asyncio

    from yeto.rl.adapters.sampler_engines.fake import FakeEngine
    from yeto.rl.engine.sampler_engine import SamplingParams
    from yeto.rl.harness.gateway.records import RecordingBackend, Segment, _Prompt, _Session

    engine = FakeEngine(token_reply=lambda prompt, i: [7, 8], routed_experts=(EXPERTS, LAYERS, TOPK),
                        top_logprobs_num=K)
    engine.start(None, 3)
    backend = object.__new__(RecordingBackend)
    backend.engine, backend.sampler_id, backend.r3_response_len = engine, "records-proxy", None
    backend.template = SimpleNamespace(stop_token_ids=(), stop_text=lambda t: None)
    backend.counters = {"t": __import__("collections").Counter()}
    session, segment = _Session("t", []), Segment(0, head=[], base=0)
    params = SamplingParams(max_tokens=4)
    asyncio.run(backend._sample(session, segment, _Prompt(stop=(), tail=(1, 2, 3)), params, kind="action"))
    asyncio.run(backend._sample(session, segment, _Prompt(stop=(), tail=(4, 5)), params, kind="action"))
    first, second = segment.turns
    assert first.routed_experts is None and first.top_ids.shape == (2, K)
    assert second.routed_experts.shape == (len(segment.token_ids) - 1, LAYERS, TOPK)
    seg = join_turns(turns_from_records({"sessions": [{"samples": [segment.sample()]}]}))
    assert seg.response_top_ids.shape == (len(seg.response_ids), K)


def _record_trial(r3_response_len, tails=((1, 2, 3), (4, 5), (6,), (7, 8, 9)), reply=(7, 8, 9)):
    """One multi-turn trial through a real RecordingBackend; its records joined."""
    import asyncio

    from yeto.rl.adapters.sampler_engines.fake import FakeEngine
    from yeto.rl.engine.sampler_engine import SamplingParams
    from yeto.rl.harness.gateway.records import RecordingBackend, _Prompt

    engine = FakeEngine(token_reply=lambda prompt, i: list(reply), routed_experts=(EXPERTS, LAYERS, TOPK),
                        top_logprobs_num=K)
    engine.start(None, 3)
    template = SimpleNamespace(stop_token_ids=(), stop_text=lambda t: None,
                               spec=SimpleNamespace(name="fake", verified=True))
    backend = RecordingBackend(engine, template, check_render=False, r3_response_len=r3_response_len)
    session_id = asyncio.run(backend.create_session("t"))
    session = backend.sessions[session_id]
    segment = session.segments[0]
    params = SamplingParams(max_tokens=8)
    for tail in tails:
        asyncio.run(backend._sample(session, segment, _Prompt(stop=(), tail=tail), params, kind="action"))
    return segment, backend.records("t")


@pytest.mark.parametrize("max_response_len", [1, 4, 9, 13, 200])
def test_r3_rows_cut_to_the_training_window_join_bit_identically(max_response_len):
    """RecordingBackend(r3_response_len=R) keeps only the rows join_turns(max_response_len=R)
    uses; the joined sample is the same as from the full rows, bit for bit."""
    full_segment, full = _record_trial(None)
    cut_segment, cut = _record_trial(max_response_len)
    ref = join_turns(turns_from_records(full), max_response_len=max_response_len, rebase=True)
    got = join_turns(turns_from_records(cut), max_response_len=max_response_len, rebase=True)
    assert got.routed_experts.dtype == ref.routed_experts.dtype == numpy.uint16
    assert got.routed_experts.shape == ref.routed_experts.shape
    assert got.routed_experts.tobytes() == ref.routed_experts.tobytes()
    assert (got.prompt_ids, got.response_ids, got.response_mask, got.response_logprobs, got.truncated) == \
        (ref.prompt_ids, ref.response_ids, ref.response_mask, ref.response_logprobs, ref.truncated)
    assert got.response_top_ids.tobytes() == ref.response_top_ids.tobytes()
    assert got.response_top_logprobs.tobytes() == ref.response_top_logprobs.tobytes()
    window = 3 + max_response_len - 1  # first prompt (3 tokens) + R - 1
    stored, whole = cut_segment.turns[-1].routed_experts, full_segment.turns[-1].routed_experts
    assert len(whole) == len(full_segment.token_ids) - 1
    if window < len(whole):  # cut: a copy of the leading rows, not a view of the full array
        assert cut_segment.turns[-1].routed_experts_cut and len(stored) == window
        assert stored.base is None and stored.tobytes() == whole[:window].tobytes()
    else:
        assert not cut_segment.turns[-1].routed_experts_cut and len(stored) == len(whole)
    assert json.loads(_records_dumps(cut))  # the HTTP route still serializes the records


def test_join_copies_the_r3_rows_it_keeps_and_refuses_a_short_window():
    turns = _three_turns()
    seg = join_turns(turns, max_response_len=4)
    assert seg.routed_experts.base is None  # not a view keeping the last turn's rows alive
    assert seg.routed_experts.tobytes() == turns[-1].routed_experts[:len(seg.prompt_ids) + 3].tobytes()
    last = turns[-1]
    short = TokenTurn(prompt_ids=last.prompt_ids, completion_ids=last.completion_ids,
                      completion_logprobs=last.completion_logprobs, completion_top_ids=last.completion_top_ids,
                      completion_top_logprobs=last.completion_top_logprobs,
                      routed_experts=last.routed_experts[:4].copy(), routed_experts_cut=True)
    with pytest.raises(TokenAlignmentError, match="routed-expert rows"):
        join_turns([*turns[:-1], short], max_response_len=4)
    with pytest.raises(TokenAlignmentError, match="routed experts"):  # uncut rows must cover every position
        TokenTurn(prompt_ids=last.prompt_ids, completion_ids=last.completion_ids,
                  completion_logprobs=last.completion_logprobs, routed_experts=last.routed_experts[:4].copy())


def test_r3_response_len_must_be_positive():
    from yeto.rl.adapters.sampler_engines.fake import FakeEngine
    from yeto.rl.harness.gateway.records import RecordingBackend

    with pytest.raises(ValueError, match="r3_response_len"):
        RecordingBackend(FakeEngine(), SimpleNamespace(), r3_response_len=0)


def _sample(index=0):
    seg = join_turns(_three_turns())
    task = AgentTask.from_mapping({"task_id": "t", "trajectory_id": f"t-{index}", "agent": "codex", "provider": "fake"})
    trajectory = AgentTrajectory(task=task, outcome=Outcome.OK, reward=float(index % 2), reason="ok",
                                 verdict=None, tokens=seg)
    finished = FinishedSample(trajectory=trajectory, segments=seg, turns=_three_turns())
    item = PromptItem(prompt_id="p", prompt="task", metadata={})
    return agentic_sample(item, finished, version=3, provenance={"engine": "fake"}, group_index=0, index=index,
                          engine_meta={"routed_experts": True, "top_logprobs_num": K})


def test_agentic_sample_has_the_single_turn_fields_and_survives_the_buffer():
    sample = decode_group(encode_group([_sample()]))[0]
    n = sample["response_length"]
    assert sample[ROLLOUT_TOP_LOGPROBS_IDS].dtype == numpy.int32 and sample[ROLLOUT_TOP_LOGPROBS_IDS].shape == (n, K)
    assert sample[ROLLOUT_TOP_LOGPROBS].dtype == numpy.float32
    assert sample[ROLLOUT_ROUTED_EXPERTS].dtype == numpy.uint16
    assert sample[ROLLOUT_ROUTED_EXPERTS].shape[0] == len(sample["tokens"]) - 1
    assert sample["metadata"]["engine"]["top_logprobs_num"] == K
    miles = miles_sample_dict(sample, r3_layers=(LAYERS + 1, 1), r3_num_experts=EXPERTS)
    assert miles[MILES_TOPK_TOKEN_IDS].shape == (n, K)
    assert miles[ROLLOUT_ROUTED_EXPERTS].shape == (len(sample["tokens"]) - 1, LAYERS + 1, TOPK)


def test_miles_train_data_conversion_accepts_agentic_samples():
    tdc = pytest.importorskip("miles.ray.rollout.train_data_conversion")
    from miles.rollout.generate_utils.rollout_topk_logprobs import validate_rollout_topk_logprobs_sample
    from miles.utils.types import Sample

    samples = []
    for i in range(2):
        d = miles_sample_dict(decode_group(encode_group([_sample(i)]))[0], r3_layers=(LAYERS, 0),
                              r3_num_experts=EXPERTS)
        d["status"] = Sample.Status.COMPLETED.value
        samples.append(Sample.from_dict(d))
    for s in samples:
        validate_rollout_topk_logprobs_sample(s, K)
        s.validate()
    args = SimpleNamespace(advantage_estimator="grpo", rewards_normalization=True, grpo_std_normalization=True,
                           n_samples_per_prompt=2, rollout_batch_size=1, reward_key=None,
                           use_dynamic_global_batch_size=False, rollout_top_logprobs_num=K, ci_test=True,
                           use_output_store=False, use_rollout_routing_replay=True,
                           use_rollout_indexer_replay=False, length_penalty_coeff=0.0, balance_data=False,
                           custom_convert_samples_to_train_data_path=None, custom_reward_post_process_path=None,
                           custom_view=None, enable_sample_ownership_checker=False, global_batch_size=2,
                           loss_type="policy_loss", multi_lora_n_adapters=0)
    data = tdc.convert_samples_to_train_data(args, samples, {}, None, None)
    assert len(data["rollout_topk_token_ids"]) == 2 and len(data["rollout_routed_experts"]) == 2
    assert data["loss_masks"][0] == list(samples[0].loss_mask)
