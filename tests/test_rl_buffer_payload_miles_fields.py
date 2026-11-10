import numpy
import pytest

from yeto.rl.engine.buffer_payload import (
    MILES_TOPK_LOG_PROBS,
    MILES_TOPK_TOKEN_IDS,
    ROLLOUT_ROUTED_EXPERTS,
    ROLLOUT_TOP_LOGPROBS,
    ROLLOUT_TOP_LOGPROBS_IDS,
    ROUTED_EXPERTS_PAD,
    SampleBufferRolloutError,
    decode_group,
    encode_group,
    miles_sample_dict,
)


def _engine_sample(t_resp=3, k=4, positions=6):
    routed = numpy.arange(positions * 3 * 2, dtype=numpy.uint16).reshape(positions, 3, 2)
    routed[-1] = ROUTED_EXPERTS_PAD
    return {
        "index": 7,
        "response_length": t_resp,
        ROLLOUT_TOP_LOGPROBS_IDS: numpy.arange(t_resp * k, dtype=numpy.int32).reshape(t_resp, k),
        ROLLOUT_TOP_LOGPROBS: -numpy.ones((t_resp, k), dtype=numpy.float32),
        ROLLOUT_ROUTED_EXPERTS: routed,
    }


def test_engine_fields_take_the_miles_sample_names_and_dtypes():
    sample = decode_group(encode_group([_engine_sample()]))[0]
    out = miles_sample_dict(sample)
    assert ROLLOUT_TOP_LOGPROBS_IDS not in out and ROLLOUT_TOP_LOGPROBS not in out
    assert out[MILES_TOPK_TOKEN_IDS].dtype == numpy.int32 and out[MILES_TOPK_TOKEN_IDS].shape == (3, 4)
    assert out[MILES_TOPK_LOG_PROBS].dtype == numpy.float32
    numpy.testing.assert_array_equal(out[MILES_TOPK_TOKEN_IDS], sample[ROLLOUT_TOP_LOGPROBS_IDS])
    routed = out[ROLLOUT_ROUTED_EXPERTS]
    assert routed.dtype == numpy.int32
    assert (routed[-1] == -1).all()
    numpy.testing.assert_array_equal(routed[:-1], sample[ROLLOUT_ROUTED_EXPERTS][:-1].astype(numpy.int32))
    assert sample[ROLLOUT_TOP_LOGPROBS_IDS] is not None  # the input mapping is left as it was


def test_a_miles_native_sample_passes_through():
    ids = numpy.zeros((2, 4), dtype=numpy.int32)
    routed = numpy.full((5, 3, 2), -1, dtype=numpy.int32)
    out = miles_sample_dict({"index": 0, "response_length": 2, MILES_TOPK_TOKEN_IDS: ids,
                             MILES_TOPK_LOG_PROBS: ids.astype(numpy.float32), ROLLOUT_ROUTED_EXPERTS: routed})
    assert out[MILES_TOPK_TOKEN_IDS] is ids
    assert (out[ROLLOUT_ROUTED_EXPERTS] == -1).all()


def test_both_top_logprob_forms_are_refused():
    sample = _engine_sample()
    sample[MILES_TOPK_TOKEN_IDS] = sample[ROLLOUT_TOP_LOGPROBS_IDS]
    with pytest.raises(SampleBufferRolloutError, match="holds both"):
        miles_sample_dict(sample)


def test_top_logprobs_that_miss_response_positions_are_refused():
    sample = _engine_sample()
    sample["response_length"] = 5
    with pytest.raises(SampleBufferRolloutError, match="response_length"):
        miles_sample_dict(sample)


def test_trainer_prompt_ids_must_match_the_engine_routing_rows():
    from yeto.rl.engine.buffer_payload import complete_prompt_tokens

    sample = {"index": 1, "prompt": "p", "tokens": [40, 41], "response_length": 2,
              "metadata": {"prompt_tokens_known": False},
              ROLLOUT_ROUTED_EXPERTS: numpy.zeros((4, 3, 2), dtype=numpy.uint16)}  # 3 prompt + 2 - 1
    assert complete_prompt_tokens(sample, lambda text: [7, 8, 9])["tokens"] == [7, 8, 9, 40, 41]
    with pytest.raises(SampleBufferRolloutError, match="imply 3 prompt ids"):
        complete_prompt_tokens(sample, lambda text: [7])


def test_moe_only_r3_gets_the_dense_layer_rows_in_front():
    sample = _engine_sample()  # R3 [6, 3, 2]: 3 MoE layers
    out = miles_sample_dict(sample, r3_layers=(4, 1))
    routed = out[ROLLOUT_ROUTED_EXPERTS]
    assert routed.shape == (6, 4, 2)
    assert (routed[:, 0] == -1).all()  # the dense layer 0 has no router
    numpy.testing.assert_array_equal(routed[:-1, 1:], sample[ROLLOUT_ROUTED_EXPERTS][:-1].astype(numpy.int32))
    assert miles_sample_dict(sample, r3_layers=(3, 0))[ROLLOUT_ROUTED_EXPERTS].shape == (6, 3, 2)
    with pytest.raises(SampleBufferRolloutError, match="has 3 layers"):
        miles_sample_dict(sample, r3_layers=(6, 1))


def test_r3_layer_layout_reads_the_text_config(tmp_path):
    import json

    from yeto.rl.adapters.miles.buffer_rollout import r3_layer_layout

    (tmp_path / "config.json").write_text(json.dumps(
        {"text_config": {"num_hidden_layers": 4, "first_k_dense_replace": 1}}))
    assert r3_layer_layout(str(tmp_path)) == (4, 1)
    assert r3_layer_layout(str(tmp_path / "missing")) is None


def test_r3_ids_must_fit_the_model_and_rows_must_not_mix_pads():
    sample = _engine_sample()  # ids 0..29, last row all pads
    assert miles_sample_dict(sample, r3_num_experts=896)[ROLLOUT_ROUTED_EXPERTS].shape == (6, 3, 2)
    with pytest.raises(SampleBufferRolloutError, match="names expert 29"):
        miles_sample_dict(sample, r3_num_experts=16)
    mixed = _engine_sample()
    mixed[ROLLOUT_ROUTED_EXPERTS][0, 0, 0] = ROUTED_EXPERTS_PAD
    with pytest.raises(SampleBufferRolloutError, match="mixes pads"):
        miles_sample_dict(mixed)


def test_r3_num_experts_reads_the_text_config(tmp_path):
    import json

    from yeto.rl.adapters.miles.buffer_rollout import r3_num_experts

    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"n_routed_experts": 896}}))
    assert r3_num_experts(str(tmp_path)) == 896
    assert r3_num_experts(None) is None
