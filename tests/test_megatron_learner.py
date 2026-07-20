"""Non-GPU-testable parts of the Megatron learner: arg parsing and the data
adapter. The Megatron-Core training path is GPU/multi-node only and validated
by a live run, not here."""

from yeto.megatron import learner as ml


def test_parse_args_maps_backend_and_parallelism():
    a = ml.parse_args(
        ["--model", "deepseek31-bf16", "--data", "org/d", "--syncer", "h:29400",
         "--expert-parallel", "24", "--tensor-parallel", "1", "--pipeline-parallel", "1",
         "--lora-r", "16", "--wire-dtype", "q4"]
    )
    assert a.island_backend == "megatron"
    assert (a.expert_parallel, a.tensor_parallel, a.pipeline_parallel) == (24, 1, 1)
    assert a.lora_r == 16 and a.wire_dtype == "q4"
    assert a.seed == 0


def test_parse_args_accepts_shared_initialization_seed():
    a = ml.parse_args(["--model", "org/model", "--data", "org/data", "--seed", "29"])
    assert a.seed == 29


def test_attention_targets_are_megatron_names_not_hf():
    # Must be mcore module names (linear_qkv / MLA splits), never HF (q_proj).
    assert "linear_qkv" in ml._ATTENTION_TARGETS
    assert "linear_q_down_proj" in ml._ATTENTION_TARGETS  # DeepSeek MLA
    assert not any("q_proj" == t for t in ml._ATTENTION_TARGETS)
    assert ml._MLP_TARGETS == ["linear_fc1", "linear_fc2"]


def test_cycle_yields_endless_input_label_batches():
    gen = ml._cycle([{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}])
    b0 = next(gen)
    assert b0["input_ids"].shape == (1, 3)
    assert (b0["labels"] == b0["input_ids"]).all()
    # endless: wraps past the 2-element dataset
    seen = [tuple(next(gen)["input_ids"][0].tolist()) for _ in range(4)]
    assert seen == [(4, 5, 6), (1, 2, 3), (4, 5, 6), (1, 2, 3)]
