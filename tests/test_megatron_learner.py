"""Non-GPU-testable parts of the Megatron learner: arg parsing and the data
adapter. The Megatron-Core training path is GPU/multi-node only and validated
by a live run, not here."""

from types import SimpleNamespace

import torch

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


def test_parse_args_defaults_to_native_mask_and_accepts_explicit_legacy():
    base = ["--model", "org/model", "--data", "org/data"]

    assert ml.parse_args(base).assistant_mask_mode == "native"
    assert (
        ml.parse_args(base + ["--assistant-mask-mode", "legacy"]).assistant_mask_mode
        == "legacy"
    )


def test_attention_targets_are_megatron_names_not_hf():
    # Must be mcore module names (linear_qkv / MLA splits), never HF (q_proj).
    assert "linear_qkv" in ml._ATTENTION_TARGETS
    assert "linear_q_down_proj" in ml._ATTENTION_TARGETS  # DeepSeek MLA
    assert not any("q_proj" == t for t in ml._ATTENTION_TARGETS)
    assert ml._MLP_TARGETS == ["linear_fc1", "linear_fc2"]


def test_cycle_shifts_labels_and_exact_loss_weights_in_micro_batches():
    blocks = [
        (torch.tensor([1, 2, 3]), torch.tensor([0.0, 1.0, 0.0])),
        (torch.tensor([4, 5, 6]), torch.tensor([0.0, 0.0, 1.0])),
    ]
    gen = ml._cycle(blocks, micro_batch_size=2)
    b0 = next(gen)
    assert b0["input_ids"].tolist() == [[1, 2, 3], [4, 5, 6]]
    assert b0["labels"].tolist() == [[2, 3, 0], [5, 6, 0]]
    assert b0["loss_mask"].tolist() == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    # Endless preload iteration wraps and produces the same full micro-batch.
    assert next(gen)["input_ids"].tolist() == b0["input_ids"].tolist()


def test_packed_data_path_propagates_mask_mode(monkeypatch):
    calls = []

    def fake_build(*args, **kwargs):
        calls.append((args, kwargs))
        return "packed"

    monkeypatch.setattr("yeto.data.build_packed_dataset", fake_build)
    args = SimpleNamespace(
        data="rows.jsonl",
        learner_id=2,
        num_learners=4,
        seq_len=1024,
        max_rows=50,
        train_on="assistant",
        assistant_mask_mode="legacy",
        tokenize="preload",
    )

    assert ml._packed_blocks(args, tokenizer="tok") == "packed"
    positional, keywords = calls[0]
    assert positional == ("rows.jsonl", "tok", 2, 4, 1024, 50)
    assert keywords == {"train_on": "assistant", "assistant_mask_mode": "legacy"}


def test_streaming_data_path_keeps_ep_ranks_on_the_same_tokens(monkeypatch):
    calls = []

    class FakeStream:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr("yeto.data.StreamingPackedBlocks", FakeStream)
    args = SimpleNamespace(
        data="rows.jsonl",
        learner_id=1,
        num_learners=3,
        seq_len=512,
        max_rows=None,
        train_on="assistant",
        assistant_mask_mode="native",
        tokenize="stream",
    )

    assert isinstance(ml._packed_blocks(args, tokenizer="tok"), FakeStream)
    _positional, keywords = calls[0]
    assert keywords == {
        "rank": 0,
        "world": 1,
        "train_on": "assistant",
        "assistant_mask_mode": "native",
    }
