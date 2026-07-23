"""Non-GPU-testable parts of the Megatron learner: arg parsing and the data
adapter. The Megatron-Core training path is GPU/multi-node only and validated
by a live run, not here."""

import json
import sys
import types
from pathlib import Path
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
    assert a.seed == 0


def test_parse_args_accepts_shared_initialization_seed():
    a = ml.parse_args(["--model", "org/model", "--data", "org/data", "--seed", "29"])
    assert a.seed == 29


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


def test_build_model_disables_bridge_ddp_and_uses_lora_signature(monkeypatch):
    seen = {}

    class FakeBridge:
        @classmethod
        def from_hf_pretrained(cls, model_id, **kwargs):
            seen["model_id"] = model_id
            seen["from_hf"] = kwargs
            return cls()

        def to_megatron_model(self, load_weights=True, wrap_with_ddp=True):
            seen["to_megatron"] = {
                "load_weights": load_weights,
                "wrap_with_ddp": wrap_with_ddp,
            }
            return ["chunk"]

    class FakeLoRA:
        def __init__(self, dim, alpha, target_modules):
            seen["lora"] = {
                "dim": dim,
                "alpha": alpha,
                "target_modules": target_modules,
            }

        def __call__(self, model, training=False):
            seen["training"] = training
            return model

    monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))
    monkeypatch.setitem(
        sys.modules,
        "megatron.bridge",
        types.SimpleNamespace(AutoBridge=FakeBridge),
    )
    monkeypatch.setitem(sys.modules, "megatron.bridge.peft", types.ModuleType("peft"))
    monkeypatch.setitem(
        sys.modules,
        "megatron.bridge.peft.lora",
        types.SimpleNamespace(LoRA=FakeLoRA),
    )
    monkeypatch.setattr("yeto.models.resolve", lambda model: f"resolved/{model}")

    args = SimpleNamespace(
        model="m",
        model_revision=None,
        trust_remote_code=True,
        expert_parallel=1,
        lora_targets="attention",
        lora_r=8,
        lora_alpha=16,
    )
    model, _bridge = ml._build_model(args, device=None)

    assert model == ["chunk"]
    assert seen["model_id"] == "resolved/m"
    assert seen["from_hf"] == {"use_safetensors": True, "trust_remote_code": True}
    assert seen["to_megatron"] == {"load_weights": True, "wrap_with_ddp": False}
    assert seen["lora"]["dim"] == 8 and seen["lora"]["alpha"] == 16
    assert "linear_qkv" in seen["lora"]["target_modules"]
    assert "share_expert_adapters" not in seen["lora"]
    assert seen["training"] is True


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


def test_save_output_best_effort_keeps_full_tuning_successful(tmp_path, caplog):
    class FailingBridge:
        def save_hf_pretrained(self, model, save_dir):
            raise AttributeError("'NoneType' object has no attribute 'megatron_to_hf'")

    assert (
        ml._save_output_best_effort(
            FailingBridge(),
            ["model"],
            tmp_path,
            SimpleNamespace(tuning="full"),
        )
        is False
    )
    assert tmp_path.exists()
    assert "HF export failed after training" in caplog.text


def test_save_output_best_effort_writes_megatron_adapter_for_lora(tmp_path, monkeypatch):
    class BridgeShouldNotRun:
        def save_hf_pretrained(self, model, save_dir):
            raise AssertionError("Bridge export should be skipped for Megatron LoRA")

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            assert model_id == "resolved/m"
            assert kwargs == {"trust_remote_code": True}
            return cls()

        def save_pretrained(self, save_dir):
            Path(save_dir, "tokenizer_config.json").write_text("{}")

    class FakeChunk:
        def __init__(self):
            self.adapter = torch.nn.Parameter(torch.ones(2, 3))
            self.base = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        def named_parameters(self):
            return [
                (
                    "decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight",
                    self.adapter,
                ),
                ("decoder.layers.0.self_attention.linear_qkv.weight", self.base),
            ]

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=FakeTokenizer),
    )
    monkeypatch.setattr("yeto.models.resolve", lambda model: f"resolved/{model}")

    args = SimpleNamespace(
        model="m",
        model_revision=None,
        trust_remote_code=True,
        tuning="lora",
        lora_targets="attention",
        lora_r=8,
        lora_alpha=16,
        expert_parallel=1,
        tensor_parallel=1,
        pipeline_parallel=1,
    )

    assert ml._save_output_best_effort(BridgeShouldNotRun(), [FakeChunk()], tmp_path, args)

    meta_path = tmp_path / ml.MEGATRON_ADAPTER_METADATA_FILE
    meta = json.loads(meta_path.read_text())
    assert meta["kind"] == "yeto.megatron.adapter"
    assert meta["base_model_name_or_path"] == "resolved/m"
    assert meta["lora"] == {
        "r": 8,
        "alpha": 16,
        "targets": "attention",
        "target_modules": ml._ATTENTION_TARGETS,
    }
    assert meta["parameter_names"] == [
        "decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight"
    ]
    assert (tmp_path / meta["weights_file"]).exists()
    assert (tmp_path / "tokenizer_config.json").exists()


def test_save_output_best_effort_drops_partial_bridge_export_for_full_tuning(tmp_path):
    class PartialBridge:
        def save_hf_pretrained(self, model, save_dir):
            Path(save_dir, "model-00001-of-00001.safetensors").write_text("partial")
            raise AttributeError("'NoneType' object has no attribute 'megatron_to_hf'")

    args = SimpleNamespace(tuning="full")

    assert ml._save_output_best_effort(PartialBridge(), ["model"], tmp_path, args) is False
    assert not (tmp_path / ".bridge-export-tmp").exists()
    assert not (tmp_path / "model-00001-of-00001.safetensors").exists()
    assert not (tmp_path / ml.MEGATRON_ADAPTER_METADATA_FILE).exists()


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
