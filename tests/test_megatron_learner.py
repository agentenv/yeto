"""Non-GPU-testable parts of the Megatron learner: arg parsing and the data
adapter. The Megatron-Core training path is GPU/multi-node only and validated
by a live run, not here."""

import sys
import types
import json
from pathlib import Path
from types import SimpleNamespace

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


def test_validate_parallelism_allows_full_tuning_model_parallelism_without_syncer():
    ml._validate_parallelism(
        SimpleNamespace(
            tuning="full",
            syncer="none",
            tensor_parallel=4,
            pipeline_parallel=2,
        )
    )


def test_validate_parallelism_rejects_full_tuning_with_syncer():
    import pytest

    with pytest.raises(NotImplementedError, match="full-parameter tuning"):
        ml._validate_parallelism(
            SimpleNamespace(
                tuning="full",
                syncer="127.0.0.1:29400",
                tensor_parallel=1,
                pipeline_parallel=1,
            )
        )


def test_validate_parallelism_rejects_lora_tensor_parallelism():
    import pytest

    with pytest.raises(NotImplementedError, match="TP=1"):
        ml._validate_parallelism(
            SimpleNamespace(
                tuning="lora",
                syncer="none",
                tensor_parallel=2,
                pipeline_parallel=1,
            )
        )


def test_validate_parallelism_rejects_synced_lora_pipeline_parallelism():
    import pytest

    with pytest.raises(NotImplementedError, match="PP>1"):
        ml._validate_parallelism(
            SimpleNamespace(
                tuning="lora",
                syncer="127.0.0.1:29400",
                tensor_parallel=1,
                pipeline_parallel=2,
            )
        )


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


def test_build_model_disables_bridge_ddp_and_uses_lora_signature(monkeypatch):
    seen = {}

    class FakeBridge:
        @classmethod
        def from_hf_pretrained(cls, model_id, trust_remote_code=False):
            seen["model_id"] = model_id
            seen["trust_remote_code"] = trust_remote_code
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
    monkeypatch.setitem(sys.modules, "megatron.bridge", types.SimpleNamespace(AutoBridge=FakeBridge))
    monkeypatch.setitem(
        sys.modules,
        "megatron.bridge.peft.lora",
        types.SimpleNamespace(LoRA=FakeLoRA),
    )
    monkeypatch.setattr("yeto.models.resolve", lambda model: f"resolved/{model}")

    args = SimpleNamespace(
        model="m",
        tuning="lora",
        expert_parallel=1,
        pipeline_parallel=1,
        lora_targets="attention",
        lora_r=8,
        lora_alpha=16,
    )
    model, _bridge = ml._build_model(args, device=None)

    assert model == ["chunk"]
    assert seen["model_id"] == "resolved/m"
    assert seen["trust_remote_code"] is True
    assert seen["to_megatron"] == {"load_weights": True, "wrap_with_ddp": False}
    assert seen["lora"]["dim"] == 8 and seen["lora"]["alpha"] == 16
    assert "linear_qkv" in seen["lora"]["target_modules"]
    assert "share_expert_adapters" not in seen["lora"]
    assert seen["training"] is True


def test_build_model_full_tuning_skips_lora_and_enables_base_params(monkeypatch):
    import torch

    seen = {}

    class FakeBridge:
        @classmethod
        def from_hf_pretrained(cls, model_id, trust_remote_code=False):
            seen["model_id"] = model_id
            return cls()

        def to_megatron_model(self, load_weights=True, wrap_with_ddp=True):
            seen["to_megatron"] = {
                "load_weights": load_weights,
                "wrap_with_ddp": wrap_with_ddp,
            }
            return [torch.nn.Linear(2, 2)]

    class LoRAShouldNotImport:
        pass

    monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.bridge", types.SimpleNamespace(AutoBridge=FakeBridge))
    monkeypatch.setitem(
        sys.modules,
        "megatron.bridge.peft.lora",
        types.SimpleNamespace(LoRA=LoRAShouldNotImport),
    )
    monkeypatch.setattr("yeto.models.resolve", lambda model: f"resolved/{model}")

    args = SimpleNamespace(
        model="m",
        tuning="full",
        expert_parallel=1,
        pipeline_parallel=1,
        lora_targets="attention",
        lora_r=8,
        lora_alpha=16,
    )
    model, _bridge = ml._build_model(args, device=None)

    assert seen["model_id"] == "resolved/m"
    assert seen["to_megatron"] == {"load_weights": True, "wrap_with_ddp": False}
    assert all(p.requires_grad for p in model[0].parameters())


def test_build_dataset_uses_current_data_api(monkeypatch):
    calls = {}

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_id, trust_remote_code=False):
            calls["tokenizer"] = (model_id, trust_remote_code)
            return cls()

    def fake_build_packed_dataset(*args, **kwargs):
        calls["dataset"] = (args, kwargs)
        return ["packed"]

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=FakeTokenizer))
    monkeypatch.setattr("yeto.models.resolve", lambda model: f"resolved/{model}")
    monkeypatch.setattr("yeto.data.build_packed_dataset", fake_build_packed_dataset)

    args = SimpleNamespace(
        model="m",
        data="rows.jsonl",
        learner_id=2,
        num_learners=4,
        seq_len=128,
        max_rows=9,
        train_on="all",
    )

    assert ml._build_dataset(args) == ["packed"]
    assert calls["tokenizer"] == ("resolved/m", True)
    dataset_args, dataset_kwargs = calls["dataset"]
    assert dataset_args[0] == "rows.jsonl"
    assert isinstance(dataset_args[1], FakeTokenizer)
    assert dataset_args[2:] == (2, 4, 128, 9)
    assert dataset_kwargs == {"train_on": "all"}


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


def test_save_output_best_effort_exports_full_tuning_with_bridge(tmp_path):
    class WorkingBridge:
        def save_hf_pretrained(self, model, save_dir):
            Path(save_dir, "config.json").write_text("{}")
            Path(save_dir, "model.safetensors").write_text("weights")

    assert (
        ml._save_output_best_effort(
            WorkingBridge(),
            ["model"],
            tmp_path,
            SimpleNamespace(tuning="full"),
        )
        is True
    )
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "model.safetensors").exists()


def test_save_output_best_effort_writes_megatron_adapter_for_lora(tmp_path, monkeypatch):
    import torch

    class BridgeShouldNotRun:
        def save_hf_pretrained(self, model, save_dir):
            raise AssertionError("Bridge export should be skipped for Megatron LoRA")

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_id, trust_remote_code=False):
            assert model_id == "resolved/m"
            assert trust_remote_code is True
            return cls()

        def save_pretrained(self, save_dir):
            (Path(save_dir) / "tokenizer_config.json").write_text("{}")

    class FakeChunk:
        def __init__(self):
            self.adapter = torch.nn.Parameter(torch.ones(2, 3))
            self.base = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        def named_parameters(self):
            return [
                ("decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight", self.adapter),
                ("decoder.layers.0.self_attention.linear_qkv.weight", self.base),
            ]

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=FakeTokenizer))
    monkeypatch.setattr("yeto.models.resolve", lambda model: f"resolved/{model}")

    args = SimpleNamespace(
        model="m",
        tuning="lora",
        lora_targets="attention",
        lora_r=8,
        lora_alpha=16,
        expert_parallel=1,
        tensor_parallel=1,
        pipeline_parallel=1,
    )

    assert ml._save_output_best_effort(BridgeShouldNotRun(), [FakeChunk()], tmp_path, args) is True

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


def test_save_output_best_effort_uses_pipeline_gathered_state(tmp_path, monkeypatch):
    import torch

    class BridgeShouldNotRun:
        def save_hf_pretrained(self, model, save_dir):
            raise AssertionError("Bridge export should be skipped for Megatron LoRA")

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_id, trust_remote_code=False):
            return cls()

        def save_pretrained(self, save_dir):
            (Path(save_dir) / "tokenizer_config.json").write_text("{}")

    class LocalStageOnly:
        def named_parameters(self):
            return []

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=FakeTokenizer))
    monkeypatch.setattr("yeto.models.resolve", lambda model: f"resolved/{model}")

    args = SimpleNamespace(
        model="m",
        tuning="lora",
        lora_targets="attention",
        lora_r=8,
        lora_alpha=16,
        expert_parallel=8,
        tensor_parallel=1,
        pipeline_parallel=2,
    )
    state = {
        "decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight": torch.ones(1),
        "decoder.layers.40.self_attention.linear_qkv.adapter.linear_in.weight": torch.ones(1) * 2,
    }

    assert (
        ml._save_output_best_effort(
            BridgeShouldNotRun(),
            [LocalStageOnly()],
            tmp_path,
            args,
            state_override=state,
        )
        is True
    )

    meta = json.loads((tmp_path / ml.MEGATRON_ADAPTER_METADATA_FILE).read_text())
    assert meta["export"] == {"pipeline_stage_gathered": True}
    assert meta["parallelism"]["pipeline"] == 2
    assert meta["parameter_names"] == sorted(state)


def test_gather_adapter_state_for_export_merges_one_replica_per_pipeline_stage(monkeypatch):
    import torch
    import torch.distributed as dist

    stage0 = {"decoder.layers.0.self_attention.linear_proj.adapter.linear_in.weight": torch.ones(1)}
    stage1 = {"decoder.layers.40.self_attention.linear_proj.adapter.linear_in.weight": torch.ones(1) * 2}

    def fake_all_gather_object(out, payload):
        assert payload["pipeline_rank"] == 0
        out[:] = [
            {"pipeline_rank": 0, "state": stage0},
            None,
            {"pipeline_rank": 1, "state": stage1},
            None,
        ]

    monkeypatch.setattr(ml, "_parallel_rank", lambda name, default=0: 0)
    monkeypatch.setattr(ml, "_adapter_state_for_export", lambda model: stage0)
    monkeypatch.setattr(dist, "all_gather_object", fake_all_gather_object)

    args = SimpleNamespace(pipeline_parallel=2)
    merged = ml._gather_adapter_state_for_export(args, ["model"], rank=0, world=4)

    assert merged == {**stage0, **stage1}


def test_gather_adapter_state_for_export_deduplicates_replicated_stage_tensors(monkeypatch):
    import torch
    import torch.distributed as dist

    state = {"decoder.layers.0.self_attention.linear_proj.adapter.linear_in.weight": torch.ones(1)}

    def fake_all_gather_object(out, payload):
        out[:] = [
            {"pipeline_rank": 0, "state": state},
            {"pipeline_rank": 1, "state": state},
        ]

    monkeypatch.setattr(ml, "_parallel_rank", lambda name, default=0: 0)
    monkeypatch.setattr(ml, "_adapter_state_for_export", lambda model: state)
    monkeypatch.setattr(dist, "all_gather_object", fake_all_gather_object)

    args = SimpleNamespace(pipeline_parallel=2)
    assert ml._gather_adapter_state_for_export(args, ["model"], rank=0, world=2) == state


def test_save_output_best_effort_drops_partial_bridge_export_for_full_tuning(tmp_path, monkeypatch):
    import torch

    class PartialBridge:
        def save_hf_pretrained(self, model, save_dir):
            Path(save_dir, "model-00001-of-00001.safetensors").write_text("partial")
            raise AttributeError("'NoneType' object has no attribute 'megatron_to_hf'")

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_id, trust_remote_code=False):
            return cls()

        def save_pretrained(self, save_dir):
            Path(save_dir, "tokenizer_config.json").write_text("{}")

    class FakeChunk:
        def __init__(self):
            self.adapter = torch.nn.Parameter(torch.ones(1))

        def named_parameters(self):
            return [("decoder.layers.0.self_attention.linear_proj.adapter.linear_in.weight", self.adapter)]

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=FakeTokenizer))
    monkeypatch.setattr("yeto.models.resolve", lambda model: f"resolved/{model}")

    args = SimpleNamespace(
        model="m",
        tuning="full",
        lora_targets="attention",
        lora_r=8,
        lora_alpha=16,
        expert_parallel=1,
        tensor_parallel=1,
        pipeline_parallel=1,
    )

    assert ml._save_output_best_effort(PartialBridge(), [FakeChunk()], tmp_path, args) is False
    assert not (tmp_path / ".bridge-export-tmp").exists()
    assert not (tmp_path / "model-00001-of-00001.safetensors").exists()
    assert not (tmp_path / ml.MEGATRON_ADAPTER_METADATA_FILE).exists()


def test_save_output_best_effort_can_skip_bridge_export(tmp_path, monkeypatch):
    import torch

    class BridgeShouldNotRun:
        def save_hf_pretrained(self, model, save_dir):
            raise AssertionError("Bridge export should be skipped")

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_id, trust_remote_code=False):
            return cls()

        def save_pretrained(self, save_dir):
            Path(save_dir, "tokenizer_config.json").write_text("{}")

    class FakeChunk:
        def __init__(self):
            self.adapter = torch.nn.Parameter(torch.ones(1))

        def named_parameters(self):
            return [("decoder.layers.0.self_attention.linear_proj.adapter.linear_in.weight", self.adapter)]

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=FakeTokenizer))
    monkeypatch.setattr("yeto.models.resolve", lambda model: f"resolved/{model}")

    args = SimpleNamespace(
        model="m",
        tuning="lora",
        lora_targets="attention",
        lora_r=8,
        lora_alpha=16,
        expert_parallel=8,
        tensor_parallel=1,
        pipeline_parallel=1,
    )

    assert (
        ml._save_output_best_effort(
            BridgeShouldNotRun(),
            [FakeChunk()],
            tmp_path,
            args,
            prefer_adapter_artifact=True,
        )
        is True
    )
    assert (tmp_path / ml.MEGATRON_ADAPTER_METADATA_FILE).exists()
