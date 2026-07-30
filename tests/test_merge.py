import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto.merge import merge_adapter


def write_artifact(root: Path, model_dir: Path):
    root.mkdir()
    (root / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "base_model_name_or_path": str(model_dir.resolve()),
            }
        ),
        encoding="utf-8",
    )
    (root / "adapter_model.safetensors").write_bytes(b"adapter")
    (root / "yeto_provenance.json").write_text(
        json.dumps(
            {
                "artifact_kind": "causal-lm-training-output",
                "model": {
                    "source": "local",
                    "requested_identifier": str(model_dir),
                    "resolved_identifier": str(model_dir.resolve()),
                    "requested_revision": None,
                    "resolved_revision": None,
                },
            }
        ),
        encoding="utf-8",
    )


def test_merge_safe_merges_and_forwards_shard_size(tmp_path, monkeypatch):
    import peft
    import transformers

    model_dir = tmp_path / "base"
    model_dir.mkdir()
    adapter_dir = tmp_path / "adapter"
    write_artifact(adapter_dir, model_dir)
    output_dir = tmp_path / "merged"
    seen = {}

    class Base:
        def to(self, device):
            seen["device"] = str(device)
            return self

    class Merged:
        def save_pretrained(self, output, **kwargs):
            seen["save"] = kwargs
            Path(output, "model.safetensors").write_bytes(b"merged")

    class Wrapped:
        def merge_and_unload(self, safe_merge):
            seen["safe_merge"] = safe_merge
            return Merged()

    class Tokenizer:
        def save_pretrained(self, output):
            Path(output, "tokenizer.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda model, **kwargs: seen.update(model=model, load=kwargs) or Base(),
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda source, **kwargs: seen.update(tokenizer_source=str(source)) or Tokenizer(),
    )
    monkeypatch.setattr(
        peft.PeftModel,
        "from_pretrained",
        lambda base, adapter: seen.update(adapter=adapter) or Wrapped(),
    )

    result = merge_adapter(
        SimpleNamespace(
            adapter_dir=str(adapter_dir),
            output_dir=str(output_dir),
            model=None,
            model_revision=None,
            trust_remote_code=False,
            device="cpu",
            dtype="bf16",
            max_shard_size="2GB",
        )
    )

    assert result == output_dir
    assert seen["safe_merge"] is True
    assert seen["save"] == {"safe_serialization": True, "max_shard_size": "2GB"}
    assert seen["load"]["torch_dtype"].is_floating_point
    record = json.loads((output_dir / "yeto_provenance.json").read_text())
    assert record["artifact_kind"] == "causal-lm-merged-model"
    assert record["artifact"]["max_shard_size"] == "2GB"
    assert record["artifact"]["safe_merge"] is True


def test_merge_refuses_unsafe_or_existing_outputs(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "base_model_name_or_path": str((tmp_path / "base").resolve()),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"

    common = dict(
        adapter_dir=str(adapter),
        output_dir=str(output),
        model=str(tmp_path / "base"),
        model_revision=None,
        trust_remote_code=False,
        device="cpu",
        dtype="auto",
        max_shard_size="5GB",
    )
    with pytest.raises(ValueError, match="no safe adapter_model.safetensors"):
        merge_adapter(SimpleNamespace(**common))

    (adapter / "adapter_model.safetensors").write_bytes(b"safe")
    output.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        merge_adapter(SimpleNamespace(**common))
