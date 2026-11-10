"""restore_base_metadata: a transformers-5 HF export of the trainer is put back into
the metadata form of the base checkpoint (found by the Qwen3-0.6B trainer + vLLM 0.11
sampler rehearsal: the release's config.json had no rope_theta, so the sampler served
every release with the default RoPE base; mean KL 0.0007 -> 0.43)."""

from __future__ import annotations

import json

from yeto.rl.adapters.miles.release_plugin import full_checkpoint_writer, restore_base_metadata

BASE_CONFIG = {"architectures": ["Qwen3ForCausalLM"], "rope_theta": 1000000, "rope_scaling": None,
               "torch_dtype": "bfloat16", "tie_word_embeddings": True, "transformers_version": "4.51.0",
               "quantization_config": {"quant_method": "mxfp4"}}
EXPORTED_CONFIG = {"architectures": ["Qwen3ForCausalLM"], "dtype": "bfloat16", "tie_word_embeddings": True,
                   "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
                   "transformers_version": "5.19.0"}


def _dirs(tmp_path):
    base, export = tmp_path / "base", tmp_path / "r1-v1"
    base.mkdir()
    export.mkdir()
    (base / "config.json").write_text(json.dumps(BASE_CONFIG))
    (base / "tokenizer_config.json").write_text(json.dumps({"additional_special_tokens": ["<|im_start|>"]}))
    (base / "tokenizer.json").write_text("{}")
    (export / "config.json").write_text(json.dumps(EXPORTED_CONFIG))
    (export / "tokenizer_config.json").write_text(json.dumps({"extra_special_tokens": ["<|im_start|>"]}))
    (export / "model.safetensors").write_bytes(b"weights")
    return base, export


def test_dropped_config_keys_and_the_tokenizer_come_back_from_the_base(tmp_path):
    base, export = _dirs(tmp_path)
    restored = restore_base_metadata(export, base)
    config = json.loads((export / "config.json").read_text())
    assert config["rope_theta"] == 1000000 and config["torch_dtype"] == "bfloat16" and "rope_scaling" in config
    # what the export wrote is kept; the base's quantization is not put on a bf16 release
    assert config["rope_parameters"] == EXPORTED_CONFIG["rope_parameters"]
    assert config["transformers_version"] == "5.19.0" and "quantization_config" not in config
    assert sorted(restored["config_keys"]) == ["rope_scaling", "rope_theta", "torch_dtype"]
    assert json.loads((export / "tokenizer_config.json").read_text()) == {"additional_special_tokens": ["<|im_start|>"]}
    assert restored["tokenizer_files"] == ["tokenizer_config.json", "tokenizer.json"]
    assert (export / "model.safetensors").read_bytes() == b"weights"


def test_the_release_writer_restores_after_the_export(tmp_path):
    base, export = _dirs(tmp_path)
    calls = []

    class Runner:
        def run(self, call):
            calls.append(call)

    class Actor:
        def export_hf(self, *, rollout_id, path):
            return ("export_hf", rollout_id, path)

    full_checkpoint_writer(Actor(), Runner(), all_ranks=False, base_dir=base)(3, export)
    assert calls == [("export_hf", 3, str(export))]
    assert json.loads((export / "config.json").read_text())["rope_theta"] == 1000000
    # without a base directory the export is left as written
    (export / "config.json").write_text(json.dumps(EXPORTED_CONFIG))
    full_checkpoint_writer(Actor(), Runner(), all_ranks=False)(4, export)
    assert "rope_theta" not in json.loads((export / "config.json").read_text())
