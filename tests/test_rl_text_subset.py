"""Text subset of a VL checkpoint: the rule, the refusals and the byte-exact write."""

from __future__ import annotations

import json
import os
import struct

import pytest

from yeto.rl.engine.text_subset import (
    NEMOTRON_H_OMNI,
    TextSubsetError,
    check_loaded,
    classify,
    plan_subset,
    rule_for_config,
    text_config,
)
from yeto.rl.engine.text_subset_materialize import (
    materialize,
    read_header,
    verify_tensor_bytes,
)

DTYPE_BYTES = {"BF16": 2, "F32": 4}


def _write_safetensors(path, tensors):
    """tensors: list of (name, dtype, shape, payload bytes)."""
    header, cursor = {}, 0
    for name, dtype, shape, payload in tensors:
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [cursor, cursor + len(payload)]}
        cursor += len(payload)
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((-(8 + len(blob))) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for _n, _d, _s, payload in tensors:
            fh.write(payload)


def _tensor(name, dtype, shape, fill):
    count = 1
    for dim in shape:
        count *= dim
    return (name, dtype, shape, bytes([fill]) * (count * DTYPE_BYTES[dtype]))


@pytest.fixture
def vl_checkpoint(tmp_path):
    """A tiny checkpoint shaped like the Nemotron VL one: text + MTP + vision."""
    src = tmp_path / "src"
    src.mkdir()
    shard_a = [
        _tensor("language_model.backbone.embeddings.weight", "BF16", (8, 4), 0x11),
        _tensor("language_model.backbone.layers.0.mixer.in_proj.weight", "BF16", (4, 4), 0x22),
        _tensor("language_model.backbone.layers.0.mixer.A_log", "F32", (4,), 0x33),
        _tensor("vision_model.blocks.0.attn.qkv.weight", "BF16", (6, 2), 0x44),
    ]
    shard_b = [
        _tensor("language_model.backbone.norm_f.weight", "BF16", (4,), 0x55),
        _tensor("language_model.lm_head.weight", "BF16", (8, 4), 0x66),
        _tensor("language_model.mtp.layers.0.mixer.in_proj.weight", "BF16", (4, 4), 0x77),
        _tensor("mlp1.0.weight", "BF16", (4, 2), 0x88),
        _tensor("vision_projector.weight", "BF16", (2,), 0x99),
    ]
    _write_safetensors(src / "model-00001-of-00002.safetensors", shard_a)
    _write_safetensors(src / "model-00002-of-00002.safetensors", shard_b)
    weight_map = {}
    for name, _d, _s, _p in shard_a:
        weight_map[name] = "model-00001-of-00002.safetensors"
    for name, _d, _s, _p in shard_b:
        weight_map[name] = "model-00002-of-00002.safetensors"
    (src / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map})
    )
    (src / "config.json").write_text(
        json.dumps(
            {
                "model_type": "nemotron_h_omni",
                "architectures": ["NemotronH_Omni_Reasoning_V3"],
                "projector_hidden_size": 16,
                "vision_config": {"model_type": "radio", "num_hidden_layers": 1},
                "llm_config": {
                    "model_type": "nemotron_h",
                    "architectures": ["NemotronHForCausalLM"],
                    "hidden_size": 4,
                    "num_nextn_predict_layers": 1,
                    "mtp_layers_block_type": ["attention", "moe"],
                },
            }
        )
    )
    (src / "tokenizer_config.json").write_text("{}")
    (src / "preprocessor_config.json").write_text("{}")
    return src


# ---------------------------------------------------------------- the rule


def test_rule_selected_by_outer_model_type(vl_checkpoint):
    config = json.loads((vl_checkpoint / "config.json").read_text())
    assert rule_for_config(config) is NEMOTRON_H_OMNI
    assert rule_for_config({"model_type": "nemotron_h"}) is None


def test_classify_splits_text_vision_and_mtp():
    rule = NEMOTRON_H_OMNI
    assert classify("language_model.backbone.layers.0.x", rule)[0] == "text"
    assert classify("vision_model.blocks.0.x", rule)[0] == "excluded_module"
    assert classify("mlp1.0.weight", rule)[0] == "excluded_module"
    assert classify("language_model.mtp.layers.0.x", rule)[0] == "excluded_text_block"


def test_unclassified_tensor_is_refused():
    with pytest.raises(TextSubsetError, match="matches neither"):
        classify("sound_tower.0.weight", NEMOTRON_H_OMNI)


def test_text_config_turns_off_the_excluded_block(vl_checkpoint):
    config = json.loads((vl_checkpoint / "config.json").read_text())
    out = text_config(config, NEMOTRON_H_OMNI)
    assert out["architectures"] == ["NemotronHForCausalLM"]
    assert out["model_type"] == "nemotron_h"
    assert out["num_nextn_predict_layers"] == 0
    assert out["mtp_layers_block_type"] == []
    assert "vision_config" not in out


def test_text_config_refuses_to_invent_an_architecture():
    with pytest.raises(TextSubsetError, match="names no architectures"):
        text_config({"llm_config": {"model_type": "nemotron_h"}}, NEMOTRON_H_OMNI)


# ---------------------------------------------------------------- the plan


def test_plan_strips_the_prefix_and_records_exclusions(vl_checkpoint):
    weight_map = json.loads((vl_checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    subset = plan_subset(weight_map.keys(), NEMOTRON_H_OMNI)
    assert subset.text_tensors == 5
    assert "backbone.embeddings.weight" in subset.renames.values()
    assert {k: len(v) for k, v in subset.excluded.items()} == {
        "vision_model.": 1,
        "mlp1.": 1,
        "vision_projector.": 1,
        "language_model.mtp.": 1,
    }


def test_plan_refuses_a_checkpoint_without_the_text_prefix():
    with pytest.raises(TextSubsetError, match="no tensor carries"):
        plan_subset(["vision_model.a", "mlp1.b"], NEMOTRON_H_OMNI)


# ---------------------------------------------------------------- the write


def test_materialize_writes_only_text_tensors(vl_checkpoint, tmp_path):
    dst = tmp_path / "dst"
    manifest = materialize(str(vl_checkpoint), str(dst))
    assert manifest["text_tensors"] == 5
    assert manifest["excluded_tensors"] == 4
    index = json.loads((dst / "model.safetensors.index.json").read_text())
    names = set(index["weight_map"])
    assert names == {
        "backbone.embeddings.weight",
        "backbone.layers.0.mixer.in_proj.weight",
        "backbone.layers.0.mixer.A_log",
        "backbone.norm_f.weight",
        "lm_head.weight",
    }
    assert not any("vision" in n or "mtp" in n or n.startswith("language_model") for n in names)
    config = json.loads((dst / "config.json").read_text())
    assert config["architectures"] == ["NemotronHForCausalLM"]
    assert config["num_nextn_predict_layers"] == 0


def test_materialize_copies_tensor_bytes_unchanged(vl_checkpoint, tmp_path):
    dst = tmp_path / "dst"
    materialize(str(vl_checkpoint), str(dst))
    checked, details = verify_tensor_bytes(str(vl_checkpoint), str(dst), sample=24)
    assert checked >= 2
    assert all(d["equal"] for d in details)


def test_materialize_preserves_dtype_and_shape(vl_checkpoint, tmp_path):
    dst = tmp_path / "dst"
    materialize(str(vl_checkpoint), str(dst))
    header, _start = read_header(str(dst / "model-00001-of-00002.safetensors"))
    assert header["backbone.layers.0.mixer.A_log"]["dtype"] == "F32"
    assert header["backbone.embeddings.weight"]["shape"] == [8, 4]


def test_materialize_carries_tokenizer_and_skips_vision_processor(vl_checkpoint, tmp_path):
    dst = tmp_path / "dst"
    manifest = materialize(str(vl_checkpoint), str(dst))
    assert "tokenizer_config.json" in manifest["carried_files"]
    assert not os.path.exists(dst / "preprocessor_config.json")


def test_manifest_records_a_sha256_per_output_shard(vl_checkpoint, tmp_path):
    dst = tmp_path / "dst"
    manifest = materialize(str(vl_checkpoint), str(dst))
    assert set(manifest["shards"]) == {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    for name, info in manifest["shards"].items():
        assert len(info["sha256"]) == 64
        assert info["file_bytes"] == os.path.getsize(dst / name)
    total = json.loads((dst / "model.safetensors.index.json").read_text())["metadata"]["total_size"]
    assert total == manifest["total_size"]


def test_materialize_refuses_a_plain_text_checkpoint(tmp_path):
    src = tmp_path / "plain"
    src.mkdir()
    (src / "config.json").write_text(json.dumps({"model_type": "nemotron_h"}))
    with pytest.raises(TextSubsetError, match="no text-subset rule"):
        materialize(str(src), str(tmp_path / "out"))


# ---------------------------------------------------------------- the check


def test_check_loaded_accepts_exactly_the_planned_set(vl_checkpoint):
    weight_map = json.loads((vl_checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    subset = plan_subset(weight_map.keys(), NEMOTRON_H_OMNI)
    loaded = {name: (2, 2) for name in subset.renames.values()}
    report = check_loaded(loaded, subset)
    assert report["tensors"] == 5
    assert report["params"] == 20


def test_check_loaded_fails_on_a_missing_parameter(vl_checkpoint):
    weight_map = json.loads((vl_checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    subset = plan_subset(weight_map.keys(), NEMOTRON_H_OMNI)
    loaded = {name: (2, 2) for name in subset.renames.values()}
    loaded.pop("lm_head.weight")
    with pytest.raises(TextSubsetError, match="1 missing"):
        check_loaded(loaded, subset)


def test_check_loaded_fails_when_a_vision_module_is_present(vl_checkpoint):
    """The check a rename-only workaround would pass and a real selection must not."""
    weight_map = json.loads((vl_checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    subset = plan_subset(weight_map.keys(), NEMOTRON_H_OMNI)
    loaded = {name: (2, 2) for name in subset.renames.values()}
    loaded["vision_model.blocks.0.attn.qkv.weight"] = (6, 2)
    with pytest.raises(TextSubsetError, match="would train and be published"):
        check_loaded(loaded, subset)


def test_check_loaded_fails_when_the_mtp_block_is_present(vl_checkpoint):
    weight_map = json.loads((vl_checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    subset = plan_subset(weight_map.keys(), NEMOTRON_H_OMNI)
    loaded = {name: (2, 2) for name in subset.renames.values()}
    loaded["mtp.layers.0.mixer.in_proj.weight"] = (4, 4)
    with pytest.raises(TextSubsetError, match="would train and be published"):
        check_loaded(loaded, subset)


def test_check_loaded_fails_on_the_wrong_parameter_total(vl_checkpoint):
    weight_map = json.loads((vl_checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    subset = plan_subset(weight_map.keys(), NEMOTRON_H_OMNI)
    loaded = {name: (2, 2) for name in subset.renames.values()}
    with pytest.raises(TextSubsetError, match="!= expected"):
        check_loaded(loaded, subset, expected_params=21)


def test_check_loaded_fails_on_a_reshaped_parameter(vl_checkpoint):
    weight_map = json.loads((vl_checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    subset = plan_subset(weight_map.keys(), NEMOTRON_H_OMNI)
    loaded = {name: (2, 2) for name in subset.renames.values()}
    with pytest.raises(TextSubsetError, match="shape the checkpoint does not record"):
        check_loaded(loaded, subset, shapes={"lm_head.weight": (8, 4)})
