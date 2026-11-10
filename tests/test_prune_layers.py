"""scripts/prune_layers.py on a small fake checkpoint (CPU, no torch, no download):
the first N layers, every expert, the embeddings and head stay; later layers and
draft tensors go; the config's layer tables follow; tensors move as bytes."""

from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "prune_layers.py"


@pytest.fixture(scope="module")
def pl():
    spec = importlib.util.spec_from_file_location("prune_layers", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]], metadata=None) -> None:
    """Write a safetensors file from ``{name: (dtype, shape, raw bytes)}``."""
    header: dict = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    offset = 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(encoded)))
        fh.write(encoded)
        for _dtype, _shape, data in tensors.values():
            fh.write(data)


def _k3_config(layers: int = 8) -> dict:
    full = [4, 8]  # 1-based: layers 3 and 7 are MLA/DSA
    return {"architectures": ["KimiK31ForConditionalGeneration"], "model_type": "kimi_k31",
            "text_config": {"num_hidden_layers": layers, "first_k_dense_replace": 1, "num_experts": 4,
                            "num_nextn_predict_layers": 0,
                            "linear_attn_config": {"full_attn_layers": full, "head_dim": 128,
                                                   "kda_layers": [i for i in range(1, layers + 1) if i not in full]},
                            "dsa_config": {"dsa_layers": full, "index_topk": 2048}},
            "vision_config": {"hidden_size": 16}}


def _tensor(name: str, fill: int, n: int = 16, dtype: str = "BF16") -> tuple[str, list[int], bytes]:
    width = {"BF16": 2, "U8": 1, "F32": 4}[dtype]
    return dtype, [n], bytes([fill % 256]) * (n * width)


def build_checkpoint(root: Path, layers: int = 8, experts: int = 4) -> dict[str, str]:
    """A tiny Kimi-K3-shaped checkpoint: one shard per layer plus two non-layer
    shards, the last of which mixes the head with a draft tensor; returns the
    weight map."""
    root.mkdir(parents=True)
    weight_map: dict[str, str] = {}
    full = {3, 7}
    for layer in range(layers):
        shard = f"model-{layer + 1:05d}-of-{layers + 2:05d}.safetensors"
        prefix = f"language_model.model.layers.{layer}"
        tensors = {f"{prefix}.input_layernorm.weight": _tensor("ln", layer)}
        if layer in full:
            tensors[f"{prefix}.self_attn.kv_a_proj_with_mqa.weight"] = _tensor("kv", layer)
            tensors[f"{prefix}.self_attn.indexer.wk.weight"] = _tensor("idx", layer)
        else:
            tensors[f"{prefix}.self_attn.A_log"] = _tensor("alog", layer, dtype="F32")
        if layer == 0:
            tensors[f"{prefix}.mlp.down_proj.weight"] = _tensor("mlp", layer)
        else:
            for e in range(experts):
                tensors[f"{prefix}.block_sparse_moe.experts.{e}.w1.weight_packed"] = _tensor("w1", layer * 10 + e, dtype="U8")
                tensors[f"{prefix}.block_sparse_moe.experts.{e}.w1.weight_scale"] = _tensor("s1", layer * 10 + e, dtype="U8")
            tensors[f"{prefix}.block_sparse_moe.gate.weight"] = _tensor("gate", layer)
        _safetensors(root / shard, tensors, metadata={"format": "pt"})
        weight_map.update({n: shard for n in tensors})
    shard_a = f"model-{layers + 1:05d}-of-{layers + 2:05d}.safetensors"
    tensors = {"language_model.model.embed_tokens.weight": _tensor("emb", 200),
               "vision_tower.encoder.blocks.0.wqkv.weight": _tensor("vis", 201),
               "mm_projector.proj.0.weight": _tensor("proj", 202)}
    _safetensors(root / shard_a, tensors)
    weight_map.update({n: shard_a for n in tensors})
    shard_b = f"model-{layers + 2:05d}-of-{layers + 2:05d}.safetensors"
    tensors = {"language_model.lm_head.weight": _tensor("head", 210),
               "language_model.model.norm.weight": _tensor("norm", 211),
               "language_model.model.draft.layers.0.eh_proj.weight": _tensor("draft", 212),
               "language_model.model.layers.42.mlp_res_norm.weight": _tensor("late", 213)}
    _safetensors(root / shard_b, tensors)
    weight_map.update({n: shard_b for n in tensors})
    (root / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": 0}, "weight_map": weight_map}))
    (root / "config.json").write_text(json.dumps(_k3_config(layers)))
    (root / "tokenizer_config.json").write_text("{}")
    (root / "configuration_kimi_k3.py").write_text("# model code\n")
    (root / "assets").mkdir()
    (root / "assets" / "logo.png").write_bytes(b"png")
    return weight_map


def _read(path: Path) -> dict[str, bytes]:
    with open(path, "rb") as fh:
        (length,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(length))
        data = fh.read()
    return {n: data[h["data_offsets"][0]:h["data_offsets"][1]] for n, h in header.items() if n != "__metadata__"}


def test_tensor_selection(pl):
    assert pl.layer_of("language_model.model.layers.12.self_attn.q_proj.weight") == 12
    assert pl.layer_of("vision_tower.encoder.blocks.3.wo.weight") is None
    assert pl.layer_of("language_model.lm_head.weight") is None
    assert pl.keep_tensor("language_model.model.layers.3.block_sparse_moe.experts.895.w2.weight_packed", 4)
    assert not pl.keep_tensor("language_model.model.layers.4.input_layernorm.weight", 4)
    assert not pl.keep_tensor("language_model.model.draft.layers.0.eh_proj.weight", 4)
    assert not pl.keep_tensor("model.layers.0.mtp.weight", 4)
    assert pl.keep_tensor("vision_tower.patch_embed.proj.weight", 4)
    assert not pl.keep_tensor("vision_tower.patch_embed.proj.weight", 4, drop_vision=True)


def test_config_tables_follow_the_kept_layers(pl):
    cfg = pl.prune_config(_k3_config(8), 4)
    text = cfg["text_config"]
    assert text["num_hidden_layers"] == 4 and text["num_nextn_predict_layers"] == 0
    assert text["linear_attn_config"]["full_attn_layers"] == [4]  # layer 3 (1-based 4) is the MLA layer
    assert text["linear_attn_config"]["kda_layers"] == [1, 2, 3]  # layer 0 (dense MLP) is KDA too
    assert text["dsa_config"]["dsa_layers"] == [4] and text["dsa_config"]["index_topk"] == 2048
    assert text["first_k_dense_replace"] == 1 and "vision_config" in cfg
    # the upstream 4-layer reference keeps the tables as they are
    kept = pl.prune_config(_k3_config(8), 4, keep_tables=True)["text_config"]
    assert kept["linear_attn_config"]["full_attn_layers"] == [4, 8] and kept["num_hidden_layers"] == 4
    # a flat config (no text_config) and generic tables
    flat = pl.prune_config({"num_hidden_layers": 6, "layer_types": list("abcdef"), "moe_layers": [1, 3, 5]}, 2)
    assert flat == {"num_hidden_layers": 2, "layer_types": ["a", "b"], "moe_layers": [1]}
    with pytest.raises(pl.PruneError, match="must be in 1..8"):
        pl.prune_config(_k3_config(8), 9)
    kinds = pl.kept_layer_kinds(_k3_config(8), 4)
    assert kinds == {0: "dense-mlp kda", 1: "kda", 2: "kda", 3: "mla"}


def test_prune_keeps_layers_experts_head_and_rewrites_mixed_shards(pl, tmp_path):
    src = tmp_path / "src"
    weight_map = build_checkpoint(src)
    dest = tmp_path / "dest"
    result = pl.prune(pl.LocalSource(src), dest, 4, log=lambda *_: None)
    index = json.loads((dest / "model.safetensors.index.json").read_text())
    kept = set(index["weight_map"])
    assert all(pl.layer_of(n) is None or pl.layer_of(n) < 4 for n in kept)
    assert not any("draft" in n for n in kept)
    assert "language_model.model.layers.42.mlp_res_norm.weight" not in kept
    assert {"language_model.lm_head.weight", "language_model.model.embed_tokens.weight",
            "language_model.model.norm.weight", "vision_tower.encoder.blocks.0.wqkv.weight"} <= kept
    experts = [n for n in kept if ".experts." in n]
    assert len(experts) == 3 * 4 * 2  # 3 MoE layers x 4 experts x (packed, scale)
    assert sorted(result["shards"]) == sorted(set(index["weight_map"].values()))
    # the layer shards and the embeddings shard are whole copies; the last shard was rewritten
    assert set(index["weight_map"].values()) == {f"model-{i:05d}-of-00010.safetensors" for i in (1, 2, 3, 4, 9, 10)}
    rewritten = _read(dest / "model-00010-of-00010.safetensors")
    assert set(rewritten) == {"language_model.lm_head.weight", "language_model.model.norm.weight"}
    assert rewritten["language_model.lm_head.weight"] == _read(src / "model-00010-of-00010.safetensors")[
        "language_model.lm_head.weight"]
    # bytes of every kept tensor are the source's bytes
    for name, shard in index["weight_map"].items():
        assert _read(dest / shard)[name] == _read(src / weight_map[name])[name]
    assert index["metadata"]["total_size"] == sum(len(_read(dest / s)[n]) for n, s in index["weight_map"].items())
    cfg = json.loads((dest / "config.json").read_text())
    assert cfg["text_config"]["num_hidden_layers"] == 4
    assert cfg["text_config"]["linear_attn_config"]["full_attn_layers"] == [4]
    # the other files travel; the source's index and config do not
    assert (dest / "tokenizer_config.json").exists() and (dest / "configuration_kimi_k3.py").exists()
    assert (dest / "assets" / "logo.png").read_bytes() == b"png"
    report = json.loads((dest / "PRUNE.json").read_text())
    assert report["layers"] == 4 and report["shards_rewritten"] == ["model-00010-of-00010.safetensors"]
    assert report["layer_kinds"] == {"0": "dense-mlp kda", "1": "kda", "2": "kda", "3": "mla"}


def test_dry_run_and_cli(pl, tmp_path):
    src = tmp_path / "src"
    build_checkpoint(src)
    out = subprocess.run([sys.executable, "-I", str(SCRIPT), str(src), str(tmp_path / "d"), "--layers", "2",
                          "--dry-run"], check=True, capture_output=True, text=True)
    plan = json.loads(out.stdout[out.stdout.index("{"):])
    assert plan["layer_kinds"] == {"0": "dense-mlp kda", "1": "kda"}
    assert not (tmp_path / "d").exists()
    bad = subprocess.run([sys.executable, "-I", str(SCRIPT), str(src), str(tmp_path / "d"), "--layers", "99"],
                         capture_output=True, text=True)
    assert bad.returncode == 1 and "must be in 1..8" in bad.stderr
    subprocess.run([sys.executable, "-I", str(SCRIPT), str(src), str(tmp_path / "d"), "--layers", "1",
                    "--drop-vision"], check=True, capture_output=True, text=True)
    index = json.loads((tmp_path / "d" / "model.safetensors.index.json").read_text())
    assert not any(n.startswith(("vision_tower.", "mm_projector.")) for n in index["weight_map"])
    assert "vision_config" not in json.loads((tmp_path / "d" / "config.json").read_text())


def test_hub_source_downloads_only_the_needed_shards(pl, tmp_path, monkeypatch):
    src = tmp_path / "src"
    build_checkpoint(src)
    fetched: list[str] = []

    class Hub(pl.HubSource):
        def list(self):
            return sorted(str(p.relative_to(src)) for p in src.rglob("*") if p.is_file())

        def file(self, name):
            fetched.append(name)
            return src / name

    pl.prune(Hub("org/model"), tmp_path / "dest", 4, log=lambda *_: None)
    shards = sorted(n for n in fetched if n.endswith(".safetensors"))
    assert shards == [f"model-{i:05d}-of-00010.safetensors" for i in (1, 2, 3, 4, 9, 10)]
    assert pl.open_source("hf://org/model").repo == "org/model"
    with pytest.raises(pl.PruneError, match="not a directory"):
        pl.open_source(str(tmp_path / "missing"))
