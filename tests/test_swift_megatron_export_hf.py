"""Export of a syncer DiLoCo state to HF (yeto/swift_megatron/export_hf.py).

Megatron-Core and mcore_bridge do not run here, so the model side is a tiny
torch module that carries exactly the attributes the export reads: pipeline
layers with a global ``layer_number``, an expert container with
``local_expert_indices``, a fused gated ``linear_fc1`` and a fused Mamba
``in_proj``, and a config with the ``mamba_*`` sizes.  Everything else (the
syncer's two file formats, the layout rebuild, the TP de-interleave) is
checked against bytes and tensors written by the test itself.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
import torch
from torch import nn

from yeto.fragments import build_layout
from yeto.protocol import layout_fingerprint
from yeto.swift_megatron.export_hf import (
    CKPT_MAGIC_V3,
    ExportError,
    GlobalTableFile,
    apply_to_mcore,
    build_layout_from_table,
    export_name_map,
    fused_chunk_sizes,
    llm_only_key,
    named_tensors,
    read_final_state,
    read_syncer_checkpoint,
    read_syncer_state,
    to_single_process,
    tp_deinterleave,
)
from yeto.swift_megatron.sync import OUTPUT_LAYER_AVG_REGEX

TP = 4
HEADS, HEAD_DIM, GROUPS, STATE = 8, 4, 2, 6          # mamba sizes of the fake model
INNER, BC = HEADS * HEAD_DIM, GROUPS * STATE         # 32, 12
IN_PROJ_ROWS = INNER + INNER + BC + BC + HEADS       # z, x, B, C, dt = 32+32+12+12+8 = 96
CONV1D_ROWS = INNER + BC + BC                        # x, B, C = 56
HIDDEN = 16
FFN = 8                                              # per gate/up half -> fc1 rows = 16


def shapes_table(fragments: int = 3, tp_size: int = TP) -> GlobalTableFile:
    """The island's contract for the fake model below."""

    shapes = {
        "embedding.word_embeddings.weight": (24, HIDDEN),
        "output_layer.weight": (24, HIDDEN),
        "decoder.layers.0.mixer.in_proj.weight": (IN_PROJ_ROWS, HIDDEN),
        "decoder.layers.0.mixer.conv1d.weight": (CONV1D_ROWS, 1, 4),
        "decoder.layers.0.mixer.norm.weight": (INNER,),
        "decoder.layers.1.mlp.linear_fc1.weight": (2 * FFN, HIDDEN),
        "decoder.layers.1.mlp.linear_fc2.weight": (HIDDEN, FFN),
        "decoder.layers.2.mlp.experts.local_experts.0.linear_fc1.weight": (2 * FFN, HIDDEN),
        "decoder.layers.2.mlp.experts.local_experts.1.linear_fc1.weight": (2 * FFN, HIDDEN),
    }
    part_dims = {
        "embedding.word_embeddings.weight": 0,
        "output_layer.weight": 0,
        "decoder.layers.0.mixer.in_proj.weight": 0,
        "decoder.layers.0.mixer.conv1d.weight": 0,
        "decoder.layers.0.mixer.norm.weight": 0,
        "decoder.layers.1.mlp.linear_fc1.weight": 0,
        "decoder.layers.1.mlp.linear_fc2.weight": 1,
        "decoder.layers.2.mlp.experts.local_experts.0.linear_fc1.weight": 0,
        "decoder.layers.2.mlp.experts.local_experts.1.linear_fc1.weight": 0,
    }
    expert = {name: ".experts." in name for name in shapes}
    return GlobalTableFile(shapes=shapes, part_dims=part_dims, expert=expert,
                           tp_size=tp_size, etp_size=1, fragments=fragments,
                           fragment_pattern="binpack")


def reference_tensors(table: GlobalTableFile) -> dict[str, torch.Tensor]:
    """A distinct, deterministic value per tensor."""

    out = {}
    for index, name in enumerate(sorted(table.shapes)):
        numel = table.numel(name)
        out[name] = (torch.arange(numel, dtype=torch.float32) + 1000.0 * (index + 1)).view(
            table.shapes[name])
    return out


def pack_fragments(table: GlobalTableFile, tensors: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    layout = build_layout_from_table(table)
    return [torch.cat([tensors[name].reshape(-1) for name, _ in fragment.tensors])
            for fragment in layout.fragments]


def write_final_state(path: Path, fragments: list[torch.Tensor]) -> None:
    with open(path, "wb") as handle:
        handle.write(struct.pack("<I", len(fragments)))
        for fragment in fragments:
            handle.write(struct.pack("<Q", fragment.numel()))
            handle.write(fragment.contiguous().numpy().tobytes())


def write_checkpoint(path: Path, fragments: list[torch.Tensor], fingerprint: bytes,
                     global_step: int = 77) -> None:
    with open(path, "wb") as handle:
        handle.write(struct.pack("<I", CKPT_MAGIC_V3))
        handle.write(b"\x00")
        handle.write(fingerprint)
        handle.write(struct.pack("<Q", global_step))
        handle.write(struct.pack("<I", len(fragments)))
        for index, fragment in enumerate(fragments):
            handle.write(struct.pack("<Q", index + 1))          # version
            handle.write(struct.pack("<Q", fragment.numel()))
            handle.write(fragment.contiguous().numpy().tobytes())
            handle.write((-fragment).contiguous().numpy().tobytes())   # momentum, skipped
        handle.write(struct.pack("<I", 0))                      # empty ledger


# ----------------------------------------------------------------- file formats
def test_final_state_round_trips_through_the_rebuilt_layout(tmp_path):
    table = shapes_table()
    tensors = reference_tensors(table)
    write_final_state(tmp_path / "final.f32", pack_fragments(table, tensors))

    state = read_final_state(str(tmp_path / "final.f32"))
    assert len(state.fragments) == build_layout_from_table(table).num_fragments
    assert state.global_step is None
    out = named_tensors(state, table)
    assert set(out) == set(tensors)
    for name, value in tensors.items():
        assert torch.equal(out[name], value), name


def test_checkpoint_carries_step_versions_and_skips_the_momentum(tmp_path):
    table = shapes_table()
    tensors = reference_tensors(table)
    fragments = pack_fragments(table, tensors)
    layout = build_layout_from_table(table)
    write_checkpoint(tmp_path / "s.ckpt", fragments, layout_fingerprint(layout), global_step=512)

    state = read_syncer_checkpoint(str(tmp_path / "s.ckpt"))
    assert state.global_step == 512
    assert state.versions == list(range(1, len(fragments) + 1))
    out = named_tensors(state, table)
    for name, value in tensors.items():
        assert torch.equal(out[name], value), name


def test_a_checkpoint_fingerprint_from_another_layout_is_refused(tmp_path):
    table = shapes_table()
    fragments = pack_fragments(table, reference_tensors(table))
    write_checkpoint(tmp_path / "s.ckpt", fragments, bytes(32))
    state = read_syncer_checkpoint(str(tmp_path / "s.ckpt"))
    with pytest.raises(ExportError, match="layout fingerprint"):
        named_tensors(state, table)


def test_a_table_with_a_different_fragment_count_is_refused(tmp_path):
    table = shapes_table(fragments=3)
    write_final_state(tmp_path / "final.f32", pack_fragments(table, reference_tensors(table)))
    state = read_final_state(str(tmp_path / "final.f32"))
    with pytest.raises(ExportError, match="does not describe this run"):
        named_tensors(state, shapes_table(fragments=5))


def test_a_truncated_artifact_is_refused(tmp_path):
    table = shapes_table()
    write_final_state(tmp_path / "final.f32", pack_fragments(table, reference_tensors(table)))
    data = (tmp_path / "final.f32").read_bytes()
    (tmp_path / "cut.f32").write_bytes(data[: len(data) - 17])
    with pytest.raises(ExportError, match="truncated"):
        read_final_state(str(tmp_path / "cut.f32"))


def test_exactly_one_source_is_required(tmp_path):
    with pytest.raises(ExportError, match="exactly one"):
        read_syncer_state()
    with pytest.raises(ExportError, match="exactly one"):
        read_syncer_state(checkpoint="a", final_state="b")


def test_the_table_file_round_trips(tmp_path):
    table = shapes_table()
    table.dump(str(tmp_path / "t.json"))
    again = GlobalTableFile.load(str(tmp_path / "t.json"))
    assert again.shapes == table.shapes
    assert again.part_dims == table.part_dims
    assert again.tp_size == table.tp_size and again.etp_size == table.etp_size
    assert again.fragments == table.fragments
    assert again.tp_of("decoder.layers.2.mlp.experts.local_experts.0.linear_fc1.weight") == 1
    assert again.tp_of("decoder.layers.1.mlp.linear_fc1.weight") == TP
    body = json.loads((tmp_path / "t.json").read_text())
    body["schema"] = "something-else"
    (tmp_path / "bad.json").write_text(json.dumps(body))
    with pytest.raises(ExportError, match="not a"):
        GlobalTableFile.load(str(tmp_path / "bad.json"))


def test_the_rebuilt_layout_is_the_one_the_sync_builds():
    table = shapes_table()
    names = sorted(table.shapes)
    expected = build_layout(
        [(n, table.numel(n)) for n in names], table.fragments, "binpack",
        avg_name_regex=OUTPUT_LAYER_AVG_REGEX,
        named_shapes={n: table.shapes[n] for n in names})
    got = build_layout_from_table(table)
    assert [f.tensors for f in got.fragments] == [f.tensors for f in expected.fragments]
    assert [f.merge_mode for f in got.fragments] == [f.merge_mode for f in expected.fragments]


# --------------------------------------------------------------- TP interleave
def megatron_shard_and_concat(whole: torch.Tensor, part_dim: int, tp_size: int,
                              chunk_sizes: list[int]) -> torch.Tensor:
    """What the island's full tensor looks like: Megatron cuts each chunk into
    ``tp_size`` pieces, rank r holds piece r of every chunk, and the sync
    concatenates the ranks' shards in rank order."""

    offsets, position = [], 0
    for size in chunk_sizes:
        offsets.append((position, size))
        position += size
    shards = []
    for rank in range(tp_size):
        pieces = [whole.narrow(part_dim, start + rank * (size // tp_size), size // tp_size)
                  for start, size in offsets]
        shards.append(torch.cat(pieces, dim=part_dim))
    return torch.cat(shards, dim=part_dim)


@pytest.mark.parametrize("chunks", [[8, 8], [32, 32, 12, 12, 8], [32, 12, 12]])
def test_tp_deinterleave_inverts_megatron_chunk_wise_sharding(chunks):
    rows = sum(chunks)
    whole = torch.arange(rows * 5, dtype=torch.float32).view(rows, 5)
    island = megatron_shard_and_concat(whole, 0, TP, chunks)
    assert not torch.equal(island, whole)                    # the interleave is real
    assert torch.equal(tp_deinterleave(island, 0, TP, chunks), whole)


def test_tp_deinterleave_is_the_identity_without_fusion_or_tp():
    tensor = torch.arange(24, dtype=torch.float32).view(8, 3)
    assert torch.equal(tp_deinterleave(tensor, 0, 1, [4, 4]), tensor)
    assert torch.equal(tp_deinterleave(tensor, 0, TP, [8]), tensor)


def test_tp_deinterleave_refuses_sizes_it_cannot_cut():
    tensor = torch.zeros(8, 3)
    with pytest.raises(ExportError, match="do not sum"):
        tp_deinterleave(tensor, 0, TP, [4, 3])
    with pytest.raises(ExportError, match="not divisible"):
        tp_deinterleave(tensor, 0, 3, [4, 4])


class FakeConfig:
    gated_linear_unit = True
    mamba_num_heads = HEADS
    mamba_head_dim = HEAD_DIM
    mamba_num_groups = GROUPS
    mamba_state_dim = STATE


def test_fused_chunk_sizes_come_from_the_config():
    config = FakeConfig()
    assert fused_chunk_sizes("decoder.layers.1.mlp.linear_fc1.weight", (2 * FFN, HIDDEN), config) \
        == [FFN, FFN]
    assert fused_chunk_sizes("...experts.linear_fc1.weight3", (2 * FFN, HIDDEN), config) == [FFN, FFN]
    assert fused_chunk_sizes("decoder.layers.0.mixer.in_proj.weight", (IN_PROJ_ROWS, HIDDEN), config) \
        == [INNER, INNER, BC, BC, HEADS]
    assert fused_chunk_sizes("decoder.layers.0.mixer.conv1d.weight", (CONV1D_ROWS, 1, 4), config) \
        == [INNER, BC, BC]
    assert fused_chunk_sizes("decoder.layers.1.mlp.linear_fc2.weight", (HIDDEN, FFN), config) is None
    assert fused_chunk_sizes("output_layer.weight", (24, HIDDEN), config) is None


def test_a_non_gated_fc1_is_not_treated_as_fused():
    class Dense(FakeConfig):
        gated_linear_unit = False

    assert fused_chunk_sizes("decoder.layers.1.mlp.linear_fc1.weight", (FFN, HIDDEN), Dense()) is None


def test_a_mamba_tensor_whose_rows_contradict_the_config_is_refused():
    with pytest.raises(ExportError, match="in_proj rows"):
        fused_chunk_sizes("a.in_proj.weight", (IN_PROJ_ROWS + 1, HIDDEN), FakeConfig())


def test_to_single_process_only_permutes_fused_tp_tensors():
    config = FakeConfig()
    whole = torch.arange(2 * FFN * HIDDEN, dtype=torch.float32).view(2 * FFN, HIDDEN)
    island = megatron_shard_and_concat(whole, 0, TP, [FFN, FFN])
    name = "decoder.layers.1.mlp.linear_fc1.weight"
    assert torch.equal(to_single_process(name, island, 0, TP, config), whole)
    assert torch.equal(to_single_process(name, island, 0, 1, config), island)
    assert torch.equal(to_single_process(name, island, -1, TP, config), island)
    plain = "decoder.layers.1.mlp.linear_fc2.weight"
    assert torch.equal(to_single_process(plain, island, 1, TP, config), island)


# ------------------------------------------------------------------ mcore side
class Layer(nn.Module):
    def __init__(self, layer_number: int):
        super().__init__()
        self.layer_number = layer_number


class MambaLayer(Layer):
    def __init__(self, layer_number: int):
        super().__init__(layer_number)
        self.mixer = nn.Module()
        self.mixer.in_proj = nn.Linear(HIDDEN, IN_PROJ_ROWS, bias=False)
        self.mixer.conv1d = nn.Conv1d(CONV1D_ROWS, CONV1D_ROWS, 4, groups=CONV1D_ROWS, bias=False)
        self.mixer.norm = nn.Module()
        self.mixer.norm.weight = nn.Parameter(torch.zeros(INNER))


class DenseLayer(Layer):
    def __init__(self, layer_number: int):
        super().__init__(layer_number)
        self.mlp = nn.Module()
        self.mlp.linear_fc1 = nn.Linear(HIDDEN, 2 * FFN, bias=False)
        self.mlp.linear_fc2 = nn.Linear(FFN, HIDDEN, bias=False)


class Experts(nn.Module):
    def __init__(self, indices: list[int]):
        super().__init__()
        self.local_expert_indices = indices
        self.local_experts = nn.ModuleList()
        for _ in indices:
            expert = nn.Module()
            expert.linear_fc1 = nn.Linear(HIDDEN, 2 * FFN, bias=False)
            self.local_experts.append(expert)


class MoELayer(Layer):
    def __init__(self, layer_number: int, indices: list[int]):
        super().__init__(layer_number)
        self.mlp = nn.Module()
        self.mlp.experts = Experts(indices)


class FakeModel(nn.Module):
    """A whole (PP=1, EP=1) export model: local names == global names."""

    def __init__(self, expert_indices=(0, 1)):
        super().__init__()
        self.config = FakeConfig()
        self.embedding = nn.Module()
        self.embedding.word_embeddings = nn.Embedding(24, HIDDEN)
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([
            MambaLayer(1), DenseLayer(2), MoELayer(3, list(expert_indices))])
        self.output_layer = nn.Linear(HIDDEN, 24, bias=False)


def test_export_name_map_uses_global_layer_and_expert_indices():
    names = set(export_name_map([FakeModel()]))
    assert "decoder.layers.0.mixer.in_proj.weight" in names        # layer_number 1 -> index 0
    assert "decoder.layers.2.mlp.experts.local_experts.1.linear_fc1.weight" in names
    assert names == set(shapes_table().shapes)


def test_export_name_map_renumbers_a_pipeline_and_expert_sharded_chunk():
    """A chunk that is *not* whole still yields global names: layer_number and
    local_expert_indices decide, which is what makes the lookup safe."""

    model = FakeModel(expert_indices=(5, 6))
    model.decoder.layers[2].layer_number = 41
    names = set(export_name_map([model]))
    assert "decoder.layers.40.mlp.experts.local_experts.5.linear_fc1.weight" in names
    assert "decoder.layers.40.mlp.experts.local_experts.6.linear_fc1.weight" in names


def island_state(table: GlobalTableFile, whole: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """What the syncer holds: each full tensor in the island's TP order."""

    config = FakeConfig()
    out = {}
    for name, value in whole.items():
        part_dim = table.part_dims[name]
        chunks = fused_chunk_sizes(name, tuple(value.shape), config)
        tp_size = table.tp_of(name)
        if chunks is None or tp_size <= 1 or part_dim < 0:
            out[name] = value
        else:
            out[name] = megatron_shard_and_concat(value, part_dim, tp_size, chunks)
    return out


def test_apply_to_mcore_writes_every_tensor_and_undoes_the_tp_interleave():
    table = shapes_table()
    model = FakeModel()
    whole = {name: torch.randn(table.shapes[name]) for name in table.shapes}
    stats = apply_to_mcore([model], island_state(table, whole), table)

    assert stats["written"] == len(table.shapes)
    assert stats["not_synchronized"] == 0
    params = export_name_map([model])
    for name, value in whole.items():
        assert torch.allclose(params[name].detach(), value, atol=0, rtol=0), name


def test_apply_to_mcore_leaves_unsynchronized_parameters_alone():
    """The frozen vision tower is not in the layout; its weights must survive."""

    table = shapes_table()
    model = FakeModel()
    model.vision_projector = nn.Linear(HIDDEN, HIDDEN, bias=False)
    before = model.vision_projector.weight.detach().clone()
    whole = {name: torch.randn(table.shapes[name]) for name in table.shapes}
    stats = apply_to_mcore([model], island_state(table, whole), table)
    assert stats["not_synchronized"] == 1
    assert torch.equal(model.vision_projector.weight.detach(), before)


def test_apply_to_mcore_refuses_a_tensor_the_model_does_not_have():
    table = shapes_table()
    tensors = island_state(table, {name: torch.zeros(table.shapes[name]) for name in table.shapes})
    tensors["decoder.layers.9.mlp.linear_fc2.weight"] = torch.zeros(HIDDEN, FFN)
    table.shapes["decoder.layers.9.mlp.linear_fc2.weight"] = (HIDDEN, FFN)
    table.part_dims["decoder.layers.9.mlp.linear_fc2.weight"] = 1
    with pytest.raises(ExportError, match="not in the export model"):
        apply_to_mcore([FakeModel()], tensors, table)
    stats = apply_to_mcore([FakeModel()], tensors, table, strict=False)
    assert stats["missing_in_model"] == 1
    assert stats["written"] == len(shapes_table().shapes)


def test_apply_to_mcore_refuses_a_non_finite_state():
    table = shapes_table()
    tensors = island_state(table, {name: torch.zeros(table.shapes[name]) for name in table.shapes})
    tensors["output_layer.weight"] = tensors["output_layer.weight"].clone()
    tensors["output_layer.weight"][0, 0] = float("nan")
    with pytest.raises(ExportError, match="non-finite"):
        apply_to_mcore([FakeModel()], tensors, table)


def test_apply_to_mcore_refuses_a_shape_mismatch():
    table = shapes_table()
    tensors = island_state(table, {name: torch.zeros(table.shapes[name]) for name in table.shapes})
    tensors["decoder.layers.1.mlp.linear_fc2.weight"] = torch.zeros(HIDDEN, FFN + 1)
    with pytest.raises(ExportError, match="model shape"):
        apply_to_mcore([FakeModel()], tensors, table)


def test_the_whole_path_from_a_checkpoint_to_the_model(tmp_path):
    table = shapes_table()
    model = FakeModel()
    whole = {name: torch.randn(table.shapes[name]) for name in table.shapes}
    fragments = pack_fragments(table, island_state(table, whole))
    layout = build_layout_from_table(table)
    write_checkpoint(tmp_path / "s.ckpt", fragments, layout_fingerprint(layout), global_step=3300)

    state = read_syncer_state(checkpoint=str(tmp_path / "s.ckpt"))
    apply_to_mcore([model], named_tensors(state, table), table)
    params = export_name_map([model])
    for name, value in whole.items():
        assert torch.allclose(params[name].detach(), value), name


# ------------------------------------------------------------------- LLM-only
def test_llm_only_key_matches_the_production_export_pair():
    assert llm_only_key("language_model.backbone.embeddings.weight") == "backbone.embeddings.weight"
    assert llm_only_key("language_model.lm_head.weight") == "lm_head.weight"
    assert llm_only_key("vision_model.encoder.layers.0.attn.qkv.weight") is None
    assert llm_only_key("vision_projector.0.weight") is None
    assert llm_only_key("mlp1.0.weight") is None
    assert llm_only_key("backbone.norm_f.weight") == "backbone.norm_f.weight"


def test_strip_vision_rewrites_the_shards_and_the_index(tmp_path):
    from safetensors.torch import save_file

    from yeto.swift_megatron.export_hf import main

    source = tmp_path / "vl"
    source.mkdir()
    shard_a = {"language_model.backbone.embeddings.weight": torch.zeros(4, 2),
               "vision_model.a.weight": torch.ones(3)}
    shard_b = {"language_model.lm_head.weight": torch.full((4, 2), 2.0),
               "mlp1.0.weight": torch.ones(2)}
    save_file(shard_a, str(source / "model-00001-of-00002.safetensors"), metadata={"format": "pt"})
    save_file(shard_b, str(source / "model-00002-of-00002.safetensors"), metadata={"format": "pt"})
    weight_map = {k: "model-00001-of-00002.safetensors" for k in shard_a}
    weight_map.update({k: "model-00002-of-00002.safetensors" for k in shard_b})
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (source / "config.json").write_text("{}")

    out = tmp_path / "llm"
    assert main(["strip-vision", "--hf-dir", str(source), "--output-dir", str(out),
                 "--config-from", str(source)]) == 0
    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == {"backbone.embeddings.weight", "lm_head.weight"}
    assert index["metadata"]["total_size"] == 2 * 4 * 2 * 4
    assert (out / "config.json").exists()

    from safetensors import safe_open

    with safe_open(str(out / "model-00002-of-00002.safetensors"), framework="pt") as handle:
        assert torch.equal(handle.get_tensor("lm_head.weight"), torch.full((4, 2), 2.0))
