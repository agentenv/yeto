import pytest

from yeto.fragments import MERGE_AVG, MERGE_RDA, build_layout, layer_index


TENSORS = [
    ("model.embed_tokens.weight", 1000),
    ("model.layers.0.q.weight", 64),
    ("model.layers.0.mlp.weight", 256),
    ("model.layers.1.q.weight", 64),
    ("model.layers.1.mlp.weight", 256),
    ("lm_head.weight", 1000),
]


def test_embedding_isolated_with_avg():
    layout = build_layout(TENSORS, 4)
    embed = layout.fragments[0]
    assert embed.merge_mode == MERGE_AVG
    assert {n for n, _ in embed.tensors} == {"model.embed_tokens.weight", "lm_head.weight"}
    assert all(f.merge_mode == MERGE_RDA for f in layout.fragments[1:])


def test_balanced_packing():
    layout = build_layout(TENSORS, 3)  # 1 embed + 2 RDA bins
    sizes = [f.numel for f in layout.fragments[1:]]
    assert sorted(sizes) == [320, 320]  # 256+64 in each bin


def test_deterministic():
    a = build_layout(TENSORS, 4)
    b = build_layout(list(reversed(TENSORS)), 4)
    assert [f.tensors for f in a.fragments] == [f.tensors for f in b.fragments]


def test_shrinks_when_few_tensors():
    layout = build_layout(TENSORS[:2], 24)
    assert layout.num_fragments == 2
    assert layout.tensor_names() == ["model.embed_tokens.weight", "model.layers.0.q.weight"]


def test_all_tensors_present_once():
    layout = build_layout(TENSORS, 4)
    assert sorted(layout.tensor_names()) == sorted(n for n, _ in TENSORS)


def test_layer_index():
    assert layer_index("base.model.layers.12.q.lora_A.weight") == 12
    assert layer_index("transformer.h.3.attn.weight") == 3
    assert layer_index("model.blocks.0.mlp.weight") == 0
    assert layer_index("model.final_norm.weight") is None


STRIDED_TENSORS = [
    ("model.embed_tokens.weight", 1000),
    ("model.final_norm.weight", 8),
] + [(f"model.layers.{i}.{part}.weight", size) for i in range(6) for part, size in (("q", 64), ("mlp", 256))]


def test_strided_interleaves_layers():
    # 6 layers over 2 RDA bins: even layers -> bin 0, odd layers -> bin 1.
    layout = build_layout(STRIDED_TENSORS, 3, pattern="strided")
    bins = layout.fragments[1:]
    layers_per_bin = [
        sorted({layer_index(n) for n, _ in b.tensors if layer_index(n) is not None})
        for b in bins
    ]
    assert layers_per_bin == [[0, 2, 4], [1, 3, 5]]


def test_strided_keeps_layer_tensors_together():
    layout = build_layout(STRIDED_TENSORS, 4, pattern="strided")
    for frag in layout.fragments[1:]:
        by_layer = {}
        for name, _ in frag.tensors:
            if layer_index(name) is not None:
                by_layer.setdefault(layer_index(name), []).append(name)
        for names in by_layer.values():
            assert len(names) == 2  # q + mlp of a layer never split


def test_strided_places_layerless_tensors_and_keeps_all():
    layout = build_layout(STRIDED_TENSORS, 3, pattern="strided")
    assert sorted(layout.tensor_names()) == sorted(n for n, _ in STRIDED_TENSORS)


def test_strided_deterministic():
    a = build_layout(STRIDED_TENSORS, 3, pattern="strided")
    b = build_layout(list(reversed(STRIDED_TENSORS)), 3, pattern="strided")
    assert [f.tensors for f in a.fragments] == [f.tensors for f in b.fragments]


def test_strided_shrinks_empty_bins():
    # 2 distinct layers cannot fill 5 bins; empties are dropped.
    layout = build_layout(TENSORS, 6, pattern="strided")
    assert layout.num_fragments == 3  # embed + one bin per layer
    assert all(f.tensors for f in layout.fragments)


def test_unknown_pattern_rejected():
    with pytest.raises(ValueError):
        build_layout(TENSORS, 4, pattern="zigzag")
