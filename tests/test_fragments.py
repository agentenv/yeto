from decoupled_diloco.fragments import MERGE_AVG, MERGE_RDA, build_layout


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
