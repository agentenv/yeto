"""Offline Miles full-parameter layout of Nemotron-H actors (no GPU, no weights).

The Nano 30B-A3B case is the island layout measured on H200 (S27
``s27-gr-probe-b``: ``[yeto-layout-probe]`` record and the layout check
error; the same hash ran in ``s27-gr-j``). The 120B case uses the real
Nemotron-3.5 Super VL config.json; its totals are checked against the
checkpoint's safetensors headers (revision fb0ee0bc, headers only).
"""

from __future__ import annotations

import json
from math import prod
from pathlib import Path

import pytest

from yeto.rl.adapters.miles.models import nemotron_h_layout as L
from yeto.rl.hybrid_params import check_expert_coverage, check_frozen_excluded
from yeto.rl.local_learner import ParameterSpec
from yeto.rl.moe_layout import check_expert_partition

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "nemotron_h"
NANO = json.loads((FIXTURES / "nemotron3_nano_30b_a3b_bf77c317_config.json").read_text())
NANO_REVISION = "bf77c3174f68ad409e1c2aa60daeb46e32d1c606"
NANO_CONFIG_SHA256 = "dd9fa380ac107b0477db5a26108db9febe6378e7bb3966a107944853ec4f76f8"
VL = json.loads((FIXTURES / "nemotron35_super_vl_120b_fb0ee0bc_config.json").read_text())
VL_REVISION = "fb0ee0bc6b3f70342afd8d84cb80dbadbaa8a1fe"
VL_CONFIG_SHA256 = "be121eafe4c174c8bc7204da2cd063baa0392c0adfd41dd82fc9e0bba5b0f9a0"
# Language-model tensors of the 120B checkpoint without the MTP block and without
# the 40 x 512 router score-correction biases (a Megatron buffer, not a master).
VL_LANGUAGE_MODEL_NUMEL = 120_668_707_840 - 40 * 512


def test_nano_tp2_ep2_matches_the_measured_island_layout():
    manifests = L.rank_manifests(L.NemotronHGeometry.from_hf_config(NANO), tp=2, ep=2)

    # s27-gr-probe-b [yeto-layout-probe] "manifests"
    assert [m.shard_id for m in manifests] == [
        "tp0-of-2.pp0-of-1.ep0-of-2.cp0-of-1.dp0-of-1",
        "tp1-of-2.pp0-of-1.ep1-of-2.cp0-of-1.dp0-of-1",
    ]
    assert [len(m.specs) for m in manifests] == [3264, 3264]
    assert [m.numel for m in manifests] == [15_792_996_640, 15_792_996_640]
    assert [L.miles_source_layout_sha256(m) for m in manifests] == [
        "1e418bcf813aa70a781c895f0bd9baec5fdd35e224c624855a11e102c1281ae1",
        "92a94a3284366506adda8bf0d0c206c8dffcb0d52ddb496ba34b6c8625002e62",
    ]
    for minimum in (8, 60):  # the probe's placeholder count and the run's --fragments
        layout = L.offline_layout(manifests, model_revision=NANO_REVISION, config_sha256=NANO_CONFIG_SHA256,
                                  matrix_merge="rda", minimum_fragments=minimum)
        assert layout.fragments.num_fragments == 60
        assert layout.layout_hash == "0ec93ba8a8610c16c086407070cda74650f306d083e4f20f37e50975b9d24270"


def test_nano_tp1_ep8_dp8_lists_every_parameter_once():
    """The S27 closing trainer (TP1/EP8/DP8, Miles data-parallel replicas)."""
    geometry = L.NemotronHGeometry.from_hf_config(NANO)
    manifests = L.rank_manifests(geometry, tp=1, ep=8, dp=8)
    (whole,) = L.rank_manifests(geometry, tp=1, ep=1)  # one rank holding every parameter

    assert [m.shard_id for m in manifests] == [f"tp0-of-1.pp0-of-1.ep{r}-of-8.cp0-of-1.dp{r}-of-8" for r in range(8)]
    owned = [spec for m in manifests for spec in m.specs]
    assert sorted(owned) == list(whole.specs)  # each parameter on exactly one rank
    for rank, manifest in enumerate(manifests):  # every rank owns its own 16 experts of each MoE layer
        experts = {int(name.rsplit("weight", 1)[1]) for name, _ in manifest.specs if ".mlp.experts." in name}
        assert experts == set(range(16 * rank, 16 * (rank + 1)))
    dense = [sum(prod(shape) for name, shape in m.specs if ".mlp.experts." not in name) for m in manifests]
    largest = max(prod(shape) for name, shape in whole.specs if ".mlp.experts." not in name)
    assert max(dense) - min(dense) <= largest
    check_expert_partition([spec for m in manifests for spec in m.parameter_specs()])
    # Offline only: not yet compared with an island's layout check on GPU.
    layout = L.offline_layout(manifests, model_revision=NANO_REVISION, config_sha256=NANO_CONFIG_SHA256,
                              matrix_merge="rda", minimum_fragments=60)
    assert layout.fragments.num_fragments == 64
    assert layout.layout_hash == "c2c3cb49ad5a3f22ae4d8613ad5eefc366b9a425f39ed61ce3d68b32c45d62b0"


def test_dense_owners_follow_the_miles_replica_rule():
    """Same owners as Miles' own ``_replica_owners`` (needs the Miles checkout)."""
    state = pytest.importorskip("miles.backends.megatron_utils.full_parameter_state")
    geometry = L.NemotronHGeometry.from_hf_config(NANO)
    manifests = L.rank_manifests(geometry, tp=1, ep=8, dp=8)
    (held,) = L._held_specs(geometry, pp=1, ep=8, ep_rank=0, vocab_divisor=128)

    class Parameter:
        def __init__(self, shape, expert):
            self.shape, self.allreduce = shape, not expert

        def numel(self):
            return prod(self.shape)

    named = [(name, Parameter(shape, ".mlp.experts." in name)) for name, shape in held.specs]
    replication = state.FullParameterReplication(global_rank=0, dense_ranks=tuple(range(8)), expert_ranks=(0,))
    miles = state._replica_owners(named, replication)
    ours = {name: rank for rank, m in enumerate(manifests) for name, _ in m.specs}
    assert {name: owner for name, (owner, _) in miles.items()} == {name: ours[name] for name, _ in held.specs}


def test_vl_geometry_is_the_language_model():
    g = L.NemotronHGeometry.from_hf_config(VL)
    assert g.layer_counts() == {"M": 40, "E": 40, "*": 8, "-": 0}
    assert (g.hidden_size, g.num_experts, g.moe_latent_size, g.moe_ffn_hidden_size) == (4096, 512, 1024, 2688)
    assert (g.shared_expert_ffn_hidden_size, g.mamba_inner, g.mtp_layers) == (5376, 8192, 1)
    (manifest,) = L.rank_manifests(g, tp=1, ep=1)
    assert manifest.numel == VL_LANGUAGE_MODEL_NUMEL
    names = [name for name, _ in manifest.specs]
    check_frozen_excluded(names)
    assert not any(".mtp." in name or "vision" in name for name in names)


def test_vl_tp8_pp4_manifests_pass_the_island_checks():
    manifests = L.rank_manifests(L.NemotronHGeometry.from_hf_config(VL), tp=8, ep=8, pp=4)
    specs: list[ParameterSpec] = [s for m in manifests for s in m.parameter_specs()]

    partition = check_expert_partition(specs)  # latent MoE: router [512, 4096], experts [2688, 1024]
    assert partition.num_experts == 512 and len(partition.moe_layers) == 40
    assert {count for _, count in partition.experts_per_owner} == {10 * 64}  # 10 MoE layers x 64 experts per rank
    assert check_expert_coverage({s.name: s.shape for s in specs}, num_experts=512).num_experts == 512
    assert max(m.numel for m in manifests) == 3_896_284_128
    # TP replicas (norms, router, latent projections) are masters on every TP rank.
    assert sum(m.numel for m in manifests) == 123_607_251_968


@pytest.mark.parametrize(("tp", "pp", "fragments", "layout_hash"), [
    (8, 4, 272, "05c1fc0cf066d54d3a41cda662b2fe8dea555242118c408186090b9523f52af7"),
])
def test_vl_offline_layout(tp, pp, fragments, layout_hash):
    manifests = L.rank_manifests(L.NemotronHGeometry.from_hf_config(VL), tp=tp, ep=tp, pp=pp)
    layout = L.offline_layout(manifests, model_revision=VL_REVISION, config_sha256=VL_CONFIG_SHA256,
                              matrix_merge="rda")
    assert layout.fragments.num_fragments == fragments
    assert layout.layout_hash == layout_hash
    assert max(f.numel for f in layout.fragments.fragments) * 4 <= 2 << 30


@pytest.mark.parametrize(("kw", "match"), [
    ({"tp": 4, "ep": 2}, "EP .* must equal TP"),
    ({"tp": 16, "ep": 16}, "Mamba groups"),
    ({"tp": 8, "ep": 8, "pp": 3}, "layers"),
    ({"tp": 2, "ep": 8, "dp": 4}, "TP 1 only"),
    ({"tp": 1, "ep": 4, "dp": 8}, "EP .* must equal DP"),
])
def test_out_of_scope_parallelism_is_refused(kw, match):
    with pytest.raises(L.NemotronHLayoutError, match=match):
        L.rank_manifests(L.NemotronHGeometry.from_hf_config(VL), **kw)


def test_latent_router_reads_the_hidden_size():
    def spec(name, shape):
        return ParameterSpec("actor", name, shape, "float32", shape[0] * shape[1], "global")

    layer = "decoder.layers.1.mlp."
    experts = [spec(layer + f"experts.linear_fc{p}.weight{e}", (8, 4) if p == 1 else (4, 8))
               for e in range(2) for p in (1, 2)]
    latent = [spec(layer + "fc1_latent_proj.weight", (4, 16))]
    check_expert_partition([spec(layer + "router.weight", (2, 16)), *latent, *experts])
    check_expert_partition([spec(layer + "router.weight", (2, 4)), *experts])  # no latent: experts read hidden
    with pytest.raises(ValueError, match="router shape"):
        check_expert_partition([spec(layer + "router.weight", (2, 4)), *latent, *experts])
    with pytest.raises(ValueError, match="latent projection"):
        check_expert_partition([spec(layer + "fc1_latent_proj.weight", (5, 16)), *experts])


def test_offline_layout_tool_prints_the_launch_values(capsys):
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "tools" / "nemotron_h_offline_layout.py"
    spec = importlib.util.spec_from_file_location("nemotron_h_offline_layout_tool", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    assert tool.main([str(FIXTURES / "nemotron3_nano_30b_a3b_bf77c317_config.json"),
                      "--model-revision", NANO_REVISION, "--tp", "2", "--matrix-merge", "rda"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["config_sha256"] == NANO_CONFIG_SHA256
    assert printed["fragments"] == 60
    assert printed["parameter_layout_sha256"].startswith("0ec93ba8")
    assert printed["fragment_bytes"]["max"] <= 2 << 30
