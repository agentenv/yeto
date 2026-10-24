"""Full-parameter sync layout for a Qwen3-MoE island with expert parallelism.

The specs use Megatron global names exactly as Miles exports them
(``named_params_and_buffers(convert_to_global_name=True)``): each rank holds
its tensor-parallel attention shard, replicated norms and router, and its own
experts named by global expert index. Ranks are emulated; no Megatron runs.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from yeto.fragments import MERGE_AVG, MERGE_RDA
from yeto.protocol import DTYPE_F32, SyncerClient, layout_fingerprint
from yeto.rl.local_learner import (
    ComponentIdentity,
    ParameterLayout,
    ParameterSpec,
    owner_affine_minimum_fragments,
    parameter_merge_mode,
)
from yeto.rl.moe_layout import check_expert_partition, describe_topology

ROOT = Path(__file__).resolve().parent.parent
CARGO = shutil.which("cargo") or str(Path.home() / ".cargo/bin/cargo")
PREFIX = "module.module."


def shard_id(tp: int, tp_size: int, ep: int, ep_size: int, pp: int = 0, pp_size: int = 1) -> str:
    # Miles FullParameterTopology.shard_id
    return f"tp{tp}-of-{tp_size}.pp{pp}-of-{pp_size}.ep{ep}-of-{ep_size}.cp0-of-1.dp0-of-1"


def qwen3_moe_specs(
    *,
    layers: int,
    hidden: int,
    heads: int,
    kv_groups: int,
    head_dim: int,
    experts: int,
    moe_ffn: int,
    vocab: int,
    tp: int,
    ep: int,
    pp: int = 1,
) -> list[ParameterSpec]:
    """Every rank's FP32 master specs of a Qwen3-MoE model (ETP 1, CP 1, DP 1).

    Within a pipeline stage the EP ranks fold onto the TP ranks (EP = TP), so
    rank r is TP rank r and EP rank r.
    """

    if ep != tp or experts % ep or layers % pp or vocab % tp:
        raise ValueError("unsupported emulated topology")
    specs = []
    per_stage = layers // pp
    for stage in range(pp):
        for rank in range(tp):
            owner = shard_id(rank, tp, rank, ep, stage, pp)

            def add(name: str, *shape: int) -> None:
                numel = int(np.prod(shape))
                specs.append(ParameterSpec("actor", PREFIX + name, tuple(shape), "float32", numel, owner))

            if stage == 0:
                add("embedding.word_embeddings.weight", vocab // tp, hidden)
            if stage == pp - 1:
                add("decoder.final_layernorm.weight", hidden)
                add("output_layer.weight", vocab // tp, hidden)
            for layer in range(stage * per_stage, (stage + 1) * per_stage):
                base = f"decoder.layers.{layer}."
                add(base + "self_attention.linear_qkv.weight",
                    (heads + 2 * kv_groups) * head_dim // tp, hidden)
                add(base + "self_attention.linear_qkv.layer_norm_weight", hidden)
                add(base + "self_attention.q_layernorm.weight", head_dim)
                add(base + "self_attention.k_layernorm.weight", head_dim)
                add(base + "self_attention.linear_proj.weight", hidden, heads * head_dim // tp)
                add(base + "pre_mlp_layernorm.weight", hidden)
                add(base + "mlp.router.weight", experts, hidden)
                local = experts // ep
                for expert in range(rank * local, (rank + 1) * local):
                    add(base + f"mlp.experts.linear_fc1.weight{expert}", 2 * moe_ffn, hidden)
                    add(base + f"mlp.experts.linear_fc2.weight{expert}", hidden, moe_ffn)
    return specs


TINY = dict(layers=2, hidden=16, heads=4, kv_groups=2, head_dim=4, experts=4, moe_ffn=6, vocab=32)
# Qwen3-30B-A3B: Hugging Face config.json and Miles scripts/models/qwen3-30B-A3B.py.
QWEN3_30B_A3B = dict(
    layers=48, hidden=2048, heads=32, kv_groups=4, head_dim=128,
    experts=128, moe_ffn=768, vocab=151936,
)


def component() -> ComponentIdentity:
    return ComponentIdentity("actor", "a" * 40, "b" * 64)


def layout(specs, fragments: int, matrix_merge: str = "rda") -> ParameterLayout:
    return ParameterLayout.create(
        algorithm="grpo",
        components=(component(),),
        specs=specs,
        num_fragments=fragments,
        fragment_strategy="owner_affine",
        matrix_merge=matrix_merge,
    )


def test_tiny_island_partitions_every_expert_once():
    partition = check_expert_partition(qwen3_moe_specs(**TINY, tp=2, ep=2))
    assert partition.num_experts == 4
    assert partition.moe_layers == (0, 1)
    # Each EP rank holds 2 experts in each of 2 layers.
    assert [count for _, count in partition.experts_per_owner] == [4, 4]


def test_dense_specs_have_no_expert_partition():
    specs = [s for s in qwen3_moe_specs(**TINY, tp=2, ep=2) if ".experts." not in s.name]
    assert check_expert_partition(specs) is None


def test_replicated_experts_are_refused_with_the_parallel_rule():
    # TP 2 with EP 1 replicates every expert on both TP ranks.
    specs = qwen3_moe_specs(**TINY, tp=1, ep=1)
    replica = [
        ParameterSpec(s.role, s.name, s.shape, s.dtype, s.numel, shard_id(1, 2, 0, 1))
        for s in specs
    ]
    with pytest.raises(ValueError, match="expert-model-parallel size = tensor-parallel"):
        check_expert_partition(specs + replica)


def test_missing_expert_and_router_mismatch_are_refused():
    specs = qwen3_moe_specs(**TINY, tp=2, ep=2)
    missing = [s for s in specs if not s.name.endswith("linear_fc2.weight3")]
    with pytest.raises(ValueError, match="experts are not 0..3"):
        check_expert_partition(missing)
    wrong_router = [
        ParameterSpec(s.role, s.name, (5, 16), s.dtype, 80, s.shard_id)
        if s.name.endswith("router.weight") else s
        for s in specs
    ]
    with pytest.raises(ValueError, match="router shape"):
        check_expert_partition(wrong_router)


def test_ungated_experts_are_accepted_and_bad_shapes_refused():
    # Nemotron-H experts are squared ReLU: fc1 holds only up, (ffn, hidden).
    specs = qwen3_moe_specs(**TINY, tp=2, ep=2)
    ungated = [
        ParameterSpec(s.role, s.name, (6, 16), s.dtype, 96, s.shard_id)
        if ".linear_fc1." in s.name else s
        for s in specs
    ]
    assert check_expert_partition(ungated).num_experts == 4
    odd = [
        ParameterSpec(s.role, s.name, (9, 16), s.dtype, 144, s.shard_id)
        if ".linear_fc1." in s.name else s
        for s in specs
    ]
    with pytest.raises(ValueError, match="not a gated or ungated MLP"):
        check_expert_partition(odd)


def owner_names_in_plan_order(lay: ParameterLayout) -> dict[str, list[str]]:
    names: dict[str, list[str]] = {}
    for fid in range(lay.fragments.num_fragments):
        _, owner = lay.fragment_owner(fid)
        names.setdefault(owner, []).extend(s.wire_name for s in lay.fragment_specs(fid))
    return names


@pytest.mark.parametrize("fragments", [4, 6, 9])
def test_layout_keeps_owner_order_and_one_merge_mode_per_fragment(fragments):
    specs = qwen3_moe_specs(**TINY, tp=2, ep=2)
    lay = layout(specs, fragments)
    assert lay.fragments.num_fragments == fragments
    # Miles installs an owner plan only when its fragments, concatenated,
    # repeat the owner's canonical (sorted) spec order exactly.
    for owner, names in owner_names_in_plan_order(lay).items():
        canonical = sorted(s for s in specs if s.shard_id == owner)
        assert names == [s.wire_name for s in canonical]
    for fid, fragment in enumerate(lay.fragments.fragments):
        modes = {parameter_merge_mode(s.name, "rda") for s in lay.fragment_specs(fid)}
        assert modes == {fragment.merge_mode}
    vocab = [f for f in lay.fragments.fragments if f.merge_mode == MERGE_AVG]
    assert sorted(n.rsplit("::", 1)[1] for f in vocab for n, _ in f.tensors) == sorted(
        [PREFIX + "embedding.word_embeddings.weight", PREFIX + "output_layer.weight"] * 2
    )


def test_average_layouts_keep_their_established_identity():
    specs = qwen3_moe_specs(**TINY, tp=2, ep=2)
    avg = layout(specs, 4, "avg")
    assert {f.merge_mode for f in avg.fragments.fragments} == {MERGE_AVG}
    assert owner_affine_minimum_fragments(specs) == 2
    # RDA needs one more fragment per owner for the vocabulary tables.
    assert owner_affine_minimum_fragments(specs, "rda") == 4
    assert layout(specs, 4, "rda").layout_hash != avg.layout_hash
    with pytest.raises(ValueError, match="outside the layout"):
        layout(specs, 3, "rda")


def test_different_expert_parallel_sizes_never_share_a_session():
    two = qwen3_moe_specs(**TINY, tp=2, ep=2)
    four = qwen3_moe_specs(**TINY, tp=4, ep=4)
    a, b = layout(two, 8), layout(four, 8)
    # Same parameters, different owners: the HELLO fingerprints differ, so the
    # syncer refuses the second island, and the island-side refusal names
    # both topologies.
    assert a.layout_hash != b.layout_hash
    assert layout_fingerprint(a.fragments) != layout_fingerprint(b.fragments)
    assert describe_topology([s.shard_id for s in two]) == "cp=1,dp=1,ep=2,pp=1,tp=2"
    assert describe_topology([s.shard_id for s in four]) == "cp=1,dp=1,ep=4,pp=1,tp=4"


def test_qwen3_30b_a3b_layout_is_balanced_and_fits_the_fragment_bound():
    # TP 4 (4 KV groups), EP 4 folded onto TP, PP 2: one 8-GPU island.
    specs = qwen3_moe_specs(**QWEN3_30B_A3B, tp=4, ep=4, pp=2)
    unique = {s.name: s.numel for s in specs if "layernorm" in s.name or "layer_norm" in s.name
              or "router" in s.name}
    wire = sum(s.numel for s in specs)
    # Sharded tensors appear once per owner; norms and routers once per TP rank.
    assert wire - 3 * sum(unique.values()) == 30_532_122_624
    assert wire == 30_570_504_192
    check_expert_partition(specs)
    lay = layout(specs, 72)
    sizes = [f.numel for f in lay.fragments.fragments]
    assert max(sizes) * 4 <= 2 << 30
    matrix = [f.numel for f in lay.fragments.fragments if f.merge_mode == MERGE_RDA]
    # Bulk fragments are within 1% of each other; the vocabulary fragments
    # (one per owner, 38.9M scalars) are the only small ones.
    assert max(matrix) / min(matrix) < 1.01
    assert len(matrix) == 64


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def syncer_binary() -> Path:
    if not Path(CARGO).exists():
        pytest.skip("needs cargo to build the syncer")
    subprocess.run([CARGO, "build", "-q"], cwd=ROOT / "syncer", check=True)
    return ROOT / "syncer/target/debug/yeto-syncer"


def rda(deltas: list[np.ndarray], weights: list[float]) -> np.ndarray:
    norms = [float(np.linalg.norm(d.astype(np.float64))) for d in deltas]
    wsum = sum(weights)
    radial = sum(n * w for n, w in zip(norms, weights)) / wsum
    direction = sum(
        (w / wsum / n) * d.astype(np.float64)
        for d, w, n in zip(deltas, weights, norms)
        if n > 0
    )
    return direction * (radial / np.linalg.norm(direction))


def avg(deltas: list[np.ndarray], weights: list[float]) -> np.ndarray:
    return sum(w * d.astype(np.float64) for d, w in zip(deltas, weights)) / sum(weights)


@pytest.mark.timeout(60)
def test_two_island_round_trip_matches_numpy(syncer_binary, tmp_path):
    specs = qwen3_moe_specs(**TINY, tp=2, ep=2)
    lay = layout(specs, 6)
    num = lay.fragments.num_fragments
    rng = np.random.default_rng(7)
    init = {s.wire_name: rng.standard_normal(s.numel).astype(np.float32) for s in specs}
    deltas = [
        {s.wire_name: (0.01 * rng.standard_normal(s.numel)).astype(np.float32) for s in specs}
        for _ in range(2)
    ]
    # Island 1 routed no token to expert 3: its update of that expert is zero.
    for name in deltas[1]:
        if name.endswith(("linear_fc1.weight3", "linear_fc2.weight3")):
            deltas[1][name][:] = 0.0
    tokens = (10, 30)
    weights = [t * t / 1 for t in tokens]  # tokens^2 / steps

    def fragment_bytes(values: dict[str, np.ndarray], fid: int) -> bytes:
        return np.concatenate([values[name] for name, _ in lay.fragments.fragments[fid].tensors]).tobytes()

    port = free_port()
    proc = subprocess.Popen(
        [str(syncer_binary), "--port", str(port), "--learners", "2", "--quorum", "2",
         "--grace-ms", "20", "--quorum-timeout-s", "10", "--total-steps", str(num),
         "--pipeline", "1", "--sync-interval-steps", "0", "--delta-correction", "none",
         "--outer-lr", "1", "--outer-momentum", "0",
         "--final-state", str(tmp_path / "final.bin")],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    clients = [
        SyncerClient(("127.0.0.1", port), island, lay.fragments, dtype=DTYPE_F32,
                     num_streams=1, connect_timeout=10, max_reconnects=0,
                     finalization_timeout=30)
        for island in range(2)
    ]
    try:
        for client in clients:
            client.start()
        for fid in range(num):
            clients[0].send_init(fid, fragment_bytes(init, fid))
        pushed = [set(), set()]
        deadline = time.monotonic() + 30
        while any(len(p) < num for p in pushed):
            assert time.monotonic() < deadline, "syncer did not permit every fragment"
            for island, client in enumerate(clients):
                client.check_health()
                for permit in client.drain_pulls():
                    client.push_fragment(
                        permit.fragment_id, permit.global_step, permit.round_attempt,
                        0, 1, 1, tokens[island],
                        fragment_bytes(deltas[island], permit.fragment_id),
                    )
                    pushed[island].add(permit.fragment_id)
            time.sleep(0.01)
        _, finals = clients[0].wait_for_final_fragments(timeout=30)
        for final in finals:
            fragment = lay.fragments.fragments[final.fragment_id]
            got = np.frombuffer(bytes(final.data), dtype=np.float32)
            offset = 0
            for name, numel in fragment.tensors:
                merge = avg if fragment.merge_mode == MERGE_AVG else rda
                expected = init[name] + merge([d[name] for d in deltas], weights)
                np.testing.assert_allclose(got[offset:offset + numel], expected, rtol=1e-5, atol=1e-6)
                if name.endswith("linear_fc1.weight3"):
                    # One-sided expert update: weighted by island 0's share only.
                    share = weights[0] / sum(weights)
                    np.testing.assert_allclose(
                        got[offset:offset + numel] - init[name], share * deltas[0][name],
                        rtol=1e-4, atol=5e-7,  # f32 cancellation against init
                    )
                offset += numel
    finally:
        for client in clients:
            client.close()
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)
