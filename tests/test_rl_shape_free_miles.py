"""Shape-free layout through the real Miles full-parameter transport, on CPU.

Fake Megatron ranks of two islands with different shapes run behind fake
trainer groups whose ``run_plugin`` calls Miles' real per-rank entry point:

* island A: 4 ranks, TP1 / EP4 / DP4;
* island B: 4 ranks, TP2 / EP2 (dense data-parallel 2, expert data-parallel 2).

Each island has a TP-split Mamba mixer (packed ``in_proj`` and ``conv1d``
blocks, row-parallel ``out_proj``, native-FP32 ``A_log``), a replicated norm
and router, four TE grouped-GEMM experts and a vocabulary table. Both
optimizer layouts run: whole FP32 masters, and a stand-in for Megatron's
distributed optimizer (each data-parallel group keeps one flat FP32 range per
rank). The yeto adapter plans the shape-free layout from the ranks' partition
reports, installs owners and free-order plans; the tests then check one layout
hash for both islands, bit-identical canonical exports, and that
export -> average -> apply leaves every rank holding exactly its slice of the
merged global model, in its FP32 masters and in its model parameters.

Needs a Miles with the ``shape_free_layout`` transport feature.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transport = pytest.importorskip("miles.ray.full_parameter_transport")
if "shape_free_layout" not in getattr(transport, "FULL_PARAMETER_TRANSPORT_FEATURES", ()):
    pytest.skip("this Miles has no shape-free layout", allow_module_level=True)

from miles.backends.megatron_utils import full_parameter_state as fps  # noqa: E402
from miles.backends.megatron_utils.full_parameter_plugin import PLUGIN_PATH, full_parameter_rank_call  # noqa: E402
from miles.backends.megatron_utils.full_parameter_state import (  # noqa: E402
    FullParameterReplication,
    FullParameterTopology,
)

from yeto.rl.adapters.miles.models.chunked_full_parameter import MilesChunkedFullParameterAdapter  # noqa: E402
from yeto.rl.engine.shape_free_layout import (  # noqa: E402
    LAYOUT_SHAPE_FREE,
    GlobalParameter,
    Granularity,
    local_from_global,
)
from yeto.rl.local_learner import ComponentIdentity  # noqa: E402

COMPONENT = ComponentIdentity("actor", "a" * 40, "c" * 64)
GRANULARITY = Granularity(tp_units=2, expert_classes=4, dense_groups=2)
EXPERTS = 4
H = 4
L = "module.module.decoder.layers"
# name -> (global shape, partition dim or None, global block sizes or None)
DENSE = {
    "module.module.embedding.word_embeddings.weight": ((8, H), 0, None),
    f"{L}.0.mixer.in_proj.weight": ((14, H), 0, (4, 4, 2, 2, 2)),
    f"{L}.0.mixer.in_proj.layer_norm_weight": ((H,), None, None),
    f"{L}.0.mixer.conv1d_weight": ((8, 1, 2), 0, (4, 2, 2)),
    f"{L}.0.mixer.A_log": ((2,), 0, None),
    f"{L}.0.mixer.out_proj.weight": ((H, 4), 1, None),
    f"{L}.1.mlp.router.weight": ((EXPERTS, H), None, None),
}
NATIVE_FP32 = {f"{L}.0.mixer.A_log"}
EXPERT_SHAPES = {"linear_fc1": (6, H), "linear_fc2": (H, 3)}


def expert_name(projection: str, expert: int) -> str:
    return f"{L}.1.mlp.experts.{projection}.weight{expert}"


def global_parameters() -> dict[str, GlobalParameter]:
    params = {}
    for name, (shape, dim, blocks) in DENSE.items():
        params[name] = GlobalParameter(name, shape, False, dim, () if dim is None else (blocks or (shape[dim],)))
    for expert in range(EXPERTS):
        for projection, shape in EXPERT_SHAPES.items():
            params[expert_name(projection, expert)] = GlobalParameter(expert_name(projection, expert), shape, True)
    return params


def global_weights(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    # Values a bf16 model can hold exactly, so model and master compare bitwise.
    return {
        name: torch.tensor(rng.standard_normal(parameter.shape), dtype=torch.float32)
        .to(torch.bfloat16)
        .to(torch.float32)
        .numpy()
        for name, parameter in global_parameters().items()
    }


def _base(owned):
    return SimpleNamespace(
        param_groups=[{"params": owned}],
        state={tensor: {"exp_avg": torch.ones_like(tensor), "exp_avg_sq": torch.ones_like(tensor)} for tensor in owned},
    )


class FakeRank:
    """One fake Megatron rank of an island with ``world`` ranks, TP ``tp`` and EP ``ep``."""

    def __init__(self, rank: int, *, world: int, tp: int, ep: int, weights, sharded_masters: bool) -> None:
        self.rank, self.role = rank, "actor"
        tp_rank, dp_rank, ep_rank = rank % tp, rank // tp, rank % ep
        dp = world // tp
        self.tp_rank, self.tp = tp_rank, tp
        self.topology = FullParameterTopology(tp_rank, tp, 0, 1, ep_rank, ep, 0, 1, dp_rank, dp)
        self.tp_ranks = tuple(dp_rank * tp + index for index in range(tp))
        self.replication = FullParameterReplication(
            global_rank=rank,
            dense_ranks=tuple(index * tp + tp_rank for index in range(dp)),
            expert_ranks=tuple(other for other in range(world) if other % ep == ep_rank),
        )
        params = global_parameters()
        per_rank = EXPERTS // ep
        local_experts = {
            expert_name(projection, expert)
            for expert in range(ep_rank * per_rank, (ep_rank + 1) * per_rank)
            for projection in EXPERT_SHAPES
        }
        self.locals = {
            name: local_from_global(parameter, weights[name], tp_rank, tp)
            for name, parameter in params.items()
            if not parameter.expert or name in local_experts
        }
        self.groups = {
            "dense": (self.replication.dense_ranks, [name for name in self.locals if not params[name].expert]),
            "expert": (self.replication.expert_ranks, [name for name in self.locals if params[name].expert]),
        }
        self.args = SimpleNamespace(
            lora_rank=0, lora_adapter_path=None, num_steps_per_rollout=1, global_batch_size=1, start_rollout_id=None
        )
        self.opt_param_scheduler = SimpleNamespace(num_steps=0)
        self.weights_backuper = SimpleNamespace(backup=lambda role: None)
        self._last_rollout_id = 0
        self.sharded_masters = sharded_masters
        self.ranges: dict[str, tuple[int, int]] = {}
        self.named, self.optimizer = self._build(params)

    def _attributes(self, parameter, name, params) -> None:
        global_parameter = params[name]
        if global_parameter.expert:
            parameter.allreduce = False
        elif global_parameter.split:
            parameter.tensor_model_parallel = True
            parameter.partition_dim = global_parameter.partition_dim
            parameter.partition_stride = 1
            if len(global_parameter.block_sizes) > 1:
                parameter.partition_sizes = [size // self.tp for size in global_parameter.block_sizes]

    def _build(self, params):
        named, wrappers, whole_owned = [], [], []
        for _group, (ranks, names) in self.groups.items():
            ordered = sorted(names)
            wrapper = SimpleNamespace(
                model_float16_groups=[[]], shard_float16_groups=[[]], model_fp32_groups=[[]], shard_fp32_groups=[[]]
            )
            owned = []
            for dtype in (torch.bfloat16, torch.float32):
                members = [name for name in ordered if (name in NATIVE_FP32) == (dtype == torch.float32)]
                total = sum(self.locals[name].size for name in members)
                shard = -(-total // len(ranks)) if self.sharded_masters else total
                low = ranks.index(self.rank) * shard if self.sharded_masters else 0
                buffer = torch.zeros(max(shard * (len(ranks) if self.sharded_masters else 1), 1), dtype=dtype)
                offset = 0
                for name in members:
                    value = torch.from_numpy(self.locals[name].copy())
                    numel = value.numel()
                    parameter = torch.nn.Parameter(buffer[offset : offset + numel].view(value.shape))
                    parameter.data.copy_(value.to(dtype))
                    self._attributes(parameter, name, params)
                    if not self.sharded_masters:
                        if dtype == torch.float32:
                            whole_owned.append(parameter)
                        else:
                            parameter.main_param = value.clone()
                            whole_owned.append(parameter.main_param)
                        self.ranges[name] = (0, numel)
                    else:
                        start, stop = max(low, offset) - offset, min(low + shard, offset + numel) - offset
                        self.ranges[name] = (start, stop) if start < stop else (0, 0)
                        if start < stop:
                            view = parameter.detach().view(-1)[start:stop]
                            if dtype == torch.float32:
                                wrapper.model_fp32_groups[0].append(parameter)
                                wrapper.shard_fp32_groups[0].append(view)
                                owned.append(view)
                            else:
                                parameter.main_param = value.view(-1)[start:stop].clone()
                                wrapper.model_float16_groups[0].append(parameter)
                                wrapper.shard_float16_groups[0].append(view)
                                owned.append(parameter.main_param)
                        elif dtype == torch.bfloat16:
                            parameter.main_param = None
                        if dtype == torch.bfloat16:
                            parameter.main_param_sharded = True
                    named.append((name, parameter))
                    offset += numel
            wrapper.optimizer = _base(owned)
            wrappers.append(wrapper)
        if not self.sharded_masters:
            return tuple(sorted(named, key=lambda item: item[0])), SimpleNamespace(optimizer=_base(whole_owned))
        return tuple(sorted(named, key=lambda item: item[0])), SimpleNamespace(chained_optimizers=wrappers)

    def master_values(self) -> dict[str, tuple[int, torch.Tensor]]:
        """``{name: (range start, FP32 values)}`` this rank's optimizer steps."""

        values = {}
        for name, parameter in self.named:
            start, stop = self.ranges[name]
            if start == stop:
                continue
            if name in NATIVE_FP32:
                values[name] = (start, parameter.detach().view(-1)[start:stop])
            elif getattr(parameter, "main_param_sharded", False):
                values[name] = (start, parameter.main_param)
            else:
                values[name] = (start, parameter.main_param.view(-1))
        return values


class FakeRay:
    def __init__(self) -> None:
        self.objects = {}

    def put(self, value):
        reference = object()
        self.objects[reference] = value.detach().clone()
        return reference

    def get(self, reference):
        return self.objects[reference]


class FakeGroup:
    """A trainer controller with the Miles group's full-parameter methods."""

    def __init__(self, ranks, current) -> None:
        self.ranks, self._current = list(ranks), current

    async def run_plugin(self, fn_path, kwargs):
        assert fn_path == PLUGIN_PATH
        results = []
        for rank in self.ranks:
            self._current["rank"] = rank
            results.append(full_parameter_rank_call(rank, **kwargs))
        return results

    async def full_parameter_shard_manifests(self):
        return await transport.shard_manifests(self)

    async def install_full_parameter_fragment_plans(self, plans):
        return await transport.install_fragment_plans(self, plans)

    async def full_parameter_partition_reports(self):
        return await transport.partition_reports(self)

    async def install_full_parameter_owner_maps(self, owner_maps):
        return await transport.install_owner_maps(self, owner_maps)

    async def export_full_parameter_chunked_cut(self, policy_version, local_step_generation=0, *, max_chunk_bytes,
                                                defer_payload=False, lazy=False):
        return await transport.export_chunked_cut(
            self, policy_version, local_step_generation, max_chunk_bytes=max_chunk_bytes,
            defer_payload=defer_payload, lazy=lazy,
        )

    async def apply_full_parameter_chunked_cut(self, cut, *, commit_token=None, max_commit_attempts=3,
                                               skip_resident_fragments=False):
        return await transport.apply_chunked_cut(
            self, cut, commit_token=commit_token, max_commit_attempts=max_commit_attempts,
            skip_resident_fragments=skip_resident_fragments,
        )


@pytest.fixture
def islands(monkeypatch):
    current = {}
    ray = FakeRay()
    monkeypatch.setattr(fps, "full_parameter_topology", lambda: current["rank"].topology)
    monkeypatch.setattr(fps, "full_parameter_replication", lambda: current["rank"].replication)
    monkeypatch.setattr(fps, "_tensor_parallel_ranks", lambda: (current["rank"].tp_ranks, 1))
    monkeypatch.setattr(fps, "_named_actor_parameters", lambda actor: actor.named)

    def copy_masters_to_model(actor):
        for _name, parameter in actor.named:
            if getattr(parameter, "main_param", None) is not None and not getattr(parameter, "main_param_sharded", False):
                parameter.data.copy_(parameter.main_param.to(parameter.dtype))

    monkeypatch.setattr(fps, "_copy_masters_to_model", copy_masters_to_model)
    monkeypatch.setattr(fps, "_resolve_ray_module", lambda _module: ray)
    monkeypatch.setattr(transport, "_resolve_ray_module", lambda _module: ray)

    def make(world: int, tp: int, ep: int, weights, *, sharded_masters: bool):
        ranks = [FakeRank(rank, world=world, tp=tp, ep=ep, weights=weights, sharded_masters=sharded_masters)
                 for rank in range(world)]
        return FakeGroup(ranks, current), ranks

    return make, ray


async def initialize(group, ray):
    return await MilesChunkedFullParameterAdapter.initialize(
        group,
        policy_version=0,
        algorithm="grpo",
        components=(COMPONENT,),
        minimum_fragments=1,
        max_fragment_bytes=96,
        max_chunk_bytes=40,
        layout_mode=LAYOUT_SHAPE_FREE,
        granularity=GRANULARITY,
    )


def payloads(adapter, cut, ray) -> list[np.ndarray]:
    result = []
    for fragment_id in range(adapter.layout.fragments.num_fragments):
        parts = list(adapter.fragment_parts(cut, fragment_id, ray_module=ray)())
        result.append(np.concatenate([np.frombuffer(bytes(part), dtype=np.float32) for part in parts]))
    return result


def target_cut(cut, ray, values):
    """``cut`` at version 1 with every fragment replaced by ``values``."""

    shards = []
    for shard in cut.shards:
        fragments, refs = list(shard.fragments), [list(row) for row in shard.chunk_refs]
        for index, descriptor in enumerate(shard.fragments):
            payload = values[descriptor.fragment_id].astype(np.float32).tobytes()
            fragments[index], refs[index] = transport.store_full_parameter_fragment_payload(
                cut, descriptor.fragment_id, payload, ray_module=ray
            )
        shards.append(replace(shard, policy_version=1, local_step_generation=0, fragments=tuple(fragments),
                              chunk_refs=refs))
    return transport.FullParameterChunkedCut(
        policy_version=1, local_step_generation=0, parameter_layout_hash=cut.parameter_layout_hash, shards=tuple(shards)
    )


SHAPES = {"A": (4, 1, 4), "B": (4, 2, 2)}


@pytest.mark.parametrize("sharded_masters", [False, True], ids=["whole", "sharded"])
def test_two_shapes_share_the_layout_and_merge_bit_exactly(islands, sharded_masters):
    make, ray = islands
    trained = {"A": global_weights(1), "B": global_weights(2)}
    built = {key: make(*SHAPES[key], trained[key], sharded_masters=sharded_masters) for key in SHAPES}
    result = {}
    for key, (group, _ranks) in built.items():
        adapter, cut = asyncio.run(initialize(group, ray))
        result[key] = (adapter, cut)
    adapters = {key: value[0] for key, value in result.items()}
    assert adapters["A"].layout.layout_hash == adapters["B"].layout.layout_hash
    exported = {key: payloads(adapters[key], result[key][1], ray) for key in SHAPES}
    merged = [(a + b) / np.float32(2) for a, b in zip(exported["A"], exported["B"], strict=True)]
    merged_global = {name: (trained["A"][name] + trained["B"][name]) / np.float32(2) for name in trained["A"]}
    params = global_parameters()
    for key, (group, ranks) in built.items():
        adapter, cut = result[key]
        target = target_cut(cut.transport_cut, ray, merged)
        applied = asyncio.run(group.apply_full_parameter_chunked_cut(target))
        assert applied == adapter.expected_parameter_tensor_count
        for rank in ranks:
            for name, (start, values) in rank.master_values().items():
                expected = local_from_global(params[name], merged_global[name], rank.tp_rank, rank.tp)
                want = expected.reshape(-1)[start : start + values.numel()]
                assert values.numpy().tobytes() == want.tobytes(), (key, rank.rank, name)
            for name, parameter in rank.named:
                expected = local_from_global(params[name], merged_global[name], rank.tp_rank, rank.tp)
                # The model holds the merged FP32 value cast to its dtype.
                expected = torch.from_numpy(expected.astype(np.float32)).to(parameter.dtype)
                assert torch.equal(parameter.detach(), expected), (key, rank.rank, name)


def test_identical_models_export_identical_bytes(islands):
    make, ray = islands
    weights = global_weights(5)
    exported = {}
    for key in SHAPES:
        group, _ranks = make(*SHAPES[key], weights, sharded_masters=True)
        adapter, cut = asyncio.run(initialize(group, ray))
        exported[key] = payloads(adapter, cut, ray)
    assert len(exported["A"]) == len(exported["B"])
    for a, b in zip(exported["A"], exported["B"], strict=True):
        assert a.tobytes() == b.tobytes()


def test_replicated_parameters_have_cross_tp_writers(islands):
    make, ray = islands
    group, _ranks = make(*SHAPES["B"], global_weights(7), sharded_masters=True)
    asyncio.run(initialize(group, ray))
    manifests = asyncio.run(group.full_parameter_shard_manifests())
    norm = f"{L}.0.mixer.in_proj.layer_norm_weight"
    holdings = {manifest.global_rank: next(item for item in manifest.holdings if item.name == norm) for manifest in manifests}
    # TP2 x DP2: all four ranks hold the norm; one owns it, three write its bytes.
    assert {holding.replica_ranks for holding in holdings.values()} == {(0, 1, 2, 3)}
    assert len({holding.owner_rank for holding in holdings.values()}) == 1
