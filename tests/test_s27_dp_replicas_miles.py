"""S27 G-AX: yeto's full-parameter adapter on a Miles trainer with DP>1 (EP spanning DP).

Four fake Megatron ranks (TP1, DP4, EP2) run the real Miles full-parameter
state, plugin and group transport in one process; only Ray and Megatron's
process groups are stand-ins.  A dense parameter is replicated on all four
ranks and an expert (TE grouped-GEMM names, ``allreduce=False``) on the two
ranks of its EP rank.  yeto builds its owner-affine layout from the Miles
manifests, installs the plans, captures the initial cut, assembles a merged
target from authoritative fragments and applies it, as the role stream does.

Checks: the layout lists every parameter exactly once; every replica of a
parameter ends bit-identical (masters and model) and equal to the merged
fragment bytes.

S27 G-BE: the same ranks under a stand-in for Megatron's distributed
optimizer (each replica group keeps its parameters in one padded flat buffer
per group, split contiguously; each rank steps the FP32 ``main_param`` of its
range) give yeto the same layout hash and initial cut, and the apply writes
the same masters and model bits.

Needs Miles with data-parallel replicas (``importorskip``).
"""

from __future__ import annotations

# ruff: noqa: I001 -- Miles is imported only after importorskip.

import asyncio
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("miles")
replicas = pytest.importorskip("miles.backends.megatron_utils.full_parameter_replicas")

from miles.backends.megatron_utils import full_parameter_state as fps
from miles.backends.megatron_utils.full_parameter_plugin import PLUGIN_PATH, full_parameter_rank_call
from miles.ray import full_parameter_transport as transport

from yeto.rl.adapters.miles.models.chunked_full_parameter import MilesChunkedFullParameterAdapter
from yeto.rl.local_learner import ComponentIdentity

WORLD, EP = 4, 2
EXPERTS_PER_RANK = 2
DENSE = {
    "embedding.word_embeddings.weight": (8, 4),
    "decoder.layers.0.self_attention.linear_qkv.weight": (6, 4),
    "decoder.layers.0.self_attention.linear_proj.weight": (4, 2),
    "decoder.layers.0.mlp.router.weight": (EP * EXPERTS_PER_RANK, 4),
    "decoder.layers.0.pre_mlp_layernorm.weight": (4,),
    "decoder.final_layernorm.weight": (4,),
    "output_layer.weight": (8, 4),
}


def _initial(name: str, shape) -> torch.Tensor:
    seed = sum(ord(character) for character in name) % 101
    return (torch.arange(int(torch.Size(shape).numel()), dtype=torch.float32) / 8 + seed).reshape(shape)


class _Rank:
    def __init__(self, rank: int) -> None:
        ep_rank = rank % EP
        self.topology = fps.FullParameterTopology(
            tp_rank=0, tp_size=1, pp_rank=0, pp_size=1, ep_rank=ep_rank, ep_size=EP,
            cp_rank=0, cp_size=1, dp_rank=rank, dp_size=WORLD)
        self.replication = fps.FullParameterReplication(
            global_rank=rank, dense_ranks=tuple(range(WORLD)),
            expert_ranks=tuple(other for other in range(WORLD) if other % EP == ep_rank))
        shapes = dict(DENSE)
        for local in range(EXPERTS_PER_RANK):
            expert = ep_rank * EXPERTS_PER_RANK + local
            shapes[f"decoder.layers.0.mlp.experts.linear_fc1.weight{expert}"] = (6, 4)
            shapes[f"decoder.layers.0.mlp.experts.linear_fc2.weight{expert}"] = (4, 6)
        self.shapes = shapes
        named = []
        for name, shape in shapes.items():
            parameter = torch.nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16))
            parameter.main_param = _initial(name, shape)
            parameter.data.copy_(parameter.main_param.to(torch.bfloat16))
            if ".experts." in name:
                parameter.allreduce = False
            named.append((name, parameter))
        self.named = tuple(named)
        masters = [parameter.main_param for _name, parameter in self.named]
        self.optimizer = SimpleNamespace(optimizer=SimpleNamespace(
            param_groups=[{"params": masters}],
            state={master: {"exp_avg": torch.zeros_like(master)} for master in masters}))
        self.args = SimpleNamespace(lora_rank=0, lora_adapter_path=None, num_steps_per_rollout=1,
                                    global_batch_size=1, start_rollout_id=None)
        self.role = "actor"
        self.opt_param_scheduler = SimpleNamespace(num_steps=0)
        self.weights_backuper = SimpleNamespace(backup=lambda role: None)
        self._last_rollout_id = 0


class _ShardedRank(_Rank):
    """The same rank under a Megatron-like distributed optimizer (sharded FP32 masters)."""

    def __init__(self, rank: int) -> None:
        super().__init__(rank)
        groups = {
            "dense": (self.replication.dense_ranks, [n for n in sorted(self.shapes) if ".experts." not in n]),
            "expert": (self.replication.expert_ranks, [n for n in sorted(self.shapes) if ".experts." in n]),
        }
        named, wrappers, self.ranges = [], [], {}
        for ranks, names in groups.values():
            total = sum(torch.Size(self.shapes[name]).numel() for name in names)
            shard = -(-total // len(ranks))
            low = ranks.index(rank) * shard
            buffer, offset = torch.zeros(shard * len(ranks), dtype=torch.bfloat16), 0
            models, views, mains = [], [], []
            for name in names:
                shape, numel = self.shapes[name], torch.Size(self.shapes[name]).numel()
                parameter = torch.nn.Parameter(buffer[offset:offset + numel].view(shape))
                parameter.data.copy_(_initial(name, shape).to(torch.bfloat16))
                parameter.main_param_sharded = True
                parameter.main_param = None
                if ".experts." in name:
                    parameter.allreduce = False
                start, stop = max(low, offset) - offset, min(low + shard, offset + numel) - offset
                if start < stop:
                    parameter.main_param = _initial(name, shape).reshape(-1)[start:stop].clone()
                    models.append(parameter)
                    views.append(parameter.detach().view(-1)[start:stop])
                    mains.append(parameter.main_param)
                    self.ranges[name] = (start, stop)
                named.append((name, parameter))
                offset += numel
            wrappers.append(SimpleNamespace(
                optimizer=SimpleNamespace(param_groups=[{"params": mains}],
                                          state={main: {"exp_avg": torch.zeros_like(main)} for main in mains}),
                model_float16_groups=[models], shard_float16_groups=[views]))
        self.named = tuple(sorted(named))
        self.optimizer = SimpleNamespace(chained_optimizers=wrappers)


class _FakeRay:
    def __init__(self) -> None:
        self.objects = {}

    def put(self, value):
        reference = object()
        self.objects[reference] = value.detach().clone()
        return reference

    def get(self, reference):
        return self.objects[reference]


class _TrainerController:
    """Miles' TrainerController full-parameter methods over in-process ranks."""

    def __init__(self, ranks, current) -> None:
        self.ranks, self._current = ranks, current
        self.full_parameter_replica_owners = {}

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

    async def export_full_parameter_chunked_cut(self, policy_version, local_step_generation=0, *,
                                                max_chunk_bytes, defer_payload=False):
        return await transport.export_chunked_cut(self, policy_version, local_step_generation,
                                                  max_chunk_bytes=max_chunk_bytes, defer_payload=defer_payload)

    async def apply_full_parameter_chunked_cut(self, cut, *, commit_token=None, max_commit_attempts=3,
                                               skip_resident_fragments=False):
        return await transport.apply_chunked_cut(self, cut, commit_token=commit_token,
                                                 max_commit_attempts=max_commit_attempts,
                                                 skip_resident_fragments=skip_resident_fragments)


@pytest.fixture
def trainers(monkeypatch):
    """``make(rank_type)`` -> (controller, ranks, fake ray); one Ray stand-in for every trainer."""
    current, ray = {}, _FakeRay()
    monkeypatch.setattr(fps, "full_parameter_topology", lambda: current["rank"].topology)
    monkeypatch.setattr(fps, "full_parameter_replication", lambda: current["rank"].replication)
    monkeypatch.setattr(fps, "_named_actor_parameters", lambda actor: actor.named)
    monkeypatch.setattr(fps, "_copy_masters_to_model", lambda actor: [
        parameter.data.copy_(parameter.main_param.to(torch.bfloat16)) for _name, parameter in actor.named])
    monkeypatch.setattr(fps, "_resolve_ray_module", lambda _module: ray)
    monkeypatch.setattr(transport, "_resolve_ray_module", lambda _module: ray)

    def make(rank_type):
        ranks = [rank_type(rank) for rank in range(WORLD)]
        return _TrainerController(ranks, current), ranks, ray

    return make


@pytest.fixture
def trainer(trainers):
    return trainers(_Rank)


def _merged(fragment_id: int, numel: int) -> torch.Tensor:
    return torch.arange(numel, dtype=torch.float32) * 0.25 - 10.0 * (fragment_id + 1)


def test_dp_replicas_cover_every_parameter_once_and_apply_identically(trainer):
    group, ranks, ray = trainer
    adapter, initial = asyncio.run(MilesChunkedFullParameterAdapter.initialize(
        group, policy_version=0, algorithm="grpo", components=(ComponentIdentity("actor", "a" * 40, "b" * 64),),
        minimum_fragments=WORLD, max_fragment_bytes=64, max_chunk_bytes=32))

    # Every parameter exactly once in the layout, under its owner's topology.
    layout_names = [spec.name for spec in adapter.layout.specs]
    held = {name for rank in ranks for name, _parameter in rank.named}
    assert sorted(layout_names) == sorted(held) and len(set(layout_names)) == len(layout_names)
    assert adapter.expected_parameter_tensor_count == len(held)
    assert adapter.layout.has_split_parameters  # 128-byte tensors span two 64-byte fragments

    sink = adapter.fragment_sink(initial, ray_module=ray)
    fragments = adapter.layout.fragments.fragments
    stored = [sink(fragment_id, adapter._fragment_version(1, fragment_id),
                   _merged(fragment_id, fragment.numel).numpy().tobytes())
              for fragment_id, fragment in enumerate(fragments)]
    target = adapter.assemble_target(initial, target_policy_version=1, fragments=stored)
    assert asyncio.run(adapter.apply(group, target, commit_token="dp-merge-1")) == len(held)

    # The merged bytes, parameter by parameter, through yeto's own layout.
    expected = {}
    for fragment_id, fragment in enumerate(fragments):
        payload, offset = _merged(fragment_id, fragment.numel), 0
        for unit in adapter.layout.fragment_slices(fragment_id):
            start, stop = unit.flat_range
            expected.setdefault(unit.spec.name, torch.full((unit.spec.numel,), float("nan")))[start:stop] = (
                payload[offset : offset + stop - start])
            offset += stop - start
    for rank in ranks:
        for name, parameter in rank.named:
            assert torch.equal(parameter.main_param.reshape(-1), expected[name]), (rank.topology.dp_rank, name)
            assert torch.equal(parameter.detach(), parameter.main_param.to(torch.bfloat16))
    # A dense parameter lives on all four ranks, an expert on two.
    copies = {name: sum(name in dict(rank.named) for rank in ranks) for name in held}
    assert {copies[name] for name in DENSE} == {WORLD}
    assert {count for name, count in copies.items() if ".experts." in name} == {WORLD // EP}


def _initialize(group):
    return asyncio.run(MilesChunkedFullParameterAdapter.initialize(
        group, policy_version=0, algorithm="grpo", components=(ComponentIdentity("actor", "a" * 40, "b" * 64),),
        minimum_fragments=WORLD, max_fragment_bytes=64, max_chunk_bytes=32))


def _apply_merged(adapter, initial, group, ray) -> None:
    sink = adapter.fragment_sink(initial, ray_module=ray)
    stored = [sink(fragment_id, adapter._fragment_version(1, fragment_id),
                   _merged(fragment_id, fragment.numel).numpy().tobytes())
              for fragment_id, fragment in enumerate(adapter.layout.fragments.fragments)]
    target = adapter.assemble_target(initial, target_policy_version=1, fragments=stored)
    assert asyncio.run(adapter.apply(group, target, commit_token="dp-merge-1")) == adapter.expected_parameter_tensor_count


def test_sharded_masters_keep_the_yeto_layout_and_apply_the_same_bits(trainers):
    from miles.ray.full_parameter_transport import full_parameter_chunked_cut_identity

    whole_group, whole_ranks, ray = trainers(_Rank)
    sharded_group, sharded_ranks, _ray = trainers(_ShardedRank)
    whole_adapter, whole_initial = _initialize(whole_group)
    sharded_adapter, sharded_initial = _initialize(sharded_group)
    assert sharded_group.full_parameter_sharded_master_topologies  # the transport routes contributions
    assert sharded_adapter.layout.layout_hash == whole_adapter.layout.layout_hash
    assert sharded_adapter.layout.specs == whole_adapter.layout.specs
    assert (full_parameter_chunked_cut_identity(sharded_initial.transport_cut)
            == full_parameter_chunked_cut_identity(whole_initial.transport_cut))
    # Some dense parameter is split across ranks, and some rank holds none of one.
    assert any(0 < stop - start < dict(rank.named)[name].numel()
               for rank in sharded_ranks for name, (start, stop) in rank.ranges.items())
    assert any(name not in rank.ranges for rank in sharded_ranks for name, _parameter in rank.named)

    _apply_merged(whole_adapter, whole_initial, whole_group, ray)
    _apply_merged(sharded_adapter, sharded_initial, sharded_group, ray)
    for whole_rank, sharded_rank in zip(whole_ranks, sharded_ranks, strict=True):
        whole_named = dict(whole_rank.named)
        for name, parameter in sharded_rank.named:
            expected = whole_named[name]
            assert torch.equal(parameter.detach().view(torch.int16), expected.detach().view(torch.int16)), name
            if name in sharded_rank.ranges:
                start, stop = sharded_rank.ranges[name]
                master = expected.main_param.reshape(-1)[start:stop]
                assert torch.equal(parameter.main_param.view(torch.int32), master.view(torch.int32)), name
