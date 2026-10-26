"""S27 G-FG: the verl Megatron engine island in a full-parameter syncer session.

The Megatron island reuses the FSDP2 island's role stream, adapter and cuts;
only its trainer group differs (``adapters/verl/megatron_full_parameter.py``
over the neutral ``yeto/rl/megatron_hf_boundary.py``).  The fake trainer here
is Megatron-shaped: a fused QKV parameter that three HF tensors fill, bf16
model parameters whose FP32 masters live in a DistributedOptimizer stand-in
(with detail the bf16 copy cannot hold), FP32 norms that are their own
masters, and a Bridge stand-in that casts to the module parameter's dtype as
Megatron-Bridge does.  The two-island cases run the real Rust syncer (the
launcher's command, as ``test_s27_verl_parity``).

``test_real_megatron_and_bridge_round_trip`` runs real Megatron-Core and
Megatron-Bridge on gloo ranks (``tests/megatron_bridge_roundtrip.py``) when
``YETO_MEGATRON_PYTHON``, ``YETO_MEGATRON_LM_PATH`` and
``YETO_MEGATRON_BRIDGE_PATH`` name them; otherwise it is skipped.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron_bridge_roundtrip import _DistributedLeaf
from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from test_s27_verl_full_parameter_sync import (  # noqa: F401 - syncer_binary is a fixture
    COMPONENT, TOKENS, _Driver, _start_syncer, needs_cargo, syncer_binary,
)
from yeto.rl import megatron_hf_boundary as boundary
from yeto.rl import megatron_masters
from yeto.rl.adapters.verl import full_parameter as fp
from yeto.rl.adapters.verl import full_parameter_stream as vfs
from yeto.rl.adapters.verl import megatron_full_parameter as mfp
from yeto.rl.contracts import LocalStepReceipt

REPO = Path(__file__).resolve().parents[1]
HF_SHAPES = {
    "model.embed_tokens.weight": (12, 4),
    "model.layers.0.self_attn.q_proj.weight": (4, 4),
    "model.layers.0.self_attn.k_proj.weight": (2, 4),
    "model.layers.0.self_attn.v_proj.weight": (2, 4),
    "model.layers.0.input_layernorm.weight": (4,),
    "model.norm.weight": (4,),
}
QKV = {"q": "model.layers.0.self_attn.q_proj.weight", "k": "model.layers.0.self_attn.k_proj.weight",
       "v": "model.layers.0.self_attn.v_proj.weight"}
# Megatron name -> HF name (or the QKV roles); bf16 parameters have an FP32 master in the optimizer.
MEGATRON = {
    "embedding.word_embeddings.weight": ("model.embed_tokens.weight", (12, 4), torch.bfloat16),
    "decoder.layers.0.self_attention.linear_qkv.weight": (QKV, (8, 4), torch.bfloat16),
    "decoder.layers.0.input_layernorm.weight": ("model.layers.0.input_layernorm.weight", (4,), torch.float32),
    "decoder.final_layernorm.weight": ("model.norm.weight", (4,), torch.float32),
}
# 16 floats per fragment: the embedding is split, and q, k and v land in different fragments.
MAX_FRAGMENT_BYTES = 64


# ----------------------------------------------------------------------------- a Megatron-shaped fake
class _Mapping:
    """A Bridge mapping stand-in: one HF tensor, or Q/K/V rows of one fused parameter."""

    def __init__(self, megatron_name: str, hf_param):
        self.megatron_param, self.hf_param = megatron_name, hf_param

    def megatron_to_hf(self, weights, module):
        if isinstance(self.hf_param, str):
            return {self.hf_param: weights}
        q, k, v = torch.split(weights, [4, 2, 2], dim=0)
        return {self.hf_param["q"]: q, self.hf_param["k"]: k, self.hf_param["v"]: v}

    def _target(self, module):
        """Bridge's ``get_module_and_param_from_name``: the whole name, else a suffix of
        it (so the embedding mapping can also write the tied output layer)."""
        parts = self.megatron_param.split(".")
        for start in range(len(parts)):
            try:
                return module.get_parameter(".".join(parts[start:]))
            except AttributeError:
                continue
        raise AssertionError(f"{self.megatron_param} is not in {type(module).__name__}")

    def hf_to_megatron(self, hf_weights, module):
        target = self._target(module)  # Bridge casts to the module parameter's dtype
        if isinstance(self.hf_param, str):
            return hf_weights.to(target.dtype)
        return torch.cat([hf_weights["q"], hf_weights["k"], hf_weights["v"]], dim=0).to(target.dtype)


class _Model(torch.nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        for name, (_hf, shape, dtype) in MEGATRON.items():
            module = self
            *path, leaf = name.split(".")
            for part in path:
                if not hasattr(module, part):
                    module.add_module(part, torch.nn.Module())
                module = getattr(module, part)
            module.register_parameter(leaf, torch.nn.Parameter(torch.randn(shape, generator=generator).to(dtype)))


class _Bridge:
    """``AutoBridge.get_conversion_tasks`` / ``export_hf_weights`` over the fake model:
    a task reads the module's parameter when it is built (as Bridge's does)."""

    def get_conversion_tasks(self, modules):
        (model,) = modules
        params = dict(model.named_parameters())
        return [SimpleNamespace(global_param_name=name, mapping=_Mapping(name, hf), param_weight=params[name],
                                megatron_module=model)
                for name, (hf, _shape, _dtype) in MEGATRON.items()]

    def export_hf_weights(self, modules, cpu=False, show_progress=True, conversion_tasks=None,
                          merge_adapter_weights=True):
        for task in conversion_tasks:
            yield from task.mapping.megatron_to_hf(task.param_weight.detach(), task.megatron_module).items()


class _Engine:
    def __init__(self, seed: int = 0):
        self.model = _Model(seed)
        bf16 = [p for p in self.model.parameters() if p.dtype == torch.bfloat16]
        leaf = _DistributedLeaf(bf16, dp_rank=0, dp_size=1, group=None, seed=7 + seed)
        self.module, self.bridge = [self.model], _Bridge()
        self.optimizer = SimpleNamespace(chained_optimizers=[leaf])

    def is_param_offload_enabled(self):
        return False

    def is_optimizer_offload_enabled(self):
        return False


class _Island:
    """``VerlIsland.call_workers`` over one in-process Megatron worker."""

    def __init__(self, engine: _Engine):
        self.worker = SimpleNamespace(actor=SimpleNamespace(engine=engine))
        self.calls: list[str] = []

    def call_workers(self, fn, *args):
        self.calls.append(fn.__name__)
        return [fn(self.worker, *args)]

    call_workers_shared = call_workers


def _hf_values(engine: _Engine) -> dict[str, torch.Tensor]:
    """What the trainer holds, in HF names: the FP32 masters."""
    out = {}
    tasks = boundary.conversion_tasks(engine.bridge, engine.module)
    boundary.export_hf(engine.bridge, engine.module, engine.optimizer, list(range(len(tasks))),
                       lambda name, tensor: out.__setitem__(name, tensor.detach().clone()))
    return out


def _step(learner_id: int, round_id: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(1000 * learner_id + round_id)
    return {name: 0.01 * torch.randn(shape, generator=generator) for name, shape in HF_SHAPES.items()}


def _train(engine: _Engine, delta: dict[str, torch.Tensor]) -> None:
    """One optimizer step stand-in: add the HF-space delta to every FP32 master
    and refresh the bf16 model copy (no HF boundary code involved)."""
    params = dict(engine.model.named_parameters())
    plist = list(params.values())
    masters = megatron_masters.full_masters(engine.optimizer, plist)
    targets = []
    for (name, param), master in zip(params.items(), masters, strict=True):
        hf = MEGATRON[name][0]
        step = delta[hf] if isinstance(hf, str) else torch.cat([delta[QKV[r]] for r in ("q", "k", "v")])
        targets.append(master.detach().clone() + step)
    with torch.no_grad():
        megatron_masters.write_masters(engine.optimizer, plist, targets)
        for param, target in zip(plist, targets, strict=True):
            param.data.copy_(target.to(param.dtype))


class _MegatronHarness:
    """One Megatron island process's trainer (the catch-up case's harness interface)."""

    names = tuple(HF_SHAPES)

    def __init__(self):
        self.engine = _Engine(seed=0)  # every island loads the same checkpoint
        self.island = _Island(self.engine)
        self.island.full_group = mfp.MegatronFullParameterGroup(self.island, shapes=HF_SHAPES, config={})

    def values(self) -> dict:
        return _hf_values(self.engine)

    @staticmethod
    def checkpoint_values() -> dict:
        return _hf_values(_Engine(seed=0))

    def train(self, learner_id: int, round_id: int) -> None:
        _train(self.engine, _step(learner_id, round_id))


def _layout(minimum_fragments: int = 1, matrix_merge: str = "avg"):
    return fp.plan_layout(fp.island_specs(HF_SHAPES), component=COMPONENT, minimum_fragments=minimum_fragments,
                          max_fragment_bytes=MAX_FRAGMENT_BYTES, matrix_merge=matrix_merge)


# ----------------------------------------------------------------------------- the HF boundary
def test_the_export_reads_fp32_masters_and_an_apply_round_trips_bit_exact():
    engine = _Engine()
    values = _hf_values(engine)
    assert set(values) == set(HF_SHAPES) and all(v.dtype == torch.float32 for v in values.values())
    # the bf16 parameters' masters carry detail the model copy does not
    copy = engine.model.embedding.word_embeddings.weight.detach().float()
    assert not torch.equal(values["model.embed_tokens.weight"], copy)
    records = [boundary.manifest(engine.bridge, engine.module, engine.optimizer)]
    index = boundary.HfTaskIndex.from_manifests(records, HF_SHAPES)
    target = {name: value + 0.125 for name, value in values.items()}
    write, read = index.apply_plan(target)
    assert read == [] and len(write) == len(MEGATRON)
    boundary.apply_hf(engine.bridge, engine.module, engine.optimizer, target, write, read)
    after = _hf_values(engine)
    assert all(torch.equal(after[name], target[name]) for name in HF_SHAPES)
    for param, master in zip(engine.model.parameters(),
                             megatron_masters.full_masters(engine.optimizer, list(engine.model.parameters())),
                             strict=True):
        assert torch.equal(param.detach(), master.to(param.dtype))


def test_a_partial_apply_reads_the_other_fused_inputs_from_the_masters():
    engine = _Engine()
    values = _hf_values(engine)
    index = boundary.HfTaskIndex.from_manifests([boundary.manifest(engine.bridge, engine.module,
                                                                   engine.optimizer)], HF_SHAPES)
    given = {QKV["k"]: values[QKV["k"]] - 1.0}
    write, read = index.apply_plan(given)
    assert write == read == [1]  # the QKV task, whose q and v are not given
    boundary.apply_hf(engine.bridge, engine.module, engine.optimizer, given, write, read)
    after = _hf_values(engine)
    assert [n for n in HF_SHAPES if not torch.equal(after[n], values[n])] == [QKV["k"]]
    assert torch.equal(after[QKV["k"]], given[QKV["k"]])


def test_the_task_index_refuses_a_boundary_bridge_does_not_cover():
    engine = _Engine()
    records = [boundary.manifest(engine.bridge, engine.module, engine.optimizer)]
    with pytest.raises(boundary.MegatronHfBoundaryError, match="exports no tensor"):
        boundary.HfTaskIndex.from_manifests(records, {**HF_SHAPES, "lm_head.weight": (12, 4)})
    with pytest.raises(boundary.MegatronHfBoundaryError, match="does not have"):
        boundary.HfTaskIndex.from_manifests(records, {k: v for k, v in HF_SHAPES.items() if k != "model.norm.weight"})
    bad = {"tasks": records[0]["tasks"], "refused": ["decoder.layers.0.self_attention.linear_qkv.weight"]}
    with pytest.raises(boundary.MegatronHfBoundaryError, match="FP32 optimizer master"):
        boundary.HfTaskIndex.from_manifests([bad], HF_SHAPES)


def test_the_task_index_joins_two_pipeline_stages(monkeypatch):
    """S27 G-FL: each stage reports every task, with the master dtype only for the ones
    it owns (``param_weight is None`` elsewhere). The index is the union, so the
    boundary covers the whole model and ``master_dtypes`` names every tensor."""
    engines = _pipeline_engines(seed=0)
    records = [boundary.manifest(engine.bridge, engine.module, engine.optimizer) for engine in engines]
    assert all(len(record["tasks"]) == len(MEGATRON) and record["refused"] == [] for record in records)
    owned = [sorted(name for _i, names, dtype in record["tasks"] if dtype for name in names)
             for record in records]
    assert owned[0] and owned[1] and set(owned[0]).isdisjoint(owned[1])
    index = boundary.HfTaskIndex.from_manifests(records, HF_SHAPES)
    assert sorted(index.by_name) == sorted(HF_SHAPES)
    assert sorted(index.master_dtypes(records)) == sorted(HF_SHAPES)
    # one stage alone still names every tensor (the task list spans all stages), but it
    # reports no master for the other stage's
    assert sorted(boundary.HfTaskIndex.from_manifests(records[:1], HF_SHAPES).by_name) == sorted(HF_SHAPES)
    assert sorted(index.master_dtypes(records[:1])) == owned[0]


def test_the_tied_output_copy_is_found_only_on_the_last_stage():
    """S27 G-FL: ``tied_output_copies`` names the output-layer weight no conversion task
    carries; a single-stage island has none, and a copy without an FP32 master is
    refused in the manifest (at island start, not mid-run)."""
    single = _Engine()
    tasks = boundary.conversion_tasks(single.bridge, single.module)
    assert boundary.tied_output_copies(single.module, tasks) == []
    assert boundary.embedding_task_index(tasks) == 0

    first, last = _pipeline_engines(seed=0)
    assert boundary.tied_output_copies(first.module, boundary.conversion_tasks(
        first.bridge, first.module)) == []
    found = boundary.tied_output_copies(last.module, boundary.conversion_tasks(last.bridge, last.module))
    assert len(found) == 1 and found[0] is last.model.output_layer.weight

    bare = _PipelineEngine(0, 1)
    bare.optimizer = SimpleNamespace(chained_optimizers=[])  # no distributed optimizer, no main_param
    record = boundary.manifest(bare.bridge, bare.module, bare.optimizer)
    assert record["refused"][0] == "output_layer.weight (the tied embedding's pipeline copy)"
    with pytest.raises(boundary.MegatronHfBoundaryError, match="FP32 optimizer master"):
        boundary.HfTaskIndex.from_manifests([record], HF_SHAPES)


def test_expert_tasks_of_one_index_are_one_task_over_the_expert_ranks():
    """Two EP ranks: task 0 names expert 0 on rank 0 and expert 1 on rank 1; one
    fragment that needs expert 1 runs task 0 on both ranks (Bridge gathers over EP)."""
    records = [{"tasks": [[0, ["model.layers.0.mlp.experts.0.up_proj.weight"], "float32"],
                          [1, ["model.norm.weight"], "float32"]], "refused": []},
               {"tasks": [[0, ["model.layers.0.mlp.experts.1.up_proj.weight"], "float32"],
                          [1, ["model.norm.weight"], "float32"]], "refused": []}]
    names = {"model.layers.0.mlp.experts.0.up_proj.weight": (2, 2),
             "model.layers.0.mlp.experts.1.up_proj.weight": (2, 2), "model.norm.weight": (2,)}
    index = boundary.HfTaskIndex.from_manifests(records, names)
    assert index.tasks_for(["model.layers.0.mlp.experts.1.up_proj.weight"]) == [0]
    assert index.apply_plan(["model.layers.0.mlp.experts.1.up_proj.weight"]) == ([0], [0])
    with pytest.raises(boundary.MegatronHfBoundaryError, match="exports no tensor"):
        boundary.HfTaskIndex.from_manifests(records[:1], names)  # expert 1 missing: refused


def test_the_hybrid_rules_pass_a_dense_model_and_refuse_missing_experts():
    mfp.check_hybrid_rules(HF_SHAPES, {name: "float32" for name in HF_SHAPES}, {"model_type": "qwen3"})
    from yeto.rl.hybrid_params import HybridParamError

    with pytest.raises(HybridParamError):
        mfp.check_hybrid_rules({"model.layers.0.mixer.experts.up_proj": (3, 2, 2)}, {}, {"n_routed_experts": 4})
    with pytest.raises(HybridParamError, match="float32"):
        mfp.check_hybrid_rules({}, {"backbone.layers.0.mixer.A_log": "bfloat16"}, {})


# ----------------------------------------------------------------------------- the group and the adapter
def _config(**overrides):
    from yeto.rl.engine.full_parameter_stream import RoleStreamConfig

    layout = _layout(minimum_fragments=overrides.pop("minimum_fragments", 1))
    values = dict(
        algorithm="grpo", role="actor", component=COMPONENT, syncer_addr=("127.0.0.1", 30100),
        learner_id=0, learner_generation=0, learner_generations=(0, 0),
        total_fragment_steps=layout.fragments.num_fragments * 2,
        expected_fragments=layout.fragments.num_fragments, expected_layout_hash=layout.layout_hash,
        local_horizon=1, optimizer_steps_per_round=1, training_contract_hash="2" * 64,
        syncer_profile_hash="3" * 64, pipeline_depth=1, algorithm_spec_sha256="a1" * 32,
        max_fragment_bytes=MAX_FRAGMENT_BYTES, max_chunk_bytes=8)
    values.update(overrides)
    return RoleStreamConfig(**values)


def test_the_megatron_layout_is_the_fsdp2_layout():
    """The same HF shapes plan the same layout: the Megatron island's adapter
    accepts the FSDP2 planner's hash (``full_parameter_plan``)."""
    harness = _MegatronHarness()
    adapter, cut = vfs.asyncio.run(vfs.VerlChunkedFullParameterAdapter.initialize(harness.island.full_group,
                                                                               config=_config()))
    assert adapter.layout.layout_hash == _layout().layout_hash
    assert cut.layout_hash == _layout().layout_hash and all(f is not None for f in cut.fragments)
    assert "remote_megatron_manifest" in harness.island.calls
    assert "remote_megatron_export_flat" in harness.island.calls


def test_an_adapter_apply_of_changed_fragments_lands_and_is_proven():
    harness = _MegatronHarness()
    group = harness.island.full_group
    adapter, cut = vfs.asyncio.run(vfs.VerlChunkedFullParameterAdapter.initialize(group, config=_config()))
    k_fragment = next(i for i, names in enumerate(adapter._names) if QKV["k"] in names)
    values = _hf_values(harness.engine)
    new = adapter.flatten(k_fragment, {**values, QKV["k"]: values[QKV["k"]] + 0.5})
    stored = vfs.VerlStoredFragment(k_fragment, 5, vfs.fragment_hash(new), new)
    target = adapter.assemble_mixed_target(cut, target_policy_version=5, authoritative_fragments=[stored])
    written = vfs.asyncio.run(adapter.apply(group, target))
    assert written >= 1
    after = _hf_values(harness.engine)
    assert torch.equal(after[QKV["k"]], values[QKV["k"]] + 0.5)
    assert torch.equal(after[QKV["q"]], values[QKV["q"]]) and torch.equal(after[QKV["v"]], values[QKV["v"]])
    assert harness.island.calls.count("remote_megatron_apply") == 1


def test_an_apply_that_does_not_land_is_refused(monkeypatch):
    harness = _MegatronHarness()
    group = harness.island.full_group
    adapter, cut = vfs.asyncio.run(vfs.VerlChunkedFullParameterAdapter.initialize(group, config=_config()))
    monkeypatch.setattr(mfp, "remote_megatron_apply", lambda *a, **k: {"parameters_written": 0})
    values = cut.fragments[0] + 1.0
    stored = vfs.VerlStoredFragment(0, 3, vfs.fragment_hash(values), values)
    target = adapter.assemble_mixed_target(cut, target_policy_version=3, authoritative_fragments=[stored])
    with pytest.raises(RuntimeError, match="did not land"):
        vfs.asyncio.run(adapter.apply(group, target))


def test_offload_is_refused_on_the_worker():
    engine = _Engine()
    engine.is_param_offload_enabled = lambda: True
    with pytest.raises(mfp.VerlMegatronSyncError, match="offload"):
        mfp.remote_megatron_manifest(SimpleNamespace(actor=SimpleNamespace(engine=engine)))


# ----------------------------------------------------------------------------- a two-stage (PP2) fake
PIPELINE = (("embedding.word_embeddings.weight", "decoder.layers.0.input_layernorm.weight"),
            ("decoder.layers.0.self_attention.linear_qkv.weight", "decoder.final_layernorm.weight"))
EMBEDDING = "embedding.word_embeddings.weight"
TIED_OUTPUT = "output_layer.weight"


class _PipelineModel(torch.nn.Module):
    """One of two pipeline stages of :class:`_Model` (S27 G-FL): it registers only its
    own parameters and, on the last stage, the tied embedding's second copy
    ``output_layer.weight`` (which Megatron fills over the embedding group and no
    Bridge conversion task carries)."""

    def __init__(self, seed: int, stage: int):
        super().__init__()
        self.share_embeddings_and_output_weights = True
        self.pre_process, self.post_process = stage == 0, stage == 1
        generator = torch.Generator().manual_seed(seed)
        for name, (_hf, shape, dtype) in MEGATRON.items():
            value = torch.randn(shape, generator=generator).to(dtype)  # drawn for every stage: same v0
            if name not in PIPELINE[stage]:
                continue
            module = self
            *path, leaf = name.split(".")
            for part in path:
                if not hasattr(module, part):
                    module.add_module(part, torch.nn.Module())
                module = getattr(module, part)
            module.register_parameter(leaf, torch.nn.Parameter(value))
        if stage == 1:
            self.add_module("output_layer", torch.nn.Module())
            self.output_layer.register_parameter(
                "weight", torch.nn.Parameter(torch.zeros(MEGATRON[EMBEDDING][1], dtype=MEGATRON[EMBEDDING][2])))


class _PipelineBridge:
    """``get_conversion_tasks`` / ``export_hf_weights`` of one stage: every stage's
    tasks, with ``param_weight`` and ``megatron_module`` None for the other stage's
    (as Bridge fills them), and an export that stands in for Bridge's PP broadcast by
    reading the owning stage's FP32 master."""

    def __init__(self, stage: int):
        self.stage, self.peers = stage, []  # peers: [(model, optimizer)], set once both stages exist

    def get_conversion_tasks(self, modules):
        (model,) = modules
        mine = dict(model.named_parameters())
        return [SimpleNamespace(global_param_name=name, mapping=_Mapping(name, hf), param_weight=mine.get(name),
                                megatron_module=model if name in mine else None)
                for name, (hf, _shape, _dtype) in MEGATRON.items()]

    def _broadcast(self, name: str):
        for model, optimizer in self.peers:
            param = dict(model.named_parameters()).get(name)
            if param is not None:
                return megatron_masters.full_masters(optimizer, [param])[0]
        raise AssertionError(f"no pipeline stage owns {name}")

    def export_hf_weights(self, modules, cpu=False, show_progress=True, conversion_tasks=None,
                          merge_adapter_weights=True):
        for task in conversion_tasks:
            weights = task.param_weight
            if weights is None:  # Bridge's broadcast_from_pp_rank: the owner's value
                weights = self._broadcast(task.global_param_name)
            yield from task.mapping.megatron_to_hf(weights.detach(), task.megatron_module).items()


class _PipelineEngine(_Engine):
    def __init__(self, seed: int, stage: int):
        self.model = _PipelineModel(seed, stage)
        bf16 = [p for p in self.model.parameters() if p.dtype == torch.bfloat16]
        leaf = _DistributedLeaf(bf16, dp_rank=0, dp_size=1, group=None, seed=7 + seed)
        self.module, self.bridge = [self.model], _PipelineBridge(stage)
        self.optimizer = SimpleNamespace(chained_optimizers=[leaf])


def _tied_output(engine) -> torch.nn.Parameter | None:
    layer = getattr(engine.model, "output_layer", None)
    return None if layer is None else layer.weight


def _set_masters(engine, values: dict) -> None:
    """Write whole FP32 values into this stage's masters and model copies."""
    params = {name: param for name, param in engine.model.named_parameters() if name in values}
    plist = list(params.values())
    targets = [values[name] for name in params]
    with torch.no_grad():
        megatron_masters.write_masters(engine.optimizer, plist, targets)
        for param, target in zip(plist, targets, strict=True):
            param.data.copy_(target.to(param.dtype))


def _pipeline_engines(seed: int = 0) -> list:
    """Two stages holding the checkpoint ``_Engine(seed)`` holds, with the last stage's
    tied copy equal to the embedding (Megatron's start-up all-reduce)."""
    reference = _Engine(seed=seed)
    masters = {name: megatron_masters.full_masters(reference.optimizer, [param])[0].detach().clone()
               for name, param in reference.model.named_parameters()}
    engines = [_PipelineEngine(seed, stage) for stage in (0, 1)]
    peers = [(engine.model, engine.optimizer) for engine in engines]
    for engine in engines:
        engine.bridge.peers = peers
        _set_masters(engine, masters)
        tied = _tied_output(engine)
        if tied is not None:
            _set_masters(engine, {TIED_OUTPUT: masters[EMBEDDING]})
            assert tied.shape == masters[EMBEDDING].shape
    return engines


def _pipeline_hf_values(engines) -> dict:
    """What the pipeline trainer holds in HF names, read through stage 0 (the reading
    rank): its own masters plus what the broadcast delivers."""
    out: dict = {}
    engine = engines[0]
    tasks = boundary.conversion_tasks(engine.bridge, engine.module)
    boundary.export_hf(engine.bridge, engine.module, engine.optimizer, list(range(len(tasks))),
                       lambda name, tensor: out.__setitem__(name, tensor.detach().clone()))
    return out


def _pipeline_train(engines, delta: dict) -> None:
    """One optimizer step stand-in per stage. The tied copy takes the embedding's step,
    as Megatron's embedding-group gradient all-reduce makes it."""
    for engine in engines:
        params = dict(engine.model.named_parameters())
        masters = megatron_masters.full_masters(engine.optimizer, list(params.values()))
        values = {}
        for (name, _param), master in zip(params.items(), masters, strict=True):
            hf = MEGATRON[EMBEDDING][0] if name == TIED_OUTPUT else MEGATRON[name][0]
            step = delta[hf] if isinstance(hf, str) else torch.cat([delta[QKV[role]] for role in ("q", "k", "v")])
            values[name] = master.detach().clone() + step
        _set_masters(engine, values)


_READER_INDEX = threading.local()


def _is_reader_override() -> bool:
    """Stand-in for ``full_parameter._is_reader`` (global rank 0) while two in-process
    islands run in two threads: the index of the worker the island of *this* thread is
    calling. Thread-local, so a single-worker island in the other thread keeps 0."""
    return int(getattr(_READER_INDEX, "value", 0)) == 0


class _PipelineIsland(_Island):
    """``VerlIsland.call_workers`` over two in-process Megatron workers, one per
    pipeline stage. Worker 0 stands in for global rank 0, the reading rank; the test
    must point ``full_parameter._is_reader`` at :func:`_is_reader_override`."""

    def __init__(self, engines):
        self.workers = [SimpleNamespace(actor=SimpleNamespace(engine=engine)) for engine in engines]
        self.calls: list[str] = []

    def call_workers(self, fn, *args):
        self.calls.append(fn.__name__)
        out = []
        for index, worker in enumerate(self.workers):
            _READER_INDEX.value = index
            try:
                out.append(fn(worker, *args))
            finally:
                _READER_INDEX.value = 0
        return out

    call_workers_shared = call_workers


class _PipelineHarness:
    """One PP2 Megatron island's trainer (the harness interface of ``_MegatronHarness``)."""

    names = tuple(HF_SHAPES)

    def __init__(self):
        self.engines = _pipeline_engines(seed=0)
        self.island = _PipelineIsland(self.engines)
        self.island.full_group = mfp.MegatronFullParameterGroup(self.island, shapes=HF_SHAPES, config={})

    def values(self) -> dict:
        return _pipeline_hf_values(self.engines)

    @staticmethod
    def checkpoint_values() -> dict:
        return _pipeline_hf_values(_pipeline_engines(seed=0))

    def train(self, learner_id: int, round_id: int) -> None:
        _pipeline_train(self.engines, _step(learner_id, round_id))

    def tied_master(self) -> torch.Tensor:
        """The FP32 master of the last stage's tied output layer."""
        engine = self.engines[-1]
        return megatron_masters.full_masters(engine.optimizer, [_tied_output(engine)])[0].detach().clone()


# ----------------------------------------------------------------------------- two islands, the real syncer
def _plan(learner_id: int, port: int, profile: str, *, rounds: int, fragments: int, async_rounds: bool,
          matrix_merge: str, pipeline: int, final_cut: str) -> dict:
    layout = _layout(minimum_fragments=fragments, matrix_merge=matrix_merge)
    assert layout.fragments.num_fragments == fragments
    return {"learner_id": learner_id, "syncer": f"127.0.0.1:{port}", "model_revision": "a" * 40,
            vfs.PLAN_KEY: {
                "async_rounds": async_rounds, "num_learners": 2, "learner_generation": 0,
                "fragments": fragments, "total_fragment_steps": rounds * fragments,
                "parameter_layout_sha256": layout.layout_hash, "syncer_profile_sha256": profile,
                "model_config_sha256": "b" * 64, "training_contract_sha256": "2" * 64,
                "algorithm_spec_sha256": "a1" * 32, "local_horizon": 1, "pipeline": pipeline,
                "wait_timeout_s": 60.0, "wan_streams": 1, "push_dtype": "f32",
                "max_fragment_bytes": MAX_FRAGMENT_BYTES, "final_cut": final_cut,
                "local_init": async_rounds, "matrix_merge": matrix_merge, "island_pool": None}}


def _run_island(learner_id: int, plan: dict, max_rounds: int, results: dict, harness_cls=None) -> None:
    harness = (harness_cls or _MegatronHarness)()
    island = harness.island
    sync, _policy_state, _stream = vfs.build_full_parameter_sync(island, plan, backend_identity_sha256="c" * 64)
    driver = _Driver()
    try:
        state = sync.start(driver).state
        for round_id in range(max_rounds):
            harness.train(learner_id, round_id)
            ids = (f"l{learner_id}-r{round_id}",)
            batch_hash = hashlib.sha256(ids[0].encode()).hexdigest()
            policy_hash = state.policy_tensor_hash()
            island.full_group.record_round(
                LocalStepReceipt(algorithm="grpo", learner_id=learner_id, learner_generation=0,
                                 base_policy_version=round_id, base_policy_hash=policy_hash,
                                 input_batch_hash=batch_hash, trajectory_ids=ids, trained_tokens=TOKENS,
                                 optimizer_steps=1, optimizer_step_succeeded=True,
                                 parameter_layout_hash=plan[vfs.PLAN_KEY]["parameter_layout_sha256"]),
                vfs.VerlRoundEvidence(round_id, policy_hash, batch_hash, TOKENS, ids))
            boundary_ = sync.boundary(driver, rollout_id=round_id, stats=None)
            state = boundary_.state
            if boundary_.stop:
                sync.published(driver, rollout_id=round_id, policy_hash=state.policy_tensor_hash())
                break
        sync.finish(driver)
        results[learner_id] = {"values": harness.values(), "events": driver.events, "state": state,
                               "rounds": round_id + 1, "calls": list(island.calls),
                               "harness": harness}
    except BaseException as error:  # noqa: BLE001 - reported by the test
        results[learner_id] = error
    finally:
        sync.close()


def _two_islands(binary, tmp_path, monkeypatch, *, rounds, fragments, async_rounds, extra, matrix_merge="avg",
                 pipeline=1, final_cut="download", plan_extra=None, harnesses=(None, None)):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "verl-megatron-sync-test-key")
    monkeypatch.setenv("TOKIO_WORKER_THREADS", "4")
    process, port, profile = _start_syncer(binary, tmp_path, fragments=fragments, rounds=rounds, extra=extra)
    results: dict = {}
    try:
        plans = [_plan(i, port, profile, rounds=rounds, fragments=fragments, async_rounds=async_rounds,
                       matrix_merge=matrix_merge, pipeline=pipeline, final_cut=final_cut) for i in range(2)]
        for plan in plans:
            plan[vfs.PLAN_KEY].update(plan_extra or {})
        threads = [threading.Thread(target=_run_island, args=(i, plans[i], rounds if not async_rounds
                                                               else 20 * rounds, results, harnesses[i]))
                   for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        assert process.wait(timeout=30) == 0
    finally:
        if process.poll() is None:
            process.terminate()
        log = process.communicate(timeout=10)[0]
    for learner_id in range(2):
        assert not isinstance(results.get(learner_id), BaseException), (results.get(learner_id), log[-3000:])
        assert learner_id in results, log[-3000:]
    return results, log


def _expected_average(rounds: int) -> dict[str, torch.Tensor]:
    """Outer lr 1, no momentum, equal weights: the masters at load plus the islands' mean step per round."""
    out = _hf_values(_Engine(seed=0))
    for round_id in range(rounds):
        for name in out:
            out[name] = out[name] + (_step(0, round_id)[name] + _step(1, round_id)[name]) / 2
    return out


def _counting_kinds(monkeypatch) -> list[int]:
    from yeto import downlink

    kinds: list[int] = []
    begin = downlink.DownlinkAnchors.begin

    def counted(self, fragment_id, version, header, numel):
        kinds.append(downlink.HEADER.unpack(header)[0])
        return begin(self, fragment_id, version, header, numel)

    monkeypatch.setattr(downlink.DownlinkAnchors, "begin", counted)
    return kinds


@needs_cargo
def test_two_megatron_islands_run_a_strict_sweep_with_bf16_broadcast_deltas(syncer_binary, tmp_path, monkeypatch):
    """Strict sweep, FULL every second broadcast and bf16 DELTA in between: the
    final cut each island downloads is the exact average of the islands' steps
    on top of the FP32 masters (atol 1e-6); both islands ACK it."""
    from yeto import downlink

    kinds = _counting_kinds(monkeypatch)
    rounds, fragments = 4, len(_layout().fragments.fragments)
    results, _log = _two_islands(syncer_binary, tmp_path, monkeypatch, rounds=rounds, fragments=fragments,
                                 async_rounds=False, plan_extra={"bcast_delta": True},
                                 extra=("--rl-sync-full-parameter-rounds", "sweep", "--rl-sync-final-cut", "download",
                                        "--rl-sync-bcast-delta-full-every", "2"))
    assert downlink.KIND_DELTA in kinds and downlink.KIND_FULL in kinds, kinds
    expected = _expected_average(rounds)
    for learner_id in range(2):
        for name, value in expected.items():
            assert torch.allclose(results[learner_id]["values"][name], value, atol=1e-6), (learner_id, name)
        acks = [f for e, f in results[learner_id]["events"] if e == "rl_full_parameter_final_ack"]
        assert acks == [{"acknowledged": True}]
        assert "remote_megatron_apply" in results[learner_id]["calls"]
    assert results[0]["state"].policy_tensor_hash() == results[1]["state"].policy_tensor_hash()


@needs_cargo
def test_a_pipeline_parallel_megatron_island_syncs_with_a_single_stage_one(syncer_binary, tmp_path, monkeypatch):
    """S27 G-FL: island 1 is PP2 (two workers, one per stage, the tied embedding's
    second copy on the last stage), island 0 is single-stage. Strict sweep with bf16
    broadcast deltas: both islands end on the exact average of their steps (atol
    1e-6) and ACK the same final cut, and the tied copy still equals the embedding."""
    from yeto import downlink

    monkeypatch.setattr(fp, "_is_reader", _is_reader_override)
    kinds = _counting_kinds(monkeypatch)
    rounds, fragments = 4, len(_layout().fragments.fragments)
    results, log = _two_islands(syncer_binary, tmp_path, monkeypatch, rounds=rounds, fragments=fragments,
                                async_rounds=False, plan_extra={"bcast_delta": True},
                                harnesses=(None, _PipelineHarness),
                                extra=("--rl-sync-full-parameter-rounds", "sweep", "--rl-sync-final-cut", "download",
                                       "--rl-sync-bcast-delta-full-every", "2"))
    assert downlink.KIND_DELTA in kinds and downlink.KIND_FULL in kinds, kinds
    expected = _expected_average(rounds)
    for learner_id in range(2):
        for name, value in expected.items():
            assert torch.allclose(results[learner_id]["values"][name], value, atol=1e-6), (learner_id, name)
        acks = [f for e, f in results[learner_id]["events"] if e == "rl_full_parameter_final_ack"]
        assert acks == [{"acknowledged": True}], (learner_id, log[-2000:])
    assert results[0]["state"].policy_tensor_hash() == results[1]["state"].policy_tensor_hash()
    # the second copy of the embedding went with it, so the last stage's head is current
    tied = results[1]["harness"].tied_master()
    assert torch.allclose(tied, expected["model.embed_tokens.weight"], atol=1e-6)


@needs_cargo
def test_two_megatron_islands_run_async_rounds_with_local_init_and_bf16_deltas(syncer_binary, tmp_path,
                                                                              monkeypatch):
    """The async defaults (quorum 1, two rounds in flight, RDA, local INIT) with
    deltas: both islands ACK the same downloaded final cut."""
    from yeto import downlink

    kinds = _counting_kinds(monkeypatch)
    rounds, fragments = 4, len(_layout(matrix_merge="rda").fragments.fragments)
    results, log = _two_islands(syncer_binary, tmp_path, monkeypatch, rounds=rounds, fragments=fragments,
                                async_rounds=True, plan_extra={"bcast_delta": True}, matrix_merge="rda",
                                pipeline=2, extra=("--rl-sync-full-parameter-rounds", "async",
                                                   "--rl-sync-final-cut", "download",
                                                   "--rl-sync-bcast-delta-full-every", "2"))
    assert downlink.KIND_DELTA in kinds and downlink.KIND_FULL in kinds, kinds
    for learner_id in range(2):
        acks = [f for e, f in results[learner_id]["events"] if e == "rl_full_parameter_final_ack"]
        assert acks == [{"acknowledged": True}], (learner_id, log[-2000:])
    for name in HF_SHAPES:
        assert torch.equal(results[0]["values"][name], results[1]["values"][name]), name


def _pool_plan(learner_id, port, profile, *, rounds, fragments):
    plan = _plan(learner_id, port, profile, rounds=rounds, fragments=fragments, async_rounds=True,
                 matrix_merge="rda", pipeline=2, final_cut="download")
    plan[vfs.PLAN_KEY]["island_pool"] = {"lease_s": 3.0, "syncer_epoch": 0}
    return plan


@needs_cargo
def test_a_restarted_megatron_island_catches_up_from_the_running_pool(syncer_binary, tmp_path, monkeypatch):
    """Island 1 dies and a new process joins the running pool by catch-up (no
    peer recovery): it takes the syncer's fragments as anchors, contributes and
    ACKs the same final cut as island 0 (the FSDP2 case of test_s27_verl_parity)."""
    from test_s27_verl_parity import run_catch_up_case

    run_catch_up_case(syncer_binary, tmp_path, monkeypatch, harness=_MegatronHarness,
                      layout=_layout(matrix_merge="rda"), plan_fn=_pool_plan)


# ----------------------------------------------------------------------------- real Megatron-Core and Bridge
_REAL = {key: os.environ.get(key) for key in ("YETO_MEGATRON_PYTHON", "YETO_MEGATRON_LM_PATH",
                                              "YETO_MEGATRON_BRIDGE_PATH")}


def _round_trip(tmp_path, tp: int, pp: int, dp: int, *, mutate: str = "", timeout: int = 900):
    out = tmp_path / f"result-{tp}-{pp}-{dp}-{mutate or 'none'}.json"
    env = {**os.environ, "OMP_NUM_THREADS": "2", "YETO_PP_MUTATE": mutate}
    proc = subprocess.run([_REAL["YETO_MEGATRON_PYTHON"], "-I", str(REPO / "tests/megatron_bridge_roundtrip.py"),
                           str(tp), str(pp), str(dp), str(out)], env=env, capture_output=True, text=True,
                          timeout=timeout)
    return proc, out


@pytest.mark.skipif(not all(_REAL.values()), reason="needs YETO_MEGATRON_PYTHON / _LM_PATH / _BRIDGE_PATH")
@pytest.mark.parametrize("tp,pp,dp", [(2, 1, 1), (1, 1, 2), (2, 1, 2), (1, 2, 1), (2, 2, 1), (1, 2, 2)])
def test_real_megatron_and_bridge_round_trip(tp, pp, dp, tmp_path):
    proc, out = _round_trip(tmp_path, tp, pp, dp)
    assert proc.returncode == 0, proc.stderr[-4000:]
    result = json.loads(out.read_text())
    assert result["ok"] and result["names"] == 24 and result["names_read_from_masters"] == 15
    assert result["partial_changed"] == ["model.layers.0.self_attn.q_proj.weight",
                                         "model.layers.1.input_layernorm.weight"]
    # S27 G-FL: with PP > 1 and tied embeddings every last-stage rank holds the second
    # copy of the embedding and writes it; with PP == 1 nobody does.
    expected = dp * tp if pp > 1 else 0
    assert result["tied_copies"] == expected and result["tied_outputs_written"] == expected


@pytest.mark.skipif(not all(_REAL.values()), reason="needs YETO_MEGATRON_PYTHON / _LM_PATH / _BRIDGE_PATH")
@pytest.mark.parametrize("mutate", ["no-tied", "stage-local-export"])
def test_the_pipeline_round_trip_catches_a_broken_cross_stage_step(mutate, tmp_path):
    """S27 G-FL mutation check: breaking the tied-copy write, or dropping what the PP
    broadcast delivered, must make the PP2 round trip fail (it passes at PP1, where
    neither step exists)."""

    broken, _out = _round_trip(tmp_path, 1, 2, 1, mutate=mutate)
    assert broken.returncode != 0, broken.stdout[-2000:]
    fine, out = _round_trip(tmp_path, 2, 1, 1, mutate=mutate)
    assert fine.returncode == 0, fine.stderr[-4000:]
    assert json.loads(out.read_text())["ok"]


# ----------------------------------------------------------------------------- launcher and island entry
MEGATRON_GPU = "modal:4xh200,modal:4xh200"


def _megatron_launch(*extra):
    from test_s27_verl_full_parameter_launch import _prepared

    from yeto.rl.adapters.verl.full_parameter_plan import model_shapes  # noqa: F401 - the planner the hash comes from

    return _prepared(_launcher_args("ports", (
        "--rl-backend", "verl", "--rl-sync-preset", "full-parameter",
        "--rl-parameter-layout-sha256", "e" * 64, "--fragments", "4", "--reward-function", "gsm8k_reward:score",
        "--tensor-parallel", "2", "--rl-placement", "fixed-partition", "--rl-rollout-gpus", "2",
        "--rl-max-policy-age", "0", *extra), gpu=MEGATRON_GPU))


def test_two_synced_megatron_islands_launch_and_their_argv_reaches_the_island_plan(tmp_path):
    from test_s27_verl_full_parameter_launch import MILES_ONLY, _flags, _island_argv

    from yeto import launcher
    from yeto.rl.adapters.verl import island_entry

    args = _megatron_launch()
    assert args.rl_sync_mode_effective == "async" and args.rl_peer_recovery is False
    profile = launcher.rl_syncer_profile(args, 2).sha256
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "qwen3"}')
    for learner_id in range(2):
        argv = _island_argv(args, learner_id)
        parsed, ignored = island_entry.parse_args(argv)
        assert _flags(ignored) <= MILES_ONLY, _flags(ignored) - MILES_ONLY
        assert parsed.tensor_parallel == 2 and not parsed.rl_single_island_no_sync
        plan = island_entry.megatron_island_plan(
            parsed, model_path=str(model), train_file="/d/train.parquet", val_file="/d/test.parquet",
            out_dir="/o", algo={"correction": "tis", "tis_upper": 2.0, "tis_lower": 0.0}, chat={},
            ray_info={"after": {"GPU": 4}}, reward_path="/r.py", reward_name="gsm8k")
        shape = plan["shape"]
        assert (shape["tp"], shape["trainer_gpus"], shape["rollout_gpus"]) == (2, 2, 2)
        section = vfs.plan_section(parsed, plan["run"].to_dict(), model_path=str(model), sync="strict",
                                   algorithm_spec_sha256="a" * 64)["full_parameter"]
        assert section["syncer_profile_sha256"] == profile and section["parameter_layout_sha256"] == "e" * 64
        assert section["async_rounds"] and "peer_recovery" not in section


def test_an_fsdp2_island_refuses_the_ascend_plugin_stack():
    """S27 G-FL: the stack patches Megatron-Core, so only an island with model
    parallelism takes it; without the flag there is no section at all."""
    from types import SimpleNamespace

    from yeto.rl.adapters.verl import island_entry
    from yeto.rl.adapters.verl.megatron_engine import ENGINE_FSDP2, ENGINE_MEGATRON

    off = SimpleNamespace(rl_ascend_plugin_stack=None)
    assert island_entry.ascend_plugin_stack_section(off, engine=ENGINE_FSDP2) is None
    assert island_entry.ascend_plugin_stack_section(off, engine=ENGINE_MEGATRON) is None
    on = SimpleNamespace(rl_ascend_plugin_stack="/data/work/sft-run/plugins")
    with pytest.raises(SystemExit, match="只接有模型并行"):
        island_entry.ascend_plugin_stack_section(on, engine=ENGINE_FSDP2)
    section = island_entry.ascend_plugin_stack_section(on, engine=ENGINE_MEGATRON)
    assert section["root"] == "/data/work/sft-run/plugins"
    assert section["env"]["STACK_REWRITES"] == "cumsum,squeeze"


def test_peer_recovery_stays_off_for_megatron_islands_with_a_store():
    from yeto.launch.rl_args import _peer_recovery_applies

    args = _megatron_launch()
    args.rl_checkpoint_store = "modal-volume://v/run"
    assert _peer_recovery_applies(args) is False  # an FSDP2 run turns it on here (test_s27_verl_full_parameter_launch)


@pytest.mark.parametrize("change,match", [
    (dict(rl_peer_recovery=True), "no peer recovery"),
    (dict(rl_sync_preset="decoupled"), "--rl-sync-preset full-parameter"),
    (dict(rl_single_island_no_sync=True, tuning="lora"), "--tuning full"),
])
def test_the_launcher_refuses_what_a_synced_megatron_island_cannot_run(change, match):
    from argparse import Namespace

    from yeto.rl.adapters.verl import run_config_rules

    values = dict(tuning="lora", rl_single_island_no_sync=False, rl_sync_preset="full-parameter",
                  rl_placement="fixed-partition", rl_max_policy_age=0, tensor_parallel=2, pipeline_parallel=1,
                  expert_parallel=1, rl_peer_recovery=False)
    assert run_config_rules.megatron_problems(Namespace(**values)) == []
    values.update(change)
    with pytest.raises(ValueError, match=match):
        run_config_rules.check_engine_shape(Namespace(**values))


def test_pipeline_parallelism_is_gated_and_the_gate_names_the_cause(tmp_path):
    """S27 G-FL: the boundary handles PP (CPU-verified), but a PP > 1 island does not
    start on cards (run s27-fl-1: the stage-1 trainer rank segfaults at the manifest
    step), so the launcher refuses the shape by default and says why.
    ``--rl-allow-pipeline-parallel-sync`` lifts the gate; the code stays in place."""
    from argparse import Namespace

    from yeto.rl.adapters.verl import megatron_engine, run_config_rules

    values = dict(tuning="lora", rl_single_island_no_sync=False, rl_sync_preset="full-parameter",
                  rl_placement="fixed-partition", rl_max_policy_age=0, tensor_parallel=1, pipeline_parallel=2,
                  expert_parallel=1, rl_peer_recovery=False, rl_allow_pipeline_parallel_sync=False)
    (problem,) = megatron_engine.sync_problems(Namespace(**values))
    assert "s27-fl-1" in problem and "VERL-MEGATRON-PP.md" in problem
    assert "--rl-allow-pipeline-parallel-sync" in problem
    with pytest.raises(ValueError, match="--pipeline-parallel 1"):
        run_config_rules.check_engine_shape(Namespace(**values))
    lifted = {**values, "rl_allow_pipeline_parallel_sync": True}
    assert megatron_engine.sync_problems(Namespace(**lifted)) == []
    assert run_config_rules.megatron_problems(Namespace(**lifted)) == []
    run_config_rules.check_engine_shape(Namespace(**lifted))  # does not raise
    # PP == 1 is unaffected either way
    for allow in (False, True):
        single = {**values, "pipeline_parallel": 1, "tensor_parallel": 2,
                  "rl_allow_pipeline_parallel_sync": allow}
        assert megatron_engine.sync_problems(Namespace(**single)) == []


# ----------------------------------------------------------------------------- the island driver over the Megatron wiring
class _CheckpointManager:
    """verl's ``standalone_checkpoint_manager``: the rollout servers get the trainer's
    bf16 model parameters (what Megatron-Bridge sends to vLLM)."""

    def __init__(self, engine: _Engine):
        self.engine = engine
        self.calls: list[int] = []
        self.rollout_weights: dict[str, torch.Tensor] = {}

    def update_weights(self, version):
        self.calls.append(int(version))
        self.rollout_weights = {n: p.detach().clone() for n, p in self.engine.model.named_parameters()}
        return {"sync/bytes": 2.0 * sum(v.numel() for v in self.rollout_weights.values())}


class _DriverIsland(_Island):
    """The parts of ``MegatronIsland`` the driver's ports read (publisher, tape)."""

    def __init__(self, engine: _Engine, out: Path, learner_id: int, tape: list):
        super().__init__(engine)
        self.learner_id, self.out, self.published, self.full_group = learner_id, out, None, None
        self.emit = lambda event, **fields: tape.append({"event": event, **fields})
        self.trainer = SimpleNamespace(standalone_checkpoint_manager=_CheckpointManager(engine))

    def append_jsonl(self, name, record):
        with (self.out / name).open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def _weights_hash(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        digest.update(name.encode() + values[name].float().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _driver_ports(island: _DriverIsland, engine: _Engine, learner_id: int, plan: dict, served: list):
    from yeto.rl.adapters.verl.ports_impl import MEMBER
    from yeto.rl.engine.driver import TrainStepMetrics, policy_token
    from yeto.rl.engine.ports import GroupMetadata, RolloutBatchHandle

    class Rollout:
        def generate(self, rollout_id, *, expected_policy_version=None):
            version, digest = island.published
            served.append((rollout_id, _weights_hash(island.trainer.standalone_checkpoint_manager.rollout_weights)))
            token = policy_token(version, digest)
            ids = (f"l{learner_id}-r{rollout_id}-s0", f"l{learner_id}-r{rollout_id}-s1")
            groups = (GroupMetadata(f"l{learner_id}-r{rollout_id}-g0", ids, token, 0.5, 0.5, TOKENS),)
            return RolloutBatchHandle(rollout_id, version, digest, groups, 1, 0, payload=None)

        def abort(self):
            return None

        def members(self):
            return frozenset({MEMBER})

    class Trainer:
        publish_offloaded = False

        def onload(self):
            return None

        def offload(self):
            return None

        def train_step(self, batch):
            _train(engine, _step(learner_id, batch.rollout_id))
            ids = tuple(s for g in batch.groups for s in g.sample_ids)
            batch_hash = hashlib.sha256("|".join(ids).encode()).hexdigest()
            receipt = LocalStepReceipt(
                algorithm="grpo", learner_id=learner_id, learner_generation=0,
                base_policy_version=batch.policy_version, base_policy_hash=batch.policy_hash,
                input_batch_hash=batch_hash, trajectory_ids=ids, trained_tokens=TOKENS, optimizer_steps=1,
                optimizer_step_succeeded=True, parameter_layout_hash=plan[vfs.PLAN_KEY]["parameter_layout_sha256"])
            island.full_group.record_round(receipt, vfs.VerlRoundEvidence(
                batch.rollout_id, batch.policy_hash, batch_hash, TOKENS, ids))
            return receipt

        def step_metrics(self):
            return TrainStepMetrics(grad_norm=0.5, loss=0.1, pg_loss=0.1, lr=1e-6, train_step=0, applied_lrs=(1e-6,))

    return Rollout(), Trainer()


def _run_driver_island(learner_id: int, plan: dict, tmp_path: Path, results: dict) -> None:
    from yeto.rl.adapters.verl import megatron_ports as mp
    from yeto.rl.adapters.verl.driver_extras import TestExitPublisher
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.driver import EventTape, IslandDriver

    out = tmp_path / f"island-{learner_id}"
    out.mkdir()
    tape: list[dict] = []
    engine = _Engine(seed=0)
    island = _DriverIsland(engine, out, learner_id, tape)
    served: list = []
    sync = None
    try:
        # run_megatron_island's wiring: sync_ports (with the Megatron group), the NCCL
        # publisher behind the test-exit wrapper, and driver_options
        island.full_group = mfp.MegatronFullParameterGroup(island, shapes=HF_SHAPES, config={})
        sync, policy_state = mp.sync_ports(island, plan, "c" * 64)
        publisher = TestExitPublisher(mp.MegatronPublisher(island), None, tape)
        spec = AlgorithmSpec()
        rollout, trainer = _driver_ports(island, engine, learner_id, plan, served)
        driver = IslandDriver(learner_id=learner_id, rollout=rollout, trainer=trainer, policy_state=policy_state,
                              publisher=publisher, algorithm=spec, sync=sync,
                              events=EventTape(out / "tape.jsonl", learner_id),
                              **mp.driver_options(plan, spec, "sha256:" + "1" * 64))
        state = driver.run()
        results[learner_id] = {"island": island, "engine": engine, "state": state, "served": served,
                               "rounds": driver.rounds_completed, "out": out}
    except BaseException as error:  # noqa: BLE001 - reported by the test
        results[learner_id] = error
    finally:
        if sync is not None:
            sync.close()


@needs_cargo
@pytest.mark.parametrize("mode", ["sweep", "async"])
def test_two_megatron_island_drivers_publish_every_round_through_the_real_syncer(syncer_binary, tmp_path,
                                                                                monkeypatch, mode):
    """The real ``IslandDriver`` over ``megatron_ports.sync_ports`` / ``driver_options``
    (the partitioned-serial profile with the strict-avg outer protocol, the Megatron
    capabilities, the fragments layout) and the NCCL publisher: every round is
    published in order, generated on its publication, and both islands ACK one final cut."""
    from test_rl_verl_megatron import D0

    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "verl-megatron-driver-test-key")
    monkeypatch.setenv("TOKIO_WORKER_THREADS", "4")
    async_rounds = mode == "async"
    merge = "rda" if async_rounds else "avg"
    rounds, fragments = 4, len(_layout(matrix_merge=merge).fragments.fragments)
    process, port, profile = _start_syncer(syncer_binary, tmp_path, fragments=fragments, rounds=rounds, extra=(
        "--rl-sync-full-parameter-rounds", mode, "--rl-sync-final-cut", "download"))
    results: dict = {}
    try:
        plans = []
        for i in range(2):
            plan = _plan(i, port, profile, rounds=rounds, fragments=fragments, async_rounds=async_rounds,
                         matrix_merge=merge, pipeline=2 if async_rounds else 1, final_cut="download")
            plan.update(sync="strict", global_rounds=rounds, groups_per_round=1, samples_per_group=2,
                        megatron=D0.to_dict(), model_path="/unused")
            plans.append(plan)
        threads = [threading.Thread(target=_run_driver_island, args=(i, plans[i], tmp_path, results))
                   for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=180)
        assert process.wait(timeout=30) == 0
    finally:
        if process.poll() is None:
            process.terminate()
        log = process.communicate(timeout=10)[0]
    for learner_id in range(2):
        result = results.get(learner_id)
        assert not isinstance(result, BaseException) and result is not None, (result, log[-3000:])
        manager = result["island"].trainer.standalone_checkpoint_manager
        assert manager.calls == list(range(len(manager.calls))) and len(manager.calls) >= 3, manager.calls
        assert [rid for rid, _h in result["served"]] == list(range(result["rounds"]))
        lines = [json.loads(line) for line in (result["out"] / "tape.jsonl").read_text().splitlines()]
        start = next(r for r in lines if r["event"] == "rl_driver_start")
        assert (start["placement"], start["execution_mode"]) == ("fixed-partition", "partitioned-serial")
        assert [r.get("acknowledged") for r in lines if r["event"] == "rl_full_parameter_final_ack"] == [True]
        assert {"remote_megatron_manifest", "remote_megatron_export_flat",
                "remote_megatron_apply"} <= set(result["island"].calls)
    if not async_rounds:
        assert results[0]["served"] == results[1]["served"]
    final = [_hf_values(results[i]["engine"]) for i in range(2)]
    for name in HF_SHAPES:
        assert torch.equal(final[0][name], final[1][name]), name
