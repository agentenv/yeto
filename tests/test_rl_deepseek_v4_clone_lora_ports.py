"""DeepSeek V4 clone-only expert LoRA on the ports Miles layout (P0 1.7b.2b).

The ports state plugin binds each packed grouped-expert adapter to one
representative name; :class:`ClonePackedAdapterPolicy` makes the sync policy
carry the clone experts only and packs them back into each EP rank's slots.
"""

from __future__ import annotations

import os
import runpy
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from yeto.rl.adapters.miles import state_plugin as sp
from yeto.rl.adapters.miles.models.deepseek_v4 import clone_lora, ports_runtime
from yeto.rl.adapters.miles.models.deepseek_v4.clone_lora import ClonePackedAdapterPolicy
from yeto.rl.adapters.miles.models.deepseek_v4.expert_clone import (
    ORIGINAL_EXPERTS,
    ORIGINAL_EXPERTS_PER_RANK,
    TOTAL_EXPERTS,
    TRAINING_EXPERTS_PER_RANK,
    training_to_logical_expert_id,
)

ROOT = Path(__file__).resolve().parents[1]
ATTENTION = "base_model.model.model.layers.0.self_attn.wq_a.lora_A.weight"
SLICE = (2, 3)


def _down_b(expert: int) -> str:
    return f"base_model.model.model.layers.0.mlp.experts.{expert}.down_proj.lora_B.weight"


# The real policy carries 43 layers x 32 clones x 6 sides; the fixture model
# has one packed down_proj B, so the carried set is narrowed to it.
LAYER0_DOWN_B_CLONES = frozenset(_down_b(e) for e in range(ORIGINAL_EXPERTS, TOTAL_EXPERTS))


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = torch.nn.Parameter(torch.full((2, 4), 0.5))
        self.packed = torch.nn.Parameter(torch.zeros(TRAINING_EXPERTS_PER_RANK, *SLICE))


class _Actor:
    def __init__(self, *, expert_parallel_rank: int = 0):
        self.module = _Model()
        self.model = [self.module]
        self.expert_parallel_rank = expert_parallel_rank
        self.args = SimpleNamespace(global_batch_size=4, tensor_model_parallel_size=1,
                                    pipeline_model_parallel_size=1, expert_model_parallel_size=8,
                                    hf_checkpoint="/ckpt")
        self.opt_param_scheduler = SimpleNamespace(num_steps=0, step=lambda increment: None)
        self._is_first_replica_megatron_main_rank = True
        ident = lambda t: t.clone()  # noqa: E731
        self._yeto_adapter_bindings = (
            sp.AdapterBinding(ATTENTION, self.module.attention, ident, ident),
            sp.AdapterBinding(_down_b(0), self.module.packed, sp._no_conversion, sp._no_conversion,
                              packed=True),
        )


def _logical_ids(rank: int) -> list[int]:
    start = rank * TRAINING_EXPERTS_PER_RANK
    return [training_to_logical_expert_id(start + slot) for slot in range(TRAINING_EXPERTS_PER_RANK)]


@pytest.fixture
def clone_policy(monkeypatch):
    monkeypatch.setattr(clone_lora, "_CLONE_EXPERT_LORA_NAMES", LAYER0_DOWN_B_CLONES)
    policy = ClonePackedAdapterPolicy(expert_parallel_rank=0, expert_parallel_size=8)
    monkeypatch.setattr(sp, "_PACKED_POLICY", policy)
    return policy


@pytest.fixture
def fake_bridge(monkeypatch):
    """Bridge export of every logical expert (originals 0, clone e = e), like the real gather."""

    exported: dict[str, torch.Tensor] = {}

    def export_adapter_weights(model, *, cpu, show_progress):
        chunk = model[0]
        yield ATTENTION.removeprefix("base_model.model."), chunk.attention, "attention"
        for expert in range(TOTAL_EXPERTS):
            value = exported.get(_down_b(expert), torch.zeros(SLICE))
            yield _down_b(expert).removeprefix("base_model.model."), value, "packed"

    bridge = SimpleNamespace(export_adapter_weights=export_adapter_weights)
    modules = {
        "miles": ModuleType("miles"),
        "miles.backends": ModuleType("miles.backends"),
        "miles.backends.megatron_utils": ModuleType("miles.backends.megatron_utils"),
        "miles.backends.megatron_utils.hf_export": ModuleType("miles.backends.megatron_utils.hf_export"),
        "miles.utils": ModuleType("miles.utils"),
        "miles.utils.megatron_bridge_utils": ModuleType("miles.utils.megatron_bridge_utils"),
    }
    modules["miles.backends.megatron_utils.hf_export"]._get_hf_bridge = lambda _path: bridge
    modules["miles.utils.megatron_bridge_utils"].patch_megatron_model = lambda _model: nullcontext()
    modules["miles.utils"].megatron_bridge_utils = modules["miles.utils.megatron_bridge_utils"]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return exported


def test_carried_names_are_the_synthesized_sparse_clone_specs():
    from tests.test_rl_export import _expanded_deepseek_v4_config
    from yeto.rl import export as rl_export

    specs = rl_export._deepseek_v4_clone_expert_lora_specs(_expanded_deepseek_v4_config(), rank=8)

    assert ClonePackedAdapterPolicy().carried_names() == frozenset(spec.name for spec in specs)


def test_packed_name_is_the_bridge_representative_of_expert_sides_only():
    policy = ClonePackedAdapterPolicy()
    fc1_a = "model.layers.3.mlp.experts.0.gate_proj.lora_A.weight"

    def side(hf_param):
        return SimpleNamespace(mapping=SimpleNamespace(hf_param=hf_param))

    assert policy.packed_name(side({"gate": fc1_a, "up": fc1_a})) == "base_model.model." + fc1_a
    assert policy.packed_name(side(ATTENTION)) is None
    assert policy.packed_name(side({"q": "a", "k": "b"})) is None


def test_bindings_name_packed_sides_by_policy_and_refuse_conversion(monkeypatch, clone_policy):
    packed = torch.nn.Parameter(torch.zeros(TRAINING_EXPERTS_PER_RANK, *SLICE))
    attention = torch.nn.Parameter(torch.zeros(2, 4))
    model = torch.nn.Module()
    model.packed, model.attention = packed, attention

    def task(param, hf_param, converted):
        mapping = SimpleNamespace(hf_param=hf_param, megatron_to_hf=lambda _t, _m: converted)
        side = SimpleNamespace(param_weight=param, param_name="p", megatron_module=None, mapping=mapping)
        return SimpleNamespace(adapter_key=hf_param, linear_in_task=side,
                               linear_out_task=SimpleNamespace(param_weight=None))

    tasks = {"experts": [task(packed, _down_b(0).removeprefix("base_model.model."), {})],
             "attention": [task(attention, ATTENTION, {ATTENTION: None})]}
    bridge = SimpleNamespace(_model_bridge=SimpleNamespace(build_adapter_conversion_tasks=lambda _m: tasks))
    hf_export = ModuleType("miles.backends.megatron_utils.hf_export")
    hf_export._get_hf_bridge = lambda _path: bridge
    monkeypatch.setitem(sys.modules, hf_export.__name__, hf_export)
    actor = SimpleNamespace(model=[model], args=SimpleNamespace(hf_checkpoint="/ckpt"))

    bindings = {b.name: b for b in sp.adapter_bindings(actor)}

    assert bindings[_down_b(0)].packed and not bindings[ATTENTION].packed
    with pytest.raises(sp.StatePluginError, match="PackedAdapterPolicy"):
        bindings[_down_b(0)].to_hf(packed)


def test_export_carries_clone_experts_only(clone_policy, fake_bridge):
    fake_bridge.update({_down_b(e): torch.full(SLICE, float(e)) for e in range(ORIGINAL_EXPERTS, TOTAL_EXPERTS)})

    out = sp._export_state(_Actor(), policy_version=3)

    assert set(out["tensors"]) == {ATTENTION} | LAYER0_DOWN_B_CLONES
    assert float(out["tensors"][_down_b(287)][0, 0]) == 287.0


def test_export_refuses_a_nonzero_original_slot(clone_policy, fake_bridge):
    actor = _Actor()
    with torch.no_grad():
        actor.module.packed[ORIGINAL_EXPERTS_PER_RANK - 1] = 1.0

    with pytest.raises(RuntimeError, match="original packed expert"):
        sp._export_state(actor, policy_version=1)


def test_export_refuses_a_missing_clone_expert(monkeypatch, clone_policy, fake_bridge):
    monkeypatch.setattr(clone_lora, "_CLONE_EXPERT_LORA_NAMES", LAYER0_DOWN_B_CLONES | {_down_b(999)})

    with pytest.raises(sp.StatePluginError, match="exported packed expert LoRA mismatch"):
        sp._export_state(_Actor(), policy_version=1)


@pytest.mark.parametrize("rank", [0, 7])
def test_apply_writes_each_ranks_clone_slots_and_zeroes_originals(monkeypatch, rank):
    monkeypatch.setattr(clone_lora, "_CLONE_EXPERT_LORA_NAMES", LAYER0_DOWN_B_CLONES)
    monkeypatch.setattr(sp, "_PACKED_POLICY", ClonePackedAdapterPolicy(expert_parallel_rank=rank,
                                                                        expert_parallel_size=8))
    actor = _Actor(expert_parallel_rank=rank)
    with torch.no_grad():
        actor.module.packed.fill_(9.0)
    tensors = {ATTENTION: torch.ones(2, 4)}
    tensors.update({_down_b(e): torch.full(SLICE, float(e)) for e in range(ORIGINAL_EXPERTS, TOTAL_EXPERTS)})

    summary = sp._apply_state(actor, tensors=tensors, policy_version=1, local_step=0, optimizer="preserve")

    packed = actor.module.packed.detach()
    assert summary["tensors"] == len(tensors)
    assert torch.count_nonzero(packed[:ORIGINAL_EXPERTS_PER_RANK]) == 0
    clones = [int(packed[slot, 0, 0]) for slot in range(ORIGINAL_EXPERTS_PER_RANK, TRAINING_EXPERTS_PER_RANK)]
    assert clones == _logical_ids(rank)[ORIGINAL_EXPERTS_PER_RANK:]
    assert torch.equal(actor.module.attention.detach(), torch.ones(2, 4))
    assert actor.module.packed.requires_grad


def test_apply_refuses_an_original_expert_in_the_policy(clone_policy):
    tensors = {ATTENTION: torch.ones(2, 4), _down_b(0): torch.zeros(SLICE)}
    tensors.update({name: torch.zeros(SLICE) for name in LAYER0_DOWN_B_CLONES})

    with pytest.raises(sp.StatePluginError, match="applied packed expert LoRA mismatch"):
        sp._apply_state(_Actor(), tensors=tensors, policy_version=1, local_step=0, optimizer="preserve")


def _fc1_name(expert: int, projection: str, side: str) -> str:
    return f"base_model.model.model.layers.0.mlp.experts.{expert}.{projection}.lora_{side}.weight"


def _clone_fc1_tensors(rank: int, side: str, slice_shape: tuple[int, int]) -> dict[str, torch.Tensor]:
    tensors = {}
    for expert in _logical_ids(rank)[ORIGINAL_EXPERTS_PER_RANK:]:
        for projection, offset in (("gate_proj", 0.0), ("up_proj", 0.5)):
            value = torch.full(slice_shape, float(expert) + (offset if side == "B" else 0.0))
            tensors[_fc1_name(expert, projection, side)] = value
    return tensors


def test_fc1_b_slot_concatenates_gate_and_up():
    policy = ClonePackedAdapterPolicy(expert_parallel_rank=3, expert_parallel_size=8)
    tensors = _clone_fc1_tensors(3, "B", (2, 4))

    writes = policy.slot_writes(_fc1_name(0, "gate_proj", "B"), (TRAINING_EXPERTS_PER_RANK, 4, 4), tensors)

    assert writes.frozen == slice(0, ORIGINAL_EXPERTS_PER_RANK)
    slot, value = writes.updates[0]
    expert = _logical_ids(3)[slot]
    assert slot == ORIGINAL_EXPERTS_PER_RANK
    assert torch.equal(value[:2], torch.full((2, 4), float(expert)))
    assert torch.equal(value[2:], torch.full((2, 4), expert + 0.5))


def test_fc1_a_requires_equal_gate_and_up():
    policy = ClonePackedAdapterPolicy(expert_parallel_rank=0, expert_parallel_size=8)
    tensors = _clone_fc1_tensors(0, "A", (2, 4))
    tensors[_fc1_name(_logical_ids(0)[-1], "up_proj", "A")] += 1

    with pytest.raises(RuntimeError, match="gate/up LoRA A tensors differ"):
        policy.slot_writes(_fc1_name(0, "gate_proj", "A"), (TRAINING_EXPERTS_PER_RANK, 2, 4), tensors)


@pytest.mark.parametrize(("rank", "size"), [(0, 4), (8, 8)])
def test_slot_writes_refuse_an_unbalanced_expert_parallel_layout(rank, size):
    policy = ClonePackedAdapterPolicy(expert_parallel_rank=rank, expert_parallel_size=size)

    with pytest.raises(ValueError, match="EP size|expert-parallel"):
        policy.slot_writes(_down_b(0), (TRAINING_EXPERTS_PER_RANK, *SLICE), {})


def test_carried_bytes_matches_the_sparse_clone_specs():
    from tests.test_rl_export import _expanded_deepseek_v4_config
    from yeto.rl import export as rl_export

    rank = 8
    specs = rl_export._deepseek_v4_clone_expert_lora_specs(_expanded_deepseek_v4_config(), rank=rank)
    per_layer = [(_fc1_name(0, "gate_proj", "A").replace("layers.0", f"layers.{layer}"), (36, rank, 4096))
                 for layer in range(43)]
    per_layer += [(_fc1_name(0, "gate_proj", "B").replace("layers.0", f"layers.{layer}"), (36, 4096, rank))
                  for layer in range(43)]
    per_layer += [(_fc1_name(0, "down_proj", "A").replace("layers.0", f"layers.{layer}"), (36, rank, 2048))
                  for layer in range(43)]
    per_layer += [(_fc1_name(0, "down_proj", "B").replace("layers.0", f"layers.{layer}"), (36, 4096, rank))
                  for layer in range(43)]

    assert ClonePackedAdapterPolicy().carried_bytes(per_layer) == 4 * sum(spec.numel for spec in specs)


def test_export_refuses_a_policy_above_the_single_return_limit(monkeypatch, clone_policy, fake_bridge):
    monkeypatch.setattr(sp, "MAX_PACKED_EXPORT_BYTES", 4 * SLICE[0] * SLICE[1] * 31)

    with pytest.raises(sp.StatePluginError, match="chunked transport"):
        sp._export_state(_Actor(), policy_version=1)


def test_distributed_optimizer_is_refused_before_any_write(monkeypatch, clone_policy, fake_bridge):
    actor = _Actor()
    actor.opt_param_scheduler = SimpleNamespace(num_steps=0, step=lambda increment: pytest.fail("scheduler moved"))
    monkeypatch.setattr(sp, "distributed_ranges", lambda _optimizer: (True, {}))
    tensors = {ATTENTION: torch.ones(2, 4), **{name: torch.ones(SLICE) for name in LAYER0_DOWN_B_CLONES}}

    with pytest.raises(sp.StatePluginError, match="distributed optimizer"):
        sp._apply_state(actor, tensors=tensors, policy_version=1, local_step=1, optimizer="preserve")
    with pytest.raises(sp.StatePluginError, match="distributed optimizer"):
        sp._export_state(actor, policy_version=1)
    assert torch.equal(actor.module.attention.detach(), torch.full((2, 4), 0.5))


def test_policy_registration_keeps_the_first_policy_and_rejects_another_type(monkeypatch):
    monkeypatch.setattr(sp, "_PACKED_POLICY", None)
    first = ClonePackedAdapterPolicy()
    sp.register_packed_adapter_policy(first)
    sp.register_packed_adapter_policy(ClonePackedAdapterPolicy())  # same type keeps the first
    assert sp.packed_adapter_policy() is first

    with pytest.raises(sp.StatePluginError, match="already registered"):
        sp.register_packed_adapter_policy(SimpleNamespace())


def test_install_clone_only_registers_the_policy_and_wraps_the_lora_factory(monkeypatch):
    monkeypatch.setattr(sp, "_PACKED_POLICY", None)
    armed = []
    monkeypatch.setattr(ports_runtime, "install_or_defer", lambda name, installer: armed.append((name, installer)))

    ports_runtime.install_clone_only()

    assert isinstance(sp.packed_adapter_policy(), ClonePackedAdapterPolicy)
    assert armed == [("miles.backends.megatron_utils.lora.bridge",
                      ports_runtime.install_clone_only_on_lora_bridge)]

    class LoraBridge:
        @staticmethod
        def create_lora_instance(args, *, target_modules):
            return SimpleNamespace(marker=args.marker)

    ports_runtime.install_clone_only_on_lora_bridge(LoraBridge)
    ports_runtime.install_clone_only_on_lora_bridge(LoraBridge)  # idempotent: one wrapper
    lora = LoraBridge.create_lora_instance(SimpleNamespace(marker="m"), target_modules=["q"])

    assert isinstance(lora, clone_lora._CloneOnlyLoraProxy)
    assert lora.marker == "m"


_DSV4_FLAGS = ("YETO_DSV4_EXPERT_CLONE", "YETO_DSV4_CLONE_ONLY_LORA", "YETO_DSV4_CLONE_ONLY_LORA_PORTS",
               "YETO_DSV4_EXPERT_FULL", "YETO_DSV4_EXPERT_FULL_PORTS", "YETO_TMS_POST_PAUSE_IDLE_S")


@pytest.mark.parametrize(("flag", "expected"), [
    ("YETO_DSV4_CLONE_ONLY_LORA_PORTS", "ports"),
    ("YETO_DSV4_CLONE_ONLY_LORA", "legacy"),
])
def test_sitecustomize_picks_the_clone_only_runtime_for_the_miles_layout(monkeypatch, flag, expected):
    calls = []
    monkeypatch.setattr(ports_runtime, "install_clone_only", lambda: calls.append("ports"))
    monkeypatch.setattr(clone_lora, "install", lambda: calls.append("legacy"))
    for name in _DSV4_FLAGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(flag, "1")

    runpy.run_path(str(ROOT / "sitecustomize.py"), run_name="_sitecustomize_clone_only_test")

    assert calls == [expected]


def test_sitecustomize_refuses_both_clone_only_runtimes(monkeypatch):
    for name in _DSV4_FLAGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("YETO_DSV4_CLONE_ONLY_LORA", "1")
    monkeypatch.setenv("YETO_DSV4_CLONE_ONLY_LORA_PORTS", "1")

    with pytest.raises(RuntimeError, match="mutually exclusive"):
        runpy.run_path(str(ROOT / "sitecustomize.py"), run_name="_sitecustomize_both_clone_test")


@pytest.mark.parametrize(("engine", "runtime", "other"), [
    ("ports", "YETO_DSV4_CLONE_ONLY_LORA_PORTS", "YETO_DSV4_CLONE_ONLY_LORA"),
    ("legacy", "YETO_DSV4_CLONE_ONLY_LORA", "YETO_DSV4_CLONE_ONLY_LORA_PORTS"),
])
def test_island_entry_switches_on_the_clone_only_runtime_of_its_engine(monkeypatch, engine, runtime, other):
    from yeto.rl.adapters.miles import island_entry

    for name in (*_DSV4_FLAGS, "SGLANG_OPT_FUSE_WQA_WKV"):
        # setenv first so monkeypatch also removes what the island entry sets.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)

    island_entry._configure_deepseek_v4_env(SimpleNamespace(), rl_engine=engine, clone_only_lora=True,
                                            expert_full=False)

    assert os.environ[runtime] == "1" and other not in os.environ
    assert os.environ["YETO_DSV4_EXPERT_CLONE"] == "1"
    monkeypatch.setenv(other, "1")
    with pytest.raises(ValueError, match="other Miles layout"):
        island_entry._configure_deepseek_v4_env(SimpleNamespace(), rl_engine=engine, clone_only_lora=True,
                                                expert_full=False)
