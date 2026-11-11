"""yeto's extension context on Miles' configuration (B0 config owner).

``ext_args`` (carrier + reader), ``island_args`` (driver facade),
``proc_state`` (process-local runtime state) and the adapter's split of the
runtime attributes. CPU only; Miles is faked where a Miles object is needed.
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field

from yeto.rl.adapters.miles import config as mc
from yeto.rl.adapters.miles import ext_args as xa
from yeto.rl.adapters.miles import island_args as ia
from yeto.rl.adapters.miles import proc_state as ps
from yeto.rl.engine import proc_state as eps

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "golden" / "decoupling"


# ------------------------------------------------------------------ fakes
class _Leaf(BaseModel):
    """A Miles leaf config: frozen, extra forbidden, carries ``yeto_ext``."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    run_uuid: str | None = None
    num_rollout: int = 10
    yeto_ext: dict[str, Any] = Field(default_factory=dict)


class _LeafWithoutTrait(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    num_rollout: int = 10


class _View(BaseModel):
    """Miles' hook view: frozen, extra allowed (what a rollout hook receives)."""

    model_config = ConfigDict(frozen=True, extra="allow")


# ------------------------------------------------------------------ ext_args
def test_ext_attr_reads_the_new_carrier_first_then_the_legacy_attribute_then_default():
    both = SimpleNamespace(yeto_ext={"rl_learner_id": 3}, yeto_rl_learner_id=99, yeto_rl_max_policy_age=2)
    assert xa.ext_attr(both, "rl_learner_id") == 3
    assert xa.ext_attr(both, "yeto_rl_learner_id") == 3  # prefixed spelling accepted
    assert xa.ext_attr(both, "rl_max_policy_age") == 2  # not in the carrier: legacy attribute
    assert xa.ext_attr(both, "rl_observe_timeline", False) is False
    with pytest.raises(AttributeError):
        xa.ext_attr(both, "rl_observe_timeline")
    assert xa.ext_has(both, "rl_learner_id") and xa.ext_has(both, "rl_max_policy_age")
    assert not xa.ext_has(both, "rl_observe_timeline")


def test_legacy_namespaces_keep_working_without_a_carrier():
    assert xa.ext_attr(SimpleNamespace(yeto_rl_learner_id=7), "rl_learner_id") == 7
    assert xa.ext_attr(argparse.Namespace(), "rl_max_policy_age", 0) == 0
    assert xa.ext_attr(argparse.Namespace(yeto_algo_plugins={"a": 1}), "algo_plugins") == {"a": 1}


def test_ext_round_trip_through_a_frozen_view_and_json():
    leaf = _Leaf(run_uuid="0" * 32)
    installed = xa.install_ext(leaf, {"yeto_rl_learner_id": 3, "rl_event_tape": "/t.jsonl",
                                      "rl_bridge_config": object()})  # driver-only: dropped
    assert installed is not leaf and leaf.yeto_ext == {}  # the source is untouched
    assert installed.yeto_ext == {"rl_event_tape": "/t.jsonl", "rl_learner_id": 3}
    wire = json.loads(json.dumps(installed.yeto_ext))
    view = _View(**installed.model_dump() | {"yeto_ext": wire})
    assert xa.ext_attr(view, "rl_learner_id") == 3
    assert xa.ext_attr(view, "rl_event_tape") == "/t.jsonl"
    assert xa.ext_attr(view, "rl_max_policy_age", 0) == 0
    with pytest.raises(Exception):  # frozen view: no setattr, as on Miles
        view.yeto_rl_learner_id = 4  # type: ignore[misc]
    # a second install merges over the first
    again = xa.install_ext(installed, {"rl_max_policy_age": 2})
    assert again.yeto_ext == {"rl_event_tape": "/t.jsonl", "rl_learner_id": 3, "rl_max_policy_age": 2}


def test_install_ext_on_a_mutable_namespace_is_in_place():
    ns = argparse.Namespace()
    assert xa.install_ext(ns, {"rl_learner_id": 1}) is ns
    assert ns.yeto_ext == {"rl_learner_id": 1}
    xa.install_ext(ns, {"rl_event_tape": "x"})
    assert ns.yeto_ext == {"rl_event_tape": "x", "rl_learner_id": 1}


def test_install_ext_fails_closed():
    with pytest.raises(TypeError, match="JSON"):
        xa.install_ext(argparse.Namespace(), {"rl_max_policy_age": object()})
    with pytest.raises(TypeError, match="YetoExtensionConfig"):
        xa.install_ext(_LeafWithoutTrait(), {"rl_learner_id": 1})


def test_wire_keys_are_unprefixed_and_iterated_sorted():
    keys = list(xa.iter_ext_keys())
    assert keys == sorted(xa.WIRE_KEYS) and len(keys) == 26
    assert all(not k.startswith("yeto_") for k in keys)
    assert xa.ext_key("yeto_rl_x") == "rl_x" and xa.attr_name("rl_x") == "yeto_rl_x"
    assert xa.attr_name("yeto_rl_x") == "yeto_rl_x" and xa.ext_key("rl_x") == "rl_x"


# ------------------------------------------------------------------ IslandArgs
def test_facade_mirrors_every_write_onto_a_mutable_namespace():
    ns = argparse.Namespace(num_rollout=4)
    island = ia.IslandArgs(ns)
    island.yeto_rl_learner_id = 3  # the island_entry assignment syntax
    island.yeto_rl_bridge_config = object()  # driver-only
    island.wandb = True
    setattr(island, "yeto_rl_grad_audit_dir", "/a")
    assert ns.yeto_rl_learner_id == 3 and ns.wandb is True and ns.yeto_rl_grad_audit_dir == "/a"
    assert island.yeto_rl_learner_id == 3 and island.num_rollout == 4
    assert getattr(island, "yeto_rl_missing", "d") == "d"
    with pytest.raises(AttributeError, match="belongs to Miles"):
        island.num_rollout = 5
    frozen = island.freeze_for_slicing()
    assert frozen is ns and ns.yeto_ext == {"rl_grad_audit_dir": "/a", "rl_learner_id": 3}
    assert "rl_bridge_config" not in ns.yeto_ext
    assert island.handoff() is ns  # pinned Miles: the namespace itself
    assert xa.ext_attr(ns, "rl_learner_id") == 3


def test_facade_over_a_frozen_config_keeps_yeto_names_in_its_context():
    leaf = _Leaf(run_uuid="1" * 32)
    island = ia.IslandArgs(leaf)
    island.yeto_rl_learner_id = 3
    island.yeto_rl_bridge_config = object()
    assert island.yeto_rl_learner_id == 3 and island.num_rollout == 10
    assert leaf.yeto_ext == {}  # nothing on the config until freezing
    with pytest.raises(AttributeError):
        island.yeto_rl_missing  # noqa: B018
    island.override_before_slice(num_rollout=5)
    assert island.num_rollout == 5 and island.miles.num_rollout == 5
    frozen = island.freeze_for_slicing()
    assert frozen is island.miles and frozen.yeto_ext == {"rl_learner_id": 3}
    assert island.handoff() is island  # the driver keeps the facade
    assert ia.miles_config(island) is frozen and ia.miles_config(leaf) is leaf
    with pytest.raises(AttributeError, match="frozen"):
        island.yeto_rl_learner_id = 4
    island.yeto_rl_learner_id = 3  # an identical re-set is a no-op
    island.yeto_rl_cell_id = "1"  # driver-only keys stay writable
    with pytest.raises(RuntimeError):
        island.override_before_slice(num_rollout=6)
    assert xa.ext_attr(island, "rl_learner_id") == 3  # the facade itself is readable


def test_facade_reads_megatron_base_args_and_refuses_non_json_wire_values():
    leaf = SimpleNamespace(raw_megatron=SimpleNamespace(base_args={"use_distributed_optimizer": False}))
    island = ia.IslandArgs(leaf)
    assert island.use_distributed_optimizer is False
    with pytest.raises(AttributeError):
        island.no_such_field  # noqa: B018
    pyd = ia.IslandArgs(_Leaf())
    pyd.yeto_rl_max_policy_age = object()
    with pytest.raises(TypeError):
        pyd.freeze_for_slicing()


def test_override_native_and_as_island_dispatch():
    ns = SimpleNamespace()
    ia.override_native(ns, save_debug_rollout_data="x")
    assert ns.save_debug_rollout_data == "x"
    island = ia.as_island(_Leaf())
    assert ia.as_island(island) is island
    ia.override_native(island, num_rollout=7)
    assert island.miles.num_rollout == 7


# ------------------------------------------------------------------ key inventory
_WRITERS = (
    "yeto/rl/adapters/miles/island_entry.py",
    "yeto/rl/adapters/miles/elastic_hook.py",
    "yeto/rl/adapters/miles/full_parameter/sao_binding.py",
    "yeto/rl/adapters/miles/config.py",
)


def _driver_writes() -> set[str]:
    """Every ``<obj>.yeto_* = ...`` / ``setattr(<obj>, "yeto_*", ...)`` in the driver-side writers."""
    names: set[str] = set()
    for rel in _WRITERS:
        tree = ast.parse((REPO / rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    for leaf in (t.elts if isinstance(t, ast.Tuple) else [t]):
                        if (isinstance(leaf, ast.Attribute) and leaf.attr.startswith("yeto_")
                                and isinstance(leaf.value, ast.Name)
                                and leaf.value.id in ("miles_args", "args", "island")):
                            names.add(leaf.attr)
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setattr"
                    and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, str) and node.args[1].value.startswith("yeto_")):
                names.add(node.args[1].value)
            if isinstance(node, ast.Dict) and rel.endswith("/config.py"):  # runtime_attrs literals
                for k in node.keys:
                    if (isinstance(k, ast.Constant) and isinstance(k.value, str)
                            and k.value.startswith(("yeto_rl_", "yeto_rollout_", "yeto_algo_"))):
                        names.add(k.value)
    return names


def test_island_args_registry_covers_every_driver_side_key():
    from yeto.rl.adapters.miles import elastic_hook
    written = {xa.ext_key(n) for n in _driver_writes()}
    written |= {"rl_" + flag.replace("-", "_") for flag, *_ in elastic_hook.TUNING}
    # constants used as names by the writers
    from yeto.rl import grad_audit
    from yeto.rl.adapters.miles.full_parameter.session import CONFIG_ATTRIBUTE
    written |= {xa.ext_key(grad_audit.GRAD_AUDIT_DIR_ATTR), xa.ext_key(CONFIG_ATTRIBUTE)}
    unknown = sorted(written - ia.ALL_KEYS)
    assert unknown == [], f"driver writes without an owner class: {unknown}"
    # and the inventory names nothing twice
    assert not (xa.WIRE_KEYS & ia.DRIVER_KEYS) and not (xa.WIRE_KEYS & ia.DRIVER_MUTABLE_KEYS)
    assert not (ia.DRIVER_KEYS & ia.DRIVER_MUTABLE_KEYS)


def test_unread_attributes_are_no_longer_written():
    gone = {"rl_algorithm_absorbed_flags", "rl_canonical_lora_names", "rl_clone_only_lora",
            "rl_expected_specs", "rl_learner_generation", "rl_learner_generations"}
    assert not gone & {xa.ext_key(n) for n in _driver_writes()}
    assert not gone & ia.ALL_KEYS


_READERS = (
    "yeto/rl/adapters/miles/rollout_meta_hook.py", "yeto/rl/adapters/miles/carry_over.py",
    "yeto/rl/adapters/miles/full_parameter/evidence.py", "yeto/rl/adapters/miles/full_parameter/session.py",
    "yeto/rl/adapters/miles/harness_glue/tool_wait.py", "yeto/rl/adapters/miles/state_plugin.py",
    "yeto/rl/adapters/miles/remote_code_prewarm.py", "yeto/rl/adapters/miles/buffer_rollout_miles.py",
    "yeto/rl/adapters/miles/models/deepseek_v4/expert_full_runtime.py",
    "yeto/rl/filters.py", "yeto/rl/grad_audit.py", "yeto/rl/algos/reward_pipeline.py",
    "yeto/rl/algos/sample_filters.py", "yeto/rl/algos/seq_adv.py", "yeto/rl/engine/events.py",
    "yeto/rl/harness/reasoning_loss_mask.py", "yeto/rl/harness/placeholder_logprob_mask.py",
)


def _is_args(node: ast.expr) -> bool:
    """``args`` / ``miles_args`` / ``<x>.args`` (not an actor, optimizer or iterator)."""
    if isinstance(node, ast.Name):
        return node.id in ("args", "miles_args")
    return isinstance(node, ast.Attribute) and node.attr == "args"


def test_miles_process_readers_use_ext_attr_and_proc_state_only():
    """No bare getattr/setattr/hasattr of a yeto name on args in the Miles-process readers,
    and every literal ``ext_attr`` key is a wire key."""
    bare: list[str] = []
    literal_keys: set[str] = set()
    for rel in _READERS:
        tree = ast.parse((REPO / rel).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id in ("getattr", "setattr", "hasattr") and len(node.args) >= 2 and _is_args(node.args[0]):
                key = node.args[1]
                if isinstance(key, ast.Constant) and isinstance(key.value, str) and (
                        key.value.startswith(("yeto_", "_yeto_"))):
                    bare.append(f"{rel}:{node.lineno} {node.func.id}({key.value})")
            if node.func.id in ("ext_attr", "ext_has") and len(node.args) >= 2:
                key = node.args[1]
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    literal_keys.add(xa.ext_key(key.value))
    assert bare == []
    # driver-only values never appear in a Miles-process reader, except through the
    # legacy fallback of names the driver also reads (sync preset in W&B telemetry)
    assert literal_keys - xa.WIRE_KEYS <= {"rl_sync_preset"}


# ------------------------------------------------------------------ proc_state
def test_proc_state_is_isolated_per_args_object_and_shared_per_run_uuid():
    a, b = SimpleNamespace(), SimpleNamespace()
    counts = ps.proc_state(a, "counts", dict)
    counts["x"] = 1
    assert ps.proc_state(a, "counts") == {"x": 1} and ps.proc_state(b, "counts") is None
    ps.set_proc_state(b, "counts", {"y": 2})
    assert ps.proc_state(a, "counts") == {"x": 1}
    assert ps.pop_proc_state(a, "counts") == {"x": 1} and ps.proc_state(a, "counts") is None
    assert ps.has_proc_state(b, "counts") and not ps.has_proc_state(a, "counts")
    # the same run id (two hook views of one Miles run) shares the state
    v1, v2 = _View(run_uuid="a" * 32), _View(run_uuid="a" * 32)
    other = _View(run_uuid="b" * 32)
    ps.set_proc_state(v1, "trained", {1, 2})
    assert ps.proc_state(v2, "trained") == {1, 2} and ps.proc_state(other, "trained") is None
    ps.pop_proc_state(v2, "trained")
    assert ps.proc_state(v1, "trained") is None


class _Frozen:
    """Frozen, weak-referenceable, no run id: identity-scoped state."""

    __slots__ = ("__weakref__",)


def test_proc_state_forgets_a_collected_identity_scoped_object():
    obj = _Frozen()
    ps.set_proc_state(obj, "k", 1)
    ident = id(obj)
    assert ident in eps._BY_ID and ps.proc_state(obj, "k") == 1
    del obj
    gc.collect()
    assert ident not in eps._BY_ID


def test_proc_state_lives_in_a_mutable_namespace_itself():
    ns = SimpleNamespace()
    ps.set_proc_state(ns, "k", 1)
    assert vars(ns)[ps.STORE_ATTR] == {"k": 1}  # dies with the namespace: no cross-test leak
    assert ps.pop_proc_state(ns, "k") == 1 and not ps.has_proc_state(ns, "k")


def _fake_miles_channel(monkeypatch, store):
    """A ``miles.rollout.rollout_runtime_stats`` keyed like Miles' (StatsKind)."""
    monkeypatch.setitem(sys.modules, "miles", types.ModuleType("miles"))
    monkeypatch.setitem(sys.modules, "miles.rollout", types.ModuleType("miles.rollout"))
    registry = types.ModuleType("miles.rollout.rollout_runtime_stats")
    registry.take = lambda kind: store.pop(kind, None)  # type: ignore[attr-defined]
    sys.modules["miles.rollout"].rollout_runtime_stats = registry  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "miles.rollout.rollout_runtime_stats", registry)


def test_take_runtime_stats_reads_the_miles_kinds_and_never_touches_args(monkeypatch):
    """k31 (F1/B5): Miles publishes under abort_discard / over_sampling / suspend /
    resume. The channel's answer is final, None included: a stale legacy attribute
    is neither returned nor reset."""
    store = {"over_sampling": {"b": 2}, "abort_discard": {"groups": 1}, "suspend": {"s": 3}, "resume": {"r": 4}}
    _fake_miles_channel(monkeypatch, store)
    args = SimpleNamespace(rollout_over_sampling_stats={"stale": 1})
    assert ps.take_runtime_stats(args, "over_sampling") == {"b": 2}
    assert ps.take_runtime_stats(args, "over_sampling") is None  # popped once; no legacy fallback
    assert args.rollout_over_sampling_stats == {"stale": 1}  # never read nor reset
    # a legacy attribute name still names its kind
    assert ps.take_runtime_stats(args, "rollout_abort_discard_stats") == {"groups": 1}
    frozen = _View()
    assert ps.take_runtime_stats(frozen, "suspend") == {"s": 3}
    assert ps.take_runtime_stats(frozen, "resume") == {"r": 4}
    assert store == {}
    with pytest.raises(ValueError, match="unknown runtime statistics kind"):
        ps.take_runtime_stats(args, "rollout_bogus_stats")


def test_take_runtime_stats_legacy_attribute_without_the_channel(monkeypatch):
    """ports-v1 fork (no rollout_runtime_stats module): the args attribute, consumed once."""
    monkeypatch.setitem(sys.modules, "miles", types.ModuleType("miles"))
    monkeypatch.setitem(sys.modules, "miles.rollout", types.ModuleType("miles.rollout"))
    monkeypatch.delitem(sys.modules, "miles.rollout.rollout_runtime_stats", raising=False)  # not a package: import raises
    args = SimpleNamespace(rollout_over_sampling_stats={"a": 1})
    assert ps.take_runtime_stats(args, "over_sampling") == {"a": 1}
    assert args.rollout_over_sampling_stats is None  # consumed once
    assert ps.take_runtime_stats(args, "over_sampling") is None
    frozen = _View(rollout_suspend_stats={"s": 1})
    assert ps.take_runtime_stats(frozen, "suspend") == {"s": 1}  # no reset possible, no error


def test_take_runtime_stats_without_miles_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "miles", None)  # import raises
    args = SimpleNamespace(rollout_abort_discard_stats={"d": 3})
    assert ps.take_runtime_stats(args, "abort_discard") == {"d": 3}
    assert args.rollout_abort_discard_stats is None


# ------------------------------------------------------------------ config.py
def _launch(attrs: dict[str, Any]) -> Any:
    return SimpleNamespace(runtime_attrs=attrs, argv=("x",), rollout_overlap=False)


def test_split_runtime_attrs_keeps_miles_and_mis_names_for_preprocess():
    attrs = {"use_distributed_optimizer": False, "tis_mode": "clip", "use_rs": False,
             "yeto_rl_dynamic_sampling_max_replacements": 3, "yeto_algo_plugins": {"a": 1},
             "yeto_rollout_quant_tier": 2}
    miles_part, yeto_part = mc.split_runtime_attrs(_launch(attrs))
    assert miles_part == {"use_distributed_optimizer": False, "tis_mode": "clip", "use_rs": False}
    assert yeto_part == {"yeto_rl_dynamic_sampling_max_replacements": 3, "yeto_algo_plugins": {"a": 1},
                         "yeto_rollout_quant_tier": 2}
    ns = mc.apply_runtime_attrs(SimpleNamespace(), _launch(attrs))  # legacy helper: everything
    assert vars(ns) == attrs


@pytest.mark.parametrize("name", sorted(p.name for p in GOLDEN.glob("*.json") if p.name != "fake_engine_tapes.json"))
def test_golden_runtime_attrs_split_into_known_owners(name):
    attrs = json.loads((GOLDEN / name).read_text())["runtime_attrs"]
    miles_part, yeto_part = mc.split_runtime_attrs(_launch(attrs))
    assert set(miles_part) <= {"use_distributed_optimizer", *mc.MIS_HOOK_FIELDS}
    assert {xa.ext_key(k) for k in yeto_part} <= ia.ALL_KEYS
    assert dict(miles_part, **yeto_part) == attrs  # nothing dropped: the golden table is intact


def test_parse_miles_args_returns_a_facade_with_the_yeto_attrs(monkeypatch):
    seen: dict[str, Any] = {}

    def fake_parse_args(preprocess_args):
        ns = argparse.Namespace(fully_async=False, use_session_server=False, partial_rollout=False,
                                actor_num_nodes=1, actor_num_gpus_per_node=1, rollout_num_gpus=1,
                                rollout_num_gpus_per_engine=1, colocate=False)
        preprocess_args(ns)
        seen.update(vars(ns))
        return ns

    monkeypatch.setitem(sys.modules, "miles", types.ModuleType("miles"))
    monkeypatch.setitem(sys.modules, "miles.utils", types.ModuleType("miles.utils"))
    arguments = types.ModuleType("miles.utils.arguments")
    arguments.parse_args = fake_parse_args  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "miles.utils.arguments", arguments)
    monkeypatch.setattr(mc, "validate_parsed_args", lambda args, launch: None)
    launch = _launch({"use_distributed_optimizer": False, "tis_mode": "clip",
                      "yeto_rl_dynamic_sampling_max_replacements": 3})
    island = mc.parse_miles_args(launch)
    assert isinstance(island, ia.IslandArgs)
    assert seen["use_distributed_optimizer"] is False and seen["tis_mode"] == "clip"
    assert "yeto_rl_dynamic_sampling_max_replacements" not in seen  # never on the raw namespace
    assert island.yeto_rl_dynamic_sampling_max_replacements == 3
    assert island.miles.yeto_rl_dynamic_sampling_max_replacements == 3  # mirrored on the pinned Miles
    assert xa.ext_attr(island.miles, "rl_dynamic_sampling_max_replacements") == 3


def test_mis_hook_config_class_only_with_the_miles_schema(monkeypatch):
    monkeypatch.setitem(sys.modules, "miles", None)
    assert mc.mis_hook_config_class() is None

    class BaseConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")

    monkeypatch.setitem(sys.modules, "miles", types.ModuleType("miles"))
    monkeypatch.setitem(sys.modules, "miles.utils", types.ModuleType("miles.utils"))
    monkeypatch.setitem(sys.modules, "miles.utils.args", types.ModuleType("miles.utils.args"))
    schema = types.ModuleType("miles.utils.args.schema")
    schema.BaseConfig = BaseConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "miles.utils.args.schema", schema)
    cls = mc.mis_hook_config_class()
    assert cls is not None and set(cls.model_fields) == set(mc.MIS_HOOK_FIELDS)
    cfg = cls(tis_mode="clip", tis_upper_bound=3.0)
    assert cfg.tis_mode == "clip" and cfg.use_rs is None
    with pytest.raises(Exception):
        cls(unknown=1)
    from yeto.rl.algos.mismatch_correction import mis_config
    import yeto.rl.engine.algorithm as alg
    assert set(mis_config(alg.AlgorithmSpec())) <= set(mc.MIS_HOOK_FIELDS)
    torch = pytest.importorskip("torch")  # the vendored hook imports it
    del torch
    from yeto.rl.algos.vendor import miles_mis
    monkeypatch.setattr(miles_mis.compute_mis_weights_with_cp, "config_class", None, raising=False)
    mc.attach_mis_hook_config()
    assert miles_mis.compute_mis_weights_with_cp.config_class is not None
    assert set(miles_mis.compute_mis_weights_with_cp.config_class.model_fields) == set(mc.MIS_HOOK_FIELDS)
