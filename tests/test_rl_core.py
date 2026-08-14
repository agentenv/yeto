from __future__ import annotations

import asyncio
import gc
import json
import sys
import threading
import types
import weakref
from types import SimpleNamespace

import pytest
import torch

import yeto.rl.bridge as bridge_module
import yeto.rl.core as core_module
from yeto.fragments import MERGE_AVG
from yeto.protocol import BcastFragment, FinalFragment, FinalManifest, PullRequest
from yeto.rl import miles
from yeto.rl.bridge import BridgeConfig, StrictRlBridge
from yeto.rl.core import (
    CanonicalTensorSpec,
    LocalRoundStats,
    PolicySnapshot,
    StrictRlInvariantError,
    bounded_tensor_groups,
    build_avg_layout,
    build_rl_fragment_layout,
    canonical_layout_hash,
    canonical_state,
    canonical_state_from_owned_tensors,
    flat_tensor,
    parse_policy_snapshot_token,
    policy_delta,
    policy_tensor_hash,
    tensors_from_flat,
)
from yeto.rl.export import adapter_targets, derive_peft_lora_specs
from yeto.rl.filters import bounded_nonzero_reward_std
from yeto.rl.miles import MilesPolicySync, _BridgeRuntime


def tensors():
    return {
        "base_model.model.z.lora_B.weight": torch.tensor([[3.0], [4.0]]),
        "base_model.model.a.lora_A.weight": torch.tensor([[1.0, 2.0]]),
    }


MODEL_REVISION = "a" * 40
LORA_CONFIG_HASH = "b" * 64


def state(version, values, **kwargs):
    return canonical_state(
        version,
        values,
        base_model_revision=kwargs.pop("base_model_revision", MODEL_REVISION),
        lora_config_hash=kwargs.pop("lora_config_hash", LORA_CONFIG_HASH),
        **kwargs,
    )


def test_canonical_lora_is_sorted_f32_cpu_and_one_avg_fragment():
    canonical = state(7, tensors())
    assert [spec.name for spec in canonical.specs] == sorted(tensors())
    assert all(value.dtype == torch.float32 for value in canonical.tensors.values())
    assert all(spec.dtype == "float32" for spec in canonical.specs)
    assert canonical.layout_hash == canonical_layout_hash(canonical.specs)
    layout = build_avg_layout(canonical.specs)
    assert len(layout.fragments) == 1
    assert layout.fragments[0].merge_mode == MERGE_AVG


def test_owned_canonical_state_revalidates_without_copying_tensor_storage():
    values = tensors()
    canonical = canonical_state_from_owned_tensors(
        7,
        values,
        base_model_revision=MODEL_REVISION,
        lora_config_hash=LORA_CONFIG_HASH,
    )

    assert canonical._owns_tensor_storage
    assert all(
        canonical.tensors[name].data_ptr() == values[name].data_ptr() for name in values
    )
    copied = state(7, values)
    assert all(
        copied.tensors[name].data_ptr() != values[name].data_ptr() for name in values
    )


def test_flat_round_trip_and_delta_use_the_exact_tensor_contract():
    base = state(2, tensors())
    flat = flat_tensor(base.tensors, base.specs)
    rebuilt = tensors_from_flat(flat, base.specs)
    assert all(torch.equal(base.tensors[name], rebuilt[name]) for name in rebuilt)
    local = state(
        2,
        {name: value + 0.5 for name, value in base.tensors.items()},
        expected_specs=base.specs,
    )
    assert torch.equal(policy_delta(local, base), torch.full_like(flat, 0.5))


def test_flatten_and_delta_preallocate_without_tensor_list_concatenation(monkeypatch):
    base = state(2, tensors())
    local = state(
        2,
        {name: value + 0.5 for name, value in base.tensors.items()},
        expected_specs=base.specs,
    )

    monkeypatch.setattr(
        core_module.torch,
        "cat",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("large policy paths must preallocate")
        ),
    )

    flat = core_module.flat_tensor(base.tensors, base.specs)
    assert torch.equal(flat, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.equal(core_module.policy_delta(local, base), torch.full_like(flat, 0.5))


def test_tensors_from_flat_does_not_copy_the_complete_flat_input(monkeypatch):
    base = state(2, tensors())
    flat = flat_tensor(base.tensors, base.specs)
    original = core_module._canonical_tensor

    def reject_full_flat_copy(tensor, name):
        if name == "flat LoRA policy":
            raise AssertionError("the complete flat policy must not be cloned")
        return original(tensor, name)

    monkeypatch.setattr(core_module, "_canonical_tensor", reject_full_flat_copy)
    rebuilt = core_module.tensors_from_flat(flat, base.specs)

    assert all(torch.equal(base.tensors[name], rebuilt[name]) for name in rebuilt)
    flat.zero_()
    assert all(torch.count_nonzero(value).item() for value in rebuilt.values())


def test_policy_delta_rejects_canonical_identity_mismatch():
    base = state(2, tensors())
    local = state(2, tensors(), base_model_revision="c" * 40)
    with pytest.raises(ValueError, match="identities differ"):
        policy_delta(local, base)
    with pytest.raises(ValueError, match="layout hash changed"):
        state(2, tensors(), layout_hash="d" * 64)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_canonical_lora_rejects_non_finite_values(bad):
    values = tensors()
    values["base_model.model.a.lora_A.weight"][0, 0] = bad
    with pytest.raises(ValueError, match="NaN or Inf"):
        state(0, values)


def test_canonical_lora_rejects_non_peft_names_and_shape_drift():
    with pytest.raises(ValueError, match="canonical PEFT"):
        state(0, {"model.weight": torch.ones(2)})
    canonical = state(0, tensors())
    changed = tensors()
    changed["base_model.model.a.lora_A.weight"] = torch.ones(2, 1)
    with pytest.raises(ValueError, match="names, shapes, or dtypes"):
        state(0, changed, expected_specs=canonical.specs)


def test_avg_layout_rejects_duplicate_names():
    spec = CanonicalTensorSpec(
        "base_model.model.x.lora_A.weight", (1, 2), "float32", 2
    )
    with pytest.raises(ValueError, match="unique"):
        build_avg_layout((spec, spec))


def test_bounded_tensor_groups_preserve_canonical_order_and_cap():
    specs = tuple(state(0, tensors()).specs)
    groups = bounded_tensor_groups(specs, max_bytes=8)

    assert tuple(spec for group in groups for spec in group) == specs
    assert all(sum(spec.numel * 4 for spec in group) <= 8 for group in groups)
    with pytest.raises(ValueError, match="exceeds the stream chunk cap"):
        bounded_tensor_groups(specs, max_bytes=4)


def test_rl_fragment_layout_is_deterministic_all_avg_binpack():
    specs = (
        CanonicalTensorSpec(
            "base_model.model.large.lora_A.weight", (1, 8), "float32", 8
        ),
        CanonicalTensorSpec(
            "base_model.model.medium.lora_A.weight", (1, 5), "float32", 5
        ),
        CanonicalTensorSpec(
            "base_model.model.small.lora_A.weight", (1, 3), "float32", 3
        ),
        CanonicalTensorSpec(
            "base_model.model.tiny.lora_A.weight", (1, 2), "float32", 2
        ),
    )

    first = build_rl_fragment_layout(specs, 2)
    second = build_rl_fragment_layout(tuple(reversed(specs)), 2)

    assert first == second
    assert [fragment.numel for fragment in first.fragments] == [10, 8]
    assert all(fragment.merge_mode == MERGE_AVG for fragment in first.fragments)
    assert first.tensor_names() == [
        "base_model.model.large.lora_A.weight",
        "base_model.model.tiny.lora_A.weight",
        "base_model.model.medium.lora_A.weight",
        "base_model.model.small.lora_A.weight",
    ]


@pytest.mark.parametrize("fragments", [0, 1, 5])
def test_rl_fragment_layout_requires_multiple_nonempty_fragments(fragments):
    specs = tuple(state(0, tensors()).specs)
    with pytest.raises(ValueError, match="fragments"):
        build_rl_fragment_layout(specs, fragments)


def test_policy_snapshot_token_binds_rollout_to_full_tensor_hash():
    first = state(3, tensors())
    same_tensors_new_progress = state(9, tensors())

    snapshot = PolicySnapshot.create(7, first, (1, 6))

    assert snapshot.policy_hash == policy_tensor_hash(same_tensors_new_progress)
    assert snapshot.token == f"yeto:7:{snapshot.policy_hash}"
    assert parse_policy_snapshot_token(snapshot.token) == (
        snapshot.rollout_id,
        snapshot.policy_hash,
    )


@pytest.mark.parametrize(
    "token",
    ["yeto:1", "yeto:-1:" + "a" * 64, "yeto:1:not-a-hash", "other:1:" + "a" * 64],
)
def test_policy_snapshot_token_rejects_values_outside_contract(token):
    with pytest.raises(ValueError, match="policy snapshot token"):
        parse_policy_snapshot_token(token)


def test_two_model_configs_use_the_same_generic_peft_path(tmp_path):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    configs = (
        transformers.LlamaConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        ),
        transformers.OPTConfig(
            vocab_size=32,
            hidden_size=8,
            ffn_dim=16,
            num_hidden_layers=1,
            num_attention_heads=2,
        ),
    )
    layouts = []
    for index, config in enumerate(configs):
        model_dir = tmp_path / str(index)
        config.save_pretrained(model_dir)
        specs = derive_peft_lora_specs(
            str(model_dir),
            None,
            rank=2,
            targets="all-linear",
        )
        layouts.append(build_avg_layout(specs))
        assert specs
    assert all(layout.fragments[0].merge_mode == MERGE_AVG for layout in layouts)


def test_auto_targets_follow_existing_moe_semantics_from_actual_config(tmp_path):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_local_experts=2,
    )
    model_dir = tmp_path / "moe-marked"
    config.save_pretrained(model_dir)

    specs = derive_peft_lora_specs(
        str(model_dir),
        None,
        rank=2,
        targets="auto",
    )

    assert set(adapter_targets(specs)) == {"q_proj", "k_proj", "v_proj", "o_proj"}


class _VersionedRuntime:
    def __init__(self):
        self.state = state(0, tensors())

    def initialize(self):
        return self.state

    def apply_global_policy(self, state):
        self.state = state

    def run_local_round(self, **kwargs):
        return LocalRoundStats(
            island_id=0,
            local_round_id=1,
            base_policy_version=0,
            active_groups=1,
            completed_groups=1,
            cancelled_groups=0,
            completed_trajectories=1,
            action_tokens=1,
            tool_wait_seconds=0.0,
            group_p50_seconds=0.1,
            group_p95_seconds=0.1,
            group_p99_seconds=0.1,
            reward_mean=1.0,
            reward_std=0.0,
            zero_variance_group_ratio=1.0,
            mean_kl=None,
            ess_ratio=None,
            clip_fraction=0.0,
            delta_l2_norm=0.0,
            rollout_seconds=0.1,
            train_seconds=0.1,
        )

    def export_local_policy(self):
        return state(self.state.policy_version + 1, self.state.tensors)

    def record_local_round(self, stats):
        pass

    def shutdown(self):
        pass


def _bridge_config(tmp_path="/tmp/yeto-rl-test-events.jsonl"):
    canonical = state(0, tensors())
    return BridgeConfig(
        syncer_addr=("127.0.0.1", 1),
        learner_id=0,
        global_rounds=1,
        groups_per_round=1,
        samples_per_group=1,
        local_optimizer_steps=1,
        expected_specs=canonical.specs,
        base_model_revision=canonical.base_model_revision,
        lora_config_hash=canonical.lora_config_hash,
        layout_hash=canonical.layout_hash,
        event_tape=str(tmp_path),
        wan_streams=0,
    )


def test_strict_bridge_start_streams_initial_policy_in_canonical_order():
    bridge = StrictRlBridge(_VersionedRuntime(), _bridge_config())
    started = []
    parts = []

    class Client:
        def start(self):
            started.append(True)

        def send_init(self, *_args):
            raise AssertionError("strict startup must not build a whole INIT payload")

        def send_init_parts(self, fragment_id, tensor_parts):
            assert fragment_id == 0
            parts.extend(bytes(part) for part in tensor_parts)
            return True

    bridge.client = Client()
    bridge.start()

    expected = b"".join(
        memoryview(bridge.initial.tensors[spec.name].numpy()).cast("B")
        for spec in bridge.specs
    )
    assert started == [True]
    assert b"".join(parts) == expected


def test_policy_delta_rejects_exported_policy_version_drift():
    runtime = _VersionedRuntime()
    bridge = StrictRlBridge(runtime, _bridge_config())
    bridge.current = bridge.initial
    with pytest.raises(ValueError, match="versions differ"):
        bridge._run_round(PullRequest(0, 1, 1))


def test_local_round_event_contains_every_init_metric(tmp_path):
    runtime = _VersionedRuntime()
    runtime.export_local_policy = lambda: state(
        0,
        {name: value + 1 for name, value in runtime.state.tensors.items()},
    )
    event_tape = tmp_path / "events.jsonl"
    bridge = StrictRlBridge(runtime, _bridge_config(event_tape))
    bridge.current = bridge.initial
    pushed = []
    bridge.client = SimpleNamespace(push_fragment=lambda *values: pushed.append(values))

    bridge._run_round(PullRequest(0, 1, 1))

    event = json.loads(event_tape.read_text())
    expected = {
        "rl/active_groups",
        "rl/completed_groups",
        "rl/cancelled_groups",
        "rl/completed_trajectories",
        "rl/action_tokens",
        "rl/tool_wait_seconds",
        "rl/rollout_seconds",
        "rl/group_p50_seconds",
        "rl/group_p95_seconds",
        "rl/group_p99_seconds",
        "rl/reward_mean",
        "rl/reward_std",
        "rl/zero_variance_group_ratio",
        "rl/global_policy_version",
        "rl/rollout_policy_version",
        "rl/mixed_version_group_count",
        "rl/local_delta_norm",
        "rl/current_vs_rollout_kl",
        "rl/ess_ratio",
        "rl/clip_fraction",
        "sync/bytes_sent",
    }
    assert expected <= event.keys()
    assert event["sync/bytes_sent"] == 48 + len(pushed[0][-1])


def test_bridge_does_not_recopy_owned_canonical_export(tmp_path, monkeypatch):
    runtime = _VersionedRuntime()
    event_tape = tmp_path / "events.jsonl"
    bridge = StrictRlBridge(runtime, _bridge_config(event_tape))
    bridge.current = bridge.initial
    bridge.client = SimpleNamespace(push_fragment=lambda *_args: None)
    values = {name: value + 0.5 for name, value in bridge.initial.tensors.items()}
    local = canonical_state_from_owned_tensors(
        bridge.initial.policy_version,
        values,
        base_model_revision=bridge.initial.base_model_revision,
        lora_config_hash=bridge.initial.lora_config_hash,
        layout_hash=bridge.initial.layout_hash,
        expected_specs=bridge.specs,
    )
    monkeypatch.setattr(
        bridge_module,
        "canonical_state",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("owned canonical export must not be cloned")
        ),
    )

    bridge.submit_local_state(
        PullRequest(0, 1, 1),
        bridge.initial,
        local,
        runtime.run_local_round(),
    )

    assert all(
        local.tensors[name].data_ptr() == values[name].data_ptr() for name in values
    )


class _FakeChunkedExport:
    _yeto_chunked_export = "owner-sharded-v1"

    def __init__(self, base, *, increment=0.5):
        self.policy_version = base.policy_version
        self.expected_names = tuple(spec.name for spec in base.specs)
        self.train_rollout_kl = 0.1
        self.ess_ratio = 0.9
        self.pg_clipfrac = 0.2
        self.train_seconds = 1.0
        self._values = {
            spec.name: base.tensors[spec.name].clone().add_(increment)
            for spec in base.specs
        }
        self._previous = []
        self.peak_group_bytes = 0
        self.discarded = False

    def take_tensors(self, specs):
        gc.collect()
        assert all(reference() is None for reference in self._previous)
        result = {spec.name: self._values.pop(spec.name) for spec in specs}
        self._previous = [weakref.ref(value) for value in result.values()]
        self.peak_group_bytes = max(
            self.peak_group_bytes,
            sum(value.numel() * value.element_size() for value in result.values()),
        )
        return result

    def finish(self):
        assert not self._values

    def discard(self):
        self.discarded = True
        self._values.clear()


def test_strict_chunk_submit_is_byte_exact_and_never_builds_full_delta(
    tmp_path,
    monkeypatch,
):
    runtime = _VersionedRuntime()
    bridge = StrictRlBridge(runtime, _bridge_config(tmp_path / "events.jsonl"))
    base = bridge.initial
    bridge.current = base
    groups = tuple((spec,) for spec in bridge.specs)
    monkeypatch.setattr(bridge, "export_tensor_groups", lambda: groups)
    monkeypatch.setattr(
        bridge_module,
        "policy_delta",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("strict chunk path must not build a full policy delta")
        ),
    )
    monkeypatch.setattr(
        bridge_module,
        "pack_tensor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("strict chunk path must not pack a full payload")
        ),
    )
    parts = []
    base_references = [weakref.ref(value) for value in base.tensors.values()]

    def push_parts(*args, **kwargs):
        streamed_parts = iter(args[-1])
        for group_index, part in enumerate(streamed_parts):
            if group_index == len(groups) - 1:
                # This is the boundary immediately before the last delta part
                # would be framed/enqueued. The complete base must already be
                # absent so syncer completion cannot overlap it with FINAL.
                gc.collect()
                assert all(reference() is None for reference in base_references)
            parts.append(bytes(part))
        with pytest.raises(StopIteration):
            next(streamed_parts)
        callback = kwargs.get("before_last_enqueue")
        if callback is not None:
            callback()
        return True

    bridge.client = SimpleNamespace(push_fragment_parts=push_parts)
    exported = _FakeChunkedExport(base)
    release_barrier = []

    def release_base():
        bridge.release_current(0)
        gc.collect()
        assert bridge.current is None
        assert all(reference() is None for reference in base_references)
        release_barrier.append(True)

    result = bridge.submit_chunked_local_state(
        PullRequest(0, 1, 1),
        base,
        exported,
        runtime.run_local_round(),
        before_last_enqueue=release_base,
    )

    expected = torch.full(
        (sum(spec.numel for spec in bridge.specs),),
        0.5,
        dtype=torch.float32,
    )
    assert b"".join(parts) == memoryview(expected.numpy()).cast("B")
    assert result.delta_l2_norm == pytest.approx(float(expected.norm().item()))
    assert exported.peak_group_bytes == max(spec.numel * 4 for spec in bridge.specs)
    assert not exported.discarded
    assert not base.tensors
    assert release_barrier == [True]
    assert bridge.pushed_step == 1


def test_strict_chunk_submit_discards_unconsumed_refs_on_sender_drop(tmp_path):
    runtime = _VersionedRuntime()
    bridge = StrictRlBridge(runtime, _bridge_config(tmp_path / "events.jsonl"))
    base = bridge.initial
    bridge.current = base
    bridge.client = SimpleNamespace(
        push_fragment_parts=lambda *_args, **_kwargs: False
    )
    exported = _FakeChunkedExport(base)

    with pytest.raises(RuntimeError, match="not completely queued"):
        bridge.submit_chunked_local_state(
            PullRequest(0, 1, 1),
            base,
            exported,
            runtime.run_local_round(),
        )

    assert exported.discarded
    assert bridge.pushed_step is None


def test_initial_policy_storage_is_released_after_authoritative_broadcast(
    tmp_path,
):
    initial = canonical_state_from_owned_tensors(
        0,
        tensors(),
        base_model_revision=MODEL_REVISION,
        lora_config_hash=LORA_CONFIG_HASH,
    )
    reference = weakref.ref(next(iter(initial.tensors.values())))
    config = _bridge_config(tmp_path / "events.jsonl")
    runtime = _BridgeRuntime(initial, SimpleNamespace())
    bridge = StrictRlBridge(runtime, config)
    payload = bridge_module.pack_tensor(
        flat_tensor(bridge.initial.tensors, bridge.specs),
        bridge_module.DTYPE_F32,
    )
    del initial
    bridge.client = SimpleNamespace(
        drain_updates=lambda: [BcastFragment(0, 0, bytearray(payload))],
        drain_pulls=list,
    )

    bridge._drain_messages()
    gc.collect()

    assert runtime.initial is None
    assert bridge.initial is None
    assert reference() is None


def test_committed_base_storage_is_released_before_next_cut_reassembly(tmp_path):
    initial = canonical_state_from_owned_tensors(
        0,
        tensors(),
        base_model_revision=MODEL_REVISION,
        lora_config_hash=LORA_CONFIG_HASH,
    )
    bridge = StrictRlBridge(
        _BridgeRuntime(initial, SimpleNamespace()),
        _bridge_config(tmp_path / "events.jsonl"),
    )
    payload = bytearray(
        bridge_module.pack_tensor(
            flat_tensor(bridge.initial.tensors, bridge.specs),
            bridge_module.DTYPE_F32,
        )
    )
    bridge.client = SimpleNamespace(
        drain_updates=lambda: [BcastFragment(0, 0, payload)],
        drain_pulls=list,
    )
    bridge._drain_messages()
    base_tensor = next(iter(bridge.current.tensors.values()))
    reference = weakref.ref(base_tensor)
    del initial, base_tensor

    bridge.release_current(0)
    gc.collect()

    assert bridge.current is None
    assert bridge.current_version == 0
    assert reference() is None


def test_terminal_wait_installs_and_finalizes_one_cached_f32_policy(tmp_path):
    runtime = _VersionedRuntime()
    bridge = StrictRlBridge(runtime, _bridge_config(tmp_path / "events.jsonl"))
    bridge.current = bridge.initial
    bridge.release_current(0)
    finalizing = threading.Event()
    finalizing.set()
    final_values = {
        name: value + 2.0 for name, value in bridge.initial.tensors.items()
    }
    payload = bytearray(
        bridge_module.pack_tensor(
            flat_tensor(final_values, bridge.specs),
            bridge_module.DTYPE_F32,
        )
    )
    manifest = FinalManifest(1, (1,))
    fragment = FinalFragment(0, 1, payload)
    acknowledged = []
    waits = []
    bridge.client = SimpleNamespace(
        finalizing=finalizing,
        check_health=lambda: None,
        wait_for_final_fragments=lambda: (
            waits.append(True) or (manifest, [fragment])
        ),
        acknowledge_finalization=lambda value: acknowledged.append(value),
    )

    installed = bridge.wait_for_global_policy(1)
    finalized = bridge.finalize()

    assert finalized is installed
    assert bridge.current is installed
    assert bridge._terminal_policy is installed
    assert waits == [True]
    assert acknowledged == [manifest]
    assert all(
        torch.equal(installed.tensors[name], final_values[name])
        for name in final_values
    )


def test_inbound_f32_policy_owns_wire_storage_without_full_clone():
    runtime = _VersionedRuntime()
    bridge = StrictRlBridge(runtime, _bridge_config())
    payload = bytearray(
        bridge_module.pack_tensor(
            flat_tensor(bridge.initial.tensors, bridge.specs),
            bridge_module.DTYPE_F32,
        )
    )

    rebuilt = bridge._state_from_payload(0, payload)
    first = rebuilt.tensors[bridge.specs[0].name].reshape(-1)
    raw = torch.frombuffer(payload, dtype=torch.float32)
    raw[0] = raw[0] + 7

    assert first[0] == raw[0]
    assert rebuilt._owns_tensor_storage


def test_miles_owned_trainable_export_becomes_canonical_without_copy():
    canonical = state(3, tensors())
    exported = SimpleNamespace(
        policy_version=canonical.policy_version,
        layout_hash=canonical.layout_hash,
        tensors=canonical.tensors,
        _yeto_owned_tensors="canonical-v1",
    )
    hook = MilesPolicySync(
        SimpleNamespace(
            yeto_rl_base_model_revision=canonical.base_model_revision,
            yeto_rl_lora_config_hash=canonical.lora_config_hash,
            yeto_rl_layout_hash=canonical.layout_hash,
        )
    )

    normalized = hook._canonical_state(exported)

    assert normalized._owns_tensor_storage
    assert all(
        normalized.tensors[name].data_ptr() == canonical.tensors[name].data_ptr()
        for name in canonical.tensors
    )


def test_miles_rollout_lifecycle_metrics_use_real_task_completion(monkeypatch):
    class Sample:
        def __init__(self, status):
            self.status = SimpleNamespace(value=status)

    class GenerateState:
        def __init__(self):
            self.pendings = set()

        def submit_generate_tasks(self, groups):
            async def finish(group):
                await asyncio.sleep(0)
                return group

            self.pendings.update(asyncio.create_task(finish(group)) for group in groups)

    upstream = types.ModuleType("miles.rollout.sglang_rollout")
    upstream.GenerateState = GenerateState
    rollout = types.ModuleType("miles.rollout")
    rollout.sglang_rollout = upstream
    package = types.ModuleType("miles")
    package.rollout = rollout
    monkeypatch.setitem(sys.modules, "miles", package)
    monkeypatch.setitem(sys.modules, "miles.rollout", rollout)
    monkeypatch.setitem(sys.modules, "miles.rollout.sglang_rollout", upstream)

    def generate(*_args, **_kwargs):
        async def run():
            state = GenerateState()
            state.submit_generate_tasks(
                [[Sample("completed")], [Sample("aborted")]]
            )
            await asyncio.gather(*state.pendings)

        asyncio.run(run())
        return object()

    _, lifecycle = miles._run_rollout_with_metrics(
        generate, SimpleNamespace(), 0, object(), False
    )
    assert lifecycle["active"] == 0
    assert lifecycle["peak_active"] == 2
    assert lifecycle["cancelled"] == 1
    assert len(lifecycle["durations"]) == 2


@pytest.mark.parametrize(
    "versions",
    [
        ["yeto:2"],
        ["yeto:3", "yeto:2"],
        [],
        ["invalid:3"],
        3,
    ],
    ids=["stale", "mixed", "empty", "invalid", "malformed"],
)
def test_generate_rollout_rejects_invalid_policy_versions_before_train(
    tmp_path, monkeypatch, capsys, versions
):
    upstream = types.ModuleType("miles.rollout.sglang_rollout")
    upstream.generate_rollout = object()
    rollout = types.ModuleType("miles.rollout")
    rollout.sglang_rollout = upstream
    package = types.ModuleType("miles")
    package.rollout = rollout
    monkeypatch.setitem(sys.modules, "miles", package)
    monkeypatch.setitem(sys.modules, "miles.rollout", rollout)
    monkeypatch.setitem(sys.modules, "miles.rollout.sglang_rollout", upstream)

    samples = [
        SimpleNamespace(
            status=SimpleNamespace(value="completed"),
            weight_versions=versions,
            index=0,
        ),
        SimpleNamespace(
            status=SimpleNamespace(value="completed"),
            weight_versions=["yeto:3"],
            index=1,
        ),
    ]
    output = SimpleNamespace(samples=[samples], metrics={})
    monkeypatch.setattr(
        miles,
        "_run_rollout_with_metrics",
        lambda *_args, **_kwargs: (
            output,
            {"active": 0, "peak_active": 0, "cancelled": 0, "durations": []},
        ),
    )
    monkeypatch.setattr(miles, "_save_completed_groups", lambda *_args: None)
    event_tape = tmp_path / "events.jsonl"
    args = SimpleNamespace(
        n_samples_per_prompt=2,
        rollout_batch_size=1,
        yeto_rl_completed_groups_path=str(tmp_path / "island.pt"),
        yeto_rl_event_tape=str(event_tape),
        yeto_rl_learner_id=7,
    )

    with pytest.raises(StrictRlInvariantError) as failure:
        miles.generate_rollout(args, 3, SimpleNamespace(buffer=[]))

    assert failure.value.metric == "mixed_version_group_count"
    assert (
        "[yeto-rl-strict-failure] mixed_version_group_count"
        in capsys.readouterr().err
    )
    event = json.loads(event_tape.read_text())
    assert event["event"] == "rl_strict_failure"
    assert event["metric"] == "mixed_version_group_count"
    assert event["island_id"] == 7


def test_generate_rollout_accepts_the_exact_decoupled_snapshot_token(
    tmp_path, monkeypatch
):
    policy_hash = "a" * 64
    token = f"yeto:3:{policy_hash}"
    upstream = types.ModuleType("miles.rollout.sglang_rollout")
    upstream.generate_rollout = object()
    rollout = types.ModuleType("miles.rollout")
    rollout.sglang_rollout = upstream
    package = types.ModuleType("miles")
    package.rollout = rollout
    monkeypatch.setitem(sys.modules, "miles", package)
    monkeypatch.setitem(sys.modules, "miles.rollout", rollout)
    monkeypatch.setitem(sys.modules, "miles.rollout.sglang_rollout", upstream)
    samples = [
        SimpleNamespace(
            status=SimpleNamespace(value="completed"),
            weight_versions=[token],
            index=index,
        )
        for index in range(2)
    ]
    output = SimpleNamespace(samples=[samples], metrics={})
    monkeypatch.setattr(
        miles,
        "_run_rollout_with_metrics",
        lambda *_args, **_kwargs: (
            output,
            {"active": 0, "peak_active": 0, "cancelled": 0, "durations": []},
        ),
    )
    monkeypatch.setattr(miles, "_save_completed_groups", lambda *_args: None)
    args = SimpleNamespace(
        n_samples_per_prompt=2,
        rollout_batch_size=1,
        yeto_rl_sync_preset="decoupled",
        yeto_rl_policy_token=token,
        yeto_rl_completed_groups_path=str(tmp_path / "island.pt"),
        yeto_rl_event_tape=str(tmp_path / "events.jsonl"),
        yeto_rl_learner_id=0,
    )

    result = miles.generate_rollout(args, 3, SimpleNamespace(buffer=[]))

    assert result is output


def test_island_checkpoint_restores_only_complete_same_policy_groups(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "island.pt"
    args = SimpleNamespace(
        actor_num_gpus_per_node=1,
        actor_num_nodes=1,
        advantage_estimator="grpo",
        yeto_rl_model="org/model",
        yeto_rl_data="org/data",
        yeto_rl_base_model_revision=MODEL_REVISION,
        yeto_rl_data_revision="d" * 40,
        expert_model_parallel_size=1,
        yeto_rl_layout_hash="c" * 64,
        lr=1e-4,
        yeto_rl_lora_config_hash=LORA_CONFIG_HASH,
        n_samples_per_prompt=2,
        num_steps_per_rollout=1,
        pipeline_model_parallel_size=1,
        over_sampling_batch_size=2,
        dynamic_sampling_max_replacements=99,
        yeto_rl_dynamic_sampling_max_replacements=8,
        rl_offload_train=False,
        offload_train=True,
        rl_distributed_timeout_minutes=99,
        distributed_timeout_minutes=7,
        yeto_rl_reward_sha256="e" * 64,
        rollout_batch_size=1,
        seq_length=128,
        seed=7,
        rollout_max_response_len=16,
        custom_generate_function_path=None,
        use_session_server=False,
        tito_model="default",
        yeto_rl_completed_groups_path=str(checkpoint),
    )

    def sample(status, version, index=0):
        value = SimpleNamespace(
            status=SimpleNamespace(value=status),
            weight_versions=[f"yeto:{version}"],
            index=index,
        )
        value.to_dict = lambda: {
            "status": status,
            "weight_versions": [f"yeto:{version}"],
            "index": index,
        }
        return value

    class Sample:
        @staticmethod
        def from_dict(value):
            version = int(value["weight_versions"][0].split(":", 1)[1])
            return sample(value["status"], version, value["index"])

    package = types.ModuleType("miles")
    utils = types.ModuleType("miles.utils")
    sample_types = types.ModuleType("miles.utils.types")
    sample_types.Sample = Sample
    package.utils = utils
    utils.types = sample_types
    monkeypatch.setitem(sys.modules, "miles", package)
    monkeypatch.setitem(sys.modules, "miles.utils", utils)
    monkeypatch.setitem(sys.modules, "miles.utils.types", sample_types)

    complete = [sample("completed", 3), sample("truncated", 3)]
    incomplete = [sample("aborted", 3), sample("completed", 3)]

    class DataSource:
        def __init__(self, buffer=None):
            self.buffer = list(buffer or [])

        def get_samples(self, _count):
            return []

        def add_samples(self, groups):
            self.buffer.extend(groups)

    source = DataSource([complete, incomplete])
    miles._save_completed_groups(args, 3, 4, source, {"reward": 1.5})
    assert source.buffer == [complete]
    payload = torch.load(checkpoint, weights_only=True)
    for name, value in {
        "model": "org/model",
        "dataset": "org/data",
        "seq_length": 128,
        "seed": 7,
        "dynamic_sampling_max_replacements": 8,
        "rl_offload_train": True,
        "rl_distributed_timeout_minutes": 7,
    }.items():
        assert payload["config"][name] == value
    assert payload["completed_groups"][0][0]["status"] == "completed"
    assert payload["completed_groups"][0][1]["status"] == "truncated"
    assert "tensors" not in payload and "local_lora" not in payload
    assert not list(tmp_path.glob(".island.pt.tmp-*"))

    restored = DataSource()
    miles._restore_completed_groups(args, 3, restored)
    assert len(restored.buffer) == 1
    args.seed = 8
    incompatible = DataSource()
    miles._restore_completed_groups(args, 3, incompatible)
    assert incompatible.buffer == []
    args.seed = 7
    stale = DataSource()
    miles._restore_completed_groups(args, 4, stale)
    assert stale.buffer == []

    used = [sample("completed", 3, 10), sample("completed", 3, 11)]
    unused = [sample("completed", 3, 20), sample("truncated", 3, 21)]
    incomplete = [sample("aborted", 3, 30), sample("completed", 3, 31)]
    upstream = types.ModuleType("miles.rollout.sglang_rollout")

    def generate(_args, _rollout_id, data_source, evaluation=False):
        assert not evaluation
        miles.queue_completed_groups(
            _args, [used, unused, incomplete], data_source.get_samples
        )
        return SimpleNamespace(samples=[used], metrics={"reward": 2.0})

    upstream.generate_rollout = generate

    class GenerateState:
        def submit_generate_tasks(self, _samples):
            pass

    upstream.GenerateState = GenerateState
    rollout = types.ModuleType("miles.rollout")
    rollout.sglang_rollout = upstream
    package = types.ModuleType("miles")
    package.rollout = rollout
    monkeypatch.setitem(sys.modules, "miles", package)
    monkeypatch.setitem(sys.modules, "miles.rollout", rollout)
    monkeypatch.setitem(sys.modules, "miles.rollout.sglang_rollout", upstream)
    checkpoint.unlink()
    source = DataSource()

    miles.generate_rollout(args, 3, source)

    assert source.buffer == [unused]
    payload = torch.load(checkpoint, weights_only=True)
    assert [sample["index"] for sample in payload["completed_groups"][0]] == [20, 21]
    assert payload["rollout_metrics"].pop("rollout_seconds") >= 0
    assert payload["rollout_metrics"] == {
        "reward": 2.0,
        "active_groups": 0.0,
        "cancelled_groups": 0.0,
        "tool_wait_seconds": 0.0,
        "group_p50_seconds": 0.0,
        "group_p95_seconds": 0.0,
        "group_p99_seconds": 0.0,
        "rl/dynamic_filter/enabled": 0.0,
        "rl/dynamic_filter/generated_groups": 2.0,
        "rl/dynamic_filter/accepted_groups": 2.0,
        "rl/dynamic_filter/dropped_groups": 0.0,
        "rl/dynamic_filter/replacement_attempts": 1.0,
    }


def test_queue_completed_groups_filters_zero_variance_groups_and_records_replacements(
    tmp_path, monkeypatch
):
    def sample(index, reward):
        return SimpleNamespace(
            status=SimpleNamespace(value="completed"),
            weight_versions=["yeto:3"],
            index=index,
            reward=reward,
        )

    kept = [sample(0, -1.0), sample(1, 1.0)]
    dropped = [sample(2, -1.0), sample(3, -1.0)]

    class DataSource:
        def __init__(self):
            self.buffer = []

        def get_samples(self, _count):
            return []

        def add_samples(self, groups):
            self.buffer.extend(groups)

    args = SimpleNamespace(
        yeto_rl_policy_version=3,
        n_samples_per_prompt=2,
        dynamic_sampling_filter_path="test.filter",
    )
    monkeypatch.setattr(
        miles,
        "_dynamic_sampling_filter",
        lambda _args: lambda _args, group: SimpleNamespace(
            keep=len({sample.reward for sample in group}) > 1,
            reason="zero_std" if len({sample.reward for sample in group}) == 1 else None,
        ),
    )
    source = DataSource()

    miles.queue_completed_groups(args, [kept, dropped], source.get_samples)

    assert source.buffer == [kept]
    assert args._yeto_dynamic_sampling_stats == {
        "generated_groups": 2,
        "accepted_groups": 1,
        "dropped_groups": 1,
        "drop_reasons": {"zero_std": 1},
    }


def test_generate_rollout_records_drop_reasons_as_numeric_metrics(monkeypatch):
    samples = [
        SimpleNamespace(
            status=SimpleNamespace(value="completed"),
            weight_versions=["yeto:3"],
            index=index,
            non_generation_time=0.0,
        )
        for index in range(2)
    ]
    output = SimpleNamespace(samples=[samples], metrics={})
    args = SimpleNamespace(
        n_samples_per_prompt=2,
        rollout_batch_size=1,
        dynamic_sampling_filter_path="test.filter",
    )
    source = SimpleNamespace(buffer=[])
    captured = {}

    package = types.ModuleType("miles")
    rollout = types.ModuleType("miles.rollout")
    upstream = types.ModuleType("miles.rollout.sglang_rollout")
    upstream.generate_rollout = lambda *_args, **_kwargs: output
    package.rollout = rollout
    rollout.sglang_rollout = upstream
    monkeypatch.setitem(sys.modules, "miles", package)
    monkeypatch.setitem(sys.modules, "miles.rollout", rollout)
    monkeypatch.setitem(sys.modules, "miles.rollout.sglang_rollout", upstream)
    monkeypatch.setattr(miles, "_restore_completed_groups", lambda *_args: None)

    def run(*_args):
        args._yeto_dynamic_sampling_stats = {
            "generated_groups": 2,
            "accepted_groups": 1,
            "dropped_groups": 1,
            "drop_reasons": {"zero_std_-1.0": 1},
        }
        return output, {
            "active": 0,
            "peak_active": 0,
            "cancelled": 0,
            "durations": [],
        }

    monkeypatch.setattr(miles, "_run_rollout_with_metrics", run)
    monkeypatch.setattr(
        miles,
        "_save_completed_groups",
        lambda _args, _policy, _next, _source, metrics: captured.update(metrics),
    )

    miles.generate_rollout(args, 3, source)

    assert captured["rl/dynamic_filter/drop_reason/zero_std_-1.0"] == 1.0
    assert all(isinstance(value, (int, float)) for value in captured.values())


def test_bounded_variance_filter_forces_progress_after_replacement_budget():
    def sample(index, reward):
        return SimpleNamespace(index=index, reward=reward)

    args = SimpleNamespace(
        yeto_rl_policy_version=0,
        yeto_rl_dynamic_sampling_max_replacements=1,
    )
    first = bounded_nonzero_reward_std(args, [sample(1, 0.0), sample(2, 0.0)])
    second = bounded_nonzero_reward_std(args, [sample(3, 0.0), sample(4, 0.0)])
    mixed = bounded_nonzero_reward_std(args, [sample(5, 0.0), sample(6, 1.0)])
    repeated = bounded_nonzero_reward_std(args, [sample(3, 0.0), sample(4, 0.0)])

    assert (first.keep, second.keep, mixed.keep, repeated.keep) == (
        False,
        True,
        True,
        True,
    )
    assert args._yeto_bounded_filter_state["forced"] == 1


def test_policy_sync_leaves_actor_offload_lifecycle_to_miles():
    calls = []

    class Actor:
        async def onload(self):
            calls.append("onload")

        async def offload(self):
            calls.append("offload")

    class PolicySync(MilesPolicySync):
        async def _initialize(self, *, actor_model, rollout_manager):
            calls.append("initialize")

    sync = PolicySync(SimpleNamespace(offload_train=True))
    asyncio.run(sync.initialize(actor_model=Actor(), rollout_manager=object()))

    assert calls == ["initialize"]


def test_miles_policy_hook_uses_public_trainable_state_api(tmp_path, monkeypatch):
    canonical = state(1, tensors())
    trainable_state = types.ModuleType(
        "miles.backends.megatron_utils.trainable_state"
    )
    trainable_state.make_trainable_state = lambda version, values: SimpleNamespace(
        policy_version=version,
        layout_hash=canonical.layout_hash,
        tensors=values,
    )
    for name in (
        "miles",
        "miles.backends",
        "miles.backends.megatron_utils",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(
        sys.modules,
        "miles.backends.megatron_utils.trainable_state",
        trainable_state,
    )

    versions = []

    class Remote:
        def __init__(self, function):
            self.function = function

        async def remote(self, *args, **kwargs):
            return self.function(*args, **kwargs)

    engine = SimpleNamespace(
        update_weight_version=Remote(lambda version: versions.append(version))
    )
    rollout_manager = SimpleNamespace(
        get_updatable_engines_and_lock=Remote(
            lambda: SimpleNamespace(rollout_engines=[engine])
        )
    )

    class Actor:
        async def apply_trainable_state(self, value, *, reset_optimizer):
            assert reset_optimizer
            self.value = value
            return 2

        async def export_trainable_state(self):
            return self.value

    actor = Actor()
    args = SimpleNamespace(
        yeto_rl_base_model_revision=canonical.base_model_revision,
        yeto_rl_lora_config_hash=canonical.lora_config_hash,
        yeto_rl_layout_hash=canonical.layout_hash,
        yeto_rl_event_tape=str(tmp_path / "events.jsonl"),
        yeto_rl_learner_id=0,
    )
    hook = MilesPolicySync(args)
    hook.actor_model = actor
    hook.rollout_manager = rollout_manager

    asyncio.run(hook._apply_global_policy(canonical))

    assert versions == ["yeto:1"]
    event = json.loads((tmp_path / "events.jsonl").read_text())
    assert event["reset_parameter_count"] == 2
    assert event["sync/global_policy_hash"]


def test_miles_strict_path_releases_both_base_aliases_before_waiting_for_cut():
    base = canonical_state_from_owned_tensors(
        0,
        tensors(),
        base_model_revision=MODEL_REVISION,
        lora_config_hash=LORA_CONFIG_HASH,
    )
    final = state(1, {name: value + 1.0 for name, value in base.tensors.items()})
    exported = _FakeChunkedExport(base)
    base_reference = weakref.ref(next(iter(base.tensors.values())))
    events = []

    class Actor:
        async def export_trainable_state_chunks(self, groups):
            assert groups
            events.append("export")
            return exported

    actor = Actor()

    class Bridge:
        def __init__(self, current):
            self.current = current
            self.specs = current.specs

        def export_tensor_groups(self):
            return tuple((spec,) for spec in self.specs)

        def submit_chunked_local_state(
            self,
            permit,
            submitted_base,
            value,
            stats,
            *,
            before_last_enqueue,
        ):
            assert permit == PullRequest(0, 1, 1)
            assert submitted_base is self.current
            assert value is exported
            assert stats.base_policy_version == 0
            value.discard()
            events.append("submit")
            before_last_enqueue()

        def release_current(self, expected_version):
            assert expected_version == 0
            self.current = None
            events.append("release")

        def wait_for_global_policy(self, expected_version):
            assert expected_version == 1
            gc.collect()
            assert base_reference() is None
            events.append("wait")
            self.current = final
            return final

        def wait_for_round(self):
            raise AssertionError("terminal strict round must not request another permit")

    hook = MilesPolicySync(SimpleNamespace(num_rollout=1))
    hook.actor_model = actor
    hook.bridge = Bridge(base)
    hook.current = base
    hook.permit = PullRequest(0, 1, 1)
    hook._round_stats = lambda *_args: _VersionedRuntime().run_local_round()

    async def apply_global_policy(value):
        assert value is final
        events.append("apply")

    hook._apply_global_policy = apply_global_policy
    del base

    asyncio.run(
        hook._after_local_train(
            rollout_id=0,
            actor_model=actor,
            rollout_data=object(),
        )
    )

    assert hook.current is final
    assert events == ["export", "submit", "release", "wait", "apply"]


def test_miles_policy_hook_builds_round_stats_without_revalidating_versions(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "island.pt"
    args = SimpleNamespace(
        actor_num_gpus_per_node=4,
        actor_num_nodes=1,
        advantage_estimator="grpo",
        yeto_rl_model="org/model",
        yeto_rl_data="org/data",
        yeto_rl_base_model_revision=MODEL_REVISION,
        yeto_rl_data_revision="d" * 40,
        expert_model_parallel_size=1,
        yeto_rl_layout_hash="c" * 64,
        lr=1e-4,
        yeto_rl_lora_config_hash=LORA_CONFIG_HASH,
        n_samples_per_prompt=2,
        num_steps_per_rollout=1,
        pipeline_model_parallel_size=2,
        over_sampling_batch_size=2,
        yeto_rl_reward_sha256="e" * 64,
        rollout_batch_size=2,
        seq_length=128,
        seed=7,
        rollout_max_response_len=16,
        custom_generate_function_path=None,
        use_session_server=False,
        tito_model="default",
        yeto_rl_completed_groups_path=str(checkpoint),
        yeto_rl_learner_id=0,
    )
    assert miles._island_checkpoint_config(args)[
        "pipeline_model_parallel_size"
    ] == 2
    torch.save(
        {
            "schema_version": miles._ISLAND_CHECKPOINT_SCHEMA,
            "config": miles._island_checkpoint_config(args),
            "policy_version": 3,
            "rollout_metrics": {
                "active_groups": 2,
                "cancelled_groups": 0,
                "tool_wait_seconds": 1,
                "group_p50_seconds": 2,
                "group_p95_seconds": 3,
                "group_p99_seconds": 4,
                "rollout_seconds": 5,
            },
        },
        checkpoint,
    )
    batches = [
        {
            "response_lengths": [1, 2],
            "sample_indices": [0, 1],
            "raw_reward": [0.0, 1.0, 2.0, 2.0],
        },
        {
            "response_lengths": [3, 4],
            "sample_indices": [2, 3],
            "raw_reward": [0.0, 1.0, 2.0, 2.0],
        },
    ]
    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(get=lambda reference: reference),
    )
    hook = MilesPolicySync(args)
    data_pack = {
        "data_ref": [SimpleNamespace(inner=batch) for batch in batches]
    }

    train_state = SimpleNamespace(
        train_rollout_kl=0.1,
        ess_ratio=0.8,
        pg_clipfrac=0.25,
        train_seconds=1.5,
    )
    stats = hook._round_stats(3, data_pack, train_state)

    assert stats.action_tokens == 10
    assert stats.reward_mean == 1.25
    assert stats.zero_variance_group_ratio == 0.5
    assert stats.rollout_seconds == 5
    assert stats.mean_kl == 0.1
    assert stats.ess_ratio == 0.8
    assert stats.clip_fraction == 0.25
    assert stats.train_seconds == 1.5


def test_miles_policy_hook_counts_data_parallel_shards_after_model_parallelism(
    monkeypatch,
):
    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(get=lambda reference: reference),
    )
    hook = MilesPolicySync(
        SimpleNamespace(
            actor_num_nodes=2,
            actor_num_gpus_per_node=8,
            tensor_model_parallel_size=8,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=8,
        )
    )
    shards = [
        SimpleNamespace(inner={"sample_indices": [0]}),
        SimpleNamespace(inner={"sample_indices": [1]}),
    ]

    assert hook._rollout_batches({"data_ref": shards}) == [
        {"sample_indices": [0]},
        {"sample_indices": [1]},
    ]
    with pytest.raises(RuntimeError, match="1 DP shards, expected 2"):
        hook._rollout_batches({"data_ref": shards[:1]})
