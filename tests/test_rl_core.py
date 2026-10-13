from __future__ import annotations

from dataclasses import replace
import gc
import json
import threading
import weakref
from types import SimpleNamespace

import pytest
import torch

import yeto.rl.bridge as bridge_module
import yeto.rl.core as core_module
from yeto.fragments import MERGE_AVG
from yeto.protocol import FinalFragment, FinalManifest, PullRequest
from yeto.rl.bridge import BridgeConfig, StrictRlBridge
from yeto.rl.core import (
    CanonicalTensorSpec,
    LocalRoundStats,
    PolicySnapshot,
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
from strict_bridge_loop import run_round


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
    assert torch.equal(
        core_module.policy_delta(local, base), torch.full_like(flat, 0.5)
    )


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
    spec = CanonicalTensorSpec("base_model.model.x.lora_A.weight", (1, 2), "float32", 2)
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


def _bridge_config(
    tmp_path="/tmp/yeto-rl-test-events.jsonl", *, send_initial_params=True
):
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
        send_initial_params=send_initial_params,
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


def test_terminal_resume_drops_initial_policy_without_sending_it():
    bridge = StrictRlBridge(
        _VersionedRuntime(),
        _bridge_config(send_initial_params=False),
    )
    started = []

    class Client:
        def start(self):
            started.append(True)

        def send_init_parts(self, *_args):
            raise AssertionError("terminal resume must not send INIT_PARAMS")

    bridge.client = Client()
    bridge.start()

    assert started == [True]
    assert bridge.initial is None


def test_policy_delta_rejects_exported_policy_version_drift():
    runtime = _VersionedRuntime()
    bridge = StrictRlBridge(runtime, _bridge_config())
    bridge.current = bridge.initial
    with pytest.raises(ValueError, match="versions differ"):
        run_round(bridge, PullRequest(0, 1, 1))


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

    run_round(bridge, PullRequest(0, 1, 1))

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
    bridge.client = SimpleNamespace(push_fragment_parts=lambda *_args, **_kwargs: False)
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


def test_terminal_wait_installs_and_finalizes_one_cached_f32_policy(tmp_path):
    runtime = _VersionedRuntime()
    bridge = StrictRlBridge(runtime, _bridge_config(tmp_path / "events.jsonl"))
    bridge.current = bridge.initial
    bridge.release_current(0)
    finalizing = threading.Event()
    finalizing.set()
    final_values = {name: value + 2.0 for name, value in bridge.initial.tensors.items()}
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
        wait_for_final_fragments=lambda: waits.append(True) or (manifest, [fragment]),
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


def _round_stats_data_pack(rewards, raw_reward):
    batch = {
        "response_lengths": [1, 2, 3, 4],
        "sample_indices": [0, 1, 2, 3],
        "rewards": rewards,
        "raw_reward": raw_reward,
    }
    return {"data_ref": [SimpleNamespace(inner=batch)]}


_ROUND_TRAIN_STATE = SimpleNamespace(
    train_rollout_kl=0.1, ess_ratio=0.8, pg_clipfrac=0.25, train_seconds=1.5
)


def _training_progress_stats(*, grad_norm, nonzero_advantage_count):
    return replace(
        _VersionedRuntime().run_local_round(),
        completed_trajectories=4,
        train_step=1,
        grad_norm=grad_norm,
        nonzero_advantage_count=nonzero_advantage_count,
    )


def _tape_events(tmp_path):
    path = tmp_path / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


