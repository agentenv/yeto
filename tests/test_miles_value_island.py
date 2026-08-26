from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch
import yeto.megatron.miles_value_island as island_module

from yeto.fragments import MERGE_AVG, MERGE_ISO
from yeto.megatron.miles_value_island import (
    MilesValueIsland,
    TensorDescriptor,
    _fragment_tensor_spans,
    _RuntimeTensor,
    _grouped_clip_grad,
    _grouped_grad_norm,
    _install_bf16_static_unscale_compat,
    _pack_flat_fragment,
    _parse_syncer,
    _unpack_flat_fragment,
    _validate_owned_ranges,
    grouped_tensor_fragment_layout,
    one_tensor_fragment_layout,
)
from yeto.protocol import DTYPE_F32
from yeto.megatron.miles_value_state import (
    reconstruct_fp32_from_bf16_remainder,
    split_fp32_to_bf16_remainder,
)


def _descriptor(name: str, shape: tuple[int, ...], merge_mode: int):
    return TensorDescriptor(
        name=name,
        local_shape=shape,
        full_shape=shape,
        tp_sharded=False,
        partition_dim=None,
        partition_stride=1,
        merge_mode=merge_mode,
    )


def test_layout_keeps_every_canonical_tensor_in_its_own_fragment():
    descriptors = [
        _descriptor("module.module.output_layer.weight", (1, 8), MERGE_ISO),
        _descriptor(
            "module.module.decoder.layers.0.input_layernorm.weight", (8,), MERGE_AVG
        ),
        _descriptor(
            "module.module.embedding.word_embeddings.weight", (16, 8), MERGE_AVG
        ),
        TensorDescriptor(
            name="module.module.decoder.layers.0.mlp.linear_fc1.weight",
            local_shape=(8, 8),
            full_shape=(32, 8),
            tp_sharded=True,
            partition_dim=0,
            partition_stride=2,
            merge_mode=MERGE_ISO,
        ),
    ]

    layout = one_tensor_fragment_layout(descriptors)

    assert layout.num_fragments == len(descriptors)
    assert all(len(fragment.tensors) == 1 for fragment in layout.fragments)
    assert layout.tensor_names() == sorted(item.name for item in descriptors)
    by_name = {fragment.tensors[0][0]: fragment for fragment in layout.fragments}
    fc1 = by_name["module.module.decoder.layers.0.mlp.linear_fc1.weight"]
    assert fc1.merge_mode == MERGE_ISO
    assert fc1.shapes == {
        "module.module.decoder.layers.0.mlp.linear_fc1.weight": (32, 8)
    }
    assert (
        by_name["module.module.embedding.word_embeddings.weight"].merge_mode
        == MERGE_AVG
    )
    # The scalar value head is 1xH; Iso is identity because k=min(shape)=1,
    # but keeping it in the canonical layout ensures it is synchronized.
    assert by_name["module.module.output_layer.weight"].merge_mode == MERGE_ISO


def test_layout_rejects_duplicate_names_and_invalid_iso_shapes():
    duplicate = _descriptor("x", (2, 2), MERGE_ISO)
    with pytest.raises(ValueError, match="not unique"):
        one_tensor_fragment_layout([duplicate, duplicate])
    with pytest.raises(ValueError, match="two-dimensional"):
        one_tensor_fragment_layout([_descriptor("x", (8,), MERGE_ISO)])
    with pytest.raises(ValueError, match="unsupported merge mode"):
        one_tensor_fragment_layout([_descriptor("x", (2, 2), 99)])


def test_grouped_layout_is_deterministic_balanced_and_never_splits_tensors():
    descriptors = [
        _descriptor("decoder.layers.1.mlp.weight", (9, 4), MERGE_ISO),
        _descriptor("decoder.layers.0.norm.weight", (4,), MERGE_AVG),
        _descriptor("decoder.layers.0.mlp.weight", (8, 4), MERGE_ISO),
        _descriptor("decoder.layers.2.mlp.weight", (7, 4), MERGE_ISO),
        _descriptor("embedding.word_embeddings.weight", (11, 4), MERGE_AVG),
        _descriptor("output_layer.weight", (1, 4), MERGE_ISO),
    ]

    first = grouped_tensor_fragment_layout(descriptors, 3, "binpack")
    second = grouped_tensor_fragment_layout(list(reversed(descriptors)), 3, "binpack")

    assert [fragment.tensors for fragment in first.fragments] == [
        fragment.tensors for fragment in second.fragments
    ]
    assert first.num_fragments == 3
    assert sorted(first.tensor_names()) == sorted(item.name for item in descriptors)
    assert len(first.tensor_names()) == len(set(first.tensor_names()))
    expected_numels = {item.name: item.numel for item in descriptors}
    for fragment in first.fragments:
        assert fragment.numel == sum(
            expected_numels[name] for name, _ in fragment.tensors
        )
        assert all(
            declared_numel == expected_numels[name]
            for name, declared_numel in fragment.tensors
        )
        assert all(
            next(item for item in descriptors if item.name == name).merge_mode
            == fragment.merge_mode
            for name, _ in fragment.tensors
        )


def test_grouped_layout_uses_requested_scale_and_complete_tensor_spans():
    descriptors = [
        _descriptor(f"decoder.layers.{index}.weight", (2, index + 1), MERGE_ISO)
        for index in range(120)
    ]
    descriptors.extend(
        _descriptor(f"decoder.layers.{index}.norm.weight", (8,), MERGE_AVG)
        for index in range(8)
    )

    layout = grouped_tensor_fragment_layout(descriptors, 96, "binpack")

    assert layout.num_fragments == 96
    by_name = {item.name: item for item in descriptors}
    for fragment in layout.fragments:
        spans = _fragment_tensor_spans(fragment, by_name)
        assert [span.name for span in spans] == [
            name for name, _ in fragment.tensors
        ]
        assert spans[0].start == 0
        assert spans[-1].end == fragment.numel
        assert all(left.end == right.start for left, right in zip(spans, spans[1:]))
        assert all(span.numel == by_name[span.name].numel for span in spans)


def test_fragment_spans_reject_wrong_numel_shape_merge_and_duplicate():
    descriptor = _descriptor("w", (2, 3), MERGE_ISO)
    by_name = {descriptor.name: descriptor}
    valid = island_module.Fragment(
        MERGE_ISO,
        [("w", 6)],
        shapes={"w": (2, 3)},
        identity_shapes={"w": (2, 3)},
    )
    assert _fragment_tensor_spans(valid, by_name)[0].end == 6

    wrong_numel = island_module.Fragment(
        MERGE_ISO,
        [("w", 5)],
        shapes={"w": (2, 3)},
        identity_shapes={"w": (2, 3)},
    )
    with pytest.raises(ValueError, match="declares 5"):
        _fragment_tensor_spans(wrong_numel, by_name)

    wrong_shape = island_module.Fragment(
        MERGE_ISO,
        [("w", 6)],
        shapes={"w": (3, 2)},
        identity_shapes={"w": (2, 3)},
    )
    with pytest.raises(ValueError, match="Iso shape"):
        _fragment_tensor_spans(wrong_shape, by_name)

    wrong_merge = island_module.Fragment(
        MERGE_AVG,
        [("w", 6)],
        identity_shapes={"w": (2, 3)},
    )
    with pytest.raises(ValueError, match="merge mode"):
        _fragment_tensor_spans(wrong_merge, by_name)

    duplicate = island_module.Fragment(
        MERGE_ISO,
        [("w", 6), ("w", 6)],
        shapes={"w": (2, 3)},
        identity_shapes={"w": (2, 3)},
    )
    with pytest.raises(ValueError, match="duplicate tensor"):
        _fragment_tensor_spans(duplicate, by_name)


def _mock_grouped_island(descriptors, *, is_leader=True):
    island = object.__new__(MilesValueIsland)
    island.is_leader = is_leader
    island.layout = grouped_tensor_fragment_layout(
        descriptors, num_fragments=1, pattern="binpack"
    )
    # This helper's tests use one merge mode, so one requested fragment is one
    # actual fragment.
    assert island.layout.num_fragments == 1
    island.descriptors_by_name = {item.name: item for item in descriptors}
    island.tensors_by_name = {
        item.name: SimpleNamespace(descriptor=item) for item in descriptors
    }
    island.fragment_spans = [
        _fragment_tensor_spans(fragment, island.descriptors_by_name)
        for fragment in island.layout.fragments
    ]
    return island


def test_grouped_pack_unpack_and_mocked_gather_use_exact_flat_offsets():
    descriptors = [
        _descriptor("z.weight", (2, 2), MERGE_ISO),
        _descriptor("a.weight", (1, 3), MERGE_ISO),
        _descriptor("m.weight", (2, 1), MERGE_ISO),
    ]
    island = _mock_grouped_island(descriptors)
    calls = []
    values = {
        item.name: torch.arange(item.numel, dtype=torch.float32).view(item.full_shape)
        + 100 * index
        for index, item in enumerate(descriptors, start=1)
    }

    def gather_one(instance, runtime):
        del instance
        calls.append(runtime.descriptor.name)
        return values[runtime.descriptor.name].clone()

    island._gather_one_canonical = MethodType(gather_one, island)
    flat = island._gather_canonical(0)
    fragment = island.layout.fragments[0]
    spans = island.fragment_spans[0]

    assert calls == [span.name for span in spans]
    assert flat is not None and flat.shape == (fragment.numel,)
    for span in spans:
        assert torch.equal(
            flat[span.start : span.end].view(span.full_shape), values[span.name]
        )
    wire = _pack_flat_fragment(fragment, flat, DTYPE_F32)
    decoded = _unpack_flat_fragment(fragment, wire, DTYPE_F32)
    assert decoded.ndim == 1
    assert torch.equal(decoded, flat)
    with pytest.raises(ValueError, match="expected"):
        _pack_flat_fragment(fragment, flat.view(1, -1), DTYPE_F32)


def test_mocked_grouped_gather_uses_same_tensor_order_on_nonleader_rank():
    descriptors = [
        _descriptor("z.weight", (2, 2), MERGE_ISO),
        _descriptor("a.weight", (1, 3), MERGE_ISO),
        _descriptor("m.weight", (2, 1), MERGE_ISO),
    ]
    island = _mock_grouped_island(descriptors, is_leader=False)
    calls = []

    def gather_one(instance, runtime):
        del instance
        calls.append(runtime.descriptor.name)
        return None

    island._gather_one_canonical = MethodType(gather_one, island)

    assert island._gather_canonical(0) is None
    assert calls == [span.name for span in island.fragment_spans[0]]


@pytest.mark.parametrize("is_leader", [True, False])
def test_mocked_grouped_apply_uses_same_tensor_order_on_every_rank(is_leader):
    descriptors = [
        _descriptor("z.weight", (2, 2), MERGE_ISO),
        _descriptor("a.weight", (1, 3), MERGE_ISO),
        _descriptor("m.weight", (2, 1), MERGE_ISO),
    ]
    island = _mock_grouped_island(descriptors, is_leader=is_leader)
    fragment = island.layout.fragments[0]
    flat = torch.arange(fragment.numel, dtype=torch.float32) if is_leader else None
    calls = []

    def apply_one(instance, runtime, leader_tensor, *, merge_alpha):
        del instance
        calls.append(
            (
                runtime.descriptor.name,
                None if leader_tensor is None else leader_tensor.clone(),
                merge_alpha,
            )
        )

    island._apply_one_canonical = MethodType(apply_one, island)
    island._apply_canonical(0, flat, merge_alpha=0.25)

    assert [name for name, _, _ in calls] == [
        span.name for span in island.fragment_spans[0]
    ]
    assert all(alpha == 0.25 for _, _, alpha in calls)
    for (_, tensor, _), span in zip(calls, island.fragment_spans[0]):
        if is_leader:
            assert tensor is not None
            assert tuple(tensor.shape) == span.full_shape
            assert torch.equal(tensor.reshape(-1), flat[span.start : span.end])
        else:
            assert tensor is None


def test_initial_anchor_covers_the_full_flat_grouped_fragment(monkeypatch):
    descriptors = [
        _descriptor("z.weight", (2, 2), MERGE_ISO),
        _descriptor("a.weight", (1, 3), MERGE_ISO),
        _descriptor("m.weight", (2, 1), MERGE_ISO),
    ]
    island = _mock_grouped_island(descriptors)
    fragment = island.layout.fragments[0]
    flat = torch.arange(fragment.numel, dtype=torch.float32)
    version = 7
    island.client = SimpleNamespace(dtype=DTYPE_F32)
    island._leader_payloads = {
        (0, version): _pack_flat_fragment(fragment, flat, DTYPE_F32)
    }
    island.anchors = [None]
    island.initial_ready = False
    island.steps_total = 11
    island.units_total = 101
    island.steps_at_reset = [0]
    island.units_at_reset = [0]
    island.fragment_versions = [0]
    applied = []

    island._wait_initial_plan = MethodType(lambda instance: [(0, version)], island)
    island._leader_value = MethodType(lambda instance, builder: builder(), island)
    island._leader_local = MethodType(lambda instance, builder: builder(), island)
    island._apply_canonical = MethodType(
        lambda instance, fid, payload, *, merge_alpha: applied.append(
            (fid, payload.clone(), merge_alpha)
        ),
        island,
    )
    island._mark_fresh_optimizer_states_initialized = MethodType(
        lambda instance: None, island
    )
    monkeypatch.setattr(island_module.dist, "barrier", lambda: None)

    island.ensure_initial_ready()

    assert island.initial_ready is True
    assert island.anchors[0] is not None
    assert island.anchors[0].shape == (fragment.numel,)
    assert torch.equal(island.anchors[0], flat.to(torch.bfloat16))
    assert len(applied) == 1
    assert applied[0][0] == 0
    assert torch.equal(applied[0][1], flat)
    assert applied[0][2] == 0.0
    assert island.steps_at_reset == [11]
    assert island.units_at_reset == [101]
    assert island.fragment_versions == [version]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("syncer.internal:29400", ("syncer.internal", 29400)),
        ("127.0.0.1:1", ("127.0.0.1", 1)),
    ],
)
def test_parse_syncer(value, expected):
    assert _parse_syncer(value) == expected


@pytest.mark.parametrize(
    "value", ["missing-port", ":29400", "host:0", "host:65536", "host:nope"]
)
def test_parse_syncer_rejects_bad_addresses(value):
    with pytest.raises(ValueError):
        _parse_syncer(value)


def test_mixed_dtype_grad_norm_uses_fixed_collective_order_and_combines_groups():
    calls = []

    def fake_get_grad_norm(grads, norm_type, grad_stats_parallel_group):
        calls.append((tuple(grad.dtype for grad in grads), grad_stats_parallel_group))
        assert len({grad.dtype for grad in grads}) <= 1
        return sum(float(grad.float().square().sum()) for grad in grads) ** 0.5

    bf16_grad = torch.tensor([3.0, 4.0], dtype=torch.bfloat16)
    fp32_grad = torch.tensor([12.0], dtype=torch.float32)

    norm = _grouped_grad_norm(
        fake_get_grad_norm,
        [fp32_grad, bf16_grad],
        grad_stats_parallel_group="grad-stats",
    )

    assert norm == pytest.approx(13.0)
    assert calls == [
        ((torch.bfloat16,), "grad-stats"),
        ((torch.float32,), "grad-stats"),
    ]


def test_grad_norm_does_not_skip_empty_dtype_collective():
    calls = []

    def fake_get_grad_norm(grads, norm_type, grad_stats_parallel_group):
        calls.append(tuple(grad.dtype for grad in grads))
        return 0.0

    _grouped_grad_norm(fake_get_grad_norm, [torch.ones(1, dtype=torch.bfloat16)])

    assert calls == [(torch.bfloat16,), ()]


def test_mixed_dtype_clip_never_passes_a_heterogeneous_tensor_list():
    bf16_param = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    bf16_param.decoupled_grad = torch.tensor([3.0, 4.0], dtype=torch.bfloat16)
    fp32_param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    fp32_param.decoupled_grad = torch.tensor([12.0], dtype=torch.float32)
    calls = []

    def fake_clip(parameters, max_norm, total_norm, use_decoupled_grad):
        dtypes = tuple(parameter.decoupled_grad.dtype for parameter in parameters)
        assert len(set(dtypes)) == 1
        calls.append(dtypes)

    _grouped_clip_grad(
        fake_clip,
        [fp32_param, bf16_param],
        max_norm=1.0,
        total_norm=13.0,
        use_decoupled_grad=True,
    )

    assert calls == [(torch.bfloat16,), (torch.float32,)]


def test_static_unscale_detects_fp32_overflow_created_by_inverse_scale(monkeypatch):
    fp32_grad = torch.tensor([torch.finfo(torch.float32).max], dtype=torch.float32)

    class Optimizer:
        is_stub_optimizer = False
        found_inf = torch.zeros(1, dtype=torch.float32)
        grad_scaler = SimpleNamespace(inv_scale=torch.tensor([1024.0]))
        config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=True)

        def _collect_main_grad_data_for_unscaling(self):
            return [fp32_grad]

        def get_grad_stats_parallel_group(self):
            return "grad-stats"

    optimizer = Optimizer()

    def fake_native_unscale(grads, found_inf, inv_scale):
        # Match the native kernel's relevant behavior: it checks the scaled
        # input but does not check whether multiplication itself overflows.
        assert all(torch.isfinite(grad).all() for grad in grads)
        for grad in grads:
            grad.mul_(inv_scale)

    monkeypatch.setattr(
        torch, "_amp_foreach_non_finite_check_and_unscale_", fake_native_unscale
    )
    monkeypatch.setattr(island_module.dist, "all_reduce", lambda *args, **kwargs: None)

    _install_bf16_static_unscale_compat(
        SimpleNamespace(bf16=True, grad_reduce_in_bf16=True, loss_scale=2**-10),
        SimpleNamespace(chained_optimizers=[optimizer]),
    )

    assert optimizer._unscale_main_grads_and_check_for_nan() is True
    assert torch.isinf(fp32_grad).all()


def test_optimizer_owned_ranges_must_exactly_partition_tp_local_parameter():
    _validate_owned_ranges("w", 11, [(6, 11), (0, 6), None])
    with pytest.raises(RuntimeError, match="gap"):
        _validate_owned_ranges("w", 11, [(0, 5), (6, 11)])
    with pytest.raises(RuntimeError, match="overlap"):
        _validate_owned_ranges("w", 11, [(0, 7), (6, 11)])
    with pytest.raises(RuntimeError, match="no optimizer rank"):
        _validate_owned_ranges("w", 11, [None, None])


class _FakeAdam:
    master_weights = True
    master_weight_dtype = torch.float32
    store_param_remainders = True

    def __init__(self):
        self.state = {}

    def initialize_state(self, parameter, store_remainders):
        assert store_remainders
        self.state[parameter] = {
            "master_param": torch.zeros_like(parameter, dtype=torch.int16),
            "exp_avg": torch.zeros_like(parameter, dtype=torch.float32),
            "exp_avg_sq": torch.zeros_like(parameter, dtype=torch.float32),
        }

    def get_unscaled_state(self, parameter, key):
        return self.state[parameter][key]

    def set_scaled_state(self, parameter, key, value):
        self.state[parameter][key].copy_(value)


class _FakeDistOptimizer:
    _state_offloader = None
    data_parallel_group = object()


def _runtime(
    model_param: torch.nn.Parameter,
    optimizer_param: torch.Tensor | None,
    adam: _FakeAdam,
    start: int | None,
    end: int | None,
) -> _RuntimeTensor:
    return _RuntimeTensor(
        descriptor=_descriptor("w", tuple(model_param.shape), MERGE_AVG),
        model_param=model_param,
        dist_optimizer=_FakeDistOptimizer(),
        adam_optimizer=adam,
        optimizer_param=optimizer_param,
        owned_start=start,
        owned_end=end,
        ownership_ranges=((start, end),)
        if start is not None and end is not None
        else (None,),
    )


def test_partial_owner_install_is_bit_exact_and_preserves_moments():
    island = object.__new__(MilesValueIsland)
    island.device = torch.device("cpu")
    model = torch.nn.Parameter(torch.zeros(7, dtype=torch.bfloat16))
    shard = model.detach().view(-1)[2:6]
    adam = _FakeAdam()
    adam.initialize_state(shard, True)
    adam.state[shard]["exp_avg"].copy_(torch.arange(4, dtype=torch.float32))
    adam.state[shard]["exp_avg_sq"].copy_(torch.arange(4, dtype=torch.float32) + 10)
    before_avg = adam.state[shard]["exp_avg"].clone()
    before_sq = adam.state[shard]["exp_avg_sq"].clone()
    runtime = _runtime(model, shard, adam, 2, 6)
    authoritative = torch.tensor(
        [0.25, -3.5, 1.00001, -0.00003, 19.1257, -8.75, 1000.1],
        dtype=torch.float32,
    )

    island._install_local(runtime, authoritative)

    expected_high, _ = split_fp32_to_bf16_remainder(authoritative)
    assert torch.equal(
        model.detach().view(torch.int16), expected_high.view(torch.int16)
    )
    reconstructed = reconstruct_fp32_from_bf16_remainder(
        shard, adam.state[shard]["master_param"]
    )
    assert torch.equal(
        reconstructed.view(torch.int32),
        authoritative[2:6].contiguous().view(torch.int32),
    )
    assert torch.equal(adam.state[shard]["exp_avg"], before_avg)
    assert torch.equal(adam.state[shard]["exp_avg_sq"], before_sq)


def test_rank_without_optimizer_slice_still_installs_complete_model_replica():
    island = object.__new__(MilesValueIsland)
    island.device = torch.device("cpu")
    model = torch.nn.Parameter(torch.zeros(5, dtype=torch.bfloat16))
    runtime = _runtime(model, None, _FakeAdam(), None, None)
    authoritative = torch.tensor([1.1, 2.2, 3.3, 4.4, 5.5], dtype=torch.float32)

    island._install_local(runtime, authoritative)

    expected_high, _ = split_fp32_to_bf16_remainder(authoritative)
    assert torch.equal(
        model.detach().view(torch.int16), expected_high.view(torch.int16)
    )


def test_master_reconstruction_transports_exact_fp32_bits_across_owners(monkeypatch):
    island = object.__new__(MilesValueIsland)
    island.device = torch.device("cpu")
    # Include negative zero and a non-canonical NaN payload: arithmetic SUM
    # is not a bit-preserving way to gather these values.
    expected_bits = torch.tensor(
        [0x3F800001, -2147483648, 0x00000001, 0x7FC01234, -1082130431, 0x00000000],
        dtype=torch.int32,
    )
    expected = expected_bits.view(torch.float32)
    high, remainder = split_fp32_to_bf16_remainder(expected)
    model = torch.nn.Parameter(high.clone())
    shard = model.detach().view(-1)[:3]
    adam = _FakeAdam()
    adam.initialize_state(shard, True)
    adam.state[shard]["master_param"].copy_(remainder[:3])
    runtime = _runtime(model, shard, adam, 0, 3)
    runtime.ownership_ranges = ((0, 3), (3, 6))

    def fake_all_gather(outputs, local, group):
        assert group is runtime.dist_optimizer.data_parallel_group
        outputs[0].copy_(local)
        outputs[1].zero_()
        outputs[1][3:6].copy_(expected[3:6])

    monkeypatch.setattr(island_module.dist, "all_gather", fake_all_gather)

    reconstructed = island._local_master(runtime)

    assert torch.equal(reconstructed.view(torch.int32), expected_bits)


def test_fresh_te_state_marks_megatron_offloader_initialized():
    class Offloader:
        _optimizer_states_initialized = False

        def mark_optimizer_states_initialized(self):
            self._optimizer_states_initialized = True

    island = object.__new__(MilesValueIsland)
    offloader = Offloader()
    dist_optimizer = _FakeDistOptimizer()
    dist_optimizer._state_offloader = offloader
    island.tensors = [SimpleNamespace(dist_optimizer=dist_optimizer)]
    island._fresh_state_dist_optimizers = {id(dist_optimizer): dist_optimizer}

    island._mark_fresh_optimizer_states_initialized()

    assert offloader._optimizer_states_initialized is True
    assert island._fresh_state_dist_optimizers == {}


def test_budget_consolidation_buffers_pipelined_pulls_in_step_order():
    class Client:
        finalization_timeout = 1.0

        def __init__(self):
            self.updates = [
                SimpleNamespace(fragment_id=1, version=20, data=b"one"),
                SimpleNamespace(fragment_id=0, version=10, data=b"zero"),
            ]
            self.pulls = [
                SimpleNamespace(
                    fragment_id=1, global_step=12, round_attempt=1
                ),
                SimpleNamespace(
                    fragment_id=0, global_step=11, round_attempt=1
                ),
            ]

        def check_health(self):
            return None

        def drain_updates(self):
            values, self.updates = self.updates, []
            return values

        def drain_pulls(self):
            values, self.pulls = self.pulls, []
            return values

    island = object.__new__(MilesValueIsland)
    island.client = Client()
    island._leader_payloads = {}

    assert island._next_budget_round(set()) == (0, 11, 1, 10)
    assert island._next_budget_round({0}) == (1, 12, 1, 20)
    assert island._leader_payloads == {(0, 10): b"zero", (1, 20): b"one"}
