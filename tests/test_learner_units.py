"""Unit tests for learner helpers that run without GPUs or a process group."""

import torch

from yeto.learner import allreduce_trainable_grads, normalize_param_name


# --- normalize_param_name -------------------------------------------------


def test_clean_names_pass_through():
    name = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
    assert normalize_param_name(name) == name


def test_strips_fsdp_prefix():
    assert (
        normalize_param_name("_fsdp_wrapped_module.base_model.model.lm_head.weight")
        == "base_model.model.lm_head.weight"
    )


def test_strips_nested_fsdp_prefixes():
    # Nested FSDP wrapping (auto_wrap_policy) inserts the segment at every
    # wrapped level.
    name = (
        "_fsdp_wrapped_module.base_model.model.model.layers.0."
        "_fsdp_wrapped_module.self_attn.q_proj.lora_B.default.weight"
    )
    assert (
        normalize_param_name(name)
        == "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"
    )


def test_strips_checkpoint_wrapper_prefix():
    name = (
        "_fsdp_wrapped_module._checkpoint_wrapped_module.layers.0.lora_A.default.weight"
    )
    assert normalize_param_name(name) == "layers.0.lora_A.default.weight"


def test_normalized_names_match_unwrapped_layout_names():
    # Fragment layouts are keyed by parameter name, so an fsdp-lora learner
    # must expose the exact names a ddp/single-GPU learner would.
    unwrapped = [
        "base_model.model.model.embed_tokens.weight",
        "base_model.model.model.layers.1.mlp.up_proj.lora_A.default.weight",
    ]
    wrapped = ["_fsdp_wrapped_module." + n for n in unwrapped]
    assert [normalize_param_name(n) for n in wrapped] == unwrapped


# --- allreduce_trainable_grads --------------------------------------------


def _param(grad):
    p = torch.nn.Parameter(torch.zeros(3))
    p.grad = grad
    return p


def test_allreduce_noop_when_world_is_one(monkeypatch):
    import yeto.learner as learner

    def boom(*a, **k):
        raise AssertionError("dist.all_reduce must not be called for world == 1")

    monkeypatch.setattr(learner.dist, "all_reduce", boom)
    p = _param(torch.ones(3))
    allreduce_trainable_grads([p], world=1)
    assert torch.equal(p.grad, torch.ones(3))


def test_allreduce_divides_by_world(monkeypatch):
    import yeto.learner as learner

    world = 4

    def fake_all_reduce(t, op=None):
        # Every rank holds the same grad, so SUM yields world * grad.
        t.mul_(world)

    monkeypatch.setattr(learner.dist, "all_reduce", fake_all_reduce)
    g = torch.tensor([1.0, -2.0, 0.5])
    p = _param(g.clone())
    allreduce_trainable_grads([p], world=world)
    # SUM over identical ranks then /world == the original grad (DDP mean).
    assert torch.allclose(p.grad, g)


def test_allreduce_skips_none_grads(monkeypatch):
    import yeto.learner as learner

    calls = []

    def fake_all_reduce(t, op=None):
        calls.append(t)

    monkeypatch.setattr(learner.dist, "all_reduce", fake_all_reduce)
    with_grad = _param(torch.full((3,), 2.0))
    without_grad = _param(None)
    allreduce_trainable_grads([with_grad, without_grad], world=2)
    assert len(calls) == 1
    assert without_grad.grad is None
    assert torch.allclose(with_grad.grad, torch.ones(3))  # 2.0 (sum stub is id) / 2


def test_fragment_probe_signal_helpers_are_stable():
    from yeto.learner import _cosine, _sigmoid

    assert abs(_sigmoid(0.0) - 0.5) < 1e-12
    assert _sigmoid(100.0) > 1.0 - 1e-12
    assert _sigmoid(-100.0) < 1e-12
    assert abs(_cosine(torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0])) - 1.0) < 1e-12
    assert abs(_cosine(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))) < 1e-12
    assert _cosine(torch.zeros(2), torch.ones(2)) == 0.0


def test_pack_flat_matches_pack_fragment():
    from types import SimpleNamespace

    from yeto.protocol import DTYPE_BF16, DTYPE_F32
    from yeto.tensor_io import fragment_flat, pack_flat, pack_fragment, unpack_fragment

    frag = SimpleNamespace(tensors=[("a", 2), ("b", 1)], numel=3)
    params = {
        "a": torch.tensor([1.25, -2.5], dtype=torch.float32),
        "b": torch.tensor([3.75], dtype=torch.float32),
    }
    flat = fragment_flat(frag, params)
    for dtype in (DTYPE_F32, DTYPE_BF16):
        assert pack_flat(flat, dtype) == pack_fragment(frag, params, dtype)
        assert torch.allclose(unpack_fragment(frag, pack_flat(flat, dtype), dtype), flat)


def test_lora_targets_resolution():
    from types import SimpleNamespace

    from yeto.learner import _ATTENTION_TARGETS, is_moe_config, resolve_lora_targets

    dense = SimpleNamespace()
    moe = SimpleNamespace(n_routed_experts=256)
    assert not is_moe_config(dense) and is_moe_config(moe)
    # auto: attention for MoE, all-linear for dense.
    assert resolve_lora_targets("auto", moe) == _ATTENTION_TARGETS
    assert resolve_lora_targets("auto", dense) == "all-linear"
    assert resolve_lora_targets("attention", dense) == _ATTENTION_TARGETS
    assert resolve_lora_targets("all-linear", moe) == "all-linear"  # warned, honored


def test_attention_target_regex_matches_common_archs():
    import re

    from yeto.learner import _ATTENTION_TARGETS

    matching = [
        "model.layers.3.self_attn.q_proj",
        "model.layers.3.self_attn.o_proj",
        "model.layers.9.self_attn.kv_a_proj_with_mqa",  # DeepSeek MLA
        "model.layers.9.self_attn.q_b_proj",
    ]
    frozen = [
        "model.layers.3.mlp.experts.17.up_proj",  # routed expert
        "model.layers.3.mlp.gate",  # router
        "lm_head",
    ]
    for name in matching:
        assert re.fullmatch(_ATTENTION_TARGETS, name), name
    for name in frozen:
        assert not re.fullmatch(_ATTENTION_TARGETS, name), name


def test_offline_first_uses_cache_hit():
    from yeto.learner import _from_pretrained_offline_first

    calls = []

    class Factory:
        @staticmethod
        def from_pretrained(model_id, **kw):
            calls.append(kw)
            if not kw.get("local_files_only"):
                raise AssertionError("went online despite cache hit")
            return "cached-model"

    assert _from_pretrained_offline_first(Factory, "org/model", trust_remote_code=True) == "cached-model"
    assert calls == [{"local_files_only": True, "trust_remote_code": True}]


def test_offline_first_falls_back_online_on_cold_cache():
    from yeto.learner import _from_pretrained_offline_first

    calls = []

    class Factory:
        @staticmethod
        def from_pretrained(model_id, **kw):
            calls.append(kw)
            if kw.get("local_files_only"):
                raise OSError("not cached")
            return "downloaded-model"

    assert _from_pretrained_offline_first(Factory, "org/model") == "downloaded-model"
    assert [c.get("local_files_only") for c in calls] == [True, None]


# --- release_lagged_broadcast (--debug-broadcast-lag-commits) --------------


def test_lag_fifo_holds_first_k_then_releases_in_order():
    from yeto.learner import release_lagged_broadcast

    queue: list = []
    released = [
        release_lagged_broadcast(queue, (v, f"flat{v}"), 2) for v in range(1, 7)
    ]
    # Warmup: the first K=2 broadcasts are held; from then on each arrival
    # releases exactly the one K commits behind it, in arrival order.
    assert released == [
        None,
        None,
        (1, "flat1"),
        (2, "flat2"),
        (3, "flat3"),
        (4, "flat4"),
    ]
    # Steady state keeps exactly K queued (the newest K arrivals).
    assert queue == [(5, "flat5"), (6, "flat6")]


def test_lag_fifo_k1_steady_state_is_one_commit_old():
    from yeto.learner import release_lagged_broadcast

    queue: list = []
    assert release_lagged_broadcast(queue, (1, "a"), 1) is None
    for v in range(2, 6):
        # Broadcast v arriving releases v-1: the applied base is always
        # exactly one commit behind the newest known commit.
        assert release_lagged_broadcast(queue, (v, "x"), 1)[0] == v - 1
    assert len(queue) == 1 and queue[0][0] == 5


def test_lag_fifo_zero_lag_passes_through_immediately():
    from yeto.learner import release_lagged_broadcast

    queue: list = []
    for v in range(1, 4):
        assert release_lagged_broadcast(queue, (v, "x"), 0) == (v, "x")
    assert queue == []


# --- --debug-broadcast-lag-commits argument validation ---------------------

_LEARNER_REQUIRED_ARGV = [
    "--model", "m",
    "--data", "d",
    "--syncer", "none",
    "--learner-id", "0",
    "--num-learners", "1",
]


def test_lag_flag_defaults_to_zero():
    from yeto.learner import parse_args

    args = parse_args(_LEARNER_REQUIRED_ARGV)
    assert args.debug_broadcast_lag_commits == 0


def test_lag_flag_requires_fixed_response_window():
    import pytest

    from yeto.learner import parse_args

    with pytest.raises(SystemExit):
        parse_args(_LEARNER_REQUIRED_ARGV + ["--debug-broadcast-lag-commits", "1"])
    # Either fixed-window flavor satisfies the requirement.
    args = parse_args(
        _LEARNER_REQUIRED_ARGV
        + ["--debug-broadcast-lag-commits", "1", "--fixed-window-microsteps", "64"]
    )
    assert args.debug_broadcast_lag_commits == 1
    args = parse_args(
        _LEARNER_REQUIRED_ARGV
        + ["--debug-broadcast-lag-commits", "4", "--fixed-window-tokens", "8192"]
    )
    assert args.debug_broadcast_lag_commits == 4


def test_lag_flag_rejects_negative():
    import pytest

    from yeto.learner import parse_args

    with pytest.raises(SystemExit):
        parse_args(_LEARNER_REQUIRED_ARGV + ["--debug-broadcast-lag-commits", "-1"])


def test_lag_flag_zero_needs_no_window():
    from yeto.learner import parse_args

    args = parse_args(_LEARNER_REQUIRED_ARGV + ["--debug-broadcast-lag-commits", "0"])
    assert args.debug_broadcast_lag_commits == 0


def test_lag_flag_rejects_q4_wire_dtype():
    # Lag mode pushes old base_versions; the syncer rejects q4 deltas whose
    # base is not current, so the combination would stall the fragment.
    import pytest

    from yeto.learner import parse_args

    with pytest.raises(SystemExit):
        parse_args(
            _LEARNER_REQUIRED_ARGV
            + [
                "--debug-broadcast-lag-commits", "1",
                "--fixed-window-microsteps", "64",
                "--wire-dtype", "q4",
            ]
        )
    # bf16 (default) and f32 remain accepted.
    args = parse_args(
        _LEARNER_REQUIRED_ARGV
        + [
            "--debug-broadcast-lag-commits", "1",
            "--fixed-window-microsteps", "64",
            "--wire-dtype", "f32",
        ]
    )
    assert args.debug_broadcast_lag_commits == 1


# --- --fixed-window-schedule (online sync-horizon changes) ------------------


def test_schedule_parser_accepts_demo_spec():
    from yeto.learner import parse_fixed_window_schedule

    assert parse_fixed_window_schedule("0:16,160:256,170:16,330:256") == [
        (0, 16),
        (160, 256),
        (170, 16),
        (330, 256),
    ]


def test_schedule_parser_tolerates_whitespace_and_trailing_comma():
    from yeto.learner import parse_fixed_window_schedule

    assert parse_fixed_window_schedule(" 0:16 , 10:256 ,") == [(0, 16), (10, 256)]


def test_schedule_parser_rejects_malformed_specs():
    import pytest

    from yeto.learner import parse_fixed_window_schedule

    for bad in (
        "",  # no entries
        ",,",  # only separators
        "16",  # missing colon
        "a:16",  # non-integer commit
        "0:h",  # non-integer window
        "-1:16",  # negative commit index
        "0:0",  # window below one microstep
        "10:16,10:256",  # equal commit indices
        "10:16,5:256",  # decreasing commit indices
    ):
        with pytest.raises(ValueError):
            parse_fixed_window_schedule(bad)


def test_scheduled_window_steps_follows_phases():
    from yeto.learner import scheduled_window_steps

    schedule = [(0, 16), (160, 256), (170, 16), (330, 256)]
    assert scheduled_window_steps(None, 64, 0) == 64  # no schedule -> base
    assert scheduled_window_steps(schedule, 64, 0) == 16
    assert scheduled_window_steps(schedule, 64, 159) == 16
    assert scheduled_window_steps(schedule, 64, 160) == 256
    assert scheduled_window_steps(schedule, 64, 169) == 256
    assert scheduled_window_steps(schedule, 64, 170) == 16
    assert scheduled_window_steps(schedule, 64, 330) == 256
    assert scheduled_window_steps(schedule, 64, 10_000) == 256


def test_scheduled_window_steps_uses_base_before_first_entry():
    from yeto.learner import scheduled_window_steps

    schedule = [(5, 256)]
    assert scheduled_window_steps(schedule, 16, 0) == 16
    assert scheduled_window_steps(schedule, 16, 4) == 16
    assert scheduled_window_steps(schedule, 16, 5) == 256


def test_window_growth_invalidates_undersized_snapshots_only():
    from yeto.learner import invalidate_undersized_snapshots

    snapshots = [
        {"c_steps": 16, "flat": "a"},  # too small for the grown window
        {"c_steps": 256, "flat": "b"},  # already fills it
        None,  # nothing cached yet
    ]
    invalidate_undersized_snapshots(snapshots, 256)
    assert snapshots[0] is None
    assert snapshots[1] == {"c_steps": 256, "flat": "b"}
    assert snapshots[2] is None


def test_window_shrink_keeps_existing_snapshots():
    from yeto.learner import invalidate_undersized_snapshots

    snapshots = [{"c_steps": 256, "flat": "a"}, {"c_steps": 300, "flat": "b"}]
    invalidate_undersized_snapshots(snapshots, 16)
    assert snapshots == [
        {"c_steps": 256, "flat": "a"},
        {"c_steps": 300, "flat": "b"},
    ]


def test_schedule_flag_defaults_off_and_parses_into_pairs():
    from yeto.learner import parse_args

    args = parse_args(_LEARNER_REQUIRED_ARGV)
    assert args.fixed_window_schedule is None
    args = parse_args(
        _LEARNER_REQUIRED_ARGV + ["--fixed-window-schedule", "0:16,160:256"]
    )
    assert args.fixed_window_schedule == [(0, 16), (160, 256)]


def test_schedule_flag_rejects_malformed_spec_at_parse_time():
    import pytest

    from yeto.learner import parse_args

    with pytest.raises(SystemExit):
        parse_args(_LEARNER_REQUIRED_ARGV + ["--fixed-window-schedule", "10:16,5:8"])


def test_schedule_flag_satisfies_lag_mode_window_requirement():
    from yeto.learner import parse_args

    args = parse_args(
        _LEARNER_REQUIRED_ARGV
        + [
            "--debug-broadcast-lag-commits", "1",
            "--fixed-window-schedule", "0:16",
        ]
    )
    assert args.debug_broadcast_lag_commits == 1
    assert args.fixed_window_schedule == [(0, 16)]


# --- --barrier-sync (true lockstep DiLoCo) ---------------------------------


def test_barrier_release_clears_only_on_a_newer_version():
    from yeto.learner import barrier_release

    # Fragment 2 was pushed from base version 4; its merge lands as version 6.
    awaiting = {2: 4}
    assert barrier_release(awaiting, 2, 6) is True
    assert awaiting == {}


def test_barrier_release_ignores_stale_or_equal_broadcasts():
    from yeto.learner import barrier_release

    # A duplicate/echo of the base (version == base) or an older version does
    # not release the barrier — the learner keeps blocking for the real merge.
    awaiting = {0: 5}
    assert barrier_release(awaiting, 0, 5) is False
    assert barrier_release(awaiting, 0, 3) is False
    assert awaiting == {0: 5}
    # The genuine merge (strictly newer) then releases it.
    assert barrier_release(awaiting, 0, 6) is True
    assert awaiting == {}


def test_barrier_release_noop_for_unwatched_fragment():
    from yeto.learner import barrier_release

    # A broadcast for a fragment the learner is not waiting on never blocks or
    # raises, and leaves the outstanding waits untouched.
    awaiting = {1: 2}
    assert barrier_release(awaiting, 3, 99) is False
    assert awaiting == {1: 2}


def test_barrier_gate_serializes_a_pipelined_round():
    """State-machine walk-through: two fragments pushed in one boundary
    (pipeline depth 2) both arm the gate; the learner stays blocked until
    BOTH merges return, then resumes — no inner step in between."""
    from yeto.learner import barrier_release

    awaiting: dict[int, int] = {}
    # Learner pushed fragment 0 (base v0) and fragment 1 (base v0) this round.
    awaiting[0] = 0
    awaiting[1] = 0
    assert awaiting  # gate closed -> inner loop must block

    # Fragment 0's merge (round t=1) arrives first.
    barrier_release(awaiting, 0, 1)
    assert awaiting == {1: 0}  # still blocked on fragment 1

    # Fragment 1's merge (round t=2) arrives.
    barrier_release(awaiting, 1, 2)
    assert not awaiting  # gate open -> learner resumes the next window


def test_barrier_sync_flag_defaults_off_and_parses():
    from yeto.learner import parse_args

    assert parse_args(_LEARNER_REQUIRED_ARGV).barrier_sync is False
    args = parse_args(
        ["--model", "m", "--data", "d", "--syncer", "h:1", "--learner-id", "0",
         "--num-learners", "1", "--barrier-sync"]
    )
    assert args.barrier_sync is True


def test_barrier_round_closure_targets_each_full_fragment_boundary():
    from yeto.learner import barrier_round_closure_target

    snapshots = [{"c_steps": 16}, {"c_steps": 16}, {"c_steps": 16}, None]
    assert barrier_round_closure_target(
        barrier_sync=True,
        shutdown=False,
        steps_total=16,
        max_local_steps=128,
        fixed_window_steps=16,
        fragment_count=4,
        global_step=1,
        fixed_window_snapshots=snapshots,
    ) == 4
    assert barrier_round_closure_target(
        barrier_sync=True,
        shutdown=False,
        steps_total=112,
        max_local_steps=128,
        fixed_window_steps=16,
        fragment_count=4,
        global_step=29,
        fixed_window_snapshots=snapshots,
    ) == 32


def test_barrier_round_closure_caps_at_32_without_extra_optimizer_step():
    from yeto.learner import barrier_round_closure_target

    snapshots = [
        {"c_steps": 16, "c_tokens": 2048, "local_step": 128}
        for _ in range(4)
    ]
    assert barrier_round_closure_target(
        barrier_sync=True,
        shutdown=False,
        steps_total=128,
        max_local_steps=128,
        fixed_window_steps=16,
        fragment_count=4,
        global_step=31,
        fixed_window_snapshots=snapshots,
    ) == 32
    assert snapshots[3] == {
        "c_steps": 16,
        "c_tokens": 2048,
        "local_step": 128,
    }
