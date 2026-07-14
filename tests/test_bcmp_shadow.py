import copy
import json
import math

import pytest
import torch

from yeto.bcmp_shadow import (
    BCMPShadowTracker,
    DROP_SCHEMA,
    RESOLUTION_SCHEMA,
    RUN_SCHEMA,
    RUN_SUMMARY_SCHEMA,
    SHADOW_SCHEMA,
    append_jsonl,
    broadcast_jump_stats,
    compute_bcmp_shadow,
    score_directions,
)
from yeto.fragments import Fragment, FragmentLayout, MERGE_RDA
from yeto.tensor_io import apply_fragment


def _problem(*, amsgrad=False):
    params = {
        "a": torch.nn.Parameter(torch.tensor([0.7, -0.3], dtype=torch.float64)),
        "b": torch.nn.Parameter(torch.tensor([0.2, 0.5, -0.4], dtype=torch.float64)),
    }
    frag = Fragment(MERGE_RDA, [(name, p.numel()) for name, p in params.items()])
    opt = torch.optim.AdamW(
        params.values(),
        lr=0.03,
        betas=(0.8, 0.9),
        eps=1e-7,
        weight_decay=0.04,
        amsgrad=amsgrad,
    )
    warmup_grads = (
        {"a": [0.2, -0.5], "b": [0.1, 0.4, -0.2]},
        {"a": [-0.1, -0.3], "b": [0.2, 0.1, 0.3]},
        {"a": [0.4, 0.2], "b": [-0.3, 0.2, 0.1]},
    )
    for row in warmup_grads:
        for name, values in row.items():
            params[name].grad = torch.tensor(values, dtype=torch.float64)
        opt.step()
        opt.zero_grad(set_to_none=True)
    for name, values in {"a": [0.3, -0.2], "b": [-0.4, 0.5, 0.1]}.items():
        params[name].grad = torch.tensor(values, dtype=torch.float64)
    return params, frag, opt


def _shadow(params, frag, opt, *, capture=True):
    return compute_bcmp_shadow(
        frag,
        params,
        opt,
        fragment_id=2,
        broadcast_version=17,
        broadcast_local_step=48,
        gradient_local_step=48,
        learner_id=3,
        rank=0,
        jump_stats={"broadcast_jump_l2": 0.125},
        capture_tensors=capture,
    )


def _state_snapshot(params, opt):
    return {
        name: {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
            for key, value in opt.state[p].items()
        }
        for name, p in params.items()
    }


def _assert_state_equal(params, opt, before):
    for name, p in params.items():
        assert opt.state[p].keys() == before[name].keys()
        for key, value in opt.state[p].items():
            old = before[name][key]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, old), (name, key)
            else:
                assert value == old


def test_shadow_is_finite_json_and_does_not_mutate_live_training_state():
    params, frag, opt = _problem()
    param_before = {name: p.detach().clone() for name, p in params.items()}
    grad_before = {name: p.grad.detach().clone() for name, p in params.items()}
    state_before = _state_snapshot(params, opt)

    result = _shadow(params, frag, opt)

    assert result.record["schema"] == SHADOW_SCHEMA
    assert result.record["event_id"] == "l3-r0-f2-v17-b48-g48"
    assert result.record["numel"] == 5
    assert result.record["tensor_count"] == 2
    assert result.record["broadcast_jump_l2"] == 0.125
    json.dumps(result.record, allow_nan=False)
    for name, p in params.items():
        assert torch.equal(p, param_before[name])
        assert torch.equal(p.grad, grad_before[name])
    _assert_state_equal(params, opt, state_before)


@pytest.mark.parametrize("amsgrad", [False, True])
@pytest.mark.parametrize("candidate", ["stock", "ray", "slab", "reset"])
def test_predicted_counterfactual_step_matches_real_torch_adamw(amsgrad, candidate):
    params, frag, opt = _problem(amsgrad=amsgrad)
    result = _shadow(params, frag, opt)
    tensors = result.tensors
    assert tensors is not None
    before = {name: p.detach().clone() for name, p in params.items()}

    if candidate != "stock":
        moments = getattr(tensors, f"{candidate}_raw_exp_avg")
        for name, p in params.items():
            opt.state[p]["exp_avg"].copy_(moments[name])
    opt.step()

    predicted = getattr(tensors, f"{candidate}_step")
    for name, p in params.items():
        actual = before[name] - p.detach()
        torch.testing.assert_close(actual, predicted[name], rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize(
    "multiple, expected_a, expected_region",
    [(-1.0, 0.0, "reset"), (0.4, 0.4, "in_range"), (2.0, 1.0, "cap")],
)
def test_projection_coefficient_clips_preconditioned_work(multiple, expected_a, expected_region):
    p = torch.nn.Parameter(torch.tensor([0.2, -0.4], dtype=torch.float64))
    params = {"p": p}
    frag = Fragment(MERGE_RDA, [("p", 2)])
    opt = torch.optim.AdamW([p], lr=0.01, betas=(0.9, 0.99), eps=1e-8)
    g = torch.tensor([0.3, -0.7], dtype=torch.float64)
    step = 7
    beta1 = opt.param_groups[0]["betas"][0]
    opt.state[p] = {
        "step": torch.tensor(float(step), dtype=torch.float64),
        "exp_avg": (1.0 - beta1**step) * multiple * g,
        "exp_avg_sq": torch.tensor([0.4, 0.8], dtype=torch.float64),
    }
    p.grad = g.clone()

    result = compute_bcmp_shadow(
        frag, params, opt,
        fragment_id=0, broadcast_version=1, broadcast_local_step=7,
        gradient_local_step=7, learner_id=0, rank=0, capture_tensors=True,
    )

    assert result.record["a_raw"] == pytest.approx(multiple, abs=2e-15)
    assert result.record["a"] == pytest.approx(expected_a, abs=2e-15)
    assert result.record["projection_region"] == expected_region
    assert result.record["ray_preconditioned_work"] == pytest.approx(
        expected_a * result.record["projection_denominator"], rel=2e-15
    )
    assert result.record["slab_preconditioned_work"] == pytest.approx(
        min(max(result.record["projection_numerator"], 0.0),
            result.record["projection_denominator"]),
        rel=2e-15,
        abs=2e-15,
    )


def test_bias_corrected_ray_a_one_stays_exactly_one_gradient_after_step():
    p = torch.nn.Parameter(torch.tensor([0.2, -0.4], dtype=torch.float64))
    params = {"p": p}
    frag = Fragment(MERGE_RDA, [("p", 2)])
    beta1, beta2 = 0.9, 0.99
    opt = torch.optim.AdamW([p], lr=0.01, betas=(beta1, beta2), eps=1e-8)
    g = torch.tensor([0.3, -0.7], dtype=torch.float64)
    step = 7
    opt.state[p] = {
        "step": torch.tensor(float(step), dtype=torch.float64),
        "exp_avg": (1.0 - beta1**step) * g,
        "exp_avg_sq": torch.tensor([0.4, 0.8], dtype=torch.float64),
    }
    p.grad = g.clone()
    result = compute_bcmp_shadow(
        frag, params, opt,
        fragment_id=0, broadcast_version=1, broadcast_local_step=7,
        gradient_local_step=7, learner_id=0, rank=0, capture_tensors=True,
    )
    assert result.record["a"] == pytest.approx(1.0, abs=2e-15)
    opt.state[p]["exp_avg"].copy_(result.tensors.ray_raw_exp_avg["p"])
    opt.step()
    new_step = step + 1
    corrected = opt.state[p]["exp_avg"] / (1.0 - beta1**new_step)
    torch.testing.assert_close(corrected, g, rtol=2e-15, atol=2e-15)


def test_conservative_slab_is_identity_in_range_and_preserves_transverse_component():
    params, frag, opt = _problem()
    # Force the old bias-corrected moment to 0.4*g plus a nonzero component.
    # The correction is zero when its total work is in range, so slab must be
    # byte-identical to the factual raw moment regardless of that component.
    for p in params.values():
        state = opt.state[p]
        step = int(state["step"].item())
        beta1 = opt.param_groups[0]["betas"][0]
        state["exp_avg"].copy_((1.0 - beta1**step) * 0.4 * p.grad)
    result = _shadow(params, frag, opt)
    assert result.record["projection_region"] == "in_range"
    for name, p in params.items():
        torch.testing.assert_close(
            result.tensors.slab_raw_exp_avg[name], opt.state[p]["exp_avg"],
            rtol=0.0, atol=0.0,
        )
    assert result.record["slab_transverse_preservation_error_l2"] < 1e-15

    # In the cap regime slab changes only the gradient-parallel coefficient;
    # the explicit diagnostic identity remains at roundoff.
    for p in params.values():
        state = opt.state[p]
        step = int(state["step"].item())
        beta1 = opt.param_groups[0]["betas"][0]
        transverse = torch.flip(p.grad, dims=[0]) * 0.3
        state["exp_avg"].copy_((1.0 - beta1**step) * (2.0 * p.grad + transverse))
    capped = _shadow(params, frag, opt)
    assert capped.record["a"] <= 1.0
    assert capped.record["slab_transverse_preservation_error_l2"] < 1e-14


def test_zero_fragment_gradient_is_explicit_fallback_and_record_stays_finite():
    params, frag, opt = _problem()
    for p in params.values():
        p.grad.zero_()
    result = _shadow(params, frag, opt)
    assert result.record["fallback"] is True
    assert result.record["fallback_reason"] == "zero_preconditioned_gradient_energy"
    assert result.record["a"] == 0.0
    json.dumps(result.record, allow_nan=False)


def test_broadcast_jump_stats_matches_blend_and_does_not_mutate_params():
    params, frag, _ = _problem()
    before = {name: p.detach().clone() for name, p in params.items()}
    local = torch.cat([params[name].detach().reshape(-1) for name, _ in frag.tensors])
    global_flat = local + torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0], dtype=torch.float64)
    stats = broadcast_jump_stats(frag, params, global_flat, merge_alpha=0.25)
    assert stats["broadcast_jump_l2"] == pytest.approx(
        0.75 * math.sqrt(1 + 4 + 9 + 16 + 25)
    )
    for name, p in params.items():
        assert torch.equal(p, before[name])


def test_future_gradient_resolution_sign_and_jsonl(tmp_path):
    params, frag, opt = _problem()
    result = _shadow(params, frag, opt)
    directions = result.tensors.ray_minus_stock
    # Resolve against exactly the candidate-minus-stock direction: cosine +1.
    for name, p in params.items():
        p.grad = directions[name].clone()
    resolution = score_directions(
        directions,
        params,
        event_id=result.record["event_id"],
        candidate="ray",
        resolved_local_step=49,
    )
    assert resolution["schema"] == RESOLUTION_SCHEMA
    assert resolution["future_gradient_dot"] > 0.0
    assert resolution["future_gradient_cosine"] == pytest.approx(1.0, abs=2e-15)

    path = tmp_path / "learner-3-rank-0.jsonl"
    append_jsonl(path, result.record)
    append_jsonl(path, resolution)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["schema"] for row in rows] == [SHADOW_SCHEMA, RESOLUTION_SCHEMA]


def test_tracker_records_all_controls_resolves_next_gradient_and_never_mutates(tmp_path):
    params, frag, opt = _problem()
    layout = FragmentLayout([frag])
    path = tmp_path / "arm" / "bcmp_shadow_learner_3.jsonl"
    tracker = BCMPShadowTracker(path, every=1, learner_id=3, rank=0)

    local = torch.cat([params[name].detach().reshape(-1) for name, _ in frag.tensors])
    global_flat = local + torch.linspace(
        -0.2, 0.2, frag.numel, dtype=local.dtype, device=local.device
    )
    assert tracker.note_broadcast(
        fragment_id=0,
        broadcast_version=9,
        local_step=3,
        fragment=frag,
        params=params,
        global_flat=global_flat,
        merge_alpha=0.0,
    )
    apply_fragment(frag, global_flat, params)
    for name, p in params.items():
        p.grad = torch.linspace(-0.3, 0.4, p.numel(), dtype=p.dtype).reshape_as(p)

    params_before = {name: p.detach().clone() for name, p in params.items()}
    grads_before = {name: p.grad.detach().clone() for name, p in params.items()}
    state_before = _state_snapshot(params, opt)
    assert tracker.before_optimizer_step(
        layout=layout, params=params, optimizer=opt, local_step=3
    ) == 1
    for name, p in params.items():
        assert torch.equal(p, params_before[name])
        assert torch.equal(p.grad, grads_before[name])
    _assert_state_equal(params, opt, state_before)

    # The factual stock step happens outside the tracker.  At the following
    # post-clip boundary, all three candidate-minus-stock directions resolve
    # against a genuinely future gradient.
    opt.step()
    for name, p in params.items():
        p.grad = torch.linspace(0.5, -0.2, p.numel(), dtype=p.dtype).reshape_as(p)
    assert tracker.before_optimizer_step(
        layout=layout, params=params, optimizer=opt, local_step=4
    ) == 0
    tracker.close(local_step=4)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["schema"] for row in rows] == [
        RUN_SCHEMA,
        SHADOW_SCHEMA,
        RESOLUTION_SCHEMA,
        RESOLUTION_SCHEMA,
        RESOLUTION_SCHEMA,
        RUN_SUMMARY_SCHEMA,
    ]
    assert rows[1]["broadcast_version"] == 9
    assert rows[1]["completed_steps_between_broadcast_and_gradient"] == 0
    assert rows[1]["upcoming_optimizer_step"] == 4
    assert {row["candidate"] for row in rows[2:5]} == {"ray", "slab", "reset"}
    summary = rows[-1]
    assert summary["broadcasts_seen"] == 1
    assert summary["broadcasts_sampled"] == 1
    assert summary["shadow_events_recorded"] == 1
    assert summary["resolutions_recorded"] == 3
    assert summary["drops_recorded"] == 0
    assert summary["shadow_wall_s"] >= 0.0
    assert summary["active_wall_s"] >= summary["shadow_wall_s"]
    assert 0.0 <= summary["shadow_wall_fraction"] <= 1.0
    assert summary["cuda_synchronized_for_timing"] is False
    json.dumps(summary, allow_nan=False)


def test_unsampled_broadcast_supersedes_sampled_pending_event(tmp_path):
    params, frag, opt = _problem()
    layout = FragmentLayout([frag])
    path = tmp_path / "shadow.jsonl"
    tracker = BCMPShadowTracker(path, every=2, learner_id=0, rank=0)
    flat = torch.cat([params[name].detach().reshape(-1) for name, _ in frag.tensors])
    assert tracker.note_broadcast(
        fragment_id=0, broadcast_version=1, local_step=3, fragment=frag,
        params=params, global_flat=flat, merge_alpha=0.0,
    )
    assert not tracker.note_broadcast(
        fragment_id=0, broadcast_version=2, local_step=3, fragment=frag,
        params=params, global_flat=flat, merge_alpha=0.0,
    )
    assert tracker.before_optimizer_step(
        layout=layout, params=params, optimizer=opt, local_step=3
    ) == 0
    tracker.close(local_step=3)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["schema"] for row in rows] == [
        RUN_SCHEMA, DROP_SCHEMA, RUN_SUMMARY_SCHEMA
    ]
    assert rows[1]["broadcast_version"] == 1
    assert rows[1]["reason"] == "superseded_before_fresh_gradient"
    assert rows[2]["drops_recorded"] == 1


def test_rebroadcast_drops_contaminated_future_gradient_resolution(tmp_path):
    params, frag, opt = _problem()
    layout = FragmentLayout([frag])
    path = tmp_path / "shadow.jsonl"
    tracker = BCMPShadowTracker(path, every=2, learner_id=0, rank=0)
    flat = torch.cat([params[name].detach().reshape(-1) for name, _ in frag.tensors])
    assert tracker.note_broadcast(
        fragment_id=0, broadcast_version=1, local_step=3, fragment=frag,
        params=params, global_flat=flat, merge_alpha=0.0,
    )
    assert tracker.before_optimizer_step(
        layout=layout, params=params, optimizer=opt, local_step=3
    ) == 1

    # The next broadcast is unsampled, but it still changes the point at which
    # the future gradient will be measured.  All old controls must be marked
    # contaminated instead of being resolved against that gradient.
    assert not tracker.note_broadcast(
        fragment_id=0, broadcast_version=2, local_step=4, fragment=frag,
        params=params, global_flat=flat, merge_alpha=0.0,
    )
    assert tracker.before_optimizer_step(
        layout=layout, params=params, optimizer=opt, local_step=4
    ) == 0
    tracker.close(local_step=4)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    drops = [row for row in rows if row["schema"] == DROP_SCHEMA]
    assert len(drops) == 3
    assert {row["candidate"] for row in drops} == {"ray", "slab", "reset"}
    assert {row["reason"] for row in drops} == {
        "rebroadcast_before_future_gradient"
    }
    assert not any(row["schema"] == RESOLUTION_SCHEMA for row in rows)
    assert rows[-1]["drops_recorded"] == 3


def test_rejects_missing_gradient_and_bad_fragment_mapping():
    params, frag, opt = _problem()
    params["a"].grad = None
    with pytest.raises(ValueError, match="has no post-clip gradient"):
        _shadow(params, frag, opt)
    bad = Fragment(MERGE_RDA, [("a", params["a"].numel() + 1)])
    params["a"].grad = torch.ones_like(params["a"])
    with pytest.raises(ValueError, match="fragment records"):
        compute_bcmp_shadow(
            bad, params, opt,
            fragment_id=0, broadcast_version=1, broadcast_local_step=1,
            gradient_local_step=1, learner_id=0, rank=0,
        )


def test_learner_cli_accepts_shadow_path_and_cadence():
    from yeto.learner import parse_args

    args = parse_args(
        [
            "--model", "m", "--data", "d", "--syncer", "127.0.0.1:9000",
            "--learner-id", "2", "--num-learners", "4",
            "--bcmp-shadow-path", "/tmp/arm/bcmp_shadow_learner_2.jsonl",
            "--bcmp-shadow-every", "7",
        ]
    )
    assert args.bcmp_shadow_path.endswith("bcmp_shadow_learner_2.jsonl")
    assert args.bcmp_shadow_every == 7
