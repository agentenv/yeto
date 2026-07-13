"""Unit tests for SCAFFOLD-lite inner control variates (yeto/scaffold.py).

Pure-math tests, no GPU / no process group / no model. They exercise the four
properties the candidate must satisfy (docs/OTHER_OPTIMIZERS.md #5):

  (a) IID workers  -> correction is ~0 (nothing to fix);
  (b) heterogeneous workers -> the correction reduces the cross-worker variance
      of the effective per-step update;
  (c) token-normalization is scale/horizon consistent;
  (d) no extra forward -> c_i is derived from the already-computed endpoint
      (the same tensor the learner pushes), not from a model call.
"""

import torch

from yeto.fragments import MERGE_AVG, Fragment
from yeto.protocol import DTYPE_F32
from yeto.scaffold import (
    VersionedControlPairs,
    effective_step_gradient,
    grad_correction,
    local_control,
    mean_control,
    zero_sum_step_diagnostics,
)
from yeto.tensor_io import pack_flat, unpack_fragment
from yeto.learner import apply_control_correction

TOKENS_PER_STEP = 4096.0  # world * micro_batch * grad_accum * seq_len
INNER_LR = 3e-4


def _window_endpoint(anchor, per_step_grad, steps, inner_lr=INNER_LR):
    """Plain-SGD endpoint of a window with a constant per-step gradient:
    theta = anchor - eta * sum_k g_k = anchor - eta * steps * g."""
    return anchor - inner_lr * steps * per_step_grad


# --- (a) IID workers: correction vanishes ---------------------------------


def test_iid_workers_zero_correction():
    torch.manual_seed(0)
    anchor = torch.randn(64)
    g = torch.randn(64)  # every worker sees the SAME gradient (IID)
    steps, tokens = 16, 16 * TOKENS_PER_STEP
    endpoints = [_window_endpoint(anchor, g, steps) for _ in range(4)]

    controls = [local_control(anchor, e, tokens) for e in endpoints]
    c = mean_control(controls, [tokens] * 4)

    for c_i in controls:
        corr = grad_correction(c_i, c, TOKENS_PER_STEP, INNER_LR)
        assert torch.allclose(corr, torch.zeros_like(corr), atol=1e-6)


def test_iid_identical_endpoints_mean_is_the_common_control():
    anchor = torch.zeros(8)
    endpoint = torch.full((8,), -0.5)
    tokens = 32 * TOKENS_PER_STEP
    controls = [local_control(anchor, endpoint, tokens) for _ in range(4)]
    c = mean_control(controls, [tokens] * 4)
    assert torch.allclose(c, controls[0])


# --- (b) heterogeneous workers: variance of effective update shrinks -------


def _cross_worker_variance(vectors):
    stacked = torch.stack(vectors)  # (M, d)
    mean = stacked.mean(dim=0, keepdim=True)
    return ((stacked - mean) ** 2).sum(dim=1).mean().item()


def test_heterogeneous_reduces_cross_worker_variance():
    torch.manual_seed(1)
    anchor = torch.randn(128)
    consensus = torch.randn(128)  # shared component
    steps, tokens = 64, 64 * TOKENS_PER_STEP

    # Each worker has the shared consensus gradient plus a large persistent
    # per-worker bias (client drift) -> heterogeneous endpoints.
    biases = [torch.randn(128) * 3.0 for _ in range(4)]
    per_step_grads = [consensus + b for b in biases]
    endpoints = [_window_endpoint(anchor, g, steps) for g in per_step_grads]

    controls = [local_control(anchor, e, tokens) for e in endpoints]
    c = mean_control(controls, [tokens] * 4)

    # Uncorrected next-window per-step gradient == the worker's own gradient
    # (persistent), corrected == grad - c_i + c.
    uncorrected = per_step_grads
    corrected = [
        effective_step_gradient(g, c_i, c, TOKENS_PER_STEP, INNER_LR)
        for g, c_i in zip(per_step_grads, controls)
    ]

    var_before = _cross_worker_variance(uncorrected)
    var_after = _cross_worker_variance(corrected)
    # Persistent drift is fully removed: corrected gradients collapse to the
    # consensus, so cross-worker variance is driven to ~0.
    assert var_after < 1e-6 * var_before
    assert var_after < var_before


def test_correction_points_from_worker_toward_consensus():
    # A single drifted worker's correction opposes its own excess gradient.
    anchor = torch.zeros(4)
    steps, tokens = 8, 8 * TOKENS_PER_STEP
    g_worker = torch.tensor([2.0, 0.0, 0.0, 0.0])
    g_consensus = torch.tensor([0.0, 0.0, 0.0, 0.0])
    e_worker = _window_endpoint(anchor, g_worker, steps)
    e_others = _window_endpoint(anchor, g_consensus, steps)

    controls = [
        local_control(anchor, e_worker, tokens),
        local_control(anchor, e_others, tokens),
        local_control(anchor, e_others, tokens),
        local_control(anchor, e_others, tokens),
    ]
    c = mean_control(controls, [tokens] * 4)
    corrected = effective_step_gradient(
        g_worker, controls[0], c, TOKENS_PER_STEP, INNER_LR
    )
    # The drifted worker's effective gradient is pulled below its raw 2.0 in
    # the drift direction (toward the consensus mean of 0.5).
    assert corrected[0] < g_worker[0]
    assert abs(corrected[0].item() - 0.5) < 1e-4


# --- (c) token-normalization is scale / horizon consistent -----------------


def test_control_invariant_to_horizon():
    # Same underlying per-token gradient, windows of different length H.
    anchor = torch.randn(32)
    g = torch.randn(32)
    c_ref = None
    for steps in (16, 64, 256):
        tokens = steps * TOKENS_PER_STEP
        endpoint = _window_endpoint(anchor, g, steps)
        c_i = local_control(anchor, endpoint, tokens)
        if c_ref is None:
            c_ref = c_i
        else:
            assert torch.allclose(c_i, c_ref, atol=1e-6)


def test_correction_scales_with_tokens_per_step():
    # Token normalization: doubling tokens/step doubles the per-step correction
    # (its per-token content is fixed).
    c_i = torch.tensor([1.0, -2.0, 0.5])
    c = torch.tensor([0.0, 0.0, 0.0])
    base = grad_correction(c_i, c, TOKENS_PER_STEP, INNER_LR)
    doubled = grad_correction(c_i, c, 2 * TOKENS_PER_STEP, INNER_LR)
    assert torch.allclose(doubled, 2.0 * base)


def test_horizon_tilt_accumulated_correction_grows_with_H():
    # A fixed per-token control drives an accumulated window correction that
    # scales with H: small at H16, larger at H256 (crossover-safe shape).
    c_i = torch.tensor([1.0, 0.0])
    c = torch.tensor([0.0, 0.0])
    per_step = grad_correction(c_i, c, TOKENS_PER_STEP, INNER_LR)
    accum = {H: per_step * H for H in (16, 64, 256)}
    n16 = accum[16].norm().item()
    n256 = accum[256].norm().item()
    assert n256 > n16
    assert abs(n256 / n16 - 16.0) < 1e-4


def test_scale_consistency_delta_and_tokens_scale_together():
    # If the delta and the tokens both scale by s (same per-token move),
    # the control is unchanged.
    anchor = torch.randn(16)
    delta = torch.randn(16)
    base_tokens = 10 * TOKENS_PER_STEP
    ref = local_control(anchor, anchor + delta, base_tokens)
    for s in (0.5, 2.0, 7.0):
        c_i = local_control(anchor, anchor + s * delta, s * base_tokens)
        assert torch.allclose(c_i, ref, atol=1e-6)


# --- (d) no extra forward: c_i reuses the already-computed endpoint --------


def test_control_derived_from_push_endpoint_only():
    # The endpoint the learner pushes IS the tensor c_i is computed from; no
    # model forward is involved. Recomputing c_i from the (copied) push payload
    # reproduces it exactly.
    torch.manual_seed(2)
    anchor = torch.randn(48)
    endpoint = anchor - INNER_LR * 20 * torch.randn(48)  # window result
    tokens = 20 * TOKENS_PER_STEP

    push_payload = endpoint.clone()  # what crosses the wire to the syncer
    c_from_live = local_control(anchor, endpoint, tokens)
    c_from_payload = local_control(anchor, push_payload, tokens)
    assert torch.equal(c_from_live, c_from_payload)


def test_mean_control_equals_token_normalized_merged_delta():
    # Server realization: c == sum_i (theta_i - A) / sum_i T_i, computed from
    # the same per-worker deltas the syncer already merges.
    torch.manual_seed(3)
    anchor = torch.randn(24)
    endpoints = [anchor + torch.randn(24) for _ in range(4)]
    tokens = [16.0, 32.0, 64.0, 8.0]  # heterogeneous windows

    controls = [local_control(anchor, e, t) for e, t in zip(endpoints, tokens)]
    c = mean_control(controls, tokens)

    merged = sum((e - anchor) for e in endpoints) / sum(tokens)
    assert torch.allclose(c, merged, atol=1e-6)


def test_local_control_rejects_nonpositive_tokens():
    a = torch.zeros(4)
    try:
        local_control(a, a, 0.0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for zero tokens")


def test_version_paired_controls_never_cross_activate():
    pairs = VersionedControlPairs()
    local_v7 = torch.tensor([7.0])
    local_v8 = torch.tensor([8.0])
    mean_v7 = torch.tensor([70.0])
    mean_v8 = torch.tensor([80.0])

    # Local v8 can race ahead of mean v7. Neither mismatched combination is
    # allowed to activate.
    pairs.add_local(8, local_v8)
    pairs.add_mean(7, mean_v7)
    assert pairs.get(7) is None
    assert pairs.get(8) is None

    pairs.add_local(7, local_v7)
    got7 = pairs.get(7)
    assert got7 is not None
    assert torch.equal(got7[0], local_v7)
    assert torch.equal(got7[1], mean_v7)

    pairs.add_mean(8, mean_v8)
    got8 = pairs.get(8)
    assert got8 is not None
    assert torch.equal(got8[0], local_v8)
    assert torch.equal(got8[1], mean_v8)

    pairs.discard_before(8)
    assert pairs.get(7) is None
    assert pairs.get(8) is not None


def test_real_f32_wire_plain_sgd_zero_sum(caplog):
    """Real codec + torch.optim.SGD path, not a symbolic update equation."""
    torch.manual_seed(4)
    workers = 4
    dim = 64
    tokens = 16 * TOKENS_PER_STEP
    frag = Fragment(MERGE_AVG, [("weight", dim)])

    # Each worker can have a different retained base version/anchor. Both its
    # local control and the server mean use the exact f32-transmitted endpoint
    # and that push's exact base anchor.
    anchors = [torch.randn(dim) for _ in range(workers)]
    endpoints = [a + 1e-3 * torch.randn(dim) for a in anchors]
    transmitted = [
        unpack_fragment(frag, pack_flat(endpoint, DTYPE_F32), DTYPE_F32)
        for endpoint in endpoints
    ]
    controls = [
        local_control(anchor, endpoint, tokens)
        for anchor, endpoint in zip(anchors, transmitted)
    ]
    mean = mean_control(controls, [tokens] * workers)
    mean_on_wire = unpack_fragment(frag, pack_flat(mean, DTYPE_F32), DTYPE_F32)
    parameters = [torch.nn.Parameter(torch.zeros(dim)) for _ in range(workers)]
    optimizers = [torch.optim.SGD([p], lr=INNER_LR) for p in parameters]
    before = [p.detach().clone() for p in parameters]
    corrections = []
    for parameter, optimizer, c_i in zip(parameters, optimizers, controls):
        parameter.grad = torch.zeros_like(parameter)
        correction = apply_control_correction(
            frag,
            {"weight": parameter},
            c_i,
            mean_on_wire,
            TOKENS_PER_STEP,
            INNER_LR,
        )
        corrections.append(correction)
        optimizer.step()
    after = [p.detach().clone() for p in parameters]

    with caplog.at_level("INFO", logger="scaffold"):
        diagnostics = zero_sum_step_diagnostics(corrections, before, after)
    assert diagnostics["correction_sum_max_abs"] < 2e-6
    assert diagnostics["displacement_sum_max_abs"] < 2e-9
    assert "before_clip correction_sum_l2" in caplog.text
    assert "after_step displacement_sum_l2" in caplog.text
