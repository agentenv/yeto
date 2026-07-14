from __future__ import annotations

import hashlib

import pytest

from yeto.capture_v2_crn_plan import CRNAuthorityError
from yeto.capture_v2_pti_crn import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    PTICRNFiniteLossObservation,
    compute_frozen_pti_finite_loss_gate,
)
from yeto.capture_v2_store import ManifestRef


def _manifest(label: str) -> ManifestRef:
    return ManifestRef(label, hashlib.sha256(label.encode()).hexdigest(), 1, False)


def _observation(
    index: int,
    *,
    k0: float = 0.0,
    k8: float = 0.003,
    action: bool = True,
) -> PTICRNFiniteLossObservation:
    return PTICRNFiniteLossObservation(
        plan=_manifest(f"plan-{index}"),
        outcome=_manifest(f"outcome-{index}"),
        boundary_sha256=hashlib.sha256(f"boundary-{index}".encode()).hexdigest(),
        fragment_id=index % 4,
        commit_seq=index,
        candidate_used_nonstock=action,
        k0_gain=k0,
        k8_gain=k8,
    )


def test_frozen_finite_loss_gate_passes_exact_balanced_positive_panel():
    rows = tuple(_observation(index, action=index < 8) for index in range(32))

    summary = compute_frozen_pti_finite_loss_gate(rows)

    assert BOOTSTRAP_REPLICATES == 20_000
    assert BOOTSTRAP_SEED == 5_318_008
    assert summary.fragment_counts == ((0, 8), (1, 8), (2, 8), (3, 8))
    assert summary.action_count == 8
    assert summary.action_fraction == 0.25
    assert summary.mean_k0_gain == 0.0
    assert summary.mean_k8_gain == pytest.approx(0.003)
    assert summary.k8_bootstrap_lower_95 == pytest.approx(0.003)
    assert summary.k8_positive_boundary_fraction == 1.0
    assert summary.k8_positive_fragment_means == 4
    assert summary.sample_gate_pass is True
    assert summary.action_rate_gate_pass is True
    assert summary.k0_gate_pass is True
    assert summary.k8_gate_pass is True
    assert summary.regression_gate_pass is True
    assert summary.finite_loss_gate_pass is True


def test_frozen_thresholds_preserve_strict_and_nonstrict_edges():
    rows = tuple(
        _observation(
            index,
            k0=-0.001,
            k8=0.002,
            action=index < 7,
        )
        for index in range(32)
    )

    summary = compute_frozen_pti_finite_loss_gate(rows)

    assert summary.action_rate_gate_pass is False  # 7/32 is below 25%.
    assert summary.k0_gate_pass is True  # Mean -0.001 is explicitly allowed.
    assert summary.k8_bootstrap_lower_95 > 0.0
    assert summary.k8_gate_pass is False  # Mean must be strictly above 0.002.
    assert summary.finite_loss_gate_pass is False


def test_frozen_gate_rejects_unbalanced_fragments_and_large_regression():
    rows = [
        PTICRNFiniteLossObservation(
            plan=_manifest(f"unbalanced-plan-{index}"),
            outcome=_manifest(f"unbalanced-outcome-{index}"),
            boundary_sha256=hashlib.sha256(
                f"unbalanced-boundary-{index}".encode()
            ).hexdigest(),
            fragment_id=0,
            commit_seq=index,
            candidate_used_nonstock=True,
            k0_gain=0.0,
            k8_gain=0.003,
        )
        for index in range(32)
    ]
    rows[-1] = PTICRNFiniteLossObservation(
        **{
            **rows[-1].__dict__,
            "k8_gain": -0.05000000000000001,
        }
    )

    summary = compute_frozen_pti_finite_loss_gate(tuple(rows))

    assert summary.sample_gate_pass is False
    assert summary.regression_gate_pass is False
    assert summary.finite_loss_gate_pass is False


def test_frozen_gate_rejects_duplicate_plan_authority():
    row = _observation(0)
    with pytest.raises(CRNAuthorityError, match="repeat a plan"):
        compute_frozen_pti_finite_loss_gate((row, row))
