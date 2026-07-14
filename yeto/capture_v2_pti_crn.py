"""Frozen finite-loss statistics for an authoritative PTI capture-v2 campaign.

The single-boundary evaluator proves same-state A/B then B/A isolation.  This
module closes the next layer: it accepts exactly one attested paired outcome
for every plan in a precommitted campaign index and computes the numerical
``k=0``/``k=8`` gates frozen by the PTI fresh-confirmation protocol.

Direction-score confidence and matched runtime overhead are intentionally not
accepted here.  Consequently this module can pass the finite-loss sub-gate but
cannot by itself authorize PTI promotion or a live experiment claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import statistics
from typing import Sequence

from .capture_v2_crn_plan import (
    CRNAuthorityError,
    CRNCampaignIndexRef,
    CRNPairedOutcomeRef,
    load_authorized_crn_plan,
    load_crn_campaign_index,
    load_crn_paired_outcome,
)
from .capture_v2_pti_adapter import (
    CRN_ACTION_CAPABILITIES,
    load_authoritative_pti_action,
)
from .capture_v2_store import CaptureObjectStore, ManifestRef


BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 5_318_008
MIN_BOUNDARIES = 32
MIN_FRAGMENT_BOUNDARIES = 8
REQUIRED_FRAGMENTS = 4
MIN_ACTION_FRACTION = 0.25
MIN_K8_MEAN_GAIN = 0.002
MIN_POSITIVE_BOUNDARY_FRACTION = 0.60
MIN_POSITIVE_FRAGMENT_MEANS = 3
MIN_K0_MEAN_GAIN = -0.001
MIN_K0_FIFTH_PERCENTILE = -0.01
MAX_INDIVIDUAL_REGRESSION = 0.05


@dataclass(frozen=True)
class PTICRNFiniteLossObservation:
    """One authority-verified boundary contribution to the frozen denominator."""

    plan: ManifestRef
    outcome: ManifestRef
    boundary_sha256: str
    fragment_id: int
    commit_seq: int
    candidate_used_nonstock: bool
    k0_gain: float
    k8_gain: float


@dataclass(frozen=True)
class PTICRNFiniteLossSummary:
    """Deterministic arithmetic for the preregistered finite-loss sub-gate."""

    boundary_count: int
    fragment_counts: tuple[tuple[int, int], ...]
    action_count: int
    action_fraction: float
    mean_k0_gain: float
    k0_fifth_percentile: float
    mean_k8_gain: float
    k8_bootstrap_lower_95: float
    k8_positive_boundary_fraction: float
    k8_fragment_means: tuple[tuple[int, float], ...]
    k8_positive_fragment_means: int
    worst_individual_gain: float
    sample_gate_pass: bool
    action_rate_gate_pass: bool
    k0_gate_pass: bool
    k8_gate_pass: bool
    regression_gate_pass: bool
    finite_loss_gate_pass: bool


@dataclass(frozen=True)
class PTICRNCampaignAnalysis:
    """Authority identity, canonical observations, and frozen finite-loss result."""

    campaign_index: CRNCampaignIndexRef
    observations: tuple[PTICRNFiniteLossObservation, ...]
    summary: PTICRNFiniteLossSummary


def _same_manifest(left: ManifestRef, right: ManifestRef) -> bool:
    return (
        left.manifest_id == right.manifest_id
        and left.sha256 == right.sha256
        and left.bytes == right.bytes
    )


def _nearest_rank(values: Sequence[float], probability: float) -> float:
    """Return the deterministic nearest-rank empirical percentile."""

    if not values:
        raise CRNAuthorityError("PTI finite-loss statistics require observations")
    ordered = sorted(values)
    rank = max(1, math.ceil(probability * len(ordered)))
    return ordered[rank - 1]


def _fragment_stratified_lower_endpoint(
    observations: Sequence[PTICRNFiniteLossObservation],
) -> float:
    by_fragment: dict[int, list[float]] = {}
    for row in observations:
        by_fragment.setdefault(row.fragment_id, []).append(row.k8_gain)
    ordered_groups = tuple(by_fragment[key] for key in sorted(by_fragment))
    rng = random.Random(BOOTSTRAP_SEED)
    replicates: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        total = 0.0
        count = 0
        for group in ordered_groups:
            total += sum(group[rng.randrange(len(group))] for _ in group)
            count += len(group)
        replicates.append(total / count)
    return _nearest_rank(replicates, 0.025)


def compute_frozen_pti_finite_loss_gate(
    observations: Sequence[PTICRNFiniteLossObservation],
) -> PTICRNFiniteLossSummary:
    """Compute frozen arithmetic over already authority-verified observations.

    Callers seeking scientifically admissible evidence must use
    :func:`analyze_authorized_pti_crn_campaign`, which constructs these rows
    only after verifying the complete campaign index and every attestation.
    """

    if isinstance(observations, (str, bytes)) or not isinstance(observations, Sequence):
        raise TypeError("observations must be a sequence")
    rows = tuple(observations)
    if not rows:
        raise CRNAuthorityError("PTI finite-loss statistics require observations")
    if any(not isinstance(row, PTICRNFiniteLossObservation) for row in rows):
        raise TypeError("observations must contain PTICRNFiniteLossObservation values")
    if len({row.plan.sha256 for row in rows}) != len(rows):
        raise CRNAuthorityError("PTI finite-loss observations repeat a plan")
    if any(
        not math.isfinite(row.k0_gain) or not math.isfinite(row.k8_gain) for row in rows
    ):
        raise CRNAuthorityError("PTI finite-loss observations must be finite")

    by_fragment: dict[int, list[PTICRNFiniteLossObservation]] = {}
    for row in rows:
        by_fragment.setdefault(row.fragment_id, []).append(row)
    fragment_counts = tuple(
        (fragment, len(group)) for fragment, group in sorted(by_fragment.items())
    )
    fragment_means = tuple(
        (
            fragment,
            statistics.fmean(row.k8_gain for row in group),
        )
        for fragment, group in sorted(by_fragment.items())
    )

    k0 = tuple(row.k0_gain for row in rows)
    k8 = tuple(row.k8_gain for row in rows)
    mean_k0 = statistics.fmean(k0)
    mean_k8 = statistics.fmean(k8)
    fifth_k0 = _nearest_rank(k0, 0.05)
    lower_k8 = _fragment_stratified_lower_endpoint(rows)
    positive_fraction = sum(value > 0.0 for value in k8) / len(k8)
    positive_fragments = sum(value > 0.0 for _, value in fragment_means)
    action_count = sum(row.candidate_used_nonstock for row in rows)
    action_fraction = action_count / len(rows)
    worst_gain = min((*k0, *k8))

    sample_pass = (
        len(rows) >= MIN_BOUNDARIES
        and len(fragment_counts) == REQUIRED_FRAGMENTS
        and all(count >= MIN_FRAGMENT_BOUNDARIES for _, count in fragment_counts)
    )
    action_pass = action_fraction >= MIN_ACTION_FRACTION
    k0_pass = mean_k0 >= MIN_K0_MEAN_GAIN and fifth_k0 > MIN_K0_FIFTH_PERCENTILE
    k8_pass = (
        mean_k8 > MIN_K8_MEAN_GAIN
        and lower_k8 > 0.0
        and positive_fragments >= MIN_POSITIVE_FRAGMENT_MEANS
        and positive_fraction >= MIN_POSITIVE_BOUNDARY_FRACTION
    )
    regression_pass = worst_gain >= -MAX_INDIVIDUAL_REGRESSION
    finite_loss_pass = sample_pass and k0_pass and k8_pass and regression_pass
    return PTICRNFiniteLossSummary(
        boundary_count=len(rows),
        fragment_counts=fragment_counts,
        action_count=action_count,
        action_fraction=action_fraction,
        mean_k0_gain=mean_k0,
        k0_fifth_percentile=fifth_k0,
        mean_k8_gain=mean_k8,
        k8_bootstrap_lower_95=lower_k8,
        k8_positive_boundary_fraction=positive_fraction,
        k8_fragment_means=fragment_means,
        k8_positive_fragment_means=positive_fragments,
        worst_individual_gain=worst_gain,
        sample_gate_pass=sample_pass,
        action_rate_gate_pass=action_pass,
        k0_gate_pass=k0_pass,
        k8_gate_pass=k8_pass,
        regression_gate_pass=regression_pass,
        finite_loss_gate_pass=finite_loss_pass,
    )


def analyze_authorized_pti_crn_campaign(
    store: CaptureObjectStore,
    *,
    campaign_index: CRNCampaignIndexRef,
    paired_outcomes: Sequence[CRNPairedOutcomeRef],
) -> PTICRNCampaignAnalysis:
    """Verify an exact closed campaign denominator, then compute its loss gate."""

    if not isinstance(campaign_index, CRNCampaignIndexRef):
        raise TypeError("campaign_index must be CRNCampaignIndexRef")
    if isinstance(paired_outcomes, (str, bytes)) or not isinstance(
        paired_outcomes, Sequence
    ):
        raise TypeError("paired_outcomes must be a sequence")
    outcome_refs = tuple(paired_outcomes)
    if any(not isinstance(ref, CRNPairedOutcomeRef) for ref in outcome_refs):
        raise TypeError("paired_outcomes must contain CRNPairedOutcomeRef values")

    index = load_crn_campaign_index(store, campaign_index)
    expected = {ref.manifest.sha256: ref for ref in index.plans}
    loaded_by_plan = {}
    for ref in outcome_refs:
        outcome = load_crn_paired_outcome(store, ref)
        if not _same_manifest(
            outcome.campaign_index_ref.manifest, campaign_index.manifest
        ):
            raise CRNAuthorityError("paired outcome belongs to another campaign")
        plan_digest = outcome.plan_ref.manifest.sha256
        if plan_digest not in expected:
            raise CRNAuthorityError("paired outcome plan is outside the campaign")
        if plan_digest in loaded_by_plan:
            raise CRNAuthorityError("paired outcomes repeat a campaign plan")
        loaded_by_plan[plan_digest] = (ref, outcome)
    if set(loaded_by_plan) != set(expected):
        raise CRNAuthorityError(
            "paired outcomes do not cover the exact precommitted campaign denominator"
        )

    observations: list[PTICRNFiniteLossObservation] = []
    candidate_required = set(CRN_ACTION_CAPABILITIES)
    control_required = {"worker_restore", "crn_train_k8"}
    for plan_ref in index.plans:
        plan = load_authorized_crn_plan(store, campaign_index, plan_ref)
        outcome_ref, outcome = loaded_by_plan[plan_ref.manifest.sha256]
        candidate = load_authoritative_pti_action(store, plan.candidate_ref).action
        if not candidate_required <= set(candidate.required_capabilities):
            raise CRNAuthorityError(
                "PTI candidate action lacks fixed CRN restore/future-eight capabilities"
            )
        if not control_required <= set(plan.stock.required_capabilities):
            raise CRNAuthorityError(
                "stock control action lacks fixed CRN restore/future-eight capabilities"
            )
        observations.append(
            PTICRNFiniteLossObservation(
                plan=plan_ref.manifest,
                outcome=outcome_ref.manifest,
                boundary_sha256=plan.boundary_ref.manifest.sha256,
                fragment_id=plan.boundary.identity.fragment_id,
                commit_seq=plan.boundary.identity.commit_seq,
                candidate_used_nonstock=candidate.action_kind == "nonstock",
                k0_gain=outcome.stock.k0_loss - outcome.candidate.k0_loss,
                k8_gain=outcome.stock.k8_loss - outcome.candidate.k8_loss,
            )
        )
    canonical = tuple(observations)
    return PTICRNCampaignAnalysis(
        campaign_index=campaign_index,
        observations=canonical,
        summary=compute_frozen_pti_finite_loss_gate(canonical),
    )


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "PTICRNCampaignAnalysis",
    "PTICRNFiniteLossObservation",
    "PTICRNFiniteLossSummary",
    "analyze_authorized_pti_crn_campaign",
    "compute_frozen_pti_finite_loss_gate",
]
