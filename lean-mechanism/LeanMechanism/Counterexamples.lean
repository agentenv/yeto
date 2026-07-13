import Mathlib
import LeanMechanism.Basic
import LeanMechanism.MergeSemantics

/-!
# Anchor-drift and momentum counterexamples

The exact rational examples below separate native transverse-momentum poison
from current-anchor contamination, then characterize the latter's one-step
threshold on a general two-dimensional quadratic.
-/

namespace LeanMechanism

noncomputable section

/-! ## A. Native momentum poison with no anchor drift -/

/-- The native momentum example is compatible with barrier semantics: its
illustrative merge event has exactly zero anchor drift. -/
theorem cexA_barrier_no_drift :
    anchorDrift ⟨(0, 0), (0, 0), (0, 0)⟩ = (0, 0) := by
  norm_num [anchorDrift]

/-- At the matched aligned learning rate, SGD strictly descends. -/
theorem cexA_sgd_descends : dL 1 100 0 (1, 0) (57 / 200, 0) < 0 := by
  rw [T2_sgd_descends]
  norm_num

/-- A transverse momentum buffer in the sharp direction flips the same aligned
step to ascent. This possibility already exists under original barrier DiLoCo:
zero drift, zero staleness, and zero overlap. It requires nonzero momentum and a
transverse buffer; it is not a claim that every training run realizes one. -/
theorem cexA_mom_ascends :
    0 < dL 1 100 0 (1, 0) (57 / 200 + 0, 0 + 243 / 2000) :=
  T2_aniso_mom_ascends_pos

/-! ## B. Anchor-drift poison with zero momentum -/

/-- The version-matched, memoryless step has exact loss change `-2`. -/
theorem cexB_vm_descends : dL 1 4 0 (2, 0) (2, 0) = -2 := by
  simp only [dL, Lq, Q]
  norm_num

/-- The version-matched step strictly descends. -/
theorem cexB_vm_descends_neg : dL 1 4 0 (2, 0) (2, 0) < 0 := by
  rw [cexB_vm_descends]
  norm_num

/-- Adding sharp-direction anchor drift changes the exact loss by `+5/2`. -/
theorem cexB_ca_ascends : dL 1 4 0 (2, 0) (2, 3 / 2) = 5 / 2 := by
  simp only [dL, Lq, Q]
  norm_num

/-- The current-anchor, memoryless step strictly ascends. -/
theorem cexB_ca_ascends_pos : 0 < dL 1 4 0 (2, 0) (2, 3 / 2) := by
  rw [cexB_ca_ascends]
  norm_num

/-- The two pseudo-gradients differ only by the pure anchor drift `(0, 3/2)`.
No momentum buffer is present (`mu = 0`), so this is distinct from example A.
Identical worker updates, quorum, data, and optimizer can therefore flip from
descent to ascent solely by changing the anchor used by the syncer. -/
theorem cexB_only_drift_differs :
    ((2 : ℝ), (3 : ℝ) / 2) = (2 + 0, 0 + 3 / 2) := by
  norm_num

/-! ## Parametric drift threshold -/

/-- Exact current-anchor loss gap at unit step and zero momentum.

The interaction is `dᵀH(delta_vm - theta)`, not `dᵀH(theta - delta_vm)`: this
sign is forced by this project's update convention `theta ↦ theta - u` and the
pseudo-gradient identity `delta_ca = delta_vm + d`. -/
theorem drift_gap_identity (a b c : ℝ) (θ δvm d : ℝ × ℝ) :
    dL a b c θ (δvm.1 + d.1, δvm.2 + d.2) - dL a b c θ δvm
      = Bil a b c d (δvm.1 - θ.1, δvm.2 - θ.2) + (1 / 2) * Q a b c d := by
  obtain ⟨θ1, θ2⟩ := θ
  obtain ⟨v1, v2⟩ := δvm
  obtain ⟨d1, d2⟩ := d
  simp only [dL, Lq, Q, Bil]
  ring

/-- Given a descending version-matched step, current anchoring flips it to
ascent exactly when the curvature-weighted drift gap exceeds the descent
margin. The relevant quantity depends on drift direction as well as magnitude:
a large drift in a flat direction can be harmless, while modest sharp-direction
drift can be catastrophic. Experiments should log this interaction, not only a
norm of `anchorDrift`. -/
theorem drift_flip_threshold (a b c : ℝ) (θ δvm d : ℝ × ℝ)
    (_hvm : dL a b c θ δvm < 0) :
    0 < dL a b c θ (δvm.1 + d.1, δvm.2 + d.2) ↔
      Bil a b c d (δvm.1 - θ.1, δvm.2 - θ.2) + (1 / 2) * Q a b c d
        > -dL a b c θ δvm := by
  have hid := drift_gap_identity a b c θ δvm d
  constructor <;> intro h <;> linarith

/-! ## Upgraded T3: system-faithful indistinguishability -/

/-- **Upgraded T3.** There exists an indistinguishable pair, not a worst-case
lower bound: at the same unit step (`eta = 1`), zero momentum (`mu = 0`), merged
delta, and buffer, a version-matched instance descends while a current-anchor
contaminated instance ascends. This is a one-step statement. The controller is
not given curvature, parameters, anchor drift, or base version. Since gain and
cosine are functions of `(delta, buffer)`, they are equal for this pair too. -/
theorem T3_upgrade_indistinguishable :
    ∃ (a1 b1 : ℝ) (θ1 : ℝ × ℝ) (a2 b2 : ℝ) (θ2 : ℝ × ℝ)
      (δ buf : ℝ × ℝ),
      buf = (0, 0) ∧
        dL a1 b1 0 θ1 δ < 0 ∧
          dL a2 b2 0 θ2 δ > 0 := by
  refine ⟨1, 1, (2, 3 / 2), 1, 4, (2, 0), (2, 3 / 2), (0, 0), rfl, ?_, ?_⟩
  · norm_num [dL, Lq, Q]
  · norm_num [dL, Lq, Q]

/-- Any controller whose complete input is only `(delta, buffer)` must return
the same action on the indistinguishable pair, although the correct one-step
descent classification differs. This existential theorem does not claim an
all-instances or long-horizon lower bound, and it permits no per-problem oracle
learning-rate input. -/
theorem T3_upgrade_controller_blind {α : Sort*}
    (C : (ℝ × ℝ) → (ℝ × ℝ) → α) :
    ∃ (a1 b1 : ℝ) (θ1 : ℝ × ℝ) (a2 b2 : ℝ) (θ2 δ buf : ℝ × ℝ),
      C δ buf = C δ buf ∧
        dL a1 b1 0 θ1 δ < 0 ∧
          0 < dL a2 b2 0 θ2 δ := by
  refine ⟨1, 1, (2, 3 / 2), 1, 4, (2, 0), (2, 3 / 2), (0, 0), rfl, ?_, ?_⟩
  · norm_num [dL, Lq, Q]
  · norm_num [dL, Lq, Q]

end

end LeanMechanism
