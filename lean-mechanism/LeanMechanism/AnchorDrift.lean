import Mathlib
import LeanMechanism.MergeSemantics

/-!
# Anchor drift identities and counterexample states

The local-delta convention in this file is deliberately named separately from
the pseudo-gradient convention in `MergeSemantics`: local deltas are upload
minus anchor, whereas pseudo-gradients are anchor minus upload.
-/

namespace LeanMechanism

noncomputable section

/-- Version-matched local delta: upload minus the learner's base parameters. -/
def localVm (e : MergeEvent) : ℝ × ℝ :=
  (e.upload.1 - e.base.1, e.upload.2 - e.base.2)

/-- Current-anchor local delta: upload minus the syncer's current parameters. -/
def localCa (e : MergeEvent) : ℝ × ℝ :=
  (e.upload.1 - e.current.1, e.upload.2 - e.current.2)

/-- In the local-delta convention, current anchoring subtracts anchor drift
from the true version-matched local delta. -/
theorem localCa_eq_localVm_sub_drift (e : MergeEvent) :
    localCa e =
      ((localVm e).1 - (anchorDrift e).1, (localVm e).2 - (anchorDrift e).2) := by
  apply Prod.ext <;> simp [localCa, localVm, anchorDrift]

/-- **R1.** Barrier synchronization implies zero anchor drift. -/
theorem R1_barrier_implies_no_drift (e : MergeEvent) (h : Barrier e) :
    anchorDrift e = (0, 0) :=
  barrier_drift_zero e h

/-- **R2.** With no anchor drift, current-anchor and version-matched
pseudo-gradients are exactly equal. -/
theorem R2_no_drift_ca_eq_vm (e : MergeEvent) (h : anchorDrift e = (0, 0)) :
    caDelta e = vmDelta e :=
  (ca_eq_vm_iff_no_drift e).2 h

/-- **R3.** Quorum constrains participation, not versions. A full-quorum commit
can still merge a worker after the server global advanced beyond its base
version, producing nonzero anchor drift. -/
theorem R3_quorum_not_no_drift :
    ∃ e : MergeEvent, ∃ base_version current_version : ℕ,
      base_version < current_version ∧ anchorDrift e ≠ (0, 0) := by
  refine ⟨⟨(0, 0), (1, 0), (0, 0)⟩, 0, 1, by norm_num, ?_⟩
  norm_num [anchorDrift]

/-- **R4.** Anchor drift comes from commit/broadcast latency overlap while a
worker continues training; it is independent of any externally injected delay. -/
theorem R4_zero_delay_not_no_drift :
    ∃ e : MergeEvent, ∃ injected_delay : ℝ,
      injected_delay = 0 ∧ anchorDrift e ≠ (0, 0) := by
  refine ⟨⟨(0, 0), (1, 0), (0, 0)⟩, 0, rfl, ?_⟩
  norm_num [anchorDrift]

/-- A committing worker together with the global version from which it trained. -/
structure Worker where
  base_version : ℕ

/-- **R5.** "All workers committed" is a participation property. Committing
workers can have different base versions, the source of per-worker anchor drift. -/
theorem R5_all_commit_not_same_version :
    ∃ w1 w2 : Worker, w1.base_version ≠ w2.base_version := by
  refine ⟨⟨0⟩, ⟨1⟩, ?_⟩
  norm_num

end

end LeanMechanism
