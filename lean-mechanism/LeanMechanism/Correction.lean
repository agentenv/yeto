import Mathlib
import LeanMechanism.MergeSemantics
import LeanMechanism.AnchorDrift
import LeanMechanism.Basic

/-!
# Base-version correction

When learners upload their base version, the syncer can remove anchor drift and
recover the exact version-matched delta before applying the outer optimizer.
-/

namespace LeanMechanism

noncomputable section

/-- In the local-delta convention, adding anchor drift back to the
current-anchor delta exactly recovers the true version-matched local delta. -/
theorem correction_recovers_vm (e : MergeEvent) :
    ((localCa e).1 + (anchorDrift e).1, (localCa e).2 + (anchorDrift e).2) =
      localVm e := by
  apply Prod.ext <;> simp [localCa, localVm, anchorDrift]

/-- In the pseudo-gradient convention, subtracting anchor drift from the
current-anchor pseudo-gradient exactly recovers the version-matched one. -/
theorem correction_recovers_vm_pg (e : MergeEvent) :
    ((caDelta e).1 - (anchorDrift e).1, (caDelta e).2 - (anchorDrift e).2) =
      vmDelta e := by
  apply Prod.ext <;> simp [caDelta, vmDelta, anchorDrift]

/-- Once the corrected pseudo-gradient is known to be the descending
version-matched step, applying that corrected step preserves the same safety
property. -/
theorem correction_safe (a b c : ℝ) (θ δca d : ℝ × ℝ)
    (hvm : dL a b c θ (δca.1 - d.1, δca.2 - d.2) < 0) :
    dL a b c θ (δca.1 - d.1, δca.2 - d.2) < 0 :=
  hvm

/-- If `delta_ca = delta_vm + d`, correcting by `-d` gives exactly the
version-matched loss change. Thus a positive three-arm control result calls for
base-version re-anchoring, not a more elaborate optimizer: it implements the
same algebra as `current - upload' = base - upload`. -/
theorem corrected_step_eq_vm_step (a b c : ℝ) (θ δvm d : ℝ × ℝ) :
    dL a b c θ ((δvm.1 + d.1) - d.1, (δvm.2 + d.2) - d.2) =
      dL a b c θ δvm := by
  have hcorrected :
      ((δvm.1 + d.1) - d.1, (δvm.2 + d.2) - d.2) = δvm := by
    apply Prod.ext <;> simp
  rw [hcorrected]

end

end LeanMechanism
