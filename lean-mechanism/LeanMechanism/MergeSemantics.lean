import Mathlib

/-!
# Merge anchor semantics

This file models the three merge semantics relevant to anchor drift: a barrier
merge, a streaming merge anchored at the learner's base version, and a
streaming merge anchored at the syncer's current version.
-/

namespace LeanMechanism

noncomputable section

/-- The parameter vectors visible at one worker merge. -/
structure MergeEvent where
  base : ℝ × ℝ
  current : ℝ × ℝ
  upload : ℝ × ℝ

/-- Current-anchor pseudo-gradient: syncer current parameters minus the upload. -/
def caDelta (e : MergeEvent) : ℝ × ℝ :=
  (e.current.1 - e.upload.1, e.current.2 - e.upload.2)

/-- Version-matched pseudo-gradient: the learner's base parameters minus the upload. -/
def vmDelta (e : MergeEvent) : ℝ × ℝ :=
  (e.base.1 - e.upload.1, e.base.2 - e.upload.2)

/-- Movement of the syncer's global parameters since the learner's base version. -/
def anchorDrift (e : MergeEvent) : ℝ × ℝ :=
  (e.current.1 - e.base.1, e.current.2 - e.base.2)

/-- A barrier merge: the syncer's current global is exactly the learner's base. -/
def Barrier (e : MergeEvent) : Prop := e.current = e.base

/-- The current-anchor pseudo-gradient is the version-matched pseudo-gradient
plus anchor drift, component by component. -/
theorem caDelta_eq_vmDelta_add_drift (e : MergeEvent) :
    caDelta e =
      ((vmDelta e).1 + (anchorDrift e).1, (vmDelta e).2 + (anchorDrift e).2) := by
  apply Prod.ext <;> simp [caDelta, vmDelta, anchorDrift]

/-- A barrier leaves no anchor drift. -/
theorem barrier_drift_zero (e : MergeEvent) (h : Barrier e) :
    anchorDrift e = (0, 0) := by
  unfold Barrier at h
  simp [anchorDrift, h]

/-- Under a barrier, current-anchor and version-matched pseudo-gradients agree. -/
theorem barrier_ca_eq_vm (e : MergeEvent) (h : Barrier e) : caDelta e = vmDelta e := by
  unfold Barrier at h
  simp [caDelta, vmDelta, h]

/-- The two streaming delta semantics agree exactly when anchor drift is zero. -/
theorem ca_eq_vm_iff_no_drift (e : MergeEvent) :
    caDelta e = vmDelta e ↔ anchorDrift e = (0, 0) := by
  constructor
  · intro h
    have hcoords :
        e.current.1 - e.upload.1 = e.base.1 - e.upload.1 ∧
          e.current.2 - e.upload.2 = e.base.2 - e.upload.2 := by
      simpa only [caDelta, vmDelta, Prod.mk.injEq] using h
    apply Prod.ext
    · simp only [anchorDrift]
      linarith [hcoords.1]
    · simp only [anchorDrift]
      linarith [hcoords.2]
  · intro h
    have hcoords : e.current.1 - e.base.1 = 0 ∧ e.current.2 - e.base.2 = 0 := by
      simpa only [anchorDrift, Prod.mk.injEq] using h
    apply Prod.ext
    · simp only [caDelta, vmDelta]
      linarith [hcoords.1]
    · simp only [caDelta, vmDelta]
      linarith [hcoords.2]

end

end LeanMechanism
