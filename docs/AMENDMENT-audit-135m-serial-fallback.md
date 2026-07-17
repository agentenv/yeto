# Prospective operational amendment: serial fallback for the 135M audit

Status: **triggered for A1 after the retired r9 campaign and before any A1/A3/A4 scientific attempt**.

## Authority and trigger

Operator Authorization 2, recorded in `/private/tmp/audit-135m-note.md`, prospectively required the parallel path to stop after a fifth distinct launch-machinery condition and named the proven serial P1 controller as the fallback. After A1D r9 terminated before science, the operator record added `FIFTH-CONDITION FALLBACK TRIGGERED — SERIAL MODE` and explicitly authorized one Spot VM at a time.

The r9 retirement evidence is recorded locally at:

`/private/tmp/audit-135m-r12-40f3f46/stages/A1/A1-R9-PARALLEL-RETIREMENT-STATUS.md`

No scientific attempt had started when this fallback was triggered. The parallel A1 path, every r9 identity, and the r9 artifact prefix are retired.

## Frozen serial binding

- The scientific randomization plan, cell IDs, commands, model/data hashes, seeds, work budgets, evaluation modes, analysis roles, outcome rules, gates, and cost ceilings are unchanged.
- The roster's registered block order is retained exactly.
- One logical slot, `v0`, is available to science. At most one Spot `a2-highgpu-1g` VM may be READY or scientifically active at a time.
- Every registered block is executed in its deterministic within-block order on that one VM. No two scientific attempts overlap.
- A block remains loss-blind until every registered cell in the block has a terminal scientific outcome. Public status may expose only mechanical progress and terminal status vocabulary, never a loss.
- A completed or diverged cell is banked permanently and is never rerun merely because a later cell in the same block suffers infrastructure failure.
- An infrastructure-failed cell receives a fresh attempt of that cell only. The retry uses the same frozen command, model, seed, data order, and work budget, with no checkpoint, optimizer state, tape, result, or other scientific state reused.
- If the active VM is Spot-preempted, its exact generation is finalized by numeric instance and boot-disk IDs before a fresh physical generation is launched. The replacement continues at the first unresolved cell.
- The normal case uses one physical generation for the complete block. A provider preemption may cause the unresolved suffix to continue on a fresh generation; completed cells remain immutable evidence.
- After a block is complete, its VM is exact-ID finalized before the next block's VM is launched, except that a final training survivor may be retained briefly for the already-registered all-at-once hidden evaluation gate.

## Capacity, cost, and evidence rails

- Spot only; on-demand fallback is forbidden.
- Survival-weighted zone order and prelaunch cost eligibility remain registered.
- The whole project is re-censused immediately before every launch. Total attached A100-equivalent capacity, including foreign workloads, remains capped at sixteen.
- The protected instance `3908640733128066700` and foreign resources are never mutated.
- File-streamed bootstraps, per-generation ownership nonces, create-only evidence, and fresh artifact prefixes remain mandatory.
- A READY accelerator must receive science or begin exact teardown before 600 seconds of idle time.
- Work-evidence validation is required before a cell can be banked.
- Access-blind objects remain unopened until the registered stage gate. Deferred evaluation still uses train-all-then-evaluate-all, one complete hidden batch, one seal, and one shared unblind.
- The corrected A1 `$140.00` hard ceiling, A3 `$31.18` hard ceiling, A4 `$138.21` hard ceiling, and separate `$40.00` per-stage pre-science abort-burn kill remain unchanged.
- Every physical generation must end with exact numeric instance/disk `NOT_FOUND` proofs and a zero attached-accelerator proof.

This amendment changes execution mechanics only. It does not authorize outcome-dependent scheduling, early unblinding, partial hidden evaluation, omission of failures, or any change to the preregistered scientific contract.
