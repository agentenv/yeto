# Prospective operational amendment: concurrent 135M audit blocks

Status: **superseded in part by tuned-baseline audit preregistration revision 1.2 before any A1/A3/A4 scientific attempt**.

Authority was recorded in the campaign handoff channel at
`/private/tmp/audit-135m-note.md` after the post-reboot controller had proved
that its first two A1 capacity attempts created no provider identity, wrote no
campaign object, incurred `$0.00`, and exposed no scientific outcome.  This
amendment changes capacity scheduling only.  The frozen audit stages, cells,
seeds, treatment grids, pairing identities, blinding rules, gates, decision
rules, hard cost ceilings, and analysis remain exactly those in
`experiment-specs/tuned-baseline-audit-prereg.{json,md}`.

## Motivation

The inherited four-slot executor chose three VMs for a six-cell paired block,
then ran blocks serially.  The first amendment allowed up to five independent
blocks at once.  After four distinct launch-machinery defects, operator
authorization 2 prospectively reduced that surface to at most two independent
blocks at once on reviewed Spot `a2-highgpu-1g` capacity: three VMs per atomic
block, at most six campaign A100s, with the existing project-global ceiling of
sixteen unchanged.

## Frozen concurrent binding

- One atomic audit block remains indivisible and loss-blind until every arm in
  that block is terminal.  For the paired tuning blocks this is three paired
  arms (six cells), dispatched in two deterministic batches over a three-VM
  lane.
- The audit-only logical-slot universe is `v0` through `v5`.  Legacy P0/P1,
  E1/E4, and non-audit scheduling retain the original `v0` through `v3`
  universe and byte-identical plan construction.
- At most two atomic blocks may overlap.  Each block receives exactly three
  disjoint logical slots.
- The block lane is a pure deterministic function of only:
  1. the frozen `audit_135m_design_contract_hash`;
  2. the planned block index; and
  3. the exact loss-blind available-slot set.
  The available slots are hash-ranked and partitioned into three-slot lanes;
  a contract-hash permutation of lane indices is indexed by
  `planned_block_index mod lane_count`.  Any contiguous batch of at most two
  blocks therefore receives disjoint lanes without consulting outcomes.
- Capacity may degrade to fewer concurrent blocks only from provider census,
  landed machine shape, preemption, or the hard cost/global-A100 rails.  Loss,
  divergence, checkpoint content, and hidden evaluation state are forbidden
  scheduling inputs.
- Attempts retain unique actual-wave indices and record their concurrent batch
  index, complete batch slot set, and three-slot lane.  Independent blocks may
  overlap; retries remain whole-block, loss-blind, fresh-attempt retries.

## Capacity and lifecycle rails

- Spot only.  On-demand fallback remains forbidden.
- Before every provider launch, the controller re-censuses the whole project
  and includes concurrently pending probes in the ceiling calculation.
- The preferred shape is direct Spot `a2-highgpu-1g`, with survival-weighted
  zones and a whole-project census immediately before every provider launch.
- Total attached A100 equivalents, including foreign workloads, may never
  exceed sixteen.  The live watchdog still exact-ID deletes campaign-owned
  generations if the global count crosses the ceiling.
- Initial assembly, including width expansion, remains bounded by 480 seconds
  from the first READY VM.  Surplus VMs are exact-ID finalized before a smaller
  final batch so no accelerator idles through another block.
- Every physical generation still has an exact numeric instance and disk ID,
  ownership nonce, create-only provider/partial/lifecycle evidence, exact-ID
  teardown, and zero-accelerator proof.  Preemption still creates a fresh
  physical generation and fresh attempt namespace.
- The revision 1.2 hard loss-blind kills are A1 `$140.00`, A3 `$31.18`, and A4
  `$138.21`.  Each stage also has a separate `$40.00` cumulative pre-science
  abort-burn kill.

## Reboot-restoration fix

The audit controller now points the dynamically loaded reviewed R0 backend at
the executing repository for `scripts/run_parallel_phase_map.py`.  Fresh
replacement rendering therefore no longer depends on an untracked copy under
`/private/tmp/yeto-p1r0-launcher/scripts/`.  Runtime packets already carry the
tracked script from the source bundle; this change makes controller-side fresh
generation rendering equally self-contained.

## Non-effects

This amendment does not authorize opening hidden objects, unstarted-stage
seeds, incomplete hidden batches, or scientific losses.  It does not change a
single scientific command, model/data hash, work budget, pairing identity,
outcome rule, gate, or analysis.  Revision 1.2 changes only the prospective
cost and launch-surface controls described above.
