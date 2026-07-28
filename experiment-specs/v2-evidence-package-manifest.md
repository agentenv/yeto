# V2 evidence-package manifest

This manifest audits the evidence claimed by the newer frozen paper text at
`/private/tmp/frozen.txt` (SHA-256
`4924f450675d1967de79fb797cae871d45a4519dbea4f62b9ce0ee658f2a77bc`).
The checked-in `paper/main.md` is an older state and is not the binding source
for this audit.

The machine-readable inventory is
`experiment-specs/v2-evidence-package-manifest.json`. It records canonical
locations, exact SHA-256 values, Mac availability, CPU-archive availability,
and provenance qualifications. The archive being audited is the non-Git
directory `c@65.19.161.135:~/h200-evac/`.

## Canonical readouts

| Study | Canonical SHA-256 | Mac | CPU evacuation | Disposition |
|---|---|---|---|---|
| G1 pilot | `c2dcd6b9...cae2b` | number-audit present | missing | not evaluable |
| G1V2 replication | `5d4eed96...2fc8` | number-audit present | missing | not evaluable |
| G3 intervention | `d4a3cde6...44c8` | number-audit present | missing | component criteria not evaluable |
| G4 initial 1.7B | `f2e70767...cded` | number-audit present | missing | not evaluable |
| G4B extension | `d58a05c4...be97` | number-audit present | missing | not evaluable |
| G4C strict original | `5f4eaf3c...d66` | present | missing | 6,804/10,000 strict-valid; not evaluable |
| G4C amended | `16bab85d...aa` | number-audit present | missing | pass, with amendment-timing caveat |
| G5 original operator audit | `9251d78a...3b` | present | missing | analyzer raised after detecting non-evaluability |
| G5B SNOO regrid | `60fd6549...e5b` | number-audit present | missing | complete; formal worse, estimator audit separate |
| G6 factorial | `7e7e9cd8...af8c` | present | missing | pass |
| G8 tuned-loss diagram | `9884f077...96c` | present | **present** | complete/evaluable |
| G9 joint scale | `4d42a5f1...bcaf` | bytes missing; ledger records SHA/results | **missing canonical** | joint fail |
| G10 fresh transfer | `46adf7b8...386f` | present | missing | `PENALTY_NULL` |
| G12 heavy-ball | `0162dbdd...ae5` | present | missing | pass |
| G13 Pythia initial | `be69b796...ef5a` | present | missing | 1/6 curves accepted; not evaluable |
| G13B Pythia regrid | `868d8aaa...f48` | present plus byte-identical rerun | missing | 3/6 accepted, 0/10,000; not evaluable |

The CPU directory does contain `g9-readout.json` with SHA
`fe49acaf...fcf`, but it has only 16/22 evidence records and no completed 7B
gate. It also contains an out-of-gate 1.7B supplement with SHA
`b572c017...48a`. Neither is the canonical joint G9 readout. The resident
`analyze_v9.py` is likewise stale (`efc18c81...ba8`) rather than the paper's
canonical terminal analyzer (`ef140f0a...ea1`).

G7/reduced-27B and G11 have no canonical readout: the frozen fold correctly
records them as running/deferred rather than analyzed results.

## Canonical analyzer inventory

| Gate/path | SHA-256 | Canonical source | CPU evacuation |
|---|---|---|---|
| G1 `scripts/analyze_e1.py` | `b8a3d542...bef1` | day-1 note only; bytes absent from reachable Git | missing |
| G1V2 `analyze_e1v2.py` | `41f7dd17...1dd` | Git `44359f5` | missing |
| G3 `analyze_v3.py` | `f608bebe...24e7` | Git `fa7206b` | missing |
| G4 `analyze_v4.py` | `654eb63d...0471` | Git `5abf65d` | missing |
| G4B `analyze_v4b.py` | `9a6bd411...ab5` | Git `9421298` | missing |
| G4C strict `analyze_v4c.py` | `b8e5470d...f27` | Git `848a18d` | missing |
| G4C amended `analyze_v4c.py` | `612d11b6...c79` | Git `ea5f034` | missing |
| G5 `analyze_v5.py` | `e574294d...a97e` | Git `63ba780` | missing |
| G5B `analyze_v5b.py` | `b6ed0959...e0ef` | Git `8316d86` | missing |
| G6 `analyze_v6.py` | `8016e03c...10d6` | amended Git `2f1d2ae` | missing |
| G7 `analyze_v7.py` | `c8351890...8f75` | v7 branch | missing |
| G8 `analyze_v8.py` | `00166147...699` | v8 branch | missing |
| G9 terminal `analyze_v9.py` | `ef140f0a...ea1` | Git `3e28f68` | stale different bytes only |
| G10 `analyze_v10.py` | `a5f32627...ff76` | Git `c2ce9eb` | missing |
| G11 `analyze_v11.py` | `74477e0d...e960` | Git `c310dc6` | missing |
| G12 `analyze_v12.py` | `e9aaa61c...f9b` | Git `c310dc6` | missing |
| G13 `analyze_v13.py` | `b939e8b7...23b` | Git `c310dc6` | missing |
| G13B `analyze_v13b.py` | `baf1cc89...927` | Git `c2893c6` | missing |

G12/G13/G13B also depend on shared core `scripts/tonight85_analysis.py`, SHA
`810256b6...53cd`; it is absent from the CPU archive.

The paper's separate two-parameter reanalysis is not the preregistered G3
analyzer. Its analyzer is
`/private/tmp/number-audit.HmdmkU/two-param-analysis/analyze_twoparam.py`, SHA
`740b9f7b...6562`, with machine results `22d18796...fb22`, master eta table
`610f8e30...d649`, master D table `f1b132a5...7427`, and report
`765c9ce0...f77`. All are missing from the CPU evacuation.

## The 2,238-versus-2,244 adjudication

The discrepancy is two deterministic analyses, not six lost or mis-added
draws:

| Path | RNG | Analyzer | Invalid T20 corrected refits | Role |
|---|---|---|---:|---|
| preregistered G3 | Python `random.Random(20260724)` | `f608bebe...24e7` | **2,244** | binding registered decision |
| paper reanalysis | NumPy `default_rng(20260724)` | `740b9f7b...6562` | **2,238** | post-hoc paper analysis |

The numeric seed is the same, but Python's MT wrapper and NumPy's generator
produce different ordered resamples. The paper currently prints 2,238. It
should explicitly cite that post-hoc path, or use 2,244 when describing the
preregistered decision. The CPU archive contains neither path nor an
adjudication record; this manifest is the first explicit linkage.

## Lean theorem map

The following paper-mapped sources are present in Git but entirely absent from
the CPU archive:

| Source | SHA-256 | Paper role |
|---|---|---|
| `FiniteHorizonOuter.lean` | `e820e78a...93b` | Theorem 2.1 / Corollary 2.2 |
| `QuadraticAlignment.lean` | `a7da4a51...ad9` | Theorem 2.3 |
| `StochasticBuffer.lean` | `582bc7e6...720` | Theorem 2.5 |
| `CorrectionCosts.lean` | `22c90d9c...4ee` | normalization variance cost |
| `ElasticInvariance.lean` | `d4277a1a...c4e` | cadence invariance |
| `NadamEquivalence.lean` | `c0fa81fa...64d0` | constant-product identity |
| `AgeCollapse.lean` | `f237641d...f8b5` | dimensionless-age bounds |

`KappaDrift.lean`, SHA `b286dc3c...99f`, is additionally required by the
mechanism chronicle but is absent from this branch as well as the CPU archive.
Its canonical copy is on `mechanism/lane-d-lean-formalizer@fd8568d` and in
`/private/tmp/yeto-h200-mechD`.

The machine manifest also enumerates every supporting Lean source and hash:
`Basic`, `Correction`, `AnchorDrift`, `MergeSemantics`, `TransferPenalty`,
`PhaseBoundary`, `Counterexamples`, and the root import file.

## Mechanism chronicle gaps

- Round 1 has three hashed lane analyzers/results on isolated Mac worktrees,
  but no single canonical round-1 adjudication JSON and nothing in the CPU
  archive.
- Round 2's analyzer (`4902dfff...6b16`) and result (`28358ada...b0e7`) exist
  only as untracked files in the original dirty Mac worktree; they are absent
  from this clean branch and the CPU archive.
- Round 3 analyzer/result (`41165a65...ef9`, `5b2a1b1e...f2b8`) and Round 3C
  analyzer/result (`e179fd64...8e6`, `86045f15...865`) are in Git but absent
  from the CPU archive.
- Round 3B has no machine-readable result file. Its `VOID` survives only in
  `/private/tmp/h200-mechR3-note.md`, along with the frozen protocol, launcher,
  and failure-log hashes.

## Bottom line

The current CPU evacuation is a partial operational archive. It is not the
complete released artifact archive described by the paper's reproducibility
statement. The canonical G8 readout and the sealed G9 predictions are present;
most canonical readouts, every canonical analyzer, every Lean source, the
completed G9 joint readout, derived tables, and mechanism adjudication files
are not. No stale or partial file should be substituted for a missing
canonical SHA.
