# Outer-Optimizer Bake-off — Exact Results

Every alternative outer optimizer tested against memoryless SGD-0.28 on the
DiLoCo momentum poison (Qwen3.5-9B, LoRA, strict-quorum streaming). Δ =
candidate − SGD-0.28 (negative = better than SGD). Noise floor ≈ 0.009.
**Product gate:** paired win **> 0.018** on ≥2 workloads AND never worse than 1
noise floor on any workload; no per-H/rank/inner-LR tuning.

## Δ vs SGD-0.28 (per horizon)

| Candidate | H16 | H64 | H256 | other cell | pattern |
|---|---|---|---|---|---|
| worker-SNR (consensus)      | **−0.0019** (win) | +0.0016      | **+0.0157 ✗** | —                     | short-H helps, long-H breaks |
| block-RMS (2nd-moment)      | **+0.0093 ✗**     | +0.0026      | **−0.0095** (win) | —                  | opposite tilt: long-H helps |
| block-Yogi (2nd-moment)     | +0.0086           | +0.0012      | n/a¹          | —                     | ~flat |
| curvature-aware momentum    | **+0.0303 ✗**     | +0.0055 ✗    | **−0.0103** (win) | innerlr-hi **+0.0283 ✗** | H256-only; loses where predicted to help |
| Iso-C (spectral)            | **−0.0078** (win) | +0.0001 (tie)| **+0.0183 ✗** | —                     | short-H helps, long-H breaks |
| capnest v2.1 (scalar cap)   | +0.0236 ✗         | +0.0179 ✗    | −0.0040 (tie) | —                     | H-stable, never a real win |
| dynamic-H                   | —                 | —            | —             | best controller **+0.011 behind** tuned SGD | SGD wins |

**Gate result: NONE passes.** No candidate reaches a >0.018 win on ≥2 workloads,
and every one is worse than SGD by ≥1 noise floor at some horizon → **ship
memoryless SGD-0.28.** The unifying finding: each candidate wins in exactly the
horizon regime the dynamics diagnostic predicts (spatial/consensus → short-H;
second-moment/spectral/curvature → long-H), and none is uniformly better — there
is no free lunch in the outer optimizer at this cadence.

## Absolute eval-loss (raw numbers)

| Candidate | H16 | H64 | H256 | innerlr-hi-H64 |
|---|---|---|---|---|
| SGD-0.28 ref            | 1.351855 | 1.357837 | 1.380456 | 1.378648² |
| worker-SNR              | 1.349991 | 1.359412 | 1.396235 | — |
| block-RMS               | 1.361138 | 1.360397 | 1.371002 | — |
| block-Yogi              | 1.360425 | 1.359039 | —¹       | — |
| curvature-aware momentum| 1.382600 | 1.362800 | 1.369068 | 1.406944 |
| Iso-C                   | 1.351354 | 1.360359 | 1.398720 | — |
| capnest v2.1            | 1.375451 | 1.375779 | 1.376484 | — |

**dynamic-H:** fixed Nesterov-μ0.9 destabilizes at horizon switches (post-switch
direction cosine **0.374** vs SGD **1.00**, capnest 0.873); tuned SGD wins, best
controller trails by **+0.011**.

## capped-nesterov-wsub (EXP2.43, directional successor to curv)

The directional worker-subspace variant, judged head-to-head against curv:
- wsub-h16 = **1.3787** vs SGD-0.28 1.351855 → **+0.027** (loses in the poison corner).
- Codex validated the implementation as faithful (correct E_b, b_perp, per-worker
  s_i, ¼-power cap) and confirmed the run used `--outer-lr 0.147`, so the result
  is genuine. Its scale-flawed disagreement cap barely engages at H16 (E_b is
  quartic in delta scale → inverted binding), and the residual aligned
  capped-Nesterov still loses to SGD. Remaining arms in progress.

## Notes
¹ block-Yogi H256 lost to node preemption mid-run; conclusion robust without it.
² curvature-aware momentum and Iso-C are compared against their **same-node** SGD
  anchors (curv: 1.352342 / 1.357306 / 1.379344 / 1.378648; Iso-C: 1.359198 /
  1.360265 / —) because they ran at a different outer-LR/config than the H-sweep
  refs. All other Δ are vs the H-sweep SGD-0.28 refs above.

Sources: exp2-41 (worker-SNR/block-RMS/block-Yogi), exp2-42 (curvature-aware),
exp2-40 (Iso-C), exp2-36 (capnest v2.1), exp2-43 (wsub), dynamic-H screen.
Raw data: s3://yeto-exp-artifacts-533462777468-us-west-2/probecommit-resume-20260710/exp2-*.
