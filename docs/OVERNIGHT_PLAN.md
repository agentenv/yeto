# Overnight Autonomous Operation Plan (2026-07-13, user asleep)

## Mandate
Take care of the GPUs, auto-continue the research, use Codex freely for
consulting. Operating unsupervised — so: FINISH the queued work, do NOT
launch new expensive experimental programs, protect against runaway spend.

## Guardrails (hard rules while user sleeps)
1. **No new expensive experiments.** Let queued runs finish; consolidate.
   Only launch a new GPU run if it's a cheap, already-designed, staged one
   (cheb-sgd exp2-44, wsub exp2-43) AND capacity/budget is comfortable.
2. **No p5/p5e or >$8/hr single instances** unsupervised.
3. **Cost guard active** (cost_guard.sh): Verda balance floor $8 -> delete
   all Verda nodes; AWS P-instance >5h -> terminate. Balance was $65.72
   (~4.7h Verda runway); AWS bills separately.
4. **Delete nodes promptly on workstream completion** (coordinator + me).
5. **Wind down gracefully** when the queue drains or balance nears floor —
   don't keep nodes idling.

## Priorities (finish, in order)
1. **mediation (exp2-39)** — paper-critical causal closure. 4/7 arms. FINISH,
   collect verdict (norm-matched inner-LR + geometry collapse + threshold).
2. **barrier crossover (exp2-45, AWS)** — top reviewer-objection remover
   (does the crossover hold under true lockstep DiLoCo?). Collect verdict.
3. **curv T4 (exp2-42, AWS)** — does curvature-aware momentum beat SGD?
4. **bake-off (exp2-41)** — worker-SNR/block-RMS/block-Yogi vs SGD. First
   cell wsnr-h64=1.3594 (~tie, slightly worse). Collect the extreme-H cells.
5. **iso (exp2-40)** — Iso-C+SGD vs SGD.
6. If budget/capacity comfortable and above are landing: cheb-sgd (exp2-44)
   — the diagnostic-recommended flagship first optimizer. Staged in
   $SP/cheb_ready.md.

## Consolidation (the real overnight deliverable)
As each workstream completes, fold results into paper/main.md (coordinator
agent owns this) and $SP/tonight_consolidated_report.md. Use Codex
(gpt-5.6-sol, `< /dev/null`) to: analyze bake-off/curv/barrier results,
draft/tighten paper sections, adversarially check any new claim.

## Morning deliverable for the user
1. A written summary at $SP/MORNING_SUMMARY.md: every verdict (mediation,
   barrier, curv, bake-off candidates, iso, cheb if run), what it means,
   updated paper status, total spend (Verda + AWS), and any decision points.
2. All nodes cleaned up (zero running instances) unless a run is legitimately
   still finishing and within budget.
3. paper/main.md placeholders filled with real numbers.

## Key facts / references
- Ship branches: rho-adaptive-v2 (most code; has a curv/wsub collision —
  cleanup deferred), rho-adaptive-curv b189540 (clean curv).
- Done + committed: 12 Lean theorems, dynamics diagnostic (Chebyshev/Krylov),
  all optimizer code, paper draft.
- SGD-0.28 refs (seed223 sync): H16=1.351855, H64=1.357837, H256=1.380456.
- Noise floor ~0.009; a <0.009 win is not a real improvement.
- 15-min global status cadence: gstatus.sh.
