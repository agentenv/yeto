# Outer-muP v3 finite-horizon preregistration

**Program ID:** `outer-mup-v3-finitehorizon`

**Status:** `PREREGISTERED`, prospective for all scientific outcomes described below.

**Registered:** 2026-07-25.

**Machine-readable companion (authoritative):**
`experiment-specs/outer-mup-v3-finitehorizon-prereg.json`.

**Prior registrations:** v1 `outer-mup-2day-prereg.{json,md}` (closed
`STOP_G1_NOT_EVALUABLE_AUDIT_PAPER`, see `outer-mup-v1-closure.md`); v2
`outer-mup-v2-confirm-prereg.{json,md}` (json sha256
`0f769dbd4e6af8de7b47d9798baf104d02595f628b8e72c0f8a86ff956c8f6f3`, md sha256
`c2ce6b2074a943866f8d5281ba861319dff6a283c621f97df26aeb97b052b163`). No v1, v2, or
pilot result, evidence file, or sealed record may be modified by v3 activity.

## 0. Object of study and scoop check

Define the tuned-LR deviation factor `D(T,mu) = [eta*(mu)/eta*(0)]/(1-mu)`, where `T`
is the number of outer commits seen by one fragment's momentum buffer (`T = S/H`,
`H = 512` fixed, `S` = inner steps per learner).

**Pilot evidence (all measured, all disclosed, none reused as v3 evidence):**

- mu-sweep at `T=5` fits `1/(1-mu^T)` within 4.4% (mu .5/.8/.9/.95 ->
  D 1.019/1.524/2.441/4.610);
- T-scan at mu.9: `T=2: 4.11, T=5: 2.44, T=10: 1.68, T=20: 0.50 (EXTRAPOLATED_LOW,
  suspect ladder artifact), T=40: 1.00 (v2 H-scan), T=160: 1.04 (v2 H-scan)`;
- the code-true Nesterov final multiplier is `(1-mu^(T+1))/(1-mu)` (Lean-checked:
  `lean-mechanism/LeanMechanism/FiniteHorizonOuter.lean`,
  `codeTrue_terminalMultiplier_closed_form`).

Pilot files hash-bound in the JSON companion: lane B S-scan
(`h200-n1:/root/yeto-results-explore/{h200-pilot-note.md,summary.csv}`, exploratory
namespace `e1x-*`, seeds 401/409 — stays labeled EXPLORATORY), lane A code audit
(`docs/finite-t-law-verification.md`), lane E buffer analysis
(`h200-n1:/root/buffer-analysis/report.md`), lane D Lean mechanism.

**Scoop check (registered):** no published finite-horizon outer-LR law and no outer
bias correction anywhere — FedAdam omits it, SNOO applies raw outer Nesterov,
OpenDiLoCo uses raw torch SGD.

## 1. Registered intervention code

Syncer flag `--outer-bias-correction` (commit
`e7930ed4af9399f1f46d128e6afdf55ea1114180`; harness forwarding
`e6bb748bdd286465fdfd60188ae310ed7ccbc7c9`): at the fragment's t-th outer commit the
applied Nesterov step is divided by `(1 - mu^(t+1))`, the exact placement for the
code's `d_t = delta_t + mu*b_t` recursion, making every commit's constant-gradient
multiplier the steady-state `1/(1-mu)` and the accumulated multiplier
horizon-invariant per commit. OFF (default) is bit-identical to production (Rust
regression test); ON with mu=0 is also bit-identical (test). The correction is
consistent with the Lean code-true algebra by construction.

## 2. Execution and registration commits

v3 executes at **this registration commit** on `h200-n1`/`h200-n2` (the intervention
build requires it). The launch manifest binds the registration commit as
`source.git_commit` plus the raw sha256 of both contract files. No GCP/AWS. Pushes
from the Mac clone only.

## 3. Design — three arms at 135M / M4 / H512 (except ARM S: M1)

Fresh seeds `{301, 311, 313, 317, 331}` everywhere (5 per curve; **307 stays
reserved**). `training_seed = int(str(seed)+str(seed))`. All ladders are explicit
numeric grids in the JSON (`stages.V3FH.eta_grid_by_curve`), 4-point sqrt2 unless
noted. Per-S pilot `eta*(0)` centers (disclosed, incl. two extrapolated pilot fits and
one registered log-log extrapolation for S=20480): `S1024: .0789, S2560: .0443,
S5120: .0309, S10240: .0242, S20480: .0213`.

### ARM T — confirmatory T-scan, raw Nesterov (200 cells)

`S in {1024, 2560, 5120, 10240, 20480}` -> `T in {2, 5, 10, 20, 40}`; `mu in {0, .9}`.

- mu0 ladders centered on the per-S pilot `eta*(0)` above.
- mu.9 ladders centered on `eta*(0) * .1 * D_center` with **pilot OBSERVED**
  `D_center`: `T=2: 4.1, T=5: 2.44, T=10: 1.68, T=40: 1.0`.
- **T=20 exception:** a WIDE 2x-spaced ladder centered at `D=0.75`, spanning
  `D 0.2652–2.1213` — registered to adjudicate the pilot's suspect `0.50`
  (EXTRAPOLATED_LOW) point in either direction.

### ARM B — bias-correction intervention (200 cells)

Same S grid, `mu in {0, .9}`, `--outer-bias-correction ON`, ladders centered on
`eta*(0)*(1-mu)`.

**Registered prediction:** the corrected optimum is horizon-INVARIANT —
`|log2 D_corrected(T)| <= 0.15` at every T while the raw arm's D spans >= 4x.

ARM B mu0 cells are bit-identical by construction to ARM T mu0 cells (proven mu0
flag-ON identity); they are run anyway as a registered production A/A integrity
control and to give ARM B its own within-arm denominator.

### ARM S — SNOO deflation (60 cells)

`M=1`, `S=2560`, `H=512` (`T=5`), tokens 327,680:

| sub-arm | outer optimizer | ladder center | provenance |
|---|---|---|---|
| (a) | mu0 (eta=1 == plain AdamW) | `1.0` | baseline tuned fairly |
| (b) | Nesterov mu.9 (SNOO-style) | `.244 = 1.0 * .1 * 2.44` | pilot law transfer |
| (c) | mu0, eta_eff-matched | each (b) eta `* 4.0951 = (1-.9^5)/.1` | the law |

**Registered question:** does (b)'s gain over (a) survive against (c)?

## 4. Work contract

Tokens per M4 arm = `S*512` (S1024: 524,288 … S20480: 10,485,760); learner inner
steps = S; global outer commits = `4*T` (four fragments, round-robin); strict quorum,
barrier sync, version-matched anchor, fixed 512-step windows, zero injected
delay/jitter, `--rho-telemetry` on every cell. Every ARM B tape row must carry
`outer_bias_correction`; ARM T/S tapes must not. S=20480 cells cost 8x S=2560;
scheduling is longest-first with greedy cost balancing over the 16 H200 slots.

## 5. Frozen analyzer

`scripts/analyze_v3.py`, frozen before launch, raw-file sha256

```text
f608bebe770bda97733db0bbb3d2f5e317a1ea8d5aa676fd28627f7d91a024e7
```

recorded here and in the JSON (`frozen_analyzer`). Per-curve quadratic fits in
`log2(eta)` on the exact registered grids, paired five-seed 10,000-replicate
bootstrap (RNG seed 20260724), None-guarded throughout — exactly the v2 estimator
family. The supervisor runs the gates; the registrar never computes gates on
outcomes.

## 6. Gates (STOP semantics per prior contracts)

- **G3a (ARM T):** PASS iff every successive-pair paired 95% CI for
  `log2 D(T_small) - log2 D(T_large)` over (2,5),(5,10),(10,20),(20,40) has lower
  endpoint > 0 (strict monotone decrease), AND the T=5 CI contains the pilot value
  `2.441338` (replication). NOT_EVALUABLE preconditions: all 10 ARM T optima
  interior, all work valid, all bootstraps VALID. FAIL/NOT_EVALUABLE -> stop, audit
  paper.
- **G3b (ARM B):** PASS iff `|log2 D_corrected(T)| <= 0.15` at >= 4 of 5 T values
  (frozen-analyzer point estimates; ARM B's own mu0 denominators). Preconditions:
  all 10 ARM B optima interior, all work valid. FAIL refutes the invariance
  prediction (publishable); NOT_EVALUABLE -> incomplete-evidence audit paper.
- **G3c (ARM S):** DESCRIPTIVE only, `snoo_verdict` vocabulary, equal-tuning-budget
  best-grid-point estimand, margin 0.01; never stops the program.

## 7. Wall clock

One stage, hard ceiling **12 h** for all arms (earliest v3 scientific process start
through final evidence seal, retries included). Breach: terminate remaining work,
mark `NOT_RUN_WALL_CEILING`, apply the registered stop rule; never extend after
seeing outcomes.

## 8. Closed vocabularies, evidence rules, fallback

Cell status, gate verdict, program verdict, and snoo verdict vocabularies are closed
(JSON authoritative). Work-evidence, retry (loss-blind, whole seed curve), and
fallback-paper duties carry over from v2 verbatim, plus mandatory disclosure of the
pilot record, every pilot-informed choice above, and the T=20 wide-ladder
adjudication outcome either way. Negative, null, reversed, unbracketed, and
non-invariant outcomes are publishable.
