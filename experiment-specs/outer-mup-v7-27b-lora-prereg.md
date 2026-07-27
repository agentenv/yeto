# Outer-muP v7: 27B FSDP+LoRA finite-horizon T-scan

Status: **REGISTERED_PREPILOT**. This contract is prospective for the wiring
smoke, the three-cell center pilot, and the conditional 48/45-cell G7 grid.
The JSON file is authoritative:
`experiment-specs/outer-mup-v7-27b-lora-prereg.json`.

Registered 2026-07-26 from clean base commit
`ea5f034096e54a2f959e6bd73890ca0e47390430`. At registration there was no
v7 FSDP+LoRA syncer smoke, pilot endpoint, main-grid endpoint, or G7 readout.
The completed full-tune 27B smoke lane is prior engineering evidence only.

## 1. Question

Does the finite-horizon outer-momentum transient law observed at 1.7B transfer
to the pinned dense 27B text model in the supported LoRA regime? G7 is a raw
production-Nesterov test with fixed `H=512`, two horizons, and no corrected or
SNOO arm:

| learner steps `S` | horizon `T=S/H` | syncer commits (`P*T`, `P=4`) |
|---:|---:|---:|
| 2,560 | 5 | 20 |
| 10,240 | 20 | 80 |

Every scientific cell contains `M=2` logical learner islands. Each island is
one four-rank torchrun over four H200s, so a cell occupies all eight GPUs on one
node. The two fleet nodes can execute exactly two cells concurrently.

## 2. Exact model, data, and LoRA state

The model is `Qwen/Qwen3.6-27B` at immutable revision
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, loaded locally as
`Qwen3_5ForCausalLM`. The causal text path has exactly 26,895,998,464
parameters and is dense: hidden size 5,120, 64 layers, intermediate size
17,408, and no expert/router fields. Both node snapshots have 29 dereferenceable
entries totaling 55,586,107,940 bytes. Their canonical filename/blob/size
inventory SHA-256 is
`32c8f34fa11f07ffde3eedb32435b39a78590ea102b7923bbc1d9b4df7b51c4c`.

The fixed train and development files are reused read-only on both nodes:

| role | rows | SHA-256 |
|---|---:|---|
| train | 13,758 | `e680a29ea8c8fc7c99efdceb4f62e485d3eed1ac2afd15bab43b506cb3f4ecaf` |
| development eval | 1,024 | `533838a0564b13519956a044d23ed8db6705ddc7ae5f0ddb96538f49460bcebc` |

A CPU-only production-tokenizer preflight found 52,831 and 52,439 packed
128-token blocks for logical learners 0 and 1, respectively. This exceeds the
long cell's 10,240 steps after four-rank partitioning. The development file has
7,631 packed blocks and 790,341 supervised tokens.

LoRA is fixed at rank 16, alpha 32, and `--lora-targets auto`. Because this is
a dense model, `auto` resolves to `all-linear`. A meta-device construction with
the production PEFT path measured exactly 992 trainable tensors and 116,727,808
trainable parameters. One complete BF16 adapter state is 233,455,616 bytes.
The frozen BF16 base is FSDP2-sharded; LoRA parameters remain replicated FP32
tensors and their gradients are explicitly all-reduced across the four ranks.

## 3. Common cell protocol

Each island uses one 128-token example per rank and one optimizer step per
microstep. Thus an island consumes 512 raw tokens per optimizer step. The fixed
window is exactly 512 optimizer steps and 262,144 raw tokens per fragment.

The inner optimizer is AdamW with LR `3e-4`, weight decay `0.01`, ten warmup
steps, and gradient clipping at 1.0. The syncer uses four binpacked fragments,
BF16 wire state, RDA matrix merge, merge alpha 0.5, no delta correction, raw
production Nesterov, strict quorum 2, pipeline depth 4, version-matched anchors,
final-only checkpointing, and rho telemetry. There is no injected delay or
jitter.

`--barrier-sync` is deliberately **off**. The learner rejects barrier mode for
world size greater than one; using it would make this registered island shape
unexecutable. G7 instead uses the supported non-barrier strict-quorum path.
Every pushed fragment still comes from an exact post-reset `H=512` snapshot
and reports `c_steps=512`, `c_tokens=262144`. This protocol choice is fixed and
is part of what G7 tests.

## 4. Required wiring smoke

Before the pilot, one short cell must prove the complete topology:

| field | value |
|---|---:|
| seed / training seed | `683` / `683683` |
| `M`, ranks/island | 2, 4 |
| `S`, `H`, `T` | 64, 16, 4 |
| momentum / outer LR | 0 / 0.28 |
| fixed-window tokens | 8,192 |
| syncer commits | 16 |

The smoke passes only if both four-rank islands initialize against one M=2
syncer; their resolved 992-tensor layouts match; all four fragments complete
16 strict-quorum commits with two learner pushes each; rho has 16 valid rows;
both adapters save; endpoint loss is finite; every learner/syncer exits zero;
and all eight GPUs are released. Failure blocks the pilot. Only one documented,
loss-blind infrastructure retry is allowed.

## 5. Registered three-cell LoRA pilot

The center pilot is three exact production-shaped `T=5`, `S=2560`, `H=512`,
`mu=0` cells at outer learning rates `{0.14,0.28,0.56}`. They use pilot seed
691 (`training_seed=691691`) and are excluded from G7. Two pilot cells run in
parallel, followed by the third on the first released node.

The 0.28 center is disclosed prior Qwen-family LoRA outer-SGD scale with the
same alpha/r ratio of 2. The factor-of-two probe is required because LoRA outer
learning rates differ materially from the 1.7B full-tune scale.

The pilot fit is `loss=a*x^2+b*x+c`, `x=log2(eta)`. If `a>0` and the
unconstrained vertex lies strictly in
`[log2(0.14)-0.5, log2(0.56)+0.5]`, the selected center is `2^vertex` without
clipping. Otherwise the finite pilot eta with minimum loss is selected, with
exact ties resolved toward the smaller eta. No finite pilot endpoint means the
main grid is blocked. All three losses, walls, fit coefficients/status, selected
center, and evidence hashes are disclosed before `GRID STARTED`.

## 6. Mechanical main-grid placement

The fixed main seeds are `{701,709,719}` with training seed formed by decimal
concatenation `seed||seed`. Let `p` be the selected pilot center. Curve centers
are then calculated without discretion:

| curve | center |
|---|---|
| `T5,mu0` | `p` |
| `T20,mu0` | `p * 0.35036736670682456` |
| `T5,mu0.9` | `p * 0.1 * 1.7416157949788522` |
| `T20,mu0.9` | `p * 0.35036736670682456 * 0.1 * 1.2806943474449415` |

The T20/T5 mu0 ratio and the two D constants come from the disclosed final
G4C point fits. They place the ladders; they are not G7 outcomes or exact-
constant claims.

The full variant uses log2 offsets `{-1.5,-0.5,+0.5,+1.5}` from every center.
Adjacent etas differ by exactly 2x. Its identity is
`2 horizons * 2 momenta * 4 etas * 3 seeds = 48 cells`.

The conditional reduced variant changes only `T20,mu0`, which uses symmetric
offsets `{-1.5,0,+1.5}`. The other three curves retain four points. Its identity
is 45 cells, and the frozen analyzer fits the one three-point quadratic exactly.

## 7. Registered 20-fleet-hour decision

A fleet-hour means one elapsed hour with the full two-node/16-H200 fleet
available to v7; it is not a GPU-hour. Let `w` be the maximum successful
end-to-end wall seconds among the three pilot cells. The prospective duration
model is `w` for each short cell and `4w` for each long cell. The controller
applies deterministic longest-processing-time scheduling to the 24 short and
24 long FULL_48 durations on two node slots.

If projected makespan is at most 20.0 fleet-hours, select `FULL_48`. If it is
greater than 20.0, select `REDUCED_T20_MU0_45`. Equality stays full. The
calculation, selected variant, exact numeric etas, and manifest hash are sealed
before any main-grid process. Actual main-grid timing or outcomes cannot reopen
the choice.

## 8. Gate feasibility simulation

The simulation ran before registration and before v7 GPU work. Its source is
the final 110-cell G4C readout, SHA-256
`16bab85dce3a83f42c8b1c91be8d91eb1ec1372cfba572f2ffd9b491044b04aa`.
At each of the 22 combined-grid coordinates, the five-seed mean was subtracted.
The resulting 110 residuals have pooled population SD `0.0632194751`; each
seed's profile is preserved jointly across eta and all four curves.

The hash-bound artifacts are:

| artifact | SHA-256 |
|---|---|
| `scripts/build_v7_noise_prior.py` | `bf7af2b0c675dec495066ef9cc405818d91ef698426afacab887db18e9d39bb7` |
| `experiment-specs/outer-mup-v7-27b-lora-noise-prior.json` | `5d684e9b296ccb35eb89d399a70eacd30cb92ee9d4a56d06acbea25e693d0876` |
| `scripts/gatesim_v7.py` | `ea3c17bf83e468edc6693e970b3275a9c44229c5d635aa69d2c1ee2e9d0b06da` |
| `experiment-specs/outer-mup-v7-27b-lora-gatesim.json` | `3428a266676b7468c243c2d5e153c68609405ab42cc610c86938344b52dea74f` |

Each simulated G7 seed samples one of the five complete residual profiles with
replacement, shared across all curves; curve profiles are interpolated by eta
rank onto the target ladders. The true mean uses the observed G4C curvature and
D constants. With the exact analyzer bootstrap and near-bracket rule:

| variant | simulations | evaluable | `P_eval` | passed | `P_pass` |
|---|---:|---:|---:|---:|---:|
| `FULL_48` | 5,000 | 5,000 | 1.000 | 5,000 | 1.000 |
| `REDUCED_T20_MU0_45` | 5,000 | 5,000 | 1.000 | 5,000 | 1.000 |

Every simulated experiment retained all 10,000 bootstrap refits. At 0.5x and
1.5x residual amplitude, both variants again had `P_eval=P_pass=1.000` over
2,000 simulations per scale and variant. This is a feasibility result under a
transported 1.7B full-tune prior, not a claim that 27B LoRA noise or constants
are identical.

## 9. Frozen G7 analysis

The frozen analyzer is `scripts/analyze_v7.py`, SHA-256
`c835189056d407535cb866c4095b49a35361391a43b8f23a3669b40914d18f75`.
Each curve is fit by ordinary least squares to the three-seed mean loss in
`x=log2(eta)`. Positive curvature is required. A vertex is accepted only when
it lies strictly between the registered endpoints extended by 0.5 log2-eta
bits on each side. Accepted outside vertices are `NEAR_BRACKETED`; the
unconstrained vertex is never clipped.

The observed transient constant is

```text
D(T) = [eta_star(mu=0.9,T) / eta_star(mu=0,T)] / (1-0.9).
```

Confidence intervals use 10,000 paired nonparametric training-seed bootstrap
draws with RNG seed `20260727`. One shared three-index draw is used at every
eta and for all four refits. At least 7,900 draws must have all four vertices
accepted and both positive D values.

The deliberately wide bands are the final observed 1.7B constants multiplied
by `[0.5,1.5]`:

| horizon | 1.7B observed constant | registered G7 band |
|---:|---:|---:|
| 5 | 1.7416157949788522 | [0.8708078974894261, 2.6124236924682784] |
| 20 | 1.2806943474449415 | [0.6403471737224707, 1.9210415211674121] |

The +/-50% width is intentional: LoRA-regime constants are unknown.

G7 is evaluable only with complete hash-valid evidence, both mu=0.9 optima
accepted, both mu=0 denominator optima accepted, and at least 7,900 accepted
bootstrap refits. It passes only if both D values lie in their bands and the
paired 95% CI for `log2 D(T=5)-log2 D(T=20)` has lower endpoint greater than
zero. Complete evaluable evidence violating a scientific condition is `FAIL`;
invalid/incomplete or unbracketed evidence is `NOT_EVALUABLE`.

## 10. Fleet gate, evidence, and rails

The v6 factorial retains priority even if GPUs look transiently idle. V7 waits
for a terminal/drained v6 note, terminal slot records, no matching controller,
comparison, learner, or syncer process on either node, and clean GPU occupancy.
Every `pgrep` expression uses bracketed names such as `[r]un_slot_v6.py`,
`[c]ompare_diloco.py`, `[y]eto.learner`, and `[y]eto-syncer`. V7 never stops,
pauses, renices, or overlaps v6.

A completed main cell requires exact command/source hashes; attempt and GPU
UUID records; two four-rank islands at exact `S`; `4*T` strict-quorum tape and
rho rows; exact fixed-window counters; finite 1,024-row loss; zero exits; and
hashes for every command, code, adapter, evaluation, tape, telemetry, log, and
checkpoint artifact. Partial work never counts.

There is at most one loss-blind infrastructure retry. The retry unit is every
eta in one `(T,mu,training-seed)` curve. Finite unfavorable loss, bracket or
band failure, bootstrap validity, and every other scientific outcome forbid a
retry. A 30-fleet-hour main-grid ceiling begins at `GRID STARTED`; it is never
extended after outcomes.

No cloud capacity is used. Prior result trees and 27B smoke artifacts are
immutable. Normal termination targets recorded v7 process groups; wildcard
`pkill` is forbidden. The coordination note is
`/private/tmp/h200-27blora-note.md` and must contain the milestones
`V7 REGISTERED sha=`, `PILOT DONE`, `GRID STARTED`, and `G7 VERDICT:`.
