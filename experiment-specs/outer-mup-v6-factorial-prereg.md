# Outer-muP v6 full-factorial preregistration

Status: **prospective and frozen before any v6 scientific process**. The machine-readable JSON is authoritative for every numeric eta, command coordinate, gate rule, and hash.

## Question and design

At SmolLM2-135M with four learners packed on one H200, how do the tuned outer-learning-rate ratios for raw and bias-corrected Nesterov depend jointly on the number of outer commits per fragment, `T`, and total learner work, `S`?

The design is the complete factorial

- `T = {2,5,10,20}`;
- `S = {2560,5120,10240}` learner steps;
- `H = S/T` exactly in every cell;
- arms `mu0`, raw Nesterov `mu=0.9`, and bias-corrected Nesterov `mu=0.9`;
- five prospectively fixed etas per arm and coordinate;
- seeds `{601,607,613,617,619}` with training seed `int(str(seed)+str(seed))`.

This gives `12 × 3 × 5 × 5 = 900` scientific cells. Each cell is one `compare_diloco.py` process packing four full-parameter learners on one GPU. It uses exact strict-quorum fixed windows, `4*T` syncer commits, `S` steps per learner, a fixed 1,024-row development set, rho telemetry, no injected delay, and no data wrap.

## H feasibility and data proof

All requested coordinates close exactly:

| T | S=2560 | S=5120 | S=10240 |
|---:|---:|---:|---:|
| 2 | H=1280 | H=2560 | H=5120 |
| 5 | H=512 | H=1024 | H=2048 |
| 10 | H=256 | H=512 | H=1024 |
| 20 | H=128 | H=256 | H=512 |

The requested H set is `{128,256,512,1024,1280,2048,2560,5120}`. The learner and comparison-runner parsers accept positive integer fixed-window steps/tokens without an upper-H cap. Exact commands bind `--fixed-window-microsteps H`, `--fixed-window-tokens 128*H`, `--learner-max-steps S`, and `--syncer-total-steps 4*T`. H values through 2,048 have already completed end-to-end in prior 135M campaigns; H=2,560 and 5,120 exercise the same fixed-window path with fewer commits and no additional model-state allocation.

The new seed bundles are isolated under `/root/yeto-data/outer-mup-v6`; no earlier data tree was changed. Both nodes independently produced the same combined input-manifest hash `b260be9ee97ebf19ca897d5a8ca638ec05527d40f31d989b22a760d2d75d6b47`. The frozen verifier found a minimum of 26,399 complete 128-token blocks (seed 617, learner 1), versus 10,240 required by the longest cell. Its report hash is `9e06988092409f1e8c8449425a7348dad2a8f3c22d6a52a6397a69e53ea88a59`.

**No requested cell is infeasible.** Any later failure to reproduce the hash, exact-window arithmetic, GPU inventory, or roomy-volume proof is an infrastructure preflight failure and blocks launch; it does not silently delete a factorial coordinate.

## Prospectively fixed eta centers

Every curve uses five adjacent sqrt-two rungs with log2 offsets `{-1,-0.5,0,0.5,1}`, i.e. multipliers `{0.5, 1/sqrt(2), 1, sqrt(2), 2}`. Thus every grid covers center/2 through 2×center, a fourfold endpoint span. Three earlier campaigns lost evaluability to boundary or nonconvex fits, so there will be no outcome-aware extension or recentering.

The `mu0` center is a log-linear interpolation/extrapolation in `S` and `H`, with a separate parameter-scale term:

```text
log2 eta0 = -4.4106518295376675
            -0.4505983838816392 log2(S/2560)
            +0.4886611783077774 log2(H/512)
            -0.17417236407974745 log2(parameters_M/135).
```

At the target 135M scale the last coordinate is zero. Inputs are the four bracketed v3 H=512 baselines, the two completed disambiguation baselines at `(H,S)=(1024,5120),(2048,10240)`, and the v4 1.7B H=512/S=2560 center pilot. The separate scale coefficient prevents the 1.7B level from being treated as a 135M target observation.

For raw Nesterov,

```text
D_code_true(T) = 1 / (1 - 0.9^(T+1))
eta_raw_center = 0.1 * D_code_true(T) * eta0_center.
```

For corrected Nesterov, the empirical measured-drift decomposition is

```text
log2 kappa(T,H) = -0.2033959883796063
                  -0.046919366440622 (T-5)
                  -0.10681443714765025 log2(H/512)
eta_corrected_center = 0.1 * kappa(T,H) * eta0_center.
```

This kappa surface is only a centering model. It is fit to the four v3 corrected points at H=512 and the two fixed-T=5 disambiguation points at H=1024/2048. The separate two-parameter audit rejected the common-correlation/equal-displacement mechanism, so v6 does not present kappa as a validated mechanistic law.

The resulting registered centers and endpoint ranges are:

| T | S | H | mu0 center `[low,high]` | raw center `[low,high]` | corrected center `[low,high]` |
|---:|---:|---:|---:|---:|---:|
| 2 | 2560 | 1280 | 0.0735731 `[0.0367866,0.147146]` | 0.0271488 `[0.0135744,0.0542975]` | 0.00638789 `[0.00319395,0.0127758]` |
| 2 | 5120 | 2560 | 0.0755401 `[0.0377700,0.151080]` | 0.0278746 `[0.0139373,0.0557491]` | 0.00609062 `[0.00304531,0.0121812]` |
| 2 | 10240 | 5120 | 0.0775596 `[0.0387798,0.155119]` | 0.0286198 `[0.0143099,0.0572395]` | 0.00580718 `[0.00290359,0.0116144]` |
| 5 | 2560 | 512 | 0.0470177 `[0.0235089,0.0940354]` | 0.0100345 `[0.00501727,0.0200691]` | 0.00408351 `[0.00204175,0.00816701]` |
| 5 | 5120 | 1024 | 0.0482747 `[0.0241373,0.0965494]` | 0.0103028 `[0.00515140,0.0206056]` | 0.00389347 `[0.00194674,0.00778694]` |
| 5 | 10240 | 2048 | 0.0495653 `[0.0247826,0.0991306]` | 0.0105782 `[0.00528912,0.0211565]` | 0.00371228 `[0.00185614,0.00742456]` |
| 10 | 2560 | 256 | 0.0335089 `[0.0167544,0.0670177]` | 0.00488333 `[0.00244166,0.00976665]` | 0.00266358 `[0.00133179,0.00532715]` |
| 10 | 5120 | 512 | 0.0344047 `[0.0172024,0.0688094]` | 0.00501388 `[0.00250694,0.0100278]` | 0.00253962 `[0.00126981,0.00507924]` |
| 10 | 10240 | 1024 | 0.0353245 `[0.0176622,0.0706490]` | 0.00514792 `[0.00257396,0.0102958]` | 0.00242143 `[0.00121072,0.00484287]` |
| 20 | 2560 | 128 | 0.0238813 `[0.0119407,0.0477626]` | 0.00268154 `[0.00134077,0.00536309]` | 0.00147665 `[0.000738323,0.00295329]` |
| 20 | 5120 | 256 | 0.0245198 `[0.0122599,0.0490395]` | 0.00275323 `[0.00137662,0.00550646]` | 0.00140793 `[0.000703964,0.00281585]` |
| 20 | 10240 | 512 | 0.0251753 `[0.0125876,0.0503506]` | 0.00282684 `[0.00141342,0.00565367]` | 0.00134241 `[0.000671203,0.00268481]` |

The JSON contains all 180 exact eta values at full precision and is authoritative over rounded table displays.

## Registered held-out prediction

For each momentum arm separately, define

```text
D(T,S,arm) = [eta*(T,S,arm) / eta*(T,S,mu0)] / 0.1.
```

The response surface is fixed as

```text
log2 D = beta0
         + beta_T log2(T/5)
         + beta_S log2(S/5120)
         + beta_TS log2(T/5) log2(S/5120).
```

It is fit by ordinary least squares on these eight coordinates:

```text
(2,2560), (2,5120),
(5,2560), (5,10240),
(10,5120), (10,10240),
(20,2560), (20,10240).
```

It must predict the four held-out coordinates, never used to fit coefficients:

```text
(2,10240), (5,5120), (10,2560), (20,5120).
```

For a held-out cell, success means

```text
abs(log2 D_pred - log2 D_obs) <= 0.2.
```

G6 is `PASS` only if **each** of raw and corrected succeeds on at least three of its four held-out cells. It is `FAIL` if the complete, bracketed evidence is evaluable but either arm misses that rule. It is `NOT_EVALUABLE` for incomplete/invalid evidence, any of the 36 unbracketed eta optima, a singular surface, or fewer than 9,500 valid refits in the registered 10,000-draw joint paired-seed bootstrap. The mu0 arm has structural `D=1`; it supplies every denominator but is excluded from the trivial response-surface gate.

The eta optimum is the interior vertex of `loss = a(log2 eta)^2 + b(log2 eta) + c` fitted to the five-seed mean loss at all five exact rungs. A required nonfinite/missing cell is never dropped. The joint bootstrap uses one common five-index seed resample for all 36 curves, both surfaces, and all held-out errors.

Frozen analyzer: `scripts/analyze_v6.py`, SHA-256 `4862681e6d7dbbf55e95ae45e15e0a0170e5a35a17a73f0df01240951b576f1a`.

## Scheduling, storage, and wall ceiling

The launch manifest deterministically shuffles within equal-S strata, then assigns cells by longest-first greedy work balance across the 16 `(node,GPU)` slots. Every per-slot queue is non-increasing in S. One cell owns one GPU and packs its four learners.

Results use the roomy volume found by v4:

```text
/root/yeto-results-v6 -> /data/yeto-results-v6
```

On both nodes `/data` is the approximately 27.9-TB XFS filesystem backed by the eight-NVMe LVM. Preflight requires the exact symlink/device and at least 1 TB free; no v6 checkpoint may land on `/`.

The immutable launch authority sets a 30-hour (108,000-second) ceiling immediately before controllers start. At the deadline, active process groups are terminated, unstarted cells are marked `NOT_RUN_WALL_CEILING`, all partial evidence is retained, and G6 becomes `NOT_EVALUABLE`. The deadline cannot be extended after outcomes.

## Fleet gate and retry rails

Registration and non-GPU preflight are allowed while waiting. **No v6 GPU work may launch** until a hash-bound gate proof establishes all of:

1. v4 has 48 unique `COMPLETED` evidence cells across h200-n1/n2;
2. `pgrep -af '[r]un_slot_v4.py'` is empty on both nodes;
3. `/private/tmp/h200-snoofix-note.md` contains a line beginning `G5 VERDICT`.

The registrar polls those conditions every 600 seconds. The launch authority binds the passing gate proof and complete manifest.

An attempt-2 retry is allowed only for a prospectively enumerated infrastructure reason and under a separate hash-bound authority. Retry scope is the whole paired-seed curve group `(T,S,arm,seed)`: all five eta rungs retry together. Retrying an individual rung, retrying because a finite loss looks poor, selecting between successful attempts, or extending a ladder is forbidden.

Every completed cell must prove exact command/source hashes, four learners each reaching S steps, exactly `4*T` strict-quorum tape rows with `c_steps=H` and `c_tokens=128*H`, matching rho telemetry, finite 1,024-row evaluation, successful exits, and hashes for all scientific artifacts. Earlier result trees are immutable.
