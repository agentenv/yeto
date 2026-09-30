# RL engine equivalence report (legacy vs ports)

- mode: real (legacy side only)
- config: benchmark_rl.py federated/strict-avg, Qwen/Qwen3-0.6B@c1899de, GSM8K (/work/data/gsm8k.jsonl, train/eval manifests == ports strict2 run), 2 islands x 1 H100, 3 global rounds, 4 groups x 8 samples, LoRA r16 all-linear, seed 17, --rl-engine legacy (post-#64: MILES_COMMIT ae475060, bundle da3464d3)
- created_unix: 1790657553.2857406
- legacy_repeats: 3
- tolerance_factor: 2.0
- tolerance_floor: 1e-06

## Tolerance (fixed from legacy repeats before the ports run)

| metric | kind | legacy noise spread | tolerance |
|---|---|---|---|
| reward_mean | abs | 0.0 | 1e-06 |
| loss | abs | 0.0 | 1e-06 |
| grad_norm | abs | 0.0 | 1e-06 |
| completed_groups | exact | - | exact |
| action_tokens | exact | - | exact |
| post_sync_hash | exact | - | exact |

## Legacy baseline v2 notes (post-#64; this supersedes `../2026-09-29-legacy-baseline`, which is pre-#64 and VOID)

- Code: copy of the integration tree `/home/michael/work/r0-integ` (branch rl-engine-ports on e21a7ff, uncommitted), with no edits. Legacy pins from its constants: agentenv/miles `6062afe` plus bundle `miles-qwen38.bundle` (sha256 da3464d3..., verified), giving `ae475060` (clean, origin agentenv/miles); agentenv/sglang `e1b57eb`; peft 0.20.0.
- Environment deviations (same as v1): the exact `MILES_IMAGE` could not be pulled (the GitHub token lacks `read:packages`), so the base is public `radixark/miles:v0.1.0` (`sha256:cd40db92...`); an env-only `.pth` shim raises Miles' 30 s router readiness wait to 300 s (megatron-bridge import takes ~25 s in this image); the TMS preload patch was not applied. See `env/`.
- Hardware: Modal app `yeto-legacy-baseline`, 1 sandbox, 2x H100 80GB HBM3, the same GPU type as the ports strict2 run.
- Runs legacy-0/1/2: all rc=0; the 3 repeats are **bit-identical** on every metric. Gradient flow is restored: grad_norm is non-zero on 5 of 6 (island, round). Island 0 round 2 has grad_norm=0.0 because reward_mean=0.0 there (every group scored 0, so every advantage is zero), which is allowed by the #64 invariant.
- loss/grad_norm are filled from Miles' `train/loss` / `train/grad_norm` log lines (events carry None); `env/tolerance.py` records the source of each value in `tolerance.json`.
- Large files (*.pt, *.safetensors, *.f32, state.ckpt*) were not copied back.

## Suggestions (not applied; the acceptance criteria were not changed)

Data (legacy-0 v2 vs ports strict2, same config and the same 2xH100 type):

| island/round | Δreward | Δgrad_norm (rel) | Δaction_tokens | hash |
|---|---|---|---|---|
| 0/1 | 0 | 0.0036 (0.6%) | 0 | differs |
| 1/1 | 0 | 0.0042 (1.0%) | 0 | differs |
| 0/2 | 0.094 | 0.251 (legacy round has zero advantages) | -108 | differs |
| 1/2 | 0.031 | 0.011 (3.7%) | +29 | differs |
| 0/3 | 0.031 | 0.045 (12.9%) | -41 | differs |
| 1/3 | 0.063 | 0.098 (58.4%) | -168 | differs |

Conclusions and suggestions:
1. The legacy repeat spread is 0 (it is fully deterministic), so the "2x repeat spread" rule collapses to the 1e-6 floor and does not measure the numerical difference between the two paths. Under the current rule 6.2 is guaranteed to FAIL, even though round 1 is effectively aligned.
2. Round 1 is the only fair point to compare: both paths start from the same weights, and reward, action_tokens and loss match exactly. grad_norm differs only by 0.6–1.0%, which is bf16 / different-kernel-path noise. Suggestion: at round 1, require group/token/reward counts to match exactly, with a relative grad_norm tolerance of about 2–3% (roughly 2–3x the observed value; this needs more data).
3. From round 2 on, the policies already differ slightly, so sampling diverges and the difference is amplified chaotically (one legacy round has all-zero rewards and the ports round does not). Per-round point-wise tolerances are therefore not meaningful there. Suggestion: set a distributional tolerance from legacy runs with **different seeds** (e.g. seeds 17–21, ≥3 of them): for each round, ports' reward_mean, grad_norm and action_tokens must lie within legacy's cross-seed mean ± k·std (or its min–max range).
4. Also run ports at least 3 times to get its own repeat noise. If ports is also bit-deterministic, that confirms determinism but still says nothing about cross-path numerical error, so it cannot replace item 3.
5. post_sync_hash cannot match across paths (the numerics differ), so replace it with within-path consistency: both islands must have the same hash in each round (true for both paths here). Cross-path, compare the relative L2 distance between the two global LoRA states after round 1 (the audit `*.f32` files hold what is needed).
6. The strongest check is teacher forcing: feed legacy's recorded round-1 rollouts (tokens and advantages) into the ports trainer and compare loss, grad_norm and the update delta. This removes sampling divergence and isolates trainer/sync equivalence.
