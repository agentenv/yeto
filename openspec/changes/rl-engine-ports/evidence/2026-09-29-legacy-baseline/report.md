> **VOID (pre-#64)**: this baseline used the legacy pin from c40a32c (MILES_COMMIT e2ad83d, before #64); its gradients are always 0. Superseded by `../2026-09-29-legacy-baseline-v2/`.

# RL engine equivalence report (legacy vs ports)

- mode: real (legacy side only)
- config: benchmark_rl.py federated/strict-avg, Qwen/Qwen3-0.6B@c1899de, GSM8K (/work/data/gsm8k.jsonl, train/eval manifests == ports strict2 run), 2 islands x 1 H100, 3 global rounds, 4 groups x 8 samples, LoRA r16 all-linear, seed 17, --rl-engine legacy
- created_unix: 1790655641.845219
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

## Legacy baseline notes (written with the header; no ports comparison performed here)

- Runs: `legacy-0`, `legacy-1`, `legacy-2`, all rc=0, Modal app `yeto-legacy-baseline`, 1 sandbox with 2x NVIDIA H100 80GB HBM3 (driver 580.95.05), the same GPU type as the ports strict2 run (Modal `H100:2`). Raw logs/events are in each `legacy-N/` directory. Per-round values are in `tolerance.json`.
- Determinism: all 3 repeats are **bit-identical** on every metric (reward, loss, grad_norm, groups, tokens, post-sync hash). So every float tolerance equals the 1e-6 floor, and counts and hashes are exact.
- **loss / grad_norm source**: legacy `rl_local_round` events carry `loss=None` and `grad_norm=None`. Ports events also carry `loss=None`. `env/tolerance.py` fills them from Miles' per-step `train/loss` / `train/grad_norm` log line (optimizer_steps=1, so step k maps to round k+1). The comparison step MUST use the same fill for ports.
- **Legacy has zero gradients**: Miles logs `train/grad_norm: 0.0` for every step on both islands. `delta_l2_norm` is about 7.9e-5 and identical on both islands (ports: 0.23), which is consistent with a weight-decay-only update. This is the known legacy defect in migration-ledger #64 (`fix/rl-lora-grad-hook`, still OPEN): `_optimizer_masters_as_model_parameters` replaces AccumulateGrad, so `main_grad` stays 0. The pinned MILES_COMMIT e2ad83d does not contain the fix. The `zero_grad_norm_with_nonzero_advantages` invariant also did not fire, because grad_norm never reaches yeto's stats (None).
- Consequence for 6.2: round-1 reward, action_tokens and step-0 loss equal the ports run exactly (same data/init/sampling). From round 1 on, grad_norm, post-sync hash and later rounds' reward/tokens diverge. The divergence comes from the legacy bug, not from ports, so a 6.2 PASS against this baseline is not achievable as specified.
- Environment deviations: see `env/`. The exact MILES_IMAGE (`ghcr.io/agentenv/miles@sha256:80c2...`) could not be pulled because the available GitHub token lacks `read:packages`. The base image was the public `radixark/miles:v0.1.0` (`sha256:cd40db92...`, same sglang v0.5.16 era as the fork base 6062afe). On top of it: agentenv/miles `6062afe` plus bundle (sha verified), giving `e2ad83d` (clean, origin = agentenv/miles); agentenv/sglang `e1b57eb`; peft 0.20.0 (the same steps as the launcher's legacy setup). An environment-only `.pth` shim (`env/yeto_legacy_router_timeout.py`) raises Miles' hard-coded 30 s router readiness wait to 300 s, because in this image megatron-bridge takes about 25 s to import in the spawned router child (first attempt failed on this: `env/legacy-0-attempt1-router-timeout.tgz`). The TMS preload patch was not applied.
- Data: the source `gsm8k.jsonl` was regenerated with the same recipe. Its sha differs (65fe7f0e vs 0615907d) but the byte size is the same, and the derived train/eval manifests (the rows actually used) are identical to the ports run.
