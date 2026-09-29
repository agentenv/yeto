# G1 plan (written and committed before launch, 2026-09-29, Agent ALGO-2a)

Tasks: rl-algo-seq-and-adv 7.2 (GSPO, optimizer_steps=2 plus optimizer_steps=1 control),
7.3 (REINFORCE++ / REINFORCE++-baseline, whiten on, kl.placement=reward), 7.4 (MaxRL, MAPO,
GDPO with the 5.4 example reward). Single island, 1 GPU, no outer sync, 3 rounds each,
`--rl-allow-unverified-mechanism` (P0 D11) for the mechanism and the supporting
undeclared mechanisms of its spec (listed in g1.py). Specs: `../../examples/*.json`.

## Code under test
- yeto: branch algo-2a at the commit that adds this plan, **plus `infra-drafts/2a-shared.patch`
  applied in the uploaded tree only** (needed: without its hunk 1 the Miles rollout process
  never imports `yeto.rl.algos.seq_adv`, so maxrl/mapo/gdpo would be unknown transforms).
  The patch SHA256 is recorded in `upload.txt`. GSPO/rpp runs do not depend on it.
- Image: `ghcr.io/michaellchung/yeto-miles-ports@sha256:5da40a07…` (= MILES_NEXT_IMAGE on
  rl-integ f6194da; Miles fork 0394715 at /root/miles, SGLang fork 9f29303).
- Model Qwen/Qwen3-0.6B @ c1899de2, LoRA r16 all-linear, gsm8k train prompts 0–11 (4 groups
  x 8 samples per round, 3 rounds), response 384, seq 1024, lr 1e-5, seed 17.
  Reward: harness `gsm8k_reward:score` (binary); GDPO: `yeto.rl.algos.gdpo_reward:reward_func`.

## Resources and reclamation
- Modal app `algo2a-g1`, one Sandbox, GPU `H100!` (exact, no H200 upgrade), asserted by
  `nvidia-smi` in `run_all.sh` (exit 90 otherwise), cpu 16, mem 128 GiB.
- Hard limits: Sandbox `timeout=5400` s (Modal side); local `timeout 5700` around the driver;
  driver `finally: sb.terminate()`; independent watchdog (setsid, survives this agent):
  `sleep 6000; modal app stop -y algo2a-g1`. After the run: `modal app stop -y algo2a-g1`,
  `modal app list` / sandbox list proof in `teardown.txt`. No volumes or named secrets
  (pull secret is an in-memory `Secret.from_dict`).
- Estimate: ~7 runs x ~8 min + ~10 min pull/setup ~ 70 min; H100 ~$3.95/h + CPU/mem
  ~$1.5/h -> ~$6.5. Cap: $20 in total for all G1 sessions of this change.

## Pass criteria per run (declared before launch; all must hold)
1. worker rc = 0 and 3 rollouts trained (3 `rl_round`/train steps x optimizer_steps logged).
2. No `StrictRlInvariantError`, `RoundFailedError`, `AdvantageTransformError`,
   `RewardPipelineError`, `CapabilityMismatch`, `AlgorithmSpecError` in the logs.
3. Every logged `train/grad_norm` is finite.
4. The Miles argv contains the mechanism's flags: gspo `--advantage-estimator gspo --eps-clip 0.0003
   --eps-clip-high 0.0004`; rpp/rpp_baseline `--kl-coef 0.01 --normalize-advantages` (reference
   model loaded); transforms `--custom-reward-post-process-path yeto.rl.algos.reward_pipeline.post_process`
   and one `rl_advantage_transform` event per round naming the transform.
5. The unverified allowance is recorded (`yeto_rl_unverified_mechanisms` / capability attestation).

Observations reported, **not** pass criteria (smoke "pass" = the chain works, not a benefit):
per-step `train/pg_clipfrac`, `train/grad_norm`, the zero-gradient decision of each round,
all-wrong / all-right group counts, non-zero advantage counts. Expectation (not a criterion):
with optimizer_steps=1 GSPO's clip fraction is ~0 (design D1).

## Failure handling
A failing run is diagnosed from its logs; only a run whose cause is identified and fixed is
rerun (at most one rerun per mechanism). No seed/prompt changes between attempts. A mechanism
that fails twice stays undeclared (7.5) with the reason recorded.

## Attempt 1 (sb-NCEOOOmOOTK45bK8f140xp, H100 80GB HBM3 asserted, 207 s) -- harness bug
- gspo_s2 worker rc=1 after 149 s: Miles `RayWorkerManager.init` -> `ServerUnavailable`,
  `HTTPConnection(host='127.0.0.1', port=8265) Connection refused` (Ray state API; the
  Ray dashboard was disabled by g1.py `include_dashboard=False`; the rl-engine-ports
  harness this was derived from uses `include_dashboard=True`). Excerpt read live from the
  sandbox (attempt1/miles_excerpt.txt); the sandbox was then terminated by hand (the
  remaining runs would fail the same way), so out.tgz could not be fetched.
- Fix: `include_dashboard=True`; run_all.sh now re-tars after every run and aborts the
  session if the first run fails. Attempt 2 reruns the same runs, otherwise unchanged.
