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

## Attempt 2 plan revision (committed before launch; supersedes "Code under test" above)
- Code under test: exactly the pushed algo-2a commit that contains this section (no local
  patch). It includes 1b F1 (dispatcher PluginRef covers seq_adv, `load_extensions`),
  ALGO-CAP 2f9f02c/6149a90 (gradient tightening, `dimension:name` allowances, pipeline
  modules no longer need `features:plugins`) and INFRA R1/R2 (infra-a 39fa0ac:
  `masked_fraction`/`clip_fraction`/`nonzero_advantages` in `rl_round_trained`).
- Entry: P0 09607d6 single island without syncer / outer sync, learner side
  (`rl_single_island_no_sync=True` in the worker arguments, `LocalOnlySync`). First real GPU
  use of that entry. The launcher's `--rl-single-island-no-sync` is not used because the
  launcher hard-codes `--optimizer-steps 1` (launcher.py) and 7.2 needs optimizer_steps=2.
- Allowances (qualified): gspo `advantage_estimators:gspo features:eps_clip
  features:clip_higher`; rpp family `advantage_estimators:<estimator> features:whiten_advantages`;
  transforms `features:<name> reward_postprocessors:custom_reward_postprocess`.
- Example specs regenerated with `grpo_knobs.with_pipeline_plugins` (make_examples.py).
- Pass criteria 1-5, observations, resources, limits and cost cap unchanged. Criterion 4
  additionally reads the `rl_round_trained` event fields (masked_fraction / clip_fraction /
  nonzero_advantages) as observations. Known open P0 items (fixture failure id, driver event
  names) do not affect these criteria.

## Attempt 2 (sb-p3xH10YsZ2hFlfXqMXikR3, H100 80GB HBM3 asserted, 557 s, code 8d7f752) -- infra defect
- gspo_s2: generation, rollout and the first round's two optimizer steps ran (Miles log: step 0
  pg_clipfrac 0.0 grad_norm 1.129; step 1 pg_clipfrac 0.4375 grad_norm 0.417 -- the sequence clip
  binds on the second step, as design D1 predicts). Then `MilesTrainerGroup.train_step` built
  `LocalStepReceipt(algorithm='gspo')` and `yeto/rl/contracts.py` refused it:
  `ValueError: unsupported local RL algorithm: 'gspo'` (`_ALGORITHMS = {"grpo", "sao"}`).
  worker rc=1; run_all aborted the session (first run failed). Evidence: attempt2/.
- Cause is outside this change (contracts.py / entry.py:265): every non-grpo estimator
  (gspo, reinforce_plus_plus, reinforce_plus_plus_baseline) fails the same way. Proposed fix:
  `infra-drafts/2a-receipt.patch`. GSPO / rpp / rpp_baseline G1 wait for it (their one allowed
  rerun is kept for after the fix).
- MaxRL / MAPO / GDPO use `advantage.estimator="grpo"` (receipt label "grpo") and are not
  affected: attempt 3 runs exactly those three with the same code (8d7f752) and harness.

## Attempt 3 (sb-NI3Hz3weM7E0P0opmx8zPd, H100 asserted, 134 s, code 8d7f752) -- my stale example
- maxrl refused before training (pre-GPU rejection working as designed):
  `[grpo_knobs_pipeline_plugins] plugins ['yeto.rl.algos.seq_adv.gdpo']: source hash differs`.
  The example specs were generated before d5ffa61 changed seq_adv.py and were not regenerated.
- Fix: regenerated examples; new test `test_example_specs_are_current` fails whenever an example
  is stale. Attempt 4 reruns maxrl/mapo/gdpo with the commit containing this fix.
