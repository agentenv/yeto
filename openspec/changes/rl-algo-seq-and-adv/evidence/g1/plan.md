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

## Attempt 4 (sb-d5jaFqTcmupKs4Pm7STePI, H100 80GB HBM3 asserted, 1158 s, code e54d2f7) -- maxrl/mapo/gdpo PASS
Per pre-declared criteria (evidence attempt4/<run>/miles.log, island-0/events.jsonl):
| run | 1 rc / trained rounds | 2 errors | 3 grad_norm per round (finite) | 4 dispatcher argv + transform events | 5 unverified recorded |
|---|---|---|---|---|---|
| maxrl | 0 / 3 | 0 | 0.339 0.598 0.662 | custom_reward_post_process_path=...post_process; 3 events | features:maxrl, reward_postprocessors:custom_reward_postprocess; rl/outer_sync=false |
| mapo | 0 / 3 | 0 | 0.456 0.439 0.380 | same; 3 events | features:mapo, ...; outer_sync=false |
| gdpo | 0 / 3 | 0 | 0.566 0.384 0.439 | same; 3 events | features:gdpo, ...; outer_sync=false |
Observations (not criteria): per round (all-right groups / all-wrong groups / non-zero advantages of 32):
maxrl 1/0/24, 1/1/16, 0/1/24; mapo 1/0/24, 1/1/16, 0/2/16; gdpo (correctness) 1/0/32, 1/1/32, 0/3/32 --
GDPO round 3 has 3 all-wrong groups on correctness yet 32 non-zero advantages from the format
component. [Corrected after review] The D8 tightening branch (R0 rule says no gradient, GDPO rule
requires one) was NOT exercised on GPU: every round had some group with scalar reward_std > 0, and
the per-round count reached the driver one round late (R2 defect below). No zero-gradient event in any round.
Defect found (INFRA R2): `rl_round_trained.nonzero_advantages` lags one round (round 0 None,
round k shows round k-1's dispatcher count: maxrl dispatcher 24,16,24 vs events None,24,16;
mapo 24,16,16 vs None,24,16). The rollout metadata hook reads the counter before the reward
post-process of the same round runs. GDPO/REINFORCE++ zero-gradient decisions would use the
previous round's count until fixed.
First real use of the P0 single-island no-sync entry (learner side): events carry
rl/outer_sync=false and rl/contains_unverified_mechanisms=true.

## Cost / cleanup (attempts 1-4)
Sandboxes: 207 s + 557 s + 134 s + 1158 s = 2056 s of 1x H100 (+16 CPU / 128 GiB).
Estimate ~ $3.1 (H100 $3.95/h + ~$1.5/h CPU/mem; not billing-confirmed); cap $20.
Both algo2a-g1 apps `stopped`, 0 tasks; watchdogs killed (teardown-attempt1.txt, teardown-attempts2-4.txt).
No volumes, no named secrets.

## Attempt 5 plan (committed before launch) -- launcher entry, GSPO / REINFORCE++ family + GDPO
- Why: attempt 2's receipt defect is fixed (infra-a e9f20cc); P0 requires the launcher entry
  `yeto launch --rl-single-island-no-sync --controller local --rl-optimizer-steps N` (algo-cap
  6d53fc3 fixed its crash). Attempt 4 used the learner side only and does not count as entry
  validation. GDPO is rerun because 5.5 needs per-round evidence of the fixed R2 channel.
- Code: the pushed commit containing this section (launch_run.sh records `git rev-parse HEAD`).
- Runs (sequential, one Modal app `yeto-algo2a-g1-<run>` each, 1x `H100!` via
  `--modal-gpu-exact`, container asserts the GPU type): gspo_s2 (optimizer_steps 2), gspo_s1
  (1), rpp, rpp_baseline, gdpo. Same model/data/seed/3 rounds/4x8 as before; binary reward
  `yeto.rl.algos.gdpo_reward:correctness_reward` (GSM8K label after "####"), GDPO
  `yeto.rl.algos.gdpo_reward:reward_func`. Allowances per launch_run.sh.
- Limits: local `timeout 2400` around each launcher; independent setsid watchdog per run
  `sleep 2700; modal app stop -y yeto-<prefix>`; after each run `yeto down <prefix>` +
  `modal app stop` + app list proof. Estimate 5 x ~12 min ~ 1 h H100 ~ $5.5; total cap for the
  change stays $20 (spent so far ~$3.1).
- Reading the result: the launcher's exit code 2 for a Modal no-sync island is an explicit
  "artifacts not pulled" notice, not a failure. Criteria are read from the streamed learner
  log (launch.log) and the event tape lines it carries: criteria 1-5 of the original plan,
  with criterion 1 = learner exit 0 in the stream and 3 `rl_round_trained` events.
  Criterion 4 for gspo: argv `--advantage-estimator gspo --eps-clip 0.0003 --eps-clip-high
  0.0004`; rpp family `--kl-coef 0.01 --normalize-advantages`.
- Additional pre-declared checks (observations unless stated):
  - gspo: per-round `masked_fraction`/`clip_fraction` in `rl_round_trained` (R1);
    expectation only: gspo_s1 clip fraction ~0.
  - gdpo (5.5 criterion, pass/fail): for each round k, `rl_round_trained.nonzero_advantages`
    equals the dispatcher's `rl_advantage_transform` count logged for round k (event
    `rollout_id` = k). Any mismatch or missing value = 5.5 not passed.
- Failure handling as before (diagnose; one rerun per mechanism after a fix).
- Attempt 5 try 1 (gspo_s2, no GPU started): Modal image build failed pulling the private image
  (`skopeo copy ... unable to retrieve`): launch_run.sh did not export the registry credentials
  the launcher reads (SKYPILOT_DOCKER_USERNAME/PASSWORD/SERVER). Fixed in launch_run.sh
  (decoded in-process from ~/.docker/config.json, never printed). Evidence
  attempt5/gspo_s2-try1-nopullcreds/. Same runs follow.
- Attempt 5 try 2 (code aa74ecf; receipt = infra-a e9f20cc whitelist, since reverted by INFRA 8cf1dec):
  - gspo_s2 (app ap-ECUGDQDcE31SQnqAtP7vgv, H100 80GB HBM3, ~9 min): learner finalized, job
    SUCCEEDED, launcher rc 2 (the pre-declared "artifacts not fetchable" notice). From the stream:
    argv gspo, eps_clip 0.0003 / eps_clip_high 0.0004; 3 rounds x 2 optimizer steps, per step
    (clipfrac, grad_norm): (0.0, 0.900) (0.1875, 0.307) | (0.0, 0.646) (0.5, 0.528) |
    (0.0, 0.478) (…second step in launch.log); all finite; no invariant error.
    **Not assessable**: criterion 1's `rl_round_trained` count, criterion 5 (unverified in the
    events) and the R1 `masked_fraction` observation -- the event tape is not in the streamed log
    and the Modal no-sync island's ~/yeto-output cannot be fetched. So this run is recorded, not
    counted as a pass. Needed from P0/INFRA: stream the event-tape lines (or pull them) for Modal
    no-sync islands.
  - gspo_s1 was stopped by hand ~1 min after start (app ap-wfkKkSassVbNcGoJIqdIyA) because the
    receipt code changed (infra-a 8cf1dec); rpp / rpp_baseline / gdpo not started.
  - All apps stopped, 0 tasks; watchdogs killed.

## Attempt 6 (committed before launch) -- same plan as attempt 5, event tape now returned
- Code: pushed commit containing this section (merges algo-cap 2bce8ed event echo, infra-a
  080bcbf round ids / 8cf1dec receipt family "grpo" for gspo/rpp). Entry unchanged
  (launcher no-sync, `--controller local`, `--rl-optimizer-steps`); the only harness change is
  copying `<run dir>/events/*.jsonl` into the evidence (launch_run.sh).
- Criteria unchanged (attempt 5 section); criterion 1 counts `rl_round_trained` and
  criterion 5 reads `rl_engine_selected` from the returned tape; 5.5 check as declared.
- Runs: gspo_s2, gspo_s1, rpp, rpp_baseline, gdpo. Spent so far ~$4 of $20.

## Attempt 6 result (code fb588a4, launcher no-sync entry, event tape returned) -- 5/5 PASS
Per run (attempt6/<run>/criteria.json, produced by attempt6/check.py from launch.log + events/):
| run | app | finalized / rl_round_trained | errors | grad_norm finite | argv | unverified + outer_sync |
|---|---|---|---|---|---|---|
| gspo_s2 | ap-JDjymkEZox76D7QLx4gz3R | yes / 3 | none | yes | gspo, eps 3e-4/4e-4 | gspo, eps_clip, clip_higher; false |
| gspo_s1 | ap-qkl3KihH8uHgbQ9GEC2unv | yes / 3 | none | yes | same | same; false |
| rpp | ap-g0SuZBHPVMakOPo65qRpnf | yes / 3 | none | yes | kl_coef 0.01, normalize_advantages True, ref loaded | rpp, whiten; false |
| rpp_baseline | ap-gyfCgbhzYujHWxsejU6ly7 | yes / 3 | none | yes | same | rpp_baseline, whiten; false |
| gdpo | ap-0qCQ2iFAbH848CZb8NjVVa | yes / 3 | none | yes | dispatcher | gdpo, custom_reward_postprocess; false |
All on "NVIDIA H100 80GB HBM3". Launcher rc 2 each (pre-declared notice). No zero-gradient events.
- GSPO clip path (per optimizer step, Miles log): gspo_s2 (clipfrac, grad_norm) =
  (0.0, 0.900) (0.1875, 0.307) | (0.0, 0.646) (0.5, 0.528) | (0.0, 0.478) (0.5, 0.246);
  gspo_s1: clipfrac 0.0 in all 3 rounds (expectation D1 held: one step -> ratio ~1).
- 5.5 criterion (declared in attempt 5): gdpo `rl_round_trained.nonzero_advantages` per round
  = 32, 24, 32; dispatcher `rl_advantage_transform` for rollout_id 0, 1, 2 = 32, 24, 32. Match,
  no missing value -> PASS (R2 channel correct on GPU).
- Observation / defect (INFRA R1): `masked_fraction` and `clip_fraction` are null in every
  `rl_round_trained` event, including gspo_s2 whose Miles log shows pg_clipfrac 0.5 --
  the per-step pg_clipfrac does not reach the round event on the real engine.
- Teardown: all apps stopped / 0 tasks (attempt6/teardown.txt); watchdogs killed.
  Attempt 6 ~ 5 x 8.5 min H100 ~ $3.9; change total ~ $8 (not billing-confirmed), cap $20.
