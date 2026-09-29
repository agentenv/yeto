# G1 re-verification through the launcher entry (plan committed before launch, 2026-09-29, ALGO-1a)

Reason (review G-2): round 1 (evidence/2026-09-29-g1) verified tis, icepop and opsm-trainer with
code bc7a330, bare allowance names and the benchmark-worker entry, without criterion 5. Those runs
were the basis of the 1a-declare declarations (tis, opsm, opsm_trainer). Code and entry have changed
since (qualified names, D11 refusal with outer sync, `--rl-single-island-no-sync`), so the three
mechanisms are re-verified once on the current code through the real entry
`yeto launch --rl-single-island-no-sync --rl-allow-unverified-mechanism <dimension:name>`.
This is the first real GPU use of that entry and is recorded separately. It is not a rerun until
pass: every run happens exactly once; a failure is reported and the matching declaration withdrawn
(the coordinator forwards that to ALGO-CAP). Harness bugs may be fixed and rerun once after a
committed diagnosis.

- Code: algo-1a at YETO_SHA (the launcher uploads the working tree of a clean `git archive` of
  YETO_SHA plus harness/gsm8k_reward.py; the upload time is recorded in each run's start_time.txt).
- Image: MILES_NEXT_IMAGE (ghcr.io/michaellchung/yeto-miles-ports@sha256:5da40a07...), registry login
  from ~/.docker/config.json exported in-process as SKYPILOT_DOCKER_* (never printed).
- Command (per run; prefix algo1a-g1c-<run>):
  `yeto launch --training-mode rl --rl-engine ports --gpu modal:1xh100 --modal-gpu-exact
   --controller local --rl-single-island-no-sync --rl-algorithm-spec <spec.json>
   --rl-allow-unverified-mechanism <each qualified name> --cluster-prefix algo1a-g1c-<run>
   --model Qwen/Qwen3-0.6B --model-revision c1899de289a04d12100db370d81485cdf75e47ca
   --data zhuzilin/gsm8k --data-revision 0cbd9f31d91ac21a7613dcbc7fef992adac459ae
   --reward-function gsm8k_reward:score --tuning lora --lora-r 16 --lora-targets all-linear
   --total-steps 3 --fragments 1 --pipeline 1 --rollout-batch-size 4 --n-samples-per-prompt 8
   --rollout-max-response-len 384 --seq-len 1024 --inner-lr 1e-5 --seed 17
   --apply-chat-template-kwargs '{"enable_thinking": false}' --trust-remote-code`
  (the R0 default-params-modal command, with the no-sync entry instead of a syncer).
- Runs: tis {tis_clip 2.0, low 0.0; corrections:tis}; icepop {[0.5,5], mismatch_metrics;
  corrections:custom, corrections:icepop, features:mismatch_metrics}; opsm-trainer {delta 1e-4,
  trainer; corrections:opsm, corrections:opsm_trainer}. Specs are identical to rounds 1/2.
- Pass criteria (all; check_g1.py with no-sync plus receipt check; no loosening):
  1-4 as in ../2026-09-29-g1/plan.md (launcher exit 0 replaces worker rc; miles.log = launch log;
  events = the island tape pulled from /root/yeto-output/rl-island-0.jsonl);
  5. rl_engine_selected has rl/outer_sync == false and rl/contains_unverified_mechanisms == true;
  6. (new, explicit receipt check) the 3 rl_local_round events have local_round_id 1,2,3,
     base_policy_version 0,1,2, completed_trajectories == 32 and action_tokens > 0; publications carry
     policy_version 0..3 in order. Receipts themselves are not emitted as events; this checks the
     receipt-derived round stats.
- Reclaim: `timeout 2700` around each launch; trap runs `yeto down`/`modal app stop -y
  yeto-algo1a-g1c-<run>` on any exit; independent watchdog (setsid) stops the app after 3000 s;
  after each run `modal app list` must show it stopped with 0 tasks.
- GPU: 1x H100! per run (asserted by --modal-gpu-exact). Estimate 3 x ~15 min = ~0.75 H100 h, ~ $3.5;
  cap $8.
- 7.1: `yeto launch` has no --dry-run; the 1-GPU resource request is recorded from the launch log
  ("gpu=H100!" in the Modal function definition) instead, labelled as such.

## tis attempt 1 (harness failure, no cloud resource created)
- `yeto launch` ran in /tmp/yeto-venv, which has no `sky`: `ModuleNotFoundError: No module named 'sky'`
  at launcher.py:3626 before any Modal call (runs/tis-attempt1-nosky). Fix: run the launcher with
  /home/michael/work/gpu-head/venv/bin/python (the venv the R0 Modal runs used), PYTHONPATH = the
  archived tree. Rerun once.
