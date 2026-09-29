# R1 channel GPU recheck (narrows known deviation 1; committed before launch, 2026-09-29)

- Purpose: verify on GPU that INFRA's R1 channel (d9bf29c) fills `clip_fraction` and
  `masked_fraction` of each `rl_round_trained` event for GSPO. It does NOT try to trigger a
  fully clipped round; the D2 full-clip relaxation remains "not covered on GPU".
- Code: YETO_SHA = integ-decl 501d71d (contains d9bf29c), private worktree /tmp/a2-r1 (run.sh
  refuses to start otherwise). Example specs current at that SHA (test_example_specs_are_current
  passes).
- Run: exactly attempt 6's gspo_s2 configuration: `yeto launch --rl-single-island-no-sync
  --controller local --rl-optimizer-steps 2 --gpu modal:1xh100 --modal-gpu-exact`, spec
  examples/gspo.json (eps 3e-4/4e-4, formally declared at 501d71d, no allowance), Qwen3-0.6B
  c1899de, zhuzilin/gsm8k 0cbd9f3, reward gdpo_reward:correctness_reward, LoRA r16, 3 rounds x 4x8,
  response 384, seq 1024, lr 1e-5, seed 17. One run, no rerun.
- Criteria (all; check.py, committed now):
  1. exit code 0 or 2 -> the tape decides; 3, 4, 5 or any other non-zero -> FAIL;
  2. 3 `rl_round_trained` events and `rl_learner_finalized` in the returned tape;
  3. for each round: `clip_fraction` and `masked_fraction` are non-null and each equals the mean
     of that round's two Miles per-step `train/pg_clipfrac` values within 1e-6 (the adapter
     aggregates with equal step weights, as upstream reports no per-step token count; rounds
     are delimited by Miles' step 0).
- Reclaim: `timeout 2400` around the launcher; hardened watchdog (own script name, setsid+nohup,
  ignores HUP/INT/TERM, heartbeat log) stops app yeto-algo2a-r1-gspo at launch+2700 s; after the
  run `yeto down` + `modal app stop` + app list proof.
- Cost ~ 10 min 1x H100 ~ $0.9; change total so far ~ $14, cap $20.

## Rerun (the one approved rerun; committed before launch)
- First run: official result not passed (checker defect: rounds delimited by Miles "step 0", but
  Miles step ids are cumulative). Evidence moved to first/.
- Fixed checker (committed with this section): Miles per-step `train/pg_clipfrac` values in log
  order, step ids asserted to be 0..n-1; rounds = consecutive groups of `--rl-optimizer-steps`
  (2) steps, matched to `rl_round_trained` sorted by rollout_id. All other criteria unchanged
  (exit 0/2 -> tape decides, 3/4/5/other FAIL; 3 rounds + finalized; per round clip_fraction and
  masked_fraction non-null and equal to the round's step mean within 1e-6).
- Same SHA 501d71d, same run.sh/config. No further runs after this one. Cost ~ $0.9.
