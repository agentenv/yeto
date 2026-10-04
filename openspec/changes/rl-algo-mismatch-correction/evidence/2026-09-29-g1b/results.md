# G1 round 2 + G2 results (sandbox sb-1kos5UHRiLNRFIevunVJT1, app algo1a-g1b, Modal H100! x1)

- GPU assert: `NVIDIA H100 80GB HBM3` (runs/gpu_name.txt). Code: YETO_SHA 3b6dfa9; harness from c8ec241.
- Entry: benchmark worker, single island, no syncer, `rl_single_island_no_sync=True`, qualified
  allowances; the harness ran the learner CLI's `_check_ports_algorithm_options(outer_sync=False)`
  before launching. The launcher-level `yeto launch --rl-single-island-no-sync` was not exercised.
- Created 17:09:01Z, terminated 17:36:54Z; app `algo1a-g1b` stopped with 0 tasks
  (modal_app_list_after_stop.txt); watchdog stopped; no volumes, no named secrets.
- Estimated cost: H100 ~28 min x ~$3.95/h = ~$1.85 + CPU/memory ~$0.35 = ~ $2.2 (not billing-confirmed).

| run | result (pre-declared criteria 1-5) | evidence |
| --- | --- | --- |
| g1-observe (3 rounds) | PASS | runs/observe/check.json |
| g1-mis (truncate, token, upper 2.0) | PASS | runs/mis/check.json |
| g1-mis-mask (mask, token, [0.5, 2.0]) | PASS | runs/mis-mask/check.json |
| g2-observe (20 rounds) | PASS; report generated | runs/g2-observe/{check.json,metrics.jsonl,report.md} |

MIS mask fractions were 0 in every round (no token outside [0.5, 2.0]). G2 data: train_rollout_kl
4.5e-4 to 7.8e-4 per round; tis_abs mean 0.0097 to 0.016; micro-batch p99 of tis_abs 0.12 to 0.15;
no token outside [0.5, 5] in any of the 20 rounds (runs/g2-observe/report.md).
