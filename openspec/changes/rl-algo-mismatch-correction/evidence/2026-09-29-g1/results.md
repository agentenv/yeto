# G1 round 1 results (sandbox sb-rHvfMERF2KVJArErjzeMps, app algo1a-g1, Modal H100! x1)

- GPU assert: `NVIDIA H100 80GB HBM3` (runs/gpu_name.txt). Image manifest: runs/image-manifest.json.
- Code: YETO_SHA (bc7a330). Entry: benchmark_rl worker, single island, syncer=None (no outer sync),
  allowances via `rl_allow_unverified_mechanism` (P0 naming at that commit: bare mechanism names).
- Created 16:43:43Z, terminated ~17:05Z (terminated_at.txt); app stopped with 0 tasks
  (modal_app_list_after_stop.txt); watchdog stopped; no volumes or named secrets created.
- Estimated cost: H100 ~22 min x ~$3.95/h = ~$1.45; CPU/memory ~$0.3; total ~ $1.8 (not billing-confirmed).

| run | result | evidence |
| --- | --- | --- |
| g1-observe | not valid: attempt 1 failed in the harness (HF `RemoteProtocolError` loading gsm8k), before any Miles process | runs/observe.driver.log |
| g1-tis | PASS (all pre-declared checks) | runs/tis/check.json |
| g1-icepop | PASS | runs/icepop/check.json |
| g1-opsm-trainer | PASS | runs/opsm-trainer/check.json |
| g1-mis, g1-mis-mask, observe rerun, G2 | not started: the coordinator paused G1 runs that need the unverified-mechanism switch until P0 provided the single-island entry without a syncer | — |

Observed values (data only): train_rollout_kl 5.7e-4 to 7.9e-4 per round, tis_abs 0.012 to 0.017,
tis_clipfrac 0 (every ratio was inside [0, 2] and [0.5, 5]), opsm_clipfrac 0 (1 optimizer step per
round, so pi_theta = pi_old and OPSM cannot trigger). TIS and IcePop gave identical losses and
grad norms because no token was clipped or masked.
