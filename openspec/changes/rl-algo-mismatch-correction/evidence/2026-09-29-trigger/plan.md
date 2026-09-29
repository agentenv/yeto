# Trigger plan: does each correction actually take effect on GPU? (committed before launch, 2026-09-29)

Why (coordinator's declaration policy): G1 ran with natural mismatch only (max |ratio-1| ~ 0.15),
so no clip/mask branch fired, and with 1 optimizer step OPSM cannot fire. The declarations tis /
opsm_trainer are kept but labelled "G1 proves only that it runs". This plan uses the smallest
configuration that should make each branch fire. The thresholds are test settings, not
recommendations.

- Entry: `yeto launch --training-mode rl --rl-engine ports --gpu modal:1xh100 --modal-gpu-exact
  --controller local --rl-single-island-no-sync --rl-allow-unverified-mechanism <dimension:name>`,
  otherwise the g1c command (Qwen3-0.6B LoRA, 4x8, 3 rounds, seed 17). Code: YETO_SHA.
- Runs (one at a time, per-user threads <= 3296 before each):
  | run | spec | steps/round | allowances |
  | --- | --- | --- | --- |
  | tis | TIS clip [0.99, 1.01] | 1 | corrections:tis |
  | icepop | IcePop [0.99, 1.01], mismatch_metrics | 1 | corrections:custom, corrections:icepop, features:mismatch_metrics |
  | mis-mask | MIS token mask [0.99, 1.01] | 1 | corrections:custom, corrections:mis_mask |
  | opsm-trainer | OPSM delta 1e-6, trainer | 2 (`--rl-optimizer-steps 2`) | corrections:opsm, corrections:opsm_trainer |
- Validity criteria (as g1c 1-6): launcher rc 0, or 2 with the "is not fetchable over ssh" line;
  Miles train steps 0..3*steps-1 in the streamed log; tape (pulled every 5 s) has rl_local_round
  1..3 with 32 trajectories and finite grad_norm, publications v0..3 with policy tokens, no
  failure events; rl_engine_selected sha == spec, allowances == the table, rl/outer_sync false;
  effect metrics finite.
- EFFECT criterion, fixed before the run: in at least one logged train step
  tis: train/tis_clipfrac > 0; icepop: train/tis_clipfrac > 0; mis-mask:
  train/mis_tis_mask_fraction_low + _high > 0; opsm-trainer: train/opsm_clipfrac > 0.
  If a run is valid but the effect is 0, that is reported as-is (declaration decision left to the
  coordinator). A run whose zero-gradient invariant fires is reported as a failure (it would mean
  the strict rule and the masking disagree); nothing is relaxed.
- Reclaim/cost: as g1c (timeout 2700 s per launch, trap app stop, setsid watchdog 3000 s, app list
  proof). ~4 x 12 min H100 = ~0.8 H100 h, ~ $3.5; cap $8. Single attempt per run; harness failures
  before an app exists may be fixed once after a committed note.
