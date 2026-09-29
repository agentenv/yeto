# G1 / G2 plan (written and committed before any launch, 2026-09-29, Agent ALGO-1a)

- Tasks: 7.1 (prep), 7.2 (G1, 1 GPU), 7.4 (G2, 1 GPU), 5.1 (import check of Miles
  `examples/.../mis.py` in the runtime image, CPU-only command in the same sandbox), 7.6 (teardown).
  G3 (7.5, two islands) is NOT in this round.
- Code: yeto branch `algo-1a` at the commit recorded in `YETO_SHA` (git archive upload).
  Image: `MILES_NEXT_IMAGE` = ghcr.io/michaellchung/yeto-miles-ports@sha256:5da40a07dabb3ea3fcf921efb4b2a21ca1178220bde40dc178c79b734fdaa540
  (Miles 0394715 at /root/miles, SGLang 9f29303). Pull via in-memory Modal Secret.from_dict built from
  the ghcr entry of ~/.docker/config.json (never printed, no named secret).
- Model/harness: R0 smoke configuration (rl-engine-ports evidence 2026-09-29-harness/smoke.py):
  Qwen/Qwen3-0.6B @ c1899de, LoRA r16 all-linear, gsm8k, 4 groups x 8 samples, 1 optimizer step
  per round, response len 384, seed 17, single island, no syncer, serial colocated
  (LoRA publish = CUDA IPC). `--rl-allow-unverified-mechanism` for exactly the mechanisms of each run
  (single island, design D11); nothing is declared on the adapter by this round.
- Cloud: Modal app `algo1a-g1`, ONE sandbox, GPU `H100!` (1x; no H200 upgrade; GPU name asserted
  to contain "H100" before any run), cpu 16, mem 128 GiB.
  Reclaim: sandbox `timeout=6000` s (Modal-side hard stop); driver wraps everything in
  try/finally -> `sb.terminate()`; independent local watchdog `timeout 6600` + trap that runs
  `modal app stop -y algo1a-g1`; after the run `modal app list` evidence of 0 tasks.
  No volumes, no named secrets.
- Estimated time: pull ~5 min + 6 G1 runs x ~9 min + G2 ~20 min = ~80 min.
  Cost estimate: H100 ~$3.95/h x 1.4 h + CPU/mem ~ $1.5 => ~ $7; budget cap $15.
- Runs (sequential, same sandbox; per-run worker timeout 1500 s; ROUNDS=3 for G1):
  | run | spec (correction) | allowances |
  | --- | --- | --- |
  | g1-observe | custom, observe_mismatch, mismatch_metrics | custom, mismatch_observe, mismatch_metrics |
  | g1-tis | tis, tis_clip 2.0, tis_clip_low 0.0 | tis |
  | g1-icepop | custom, icepop_function, [0.5, 5.0], mismatch_metrics | custom, icepop, mismatch_metrics |
  | g1-opsm-trainer | opsm, delta 1e-4, source trainer | opsm, opsm_trainer |
  | g1-mis | custom, vendored MIS, token, truncate, upper 2.0 | custom, mis |
  | g1-mis-mask | custom, vendored MIS, token, mask, [0.5, 2.0] | custom, mis_mask |
  | g2-observe | as g1-observe, ROUNDS=20 | same | (only if g1-observe passes)
- G1 pass criteria per run (pre-declared; all must hold; check_g1.py, no manual judgement):
  1. worker rc 0 and exactly ROUNDS `rl_local_round` events, ROUNDS+1 `rl_publication` events, each
     publication with a `rl/policy_token`; no `rl_invariant_failed`/`rl_round_failed`/
     `rl_algorithm_mismatch` events and no `StrictRlInvariantError`/`PolicyIdentityError` in miles.log;
  2. `rl_engine_selected` has `rl/algorithm_spec_sha256` == local spec sha256 and
     `rl/unverified_mechanisms` == the allowances;
  3. every Miles `train/` step dict (ROUNDS of them) contains the mechanism's keys, all finite:
     observe: tis, tis_abs, mismatch_outside_0p5_5, tis_abs_p50/p90/p99, ois, train_rollout_kl, ess_ratio;
     tis / icepop: tis, tis_abs, tis_clipfrac, ois, train_rollout_kl;
     opsm-trainer: opsm_clipfrac, train_rollout_kl;
     mis / mis-mask: ois and >= 1 key starting with `mis_` (mis-mask: mis_tis_mask_fraction_low/high);
  4. every `rl_local_round` has finite grad_norm (a zero grad_norm would have failed the run under the
     current driver rule; no masked-round relaxation exists without p0-driver.patch).
  A run that fails is recorded as failed; it is rerun at most once and only after a diagnosed and
  committed fix (harness bugs only; mechanism code changes restart the plan).
- G2 (7.4): observe-only, ROUNDS=20, same config. Report per round: train_rollout_kl, tis_abs mean
  and p50/p90/p99 (micro-batch quantiles averaged by Miles), ess_ratio, mismatch_outside_0p5_5.
  Report regenerated from the extracted metrics jsonl by `report_g2.py`. Data only; no
  recommendation; states execution mode (colocated-serial, CUDA IPC publish), model and that results
  do not extrapolate to partitioned mode (NCCL broadcast), larger models or MoE.
- 5.1: in the same sandbox, `cd /root && python -c "import examples.infra_features.train_infer_mismatch_helper.mis"`
  with PYTHONPATH=/root/miles, and the same from /work/yeto, output recorded; license line from /root/miles/LICENSE.
YETO_SHA updated before any G1 run: header comment of vendored MIS corrected after the 5.1 check (no code change)
