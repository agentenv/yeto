# CTTN Shadow Diagnostic v1

`cttn_shadow_v1` answers the CTTN oracle question without changing the
training trajectory. Every merge commits ordinary SGD at outer LR 0.28:

```text
d_t = g_t
theta <- theta - 0.28 g_t
```

An independent counterfactual buffer is maintained as
`b_{t+1} = 0.9 b_t + g_t`. At 32 deterministic, fragment-local stratified
sample points (eight for each of four fragments), the sidecar uses one shared
HVP/Lanczos sketch to compute both the matrix CTTN transverse vector and the
scalar-control vector. Neither vector is applied. Four global merges later,
the same fragment's new pseudo-gradient resolves

```text
A_t = z_t^T g_{t+4} / (||P_{g_t^perp}(b_t)|| ||g_{t+4}||).
```

Resolved samples are written into the event tape with
`cttn_shadow_sample_step`, `cttn_shadow_future_step`, bind, retention,
`ritz_max`, and matrix/scalar alignment fields. The comparison harness also
writes `cttn_shadow_analysis.json` and applies the frozen decision:

- `NO-GO`: at least 30/32 matrix samples bind, mean matrix retention is below
  0.20, and the mean matrix alignment is non-positive in at least 3/4
  fragments.
- `TRIGGER`: mean matrix alignment is positive in at least 3/4 fragments, at
  least 24/32 matrix alignments are positive, and the paired mean
  `matrix - scalar` alignment is positive.
- Otherwise: `INCONCLUSIVE`.

## Locked 32-point run

This is the H16, rank-16, seed-223 protocol. It uses learners on GPUs 0-3 and
the HVP sidecar on GPU 4. It runs one SGD trajectory; no CTTN action is ever
committed.

```bash
python scripts/compare_diloco.py \
  --model qwen35-9b \
  --data trl-lib/Capybara \
  --settings m4 \
  --baseline-loss 0.0 \
  --delta-correction none \
  --outer-optimizer cttn-shadow \
  --outer-lr 0.28 \
  --outer-momentum 0 \
  --token-budget 700000 \
  --seq-len 64 \
  --micro-batch-size 1 \
  --inner-lr 0.001 \
  --lora-r 16 \
  --lora-alpha 32 \
  --eval-rows 64 \
  --max-rows 5000 \
  --shuffle-rows-seed 223 \
  --training-seed 223223 \
  --device cuda \
  --gpu-slots 4 \
  --gpu-offset 0 \
  --action-probe-gpus 4 \
  --action-probe-anchor-manifest experiment-results/EXP2/anchor_capybara_11000_256_manifest.json \
  --action-probe-seq-len 64 \
  --action-probe-panels 8 \
  --action-probe-blocks-per-panel 1 \
  --action-probe-timeout-s 600 \
  --action-probe-startup-timeout-s 1800 \
  --cttn-shadow-samples 32 \
  --fixed-window-tokens 2048 \
  --fixed-window-microsteps 16 \
  --pad-to-fixed-window-tokens \
  --freeze-delta-before-delay \
  --learner-push-delay-ms 0,0,0,0 \
  --learner-delay-jitter-ms 0 \
  --syncer-total-steps 320 \
  --learner-max-steps 2500 \
  --strict-quorum \
  --arm-timeout-min 240 \
  --work-dir experiment-results/EXP2/cttn-shadow-seed223-r16-h16/work \
  --report-dir experiment-results/EXP2/cttn-shadow-seed223-r16-h16/report
```

To re-run only the frozen decision from a completed tape:

```bash
python scripts/analyze_cttn_shadow.py \
  experiment-results/EXP2/cttn-shadow-seed223-r16-h16/work/m4/tape.jsonl \
  --expected-samples 32
```
