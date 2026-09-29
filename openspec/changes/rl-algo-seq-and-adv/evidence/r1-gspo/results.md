# R1 GPU recheck -- rerun result (YETO_SHA 501d71d; plan + fixed checker 5b0ae97, prefix note 9415033)

Run 21:33:40Z-21:42:49Z, 1x "NVIDIA H100 80GB HBM3", launcher exit code 2 (tape decides).
**Result (pre-declared fixed checker): PASS** (run/check.json):
| round | Miles pg_clipfrac (2 steps) | mean | clip_fraction | masked_fraction |
|---|---|---|---|---|
| 0 | 0.0, 0.1875 | 0.09375 | 0.09375 | 0.09375 |
| 1 | 0.0, 0.5 | 0.25 | 0.25 | 0.25 |
| 2 | 0.0, 0.5 | 0.25 | 0.25 | 0.25 |
3 rl_round_trained events, rl_learner_finalized present. The R1 channel (INFRA d9bf29c) fills
clip_fraction and masked_fraction on GPU. No full-clip round occurred: the D2 full-clip
relaxation remains not covered on GPU.
History: first run (first/) officially not passed (checker round-delimiter defect); a pre-launch
refusal (same-name tape guard, no cloud resource) in rerun-prelaunch-refused/.
Resources: app yeto-algo2a-r1-gspo2 ap-sEPPygSJmxCvTy4kz7Y1bk stopped 21:42:42Z, 0 tasks; watchdog
killed after teardown. ~9 min H100 ~ $0.9 (estimate).
