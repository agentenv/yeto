# R1 GPU recheck result (YETO_SHA 501d71d; plan committed f6e25b4 before launch)

Run 21:21:13Z-21:30:45Z, 1x H100 ("H100!" via --modal-gpu-exact), launcher exit code 2
(no-sync notice; the tape decides). Learner finalized; tape returned.

**Official result (pre-declared check.py): NOT PASSED** -- run/check.json: `three_rounds` false and
`per_round_non_null_and_consistent` false. Cause: the plan delimited rounds by Miles "step 0",
but Miles numbers train steps cumulatively over the run (step 0..5 for 3 rounds x 2 steps), so the
script put all six per-step values into one "round" and compared only round 0 with their mean.
This is a defect of the pre-declared judging script; per the rules the result is not changed.

Observation only (not a pass):
| round | Miles pg_clipfrac (steps 2k, 2k+1) | pair mean | rl_round_trained clip_fraction | masked_fraction |
|---|---|---|---|---|
| 0 | 0.0, 0.1875 | 0.09375 | 0.09375 | 0.09375 |
| 1 | 0.0, 0.5 | 0.25 | 0.25 | 0.25 |
| 2 | 0.0, 0.5 | 0.25 | 0.25 | 0.25 |
All non-null and equal to the per-round mean -- consistent with the R1 channel working on GPU.
No full-clip round occurred (clip fraction <= 0.25), so the D2 full-clip relaxation is still not
covered on GPU.

Resources: app yeto-algo2a-r1-gspo ap-xbkY9rLSJ3vjXCBmcuSrVi stopped 21:30:38Z, 0 tasks; hardened
watchdog ran with heartbeats and was killed by run.sh after teardown ("Killed" line). ~9.5 min
H100 ~ $0.9 (estimate). Change total ~ $15 (estimate, not billing-confirmed), cap $20.
