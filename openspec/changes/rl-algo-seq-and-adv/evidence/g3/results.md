# 7.6 G3 result (MaxRL, two Modal islands + local syncer :29420, strict-avg, YETO_SHA 4652f73)

Run 20:18:04Z-20:39:07Z (run/start_time.txt, end_time.txt). Head exit code **143**.

| criterion (plan.md) | result |
|---|---|
| exit-code rule: 0 pass, 2 -> tapes decide, 3/other -> FAIL | **143 -> FAIL** (see below) |
| 1 syncer: 3 outer steps, responders 2, no stale | PASS (base_version 0,1,2; responders 2) |
| 2 same algorithm sha on both islands, no unverified | PASS |
| 3 rl_publication hash equal for v0..3 | PASS (v0 373abf70..., v1 343af438..., v2 4750813e..., v3 e736949b... on both) |
| 4 3 finite rounds, rl_learner_finalized, no failure events | **NOT ESTABLISHED by the pre-declared script** (run/check-as-declared.json: false). Observation only: by event count each tape has 3 rl_local_round + 3 rl_round_trained (finite grad_norm), rl_learner_finalized, no failure event |

**Verdict (official): 7.6 not passed** -- exit code 143, and criterion 4 not established by the pre-declared script. Criteria 1-3 hold.
What happened (launch.log lines 11389-11652): the syncer logged "training complete after 3
outer steps" with both learners finalized at 20:36:09Z; island 0's job SUCCEEDED; the local
syncer was then terminated (exit 143) while island 1 was still shutting down; island 1 printed
`learner 1 finalized` and then `KeyboardInterrupt` inside `ray.shutdown`, its Modal job ended as
FAILED, and the launcher started "recovery", restarted the syncer and retried relaunching
island 1 in a loop (the app was already stopped). This is the launcher hang after an island
failure that P0 fixed from integ-decl c098b5b (exit code 4); 4652f73 does not contain it. I
stopped the hung head process by hand at ~20:39Z (hence 143); the harness then ran its cleanup.

Checker note: harness/check_g3.py as committed before launch counted rounds by distinct
`rollout_id`; `rl_local_round` has no rollout_id, so it reported criterion 4 false
(run/check-as-declared.json) -- this is the official result. The post-run count by events
(run/check.json) is a change of judging method after the fact and is an observation only.

Observations: rounds 0/1 grad_norm 0.288/0.296 (island 0) and 0.285/0.785 (island 1). Round 2
on both islands: nonzero_advantages 0 (every group all-right or all-wrong under MaxRL) and
grad_norm 0.0 with no invariant failure -- the "no gradient expected" branch exercised on GPU.
rl_advantage_transform: 3 per island.

Resources: Modal app yeto-algo2a-g3 (ap-z1Tie8sBSPMxHwpMN0pGrY), 2x H100 ("H100!" via
--modal-gpu-exact) for ~18 min of GPU time (app created ~20:18Z, stopped 20:36:20Z). Estimate
~0.6 H100 h + CPU ~ $3 (not billing-confirmed). After: app stopped, 0 tasks
(run/app_list_final.txt); local syncer for :29420 gone, port 29420 closed (run/port_after.txt).
The first watchdog (pid in run/watchdog.pid) was found gone mid-run (cause not determined); a
replacement watchdog with the same deadline was started (run/watchdog2.txt) and killed after
teardown. Remaining yeto-syncer / watchdog processes on this host belong to algo-1a.

Rerun (needs a new committed plan): base on an integ-decl SHA >= c098b5b (exit code 4 = FAIL
added to the reading); same criteria.

## Watchdog disappearance (investigation) -- RESOLVED (see correction below)
- The watchdog was `setsid nohup bash -c "sleep 3300; modal app stop -y yeto-algo2a-g3; pkill -f ..."`
  (pid 1836651); at 20:38Z the pid no longer existed, while the head (timeout 3000) was still running.
  Nothing in this change's scripts kills it before the run ends. Other agents on the host run
  identical-looking watchdogs (`bash -c "sleep 3300; ... modal app stop ..."`, algo-1a G3 harnesses)
  and tear them down; a pattern-based kill (`pkill -f "sleep 3300..."` or similar) by another
  session would match ours too. Not confirmed (no log of the kill) -- cause undetermined.
- Hardening for the rerun (harness/run_g3.sh, harness/watchdog.sh; committed before any rerun):
  a separately named script `/tmp/algo2a/algo2a-g3-watchdog.sh` (no `sleep 3300` in its command
  line), started with setsid + nohup in its own session/process group, ignoring HUP/INT/TERM (only
  SIGKILL stops it), writing a heartbeat line every 60 s to run/watchdog.log; the harness checks it
  every minute and restarts it with the same deadline if it is gone, and logs that.

### Correction (2026-09-29, after the main agent's notice) -- revised after final review
- algo-1a reported running `pkill -x -f "sleep 3300"` at about 20:37-20:40Z (algo-1a progress, 7ebfc59).
- **The timing does not match that window**: this run's local syncer was "Terminated" at
  20:36:09-20:36:14Z (launch.log) and the Modal app's stopped_at is 20:36:20Z -- both before
  20:37Z. The order is also the reverse of the watchdog's commands (watchdog: `modal app stop`
  first, then `pkill` of the syncer; observed: syncer terminated first, app stopped after).
- Conclusion: the failure was most likely triggered by an external process being ended (the
  syncer terminated, then island 1's `KeyboardInterrupt` in `ray.shutdown`, job FAILED, launcher
  recovery loop -- the loop being the launcher defect fixed in a602fa2). Attribution to algo-1a's
  pkill is **not established**. The watchdog's disappearance itself remains consistent with that
  pkill but is also unproven.
- The official result of this run stays as pre-registered: **not passed** (exit 143; criterion 4
  not established by the pre-declared script). The one allowed rerun is reported in rerun/results.md.
