# 7.6 G3 rerun result (YETO_SHA a602fa2; plan rerun/plan.md, committed 1698744 before launch) -- PASS

Run 20:52:07Z-21:07:53Z. Head exit code **0**. Checker (fixed before launch) -> rerun/check.json:
| criterion | result |
|---|---|
| exit code (0/2 -> tapes decide) | 0 |
| 1 syncer: 3 outer steps, responders 2, no stale | PASS (base_version 0,1,2; responders 2) |
| 2 same algorithm sha, no unverified | PASS |
| 3 rl_publication hash equal v0..3 | PASS (v0 373abf70..., v1 2adc67b1..., v2 84b1dc6d..., v3 8212b241... on both islands) |
| 4 3 rl_local_round + 3 rl_round_trained finite, finalized, no failure events | PASS |
Both island jobs SUCCEEDED; both learners finalized; no recovery, no syncer restart.
Observations: grad_norm island 0 = 0.288 / 0.757 / 0.515, island 1 = 0.285 / 0.754 / 0.553;
nonzero_advantages per round 16 / 24 / 16 on both islands (rollouts identical at the same policy
and seed on both islands). 0 unparsable tape lines.
Watchdog: hardened script ran from launch (pid 2194855, own pgid), 16 heartbeats, no restart
needed, killed by the harness after teardown (the "Killed" line in the driver output is that kill).
Resources: app yeto-algo2a-g3 ap-aivVXMrZwnfkkB7zj2rQFm, 2x H100 (--modal-gpu-exact), created
~20:52Z, stopped 21:07:56Z, 0 tasks (app_list_final.txt); local syncer gone, port 29420 closed.
Cost ~ 2 x 16 min H100 + CPU ~ $2.9 (estimate, not billing-confirmed).
