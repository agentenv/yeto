# G3 result (TIS, two Modal islands + local syncer on :29410, strict-avg)

Attempt 1: harness path error before any cloud resource (plan.md). Attempt 2: ran 17:59:55Z to
18:16:58Z, head exit 0.

| criterion | result |
| --- | --- |
| 1 head rc 0; 3 outer steps with 2 responders and 0 stale | PASS (syncer tape run/yeto-tape.jsonl: base_version 0,1,2, responders 2) |
| 2 same algorithm sha on both islands, no unverified | PASS |
| 3 publication hash equal for v0..3 | NOT ESTABLISHED: equal for v0 and v1; island 1's pulled tape ends mid-line after v1 (the last 20 s pull raced the container exit), so v2/v3 are missing for island 1 |
| 4 3 finite rounds per island, no failure events | NOT ESTABLISHED for island 1 (2 rounds in its truncated tape; island 0 has 3) |
| 5 train/tis, tis_abs, tis_clipfrac for all 3 steps of both islands | PASS (launch log) |

**Verdict: G3 not passed.** The pre-declared criteria 3 and 4 cannot be evaluated from the collected
evidence. This is not a mechanism failure: the syncer tape shows 3 strict steps with both
islands. Nothing was relaxed. A rerun needs a committed harness fix for the final tape collection
(for example a faster pull and a final pull before teardown).

Resources: Modal app yeto-algo1a-g3, 2x H100 for ~17 min (~0.57 H100 h, ~ $2.3 estimate). Afterwards
`modal container list` shows no algo1a container, the app no longer shows in `modal app list`
(run/app_list_final.txt), local syncer exited, port 29410 closed, watchdog killed.
