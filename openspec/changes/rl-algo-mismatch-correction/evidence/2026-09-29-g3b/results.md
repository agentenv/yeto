# G3 rerun (TIS, two Modal islands + local syncer :29410), 19:30:20Z to 19:44:18Z, head exit 0

| criterion | result |
| --- | --- |
| 1 rc_ok (exit 0); 3 outer steps with 2 responders and 0 stale | PASS |
| 2 same algorithm sha, no unverified | PASS |
| 3 publication hashes equal for v0..v3 across islands | PASS (launcher-returned tapes, both with rl_learner_finalized) |
| 4 3 finite rl_local_round per island, no failure events | FAIL: the launcher-returned tapes (the declared evidence source) contain no rl_local_round events at all (per island: 30 driver_phase, 4 policy_apply, 4 publication, 1 finalized) |
| 5 tis metrics each step, both islands | PASS |

**Verdict: not passed.** Criterion 4 cannot be met from the declared source. The backup container
pulls (island-*.jsonl, not used for the verdict) do contain rl_local_round events (2 per island before
truncation), so the launcher's event return (P0 4373cd9) appears to drop rl_local_round. That is
reported to P0; nothing was relaxed.
Resources: 2x H100 ~14 min (~0.47 H100 h, ~ $1.9 estimate); no algo1a container left, port 29410 closed.
