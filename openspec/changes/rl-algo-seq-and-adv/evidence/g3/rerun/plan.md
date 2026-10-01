# 7.6 G3 rerun plan (the one allowed rerun; committed before launch, 2026-09-29, Agent ALGO-2a)

Same task, topology, spec, args and criteria as ../plan.md, with these declared changes:
- YETO_SHA = a602fa2 (integ-decl with P0's close-down fix: an island that finalized and then
  ends FAILED counts as success and does not trigger recovery; failure waits up to 60 s for
  finalized; no syncer restart after all finalized; exit 4 only when an island is given up).
  Checked out in a private worktree (/tmp/a2-g3). Example specs are current at a602fa2
  (`test_example_specs_are_current` 6 passed, make_examples.py no diff); rerun/maxrl.json is
  byte-identical to ../maxrl.json. Syncer sources unchanged since 4652f73 (same binary, sha in
  ../syncer_sha256.txt). Dry-run at a602fa2: 2 islands, outer_sync true, no unverified mechanisms.
- Checker (fixed before this launch, committed with this plan): criterion 4 counts 3
  `rl_local_round` and 3 `rl_round_trained` events per island (finite grad_norm), plus
  rl_learner_finalized and no failure events. Criteria text unchanged:
  1. syncer tape: 3 outer steps, responders 2, no stale;
  2. both tapes: same rl/algorithm_spec_sha256 == sha256(rerun/maxrl.json at YETO_SHA), no unverified;
  3. rl_publication hash equal on both islands for policy_version 0..3;
  4. as above.
- Exit code reading: 0 or 2 -> the tapes decide (criteria 1-4 must hold); 3, 4 or any other
  non-zero -> FAIL (no override by the tapes).
- Watchdog: hardened (harness/watchdog.sh: own name, setsid+nohup, ignores HUP/INT/TERM, 60 s
  heartbeat to rerun/watchdog.log, harness re-checks every minute and restarts with the same
  deadline). Hard limits: head `timeout 3000`; watchdog deadline launch+3300 s stops app
  yeto-algo2a-g3 and the local syncer; trap cleanup on exit.
- Port: 29400 and 29410 in use; 29420 checked free by the harness right before launch
  (rerun/port_check.txt); abort if busy.
- Evidence: rerun/ (launch.log, events/*.jsonl, yeto-tape.jsonl, check.json, app list, port).
- Single run; no further rerun. Cost ~ $3 (change total so far ~ $11, cap $20).
- Command: `cd evidence/g3 && bash harness/run_g3.sh rerun`, then
  `/tmp/yeto-venv/bin/python harness/check_g3.py rerun`.
