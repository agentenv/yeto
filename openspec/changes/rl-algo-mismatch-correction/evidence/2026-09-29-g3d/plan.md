# IcePop two-island G3, single revised rerun (task 7.5), committed before launch, 2026-09-29

Allowed once by the coordinator after ../2026-09-29-g3c (IcePop criterion 5 FAIL by the checker: one
log line lost its island prefix). There is no further run after this one, whatever the result.
- Code: YETO_SHA = algo-1a after merging origin/integ-decl, which contains c098b5b (the launcher tears
  everything down on an island failure and exits 4) and INFRA 0c2cd99 (per-island rl_round_trained
  with a `mismatch` dict).
- Unchanged: topology (two Modal H100! islands, local syncer :29410 checked free), icepop.json
  (IcePop [0.5, 5], no mismatch_metrics), model and batch settings, criteria 1-4, evidence source
  for 1-4 (launcher-returned tapes), reclaim (timeout 3000 on the head, independent setsid watchdog
  that stops the app by name after 3300 s and kills the local syncer), cost cap $8.
- Exit code: only 0 passes; 2, 3 and 4 fail.
- Criterion 5, source revised before this run: on each island, all 3 rl_round_trained events in
  the launcher-returned tape carry a `mismatch` dict containing tis, tis_abs and tis_clipfrac
  (with or without a `train/` prefix). If that does not hold on either island, the source falls back
  to island-prefixed log lines for steps 0-2 of both islands. Unprefixed lines are not counted, and
  the run may fail again. The checker records which source it used.
