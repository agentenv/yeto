# G3 rerun plan (task 7.5), committed before launch, 2026-09-29

Same as ../2026-09-29-g3/plan.md (topology: two Modal H100! islands plus a local syncer on port 29410,
checked free; strict-avg; TIS spec tis.json; model and batch settings unchanged; criteria 1-5; reclaim; cost cap $8), except:

- Code: YETO_SHA (after merging origin/algo-cap 4373cd9: every Modal ports island's tape is returned to
  `<run dir>/events/<island>.jsonl`; a tape without rl_learner_finalized becomes .incomplete and the
  launcher exits with 3).
- Evidence source: island tapes = the launcher-returned `$YETO_RUNS_DIR/algo1a-g3b/events/algo1a-g3b-l{0,1}-modal.jsonl`
  (the container puller is only a backup and is not used for the verdict). Syncer tape = local ~/yeto-output/yeto-tape.jsonl.
- Exit code: only 0 passes criterion 1 (key renamed `1_rc_ok`). Exit 3 means incomplete evidence and counts as a failure;
  exit 2 also fails here because a run with a syncer delivers the checkpoint.
- Why rerun: attempt 2 of ../2026-09-29-g3 could not establish criteria 3 and 4 (truncated container pull). The launcher
  now returns the tapes. This is a single rerun, not a rerun until pass.
- IcePop two-island: required by 7.5, but a multi-island run may use declared mechanisms only
  (D11), and IcePop (corrections:custom + corrections:icepop) is not declared on the Miles adapter.
  It is therefore not run in this plan and stays pending until the integration branch declares it.
  Its run will get its own committed plan with the same criteria.
- Est. 2x H100 ~20 min, ~ $3.
