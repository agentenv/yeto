# G3 rerun 3: TIS and IcePop two islands (task 7.5), committed before launch, 2026-09-29

Everything as ../2026-09-29-g3b/plan.md (topology, local syncer :29410, criteria 1-5 with `1_rc_ok`
= exit 0 only, exit 2/3 fail, evidence = launcher-returned tapes `<run dir>/events/*.jsonl`, reclaim,
cost cap $8 per run), except:
- Code: YETO_SHA = algo-1a after merging origin/integ-decl 5b94cf7 (single tape writer echoes every
  event, including rl_local_round).
- Two runs, sequential, each only with per-user threads <= 3296 and port 29410 free:
  `tis` (tis.json, same spec as before) and `icepop` (icepop.json: IcePop [0.5, 5.0], **without**
  mismatch_metrics, because features:mismatch_metrics is being withdrawn and a multi-island run may
  use declared mechanisms only; required mechanisms: corrections:icepop plus the defaults).
- Criterion 5 for icepop reads the same keys (train/tis, train/tis_abs, train/tis_clipfrac), which
  IcePop reports whenever it runs (use_tis is emitted).
- One attempt each; a failure is reported as it is.
