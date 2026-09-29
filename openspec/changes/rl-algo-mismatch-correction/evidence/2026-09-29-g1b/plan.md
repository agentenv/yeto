# G1 round 2 + G2 plan (written and committed before launch, 2026-09-29, Agent ALGO-1a)

Same as ../2026-09-29-g1/plan.md (model, image, harness, pass criteria 1-4, G2 report content, cost
rules, reclaim) except:

- Code: `algo-1a` at YETO_SHA (after merging origin/algo-cap 2f9f02c: qualified allowance names
  `dimension:name`, D11 refusal with any outer sync, `--rl-single-island-no-sync`).
- Entry: the same benchmark-worker harness (single island, no syncer) with
  `rl_single_island_no_sync=True` in the learner arguments, and the harness calls the learner CLI's
  startup check `_check_ports_algorithm_options(args, outer_sync=False)` before launching (the worker
  path skips CLI parsing). The launcher-level `yeto launch --rl-single-island-no-sync` entry is NOT
  exercised by this round (recorded as not done).
- Extra pass criterion 5: `rl_engine_selected` carries `rl/outer_sync == false` and
  `rl/contains_unverified_mechanisms == true`.
- Harness fix from round 1: model and dataset pre-downloaded with retries (predownload.py).
- Runs, in order: g1-observe (3 rounds), g1-mis (3), g1-mis-mask (3), g2-observe (20 rounds; its
  checks are criteria 1-5 at 20 rounds; report.md/metrics.jsonl by report_g2.py).
  tis / icepop / opsm-trainer are not rerun (round 1 passed on the same mechanism code).
- Cloud: Modal app `algo1a-g1b`, one sandbox, `H100!` x1 (GPU name asserted), cpu 16, mem 128 GiB,
  sandbox timeout 6000 s + local watchdog 6600 s + app stop; est. ~60 min, ~ $5; cap $12.
