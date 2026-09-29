# Elastic image smoke plan (written before launch, 2026-09-29)
- Image: ghcr.io/michaellchung/yeto-miles-ports:d002615-9f29303@sha256:9f0977dad5376b2cfd8bca7a15b2f78bd3ce7ffe199e5176c3cd6628ec7e6245
  (miles d002615f44816ae243169643dc093eb2c29029d8, sglang 9f29303; same base; pure-Python gate passed:
  27 changed files vs 9e4260d, all .py under miles/ or tests/, no deletions).
- Modal app `img-smoke`, one Sandbox, 1x L40S, cpu 8 / 64 GiB, Sandbox timeout=2400 s, local `timeout 3000`,
  `finally: terminate` + `modal app stop -y img-smoke`, independent watchdog (55 min).  Estimate < $1, budget $5.
- Script: ../smoke_in_image.sh at this commit (pins read from yeto/rl; import check uses PYTHONPATH=/root/miles).
- Pass = all checks PASS: L40S; manifest == pins; git identity; imports -> forks (+ run_plugin);
  launcher ports setup reuses image forks; upstream parse_args tests (test_rl_miles_adapter_config.py,
  test_rl_argv_snapshot.py) pass.  Any failure: diagnose before any rerun.

## Run 1 (sb-uvZmdwRnk9jXk3kLZ0U7Ts, L40S, 580 s): all PASS except parse_args_tests (1 failed, 46 passed, 1 skipped)
The script kept only the last 15 pytest lines, so the failing test id was lost (script defect).
A CPU-only rerun (sb-RhFig36ejb7t7uFPOYFNIc) is not informative (upstream parse_args needs libcuda).
Diagnostic run 2: the same two test files only, on 1x L40S, with -rfs --tb=short and full output --
to identify the failure, not to retry it into passing; its result stands either way.

## Run 2 result (sb-t1GlvtiBBvlpmDbnCvZmOK, L40S): 46 passed, 1 skipped, 1 failed
FAILED tests/test_rl_miles_adapter_config.py::test_upstream_parse_args_accepts_translation[False]
  MilesConfigError raised by yeto/rl/engine/miles_adapter/config.py:527 (translate_run_config):
  "a LoRA fixed partition publishes over NCCL broadcast every round; the trainer must stay
  resident (no offload_train)".  This fires in yeto's own translation before any Miles call
  (parse_miles_args is never reached): the non-colocated test fixture keeps offload_train while
  the INFRA guard (b618d8d / 63bedc4) refuses it.  Image-independent (would fail on any image);
  owner INFRA (miles_adapter + its test).  [True] (colocated) passes on the elastic Miles.
Conclusion: image smoke PASS on every image check; parse_args: colocated PASS, non-colocated
blocked by a yeto test/guard inconsistency, not by the image.

## Cost / cleanup
L40S: 580 s + ~300 s (run 2 incl. cached pull); CPU 4c/16GiB ~ 3 min.  Estimate ~ $0.6.
All sandboxes terminated in `finally`, `modal app stop -y img-smoke` after each; app list: all
img-smoke apps stopped (modal_app_list_after_stop.txt); watchdog killed.  No volumes/secrets.
