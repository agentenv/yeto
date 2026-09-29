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
