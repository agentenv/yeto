# Miles 0af62f4 image smoke plan (written before launch, 2026-09-29)
- Image ghcr.io/michaellchung/yeto-miles-ports:0af62f4-9f29303@sha256:c6f5455c8a88d131c780cf99d07c3fdd913d2beda25b6d94739db06916d602ed
  (miles 0af62f4 = d002615 + LoRA bridge calculate_per_token_loss; sglang 9f29303; pure-Python gate passed).
- Modal `img-smoke`, 1 Sandbox, 1x L40S, cpu 8/64 GiB, Sandbox timeout=2400, local timeout 3000,
  terminate in finally + `modal app stop -y`, independent 55-min watchdog.  Estimate < $1; budget $5.
- Script ../smoke_in_image.sh at this commit: + check lora_bridge_per_token_loss (grep the assignment in
  /root/miles/miles/backends/megatron_utils/lora/bridge.py).
- Pass = every image check PASS.  Known beforehand (run of d002615): parse_args
  test_upstream_parse_args_accepts_translation[False] fails in yeto's translate_run_config guard
  (config.py:527) before Miles is called -- image-independent; expected to recur; [True] must pass
  and nothing else may fail.
