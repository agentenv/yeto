# 4.4 parse check on the pinned Miles 5c1b49eb (ALGO-2b, 2026-09-30)

- Miles source: `/tmp/img-verify-5c1b/root/miles` = `/root/miles` extracted from the
  pinned image `ghcr.io/michaellchung/yeto-miles-ports:5c1b49e-9f29303@sha256:17d428a2…`
  by Agent IMG (`git rev-parse HEAD` = 5c1b49ebccbc7508c1d9ef89eacc2db3e448b6ba, printed in parse.log).
- Step 1 `gen_argv.py` (yeto venv, this branch): AlgorithmSpec -> `algorithm_argv` -> `argv.json`.
- Step 2 `parse_argv.py` (`/home/michael/work/miles-next-venv`, `PYTHONPATH=<image /root/miles>`):
  Miles' own `get_miles_extra_args_provider()` parser + `validate_policy_loss_variant_args`.
  Result `parse.log`: cispo / sapo / gmpo argv parse with no unknown flags and are accepted with
  the values yeto set; gmpo + `--calculate-per-token-loss` is rejected by Miles, as yeto rejects it
  before launch.
- `upstream_flags_test.log`: `tests/test_rl_algorithm_flags_upstream.py` in the same environment
  (20 passed; the fork flags are no longer exempt because 5c1b49eb is in `FORK_COMMITS`).
- NOT covered: the full upstream `parse_args` + Megatron `validate_args` (the Megatron half needs a
  CUDA container; miles-next-venv has no `megatron.training`; no local docker, no GPU this round).
