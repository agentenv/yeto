# Run health breakers

Two pieces watch an RL run's numerics:

- `yeto breaker`: a read-only check over a run's event tapes. It prints every
  breaker with its window value and limit and exits 1 when one trips, so a cron
  job or a supervisor can page or call `yeto stop-run RUN`. It never stops
  anything itself.
- `YETO_RL_XHW_GUARD`: the in-run guard of the trainer driver. It judges each
  sampler source every trained round and can warn, quarantine, down-weight or
  stop.

## `yeto breaker`

```bash
yeto breaker --trainer-log ~/.yeto/runs/<run>/launcher.log \
  --sampler-tapes <dir-of-pulled-sampler-tapes> [--window 3] \
  [--sampler-hardware auto|amd|nvidia] [--noise-floor X] \
  [--logprob-warn X] [--logprob-act Y] [--flips flips.json] [--json]
```

- `--trainer-log`: launcher logs (lines with `YETO_RL_EVENT {json}`), JSONL
  event tapes, or directories of them.
- `--sampler-tapes`: sampler event tapes (`events.<incarnation>.jsonl`), as
  files, globs or directories (every `*.jsonl` below them).
- Exit codes: `0` all ok (warnings included), `1` a breaker tripped, `2` bad
  input (a missing file, inconsistent limits).

| Breaker | Value (trailing `--window` trained steps) | Trips |
|---|---|---|
| `logprob_diff` | mean `train_rollout_logprob_abs_diff` | above the hardware limit (below) |
| `is_clipped` | mean `sc_is_clipped` | > 0.05 |
| `is_weight` | mean `sc_importance_weight` | < 0.9 |
| `stale` | stale groups / (stale + trained) | > 0.30 |
| `aborted` | agentic trials lost / trials (last 3 x window sampler rounds) | > 0.20 |
| `zero_call` | agentic trials retried before any model call / trials | > 0.05 |
| `flips` | changed / total weights of the last release (`--flips`, `{release: {changed, total}}`) | > 0.01 |
| `nan_skip_consecutive` | rounds skipped in a row for a non-finite grad norm | > 2 |
| `nan_skip_fraction` | skipped / trained rounds (judged from 20 trained rounds on) | > 0.05 |
| `r3_rows` | sampler records reporting a routed-experts row mismatch | > 0 |
| `xhw <sampler>/<card>` | the last `rl_xhw_mismatch` row of each source | level `act`, or the guard stopped |

### Log-prob mismatch limits by sampler hardware

The train/rollout log-prob difference is not zero even with identical weights:
the sampler's card and engine round differently from the trainer's recompute.
The limits therefore depend on where the rollouts were sampled.

| Hardware class | warn | trip | Measured (Kimi-K3.1, step 1, pure numerics) |
|---|---|---|---|
| `amd` (ROCm, MI-series cards) | 0.10 | 0.15 | MI355X vLLM -> B300 Megatron 0.076-0.099 |
| `nvidia` (CUDA cards) | 0.04 | 0.052 | B300 SGLang -> B300 Megatron 0.023-0.028 |

The `amd` limits are the in-run guard's defaults. With `--sampler-hardware auto`
(the default) the class comes from the sampler tapes (`rl_sampler_engine_round`
`source`, `family/card/engine@version/...`) or the `rl_xhw_mismatch` rows. A run
with any AMD sampler is judged by the `amd` limits, since the step value mixes
the tokens of every source. When no tape tells, the `amd` limits apply and the
line says `sampler hardware unknown`.

`--noise-floor X` (the mean abs diff of two repeat runs with the same weights)
raises the trip limit to at least `1.5 * X` and the warn limit to at least `X`.
`--logprob-warn` and `--logprob-act` set the limits directly.

## `YETO_RL_XHW_GUARD` (in-run, per source)

Opt-in. The launcher forwards it to the trainer island. With several sampler
sources (cards, engines), the driver attributes the step metrics to each source
by its token share and judges each one on a trailing window. It emits
`rl_xhw_mismatch` every trained round.

```bash
YETO_RL_XHW_GUARD=1        # defaults
YETO_RL_XHW_GUARD=warn_abs_diff=0.10,act_abs_diff=0.15,warn_kl=0.03,act_kl=0.06,warn_clip=0.01,act_clip=0.03,window=3,min_rounds=2,action=quarantine,quarantine_rounds=8,weight=0.5
```

| Key | Default | Meaning |
|---|---|---|
| `warn_abs_diff` / `act_abs_diff` | 0.10 / 0.15 | `train_rollout_logprob_abs_diff` limits |
| `warn_kl` / `act_kl` | 0.03 / 0.06 | `train_rollout_kl` limits |
| `warn_clip` / `act_clip` | 0.01 / 0.03 | `sc_is_clipped` limits |
| `window` | 3 | trailing trained steps per source |
| `min_rounds` | 2 | rounds a source needs before it can be acted on |
| `action` | `quarantine` | `warn`, `quarantine` (skip the source's groups for `quarantine_rounds` rounds, then let one probe round through), `downweight` (scale its advantages by `weight`; exact only without GRPO std normalization), `stop` (the driver raises) |
| `quarantine_rounds` | 8 | rounds of a quarantine |
| `weight` | 0.5 | advantage scale of `downweight` |

Every key is optional; `0`, `off` or unset turns the guard off. The defaults
sit above the measured MI355X vLLM band. For an all-NVIDIA fleet, tighten them
to `warn_abs_diff=0.04,act_abs_diff=0.052`.
