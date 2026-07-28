# v17: reproduce and overturn

**Status:** `REGISTERED_PRE_OUTCOME`  
**Authority:** design and gatesim only; `NO_LAUNCH_AUTHORITY`

v17 selects the strongest survey result that combines a primary-source numeric
gain, exact runnable hyperparameters, public data, and open code: Kallusky et
al.'s SNOO example with `K=100`, outer `eta=.8`, and `mu=.75`. The paper reports
that its 300M run reaches AdamW's final validation loss in 78% of the steps, a
1.28× step speedup. The local primary-source extraction has SHA-256
`b8a1fce9c0ddcb8debdc9c9bc3e714ac60d0c93895922cc5d44b6d6a35ef4aec`.

No data materialization, result root, launch manifest, queue, controller, or
GPU process is authorized by this registration.

## Quoted original protocol

The primary source states:

> “All models were trained using Fully Sharded Data Parallel (FSDP) with a
> global batch size of B = 64 sequences and a sequence length of L = 8,192.
> The training budget was set to achieve a token-to-parameter ratio of
> approximately 30:1. Gradient clipping with a maximum norm of 1.0 was
> employed to ensure stability.”

It fixes AdamW LR `3e-4`, weight decay `.01`, betas `.9/.95`, epsilon `1e-8`,
warmup followed by linear decay to a `.1` minimum factor. It also says the
algorithm “does not reset the inner optimizer's states or learning rate
schedule between outer steps.” Those details, not just the three outer
hyperparameters, are registered here.

## Disclosed 1.7B bridge

The source uses dense OSS Llama-3 models through 1B and illustrates the selected
run at 300M. v17 uses pinned SmolLM2-1.7B revision
`effd688a12921b4cc83e3312b6feb579f70f9c71`. This is a protocol reproduction
under a model-family/scale bridge, not byte-identical replication.

At 64×8,192 tokens per step, 98,000 steps consume 51,380,224,000 tokens, or
30.02275 tokens/parameter. The K-divisible count gives exactly 980 complete
outer updates. Warmup is 12,760 steps, obtained by scaling the paper's 1B
7,500/57,600 fraction. The dataset is English C4 at pinned revision
`1588ec454efa1a09f29cd18ddd04fe05fc8653a2`; deterministic packing and a fixed
4,096-sequence validation set must be hash-materialized before any launch.

## Phase A: reproduce

Seven paired seeds compare fixed AdamW with the fixed SNOO recipe for the full
budget. Validation runs every 500 steps. A nonincreasing isotonic fit determines
the first step at which SNOO reaches the paired AdamW final threshold. Failure
to cross is right-censored to gain 1.0 and counts against reproduction.

Phase A reproduces only when the mean log2 step speedup is at least
`log2(1.15)=0.201634` and its seven-seed 95% Student interval has a lower
endpoint above zero.

## Phase B: age-matched per-arm retune

SNOO (`mu=.75`) and Lookahead (`mu=0`) both use K=100, exactly 98,000 inner
steps, exactly 980 outer updates, identical paired token order, and an
uninterrupted AdamW state/schedule clock. Each independently receives the same
3×3 grid: inner LR `{1.5e-4,3e-4,6e-4}` crossed with outer eta
`{.5,.8,.95}`. AdamW independently receives the same three inner LRs.

A fully specified two-round successive-halving procedure uses separate tuning
seeds at approximately 5 and 10 tokens/parameter. The selected configurations
and all selection arithmetic must be committed and pushed before the seven
fresh full-budget confirmation seeds begin.

Against the selected AdamW final threshold, the survival estimand is paired
log2 step speedup of selected SNOO minus selected Lookahead. Survival requires
the lower 95% endpoint to exceed `log2(1.05)=0.070389`. Failure of this positive
criterion is not called statistical equivalence unless the separately reported
interval lies wholly inside the ±1.05× equivalence band.

## Closed vocabulary and gatesim

The final vocabulary is exactly:

- `GAIN_REPRODUCED_AND_SURVIVES`;
- `GAIN_REPRODUCED_VANISHES`;
- `GAIN_NOT_REPRODUCED`.

Phase A failure takes precedence. If Phase A reproduces, Phase B's positive
survival criterion selects between the first two labels.

With seven confirmation seeds and a conservative 0.10-bit seed SD, the
20,000-draw gatesim assigns 0.99990 probability to
`GAIN_REPRODUCED_VANISHES` under reproduction plus a tuned null. Under
reproduction plus a genuine 1.15× tuned momentum advantage, it assigns 0.82370
probability to `GAIN_REPRODUCED_AND_SURVIVES`. `P_evaluable=1.000` in both
complete-data scenarios.

These are decision-power calculations, not a reproduction result.

Machine contract:
`experiment-specs/outer-mup-v17-reproduce-overturn-prereg.json`.
Gatesim report:
`experiment-specs/outer-mup-v17-reproduce-overturn-gatesim.json`.
