# RL Algorithms

This document describes how a Miles RL island's training objective is
specified, validated and translated into Miles flags. The runtime, launch,
synchronization and recovery contract is in [MILES_RL.md](MILES_RL.md).

## Algorithm specs (`--rl-algorithm-spec`)

The training objective is described by one `AlgorithmSpec`
(`yeto/rl/engine/algorithm.py`), and the Miles algorithm flags are generated
only from it. The spec has these groups: `advantage` (estimator,
`std_normalization`, `rewards_normalization`, `whiten`, `reward_postprocess`,
`reward_binary`), `loss` (`variant`, `eps_clip`, `eps_clip_high`,
`eps_clip_c`, `aggregation`, `reducer`, `custom_loss`), `kl` (`placement`,
`coef`, `estimator`, `unbiased`), `correction`, `sampling`, `execution`,
`entropy_coef` and `plugins`. A plugin is written as `{path, sha256}` (the
SHA256 of the plugin module's source file). Only the `yeto.` and `miles.`
namespaces are accepted. The learner re-hashes and imports every plugin before
it joins outer sync, and refuses to start if the hash differs.

Where the spec comes from:

1. `--rl-algorithm-spec PATH`, a v1 or v2 JSON file. `yeto launch` and
   `python3 -m yeto.rl.adapters.miles.island_entry` both accept it.
2. Without that flag, the CLI builds the default GRPO spec from the run
   flags.
3. Mapped Miles flags found in the extra argv are absorbed into the spec.

A spec that only uses the original fields keeps the v1 canonical JSON, so its
hash is unchanged. Setting any newer field switches the spec to
`yeto-rl-algorithm-spec-v2`.

### Absorb and reject

`yeto/rl/adapters/miles/algorithm_flags.py` maps each Miles flag to a spec
field. Every flag in that table is adapter-owned:

- A mapped flag in the extra argv is absorbed. It is recorded as
  `rl/algorithm_absorbed_flags` in the `rl_engine_selected` event.
- A value that disagrees with the spec is refused, and the error shows both
  values.
- A flag that changes the objective but has no mapping yet is refused. These
  are the flags in `UNMAPPED_OBJECTIVE_FLAGS`, for example
  `--rollout-temperature`.

### KL placement

- `kl.placement=loss` translates to
  `--use-kl-loss --kl-loss-coef C --kl-loss-type T`, and `T` must be given.
- `placement=reward` translates to `--kl-coef`. With `grpo` or `gspo` and a
  coefficient above 0 it is refused, because Miles drops a KL placed in the
  reward for these estimators; use `placement=loss` instead.
- A v1 `kl_coef` is read as `placement=reward`. `0.0` still emits
  `--kl-coef 0.0`, so the hash is unchanged. `placement=none` emits no KL flag,
  and Miles loads no reference model.

### Capabilities and execution

The engine declares what it supports per mechanism dimension:
`advantage_estimators`, `losses`, `loss_aggregations`, `kl_placements`,
`corrections`, `reward_postprocessors`, `dynamic_sampling_filters` and
`features`. It also declares an `execution` block: `critic=false`,
`max_policy_staleness=0` and `rollout_logprobs=true`.

Before any GPU process exists, the driver handshake refuses:

- every mechanism the spec requires that the engine does not declare;
- a critic;
- a policy age above `execution.max_policy_staleness`, which is fixed at 0;
- every entry of the rejection matrix:
  - TIS together with `use_rollout_logprobs`;
  - reward KL together with loss KL;
  - `gspo` without an explicit clip range;
  - a mechanism that needs a binary reward when the reward is not declared
    binary;
  - `reinforce_plus_plus*` without `whiten`.

### Declarations

Declarations are defined by `MILES_DECLARED` in
`yeto/rl/adapters/miles/entry.py`. A mechanism is declared only once it is
known to take effect on GPU; a declaration says nothing about training gains.
Some declarations apply only under specific Miles pins
(`MILES_DECLARED_PINS`).

Under a correction that makes Miles set `use_tis` (tis, icepop, mis_mask,
mismatch_observe), `correction.mismatch_metrics` is claimed by that correction
(`CORRECTION_COMPANIONS`), because the flag has no effect there. Under a
generic custom function it is still a separate, undeclared mechanism.

Settings an estimator mandates are claimed by that estimator's mechanism in
that combination only (`ESTIMATOR_COMPANIONS`): GSPO's explicit clip range and
the REINFORCE++ family's advantage whitening. The same settings under grpo are
still separate, undeclared mechanisms.

**User code is never declared.** A declaration covers one piece of code, not
any code a user may supply. These mechanisms never enter `MILES_DECLARED`;
`entry.NEVER_DECLARABLE` lists them, and a test asserts that the list and
`MILES_DECLARED` do not overlap:

- `corrections:custom` (a generic custom correction function);
- `features:plugins` (plugins that no registered Yeto extension owns);
- `losses:custom_loss` (`--custom-loss-function-path`);
- `features:custom_pg_loss_reducer` (any reducer other than the vendored
  Dr.GRPO reducer; that reducer is claimed by `loss_aggregations:constant`
  through `register_named_reducer` at its pinned source hash).

The Yeto reward dispatcher (`reward_postprocessors:custom_reward_postprocess`)
is Yeto code and can be declared.

Each declared mechanism stands on its own; combinations of declared
mechanisms (for example tis+opsm_trainer) are accepted but not separately
verified.

"Expressible, not enabled" means the spec can describe and translate a
mechanism, but `miles_capabilities` does not declare it.

### Unverified-mechanism allowance

`--rl-allow-unverified-mechanism DIMENSION:NAME` (repeatable, for example
`features:clip_higher`) exempts only the named mechanisms from the "not
declared" check, and only on a single-island run **without outer sync**
(`--rl-single-island-no-sync`, or the dry run, which models a sync-less single
island). With two or more islands or with outer sync the launch is refused.
Every other check still applies. The allowance does not change the hash. It is
recorded as `rl/unverified_mechanisms` in the event and in
`yeto_rl_provenance.json`, which is also marked
`contains_unverified_mechanisms`.

### Island consistency and provenance

`yeto launch` builds the spec once and sends each island two things:

- its canonical JSON, as `~/yeto-rl/algorithm_spec.json`;
- `--rl-expected-algorithm-sha256`.

Each learner compares its own hash with the expected one before it joins outer
sync:

- on a mismatch it writes an `rl_algorithm_mismatch` event and exits;
- when the expected hash is missing (a learner started by hand) it writes a
  warning event.

With no algorithm option given, the island learner command still carries
`--rl-expected-algorithm-sha256 <hash>`, the tape ends with an
`rl_learner_finalized` record, and the `rl_engine_selected` event carries
`rl/algorithm_spec` (canonical JSON) and `rl/algorithm_absorbed_flags` (`{}`
by default). The Miles argv of default GRPO is unchanged by the spec layer.

`yeto-rl-export --rl-algorithm-spec PATH` writes `algorithm_spec`, the
canonical JSON, and `algorithm_spec_sha256` to the provenance.

### Dry run

`python3 -m yeto.rl.adapters.miles.algorithm_flags --dry-run
[--rl-algorithm-spec PATH] [--extra "<miles argv>"]
[--rl-allow-unverified-mechanism NAME]` runs the same steps as the learner,
with no engine: resolve, absorb, rejection matrix, then the Miles adapter's
declaration. It prints the canonical spec, the hash, the absorbed flags, the
Miles algorithm flags and the verdict. The exit code is 0 only when the spec
is accepted.

```bash
M="python3 -m yeto.rl.adapters.miles.algorithm_flags"
$M --dry-run          # default GRPO: v1 schema, accepted
echo '{"schema":"yeto-rl-algorithm-spec-v2","loss":{"eps_clip_high":0.28}}' > clip_higher.json
$M --dry-run --rl-algorithm-spec clip_higher.json
$M --dry-run --extra "--eps-clip-high 0.28"      # absorbed; same hash as the file
$M --dry-run --rl-algorithm-spec clip_higher.json --extra "--eps-clip-high 0.3"
    # rejected: ... sets loss.eps_clip_high=0.3 but the algorithm spec has ...=0.28
$M --dry-run --extra "--kl-coef 0.1"   # rejected: ... use kl.placement='loss'
```

### Extending

Add a module under `yeto/rl/algos/` and one line to
`yeto.rl.algos.EXTENSION_MODULES`. The module registers what it needs:

- in `algorithm.py`: `register_field`, `register_mechanism`,
  `register_rejection`, `register_launch_check`, `register_island_check`,
  `register_runtime_attrs`, `register_gradient_rule`;
- in `algorithm_flags.py`: `register_flag`.

A registered field enters the v2 canonical JSON only when it differs from its
default, so registering one never changes an existing hash. The mechanism is
declared in `yeto/rl/adapters/miles/entry.py` only after a single-GPU smoke
shows it takes effect.

### When Miles is upgraded

Update `MILES_NEXT_COMMIT` in `yeto/rl/adapters/miles/pins.py`, then re-review
the objective-changing flags of the new `miles/utils/arguments.py` (the
algorithm, rollout and reward groups):

1. Add each new flag to the mapping table or to `UNMAPPED_OBJECTIVE_FLAGS`.
2. Run `tests/test_rl_algorithm_flags_upstream.py` in the upstream venv:

   ```bash
   PYTHONPATH=<miles checkout>:$PWD:$PWD/tests \
     <miles venv>/bin/python -m pytest -q tests/test_rl_algorithm_flags_upstream.py
   ```

   It checks that every listed flag exists upstream and that upstream
   `parse_args` accepts every non-default mapping.
3. Re-check the estimator and KL rules in `algorithm.py` against Miles'
   advantage and train-spec code.
4. Re-copy the vendored MIS helper and re-check `ICEPOP_SOURCE_SHA256` (see
   [Train-inference mismatch corrections](#train-inference-mismatch-corrections)).

## GRPO-family knobs

Registered by `yeto/rl/algos/grpo_knobs.py`. Every mechanism below can be
expressed and translated; whether it is declared is defined by
`MILES_DECLARED`.

| Mechanism | Spec | Miles argv | Constraint (checked before any GPU process) |
|---|---|---|---|
| clip-higher | `loss.eps_clip_high` (and `eps_clip`) | `--eps-clip-high` | finite, > 0 |
| dual-clip | `loss.eps_clip_c` | `--eps-clip-c` | > 1 (Miles asserts the same, later) |
| token aggregation | `loss.aggregation="token"` | `--calculate-per-token-loss` | |
| Dr.GRPO, no std | `advantage.std_normalization=false` | `--disable-grpo-std-normalization` | |
| Dr.GRPO, constant denominator | `loss.aggregation="constant"`, `loss.constant_denominator=D`, `loss.reducer=yeto.rl.algos.reducers.constant_denominator_reducer` | `--custom-pg-loss-reducer-function-path` | D finite > 0; not with token aggregation (one enum; `--calculate-per-token-loss` in extra argv conflicts); CP = 1 |
| KL loss | `kl.placement="loss"`, `coef`, `estimator` ∈ k1/k2/k3/low_var_kl, `kl.ref_model={source, revision}` | `--use-kl-loss --kl-loss-coef --kl-loss-type` | `ref_model` required; its revision must equal the island's `--model-revision` |
| entropy | `entropy_coef` | `--entropy-coef` | finite |
| over-sampling | `sampling.over_sampling_batch_size` | `--over-sampling-batch-size` (from the run config) | needs `sampling.filter`; ≥ rollout batch size |
| overlong penalty | `advantage.reward_shapers=[{name: overlong_penalty, max_length, cache_length}]` + the dispatcher | `--custom-reward-post-process-path yeto.rl.algos.reward_pipeline.post_process` | 0 < cache_length ≤ max_length ≤ `rollout_max_response_len` |
| overlong filter | `sampling.overlong_filter=true` | none (the shared `--rollout-sample-filter-path` hook) | |

**Dr.GRPO and RLOO.** The constant-denominator reducer is vendored from the
upstream Miles DrGRPO example reducer. The only change is that D comes from
the spec instead of the constant 1000. It applies to pg_loss only; clipfrac,
KL and entropy keep the default reducer. RLOO is not implemented separately:
with std normalization off its advantage is G/(G−1) times the Dr.GRPO
advantage, and Adam is nearly invariant to a constant gradient scale.

**KL loss.** Miles loads the `--ref-load` model only when `kl_coef != 0` or
`use_kl_loss`, so KL loss adds a reference model and one extra reference
forward pass. `kl.ref_model` is part of the algorithm hash, so two islands
with different references disagree before outer sync. The learner also
refuses a reference revision that differs from `--model-revision`
(`rl_algorithm_island_rejected`).

**Reward dispatcher.** `yeto.rl.algos.reward_pipeline.post_process` is the
only reward post-processing hook. It is emitted only when the spec selects a
reward shaper or a non-default `advantage.transform`; default GRPO keeps
Miles' built-in path, and its argv is unchanged. The dispatcher runs
`raw → shapers → advantage transform`. `grpo_default` is element-wise equal
(`torch.equal`) to Miles `_post_process_rewards`: prompt groups, one reward
per multi-segment rollout, the error on inconsistent siblings, G=1, std=0,
`+1e-6`, and the estimator and `rewards_normalization` gates. The comparison
is `tests/test_rl_reward_pipeline_equivalence.py`; it pins the SHA256 of
Miles' `train_data_conversion.py`. Details:

- Multi-LoRA is refused, because a custom post-process cannot see
  `prompt_group_sizes`.
- A batch with no `group_index` and no fixed fan-out falls back to one
  whole-batch group, as Miles does, and emits `rl_reward_group_fallback`.
- The configuration reaches Miles as `args.yeto_algo_plugins = {config,
  sha256}`, set through the spec's runtime attrs. The dispatcher and reducer
  re-hash it and refuse a mismatch.

**Adding an advantage transform.**

1. Register it in `reward_pipeline.py` with `register_advantage_transform(name,
   fn(args, samples, rewards, groups, params), validate=...)`. This keeps it
   under the dispatcher's PluginRef hash.
2. Select it with `advantage.transform=name` plus `transform_params`.
3. Register its mechanism and leave it undeclared until a single-GPU smoke
   shows it takes effect.
4. Add a CPU test that compares it with a hand computation, and with
   `grpo_default` in its degenerate case.

Any edit to `reward_pipeline.py` changes the dispatcher's source hash:
regenerate specs that pin it (for example `examples/rl_algorithms/*.json`).

**Overlong penalty (DAPO).** For a response of length L:

- L ≤ Lmax − Lcache: penalty 0;
- Lmax − Lcache < L ≤ Lmax: penalty (Lmax − Lcache − L)/Lcache;
- L > Lmax: penalty −1.

The penalty is added to the raw reward before group normalization. L is the
summed length of a rollout's segments, so siblings stay consistent. The
shaped reward is what Miles logs as raw reward. The original reward is kept
in `sample.metadata["yeto_raw_reward"]`, and an `rl_reward_shaping` event
summarizes raw against shaped rewards.

**Overlong filter vs DAPO.** Truncated samples get `remove_sample=True` in
`record_trained_groups`:

- Their loss mask becomes all zero, but their reward still enters the group
  mean and std, so the other samples' advantages are unchanged.
- Under the default sample-mean aggregation a removed sample contributes 0 to
  the numerator. It still counts in the `global_batch_size` divisor, which
  dilutes the others.
- Under token aggregation it adds 1 to the reported token count
  (`clamp_min`).

DAPO's paper masks truncated samples in the loss but does not say whether
their reward enters the group statistics. The rollout metadata carries
`filtered_samples`, which the ledger records in the terminal state `filtered`.
Over-sampling leaves no reusable remainder: once a rollout has its batch,
Miles does not return further kept groups to the buffer; only samples aborted
under `--partial-rollout` go back. A round in which every non-zero-variance
group was filtered completely does not trip the zero-gradient invariant.

**Over-sampling and outer averaging.** strict-avg and decoupled average the
islands' deltas with equal weights, not weighted by sample count. Islands that
train on different numbers of samples therefore count equally.

```bash
M="python3 -m yeto.rl.adapters.miles.algorithm_flags"
$M --dry-run --rl-algorithm-spec examples/rl_algorithms/dr-grpo.json
```

## Sequence-level ratio and advantage variants

Registered by `yeto/rl/algos/seq_adv.py`. Declared support is not a claim of
benefit.

| Mechanism (`dimension:name`) | Spec | Engine argv |
|---|---|---|
| GSPO (`advantage_estimators:gspo`) | `advantage.estimator="gspo"` + explicit `loss.eps_clip` / `loss.eps_clip_high` | `--advantage-estimator gspo --eps-clip .. --eps-clip-high ..` |
| REINFORCE++ (`advantage_estimators:reinforce_plus_plus`) | `advantage.estimator`, `advantage.whiten=true`, optional `kl.placement="reward"` | `--normalize-advantages [--kl-coef ..]` |
| REINFORCE++-baseline (`advantage_estimators:reinforce_plus_plus_baseline`) | same | same |
| MaxRL (`features:maxrl`), MAPO (`features:mapo`) | `advantage.transform`, `advantage.reward_binary=true`, dispatcher + `plugins=[seq_adv ref]` | `--custom-reward-post-process-path yeto.rl.algos.reward_pipeline.post_process` |
| GDPO (`features:gdpo`) | `advantage.transform="gdpo"`, `advantage.gdpo={components:[{name,weight}], whiten:true}`, dispatcher + `plugins` | same |

Example specs (PluginRef SHA256s regenerated by
`tests/fixtures/seq_adv_examples/make_examples.py`) live in
`tests/fixtures/seq_adv_examples/`. Check one without an engine:

```bash
python3 -m yeto.rl.adapters.miles.algorithm_flags --dry-run \
  --rl-algorithm-spec tests/fixtures/seq_adv_examples/maxrl.json
```

**GSPO.** The clip range must be explicit (`gspo_noclip.json` is rejected: the
engine default `--eps-clip 0.2` is a token-level value; the GSPO paper uses
3e-4 / 4e-4). Yeto sets no default. Watch the per-round clip fraction
(`masked_fraction`, from Miles `pg_clipfrac`): a round whose every sequence is
clipped has a legitimately zero gradient and does not trip the zero-gradient
invariant; an unknown clip fraction keeps the strict rule. With a single
optimizer step per round Miles recomputes the old policy with the current
weights, so the ratio is ~1 and clipping never binds: GSPO then is a
sequence-ratio GRPO (clip fraction ~0). GSPO combined with an advantage
transform is not opened.

**REINFORCE++ family.** `advantage.whiten=true` is required (Miles asserts
it). Whitening is `--normalize-advantages`: token-level, all-reduced inside
the island's DP group only; statistics never cross islands or enter the outer
protocol. Each island's advantages are therefore normalized by its own
statistics. `kl.placement="reward"` is allowed and enters the advantage
(loads the reference model). `advantage.gamma` (`--gamma`) is mapped but only
1.0 is open (`rpp_gamma.json` is rejected: supported values [1.0]); with any
non-critic estimator a non-default gamma is rejected.

**MaxRL / MAPO** require a binary {0,1} reward, declared by
`advantage.reward_binary=true` and checked at runtime after reward shaping (a
non-binary reward fails the round); they are refused with the
`overlong_penalty` reward shaper (use `sampling.overlong_filter`). They compute
per rollout (multi-segment rollouts merged, result shared by every segment),
only with `advantage.estimator="grpo"`. MaxRL: `(r - mean)/mean`, 0 for
all-wrong groups and single-sample groups. MAPO:
`(1-lam)(r-mu)/sigma + lam(r-mu)/mu`, `lam = 1-4p(1-p)`, sigma as Miles' GRPO
(unbiased std + 1e-6); at p=0.5 it equals Miles' GRPO normalization exactly.

**GDPO reward vector.** The reward function writes
`sample.metadata["yeto_reward_components"] = {name: float}` for exactly the
declared components (example: `yeto.rl.algos.gdpo_reward.reward_func`,
correctness + format); a missing, extra or non-finite component fails the
round (no default fill). Each component is normalized inside its group, the
weighted sum is whitened over this island's whole rollout batch (sample
level, island-local). The scalar `sample.reward` is only for metrics.

All three transforms run in the Yeto dispatcher; their module
(`yeto.rl.algos.seq_adv`) must be listed in `plugins` so its source SHA256
enters the algorithm hash. Each round emits `rl_advantage_transform`
(all-wrong / all-right groups, non-zero advantages).

## Policy-loss variants

Registered by `yeto/rl/algos/loss_variants.py`. The computation is a variant
branch in the Miles fork (`--policy-loss-variant`); Yeto only describes,
translates and validates. A launch is refused on any Miles pin outside
`yeto.rl.algos.loss_variants.FORK_COMMITS`. The variants are not declared, so
only the single-island `--rl-allow-unverified-mechanism losses:<v>` entry can
run them.

The variants replace only the token-level pg_loss inside `--loss-type
policy_loss`; TIS / IcePop / OPSM weights, KL loss, entropy and the
aggregation are applied afterwards by Miles' unchanged code (variant loss
first, then the correction weight, then aggregation). With rho = pi_theta /
pi_old and A the advantage:

| Variant (`loss.policy_loss_variant`) | Token loss | Parameters (default) | Miles argv |
|---|---|---|---|
| `policy_loss` (default) | PPO clip (unchanged) | -- | nothing emitted (default GRPO argv and hash unchanged) |
| `cispo` | -sg(clip(rho, 1-eps_l, 1+eps_h)) * A * log pi_theta; every token keeps a gradient; token-normalized | `loss.eps_clip` (eps_l) / `loss.eps_clip_high` (eps_h), **required** (no default); `loss.aggregation="token"` **required** | `--eps-clip .. --eps-clip-high .. --calculate-per-token-loss --policy-loss-variant cispo` |
| `sapo` | -sigmoid(tau (rho-1)) * 4/tau * A; tau = tau_pos for A>0, tau_neg otherwise; no hard mask; per-sequence mean (Miles default aggregation) | `loss.sapo_tau_pos` (1.0), `loss.sapo_tau_neg` (1.05) | `--policy-loss-variant sapo --sapo-tau-pos .. --sapo-tau-neg ..` |
| `gmpo` | one-sided (arXiv:2507.20673v3 eq. 4, in log space): l_t = sign(A) min(sign(A) log rho_t, sign(A) clamp(log rho_t, -delta_l, delta_h)); sequence ratio exp(mean_t l_t) over the whole sequence; -ratio * A. For A>0 only log rho > delta_h is clipped, for A<0 only log rho < -delta_l | `loss.gmpo_log_clip_low` (0.4), `loss.gmpo_log_clip_high` (0.4) | `--policy-loss-variant gmpo --gmpo-log-clip-low .. --gmpo-log-clip-high ..` |

`loss.variant` stays `--loss-type`; the variant is its own field. A variant's
parameters enter the canonical form (and the hash) exactly when that variant
is selected, then always (even at the default value), and are always emitted
explicitly (never the fork's own default). Absorbed from extra argv, the fork
flags follow the usual rule: a value that differs from the spec fails before
launch naming both values.

Refused before any GPU process:

- a variant with `advantage.estimator="gspo"` (both define the ratio);
- a variant with `loss.eps_clip_c` (dual-clip is for the PPO clip objective);
- a variant with `loss.variant="custom_loss"`;
- CISPO without both `loss.eps_clip` and `loss.eps_clip_high` (the clip range
  must be in the hash);
- CISPO without `loss.aggregation="token"` (see the paper notes below);
- GMPO with `loss.aggregation="token"` (`--calculate-per-token-loss`; the
  fork refuses it too);
- SAPO / GMPO with `loss.eps_clip` / `loss.eps_clip_high` (no effect there);
- a parameter of another variant (e.g. `loss.sapo_tau_pos` with `gmpo`);
- tau or delta that is not a positive finite number (the error names the
  field);
- GMPO with context parallel size > 1;
- any variant on a Miles pin without the fork commit (launch check
  `[loss_variants]`).

Zero-gradient invariant: CISPO and SAPO keep the GRPO rule (a round whose
every ratio is out of range still expects a gradient). GMPO may legitimately
produce no gradient when every token is clipped in log space: the round is
relaxed only when the round's global GMPO clip fraction is 1:
sum(`gmpo_clip_num`) / sum(`gmpo_clip_den`) over the round's optimizer steps,
the two counts the fork reports under the same final loss mask (tokens with
A != 0; sequences with all-zero advantages count in neither). Only the ratio
is meaningful (Miles' micro-batch averaging scales both; under CP every rank
adds the same counts). Fork `pg_clipfrac` keeps its own per-sequence-mean
definition and is not used by the rule. The fraction travels as
`TrainStepMetrics.clip_fraction`, never as `masked_fraction` (which
corrections fill with their own mask). Missing counts or a zero denominator
keep the GRPO rule. A non-finite grad norm fails under every variant.

Paper notes. CISPO (MiniMax-M1, arXiv:2506.13585 eq. 4-5) divides by the
total token count of the group, i.e. per-token loss; Miles' default is the
per-sample mean, so Yeto opens CISPO only with `loss.aggregation="token"`
(Miles normalizes by the batch token count -- the same up to how groups share
a batch). The paper sets no effective lower bound (eps_low large, only
eps_high tuned): a paper-like run uses `loss.eps_clip >= 1`. SAPO
(arXiv:2511.20347v2 eq. 5-6) is a per-sequence mean, which is Miles' default
aggregation. GMPO (arXiv:2507.20673v3 eq. 4) matches the official code.

Outer sync: the variants only change the loss and are orthogonal to
strict-avg and decoupled; islands run serially (staleness 0, pi_old is this
round's start).

```bash
python3 -m yeto.rl.adapters.miles.algorithm_flags --dry-run \
  --extra "--policy-loss-variant sapo" --rl-allow-unverified-mechanism losses:sapo
# miles_argv ends with --policy-loss-variant sapo --sapo-tau-pos 1.0 --sapo-tau-neg 1.05
python3 -m yeto.rl.adapters.miles.algorithm_flags --dry-run \
  --extra "--policy-loss-variant gmpo --sapo-tau-pos 1.2" \
  --rl-allow-unverified-mechanism losses:gmpo
# rejected: [loss_variant_params] ['loss.sapo_tau_pos'] only apply to ...
```

Upgrading Miles: the variant lives in one branch point of the fork's
`policy_loss_function` plus its math utilities and arguments. After every
rebase of the fork, rerun the fork's variant tests, add the reviewed commit to
`FORK_COMMITS` when `MILES_NEXT_COMMIT` and the image digest move, and declare
a variant in `entry.MILES_DECLARED` only after its GPU smoke passed.

## Critic family: PPO, VAPO, SAO, CompactionRL

The Miles adapter declares `execution.critic=false`, so a critic algorithm
launches only as a single-island run without outer sync
(`--rl-single-island-no-sync`), with `--rl-allow-unverified-mechanism` naming
`advantage_estimators:ppo`, `execution:critic` and each fork feature it uses;
every two-island critic run is refused (`check_unverified_allowance`).

All four algorithms are the Miles shared actor/critic PPO
(`--advantage-estimator ppo`; Miles derives `use_critic`) and differ only in
the GAE variant, the value loss and a few loss terms. Yeto declares,
validates, translates and orchestrates; the math lives in Miles or in the
fork.

**Fork pin and overlay.** The GAE variants, the positive-example LM loss, the
SAO math, the tied-embedding critic value head and `--critic-updates-per-step`
exist only in a fork branch. `yeto/rl/algos/critic_fork.py` pins it as
`CRITIC_FORK_PIN` (`FORK_COMMITS`) and the spec-level `*_not_at_pin`
rejections check it. The island image does not carry these flags; they reach
the island only through the setup-time overlay (`--rl-miles-overlay auto`,
patches in `yeto/rl/adapters/miles/overlays/`, see "Clouds, controller and
images" in [MILES_RL.md](MILES_RL.md#clouds-controller-and-images)).

### Spec fields and defaults

The fields enter the canonical JSON and the hash only when
`execution.needs_critic=true`; on any other algorithm a non-default critic
field is refused (`[critic_fields_without_critic]`). GRPO hashes and the
default argv are unchanged.

| field | default on a critic spec | Miles / fork argv |
| --- | --- | --- |
| `advantage.gamma` | 1.0 | `--gamma` (always emitted) |
| `advantage.lambd` | 1.0 | `--lambd` (always emitted) |
| `advantage.lambd_mode` | `fixed` (`length_adaptive`: lambda = 1 - 1/(alpha*l)) | fork `--gae-lambd-mode` (only when not `fixed`) |
| `advantage.alpha` | none; required with `length_adaptive` | fork `--gae-length-alpha` |
| `advantage.gae_variant` | `vanilla` (`decoupled`, `cross_segment`) | fork `--gae-variant` (only when not `vanilla`) |
| `advantage.critic_lambd` | 1.0, filled in automatically with `decoupled` | fork `--gae-critic-lambd` |
| `critic.value_clip` | 0.2 | `--value-clip` |
| `critic.critic_lr` | none = inherit the actor lr (Miles default) | `--critic-lr` |
| `critic.critic_lr_warmup` | none | `--critic-lr-warmup-iters` |
| `critic.critic_updates_per_step` | 1 | fork `--critic-updates-per-step` (alias `--num-critic-epochs`; only when != 1) |
| `critic.value_loss` | `mse` (`hl_gauss`, SAO only) | fork `--value-loss-type classification ...` (SAO argv) |
| `critic.init` | `copy_actor_backbone` (`load` when `critic.load` is set) | stage W, see below |
| `critic.warmup_steps` | 0 | stage W `--num-critic-only-steps`; the main stage always gets `0` |
| `critic.param_mode` | `full` (`lora` reserved, refused) | -- |
| `loss.positive_lm_coef`, `positive_lm_source`, `positive_lm_reward_threshold` | none | fork `--positive-example-lm-loss-coef`, `--positive-example-source`, `--positive-example-reward-threshold` (only with source `reward`) |

`--gamma`, `--lambd`, `--value-clip`, `--num-critic-only-steps`,
`--critic-load`, `--critic-lr` and the fork flags are adapter-owned: absorbed
from the extra argv into the spec, refused on a value conflict, never passed
through raw.

### Rejections (before any GPU process)

Spec rules (`yeto/rl/engine/algorithm.py`, `yeto/rl/algos/critic.py`,
`yeto/rl/algos/sao.py`):

- `[critic_estimator]`: `ppo` without `execution.needs_critic=true`;
- `[critic_fields_without_critic]`: critic fields on a critic-free algorithm;
- `[critic_reward_kl]`: `kl.placement=reward` with `kl.coef != 0` (Miles
  shared PPO asserts `kl_coef == 0`); use `kl.coef=0` or `placement=loss`;
- `[critic_param_mode]`: `critic.param_mode=lora` (planned, not implemented)
  and `lora_*` fields without it;
- `[critic_not_at_pin]`: `value_loss=hl_gauss` without
  `policy_objective=sao_dis` (or without the fork pin); `alpha` without
  `length_adaptive`; `critic_lambd` without `decoupled`; `hl_gauss_bins`
  without `hl_gauss`;
- `[critic_init]`: `init=load` without `critic.load`; `critic.load` with
  `copy_actor_backbone`; `warmup_steps > 0` with `init=load`;
- `[positive_lm_threshold]`, `[sao_dis_fields]`, `[sao_not_at_pin]`;
- absorbing `--gae-variant cross_segment` (the fork's one-sample-per-rollout
  layout) is refused; see CompactionRL below.

Run rules (launch check `critic_shared_ppo`, Miles shared PPO limits): elastic
reconfiguration and `--indep-dp`; `--deploy-component trainer`;
`--critic-num-nodes` / `--critic-num-gpus-per-node` different from the actor's
(the critic shares the actor GPUs); `decoupled` outer sync. The driver also
refuses strict-avg with a critic but no critic syncer.

### Warm-up: two stages

`critic.init=copy_actor_backbone` builds the critic from the initial actor
checkpoint with a new value head. `critic.warmup_steps=N > 0` adds stage W:

1. **Stage W** (`yeto/rl/critic_warmup.py`): one plain Miles `train.py`
   launch, not the ports driver: `--num-rollout N --num-critic-only-steps N
   --critic-load <actor checkpoint> --critic-save <dir> --save-interval N`,
   with the main stage's algorithm flags. The actor never trains; the product
   manifest records the critic hash and checks the actor hash is unchanged.
   One product is reused by every island (keyed by algorithm + initial actor).
2. **Main stage**: `--num-critic-only-steps 0 --critic-load <stage-W
   product>`; the product hash enters the receipt
   (`CriticRunConfig.init_sha256`).

```bash
python3 -m yeto.rl.critic_warmup --dry-run \
  --extra "--advantage-estimator ppo --num-critic-only-steps 50" \
  --actor-checkpoint /ckpt/actor --critic-save /ckpt/critic-w
# stage_w_argv: ... --num-rollout 50 --num-critic-only-steps 50
#               --critic-load /ckpt/actor --critic-save /ckpt/critic-w --save-interval 50
# main_argv:    ... --num-critic-only-steps 0 --critic-load <stage-W product>
```

### GAE variants (fork `--gae-variant`, one extension point)

| variant | meaning | used by |
| --- | --- | --- |
| `vanilla` (default) | Miles GAE, element-wise unchanged | PPO |
| `length_adaptive` (`lambd_mode`) | lambda = 1 - 1/(alpha*l), l = response length | VAPO (alpha 0.05), SAO and CompactionRL (alpha 1.5) |
| `decoupled` | advantage uses the policy lambda, value target = GAE(`critic_lambd`) + V | VAPO, SAO |
| `cross_segment` -> fork `cross_segment_per_sample` | each compaction segment is its own sample, reward at each segment end, bootstrap 0; local GAE times (gamma*lambda)^N_{>s} (paper eq. 13-15) | CompactionRL |
| fork `cross_segment` (old) | whole rollout as one sample with per-token `segment_ids`; earlier segments carry no terminal reward | kept in the fork as the explicit control mode `--gae-variant cross_segment_whole_rollout`; the ambiguous name `cross_segment` is refused by Yeto |

The fork's GAE tests compare against `tests/rl_gae_reference.py`.

### Outer sync, round-cut, ledger

- **Two syncers**: the launcher starts a second syncer for the critic (port
  29401, layout `critic_layout_hash`, its own checkpoint and tape;
  `dry_run_plan` shows a `critic_syncer` block only for critic algorithms).
  `DualStrictAvgSync` commits both channels atomically: a round is applied
  only when actor and critic both reach v+1, otherwise neither is applied
  (`CrossChannelCommitError`).
- **fp32 master weights**: the critic channel averages and writes back the
  fp32 master weights; the critic optimizer state is kept across rounds (the
  actor's is reset as before). `check_unverified_allowance` still refuses any
  critic run with outer sync.
- **Round-cut checkpoint**: `save_cut` stores critic weights, optimizer and
  scheduler state under `<cut>/critic/`; `restore_cut` refuses a critic round
  that differs from the actor round, a missing pointer, or tampered weights;
  resharded restore with a critic is refused.
- The tape and ledger record the critic weight hash, `value_loss` and
  explained variance.

### Algorithms

**PPO.** `advantage.estimator=ppo`, `execution.needs_critic=true`; defaults
above.

```bash
M="python3 -m yeto.rl.adapters.miles.algorithm_flags"
A=--rl-allow-unverified-mechanism
$M --dry-run --extra "--advantage-estimator ppo"
# rejected: advantage estimator 'ppo' not supported (...)
$M --dry-run --extra "--advantage-estimator ppo" $A advantage_estimators:ppo $A execution:critic
# miles_argv: --advantage-estimator ppo --gamma 1.0 --lambd 1.0 --value-clip 0.2 --num-critic-only-steps 0
$M --dry-run --extra "--advantage-estimator ppo --kl-coef 0.1" $A advantage_estimators:ppo $A execution:critic
# rejected: [critic_reward_kl] kl.placement='reward' with kl.coef=0.1 and a critic ...
```

**VAPO** (`yeto/rl/algos/vapo.py::vapo_spec()`, arXiv 2504.05118v3). PPO +
decoupled GAE (critic lambda 1.0) + length-adaptive policy lambda with
**alpha = 0.05** (the paper value; the fork default 1.5 is SAO's) +
clip-higher (0.2 / 0.28) + token-level loss + positive-example LM loss
**mu = 0.1** + value warm-up 50 steps; critic lr 2e-6. Choices:

- the critic is initialised from the actor backbone (`copy_actor_backbone`)
  plus stage-W warm-up, and the warm-up is checked for return fit,
  calibration and explained variance before the main stage;
- "correct sample" uses an explicit success predicate (the reward function's
  boolean `success`/`is_correct` metadata, `loss.positive_lm_source=success`)
  rather than an implicit reward threshold; `positive_lm_source=reward` (reward
  above `positive_lm_reward_threshold`) is accepted only when the spec
  declares the reward binary;
- the positive-example NLL is normalised by the global number of
  positive-example tokens (paper eq. 9).

Differences from the paper: value model initialised from the actor, not from
the reward model; success predicate defined by Yeto; value_clip (0.2), KL (0)
and the warm-up schedule are not given by the paper; group sampling (16 per
prompt, 512 prompts) is a run setting, not part of the hash.

```bash
$M --dry-run --extra "--advantage-estimator ppo --gae-variant decoupled --gae-lambd-mode length_adaptive \
  --gae-length-alpha 0.05 --eps-clip 0.2 --eps-clip-high 0.28 --calculate-per-token-loss \
  --positive-example-lm-loss-coef 0.1 --positive-example-source success \
  --critic-lr 2e-6 --num-critic-only-steps 50" \
  $A advantage_estimators:ppo $A execution:critic $A features:gae_decoupled \
  $A features:gae_length_adaptive $A features:positive_example_lm_loss
# algorithm_spec_sha256 equals vapo_spec(); miles_argv ends with
#   --num-critic-only-steps 0 --gae-variant decoupled --gae-critic-lambd 1.0
#   --gae-lambd-mode length_adaptive --gae-length-alpha 0.05
#   --positive-example-lm-loss-coef 0.1 --positive-example-source success
# (--num-critic-only-steps 50 is the warm-up length: it goes to stage W, the main stage gets 0)
```

**SAO** (`yeto/rl/algos/sao.py::sao_algorithm_spec(recipe)`): the SAO
streaming recipe as an AlgorithmSpec: `policy_objective=sao_dis`,
gamma/lambda 1/1, decoupled + length-adaptive alpha 1.5, HL-Gauss value loss
(51 bins), critic lr 5e-6, `critic_updates_per_step = num_critic_epochs = 2`,
two roles with separate layouts and syncers. Not ported: re-forwarding the
updated value to the actor, offline value pretrain; `critic_freeze_attention`
is in the fork but not in the spec. The SAO streaming entry
(`--rl-sync-preset sao-streaming-full`) is unchanged and stays the supported
way to run SAO.

**CompactionRL** (`yeto/rl/algos/compactionrl.py::compactionrl_spec()`,
arXiv 2607.05378). Rollout side `yeto/rl/compaction.py`: compaction when
`C - |h_t| < 10240`, same-policy summary, rebuilt context with the last k=2
steps, at most 3 compactions; each segment is one sample with the shared
reward and metadata `tokens_after` (N_{>s}) and `gae_length`. Training side:
local GAE per segment times the cross-segment decay (gamma*lambda)^N_{>s}; the
lambda length l = the **total optimised tokens of the whole rollout** (all
segments share one lambda); alpha = 1.5; gamma = 1; KL 0; token-level loss;
critic lr 3e-6; 2 critic updates per policy update; 50-step warm-up; the
critic target is the **uncorrected local return** (local advantage + V); the
old one-sample implementation stays as the explicit control
`cross_segment_whole_rollout`. The 9-section summary template is a Yeto
draft (the paper gives no section names).

```bash
$M --dry-run --extra "--advantage-estimator ppo --gae-variant cross_segment_per_sample \
  --gae-lambd-mode length_adaptive --gae-length-alpha 1.5 --calculate-per-token-loss \
  --critic-lr 3e-6 --critic-updates-per-step 2 --num-critic-only-steps 50" \
  $A advantage_estimators:ppo $A execution:critic $A features:gae_cross_segment \
  $A features:gae_length_adaptive $A features:critic_multi_update
# algorithm_spec_sha256 equals compactionrl_spec(); miles_argv:
#   --advantage-estimator ppo --calculate-per-token-loss --gamma 1.0 --lambd 1.0 --value-clip 0.2
#   --critic-lr 3e-06 --critic-updates-per-step 2 --num-critic-only-steps 0
#   --gae-variant cross_segment_per_sample --gae-lambd-mode length_adaptive --gae-length-alpha 1.5
# the same with --gae-variant cross_segment is rejected
#   ("... use --gae-variant cross_segment_per_sample")
```

## Train-inference mismatch corrections

Registered by `yeto/rl/algos/mismatch_correction.py`. On the serial island
driver the behavior policy (SGLang generating with `W_r`) and `pi_old`
(Megatron re-scoring with `W_r`) are the **same weights**: the per-group
policy token enforces it. The ratio `exp(train_old - rollout)` therefore
measures only the numeric difference between the two engines (kernels, bf16,
the LoRA weight publish path). These mechanisms correct that difference; they
are not an off-policy license: `execution.max_policy_staleness` stays 0 and a
non-zero value is rejected before any GPU process.

Every threshold is explicit. Yeto sets no default and does not inherit Miles'
parser defaults (for example `--tis-clip-low 0`); the translated argv always
carries the values from the spec. At most one importance-weighting correction
(observe-only, TIS, IcePop or MIS) can be selected, because they share
`correction.method` and Miles has one `--custom-tis-function-path`. Selecting
two of them through extra argv is a conflict that names both flags. OPSM alone
uses `correction.method: "opsm"`; combining OPSM with TIS, IcePop or MIS is
not supported. Observe-only can never be combined with another mechanism.

| mechanism | spec (`correction`) | Miles argv / attributes | masks tokens |
| --- | --- | --- | --- |
| `mismatch_observe` | `method: custom`, `function: yeto.rl.algos.mismatch_observe.observe_mismatch`, `mismatch_metrics: true` | `--use-tis --custom-tis-function-path ... --get-mismatch-metrics` | no (loss and gradient `torch.equal` to no correction) |
| `tis` | `method: tis`, `tis_clip`, `tis_clip_low` | `--use-tis --tis-clip H --tis-clip-low L` | no |
| `icepop` | `method: custom`, `function: miles...corrections.icepop_function`, `tis_clip_low` < `tis_clip` | `--use-tis --tis-clip H --tis-clip-low L --custom-tis-function-path ...` | yes |
| `opsm_trainer` / `opsm_rollout` | `method: opsm`, `opsm_delta`, `opsm_old_logprob_source` | `--use-opsm --opsm-delta d` (+ `--use-rollout-logprobs` for `rollout`) | yes (sequences) |
| `mis` / `mis_mask` | `method: custom`, `function: yeto.rl.algos.vendor.miles_mis.compute_mis_weights_with_cp`, `mis_level`, `mis_mode`, `mis_upper_bound` (+ `mis_lower_bound` for clip/mask), `mis_batch_normalize` | `--use-tis --custom-tis-function-path ...` plus namespace attributes `tis_level`, `tis_mode`, `tis_*_bound`, `tis_batch_normalize`, `use_rs=false` | `mis_mask` only |

Numeric tests against the Miles sources at `MILES_NEXT_COMMIT` are in
`tests/test_rl_mismatch_observe.py` (run in the Miles venv); spec,
translation and rejection tests are in `tests/test_rl_mismatch_correction.py`.
Which corrections are declared is defined by `MILES_DECLARED`; any other
mechanism fails at startup with a list of the supported ones. For a
single-island smoke only, `--rl-single-island-no-sync
--rl-allow-unverified-mechanism corrections:<name>` (and `features:<name>`
where needed) admits them. With one optimizer step per round, OPSM cannot
trigger by construction, because pi_theta = pi_old. For the same reason
`ess_ratio` and `ois` are always 1: they are pi_theta/pi_old statistics and do
not reflect the train/inference mismatch.

Threshold meaning. The literature values below are shown only for
orientation:

- TIS: the weight is `clamp(ratio, tis_clip_low, tis_clip)` and multiplies the
  per-token PPO loss. No token is masked. Miles' own example uses `C = 2`.
- IcePop: a token whose ratio lies in `[tis_clip_low, tis_clip]` gets the
  weight `ratio`; any other token gets weight 0. The paper's interval is
  `[0.5, 5]`.
- OPSM: a sequence is masked when its advantage is negative and the
  sequence-level `mean(pi_old - pi_theta) > opsm_delta`. With
  `opsm_old_logprob_source: "trainer"` (the default, as in Miles) `pi_old` is
  the Megatron re-score, so OPSM only masks sequences that the inner
  mini-batches moved too far; it does not cover train/inference mismatch.
  `"rollout"` requires `use_rollout_logprobs: true`
  (`--use-rollout-logprobs`), which **also replaces `pi_old` in the PPO
  ratio**, not only in OPSM. Because of that, it is rejected together with
  TIS. The DeepSeek-V3.2 report uses the inference-side logprobs.
- MIS: `mis_mode` is `truncate` (cap at the upper bound), `clip` (clamp to
  `[lower, upper]`) or `mask` (zero the tokens outside the interval), and
  `mis_level` is `token`, `sequence` or `geometric` (geometric mean). Miles'
  example suggests `[0.9999, 1.0001]` for the geometric level. Rejection
  sampling and the veto threshold of Miles' `mis.yaml` are not exposed. MIS is
  a verbatim copy of the upstream Miles train/infer mismatch helper
  (Apache-2.0; header in `yeto/rl/algos/vendor/miles_mis.py`), because plugins
  must live under `yeto.`/`miles.`. On a Miles upgrade, re-copy it and rerun
  `tests/test_rl_mismatch_observe.py`, which checks that the copy equals the
  original. Do the same with `ICEPOP_SOURCE_SHA256`.

Zero-gradient rule: `icepop`, `opsm_*` and `mis_mask` may legitimately mask a
whole round. A round with zero gradient is accepted only when the trainer
reports `masked_fraction == 1.0`. The helper
`mismatch_correction.masked_fraction_from_metrics` derives it from the Miles
metrics: `tis_clipfrac` for IcePop and `mis_tis_mask_fraction_low + _high` for
MIS. It returns None for OPSM, whose `opsm_clipfrac` is not a token fraction.
An unknown fraction keeps the strict rule. A non-finite `grad_norm` always
fails.

Metrics: `tis`, `tis_abs`, `ois`, `train_rollout_kl`,
`train_rollout_logprob_abs_diff` and `ess_ratio`, plus
`mismatch_outside_0p5_5` for observe-only (the fraction of tokens outside the
IcePop reference interval `[0.5, 5]`). They appear in Miles' `train/*` log.
The serial colocated mode publishes LoRA weights over CUDA IPC, while a
partitioned placement uses NCCL broadcast, so a mismatch measured in one mode
does not carry over to the other.

Example specs for every correction, checked through the dry run by
`tests/test_rl_mismatch_correction.py`, are in
[MILES_RL.md](MILES_RL.md#mismatch-correction-spec-examples).
