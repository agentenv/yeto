# Design

## Context

Backend facts are from `infra-drafts/SESSION23/sampler-trainer-startup-explore.md` §9.2–§9.4 and from `infra-drafts/SESSION24/DISCUSS.md` D2 (code reading on verl fork `acad9875` and Miles fork `s17-infra-fork-m`; no GPU run). User decisions are from explore §10 and DISCUSS.md D2 (2026-10-14).

Terms:
- Start version of a trajectory: the policy version that generated its first response token.
- Version gap of a trajectory: current trainer version minus its start version.
- `k`: the admission limit. A new trajectory is not started when its start version would be more than `k` behind.
- `A_hard`: the consumption limit. It is the existing `max_policy_age`. `A_hard ≥ k`.
- `π_behav`: the per-token logprob that the rollout engine returned (behavior policy).
- `π_prox`: the proximal policy. The trainer recomputes it as the "old logprob" before the update.
- `C`: the upper truncation bound of the importance weight.

Defaults: `A_hard` keeps its current default, so the default behavior does not change. Recommended values go into example configs only.

## Decisions

### D1. Neutral fields, written in full
`CorrectionSpec` gets `method`, `level`, `lower`, `upper`, `veto`, `batch_normalize`, `ratio_mask`. `lower` and `upper` are required when `method` is not `none`. `upper` is `C`. The reason: the backends have different defaults (explore §9.4), so a missing field gives different math on two islands.

`C` has no agreed default. Miles and verl both default to 2.0 (DISCUSS.md D2). The AReaL example may use 5.0 (unverified). Example configs use 2.0 until a GPU run gives evidence.

### D2. Decoupled PPO is the training objective
All response tokens enter the loss. For each token:
- importance weight `w = min(π_prox / π_behav, C)`;
- PPO clip ratio `π_θ / π_prox`, clipped as usual.

This matches decoupled PPO in AReaL. Code reading says both backends can express it without source changes (DISCUSS.md D2, unverified on GPU).

Miles mapping:
- `--use-tis --tis-clip C` (plus `--tis-clip-low` from `lower`).
- `--use-rollout-logprobs` is refused: it makes `π_rollout` the clip anchor, so `π_prox` is lost.
- `--skip-actor-forward-only` is refused: it skips the forward pass that computes `π_prox`.

verl mapping:
- `algorithm.rollout_correction.bypass_mode=False`, `rollout_is=token`, `rollout_is_threshold=C`.
- The `fully_async` yaml sets `bypass_mode: True` (`verl/experimental/fully_async_policy/config/fully_async_ppo_trainer.yaml:98`, `fully_async_ppo_megatron_trainer.yaml:97`, checked) and `rollout_is` defaults to `sequence` (DISCUSS.md D2). The adapter writes all three values explicitly.

`π_prox` differs between backends. Miles recomputes it on the weights before each update. verl takes it from the weights of the first local step of the current cycle (DISCUSS.md D2). The adapter records the anchor policy (`prox_anchor=per_step` or `first_local_step`) in the manifest and the tape. A run that mixes the two across islands gets `correction_config_warning`.

L6.6 result (code reading, verl fork `acad9875`, checked): no Python code in the fork reads `actor.use_rollout_log_probs`; only the config dataclasses declare it. The `fully_async` yaml value `True` has no effect. With `bypass_mode=False`, `FullyAsyncTrainer._compute_old_log_prob` recomputes `old_log_probs` on the weights of local step 1 of the sync cycle. A CPU test (`tests/test_rl_verl_offpolicy_correction.py`) fails if a reader appears. Not run on GPU.

### D3. Backend mapping table (ratio masks and other methods)

| Neutral | Miles | verl |
|---|---|---|
| `tis`, token (default for decoupled PPO) | `--use-tis --tis-clip=upper --tis-clip-low=lower` | `rollout_is=token`, `rollout_is_threshold=upper`; `lower>0` cannot map → warning |
| `ratio_mask=icepop`, token | `--custom-tis-function-path=…icepop_function`, `--tis-clip*` = range | `rollout_is=token`, `rollout_is_threshold="lower_upper"` |
| `ratio_mask=mis`, token or sequence | MIS plugin, `tis_mode=mask`, bounds | `rollout_rs=token_k1` or `seq_mean_k1` (geometric) with bounds |
| `opsm` | `--use-opsm --opsm-delta` | no mapping → warning |
| `veto` | MIS plugin `rs_veto_threshold` | no direct mapping → warning |

`ratio_mask` defaults to `none`. Rows that say "warning" do not refuse in phase 1. The adapter emits `correction_config_warning` with the field, the backend, and the reason. The event records the setting that actually ran.

Loss denominator after a ratio mask: Miles keeps the original denominator. The neutral meaning is "denominator unchanged". L6.6 result (CPU, verl and Miles functions loaded from source, checked): verl IcePop (`rollout_is_threshold="lower_upper"`) sets the weight to 0 and keeps `response_mask`, so the denominator is unchanged and the weights equal Miles `icepop_function`. verl rejection sampling (`rollout_rs`, the MIS row) removes tokens from `response_mask`; `batch_num_tokens` is the sum of that mask, so the denominator shrinks. No verl setting keeps it, so the MIS row on verl emits a `ratio_mask` warning. TIS with `lower=0` gives equal weights on both backends; with `lower>0` verl does not raise small weights (warning row in D3).

### D4. Version gap from the start version
The gap of a trajectory is the current trainer version minus its start version. Miles and yeto compute the gap from the oldest token version in the group. With continued generation the oldest token is the first token, so the two rules give the same value (inferred; not checked case by case). The spec requires a value equal to the start-version rule. L6.12 checks the equality on recorded tapes. If it fails, the adapter computes the start version from `TokenProvenance`.

L6.12 result (2026-10-15, L6.10 GPU runs, `s1-runs/s24-l6.10/`, `A_hard=1`): the equality holds on all recorded data, but the continued case was not exercised.
- Miles (two runs, 2,560 samples, about 1.2 M response tokens, `--rl-debug-dump`): every response token has a version span. In every sample the first-token version equals the oldest version, and versions never decrease along a sample. No sample spans more than one version (no carry-over happened in Miles `fully_async`), so a continued trajectory across two weight updates is still checked only by the CPU test.
- verl reports only the first and last version per trajectory (`min/max global_steps`). The start version is the first one by construction. In the verl run with thinking on, 54 of 320 groups spanned two versions. No per-token check is possible on verl.
No change to the adapter is needed now. Recheck on a Miles run that carries groups across updates.

### D5. Three limits
1. Admission (`k`), approximate:
   - Miles: buffer capacity plus `--max-weight-staleness` at consumption (`miles/rollout/fully_async_data_buffer.py:197`, checked).
   - verl: `async_training.staleness_threshold` sample quota (yaml default `0.1`, `fully_async_ppo_trainer.yaml:12`, checked).
   - Neither backend refuses a new trajectory by its exact start version. Exact admission is deferred; the decision depends on the drop ratio in L6.10.
2. In flight: generation continues across weight updates. Miles uses partial rollout; verl uses `partial_rollout=True` (yaml default, `fully_async_ppo_trainer.yaml:23`, checked). Miles `mask_offpolicy_in_partial_rollout` SHALL be off: when on, Miles masks the old-version tokens of a continued sample (`miles/rollout/sglang_rollout.py:291`, checked), which is the version mask this design rejects. The adapter refuses it.
3. Consumption (`A_hard`): a group whose gap exceeds `A_hard` is dropped whole. Miles does this through `--max-weight-staleness`. verl has no drop by version at consumption; the yeto verl adapter adds it at round collection.

`SUPPORT.max_policy_age` is 1 on both adapters today (`yeto/rl/adapters/miles/policy_age.py:36`, `yeto/rl/adapters/verl/policy_age.py:22`, on `s23-g3-fp`, checked). L6.9 raises the cap so that `A_hard ≥ 2` is allowed. The new cap value is set by L6.9 together with the backend limits.

### D6. Monitoring
Per round, emit `offpolicy_gate` with:
- counts per reason (`kept`, `group_dropped`, and backend admission drops where the backend reports them);
- a response length histogram of dropped trajectories, with bucket edges fixed in code;
- the ratio of truncated tokens (`w` hit `C`) by turn position. Miles gives token-level data. verl gives only trajectory-level min and max version, so on verl the adapter reports per-trajectory values and marks the field `granularity=trajectory`.

The dashboard reads this event.

### D7. `cross_version_is` is observe-only
It does not change the loss today (call site `yeto/rl/adapters/miles/carry_over.py:326` per explore). It keeps the per-version-segment statistics that `PolicyAgeGovernor` needs. Its clip bounds come from `CorrectionSpec.lower/upper`. `StalenessPolicy.correction` derives from the spec instead of the fixed `"tis"`.

### D8. Phase 1 contract: manifest and tape, not hash
The correction group is removed from the canonical JSON that produces `algorithm_sha256`. A test locks that a change in correction fields does not change the hash in phase 1. The full group, the backend-mapped values, and `prox_anchor` go into the run manifest and each island tape. The launcher compares the groups of all islands at start. A difference gives a warning and an event.

The MIS plugin path is a PluginRef. Its source hash already enters the spec (`algorithm.py:435` per explore). Phase 1 keeps that.

### D9. Phase 2 (later)
After phase 1 has one passing GPU run, the correction group enters `algorithm_sha256`, and unmappable settings and cross-island differences are refused at launch (same refusal path as `yeto/rl/adapters/verl/entry.py:34`). This needs a hash migration under `hash-migration.md`. L6b SHALL land at the same version boundary as the semantic hash replacement of change `rl-identity-tiering`, so that the hash changes once.

## Alternatives considered

### A1. Two thresholds with version masks (rejected 2026-10-14)
The earlier design had `A_soft ≤ A_hard`. A token gap `≤ A_soft` got TIS. A gap between `A_soft` and `A_hard` got a mask: whole turns for agentic tasks, a token prefix for other tasks. A gap over `A_hard` dropped the group.

Reasons to reject:
- Position bias. The oldest tokens are always the first tokens or turns. The opening turns get no gradient, often, and reward credit moves to later turns.
- A mask does not fix the real error. Token-level IS cannot correct the state distribution shift. The error term of the policy improvement bound depends on the oldest version used. So the main control is a version limit at rollout, and IS is a supplement.
- Cost: the mask needs per-turn version records and a loss denominator aligned across backends.
- Industry practice (AReaL, Solar Open 2, verl) is continued generation, all tokens in the loss, and IS correction (DISCUSS.md D2).

`A_soft` is removed.

### A2. Ratio masks (IcePop, MIS) as the main tool (rejected as default)
A ratio mask looks at how far the policies differ, not how old the sample is. Both backends have it. It stays available through `ratio_mask`, off by default, because it drops signal like a version mask and it does not give an interpretable hard limit.

### A3. Exact admission now (deferred)
Exact admission needs a Miles source change. The user chose the approximation first and a decision after L6.10.

## Risks

- Approximate admission can admit trajectories older than `k`. Consumption drops them at `A_hard`, which wastes rollout compute. The drop ratio in `offpolicy_gate` is the control.
- `π_prox` differs between Miles and verl (D2). Cross-backend metrics are not exactly comparable.
- verl loss denominator after a ratio mask and verl `use_rollout_log_probs` with `bypass_mode=False`: checked on CPU by L6.6 (see D2, D3). Not run on GPU.
- Whether sglang returns `weight_versions` per token is partly verified. In L6.10 the Miles island's SGLang returned version spans for all 2,560 samples and they covered every response token. No version changed inside one generate call, so a call with more than one span was not observed. The truncation-by-turn metric and D4 depend on it.
