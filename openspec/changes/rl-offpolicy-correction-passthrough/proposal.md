# Proposal

## Why

A trainer that starts and stops, and a sampler that keeps generating while the trainer trains, produce samples that are 0 to A versions old. Long trajectories also cross several versions. The trainer must correct these samples, not only filter them.

Today (source: `infra-drafts/SESSION23/sampler-trainer-startup-explore.md` §5 and §9):
- yeto has one hard filter: a group is dropped when its oldest token is older than the limit (`split_by_age`, `yeto/rl/engine/policy_age.py:115`).
- yeto caps `max_policy_age` at 1 on both backends (`SUPPORT`, `yeto/rl/adapters/miles/policy_age.py:36` and `yeto/rl/adapters/verl/policy_age.py:22` on `s23-g3-fp`). A limit of 2 or more is refused today.
- The neutral `CorrectionSpec.method` has only `none | tis | opsm | custom` (`yeto/rl/engine/algorithm.py:783-792` on `d6c58fb8`). The verl adapter accepts only `none, tis` (`yeto/rl/adapters/verl/config.py:28`).
- Both backends already implement truncated importance sampling (TIS), IcePop, and masked importance sampling (MIS). Their defaults differ (explore §9.4).

User decisions (2026-10-14; source: `infra-drafts/SESSION24/DISCUSS.md` D2, which replaces the two-threshold mask decision Q3 of explore §9.1):
- Use decoupled PPO, close to AReaL. Do not mask tokens by version. The `A_soft` threshold is removed.
- The version gap of a trajectory is the current trainer version minus the trajectory start version.
- Three limits: admission (do not start a trajectory that is already more than `k` versions behind), in flight (continue generation across weight updates; do not abort), consumption (drop the whole group when the gap is over `A_hard`).
- All response tokens enter the loss. The importance weight is `π_prox / π_behav`, truncated at `C`. The PPO clip anchor is `π_prox`.
- Admission uses existing backend mechanisms. Miles source is not changed. Exact admission is deferred until GPU runs show the drop ratio.
- verl and Miles are both in scope. Each backend uses its own async mode: Miles `fully_async`, verl `verl/experimental/fully_async_policy`.
- Ratio masks (IcePop, MIS) stay in the backends. yeto passes their settings through. They are optional and off by default.
- Phase 1 (explore §10): correction settings go into the run manifest and the tape. A mismatch gives a warning and an event. It does not refuse the launch and it does not enter the algorithm hash.

## What Changes

- Extend the neutral `CorrectionSpec` with explicit fields: `method` (`none | tis | icepop | mis_mask | opsm | custom`), `level` (`token | sequence | geometric`), `lower`, `upper`, `veto`, `batch_normalize`, `ratio_mask` (default off). yeto writes every field. No field takes a backend default.
- Map decoupled PPO to both backends. Miles: `--use-tis --tis-clip C`; `--use-rollout-logprobs` and `--skip-actor-forward-only` are refused. verl: `bypass_mode=False`, `rollout_is=token`, `rollout_is_threshold=C`, set explicitly over the `fully_async` yaml defaults.
- Version gap from the trajectory start version. The existing oldest-token rule is accepted only where it equals the start version.
- Admission by approximation: Miles buffer capacity plus `--max-weight-staleness`; verl `async_training.staleness_threshold`.
- In flight: Miles partial rollout with `mask_offpolicy_in_partial_rollout` off; verl `partial_rollout=True`.
- Consumption drop over `A_hard` (`max_policy_age`). Miles has it; yeto adds it in the verl adapter. Raise the `SUPPORT.max_policy_age` cap so that `A_hard ≥ 2` is allowed.
- Monitoring: `offpolicy_gate` with counts per reason, length histogram of dropped trajectories, truncation ratio by turn position.
- Phase 1 warn-only contract. Phase 2 (L6b): the correction group enters `algorithm_sha256`.

## Capabilities

### New Capabilities
- `rl-offpolicy-correction`: neutral correction settings passed to the backend, decoupled PPO on Miles and verl async modes, the three version limits, drop and truncation monitoring, and the phase 1 warn-only contract.

### Modified Capabilities
<!-- Existing algorithm-hash requirements are not changed in phase 1: the correction group is excluded explicitly. Phase 2 will modify them. -->

## Impact

- `yeto/rl/engine/algorithm.py` (`CorrectionSpec`, hash exclusion), `yeto/rl/algos/mismatch_correction.py`.
- `yeto/rl/adapters/miles/algorithm_flags.py`, `yeto/rl/adapters/miles/policy_age.py`, `yeto/rl/adapters/verl/config.py`, `yeto/rl/adapters/verl/policy_age.py`, verl round collection (`fully_async_round.py`).
- `yeto/rl/engine/policy_age.py`, `yeto/rl/engine/island_ledger.py` (`StalenessPolicy.correction` default).
- Run manifest and tape writers; launcher cross-island compare.
- Depends on P0-S per-trajectory version evidence (schema 4) and the S23 integration branch. These are not on `d6c58fb8`; see GOAL.md prerequisites.
