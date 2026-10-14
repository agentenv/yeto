# Proposal

## Why

PM item 6 says the tamper-proof signature on reward results is not needed. The user agreed to remove it (`infra-drafts/SESSION23/P2-reward-hash.md`, 00-DECISIONS Q3). The reasons: rewards are produced inside a controlled sandbox, the sandbox does not escape, and the signature does not stop a real threat. Yet it forces every run path to distribute and check one key. The signature is also an obstacle to connecting Harbor (as stated in README line 46; this code reading found no Harbor-related code in yeto, so the exact form of the obstacle is **unverified**).

Compatibility hashes are correctness checks and MUST stay. The island HELLO HMAC is cross-host network authentication. Q3 decided to keep it.

## What Changes

**Remove (reward tamper-proofing)**
- `yeto/rl/tbench_outcome.py`: remove `sign_outcome`, `build_signed_metadata`, the signature verification in `verify_outcome`/`verified_outcome`, `validate_hmac_key_source`, `_secret`, `MAC_KEY`, `HMAC_ENV`, `HMAC_FILE_ENV`. Keep `build_outcome`, `canonical_outcome`, `_validated` (field, status, and range checks).
- `yeto/rl/harness/codex/reward.py`: remove `MAC_KEY`, `_hmac_key`, `sign_outcome`, and the MAC comparison in `_verified_outcome`. Keep the schema, admission, retry ledger, and reward range check.
- Signing callers: `agent.py:720`, `codex_harness_agent.py:2376`, `codex_openenv_agent_function.py:427,482,687`, `tbench_reward.py`, `harness_glue/codex_generate.py`, `tbench_direct_preflight.py:130`, `sao_streaming_runtime.py:535`, `harness/codex/preflight.py`.
- Key distribution: remove `TBENCH_REWARD_HMAC_KEY`, `TBENCH_REWARD_HMAC_KEY_FILE`, `SECRLENV_REWARD_HMAC_KEY` from the pass-through and key lists in `launcher.py:1400,3488`, `preflight.py:220`, `modal_rlf_baseline.py:49`.
- `yeto/rl/trajectory_evidence.py`: the outcome evidence hash no longer mixes in the mac (`_outcome_evidence_hash`, lines 196/233/273). Raise the evidence schema to 3. **BREAKING**: newly written evidence hashes differ from old evidence.
- (D1 decided: moved out of this change, not removed) `yeto/island_credential_guard.py` and its call sites.

**Keep (correctness)**
- `BackendIdentity` in `backend_identity.py` (engine, engine_commit, device_family, param_map_sha256, compat_group) and `session_contract_hash`; `lr_schedule_sha256`; the algorithm hash; `training_contract_hash`; `ExecutionProfile.contract_hash`.
- `reward_contract_hash` (`rl/contracts.py:50`, `trajectory_evidence.py:82`).
- `episode_id` in `rlf/contracts.py`.
- The island HELLO HMAC and the per-frame HMAC in elastic mode (`seal`/`open` at `syncer/src/elastic.rs:146-167`), and `YETO_ISLAND_HMAC_KEY`.

## Decisions Pending

**Decided (2026-10-13, user): D1-D4 all as recommended.** For D1, the credential guard is not handled in this change. It moves out for separate evaluation.

**D1: Whether to remove `island_credential_guard` in this change.**
- Fact: it checks whether the learner island process holds cloud credentials. It has nothing to do with rewards. There are 8 call sites in 7 files, not the 4 that P2 states.
- Option A: remove it in this change. Option B: move it out of this change and evaluate it separately. Option C: keep it.
- Recommend B. Reason: it is not a reward hash. Mixing it in would blur the scope of "remove tamper-proofing". It guards against credential leaks to the island, and the sandbox-escape argument does not apply to it.

**D2: Whether old evidence (schema 1 and 2, with mac) must stay readable.**
- Option A: the reader keeps accepting 1 and 2, checks only the structure, and does not verify the mac. Option B: accept only 3, and old evidence becomes invalid.
- Recommend A. Reason: evidence from old runs is still used for replay comparison. Read-only compatibility costs about 30 lines. Writing uses only 3.

**D3: The exact form of the Harbor obstacle.**
- Fact: yeto code has no Harbor reference. The obstacle is **unverified**.
- Option A: proceed on the assumption that removing the signature clears it, and add a fix when Harbor is connected. Option B: first have the user or PM explain the obstacle, then decide the scope.
- Recommend A. Reason: the user already agreed to remove the signature. This does not depend on a Harbor conclusion.

**D4: Whether to keep the metadata in the `outcome` field that is unrelated to the signature (benchmark, verifier, domain separator string).**
- Recommend keeping it. Reason: replay comparison and reward_contract_hash need it. The domain separator string can stay in the evidence hash.

## Boundary with rl-code-quality-cleanup

- Task 4.8 there, "embed the algorithm hash in the island contract hash", changes the **composition of a compatibility hash**. This change **does not change** the composition or value of any compatibility hash.
- Phase 4 there, "merge the TB2 and RLF trust paths into one", merges `tbench_outcome.py` and `harness/codex/reward.py`. This change first removes the signature on both sides and does not merge the files. The merge is left to that change.
- If both changes touch `trajectory_evidence.py`, this change lands first (schema 3), and that change builds on 3.

## Capabilities

### New Capabilities
- `rl-reward-integrity-scope`: The scope of reward integrity checks. Reward results get only structure and range checks and no key signature. Compatibility hashes and the island HMAC stay.

### Modified Capabilities
(None. The main specs have no requirement about reward signatures.)

## Impact

- Code: the files listed above. About 330-420 lines of source can be removed (P2 estimate, not recounted line by line in this reading).
- Tests: `test_rl_tbench_outcome.py`, `test_rlf_env_lease.py`, `test_secrlenv_adapter.py`, `test_harness_codex_openenv.py`, `test_rl_trajectory_evidence.py`, `test_rl_tbench_trajectory_evidence.py`, `test_codex_head_staging.py`, `test_rl_tbench_direct_preflight.py`, `test_rlf_baseline.py`, and others need rewriting. `test_secret_handling_hardening.py` **cannot be deleted whole** (see design.md).
- Must keep passing: `test_rl_backend_identity.py`, `test_rl_elastic_backend_identity.py`, `test_hw_compat_group.py`, `test_export.py`.
- External: the three reward key environment variables are no longer needed. Setting them does not cause an error (to be implemented as ignored).
