## 1. Prerequisites

- [x] 1.1 User decides D1-D4 in proposal.md (2026-10-13: all as recommended)
- [x] 1.2 Use grep on `scripts/` and `tools/` for places that compare evidence by mac, and record the result in design.md (result: none. Only `tools/probes/build_tbench21_sao_diloco_plan.py` had one gate name, and it is removed)

## 2. Remove the reward signature (first PR)

- [x] 2.1 `tbench_outcome.py`: remove signing, signature verification, and key reading; keep build/canonical/_validated
- [x] 2.2 `harness/codex/reward.py`: remove MAC_KEY, _hmac_key, sign_outcome, and the MAC comparison
- [x] 2.3 Change the signing callers: agent.py, codex_harness_agent.py, codex_openenv_agent_function.py, tbench_reward.py, harness_glue/codex_generate.py
- [x] 2.4 Remove the key-source checks in preflight: tbench_direct_preflight.py, sao_streaming_runtime.py, harness/codex/preflight.py
- [x] 2.5 Remove the three reward key variables from the key lists in launcher.py, preflight.py, modal_rlf_baseline.py, codex_openenv_agent_function.py
- [x] 2.6 `trajectory_evidence.py`: add schema 3 whose outcome hash has no mac; schema 1 and 2 stay readable, with only a structure check and no mac verification (decided in D2)
- [x] 2.7 Rewrite the affected tests (the Impact list in proposal.md); remove the cases that test only the signature
  - Addition: `test_secrlenv_adapter.py` tests the external package `yeto_miles_secrlenv`. It has always been skipped on this machine, and this change did not touch it
  - Addition: `test_rl_trajectory_evidence.py` also depended on the external package and was always skipped. It now uses `harness/codex/reward.py` in the repository, so it actually runs
- [x] 2.7b Recompute the three source pin values: `agent.py`, `codex_generate.py`, `codex_harness_agent.py` (`pins.py`, `yeto/rl/__init__.py`). The files changed, so the pin values must change with them
- [ ] 2.8 Run the local safe test set; the four tests backend_identity, elastic_backend_identity, hw_compat_group, and export MUST pass

- [x] 2.9 Add the `reward-function` evidence kind and choose the kind from the island's reward function (design.md implementation record item 9)

## 3. Credential guard (per D1)

- This section is not done in this change: D1 decided to move the credential guard out of this change (2026-10-13)

## 4. Wrap-up

- [x] 4.1 Use grep to confirm that `TBENCH_REWARD_HMAC_KEY`, `SECRLENV_REWARD_HMAC_KEY`, and `trusted_outcome_hmac` have 0 hits in yeto/
- [x] 4.2 Use grep to confirm that `YETO_ISLAND_HMAC_KEY` and seal/open in elastic.rs are unchanged
- [ ] 4.3 Tell rl-code-quality-cleanup phase 4: the evidence schema is now 3, and the trust-path merge builds on it
