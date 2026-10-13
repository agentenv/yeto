# Design

## Code-reading check (2026-10-13, read-only)

| P2 conclusion | Check result |
|---|---|
| tbench_outcome.py has sign/build_signed/verify/validate_hmac_key_source | Holds. The file has 274 lines. |
| reward.py has OUTCOME_KEY/MAC_KEY/_hmac_key | Holds. `_hmac_key` reads `SECRLENV_REWARD_HMAC_KEY` and falls back to `SECRLENV_BEARER_TOKEN`. |
| trajectory_evidence.py mixes the mac into the evidence hash; note `_LEGACY_SCHEMA` | Holds, but incomplete: schema 1 and schema 2 **both** mix in the mac. So a new schema 3 is needed. Changing only legacy is not enough. |
| island_credential_guard.py has 70 lines and 4 call sites | **Overturned**: the 70 lines hold. There are 8 call sites, in `learner.py`, `megatron/learner.py`, `diffusion/learner.py`, miles `island_entry.py`, verl `island_entry.py`, `verl_main.py` (2 sites), and `fully_async_runner.py`. It checks cloud credentials, not rewards. |
| test_secret_handling_hardening.py can be deleted whole (only the guard part if Q3 is kept) | **First half overturned**: Q3 is decided to keep it. So only the 7 guard-related tests can be deleted (near lines 235-271 and 346). The other tests cover HELLO HMAC, the pinned contract, and key routing. They must stay. |
| Elastic mode signs every frame (elastic.rs:146-167) | Holds. `seal`/`open` are in that line range. |
| backend_identity fields and session_contract_hash | Holds. `session_contract_hash = sha256(domain ‖ layout_fingerprint ‖ identity_sha256)`. `lr_schedule_sha256` and `ISLAND_CONTRACT_DOMAIN` also exist. P2 did not list them. They stay too. |
| episode_id is in rlf/contracts.py | Holds (line 37, uuid4 default). |
| Removing the signature clears the Harbor obstacle | **Unverified**. `yeto/` and `tests/` have no Harbor reference. |

## Decisions

1. **Remove only the signature and the key. Keep the structure checks.** The field set, status enum, and reward range check in `_validated` stay. Reason: they guard against code errors, not attacks.
2. **Raise the evidence schema to 3.** The schema 3 outcome hash is `hash(domain, canonical_outcome)` and has no mac. Writing uses only 3. Reading 1 and 2 follows D2.
3. **Metadata key names.** Keep `OUTCOME_KEY` unchanged, to avoid affecting replay tools. Delete `MAC_KEY`. When reading, ignore an old mac field.
4. **Key environment variables.** Remove them from the pass-through list and from the preflight check. Do not add a check that fails when they are set.
5. **Do not touch the island HMAC.** The HMAC in `syncer/src/elastic.rs`, `main.rs`, `server.rs` and in the Python files `elastic_client.py`, `protocol.py` is out of scope for this change.
6. **Order of work.** Step 1: remove the reward signature and raise the evidence schema. Step 2: handle the credential guard according to the D1 result (recommended: move it out of this change).

## Risks

- A replay tool may deduplicate or compare by mac. After the schema raise, it would treat old and new evidence as different. Mitigation: replay comparison compares only `reward_contract_hash` and the canonical outcome. Whether any tool compares by mac is **unverified**. During implementation, check `scripts/` and `tools/` with grep.
- Running in parallel with rl-code-quality-cleanup phase 4 may cause conflicts. Mitigation: merge this change first.
