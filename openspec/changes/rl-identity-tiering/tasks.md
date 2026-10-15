# Tasks

Source: `infra-drafts/SESSION24/DISCUSS.md` D1. Effort is in sub-agent person-days and is an estimate. Estimates for I3 and I4 come from the D1 launcher study (read code, no GPU). Other estimates are new and not from a study. Total estimate: about 13–15.5 d, plus multi-cloud checks on real machines (not estimated).

## Prerequisite

- [x] P.1 The S23 integration branch is merged to agentenv main, and this change is rebased on it (same as `rl-trainer-start-stop` and `rl-offpolicy-correction-passthrough`). GPU: no.

## I1. Now: dual write and tolerant plugin refs

- [x] L9.1 Add `AlgorithmSpec.semantic_sha256()` (design D1). Write it, and the code identity field, into run metadata and `rl_driver_start`. Keep all old fields. Acceptance: CPU tests; a plugin-only change does not change the semantic hash; a correction change does. GPU: no. Estimate: 1–1.5 d (new estimate)
- [x] L9.2 `PluginRef` accepts no `sha256`; a different `sha256` gives a warning (migration step 2). Acceptance: CPU test with an old spec file with a stale hash. GPU: no. Estimate: 0.5 d (new estimate)

## I2. Read before the version boundary

- [x] L9.3 Read `yeto/rl/engine/cut.py` (`verify_cut`), `yeto/cloud/replace.py`, and `yeto/rl/elastic_client.py`. Write down where code is fetched and whether an old cut holds enough spec to recompute the semantic hash. Acceptance: findings written back into design.md "Risks and open items". GPU: no. Estimate: 0.5 d (new estimate)

## I3. Code pack (development mode)

- [x] L9.4 New `yeto/codepack.py`: commit + patch with untracked files → tar → sha256 → upload (design D2). Launcher makes it at first start and writes digest, URI, and mode into `_write_run_manifest` (`yeto/launcher.py:8183`). Island setup downloads, unpacks, and checks the digest. Acceptance: CPU tests with a fake object store; a changed local tree after start does not change the island code. GPU: no. Estimate: 3–4 d (study) (GPU verified 2026-10-15 with modal-volume storage, evidence: infra-drafts/SESSION24/gpu/L6.10-review.md; not verified: s3 backend, sky machines, `.skyignore`; `--rl-code-pack` default is off)
- [ ] L9.5 Replacement, re-JOIN, and cut resume use the manifest pack; HELLO/JOIN carry the digest; the syncer rejects a mismatch by default (design D3). Acceptance: real Rust syncer + fake islands; a mismatched JOIN is rejected; resume uses the manifest pack. GPU: no. Estimate: 1.5 d (study) (partial: manifest-level pack digest check at island start only; HELLO/JOIN frames unchanged; the real Rust syncer rejection of a mismatched JOIN moves to L9.7)
- [x] L9.6 Explicit upgrade with semantic hash check and manifest record (design D4). Acceptance: CPU tests for allowed and refused upgrades. GPU: no. Estimate: 1 d (new estimate)

## I4. Version boundary (with L6b of rl-offpolicy-correction-passthrough)

- [ ] L9.7 Island contract uses `semantic_sha256()`; domain `yeto-rl-island-contract-v3`; L1 checks become pack digest checks (design D6). Acceptance: CPU tests; real syncer rejects a v2 island. GPU: no. Estimate: 1–1.5 d (new estimate)
- [ ] L9.8 Cut resume compatibility: `verify_cut` uses the semantic hash and recomputes it for old cuts; old elastic checkpoints stay readable (migration step 4; depends on L9.3). Acceptance: CPU tests with a cut and an elastic checkpoint from before the change. GPU: no. Estimate: 1–1.5 d (new estimate)
- [ ] L9.9 GPU check: ride on the L6b / L6.10 GPU check; record the pack digest in every island and one cut resume with the manifest pack. GPU: yes, review before launch, no extra launch. Estimate: 0.5 d

## I5. Golden files and cleanup

- [ ] L9.10 Golden generator drops code hashes (design D5); re-record once; one final `hash-migration.md` entry. Acceptance: golden test passes; a plugin-only change needs no re-record. GPU: no. Estimate: 1 d (new estimate)
- [ ] L9.11 After two production runs pass under the new rules, delete the L2 hard reject and the L7 golden fields. Acceptance: CPU safety test set passes. GPU: no. Estimate: 0.5 d (new estimate)

## I6. Release mode (later)

- [ ] L9.12 Put yeto into an image layer; the image digest is the pack digest; launcher stops syncing the local tree in release mode (today: sky `workdir=str(REPO_ROOT)` at `yeto/launcher.py:1244`, Modal `add_local_dir` at `yeto/modal_runner.py:907-919`). Acceptance: one launch on each cloud uses the image code only. GPU: yes, review before launch. Estimate: 2 d (study) + cost not set
