# Design

## Context

Sources: `infra-drafts/SESSION24/DISCUSS.md` D1 (decided 2026-10-14) and `infra-drafts/SESSION24/identity-check-explore.md`. Line references marked "explore" come from that file and are not checked again here. Other line references are checked on worktree `s23-g3-fp`.

Current layers (explore §1): L1 source tree hash (`yeto/provenance.py:120-132`), L2 plugin source hash (`algorithm.py:255-258`), L3 `AlgorithmSpec.sha256()` (`algorithm.py:1269-1275`, includes L2), L4 execution contract, L5 island contract (`backend_identity.py:118`), L6 backend identity (includes the Miles commit), L7 runtime fingerprint. Line numbers are from explore.

## Goals

- Reject only differences that change numerics.
- Pin the code of a run, so that every island and every resume runs the same code.
- Remove golden re-records and migration entries for pure code changes.

## Non-Goals

- Policy version tags (`yeto:<v>:<hash>`) are weight identity, not code identity. They do not change.
- L4 execution contract stays as it is.
- Release image build (yeto in an image layer) is a later task in this change, after development mode works.

## Decisions

### D1. Semantic hash
`AlgorithmSpec.semantic_sha256()` hashes the canonical JSON of: hyperparameters, correction configuration, advantage, loss, filter names, reference model, learning-rate schedule hash, and backend engine name + card type (`compat_group`) + parameter name mapping hash. Plugin entries keep the path and drop `sha256`. The engine commit is not in it. A mismatch is a hard reject at the handshake and at cut resume.

### D2. Run code pack
- Development mode: `git rev-parse HEAD` + `git diff` of tracked changes + all untracked, non-ignored files → tar → sha256. The launcher uploads it to the object storage of the run (S3 bucket, Modal Volume, or nebius-fs; these already hold checkpoints, explore/DISCUSS D1). It is kept at least until the run ends.
- Release mode: the image digest is the digest. No upload.
- The launcher makes the pack once, at the first start of the run. It writes `code_pack.digest`, `code_pack.uri`, and `code_pack.mode` into the run manifest (`_write_run_manifest`, `yeto/launcher.py:8183`).
- Island setup downloads the pack, unpacks it as the workdir, computes the digest again, and stops on a mismatch.
- Replacement, re-JOIN, and cut resume read the manifest and use the same pack. The local tree of the submitter is not used.

### D3. Handshake
HELLO and JOIN carry the pack digest. The syncer rejects a different digest by default. In normal use a mismatch does not happen; a mismatch means a path went around the launcher. L1 checks change to a pack digest check.

### D4. Explicit upgrade
An operator can change the pack of a running run only by an explicit upgrade command. The launcher allows it only if the semantic hash of the new pack equals the old one. The manifest appends the record "pack A → pack B" with time and reason.

### D5. Golden files and hash-migration
Golden files drop `plugins[*].source_sha256`, `ports_runtime_fingerprint`, and `engine_commit` in `backend_identity`. They keep the Miles argv digest. `hash-migration.md` gets an entry only when a semantic hash changes.

### D6. Version boundary
The island contract switches from L3 to the semantic hash, with domain `yeto-rl-island-contract-v3`. The pack digest goes into the handshake. These two changes ship at the same version boundary as L6b of `rl-offpolicy-correction-passthrough`. Old and new islands cannot run together after it, as with earlier pin updates.

## Migration

1. Dual write: islands write `code_pack_digest` and `semantic_sha256` into run metadata and `rl_driver_start`. Old fields stay. Now.
2. `PluginRef` accepts no `sha256`. A present and different `sha256` gives a warning, not a reject. Now.
3. Code pack, handshake digest, and semantic hash in the island contract v3. At the L6b version boundary.
4. Cut resume compatibility: `verify_cut` compares `semantic_sha256`; for an old cut without it, recompute it from the spec in the old manifest. This is inferred from explore §6; `cut.py` is not read yet. Old elastic checkpoints stay readable and use the old comparison rule.
5. Golden simplification: change the generator, re-record once, write one final `hash-migration.md` entry.
6. Delete old paths (L2 hard reject, L7 golden fields) after two production runs pass under the new rules.

## Risks and open items

- "Same config, different math" code changes are not caught by the semantic hash. The strict pack digest catches them inside one run. Between runs, the code review and the golden tape catch them.
- Unverified: the actual code fetch paths in `yeto/cloud/replace.py` and `yeto/rl/elastic_client.py`.
- Unverified: whether `cut.py` keeps enough of the old spec to recompute `semantic_sha256`.
- Pack size and upload time for a work tree with large untracked files are not measured. The pack honors `.gitignore`.
