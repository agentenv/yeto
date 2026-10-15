# Proposal

## Why

Yeto checks identity with seven hash layers (`infra-drafts/SESSION24/identity-check-explore.md` §1). Two of them cost much work and have no record of stopping a real error:
- The plugin source hash (L2) is part of `AlgorithmSpec.sha256()` and is a hard reject at island start (`yeto/rl/engine/algorithm.py:301-309`, source: explore §1). A one-line plugin change makes old spec files unusable.
- Golden files record plugin source hashes. `hash-migration.md` has more than 10 entries from S17 to S23 that only say "plugin source hash changed" (explore §1, §2).

Also, the launcher does not pin the yeto code. Sky syncs the local work tree (`workdir=str(REPO_ROOT)`, `yeto/launcher.py:1244`). Modal mounts the local directory (`add_local_dir`, `yeto/modal_runner.py:907-919`). Both lines are checked on `s23-g3-fp`. A replacement island or a resume uses the current local tree of the submitter, not the code of the run.

The user decided the design on 2026-10-14 (`infra-drafts/SESSION24/DISCUSS.md` D1).

## What Changes

- **Semantic hash, hard reject.** New `AlgorithmSpec.semantic_sha256()`. It covers hyperparameters, correction configuration, learning-rate schedule, and backend engine name + card type + parameter name mapping. It does not include plugin source hashes. A mismatch is a hard reject.
- **Code identity = run code pack digest.** One digest per run replaces the plugin source hash (L2), the source tree hash (L1), and the Miles commit as code identity.
  - At the first start of a run, the launcher makes a code pack. Release mode: the image digest. Development mode: git commit + a patch of all uncommitted changes, including untracked files, packed and uploaded to object storage. The pack is kept at least until the run ends.
  - The digest and the pack location go into the run manifest (`_write_run_manifest`, `yeto/launcher.py:8183`).
  - Island replacement, re-JOIN, and cut resume always get the pack from the manifest. They do not use the local tree.
  - The handshake compares the pack digest. The default is strict.
  - A code change in a running run needs an explicit upgrade. The upgrade is allowed only when the semantic hash is the same. The manifest records "pack A → pack B".
- **Release mode.** A named version for production training. The image digest is the pack digest. This needs yeto inside an image layer. It is later work.
- **Golden files** record no code hash. `hash-migration.md` gets an entry only when the semantic hash changes.

## Capabilities

### New Capabilities
- `rl-run-identity`: semantic hash with hard reject, run code pack with manifest, strict handshake on the pack digest, explicit code upgrade, and golden files without code hashes.

### Modified Capabilities
<!-- None on main. The island contract domain changes to v3 at the same version boundary as L6b of rl-offpolicy-correction-passthrough; that change is not edited here. -->

## Impact

- `yeto/rl/engine/algorithm.py` (`AlgorithmSpec.semantic_sha256`, `PluginRef`), `yeto/rl/engine/backend_identity.py` (island contract v3), `yeto/rl/engine/cut.py` (`verify_cut`), `yeto/provenance.py` (L1), new `yeto/codepack.py` (name is a proposal).
- `yeto/launcher.py` (`_write_run_manifest`, sky workdir), `yeto/modal_runner.py` (workdir mount), `yeto/cloud/replace.py`, `yeto/rl/elastic_client.py` (JOIN field). Code paths in the last two are not verified.
- `syncer/src/elastic_server.rs` (handshake digest check).
- `tests/golden/decoupling/*.json`, `tests/decoupling_golden.py`.
- Depends on the S23 integration merge, as `rl-trainer-start-stop` and `rl-offpolicy-correction-passthrough` do.
