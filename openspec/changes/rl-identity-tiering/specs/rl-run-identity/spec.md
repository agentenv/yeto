# Spec Delta

## Purpose

Reject only identity differences that change numerics, and pin the code of a run to one code pack.

## ADDED Requirements

### Requirement: Semantic hash hard reject
`AlgorithmSpec` SHALL provide `semantic_sha256()` over hyperparameters, correction configuration, learning-rate schedule, and backend engine name, card type, and parameter name mapping. It SHALL NOT include plugin source hashes or the engine commit. The syncer and cut resume SHALL reject a semantic hash mismatch.

#### Scenario: Plugin code change only
- **WHEN** a plugin file changes and the spec configuration does not change
- **THEN** `semantic_sha256()` does not change

#### Scenario: Correction configuration differs
- **WHEN** two islands have different correction configurations
- **THEN** the syncer rejects the second island

### Requirement: Run code pack
At the first start of a run, the launcher SHALL make one code pack and SHALL write its digest and location into the run manifest. In development mode the pack SHALL contain the git commit and a patch of all uncommitted changes, including untracked files, and SHALL be kept in object storage at least until the run ends. In release mode the image digest SHALL be the pack digest.

#### Scenario: Replacement island
- **WHEN** an island is replaced after the local work tree of the submitter changed
- **THEN** the new island runs the pack from the run manifest, and its digest equals the manifest digest

#### Scenario: Cut resume
- **WHEN** a run resumes from a cut
- **THEN** the islands use the pack named in the run manifest

### Requirement: Strict pack digest handshake
HELLO and JOIN SHALL carry the code pack digest. The syncer SHALL reject a digest that differs from the run digest, by default.

#### Scenario: Island outside the launcher
- **WHEN** an island joins with a pack digest that differs from the run digest
- **THEN** the syncer rejects the JOIN and names both digests

### Requirement: Explicit code upgrade
A run SHALL change its code pack only through an explicit upgrade. The upgrade SHALL be allowed only when the semantic hash of the new pack equals the current one. The run manifest SHALL record the old and the new digest.

#### Scenario: Upgrade that changes numerics
- **WHEN** an operator upgrades to a pack whose semantic hash differs
- **THEN** the upgrade fails and the manifest does not change

### Requirement: Golden files without code hashes
Golden files SHALL NOT record plugin source hashes, the runtime fingerprint, or the engine commit. A `hash-migration.md` entry SHALL be required only when a semantic hash changes.

#### Scenario: Plugin change
- **WHEN** a plugin file changes and the semantic hash does not change
- **THEN** the golden test passes without a re-record

### Requirement: Old specs and old cuts stay usable
`PluginRef` SHALL accept a plugin entry without `sha256`, and SHALL warn, not reject, when a given `sha256` differs. Cut resume SHALL accept a cut written before this change when the recomputed semantic hash matches.

#### Scenario: Old spec file with stale plugin hash
- **WHEN** an old spec file has a plugin `sha256` that differs from the current file
- **THEN** the island starts and logs a warning
