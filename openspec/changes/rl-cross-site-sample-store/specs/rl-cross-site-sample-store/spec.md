# Spec Delta

## Purpose

Let sampler islands and a trainer at different sites share the durable sample buffer and policy releases through an object store (Alibaba Cloud OSS) that every site reaches.

## ADDED Requirements

### Requirement: Object store backend
The system SHALL provide an OSS store implementation of the checkpoint store interface, selected through the store registry. Buffer, release, and resume code SHALL NOT import it directly. Credentials SHALL come from environment variables or a referenced file, never from tracked files.

#### Scenario: OSS store selected by configuration
- **WHEN** the cut store is `oss://bucket/prefix` and the launch layer exports the OSS store implementation
- **THEN** the buffer and the releases are written through that implementation, and no core module branches on the backend

### Requirement: Sharded buffer
Each sampler island SHALL write only to its own buffer shard. The trainer SHALL read all shards and SHALL record a per-shard consumed position in the cut record.

#### Scenario: Two samplers at two sites
- **WHEN** an NPU sampler on the local cluster and a PPU sampler write to the same buffer
- **THEN** each writes its own shard, and the trainer's cut record contains one position per shard

#### Scenario: Restore after crash
- **WHEN** the trainer restarts from a cut
- **THEN** every shard resumes at the position stored in that cut, and samples after it are consumed again

### Requirement: Complete releases only
A sampler SHALL load a release only if every file named in its `RELEASE.json` exists with the stated size.

#### Scenario: Reader during upload
- **WHEN** a sampler reads `LATEST.json` while the next release is still uploading
- **THEN** it sees the previous release, or the new release with all its files

### Requirement: Cut store reachability
At launch, the trainer scheduler SHALL check that the trainer and every sampler reach the configured cut store, and SHALL refuse the launch if one does not. It SHALL NOT require a Modal volume.

#### Scenario: OSS with an NPU trainer and a PPU sampler
- **WHEN** the cut store is `oss://…`, the trainer runs on the local NPU cluster, and a sampler runs on a PPU machine
- **THEN** the launch passes the check only if both sites reach the store
