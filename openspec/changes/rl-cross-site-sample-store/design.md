## Context

- Buffer: append-only `index.jsonl` plus payloads; payload written before its index line; torn last line ignored; one consumer; appenders on one host (`flock`).
- Watermark: kept in the cut record (`ConsistentCut.buffer_watermark`), restored with `reset_to`.
- Releases: `r<seq>-v<version>/` with `RELEASE.json`, then `LATEST.json` as the commit point.
- Store interface: `CheckpointStore(root)` with `commit()`, `reload()`, `describe()`, `list()`; `store_for(path)` builds the implementation named by `YETO_RL_STORE_IMPL` (`yeto/rl/engine/resume.py`). `ModalVolumeStore` is the model for a store that needs an explicit commit and reload.
- Sites: local Ascend cluster behind a VPN that routes only 172.28.11.0/24; PPU machines at Alibaba Cloud; the head on the submitting server. Public reachability of OSS from the cluster nodes is not verified.

## Decisions

### D1. OssStore follows the ModalVolumeStore pattern
Islands write files under a local staging root as today. `commit()` uploads new and changed files under that root to `oss://<bucket>/<prefix>/`; `reload()` downloads new objects. The implementation uses the S3-compatible API with the OSS endpoint: boto3 with `endpoint_url`, virtual-hosted addressing, SigV4, and checksums only where required (not tested against OSS). The store is always staged (no mount). Creates send `x-oss-forbid-overwrite`. It reads the endpoint, bucket, and credential source from the environment, like `store_from_env` in the Modal store. It is registered by name, so `sample_buffer`, `policy_release`, and `resume` do not import it.

### D2. One shard per sampler island
Object stores have no `flock` and no atomic append. Each sampler island owns one shard and is its only writer, so no lock is needed. Shard records are numbered objects (`sample-buffer/shards/<island>/<n>.json`); a reader lists them in order and stops at the first gap. The object store says whether shards are needed (`buffer_shards`: true for OSS, false for S3, which keeps one index with conditional creates).

Implemented: the trainer (the one consumer) is the only writer of the merged index `sample-buffer/index/<n>.json`. Each refresh appends the new shard records in `appended_at` order (then shard name) and keeps `shard` and `shard_pos` in each record. When every shard has ended, the trainer writes the end record.

### D3. Watermark as a position vector
The cut record stores `{shard: position}`. Restore resets every shard. A shard that is new after the cut starts at 0. Single-shard stores keep today's scalar watermark as a one-entry vector.

Implemented differently: the cut record keeps the scalar watermark, which is a position in the durable merged index. The per-shard vector follows from it (`FileSampleBuffer.shard_positions`), because each merged record names its shard and position. Restore resets to the scalar; the merge continues after the records already merged. Policy releases carry the trainer's per-shard read positions (`buffer_shards`), so each sampler caps its backlog on its own shard. Reason: the scalar watermark is used by the cut record, the Miles round cut and the trainer commands; a vector there would change all of them without new behavior.

### D4. Commit order and reader check
Writers upload weights and `RELEASE.json`, then overwrite `LATEST.json`. Readers verify that every file named in `RELEASE.json` exists with the stated size before loading. The reader check does not depend on the store's consistency model; the consistency of OSS overwrites is not verified here.

### D5. Store choice is configuration
The cut store URI selects the backend (`/abs/path`, `modal-volume://`, `oss://`). The trainer scheduler checks at launch that the trainer and every sampler reach the store, instead of requiring `modal-volume://`. Implemented as a static check by store kind and island cloud: Modal islands reach only `modal-volume://`; other islands reach `s3://` and `oss://`. The network check on the node is left to the node preflight (change `rl-ssh-node-pool`, N1.5). Stage 1 (one site) may use a shared directory on the cluster if the survey finds one.

### D6. Large releases
A W4A8 release is about 301 GB. Uploads use multipart transfer with parallel parts; the part size and parallelism are configuration (`YETO_RL_STORE_PART_MB`, default 64; `YETO_RL_STORE_PART_CONCURRENCY`, default 8; the part size grows so that one file needs at most 10,000 parts). Downloads use parallel ranged reads pinned to the object ETag. Only files that change between versions need upload; for full-parameter training this is nearly every file, so no saving is assumed.

### Non-goals and rejected alternatives
- Rejected: a lock service for multi-writer appends; shards are simpler and need no service.
- Rejected: mounting OSS with a FUSE file system (ossfs) as the main path; partial-write visibility and fsync behavior are not known. It stays a later option.
- Rejected: self-hosted MinIO (v1 recommendation); the user chose OSS.

## Risks
- Transfer time per release may be longer than one trainer round; samplers then lag more versions behind, and the version gate drops more groups.
- Listing cost and latency grow with the index object count; compaction is a later task.
- Cluster nodes may need a proxy to reach OSS; the head reachability preflight (change `rl-ssh-node-pool`, N1.5) checks it.
