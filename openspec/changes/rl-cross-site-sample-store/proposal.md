# Proposal

## Why

The S24 sampler/trainer hand-off assumes that both islands mount one store: `yeto/rl/engine/policy_release.py` puts releases in "the store both islands already mount", and `yeto/rl/engine/sample_buffer.py` (`FileSampleBuffer`) requires concurrent appenders to share one host (`fcntl.flock` on the index). The trainer scheduler also requires a `modal-volume://` cut store (`yeto/launch/trainer_loop.py:181-182`).

In the S25 fleet no island runs on Modal. The trainer runs on the local Ascend cluster (behind a VPN) or on Alibaba PPU machines, and samplers run on the other site. No file system is mounted on both sites. The user chose Alibaba Cloud OSS as the cross-site store. OSS has an S3-compatible API.

yeto already has the parts to reuse: the `CheckpointStore` interface with `commit`/`reload`, and a registry that selects a store implementation through `YETO_RL_STORE_IMPL` (`yeto/rl/engine/resume.py:119-178`). The Modal store (`yeto/cloud/modal_ckpt_store.py`, `ModalVolumeStore`) is one implementation. The sample-pool index draft (`yeto/rl/engine/sample_pool.py`) already lists `s3://` URIs. This change adds an OSS implementation in the same place and makes the buffer safe for writers on several hosts.

## What Changes

- `OssStore`: a `CheckpointStore` implementation for OSS through the S3-compatible API, registered under the existing store registry. `commit` uploads what the island wrote under its local staging root; `reload` downloads what other writers committed. Core code does not branch on the backend.
- Sharded buffer: each sampler island appends only to its own shard (`sample-buffer/shards/<island>/`). Index lines become numbered objects. The trainer reads all shards in a fixed order. The consumed watermark becomes a per-shard position vector in the cut record.
- Release commit on object stores: weights first, then `RELEASE.json`, then `LATEST.json`. A reader loads a release only if every file named in `RELEASE.json` exists with the stated size. This applies to BF16 releases and to the W4A8 releases of change `rl-hetero-sampler-engines`.
- The trainer scheduler accepts any cut store that the trainer and every sampler reach, not only `modal-volume://`.
- Same-site runs (stage 1: NPU trainer and NPU sampler on the local cluster) may use a shared directory instead of OSS. The store URI selects the backend.

## Non-goals

- No peer-to-peer weight transfer between islands.
- No change to the syncer protocol.
- No other cloud object store in this change (the interface allows one later).
- No cost model for OSS traffic.
- No credentials in files under version control. Access keys come from environment variables or a file path named in configuration.

## Capabilities

### New Capabilities
- `rl-cross-site-sample-store`: sharded durable buffer and policy releases on an object store (OSS) that every site reaches.

### Modified Capabilities
- None in `openspec/specs/`. It changes the single-host limit stated in the draft `rl-trainer-lifecycle` capability.

## Impact

- Code: new `yeto/cloud/oss_store.py`; `yeto/rl/engine/sample_buffer.py`, `yeto/rl/engine/policy_release.py`, `yeto/rl/engine/resume.py` (registry entry only), `yeto/launch/trainer_loop.py`, `yeto/launch/storage.py`.
- Operations: one OSS bucket; access keys from the user.
- Risk: one W4A8 release of GLM5.3-Flash is about 301 GB (size of the `checkpoint/` directory that vllm-ascend loads); one BF16 release is 598.5 GiB. Transfer time between the local cluster, PPU machines, and OSS is not measured.
