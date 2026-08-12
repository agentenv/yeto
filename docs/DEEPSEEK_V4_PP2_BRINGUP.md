# DeepSeek V4 PP2 bring-up

## Status

Pipeline-parallel (`PP=2`) support for the DeepSeek V4 expert-full Miles RL
path is implemented on `fix/secrlenv-r64-pp2-support`. Component and pinned
multi-node validation have passed. Daemon-backed end-to-end acceptance is
still in progress, so this should not yet be described as generally
production-ready PP2 support.

The next acceptance gate is a fresh two-node, 16-GPU run on `h200-n4` and
`h200-n5`. The preceding v56 gate reached daemon-backed rollout, exposed a
missing task-image preflight, and was stopped with its state preserved. A
four-node DP2/full32 acceptance run remains gated on the fresh DP1 run.

This document records the implementation contract, validation evidence,
failure history, and operational handoff. It deliberately contains no
environment contents, secrets, prompts, samples, responses, sessions, or
payload-bearing logs.

## Supported bring-up topology

The minimum end-to-end gate uses:

| Dimension | Value |
| --- | --- |
| Nodes / GPUs | 2 H200 nodes, 8 GPUs per node, 16 GPUs total |
| Trainer parallelism | TP8 x PP2 x EP8 x DP1 |
| Rollout | SGLang TP8 x DP1, colocated through train offload |
| Transformer layers | 43, split 22 / 21 across the two PP stages |
| PP stage 0 | canonical layers 0-21 |
| PP stage 1 | canonical layers 22-42; physical local layer 0 maps to global layer 22 |
| Expert-full selection | 16 experts |
| Expected selected-expert source balance | `[4,4,4,4,0,0,0,0]` on each PP stage |
| Optimizer budget | one rollout/train cycle and checkpoint step 1 |

Four nodes with TP8 x PP2 produce DP2. That duplicates each pipeline stage and
adds replica collectives; it does not increase model-sharding capacity. For
that reason, DP2/full32 is treated as a scale acceptance test rather than the
minimum PP2 smoke.

## Implemented PP2 contract

### Private service routing

Ray, Miles, the router, the session server, and SGLang are explicitly bound to
`tailscale0`. This prevents workers or services from advertising unreachable
default/public addresses. The active two-node gate uses the nodes' private
Tailscale addresses only.

### Pipeline-local injection and canonical identity

Megatron names the first physical layer on a later PP stage as local layer 0,
while the HF/PEFT policy identity remains global. For PP greater than one,
adapter injection therefore uses pipeline-local wildcard targets before LoRA
construction. Export and apply retain strict canonical global names.

Attention export and apply:

- enumerate specifications in canonical order on every rank;
- map PP-local parameters to canonical HF names;
- keep rank 0's valid Bridge PP-broadcast copies and physical-owner copies;
- validate owner and replica metadata across the distributed world;
- require exact canonical name, shape, dtype, and layout coverage;
- retain the complete attention state only where required for publication.

### BF16 task Bridge versus FP8 rollout Bridge

The trainer loads the BF16 `ref_load` model layout, while rollout uses the FP8
`hf_checkpoint` layout. Those layouts have different expert key spaces. Bridge
conversion-task discovery must use the BF16 trainer layout; the FP8 Bridge is
still used for rollout quantization and post-processing.

Both actor-side state conversion and base-weight publication now use one
cached BF16 task Bridge for task discovery and export. This preserves the FP8
rollout boundary without silently dropping canonical expert tasks.

### FP32 optimizer-master semantics

Canonical trainable state is read from and applied to complete FP32 optimizer
masters under `no_grad`. After apply, masters are copied back to model
parameters. This matches the pinned Miles/Megatron optimizer contract and
avoids a BF16 round trip changing the strict policy hash.

### Owner-sharded expert export

Full16 contains exactly 2,064 expert tensors:

```text
43 layers x 16 experts x 3 matrices = 2,064 tensors
2,064 x 4096 x 2048 x 4 bytes = 69,256,347,648 bytes = 64.5 GiB
```

Each tensor is retained only by the minimum global rank among its validated
canonical owners. DP replicas still participate and must agree byte-for-byte,
but they do not return duplicates. The Ray train group merges the rank
fragments without cloning and rejects missing names, duplicate names, source
rank mismatches, layout disagreements, or peer-provided authoritative metrics.

For TP8 x PP2 x EP8 full16, the largest owner shards are:

- PP stage 0: 8,858,370,048 bytes (8.25 GiB);
- PP stage 1: 8,455,716,864 bytes (7.875 GiB).

### Transactional chunked apply

Applying the complete state uses this protocol:

```text
begin -> sequential Ray chunks (at most 992 MiB each) -> finish
```

Only one top-level Ray `ObjectRef` is live at a time. Rank 0 receives that one
resolved chunk, while every rank enters the same mapping collectives. Each
chunk is checked for canonical order, exact names and shapes, CPU FP32
contiguity, finite values, and exact byte counts.

`finish` performs master-to-model copy, scheduler alignment, optimizer-state
reset, actor backup, and the final barriers. The new policy version is not
published before those operations complete.

### Persistent JIT cache

Fresh runs may opt into `/data/yeto-rl/jit-cache`. The namespace is bound to
the pinned Docker, Miles, SGLang, H200, and driver identities. Only the narrow
DeepGEMM, TVM-FFI, Triton, FlashInfer, SGLang, CUDA, TileLang, CuPy,
CUTLASS-Python, and TorchInductor cache paths are mounted.

Do not mount broad `/data` or `/root/.cache`, overlap writers in the same host
namespace, or delete the cache automatically.

### SecrlEnv task-image preflight

The source-attested `TaskPack` is loaded and its SHA-256 is verified before
any task-image operation. Service image pins are deduplicated, conflicting
image IDs are rejected, and each existing or newly pulled image must match its
exact content ID. Registry-qualified digest references may be pulled; missing
raw image IDs fail without attempting a pull.

This preflight runs independently on every host before the episode daemon is
started and before any GPU setup. It checks that `/data` has at least 2 TiB
free before preflight and before and after every pull. Output is one aggregate
readiness record; it does not reveal task IDs, service names, or image refs.
Per-episode image verification remains in place as defense in depth.

## Validation evidence

The PP2 transport source tested by v56 is
`5aa2f2433ded4e58ae55bb959d22efd2f9188c5b814910c3d096b41fb19969ca`.

Automated validation completed:

- 28 expert-full runtime tests;
- 42 runtime and validator tests;
- 172 focused tests;
- 255 broader RL tests;
- 100 DeepSeek/learner tests with one expected skip;
- Ruff, Python compilation, and `git diff --check`.

A pinned two-node world16 validator used TP8 x PP2 x EP8, DDP, and the real
optimizer. Both containers exited naturally with code 0. It proved:

- exact attention mapping and FP32 master round trip;
- stage-local to canonical layer mapping;
- BF16 task-Bridge coverage;
- selected-expert balance `[4,4,4,4,0,0,0,0]` on both stages;
- an exact, one-copy union of 192 tensors in the four-layer fixture;
- the full16 64.5-GiB geometry and maximum owner-shard sizes.

A separate pinned Ray 2.56.1 probe proved that a top-level `ObjectRef` resolves
to a dictionary at actor entry and that sequential chunks have one active
chunk with no overlap.

The component validator does not prove checkpoint load, TMS lifecycle,
rollout, optimizer step, authoritative checkpoint, or natural end-to-end
finalization. Those remain acceptance milestones for a fresh DP1 run.

## Failure history and fixes

| Run | Finding | Resolution |
| --- | --- | --- |
| v46-v48 | Ray/Miles/router services advertised unreachable default/public addresses | Bind the complete service plane to `tailscale0` |
| v49 | Each PP stage independently expected the complete global attention-LoRA set | Add distributed canonical attention union |
| v50 | Later PP stage's first canonical tensor, layer 22, had no retained owner | Preserve rank 0's valid PP-broadcast copy and canonical owner metadata |
| v52 | Global concrete injection targets did not match later-stage physical local layer 0 | Use PP-local wildcard injection while preserving canonical global identity |
| v54 | Conversion tasks were discovered from the FP8 rollout layout instead of the BF16 trainer layout | Use `ref_load` for actor and base-publication task Bridges |
| v55 | A Ray rank vanished at the monolithic 64.5-GiB export/serialization boundary | Replace monolithic export/apply transport with owner shards and bounded sequential chunks |
| v56 | Daemon-backed rollout could not provision tasks because task images were absent on the hosts | Attest the task pack and prefetch every exact image identity before daemon or GPU startup |

The v55 worker ended with Ray `SYSTEM_ERROR`/EOF and no preserved numeric
signal. Docker OOM, Ray/host/cgroup memory pressure, GPU Xid, Python traceback,
and a core dump were not observed. The monolithic object boundary is the
evidenced fault surface; a specific signal or deeper native cause is not
claimed.

Failed run state is preserved and never reused.

## Staged acceptance

The progression is intentionally ordered:

1. Real two-GPU TP1 x PP2 mapping and FP32-master component validation — passed.
2. Pinned two-node world16 TP8 x PP2 x EP8 validator — passed.
3. Two-node full16 TP8 x PP2 x EP8 DP1 daemon-backed smoke (v56) — reached rollout, then stopped on missing task-image infrastructure.
4. Fresh two-node DP1 gate with fail-closed task-image preflight — pending.
5. Fresh four-node n4-n7 full32 TP8 x PP2 x EP8 DP2 acceptance — gated on step 4.

The DP1 gate succeeds only after all of the following are evidenced:

1. TMS backup/offload and restore;
2. initial canonical attention and owner-sharded expert export;
3. syncer `INIT_PARAMS`;
4. daemon-backed rollout;
5. transactional sequential policy apply;
6. one optimizer step;
7. authoritative checkpoint at step 1;
8. final cut and training complete;
9. natural exit 0 from both containers.

After success, use an officially prepared, freshly attested run ID for DP2.
Never restart or reuse failed run state.

## Latest gate handoff

As of 2026-08-12 03:35 UTC:

| Item | Value |
| --- | --- |
| Run ID | `dsv4-e288-safety32-full16-pp2-smoke-v56` |
| Hosts | `h200-n4`, `h200-n5` only |
| Plan attestation | `391e37321bf766ad7f9889870ca03dd3ca968637467bedc4968d85027f56d572` |
| Source attestation | `5aa2f2433ded4e58ae55bb959d22efd2f9188c5b814910c3d096b41fb19969ca` |
| JIT compatibility | `efeaa0f219199514953bebba61fc1fd6dff5ab4aba75c0b0933807650957dd97` |
| Monitor | `monitor-staged-pp2-validation-every-minute` (hourly schedule) |
| Last completed stage | model load, TMS lifecycle, PP2 restore, and entry into daemon-backed rollout |
| Current health | v56 stopped safely; containers, daemons, and syncer stopped; GPUs idle; state preserved |
| Next step | officially prepare a fresh DP1 run with the task-image preflight, then require rollout, apply, optimizer, checkpoint, final cut, and natural exits |

Operational state is stored under run-ID-specific SSH-harness, daemon, and TMS
roots. Preserve those roots. Do not collect until a safe disk check has passed.

Monitor only narrow operational metadata:

- both container and daemon states;
- Tailscale service addresses;
- PP 22/21 mapping and selected-expert source balance;
- attention export/apply and expert owner-shard/chunk milestones;
- JIT cache and TMS backup/restore milestones;
- `INIT_PARAMS`, rollout, optimizer, checkpoint, finalization, and syncer state;
- available memory, `/data` free space, GPU utilization/memory, and fatal counts.

Never inspect or publish environment files, secrets, session metadata, prompts,
samples, responses, episode/tool payloads, or payload-bearing logs. Do not use
the unfiltered harness status command for this run.

Stop safely if any host has:

- less than 300 GiB `MemAvailable`;
- Ray memory pressure;
- less than 2 TiB free on `/data`;
- a fatal/OOM/mapping/distributed/NCCL/transport error.

Specifically watch for recurrence of:

- v55 `SYSTEM_ERROR`/EOF at export;
- v54 `expert-full conversion tasks do not cover local parameters`;
- incomplete attention export or missing pipeline owner;
- public-address `ConnectTimeout`.

## Branch and commit note

The implementation, required run support, pinned runtime bundle, tests, and
this document live on `fix/secrlenv-r64-pp2-support`. Its history is based on
`origin/main`; it is intended for review and remains unmerged.
