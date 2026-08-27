# Learner ↔ Syncer wire protocol (v4)

Transport is a connection group of `1 + S` persistent TCP sockets per
learner: stream 0 carries control traffic and streams 1..S carry striped bulk
traffic. All integers are little-endian. Tensors are contiguous bytes in the
session dtype and fragments concatenate tensors in HELLO layout order.

The syncer owns global step `t` and drives a pull-based, per-fragment schedule;
learners continue local optimization while network work proceeds.

## Framing and allocation limits

```text
frame := magic:u32 (0xD170C0DE) | type:u8 | len:u64 | payload[len]
```

Both peers validate the type and length before allocating or reading the
payload:

- the first frame must be HELLO or DATA_HELLO and is capped as layout metadata;
- PULL, HEARTBEAT, BUDGET_DONE, FINAL_MANIFEST, FINAL_ACK, SHUTDOWN, and ERROR
  have exact or small negotiated limits;
- each CHUNK is limited to its 24-byte envelope plus one 4 MiB slice;
- INIT, PUSH, BCAST, FINAL_FRAGMENT, and reassembled-inner limits derive from
  the negotiated fragment numel and dtype.

Fragment IDs and exact decoded tensor lengths are checked again before work is
accepted. Oversized/truncated/overlapping chunks, invalid IDs, and arithmetic
or allocation overflows are errors rather than panics.

## Message types

| type | name | direction | payload |
|---:|---|---|---|
| 1 | HELLO | learner → syncer | protocol_version:u16 (=4), learner_id:u32, connection_generation:u64, dtype:u8 (1=f32, 2=bf16, 3=q4), num_fragments:u32, per-fragment layout, layout_fingerprint:[u8;32], session_contract_hash:[u8;32], optional syncer_profile_hash:[u8;32], num_streams:u16 |
| 2 | INIT_PARAMS | learner → syncer | fragment_id:u32, full tensor bytes; only learner 0 may initialize |
| 3 | PULL_REQ | syncer → learner | fragment_id:u32, global_step:u64, round_attempt:u32 |
| 4 | PUSH_FRAGMENT | learner → syncer | learner_id:u32, fragment_id:u32, global_step:u64, round_attempt:u32, base_version:u64, local_step:u64, c_steps:u32, c_tokens:u64, base-relative learner-delta bytes |
| 5 | BCAST_FRAGMENT | syncer → learner | fragment_id:u32, version:u64, full global tensor bytes |
| 6 | HEARTBEAT | learner → syncer | learner_id:u32, local_step:u64 |
| 7 | SHUTDOWN | syncer → learner | empty; sent only to a generation whose final ACK was accepted |
| 8 | DATA_HELLO | learner → syncer | protocol_version:u16 (=4), learner_id:u32, connection_generation:u64, stream_idx:u16 |
| 9 | CHUNK | either | msg_id:u64, total_len:u64, offset:u64, bytes |
| 10 | ERROR | syncer → learner | UTF-8 fatal protocol/session error |
| 11 | FINAL_MANIFEST | syncer → learner | revision:u16 (=1), final_global_step:u64, num_fragments:u32, expected_version:u64 × num_fragments |
| 12 | FINAL_ACK | learner → syncer | revision:u16 (=1), final_global_step:u64 |
| 13 | FINAL_FRAGMENT | syncer → learner | fragment_id:u32, version:u64, authoritative full tensor bytes, always little-endian f32 |
| 14 | BUDGET_DONE | learner → syncer | exact_local_steps:u64 |

HELLO fragment layout is, for every fragment: `merge_mode:u8` (0=avg, 1=rda,
2=iso), `num_tensors:u32`, then `numel:u64 × num_tensors`; iso additionally
appends `(rows:u64, cols:u64)` for every tensor.

`layout_fingerprint` is SHA-256 over a canonical encoding of every fragment's
merge mode and every tensor's fully qualified name, order, numel, and full
shape. The numeric layout drives decoding; the fingerprint prevents two
learners with equal-sized but semantically reordered tensors from entering
the same session.

`session_contract_hash` binds reconnects and checkpoint recovery to one
client experiment. A fresh run must use a fresh contract; reusing a contract
means that checkpoint continuation is intentional. Profile-bound clients
additionally append
`syncer_profile_hash`, the canonical identity of the server's parsed
merge/schedule/recovery configuration. The server derives it from its actual
configuration, excluding only the listener port and concrete output paths,
and rejects a mismatch. `--require-profile-binding` also rejects an omitted
profile hash; leaving it disabled preserves generic protocol-v4 clients.

The version is checked before a connection can enter a session. A mismatch
returns ERROR and closes the connection; an older payload is never guessed or
silently decoded as v4. The first accepted HELLO fixes the session dtype,
layout, semantic fingerprint, session contract, and optional server profile
binding. Every later HELLO must match them exactly,
and learner IDs must lie in the configured `0..M` launch set. The learner ID
repeated inside PUSH and HEARTBEAT must match the connected group.

## Striping

HELLO declares `num_streams`; each additional socket attaches with a
versioned DATA_HELLO carrying the same learner ID and connection generation.
Small control frames, including BUDGET_DONE, FINAL_MANIFEST, and FINAL_ACK, use
stream 0.
INIT, PUSH, BCAST, and FINAL_FRAGMENT are serialized as normal inner frames,
sliced into 4 MiB chunks, and sent round-robin over data streams. With no data
streams they use stream 0 directly. A message never straddles connection
generations.

## Base-relative learner deltas

For every wire dtype, a learner retains the exact raw global fragment from its
last accepted BCAST, before optional alpha blending:

```text
learner_delta d_m,p = local θ_m,p − raw_anchor_m,p
outer_gradient Δ_m,p = −d_m,p = raw_anchor_m,p − local θ_m,p
```

The learner sends `d_m,p`; the syncer negates it exactly once at decode time.
Avg, RDA, iso, directional correction, and Nesterov consume signed outer
gradients directly. The coordinator never reconstructs a learner parameter by
adding a delta to its current state.

This matters under staleness. If the learner's raw anchor is one or more
versions old, its payload still contains only local progress since that exact
anchor. Intervening global drift cannot leak into the outer gradient.
`base_version` remains available for staleness telemetry and policy. A future
base version is invalid.

PULL can overtake the initial striped BCAST, so learners defer a response until
that fragment has a raw anchor. Subsequent broadcasts are accepted
monotonically per fragment: lower versions are dropped; an identical
equal-version replay is idempotent; different bytes for the same version are a
fatal protocol error. This prevents out-of-order stripes from regressing the
anchor or base version.

## Round membership, quorum, and generations

At launch, a round captures an immutable set of
`(learner_id, connection_generation)` members and
`K = min(--quorum, captured_members)`. Only that exact generation may answer.
A learner joining or reconnecting mid-attempt neither counts toward nor blocks
it. Disconnecting does not erase an already accepted response.

A new generation for a duplicate logical learner ID supersedes the old one for
future launches. The old generation may still finish an attempt that captured
it. Its eventual disconnect cannot remove the replacement generation from the
registry. Duplicate, wrong-generation, future-base, and out-of-round PUSHes
are rejected deterministically; the first valid PUSH per member wins.

The syncer must reach frozen K before merging. After quorum it applies the
adaptive grace window. If the quorum deadline expires below K, all partial
responses are discarded and a new attempt captures the then-current member
set. `round_attempt` is echoed in PUSH, so a delayed response from a discarded
attempt cannot enter its retry.

Each client redial creates a fresh nonzero connection generation and repeats
HELLO/DATA_HELLO. The syncer sends every initialized fragment at its current
version to a valid new generation.

## Merge and broadcast

Learner weight is `w_m = c_tokens² / c_steps`. Avg computes the weighted mean
of signed outer gradients. RDA computes weighted radial/directional averaging
per tensor. Iso direct-averages a matrix gradient and flattens its singular
spectrum. Optional directional correction operates on each signed outer
gradient against the fragment momentum.

The outer step is f32 Nesterov:

```text
buf ← μ·buf + Δ
Θ   ← Θ − lr·(Δ + μ·buf)
```

For a fresh single-learner response, this is equivalent to the previous
full-parameter behavior because `Δ = anchor − local`; the stale case is now
well-defined instead of incorporating coordinator drift.

BCAST carries the updated full fragment. The learner applies
`θ ← α·θ_local + (1−α)·Θ`, but records the unblended raw Θ as its next anchor.

The coordinator is one sequential scheduler actor. Pipelined rounds target
distinct fragments, and merge + version update + broadcast for a completion
is serialized. No two completions mutate one fragment concurrently.

## Q4 pushes

Q4 uses the same learner-delta semantics. INIT and BCAST stay bf16; PUSH is
block-quantized E3M0. Values are grouped in blocks of 256: each block contains
an f32 absmax scale and 128 packed-nibble bytes. A nibble has one sign bit and
a three-bit level; level 0 is zero and level `L∈1..7` decodes to
`sign · 2^(L−7) · scale`.

The syncer decodes and negates Q4 exactly like f32/bf16. A stale Q4 response
does not require a historical global snapshot and is admitted under the same
base-version policy.

## Authoritative finalization

After every launched round completes, the coordinator freezes the logical
learner IDs represented by live current groups. It writes the final checkpoint
and optional final-state file, then sends every authoritative fragment through
FINAL_FRAGMENT followed by FINAL_MANIFEST. FINAL_FRAGMENT is always f32,
independent of the ordinary session dtype, and uses a monotonic cache separate
from BCAST. A bf16 or q4 BCAST at the same fragment/version therefore cannot
satisfy or conflict with terminal delivery.

Control and striped data streams may reorder. A learner waits until it holds
the manifest and the exact f32 FINAL_FRAGMENT version for every fragment,
overwrites the trainable parameters with normal alpha blending disabled, and
only then sends FINAL_ACK. Applying to a lower-precision destination parameter
may cast to that parameter's storage dtype; the coordinator checkpoint remains
the exact f32 source of truth.

Final membership is generation-aware. An ACK is valid only from a connection
generation that the coordinator successfully queued the frozen cut to. A
pending logical learner may reconnect, receive the full cut on its new current
generation, and ACK from that generation. Once a valid ACK is accepted it is
never erased by a later disconnect. SHUTDOWN is sent only to still-connected
generations whose ACK was accepted. A missing cut or ACK fails after a bounded
wait rather than falling back to a locally blended save.

FINAL_MANIFEST and FINAL_ACK carry `FINALIZATION_REVISION=1`; mismatches are
fatal. A learner also rejects SHUTDOWN before it has applied and initiated the
ACK for the exact manifest.

## Learner-budget cutoff and consolidation restart

`BUDGET_DONE` is benchmark-only and is accepted only when the syncer and every
learner receive the same explicit positive learner-budget target. A trailing
byte, wrong step count, or duplicate report is fatal; learner identity comes
from the connection group.

The benchmark closes in two syncer processes on the same endpoint:

1. In the cutoff process, the first accepted `BUDGET_DONE` closes ordinary
   issuance and cancels all in-flight rounds as whole rounds. After exactly
   one report from every configured learner, the syncer writes an unmarked
   recovery checkpoint and exits. It sends no final manifest or shutdown.
2. The harness reads that checkpoint's global step `C`, restarts the syncer
   from it, and requests exactly `F` ordinary rounds, where `F` is the fragment
   count. Quorum is the full configured learner set. Gather/Iso work may be
   pipelined, but coordinator commits remain strictly ordered by round `t`.

Learners stop optimizer and data work before sending `BUDGET_DONE`, retain
their frozen trainable parameters, and reconnect to the restarted syncer. The
fresh connection drops queued pulls and broadcasts from the cutoff process.
The resumed syncer broadcasts its checkpoint state normally. Across the next
`F` round-robin pulls, each frozen learner answers every fragment with the
existing `PUSH_FRAGMENT` frame. Its payload is
`frozen_local - resumed_global`, with `local_step` and `c_steps` equal to the
budget and `c_tokens` equal to its exact raw unit count.

These are ordinary scheduler rounds: there is no terminal message or terminal
scheduler branch. The existing merge, correction, outer optimizer, checkpoint,
final-manifest, and acknowledgement paths are reused unchanged. Only normal
completion of all `F` rounds can produce a marked final checkpoint. Before
export, the harness also requires the event tape to show every configured
learner in every one of these rounds.

## Snapshots and event tape

The sequential coordinator mutates state only when committing the next
contiguous round `t`. Other rounds may still be gathering or running the Iso
polar iteration, but their owned buffers cannot touch coordinator state. A checkpoint at
that cut is therefore consistent. Snapshots contain global/per-fragment
versions, f32 params, momentum, and the cumulative learner ledger. Cutoff and
terminal checkpoints explicitly drain the Iso worker pool first; a poisoned worker
prevents checkpoint publication. The Iso worker pool bounds startup, every
full-matrix request, and drain. Any deadline expiry kills the directly owned
Python child, installs a persistent pool poison, discards queued work, and
makes drain/finalization fail. Pool capacity covers the complete admitted
queued-plus-running job lifetime; it is not released at dequeue.

New snapshots use checkpoint V3. Its fixed prefix is:

```text
magic:u32 = 0xD1705A80
iso_backend:u8                 # 0=scalar, 1=torch-svd
semantic_layout_fingerprint:32 # exact accepted HELLO SHA-256
global_step:u64
```

The fingerprint covers fragment order, merge modes, ordered tensor names,
lengths, and shapes. Resume compares both backend and fingerprint before
installing any checkpoint state. Fragment count and flat `numel` equality are
not sufficient: two grouped layouts can have identical flat sizes while
assigning those values and momentum slots to different tensors.

Legacy behavior is deliberately conservative:

- V1 (`0xD1705A7E`) has neither backend nor fingerprint. Python readers can
  inspect/export it, but the syncer never resumes it because an old Torch-SVD
  checkpoint cannot be distinguished from a scalar checkpoint.
- V2 (`0xD1705A7F`) records the backend but not the fingerprint. Python readers
  can inspect/export it. The syncer resumes it only when the backend matches
  and every fragment contains exactly one tensor. Grouped V2 resume fails
  closed.
- V3 records both and is the only format written by current syncers. A
  fingerprint mismatch is rejected even when fragment counts and every flat
  fragment size match.

Legacy V1/V2 export still depends on rebuilding the exact historical layout
from the original model and fragmentation flags; the file cannot prove that
identity. A grouped legacy checkpoint requires an externally audited
migration that supplies its original semantic fingerprint. V1 additionally
requires authoritative knowledge of its original ISO backend. Neither value
is guessed automatically.

Pipelined rounds may finish gathering out of order, but the coordinator merges,
checkpoints, and broadcasts them strictly in ascending global-step order. A
checkpoint is therefore always a contiguous prefix, and resume begins at the
first uncommitted step rather than skipping an older in-flight round. With
`--checkpoint-every 1`, each checkpoint is durable before its matching
non-terminal broadcast becomes externally visible.

Periodic checkpoints remain controlled by `--checkpoint-every`, but the
coordinator always rewrites `--checkpoint-path` at the final quiescent cut
before terminal delivery. Thus a total-step count not divisible by the
periodic interval still leaves an atomic final checkpoint.

When `--mark-final-checkpoint` is explicitly enabled, the syncer creates
`<checkpoint-path>.final` only after that terminal checkpoint write. The
marker contains `YETO_FINAL_V1` and the checkpoint `global_step`; periodic,
cutoff-incomplete, timeout, and error checkpoints remain unmarked recovery
state. The marker is adjacent metadata and does not change the checkpoint or
wire format. The normal `yeto launch` syncer enables this marker without
changing its learner-owned artifact path.

Every merge appends event-tape fields for protocol/delta semantics, attempt,
launch base version, frozen members and generations, responders, staleness,
weights, counters, quorum/grace timings, and sync latency.
