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
- PULL, HEARTBEAT, FINAL_MANIFEST, FINAL_ACK, SHUTDOWN, and ERROR have exact
  or small negotiated limits;
- each CHUNK is limited to its 24-byte envelope plus one 4 MiB slice;
- INIT, PUSH, BCAST, FINAL_FRAGMENT, and reassembled-inner limits derive from
  the negotiated fragment numel and dtype.

Fragment IDs and exact decoded tensor lengths are checked again before work is
accepted. Oversized/truncated/overlapping chunks, invalid IDs, and arithmetic
or allocation overflows are errors rather than panics.

## Message types

| type | name | direction | payload |
|---:|---|---|---|
| 1 | HELLO | learner → syncer | protocol_version:u16 (=4), learner_id:u32, connection_generation:u64, dtype:u8 (1=f32, 2=bf16, 3=q4), num_fragments:u32, per-fragment layout, layout_fingerprint:[u8;32], num_streams:u16 |
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

HELLO fragment layout is, for every fragment: `merge_mode:u8` (0=avg, 1=rda,
2=iso), `num_tensors:u32`, then `numel:u64 × num_tensors`; iso additionally
appends `(rows:u64, cols:u64)` for every tensor.

`layout_fingerprint` is SHA-256 over a canonical encoding of every fragment's
merge mode and every tensor's fully qualified name, order, numel, and full
shape. The numeric layout drives decoding; the fingerprint prevents two
learners with equal-sized but semantically reordered tensors from entering
the same session.

The version is checked before a connection can enter a session. A mismatch
returns ERROR and closes the connection; an older payload is never guessed or
silently decoded as v4. The first accepted HELLO fixes the session dtype and
layout and semantic fingerprint. Every later HELLO must match them exactly,
and learner IDs must lie in the configured `0..M` launch set. The learner ID
repeated inside PUSH and HEARTBEAT must match the connected group.

## Striping

HELLO declares `num_streams`; each additional socket attaches with a
versioned DATA_HELLO carrying the same learner ID and connection generation.
Small control frames, including FINAL_MANIFEST and FINAL_ACK, use stream 0.
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

## Snapshots and event tape

The sequential coordinator mutates state only at serialized round completion,
so a checkpoint at that cut is consistent while other pipelined rounds are
still gathering. Snapshots contain global/per-fragment versions, f32 params,
momentum, and the cumulative learner ledger.

Periodic checkpoints remain controlled by `--checkpoint-every`, but the
coordinator always rewrites `--checkpoint-path` at the final quiescent cut
before terminal delivery. Thus a total-step count not divisible by the
periodic interval still leaves an atomic final checkpoint.

Every merge appends event-tape fields for protocol/delta semantics, attempt,
launch base version, frozen members and generations, responders, staleness,
weights, counters, quorum/grace timings, and sync latency.
