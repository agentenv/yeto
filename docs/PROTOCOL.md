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
| 1 | HELLO | learner → syncer | protocol_version:u16 (=4), learner_id:u32, connection_generation:u64, dtype:u8 (1=f32, 2=bf16, 3=q4, 4=f32 with bf16 pushes, 5=f32 with int8 pushes; flag 0x80 = bf16 downlink deltas), num_fragments:u32, per-fragment layout, layout_fingerprint:[u8;32], session_contract_hash:[u8;32], optional syncer_profile_hash:[u8;32], num_streams:u16, optional compat_group trailer (`b"YCG1"`, u32 length, 1–64 bytes utf-8), optional extension trailer (`b"YEX1"`, flags:u8, syncer_epoch:u64; see "Syncer epoch and peer recovery") |
| 2 | INIT_PARAMS | learner → syncer | fragment_id:u32, full tensor bytes; only learner 0 may initialize |
| 3 | PULL_REQ | syncer → learner | fragment_id:u32, global_step:u64, round_attempt:u32 |
| 4 | PUSH_FRAGMENT | learner → syncer | learner_id:u32, fragment_id:u32, global_step:u64, round_attempt:u32, base_version:u64, local_step:u64, c_steps:u32, c_tokens:u64, routed-token tag block (only when HELLO flag bit 2 is set; see "Expert token weights"), base-relative learner-delta bytes |
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
| 40 | RECOVERY_CLOCK | syncer → recovering learner | t_s:u64, n:u32, V_s:u64 × n, donor:u32 (0xFFFFFFFF = none) |
| 41 | PEER_SNAPSHOT_REQ | syncer → donor | recovering_id:u32, recovering_generation:u64, t_s:u64, n:u32, V_s:u64 × n |
| 42 | PEER_SNAPSHOT_READY | donor → syncer | recovering_id:u32, recovering_generation:u64, n:u32, V_d:u64 × n, sha256:[u8;32], len:u32, locator utf-8 |
| 43 | RECOVERY_SOURCE | syncer → recovering learner | kind:u8 (1 peer, 2 degraded), donor:u32, n:u32, V_d:u64 × n, sha256:[u8;32], len:u32, text utf-8 (locator, or the degraded reason) |
| 44 | RECOVERY_RESTORED | recovering learner → syncer | has_versions:u8, n:u32, held:u64 × n (has_versions 0 and n 0: resend every fragment) |
| 45 | RECOVERY_FAILED | recovering learner → syncer | len:u32, reason utf-8 |
| 46 | RECOVERY_DONE | syncer → recovering learner | t_s:u64, done_step:u64, replayed:u32, n:u32, versions:u64 × n |

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

## HELLO authentication (secret-handling-hardening)

When the syncer has an island HMAC key (`YETO_ISLAND_HMAC_KEY`; required
unless `--allow-unauthenticated-islands`), the HELLO payload is:

```
body := version .. num_streams:u16 [ b"YCG1" | u32 n | n bytes utf-8 ]   # compat_group trailer optional
HELLO payload := body | mac:[u8;32]
mac := HMAC-SHA256(key, b"yeto-hello-mac-v1\0" | body)
```

- Order: the compat_group trailer (clear text, 8 + n bytes, 1 <= n <= 64) is
  part of `body`; the 32-byte MAC is always the last field. The MAC therefore
  covers the card-type group, so it cannot be changed in transit.
- The syncer first strips and checks the last 32 bytes, then parses `body`
  exactly as above (profile hash, num_streams, trailer detection).
- Without a key nothing is appended (wire bytes unchanged).
- A wrong or missing MAC gets ERROR `HELLO authentication failed: ...` on that
  connection only; the client maps it to metric `island_auth_failed` and does
  not retry. DATA_HELLO carries no MAC; it can only join the 64-bit random
  connection generation of an authenticated HELLO.
- `--expected-island-contract <hex32>`: the session contract must equal
  `sha256(b"yeto-rl-session-contract-v2\0" | layout_fingerprint | pinned)`.

## Card-type compatibility group (yeto-framework-decoupling 7.7)

- Every RL island declares `compat_group` = `"<vendor>-<card>"` in lower case,
  for example `nvidia-h100`, `nvidia-h200`, `ascend-910b`. The value comes from
  the card catalog `yeto/hw/catalog.py`. The launcher sets it from the `--gpu`
  card type (env `YETO_RL_COMPAT_GROUP`). An unknown card stops the launch; there
  is no default.
- The group is a field of `BackendIdentity` (schema v2), so it is inside the
  identity hash and the HELLO session contract.
- The group is also sent in clear text as the HELLO trailer above. The syncer
  pins the first admitted group. A HELLO with another group gets ERROR
  `兼容组不同：<a> 对 <b>，容差未标定` on that connection only. The syncer does not
  exit and the admitted islands keep running. H100 and H200 are different groups.
- Phase 1 rule: groups must be identical. `CARD_PAIR_TOLERANCE` in the catalog is
  the key for calibrated card-pair tolerances (phase 2); it is empty now.
- Driver version and CUDA version are recorded only: island log line
  `[yeto-island] compat_group=...`, event `rl_island_hardware`, dashboard island
  row. They never cause a refusal (user decision 2026-10-09). Reason: the checks
  that run during training compare sha256 hashes only and do not depend on the
  hardware; recompute comparisons run only in tests.
- Version boundary: a syncer without this change refuses the HELLO trailer and
  the longer JOIN; this syncer refuses a JOIN without the field.

### What the syncer exchanges (7.7d)

The syncer only exchanges deltas in a neutral format: flat f32 or bf16 tensors
(q4 only as a wire compression of the same flat tensor), laid out by the HELLO
fragment layout, which is built from the canonical parameter names
(`yeto.rl.core` canonical specs; each backend maps its own names through its
parameter-name map, whose hash is in the identity). No engine-specific layout,
sharding or device format crosses the syncer. This is the precondition for
phase 2 (merging different card types after the tolerance is calibrated).

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

## Expert token weights

With `--expert-token-weights` the syncer weights each expert tensor of a
mixture-of-experts layout by the tokens the island routed to that expert
during the push's outer interval. Every HELLO must set flag bit 2 of the
extension trailer, and a HELLO with that bit is refused by a syncer without
the option. Each PUSH_FRAGMENT then carries, after c_tokens:
`n:u32` (the fragment's tensor count) and per tensor in layout order
`routed:u64, layer_routed:u64, layer_experts:u32`. A dense tensor sends
`(0, 0, 0)`. Tensor `t` of island `m` merges with weight
`w_m · routed · layer_experts / layer_routed`, where `w_m` is the
`--learner-weight` weight; the scale is 1 under even routing and 0 for an
expert the island never routed to (or a layer without tokens). Without the
option the HELLO and PUSH_FRAGMENT bytes are unchanged.

## Syncer epoch and peer recovery

The HELLO extension trailer is the last 13 bytes of the HELLO body (inside
the HMAC): `b"YEX1"`, flags (bit 0: syncer_epoch is set, bit 1: recover, bit 2:
expert token tags), syncer_epoch:u64. A client that sets no flag sends no
trailer.

Syncer epoch: a HELLO whose epoch is lower than `--syncer-epoch` is refused
with an ERROR that starts with `fenced: syncer_epoch`; the client does not
retry it. A HELLO without an epoch is admitted. The epoch and the retired
connection generations are written to `<checkpoint>.membership` whenever they
change and restored on `--resume`; the configured epoch only raises the
restored one.

Lease expiry: with `--island-lease-s`, a silent island gets the ERROR
`island lease expired: re-JOIN required`. The client redials with a new
generation (this first redial does not use `max_reconnects`); the syncer
catches it up and fences its pushes whose base predates the catch-up. The
catch-up is one BCAST_FRAGMENT per fragment at its newest version, which may
skip rounds; the client keeps at most the newest version per fragment, so an
island applies such a jump at once (alpha = 0).

With any membership option the fragment-round server starts its first round
once the launch quorum (K, or `q_min` of a capacity quorum) is connected and
initialized; without one it waits for all `--learners`.

Peer recovery (an island that lost its trainer state, HELLO with the recover
flag):

1. The syncer registers the connection but sends it no pulls and no
   broadcasts. After pending checkpoint publishes are durable it sends
   RECOVERY_CLOCK `(t_s, V_s)` and PEER_SNAPSHOT_REQ to the current member
   with the fewest unanswered pulls.
2. The donor keeps training. Once every fragment version it holds is at least
   `V_s[p]` it writes its parameters and inner optimizer state to the
   snapshot store and answers PEER_SNAPSHOT_READY. The syncer relays only this
   record (RECOVERY_SOURCE); it never carries the snapshot, and islands never
   stream it to each other. The store is a directory both islands see, or an
   object store (`yeto.rl.peer_recovery.SnapshotStore`). A snapshot is a set
   of files plus a manifest with the sha256 and size of every file; the
   message sha256 is the manifest's, and the locator names the snapshot.
3. The recovering island checks the sha256 (manifest and every file) and the
   trainer layout, restores, and sends RECOVERY_RESTORED with the versions it
   holds, or with none to receive every fragment (a sharded trainer). If it
   cannot use the snapshot it sends RECOVERY_FAILED and the syncer asks the
   next donor. A restored snapshot is removed; the locator starts with the
   creation time, and snapshots older than a TTL (a donor or recovering
   island died) are removed by the donors.
4. The syncer resends the current value of every fragment whose version is
   newer than the held version (broadcasts overwrite, so this equals
   replaying every update since the donor's snapshot; no update log is kept),
   sends RECOVERY_DONE, puts the island on the roster and rebinds its
   unanswered pulls. The `recovery` event-tape record holds the steps from
   the clock to DONE and whether they exceeded `--recovery-bound-steps`.

With no donor, after `--recovery-max-donor-tries` donors, or outside the
training loop (initialization, budget cutoff, finalization) the island gets
the degraded source: it restores on its own and receives every fragment.
A donor that disconnects or misses `--recovery-donor-timeout-s` is replaced.

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

## Streaming weighted fold (`--weighted-fold streaming`)

Rounds whose responder set or weights are fixed at launch (quorum covers the
roster, equal weights, no HeLoCo) always fold pushes in learner-id order,
bit-identical to the buffered merge. Other rounds (quorum below the roster,
grace window, tokens²/steps weights) buffer every push until the round
closes by default. With `--weighted-fold streaming` they instead add each
accepted push on arrival to unnormalised f32 sums (Σw·Δ; for RDA also
Σw·Δ/‖Δ‖ per tensor and the f64 scalar Σw‖Δ‖) and divide by Σw at close.
The push buffer is released immediately, so a round holds two accumulators
plus the push being decoded. The result equals the buffered merge up to f32
rounding and is deterministic for a given arrival order.

## Fragment-sharded head (`--shard-index i --shard-count N`)

One logical syncer may run as N processes. Shard `i` owns the fragments
`[⌊i·P/N⌋, ⌊(i+1)·P/N⌋)`; it allocates, merges, broadcasts and checkpoints
only those (other fragments are stored with numel 0 at version 0).

- Global steps are interleaved: shard `i` runs `t ≡ i+1 (mod N)`; its k-th
  round (k = ⌊(t−1)/N⌋) merges its fragment `start + k mod len`. Every step
  has exactly one owner, computable from `t`.
- A fragment's version is the step of its last merge. Each fragment has one
  owner, so versions are globally unique and the union of the shards'
  versions is the head's vector clock. Shards exchange no messages.
- Shards advance independently: steps of different shards are not ordered
  in wall time. Within a shard, commits stay in strict step order and the
  checkpoint is a contiguous cut of that shard's steps.
- An island connects to every shard (same learner id, HELLO and layout) and
  sends INIT_PARAMS and PUSH_FRAGMENT for fragment `p` only to `p`'s owner.
  PULL and BCAST from all shards are merged. A shard refuses INIT for a
  fragment it does not own.
- Each shard sends its own FINAL_FRAGMENTs and a FINAL_MANIFEST over all P
  fragments in which only its own versions are meaningful. The island waits
  for every shard, takes each version from its owner, and ACKs each shard's
  manifest separately. The terminal cut is the union.
- `N = 1` (default) is the unsharded syncer, unchanged. Strict policy
  sweeps, dynamic membership and learner-budget cutoff refuse `N > 1`.

## bf16 pushes over an f32 session (dtype 4)

Dtype 4 keeps INIT, BCAST and FINAL exact f32 and sends only PUSH deltas as
bf16 (round-to-nearest-even). The syncer decodes and negates them like dtype 2
and keeps f32 masters and momentum. The full-parameter policy sweep accepts
dtype 1 or 4 (S26 E3, `--rl-sync-push-dtype bf16`).

## int8 pushes over an f32 session (dtype 5)

Dtype 5 sends PUSH deltas as block-scaled int8: u32 block size B
(32..1048576), then per block of B values (the last block may be shorter) an
f32 scale (block absmax / 127) and one i8 per value (`-128` is invalid).
A value decodes to `q * scale`, so its error is at most absmax / 254. The
learner picks B (launcher `--rl-sync-push-int8-block`, default 128); it is
bound into the role-stream session contract. INIT, BCAST and FINAL stay f32.

## bf16 downlink deltas (dtype flag 0x80)

On an f32-bulk session (dtype 1, 4 or 5) the flag 0x80 changes BCAST_FRAGMENT
to `fragment_id:u32, version:u64, kind:u8, base_version:u64,
anchor_sha256:[u8;32]`, then the tensor: kind 0 (FULL) carries f32 values,
kind 1 (DELTA) carries `bf16(theta - anchor)`. Syncer and learner hold the
same f32 anchor per fragment: FULL sets it, DELTA adds the decoded delta in
f32; the digest is checked after every update. Every N-th version of a
fragment is FULL (`--bcast-delta-full-every N`, N >= 2; the syncer refuses a
flagged session without it and the reverse). Initial and reconnect re-sends
are always FULL with the shared anchor. A learner with a missing or wrong
anchor drops its connection, and the reconnect re-send restores it. The
syncer's masters, momentum, checkpoints and FINAL are unchanged and exact;
anchors are not checkpointed, so a resumed syncer starts with FULL.

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
wait rather than falling back to a locally blended save. With
`--island-lease-s`, the final cut stops waiting for a learner whose lease
expires during finalization, or that stays disconnected for one lease; the run
finalizes when every other learner acknowledged, and fails if none did.

FINAL_MANIFEST and FINAL_ACK carry `FINALIZATION_REVISION=1`; mismatches are
fatal. A learner also rejects SHUTDOWN before it has applied and initiated the
ACK for the exact manifest.

### Manifest-only final cut

`--final-cut-payload manifest` (default `full`) sends FINAL_MANIFEST alone: no
FINAL_FRAGMENT. A learner of such a syncer waits for the manifest
(`SyncerClient.wait_for_final_manifest`), checks the final step and versions,
lets its queued pushes drain, and sends FINAL_ACK without applying anything:
its trainer keeps its local weights. The final parameters exist only in the
syncer's final checkpoint (and `--final-state`). The ACK, membership,
generation and grace rules above are unchanged. A learner must know the mode
in advance, so it is part of the semantic profile hash: the domain
`yeto-syncer-final-cut-manifest-only-v1\0` is appended when it is on (off
keeps every existing hash). The launcher selects it for the full-parameter
preset with `--rl-sync-final-cut ack`, the default for asynchronous rounds
in `--rl-sync-mode async` unless islands evaluate (`--rl-eval-interval`; the
strict sweep and `--rl-sync-mode strict` keep the full cut by default); `--rl-sync-final-cut download` keeps the full cut,
for example to evaluate the final policy on an island or to compare island
and syncer parameters bit for bit. For a 30B-parameter model the full cut is
about 120 GB of f32 per island; the coordinator sends it to one island after
another.

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
   count. Quorum is the full configured learner set. Gather/SVD work may be
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
contiguous round `t`. Other rounds may still be gathering or running exact
SVD, but their owned buffers cannot touch coordinator state. A checkpoint at
that cut is therefore consistent. Snapshots contain global/per-fragment
versions, f32 params, momentum, and the cumulative learner ledger. Cutoff and
terminal checkpoints explicitly drain the SVD pool first; a poisoned worker
prevents checkpoint publication. The exact-SVD pool bounds startup, every
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

## RL sync modes (`--rl-sync-mode`)

A launch with two or more islands on the strict-avg (LoRA), decoupled or
full-parameter preset fills the sync options it leaves unset from the sync
mode. An option given on the command line always wins. One island,
`sao-streaming-full`, `--experimental-rl-sync` and algorithms with a critic
always take the strict values.

| Option | `async` (default) | `strict` (control) |
| --- | --- | --- |
| `--rl-sync-full-parameter-rounds` (full-parameter) | `async`, at most 2 fragment rounds in flight | `sweep` |
| `--rl-sync-quorum` | 1 | every island |
| `--rl-sync-grace-ms` | 300000: cap of the adaptive grace window | 0 (no grace) |
| `--rl-sync-max-base-lag` | 1 round | 0 |
| `--rl-sync-learner-weight` | `tokens2-over-steps` | `equal` |
| `--rl-sync-matrix-merge` (decoupled, full-parameter) | `rda` (embedding fragments stay averaged) | `avg` |
| `--rl-sync-outer-lr` / `--rl-sync-outer-momentum` | 1.0 / 0 below 4 local rounds per sync (H), 0.7 / 0.9 from H = 4 | the preset's |
| `--rl-sync-push-dtype` (full-parameter) | `int4` (linear, block 128, blocks restart at tensor boundaries) | `f32` |
| `--rl-peer-recovery` (full-parameter) | on with async rounds, a fixed roster, one syncer process, no broadcast deltas and a snapshot store; else off | off |

The grace window adapts to the islands' measured step time through the
syncer's `--grace-gamma` and `--grace-tau`, which keep their defaults; no
value of gamma is prescribed, the cap only bounds a slow round. The
broadcast overwrites the islands' fragments (merge alpha 0) in both modes.
A quorum of 1 needs a base lag of at least 1 round: otherwise the push of an
island that missed the quorum is a strict failure of a fixed roster.
Asynchronous full-parameter islands train until the syncer stops them, so
`--rl-lr-schedule auto` gives them a constant learning rate.

`--rl-sync-mode strict` produces the syncer and island commands of the
launcher before the modes existed (tested byte for byte).

Resume: every syncer start with a checkpoint path writes the semantic
profile hash to `<checkpoint>.profile`. The async mode passes
`--require-profile-record`: `--resume` then refuses a checkpoint whose record
differs from the syncer's profile, or a checkpoint without the record:
such a checkpoint was written before the modes, under the strict values, and
must be resumed with `--rl-sync-mode strict` (or the session's original
explicit options). The strict-mode command does not pass the flag.

## Island pool (dynamic membership)

The syncer has one round loop. Without membership options the roster is
fixed: every island answers every round (strict RL). Any of the options below
makes the islands a dynamic pool; unset options keep the fixed-roster value,
and the syncer and island command lines stay unchanged.

The launcher takes the `--rl-sync-*` membership options for the strict-avg
(LoRA), decoupled and full-parameter presets; the full-parameter preset needs
`--rl-sync-full-parameter-rounds async`. The launcher refuses them for the
full-parameter sweep, `sao-streaming-full`, `--experimental-rl-sync`,
`--syncer-shards` > 1 and algorithms with a critic. The rounds must be able
to merge without some island (`--rl-sync-quorum-theta`, `--rl-sync-quorum`
below the island count, `--rl-sync-grace-ms` or `--rl-sync-max-base-lag` >= 1).

| Syncer flag | Launcher option | Effect |
|---|---|---|
| `--quorum-theta F`, `--q-min N` | `--rl-sync-quorum-theta`, `--rl-sync-q-min` | merge once the arrived islands hold F of the capacity the round expects (at least N islands); replaces `--quorum` |
| `--soft-deadline-s S` | `--rl-sync-soft-deadline-s` | merge what arrived after S seconds even below quorum; also caps the grace window |
| `--carry-gamma G` | `--rl-sync-carry-gamma` | weight `G^lag` for a push whose base lags the round; the largest lag is `--max-base-lag` (`--rl-sync-max-base-lag`) |
| `--island-lease-s L` | `--rl-sync-island-lease-s` | an island silent for L seconds leaves the pool and must re-JOIN |
| `--syncer-epoch E` | `--rl-syncer-epoch` | a HELLO with a lower epoch is refused (fencing) |
| `--final-grace-s S` | `--rl-sync-final-grace-s` | after the last step, wait S seconds for the connected islands to ACK the final cut, then finish without the missing ones |
| `--max-learners N` | `--rl-sync-max-learners` | admit island ids 0..N-1 (islands may join late); not for full-parameter, whose island session binds the launched roster |

Every island in a pool heartbeats every lease/3 (default lease 150 s,
`yeto.rl.pool_client.LeaseHeartbeat`). A lease expiry gets
`MSG_ERROR "island lease expired: re-JOIN required"`; the client redials with
a new connection generation (outside its reconnect budget) and is caught up
to the current versions. Per preset:

- strict-avg (`PoolIslandClient`, session `PoolAvgSync`) syncs its LoRA vector
  as one f32 fragment, answers the newest PULL it has not answered with
  `local - base` against the base it holds, and applies the newest broadcast.
- decoupled (`yeto.rl.decoupled.DecoupledRlBridge`) applies the newest
  broadcast of each fragment, drops a PULL whose round closed without it (a
  newer PULL of the fragment supersedes it), and answers the PULLs the syncer
  replays on a new connection. A lost link redials up to 5 times.
- full-parameter, asynchronous rounds (`MilesFullParameterRoleStream`,
  `FullParameterSync`) overwrites a fragment with a broadcast that skips
  versions (catch-up) and releases its retained pushes of the skipped rounds.
  A restarted island process (new connection generation, same island id)
  joins a running pool: the syncer admits it (it supersedes a live older
  generation, or re-JOINs after a lease expiry) and resends every fragment at
  its current version. The island takes the newest version of each fragment
  as its initial cut: the trainer weights are overwritten with it and it
  becomes the anchor of the island's next push. Only island 0 sends INIT; a
  restarted island 0 sends it again and the running syncer ignores it. The
  restarted island starts with a fresh optimizer state (peer recovery,
  `--rl-peer-recovery`, restores it where available). At the end the island
  ACKs the final cut like a strict
island (by default the manifest only, see "Manifest-only final cut"); a push
that races the end ends the island normally (`elastic_finished` on its tape).

With a pool the syncer also writes island-pool records on its event tape
(`{"kind": ..., "syncer_epoch": ..., "time_unix": ...}`):

| kind | When |
|---|---|
| `pool_join` | an island connected (`catch_up` true after the run started) |
| `pool_leave` | an island left: `reason` from its LEAVE, `lease_expired`, or `disconnected` |
| `delta_accepted` / `delta_carried_over` / `delta_rejected` | a push was merged on time, merged with `lag` and `discount`, or refused (`carry_lag_exceeded`, `fenced`, `out_of_round`, ...) |
| `outer_step` | a round merged: `arrived`, capacities, `timed_out`, normalized `weights` `[island, base_version, weight]`, `absent` |
| `final_grace_expired` | the final grace window ended with islands missing |

It keeps `status.json` next to the tape (schema
`yeto.syncer.elastic-status/v1`: state, syncer epoch, outer version, live
islands with capacity and remaining lease).

### Local init (INIT_DIGEST / INIT_CONFIRM)

When every island loads the same base checkpoint (`--rl-sync-init-mode
local`, the default with `--rl-sync-full-parameter-rounds async`), each island
sends INIT_DIGEST (type 28) on its control socket right after every HELLO
until its initial cut arrives: `u32 learner_id | [32] layout_hash | u32
first_fragment | u32 count | count x [32] sha256`. Each digest is the SHA-256
of the fragment's INIT_PARAMS tensor bytes; the list covers exactly the
fragments the receiving syncer (shard) owns. Only f32-bulk sessions accept it.

Learner 0 still sends INIT_PARAMS, so the syncer's f32 masters, merges,
checkpoints and resume are unchanged. The syncer checks that every island's
layout hash and digests agree and that each applied INIT hashes to the agreed
digest. Any difference stops the run as the strict failure
`init_digest_mismatch`; every connected island first receives MSG_ERROR
`init digest mismatch: ...` and fails with the same metric.

When the checks pass on a fresh start, an island that sent digests receives
INIT_CONFIRM (type 29: `[32]` SHA-256 over the concatenated digests) instead
of the version-0 broadcast and keeps its own weights as version 0. An island
without digests, a resumed syncer, and late joiners still get the broadcast.

### Island departure (LEAVE)

An island that leaves a pool sends LEAVE (type 27: `u32 learner_id | u8
reason`) and closes its connection. Only the class of the departure goes on
the wire; for a failed departure the exception type and text go to the
island's event tape (`island_leave`), and a failed send is recorded as
`elastic_leave_failed`. The codes are defined in `yeto/rl/pool_client.py`
(`LeaveReason`) and `syncer/src/protocol.rs` (`LEAVE_REASON_NAMES`); a test
pins both.

| Code | Name | Sent when |
|---|---|---|
| 0 | FINISHED | the run ended normally, or the syncer finished first |
| 1 | LEASE_EXPIRED | never sent: the syncer assigns it when an island is silent for the lease |
| 2 | STOPPED | KeyboardInterrupt / SystemExit |
| 3 | PREEMPTED | a droppable island received a cloud reclaim notice |
| 4 | INFRA_FAILED | node or trainer lost, reconfiguration or storage failure, any unmapped error |
| 5 | TRAINING_FAILED | a training check failed (publication, policy identity, optimizer step, non-finite values) |
| 6 | CONFIG_MISMATCH | incompatible configuration |

The exception table is `yeto/rl/engine/leave_reason.py`. An island sends at
most one LEAVE: a reclaim listener's PREEMPTED is not overwritten by the
driver's close. A SIGTERM that kills the process sends nothing; the syncer
records `disconnected` (or `lease_expired` after the lease).

At the end of the run the launcher waives the missing `rl_learner_finalized`
record only for islands whose last syncer `pool_leave` reason is `finished`,
`stopped`, `preempted` or `lease_expired`; any other reason keeps the run
failed.

To add an island to a running pool there is no CLI: start another island
process against the same syncer address with the same `--rl-syncer-epoch`,
the same HMAC key and a compatible backend identity and compat group (and an
id below `--max-learners`). It is caught up to the current base.

## Task dispenser (S27 P2, `--dispense`)

The actor syncer can hand out the rows of `--data` that the training islands
train (`syncer/src/dispense.rs`, launcher `--rl-data-split dispense`). An
island process opens its own connection for it, like BW_PROBE, never part of
a learner session; shard 0 of a sharded head only.

Positions: the dispenser gives out positions of one global sequence; position
`p` is row `pass_order(rows, seed, p // rows)[p % rows]` (island side,
`yeto.rl.engine.task_dispense.PositionMap`). The first island fixes `rows`
and the data fingerprint; a later island with other values is refused.

| type | name | direction | payload |
|---:|---|---|---|
| 50 | TASK_HELLO | island → syncer | version:u16 (=1), learner_id:u32, incarnation:u64, rows:u64, fingerprint:[u8;32]; or with streams (S27 P3) version:u16 (=2), learner_id:u32, incarnation:u64, fingerprint:[u8;32], n:u16 (1..256), n × (rows:u64, weight:u64 (1..2^62), name_len:u16 (≤ 64), name:utf-8), m:u16, m × stream:u16 (m = 0: every stream); then HMAC-SHA256(key, `b"yeto-task-hello-mac-v1\0"` \| body) when the syncer has an island HMAC key |
| 51 | TASK_WELCOME | syncer → island | version:u16 (the TASK_HELLO's), lease_ms:u64, passes:u64 (0 = unlimited), max_block:u32, reconnect:u8, superseded:u8, superseded_incarnation:u64, reclaimed:u64, next:u64 |
| 52 | TASK_CLAIM | island → syncer | seq:u64, count:u32 |
| 53 | TASK_GRANT | syncer → island | status:u8 (0 grant, 1 wait, 2 end), block:u64, retry_ms:u32, n:u32, n × (start:u64, end:u64) |
| 54 | TASK_REPORT | island → syncer | seq:u64, n:u32, n × (block:u64, kind:u8 (1 trained, 2 dropped, 3 requeue, 4 release), m:u32, m × (start:u64, end:u64)) |
| 55 | TASK_ACK | syncer → island | settled:u64, requeued:u64, exhausted:u64 |
| 56 | TASK_HEARTBEAT | island → syncer | empty (answered with TASK_ACK) |

Streams (S27 P3, `--data-mix` and `--rl-island-task-labels`): the rows may be
split into streams, one per (source data set, task label). A position then
carries its stream in bits 40 and up: position `(s << 40) | k` is the `k`-th
position of stream `s`, row `rows_s[pass_order(len(rows_s), seed, k //
len(rows_s), stream=s)[k % len(rows_s)]]`. Every island declares the same
stream table (rows, weight, name); a TASK_HELLO with another table is a
mismatch. An island may accept only some streams (its task labels); it then
gets only their positions. Fresh positions are interleaved across the
accepted streams by weight (Tijdeman's earliest-deadline rule over every
stream's weight; when none of the accepted streams is due, the one furthest
behind): while every claiming island accepts every stream, after `n` fresh
positions stream `s` has `n * w_s / sum(w)` of them to within one position.
Queued positions of the accepted streams come first, in stream order.
`--dispense-passes N` counts per stream. A claim that finds nothing gets WAIT
while a live holder still holds a position of one of the claimer's streams,
END once all of them are exhausted and reported. One stream with weight 1 and
an empty name, accepted by every island, is the P2 dispenser: the island then
speaks version 1 and positions are row sequence numbers.

Sampler members (S27 P4; shared with the spot rollout subscribers and sample
relay of S27 G-EA, `syncer/src/members.rs`): every head service that keeps
per-process state for islands outside the learner session uses one membership
rule.

- Member = `(id, incarnation)`. `id`: a training island's learner id; a
  sampler island's `2^31 | (u32_be(SHA-256("yeto-sampler:" + sampler_id)[0..4]) & 0x7fffffff)`
  (`sampler_id` = its buffer shard name, `--rl-sampler-node`;
  `task_dispense.sampler_holder_id`, `spot_rollout.release_store.subscriber_id_for`).
  `incarnation`: one nonzero random u64 per island process, the same on every
  channel the process opens (TASK_HELLO, SUBSCRIBE, RELAY_HELLO):
  `task_dispense.member_incarnation()` (from `YETO_SAMPLER_INCARNATION`, the
  sampler process id that Ray workers inherit).
- A new incarnation of an id supersedes the current one at once; a member
  silent on every channel for the lease (`--dispense-lease-s`) expires. A
  superseded or expired member is retired: its later requests on any channel
  are refused (`fenced: ...`); each service returns what the member held (the
  dispenser: its unreported positions, to the front of the queue).
- A fenced process opens again with a new incarnation on all its channels.
- One `Members` table per head process (target; today the dispenser owns its
  own table and the spot rollout services keep their connection state).

Sampler islands (S27 P4) claim like training islands; their `learner_id` is
`2^31` plus 31 bits of SHA-256 of their sampler id (buffer shard name), so a
relaunched sampler supersedes its old process and the ids never meet a
training island's learner id. A sampler reports a position trained once its
group is in the sample buffer.

Every request gets exactly one answer, in order. `seq` numbers an island
process's CLAIM and REPORT requests (strictly increasing); a request resent
after a lost answer repeats its `seq` and gets the stored answer, so it is
never applied twice. A refusal is MSG_ERROR and closes the connection:
`fenced: dispense ...` (the process is retired: the client drops its blocks
and opens a new incarnation), `dispense mismatch: ...` (other data: fatal),
`dispense refused: ...` (a malformed or impossible request: fatal).

Semantics:

- A holder is one island process (`learner_id`, random `incarnation`). A new
  incarnation of an island supersedes the old one at once. A holder silent
  for `--dispense-lease-s` (default 300) expires. In both cases the holder is
  retired and every position it did not report goes to the front of the
  queue; the next claim of any island takes them first. Expiry is checked at
  every request, so no timer is needed.
- A claim gets queued positions first, then fresh ones. With
  `--dispense-passes N` a claim that finds nothing gets WAIT while some island
  still holds positions, END once everything is reported. Without it the
  passes never end and a claim never waits.
- A report settles positions: trained, dropped (consumed, not trained),
  requeue (given out again, at most `--dispense-max-redo` times, then settled
  as exhausted), release (given out again, not a redo). A range must lie in
  the block's unreported positions; a bad entry refuses the whole report. A
  block whose positions are all reported is retired, so the state grows with
  the blocks in flight, not with the rows.
- A position is generated at least once and reported trained at most once.
- Checkpoint (record `YDS2` with the stream table; a P2 `YDS1` record still
  restores as one stream): the dispenser record is written to `<checkpoint>.dispense`
  at every checkpoint cut, before the checkpoint itself. `--resume` restores
  the record of that cut: every holder of the record is retired and its
  unreported positions are queued first; positions settled after the cut are
  given out again (the model returned to the cut too).
- Log: `--dispense-log` gets one JSON line per event (`dispense_streams` once
  with the stream table, `dispense_open` with the accepted streams (`accept`,
  null: every stream), `_grant` with the first and last pass, `_pass_start`
  with its stream, `_wait`, `_end`,
  `_report`, `_block_done`, `_reclaim` with the reason `lease_expired` or
  `superseded`, `_fenced`, `_checkpoint`, `_restore`; a restart reclaims
  every holder of the record inside `_restore`).
  `python -m tools.eval.dispense_check` replays it.
