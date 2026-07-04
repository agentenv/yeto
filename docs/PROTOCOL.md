# Learner ↔ Syncer wire protocol (v3)

Transport: a **connection group** of `1 + S` persistent TCP sockets per
learner (stream 0 = control, streams 1..S = data). A single cross-region TCP
stream is congestion-window-limited; striping large payloads across parallel
streams multiplies WAN throughput (the GridFTP trick). gRPC was considered
and rejected: protobuf encode/copy and HTTP/2 framing sit exactly on the bulk
tensor path and add overhead without adding anything the 7-message control
plane needs.

All integers little-endian. Tensors are raw contiguous bytes in the declared
dtype (fragment = concatenation of its tensors, in layout order).

The syncer owns the global step `t` and drives a pull-based, per-fragment
sync schedule; learners never block on the WAN.

## Framing

```
frame := magic:u32 (0xD170C0DE) | type:u8 | len:u64 | payload[len]
```

## Message types

| type | name           | direction        | payload |
|------|----------------|------------------|---------|
| 1    | HELLO          | learner → syncer | learner_id:u32, dtype:u8 (1=f32, 2=bf16, 3=q4), num_fragments:u32, then per fragment: merge_mode:u8 (0=avg, 1=rda), num_tensors:u32, numel:u64 × num_tensors |
| 2    | INIT_PARAMS    | learner → syncer | fragment_id:u32, tensor bytes (Θ_p^(0)); learner 0 sends all fragments once; syncer ignores if already initialized |
| 3    | PULL_REQ       | syncer → learner | fragment_id:u32, global_step:u64 |
| 4    | PUSH_FRAGMENT  | learner → syncer | learner_id:u32, fragment_id:u32, global_step:u64 (echoed from PULL_REQ), base_version:u64 (version of this fragment the learner last applied), local_step:u64, c_steps:u32, c_tokens:u64, tensor bytes (current θ_m,p; under dtype=q4: the q4-encoded delta θ_m,p − Θ_p(base_version)) |
| 5    | BCAST_FRAGMENT | syncer → learner | fragment_id:u32, version:u64 (new global step t), tensor bytes (Θ_p^(t)) |
| 6    | HEARTBEAT      | learner → syncer | learner_id:u32, local_step:u64 |
| 7    | SHUTDOWN       | syncer → learner | empty (training reached T global steps) |
| 8    | DATA_HELLO     | learner → syncer | learner_id:u32, stream_idx:u16 (attaches this socket to the learner's group as a data stream) |
| 9    | CHUNK          | either           | msg_id:u64, total_len:u64, offset:u64, bytes (slice of an inner frame) |

## Striping

- HELLO (on stream 0) carries `num_streams:u16` after the layout; the learner
  then opens that many extra sockets, each introduced by DATA_HELLO.
- Small messages (HELLO, PULL_REQ, HEARTBEAT, SHUTDOWN) travel unchunked on
  stream 0.
- Large messages (INIT_PARAMS, PUSH_FRAGMENT, BCAST_FRAGMENT) are serialized
  as a normal inner frame, split into fixed-size chunks (4 MiB), and the
  chunks are sent round-robin across the data streams wrapped in CHUNK
  envelopes. `msg_id` increases monotonically per sender per group; the
  receiver reassembles by (group, msg_id) and parses the inner frame when all
  bytes arrived. With zero data streams, large messages go on stream 0.

## Semantics

- **Fragments**: trainable parameters are partitioned into `P` fragments,
  either by balanced greedy bin-packing over tensors (default) or by
  depth-interleaved transformer layers (`--fragment-pattern strided`, the
  Streaming DiLoCo pattern: layer i → fragment i mod P). Embedding-like
  tensors go to their own fragment with merge_mode=avg; everything else uses
  RDA. All learners must declare identical layouts in HELLO.
- **Schedule**: syncer global step `t = 1..T`; at each step exactly one
  fragment `p = (t-1) mod P` syncs (P = H round-robin). Up to `--pipeline`
  rounds are in flight at once (default 2, Decoupled DiLoCo's "two fragments
  in flight"), always targeting distinct fragments; rounds may complete out
  of order (per-fragment versions, monotonic global step).
- **Pull**: syncer sends PULL_REQ(p, t) to all connected learners. A learner
  answers at its next inner-step boundary, and only once `c_steps[p] ≥ 1`
  (this self-clocks the syncer to learner pace). Responses carry the counters
  `c_steps[p]`, `c_tokens[p]` accumulated since the learner last *received*
  fragment p. At each boundary the learner applies received broadcasts
  BEFORE answering pulls: with pipelined rounds, the pull opening fragment
  p's next round (control stream) can overtake the broadcast that closed its
  previous one (data streams), and answering first would push a stale
  base_version.
- **Quorum + grace**: syncer waits for `K` PUSHes for round `t`, then an
  adaptive grace window to admit stragglers, then merges. The window is
  γ · (τ·ξ_step − ξ_quorum − ξ_sync) clamped to [0, `--grace-ms`]: wait only
  within the slack the learners' compute overlap leaves free (Decoupled
  DiLoCo Eq. 3). ξ_step is the slowest learner's inner-step time, estimated
  from the local_step counters on consecutive pushes; before any estimate
  exists the full `--grace-ms` cap is used. Late PUSHes for an
  already-merged round are dropped.
- **Merge**: per-learner outer gradient Δ_m,p = Θ_p(prev) − θ_m,p, anchored at
  the syncer's own previous fragment value. Learner weights
  w_m = c_tokens² / c_steps (quantity × quality). merge_mode=avg: weighted
  mean of deltas. merge_mode=rda: per-tensor radial-directional averaging —
  weighted mean of norms × normalized weighted mean of unit directions
  (φ(0) := 0; degenerate direction falls back to weighted mean).
- **Delta correction** (`--delta-correction heloco`, default): before
  merging, each learner's outer delta is corrected per tensor against the
  outer momentum (HeLoCo, arXiv 2606.00271): cos ≥ 0.2 passes through;
  anti-aligned deltas have their opposing component shrunk (bounded by
  β_max = 0.5); weakly-aligned deltas rotate toward the momentum preserving
  magnitude. Corrections are confidence-scaled by ‖Δ‖/(‖Δ‖+κ‖m‖) and skip
  entirely while the momentum is empty (early rounds). `none` disables.
- **Outer step**: SGD with Nesterov momentum on the syncer, f32 state,
  per-fragment: buf ← μ·buf + Δ; Θ ← Θ − lr·(Δ + μ·buf).
- **Broadcast**: BCAST_FRAGMENT(p, t) to all connected learners. A learner
  applies θ ← α·θ_local + (1−α)·Θ_p^(t) with α = `--merge-alpha`
  (default 0.5; α = 0 overwrites), keeping a share of the inner steps taken
  while the merge was in flight (Streaming DiLoCo / HALoS delayed-application
  blending). Either way it resets that fragment's counters and adopts `t` as
  its global step.
- **Recovery**: a (re)connecting learner sends HELLO; syncer replies with
  BCAST_FRAGMENT for every initialized fragment at that fragment's version.
- Merge math runs in f32 on the syncer regardless of wire dtype.

## Q4 delta pushes (dtype = 3)

Full parameters do not survive 4-bit encoding, but push *deltas* have a
small dynamic range, so a q4 session quantizes only PUSH_FRAGMENT payloads;
INIT_PARAMS and BCAST_FRAGMENT travel as bf16 (`bulk_dtype`).

- **Encoding**: values are grouped into blocks of 256; each block is an f32
  absmax scale followed by 128 bytes of packed nibbles (two values per byte,
  low nibble first). A nibble is 1 sign bit (bit 3) + a 3-bit level: level 0
  is exactly zero; level L ∈ 1..7 decodes to sign · 2^(L−7) · scale (E3M0).
  Encoding rounds to the nearest level in log space; magnitudes below
  2^−6.5 · scale round to zero. 4.125 bits/value ≈ 3.9× smaller than bf16.
- **Anchor**: the learner keeps, per fragment, the raw global value it last
  received (pre-blend, so anchors equal Θ_p at base_version exactly; before
  any broadcast this is the base-model value, identical on all learners) and
  pushes δ = θ_m,p − anchor. The syncer reconstructs θ_m,p = Θ_p(base_version)
  + δ in f32 and merges as usual.
- **Staleness**: reconstruction needs Θ_p at the learner's base_version, and
  the syncer holds only the current value; a q4 push whose base_version does
  not match the fragment's current version is dropped and logged ("stale q4
  delta dropped") instead of being admitted the way stale full-tensor pushes
  are. Matching is the steady state — learners re-anchor on every broadcast.

## Consistent snapshots

A sharded coordinator would need vector clocks plus Chandy-Lamport markers to
snapshot consistently across in-flight inter-shard messages. Here the syncer
is a single sequential actor, so the marker algorithm degenerates: state only
changes when a round completes (merge + broadcast, serialized in one task),
so a checkpoint at any completion is consistent by construction — other
pipelined rounds are still gathering and have not touched state. A
crash-resume loses those in-flight gathers; their fragments simply merge on
a later cycle, which the quorum design already tolerates. The snapshot
persists:

- global step t and per-fragment versions,
- global parameters Θ and outer (Nesterov) momentum,
- the cumulative merged-token/step ledger per learner.

Learner consistency on restore holds because recovery is idempotent: a learner
(re)connecting after a syncer restart receives the full fragment rebroadcast,
bounding its staleness by one sync cycle H. Pushes anchored at a base_version
older than the syncer's previous version for that fragment are logged as
stale (the weight formula already compensates; detection is for the event
tape). Every merge appends a JSONL event-tape record: step, fragment,
responders with base versions, weights, token counts.
