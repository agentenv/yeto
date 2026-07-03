# Learner ↔ Syncer wire protocol (v2)

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
| 1    | HELLO          | learner → syncer | learner_id:u32, dtype:u8 (1=f32, 2=bf16), num_fragments:u32, then per fragment: merge_mode:u8 (0=avg, 1=rda), num_tensors:u32, numel:u64 × num_tensors |
| 2    | INIT_PARAMS    | learner → syncer | fragment_id:u32, tensor bytes (Θ_p^(0)); learner 0 sends all fragments once; syncer ignores if already initialized |
| 3    | PULL_REQ       | syncer → learner | fragment_id:u32, global_step:u64 |
| 4    | PUSH_FRAGMENT  | learner → syncer | learner_id:u32, fragment_id:u32, global_step:u64 (echoed from PULL_REQ), base_version:u64 (version of this fragment the learner last applied), local_step:u64, c_steps:u32, c_tokens:u64, tensor bytes (current θ_m,p) |
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

- **Fragments**: trainable parameters are partitioned into `P` fragments by
  balanced greedy bin-packing over tensors. Embedding-like
  tensors go to their own fragment with merge_mode=avg; everything else uses
  RDA. All learners must declare identical layouts in HELLO.
- **Schedule**: syncer global step `t = 1..T`; at each step exactly one
  fragment `p = (t-1) mod P` syncs (P = H round-robin, double-buffered).
- **Pull**: syncer sends PULL_REQ(p, t) to all connected learners. A learner
  answers at its next inner-step boundary, and only once `c_steps[p] ≥ 1`
  (this self-clocks the syncer to learner pace). Responses carry the counters
  `c_steps[p]`, `c_tokens[p]` accumulated since the learner last *received*
  fragment p.
- **Quorum + grace**: syncer waits for `K` PUSHes for round `t`, then a grace
  window (γ · slack, approximated by a configurable duration) to admit
  stragglers, then merges. Late PUSHes for an already-merged round are dropped.
- **Merge**: per-learner outer gradient Δ_m,p = Θ_p(prev) − θ_m,p, anchored at
  the syncer's own previous fragment value. Learner weights
  w_m = c_tokens² / c_steps (quantity × quality). merge_mode=avg: weighted
  mean of deltas. merge_mode=rda: per-tensor radial-directional averaging —
  weighted mean of norms × normalized weighted mean of unit directions
  (φ(0) := 0; degenerate direction falls back to weighted mean).
- **Outer step**: SGD with Nesterov momentum on the syncer, f32 state,
  per-fragment: buf ← μ·buf + Δ; Θ ← Θ − lr·(Δ + μ·buf).
- **Broadcast**: BCAST_FRAGMENT(p, t) to all connected learners. Learners
  overwrite the fragment (α = 0), reset that fragment's counters, and adopt
  `t` as their global step.
- **Recovery**: a (re)connecting learner sends HELLO; syncer replies with
  BCAST_FRAGMENT for every initialized fragment at that fragment's version.
- Merge math runs in f32 on the syncer regardless of wire dtype.

## Consistent snapshots

A sharded coordinator would need vector clocks plus Chandy-Lamport markers to
snapshot consistently across in-flight inter-shard messages. Here the syncer
is a single sequential actor, so the marker algorithm degenerates:
between rounds (after broadcasting step t, before pulling t+1) the channel
state is irrelevant to global correctness, and a checkpoint at that quiescent
cut is consistent by construction. The snapshot persists:

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
