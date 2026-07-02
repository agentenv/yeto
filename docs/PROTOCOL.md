# Learner ↔ Syncer wire protocol (v2)

Transport: one persistent TCP connection per learner to the syncer. All
integers little-endian. Tensors are raw contiguous bytes in the declared dtype
(fragment = concatenation of its tensors, in layout order).

The design follows Decoupled DiLoCo (arXiv 2604.21428) Algorithms 1–2: the
syncer owns the global step `t` and drives a pull-based, per-fragment sync
schedule; learners never block on the WAN.

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
| 4    | PUSH_FRAGMENT  | learner → syncer | learner_id:u32, fragment_id:u32, global_step:u64 (echoed from PULL_REQ), local_step:u64, c_steps:u32, c_tokens:u64, tensor bytes (current θ_m,p) |
| 5    | BCAST_FRAGMENT | syncer → learner | fragment_id:u32, version:u64 (new global step t), tensor bytes (Θ_p^(t)) |
| 6    | HEARTBEAT      | learner → syncer | learner_id:u32, local_step:u64 |
| 7    | SHUTDOWN       | syncer → learner | empty (training reached T global steps) |

## Semantics

- **Fragments**: trainable parameters are partitioned into `P` fragments by
  balanced greedy bin-packing over tensors (paper Appendix C). Embedding-like
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
  BCAST_FRAGMENT for every initialized fragment at the current version.
- Merge math runs in f32 on the syncer regardless of wire dtype.
