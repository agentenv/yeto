# Learner ↔ Syncer wire protocol (v1)

Transport: one persistent TCP connection per learner to the syncer. All
integers little-endian. Tensors are raw contiguous bytes in the declared dtype.
The syncer never blocks learners: learners push fragments on schedule and
apply broadcast fragments whenever they arrive.

## Framing

```
frame := magic:u32 (0xD170C0DE) | type:u8 | len:u64 | payload[len]
```

## Message types

| type | name           | direction        | payload |
|------|----------------|------------------|---------|
| 1    | HELLO          | learner → syncer | learner_id:u32, num_fragments:u32, dtype:u8 (1=f32, 2=bf16), numel per fragment: u64 × num_fragments |
| 2    | INIT_PARAMS    | learner → syncer | fragment_id:u32, tensor bytes (initial Θ_p^(0); only learner 0 sends; syncer ignores if already initialized) |
| 3    | PUSH_FRAGMENT  | learner → syncer | learner_id:u32, fragment_id:u32, base_version:u64 (global step t of the Θ_p this delta is anchored to), steps:u32 (c^steps), tokens:u64 (c^tokens), tensor bytes (θ_m,p current value) |
| 4    | BCAST_FRAGMENT | syncer → learner | fragment_id:u32, version:u64 (new global step t), tensor bytes (Θ_p^(t)) |
| 5    | HEARTBEAT      | learner → syncer | learner_id:u32, local_step:u64 |
| 6    | SHUTDOWN       | either           | empty |

## Semantics

- **Fragments**: the model's parameters are partitioned into `P` contiguous
  fragments (groups of tensors, flattened). Fragment `p` syncs on the
  round-robin schedule `t mod H == t_p` where `t_p = p * H / P` (double
  buffering: `P = H` gives one fragment per outer step slot).
- **Push**: when a learner's per-fragment inner-step counter reaches the
  fragment's schedule point, it pushes θ_m,p plus its counters and keeps
  training (no blocking).
- **Merge**: the syncer waits for a quorum of `K` pushes for fragment `p`
  (plus an optional grace window), computes per-learner deltas
  Δ_m,p = Θ_p(anchor) − θ_m,p, merges them with token/step-weighted RDA,
  applies the Nesterov outer step, bumps the fragment version, and broadcasts.
- **Apply**: learners apply BCAST_FRAGMENT between inner steps and reset that
  fragment's counters. Stale pushes (base_version older than the syncer's
  current fragment version) are down-weighted or dropped per config.
- **Recovery**: a (re)connecting learner sends HELLO; the syncer replies with
  BCAST_FRAGMENT for all fragments at the current version.

Merge math internals are computed in f32 on the syncer regardless of wire dtype.
