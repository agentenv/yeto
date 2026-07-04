# Design notes

Moved here from the README; docs/PROTOCOL.md has the wire-level detail.

- **Merging**: per-learner outer gradient Δ_m,p = Θ_p(prev) − θ_m,p anchored
  at the syncer's previous fragment value; learner weights
  w_m = c_tokens²/c_steps (quantity × quality); weighted RDA per tensor on
  non-embedding fragments, direct averaging on the embedding fragment (whose
  deltas lack the near-orthogonality that motivates RDA).
- **Broadcast blending**: learners apply a merged fragment as
  θ ← α·θ_local + (1−α)·Θ_global (`--merge-alpha`, default 0.5) instead of
  overwriting, keeping the inner steps taken while the merge was in flight
  (Streaming DiLoCo / HALoS; at large fleets prefer α=0 — Decoupled DiLoCo's
  ablation found overwrite wins as M grows).
- **Adaptive grace**: the post-quorum straggler window sizes itself to the
  learners' compute slack each round (γ·(τ·ξ_step − ξ_quorum − ξ_sync),
  capped by `--grace-ms`), instead of a fixed wait. The ξ estimates are
  EMA-smoothed per learner, as the paper prescribes.
- **Pipelined rounds**: up to `--pipeline` fragment rounds are in flight at
  once (default 2 — Decoupled DiLoCo's "two fragments in flight" at τ=2), so
  one fragment's quorum/grace/WAN latency never delays pulling the next.
  Concurrent rounds always target distinct fragments (depth is clamped to
  P); merges stay serialized in one scheduler task; rounds may complete out
  of order (per-fragment versions, monotonic global step). `--pipeline 1`
  recovers serial rounds.
- **Delta correction**: stale learner deltas that oppose the outer momentum
  are shrunk/reoriented per tensor before merging (HeLoCo;
  `--delta-correction none` disables).
- **Transport**: custom binary framing over parallel TCP streams (control on
  stream 0; 4 MiB chunks striped across data streams). gRPC was evaluated and
  rejected — protobuf copies and HTTP/2 framing sit on the bulk tensor path.
- **Q4 pushes**: `--wire-dtype q4` sends learner pushes as blockwise 4-bit
  E3M0 deltas against the last received broadcast (~3.9× less learner egress
  than bf16); broadcasts and init stay bf16. See docs/PROTOCOL.md.
- **Fragment patterns**: `--fragment-pattern binpack` (default,
  size-balanced) or `strided` (transformer layer i → fragment i mod P,
  interleaving depth across fragments as in Streaming DiLoCo).
- **Snapshots**: the single-actor syncer checkpoints at the quiescent cut
  between rounds (params, momentum, per-fragment versions, merged-token
  ledger). `--resume` restores; a JSONL event tape records every merge.
- **Fine-tuning**: `--tuning lora` (default) syncs only adapter weights —
  fragments are megabytes, so the syncer and WAN stay cheap even for large
  models. `--tuning full` syncs everything.
- **Task backends**: what yeto trains is a plug-in. The sync core (syncer,
  fragments, wire protocol, fleet orchestration) is task-agnostic; each *task*
  registers a `TaskBackend` (`yeto/backends/`) that owns its CLI flags,
  validation, learner-task construction, fit warnings, and export, reached
  through `get_backend(--task)`. `lm` (text LMs, the default) and `nava`
  (multimodal diffusion — a self-contained package under `yeto/nava/`) ship
  today. Within `lm` the intra-island trainer is a second plug-in, an
  `IslandEngine` (`--island-backend torch|megatron`; see docs/MEGATRON.md). A
  new task or engine registers itself with no edits to the CLI, launcher, or
  export dispatch.
- **Model sizing**: `deepseek4flash` (DeepSeek-V4-Flash, 284B MoE) needs
  ~568 GB for frozen bf16 weights — more than 8×A100-40GB (320 GB); use
  ≥16×80GB GPUs per learner, or pick `gemma4` (12B) / any smaller HF id.
  `yeto shape` automates this sizing.
- **Loss masking**: `--train-on assistant` (default) puts loss only on
  assistant-message tokens (plus the closing EOS); `--train-on all` trains
  on every token. Tokenization streams asynchronously in DataLoader workers
  (`--tokenize preload` to materialize upfront).
- **Resilience**: learners reconnect automatically through syncer restarts
  and WAN drops (exponential backoff; work continues locally during the
  outage and re-merges after the post-reconnect rebroadcast). The syncer
  checkpoint is the single durable source of truth — recover a model from it
  with `yeto-export --checkpoint yeto-state.ckpt --model <id> --output-dir out/`
  even if every learner is gone.
