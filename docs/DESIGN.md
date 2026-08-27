# Design notes

Moved here from the README; docs/PROTOCOL.md has the wire-level detail.

- **Merging**: every learner sends its base-relative update
  `d_m,p = θ_m,p − raw_anchor_m,p`; the syncer converts this once to the
  signed outer gradient `Δ_m,p = −d_m,p`. Merge math never subtracts a stale
  learner parameter from the current global fragment, so intervening global
  drift cannot enter the update. Learner weights
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
  P). Gather and exact-SVD compute may complete out of order, but the single
  coordinator commits Nesterov state, fragment versions, ledger/event tape,
  checkpoints, and broadcasts strictly in global fragment-step order, so
  every recovery checkpoint is a contiguous prefix.
  `--pipeline 1` recovers fully serial rounds.
- **Exact-SVD worker pool**: `--iso-worker-devices` starts one persistent
  Torch worker per listed device. Each bounded-queue job is one complete
  canonical f32 matrix on exactly one GPU; learner/TP shards never enter the
  pool. Worker failure poisons the pool, and both cutoff and terminal
  checkpoints require an explicit drain. The Miles six-island supervisor
  defaults to `cuda:0,...,cuda:7` and a bounded pipeline window of 16 on a
  dedicated host, allowing AVG fragments and uneven SVD sizes to overlap.
  Startup, request, and drain deadlines are bounded by
  `YETO_ISO_WORKER_{STARTUP,REQUEST,DRAIN}_TIMEOUT_S`; timeout diagnostics
  identify the worker/device/request where applicable, kill the direct Python
  child, and permanently poison the pool. Queue capacity bounds admitted
  queued-plus-running matrices because each resident permit is retained until
  its job completes or is discarded. It does not account for input vectors
  already allocated by callers waiting for admission.

  Production runs the syncer inside `miles_node` with a direct Python
  executable (`ISO_WORKER_PYTHON=python3`). Killing a `docker run` CLI does not
  prove that Docker stopped the daemon-owned CUDA process, so the Docker helper
  refuses persistent pool launches rather than claiming that cancellation is
  safe.
- **Frozen rendezvous**: a round attempt captures learner connection
  generations and quorum at launch. Joins/reconnects apply only to future
  attempts, disconnects do not erase accepted work, and a below-quorum
  timeout discards partial responses before retrying with a new attempt ID.
- **Sync-interval sensitivity** (legacy measurement, gemma4/Lean-Workbook,
  500k tokens, M=2, held-out eval CE): at the design-point sync interval
  (H≈24 inner steps per fragment) DiLoCo matched the synchronous run within
  noise (+0.5%); at H≈2 — what a LAN/localhost fleet produces naturally,
  since rounds complete as fast as learners answer — the stock outer
  optimizer (Nesterov 0.7/0.9) over-drove correlated deltas and cost ~+9%
  (α=0 overwrite was much worse; the direct-RDA configuration recovered to
  ~+3%). The old harness used a one-island baseline, so these numbers remain
  useful directional evidence but are not equal-hardware throughput results.
  The current `scripts/compare_diloco.py` names H≈24 `m2`, names H-disabled
  `unthrottled`, pairs every arm with a matching `baseline-mM`, and repeats
  training seeds; see `docs/LM_BENCHMARK.md`. WAN round latency yields large H
  by itself; for low-latency fleets the syncer self-throttles to
  `--sync-interval-steps` (default H=24, sized from measured learner step
  time; 0 disables), with `--min-round-interval-ms` as a manual floor on top.
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
- **Snapshots**: the single-actor syncer checkpoints after a contiguous
  committed prefix (params, momentum, per-fragment versions, merged-token
  ledger); uncommitted gathers/SVD results never touch state. Checkpoint V3
  also stores the ISO backend ID and exact 32-byte HELLO semantic-layout
  fingerprint, both validated before any restore mutation. `--resume`
  restores; a JSONL event tape records every merge in strict step order.
- **Fine-tuning**: `--tuning lora` (default) syncs only adapter weights —
  fragments are megabytes, so the syncer and WAN stay cheap even for large
  models. `--tuning full` syncs everything.
- **Model sizing**: `deepseek4flash` (DeepSeek-V4-Flash, 284B MoE) needs
  ~568 GB for frozen bf16 weights — more than 8×A100-40GB (320 GB); use
  ≥16×80GB GPUs per learner, or pick `gemma4` (12B) / any smaller HF id.
  `yeto shape` automates this sizing.
- **Loss masking**: `--train-on assistant` (default) tokenizes with the
  model's selected native chat template and uses its exact
  `assistant_masks` output. The template's `{% generation %}` blocks decide
  which assistant control/content/EOS tokens carry loss; Yeto does not add
  control tokens after templating. A selected template without generation
  tracking fails clearly. `--assistant-mask-mode legacy` explicitly restores
  the synthetic `<|role|>` compatibility format, while `--train-on all`
  retains the existing all-token behavior. Tokenization streams
  asynchronously in DataLoader workers (`--tokenize preload` to materialize
  upfront).
- **Resilience**: learners reconnect automatically through syncer restarts
  and WAN drops (exponential backoff; work continues locally during the
  outage and re-merges after the post-reconnect rebroadcast). The syncer
  checkpoint is the single durable source of truth. Recover a causal LM with
  `yeto-export --checkpoint yeto-state.ckpt --model <id> --output-dir out/`,
  or use `yeto-diffusion-export` with the same checkpoint/model/output flags
  for a diffusion adapter, even if every learner is gone.
