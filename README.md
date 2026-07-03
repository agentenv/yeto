# Yeto

**Yeto** is an efficient, low-cost post-training tool: it fine-tunes language
models across cheap, geographically scattered GPU clusters (spot instances,
mixed regions, mixed clouds) launched with the [SkyPilot](https://skypilot.co)
SDK.

Yeto's asynchronous synchronization algorithm is based on **Decoupled DiLoCo**
([Douillard et al., arXiv 2604.21428](https://arxiv.org/abs/2604.21428)).

## Architecture

```
                 ┌──────────────────────────────┐
                 │  syncer (Rust, hot path)     │
                 │  fragment ingest · RDA merge │
                 │  Nesterov outer step · bcast │
                 └──────┬───────┬───────┬───────┘
                        │  TCP (binary framing, WAN)
        ┌───────────────┤               ├───────────────┐
 ┌──────┴──────┐  ┌─────┴───────┐  ┌────┴────────┐
 │ learner 0   │  │ learner 1   │  │ learner 2   │   … one per --gpu entry
 │ us-east-2   │  │ us-east-1   │  │ us-west-2   │
 │ 8×A100 node │  │ 8×A100 node │  │ 8×A100 node │   (PyTorch, AdamW inner opt)
 └─────────────┘  └─────────────┘  └─────────────┘
```

- **`yeto` CLI** — parses the `--gpu` spec, launches one
  SkyPilot cluster per learner plus a syncer VM, wires up IPs/ports, streams logs.
- **`syncer/`** — Rust implementation of the latency-sensitive syncer:
  async TCP server, per-fragment sync schedule (interval `H`, round-robin
  offsets), quorum-`K` gather with grace window, token/step-weighted RDA
  merge, Nesterov outer optimizer, consistent checkpoints, JSONL event tape.
- **`yeto/`** — Python package: learner training loop (HF transformers +
  AdamW inner steps, background fragment push/pull), data loading, loss
  functions, GPU-spec parsing, SkyPilot orchestration.

## Usage

```bash
pip install "yeto[launcher] @ ."

yeto launch \
  --gpu aws:8xa100@us-east-2,aws:8xa100@us-east-1,aws:8xa100@us-west-2 \
  --model deepseek4flash \
  --data armand0e/claude-fable-5-claude-code \
  --loss-function cross_entropy
```

`launch` provisions a small on-demand **head VM** that hosts both the
syncer and the fleet controller (SkyPilot managed-jobs style): the
submitting machine is free to disconnect the moment submission finishes,
and **Ctrl-C merely detaches** from the log stream. Runs are named by
`--cluster-prefix` (default `yeto`). `--controller local` keeps the
controller on your host instead. The head VM stays up after the run
until you `yeto down <run>`.

```bash
yeto status                # table of known runs
yeto logs <run>            # re-attach to a run's log stream (--no-follow to dump)
yeto down <run>            # stop the run's worker and tear down its clusters
```

`--gpu` grammar: `cloud:[nodes x]<count>x<gpu>@region`, comma-separated; one
entry per learner. E.g. `aws:4x8xa100@us-east-2` = one learner cluster of
4 nodes × 8×A100 in us-east-2.

Learners run on **spot instances by default** (pass `--on-demand` to opt
out); the syncer VM is always on-demand — it is the cheap, stateful
coordinator, and its checkpoint/resume covers learner preemptions.

## Design notes

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
  capped by `--grace-ms`), instead of a fixed wait.
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
- **Model sizing**: `deepseek4flash` (DeepSeek-V4-Flash, 284B MoE) needs
  ~568 GB for frozen bf16 weights — more than 8×A100-40GB (320 GB); use
  ≥16×80GB GPUs per learner, or pick `gemma4` (12B) / any smaller HF id.
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

## Testing

    python3 -m pytest tests/          # includes a real syncer+learner loop
    (cd syncer && cargo test)
