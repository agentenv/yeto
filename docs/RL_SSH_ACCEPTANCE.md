# Miles RL SSH Acceptance

`yeto-rl-ssh` runs the current fixed-roster Miles RL path directly on existing
GPU hosts. It is an acceptance and failure-injection tool, not a replacement
for `yeto launch`: it does not provision machines or supervise them after the
requested recovery actions.

The harness always creates two logical islands. Each `--host` value describes
one island and may contain one SSH target or a comma-separated node list. The
first node is that island's Ray head; the first node of island 0 also runs the
Yeto syncer. All nodes use the same `--gpus-per-node` value.

## Prerequisites

The controller needs this Yeto checkout, `ssh`, and `rsync`. Every remote node
needs Docker with the NVIDIA runtime, Git, and the requested number of GPUs.
The syncer node additionally needs Cargo and a C compiler. Use dedicated test
hosts: the harness stops an existing Ray runtime inside its containers and
uses host networking.

All SSH host names must be mutually reachable over a private network. Allow
TCP 29400 to the syncer and TCP 6379 within each island; do not expose either
port publicly. The model must use an immutable Hugging Face revision. Prompt
data may use either an immutable Hugging Face revision or a local file or
directory on the controller.

For secrets, create the same environment file on every node and pass its path
relative to remote `$HOME`. The path, but never its contents, is stored in the
plan. It can contain `HF_TOKEN` or `CYBERGYM_API_KEY`:

```bash
mkdir -p ~/.config/yeto
chmod 700 ~/.config/yeto
printf 'HF_TOKEN=%s\n' "$HF_TOKEN" > ~/.config/yeto/rl.env
chmod 600 ~/.config/yeto/rl.env
```

## Prepare And Start

For two single-node, four-GPU islands:

```bash
yeto-rl-ssh prepare \
  --host alice@island-a \
  --host alice@island-b \
  --gpus-per-node 4 \
  --syncer-address island-a:29400 \
  --remote-env-file .config/yeto/rl.env \
  --run-id miles-rl-acceptance \
  -- \
  --model Qwen/Qwen3.6-27B \
  --model-revision <immutable-model-commit> \
  --data <org/prompt-dataset> \
  --data-revision <immutable-dataset-commit> \
  --reward-function project.rewards:score \
  --total-steps 2 \
  --rollout-batch-size 4 \
  --n-samples-per-prompt 2 \
  --rollout-max-response-len 512 \
  --seq-len 1024 \
  --lora-r 8 \
  --trust-remote-code
```

For two multi-node islands, group each island's nodes in one value:

```bash
--host alice@a0,alice@a1 --host alice@b0,alice@b1 --gpus-per-node 4
```

The prompt/sample product for each round must divide evenly across every
island's `nodes * gpus-per-node` data-parallel ranks. Preparation runs the
normal Miles RL preflight, resolves provenance, fixes the current Miles commit
and image digest, and writes a content-bound plan under
`~/.yeto/ssh-runs/<run-id>/plan.json` by default.

For local prompt data, pass a local path and omit `--data-revision`:

```bash
--data ./prompts.jsonl
```

`prepare` binds the file contents, or a directory's relative file names and
contents, into the plan with SHA-256. Symlinks are rejected. `deploy` verifies
that identity again, copies the data to every island node, and mounts it
read-only at `/workspace/data`. Changing the local data after preparation
causes deployment to fail rather than running a different dataset under the
same plan. This local-data option belongs to the explicit SSH harness; the
public SkyPilot RL launcher continues to require a revision-pinned Hugging
Face dataset.

For a decoupled, variance-aware run, add these options to the learner
arguments. They preserve the optimizer budget while rejecting zero-variance
GRPO groups and generating replacements:

```bash
--rl-sync-preset decoupled \
--fragments 8 \
--pipeline 2 \
--local-rl-rounds-per-sync 4 \
--rollout-batch-size 4 \
--n-samples-per-prompt 8 \
--over-sampling-batch-size 16 \
--dynamic-sampling-filter-path \
  miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
```

For CyberGym, also bound replacement sampling and use the safer large-model
weight-publication path:

```bash
--dynamic-sampling-max-replacements 8 \
--rl-offload-train \
--rl-distributed-timeout-minutes 10
```

The stock Miles filter path is rewritten to Yeto's bounded filter. It prefers
non-zero-variance groups, rejects at most eight zero-variance groups, then
accepts a bounded fallback and records `rl/dynamic_filter/forced_groups`.
This avoids an unbounded replacement loop while retaining the variance
preference. `--rl-offload-train` rebuilds process groups before the external
weight update, and the distributed timeout makes a stuck barrier fail fast.

The harness records these settings in the content-bound plan and forwards
them to both learners and the syncer. With `--total-steps 55`, decoupled
execution uses `8 * 55 = 440` fragment steps, while still applying one local
optimizer step per rollout. Use the exact `--seq-len`, model revision, Miles
digest, and dataset manifest from the reference run when making a matched
comparison.

```bash
PLAN="$HOME/.yeto/ssh-runs/miles-rl-acceptance/plan.json"
yeto-rl-ssh start --plan "$PLAN"
yeto-rl-ssh status --plan "$PLAN"
```

`start` deploys the attested source, checks out the current detached Miles
revision on every node, recomputes the deployed Rust syncer-source hash before
the Cargo build, starts both Ray islands, and launches one current Yeto/Miles
learner per island. Re-running it with the same plan is idempotent, including
after an interrupted deployment.

## Failure Checks

Restarting a learner preserves its logical ID, completed-group checkpoint,
event tape, and f32 audit evidence:

```bash
yeto-rl-ssh kill-learner --plan "$PLAN" --learner-id 1
yeto-rl-ssh restart-learner --plan "$PLAN" --learner-id 1
```

A dead syncer connection causes current Miles islands to exit. Restart the
syncer from its authoritative checkpoint, then restart both logical learners:

```bash
yeto-rl-ssh kill-syncer --plan "$PLAN"
yeto-rl-ssh restart-syncer --plan "$PLAN"
yeto-rl-ssh restart-learner --plan "$PLAN" --learner-id 0
yeto-rl-ssh restart-learner --plan "$PLAN" --learner-id 1
```

## Collect And Verify

After both learner heads exit successfully:

```bash
yeto-rl-ssh collect --plan "$PLAN"
yeto-rl-ssh verify --plan "$PLAN" \
  --export-dir "$HOME/yeto-artifacts/miles-rl-acceptance"
```

Collection retains every container log and status, each island's local
checkpoint and event tape, f32 round evidence, and the syncer's log, event
tape, and authoritative checkpoint. Verification requires:

- one ordered fixed-roster f32 AVG commit per configured round;
- identical canonical f32 bases on both islands;
- an independent average of both recorded deltas at every round;
- each average to equal the next exact base and final syncer checkpoint;
- the complete two-member ledger and final policy hash on both islands.

When `--export-dir` is present, verification also exports the authoritative
checkpoint through the standard Yeto RL PEFT exporter. Use `stop` only for an
intentionally aborted run.
