# Two-H200 strict RL acceptance over SSH/Tailscale

`yeto-rl-ssh` runs the production strict syncer on the first H200 host and one
production Miles learner on GPU 0 of each of two existing hosts. It reuses the
normal `yeto launch --training-mode rl` preflight, so model/data revisions,
reward source, LoRA layout, Yeto source, Miles commit, image digest, workload,
and the strict roster all land in one canonical manifest before either GPU is
started.

## Host names and addresses

Suppose the machines are:

- H200 A: SSH target `walden@h200-a`, Tailscale/MagicDNS name `h200-a`
- H200 B: SSH target `walden@h200-b`, Tailscale/MagicDNS name `h200-b`

The earlier `<other-host>` placeholder simply meant the opposite machine. On
A it is `h200-b`; on B it is `h200-a`. The harness does not need that
placeholder: the controller opens SSH connections to both `--host` targets.

`<syncer-ip>` means the Tailscale address of the machine running the syncer.
The harness runs the syncer on the first `--host`, so here it is `h200-a` (or
A's `100.x.y.z` address), and the full address is `h200-a:29400`. If an SSH
alias is also valid through Tailscale MagicDNS, omitting `--syncer-address`
uses the first host name automatically. Otherwise discover and pass A's IP:

```bash
ssh walden@h200-a tailscale ip -4
ssh walden@h200-b tailscale ping 100.x.y.z
```

Keep TCP 29400 private to the tailnet (Tailscale ACL and/or host firewall); the
strict protocol is not a public Internet service.

## Prerequisites

The controller needs this checkout, Python dependencies, `ssh`, and `rsync`.
Both H200 hosts need:

- working SSH and Tailscale connectivity;
- Docker plus the NVIDIA container runtime (`docker run --gpus device=0 ...`);
- `git` and enough disk for the pinned image, model, and rollout state.

H200 A additionally needs Rust/Cargo and a C compiler to build the exact syncer
source copied by the harness. The run uses GPU 0 on each host. The preflight
fails if GPU 0's reported machine is not an H200.

For private Hugging Face inputs, run `hf auth login` on both hosts (their
`~/.cache/huggingface` directories are mounted into the containers), or create
the same remote env file on both hosts and pass a path such as
`--remote-env-file .config/yeto/rl.env`. That file may contain `HF_TOKEN=...`;
the harness does not copy it or store its contents in the plan.

Install the new entry point after updating the checkout:

```bash
python3 -m pip install -e .
```

You can equivalently replace `yeto-rl-ssh` below with
`python3 -m yeto.rl.ssh_harness`.

## Prepare and run

Use an immutable 40-character model revision. A local prompt dataset is
content-hashed and copied to both hosts; a Hugging Face dataset instead needs
its immutable `--data-revision`. The reward must be a batch callable inside
this repository with signature `fn(args, samples) -> sequence[float]` (sync or
async), selected as `package.module:function`.

```bash
yeto-rl-ssh prepare \
  --host walden@h200-a \
  --host walden@h200-b \
  --syncer-address h200-a:29400 \
  --run-id h200-rl-acceptance \
  -- \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --model-revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --data /absolute/path/to/prompts.jsonl \
  --reward-function your_package.cybergym_reward:score \
  --learner-image docker:radixark/miles@sha256:95b3afa9ee4313f5633e6ed3779c8276353cc8e24a2462e4f54ec0d5978fbae7 \
  --rl-global-rounds 2 \
  --rl-groups-per-island-round 1 \
  --rl-samples-per-group 2 \
  --rl-local-optimizer-steps 1 \
  --lora-r 8 \
  --lora-targets attention \
  --seq-len 512 \
  --trust-remote-code
```

Preparation resolves immutable provenance and derives the exact PEFT LoRA
layout locally. It prints the plan path, normally:

```text
~/.yeto/ssh-runs/h200-rl-acceptance/plan.json
```

Start and inspect the run:

```bash
PLAN="$HOME/.yeto/ssh-runs/h200-rl-acceptance/plan.json"
yeto-rl-ssh start --plan "$PLAN"
yeto-rl-ssh status --plan "$PLAN"
```

`start` copies the attested source and manifest into a run-specific directory
on both machines, checks Docker/H200/Miles identities, builds and starts the
syncer on A, proves `h200-a:29400` is reachable from both machines, and starts
logical learner 0 on A and learner 1 on B. Re-running it is idempotent for the
same plan; a different manifest cannot reuse the remote run directory.

## Recovery checks

Use a separate acceptance run when injecting failures. Learner restart keeps
the same logical ID, result cache, audit records, and remote run directory:

```bash
yeto-rl-ssh kill-learner --plan "$PLAN" --learner-id 1
yeto-rl-ssh restart-learner --plan "$PLAN" --learner-id 1
```

The syncer restart uses its authoritative checkpoint:

```bash
yeto-rl-ssh kill-syncer --plan "$PLAN"
yeto-rl-ssh restart-syncer --plan "$PLAN"
```

`kill-*` sends SIGKILL intentionally. `status` shows the container exit code,
recent learner output, syncer liveness, and final/fatal marker presence. A
strict run never shrinks from two learners; if one is not restarted, the run
waits (or fails at an explicitly configured round timeout).

## Collect and verify

After both learners and the syncer exit successfully:

```bash
yeto-rl-ssh collect --plan "$PLAN"
yeto-rl-ssh verify \
  --plan "$PLAN" \
  --export-dir "$HOME/yeto-artifacts/h200-rl-acceptance"
```

Verification fails closed unless all of these agree:

- canonical manifest hash, pinned layout, two-member roster, and final step;
- authoritative checkpoint plus the final marker written only after both
  learners applied and acknowledged the terminal policy;
- one ordered fixed-roster syncer commit and one trainer/rollout apply event
  per learner per round;
- each learner's exact raw-f32 base and transmitted delta, including payload
  digests;
- an independent learner-ID-ordered f32 average for every round, the next
  round's exact base, and the final checkpoint tensor/hash;
- optional standard PEFT export and clean-process reload with the same
  canonical policy hash.

Collected evidence remains next to the plan under `artifacts/`. `stop` is for
an intentionally aborted or stuck run; successful processes terminate on
their own.
