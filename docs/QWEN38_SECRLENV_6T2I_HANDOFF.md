# Qwen3.8 SecRLEnv 6-train/2-inference handoff

Snapshot time: `2026-08-22T02:43:56Z`

> **Architecture status (2026-08-24):** preserve this run as a diagnostic
> single-island full-parameter result, but do not use its `6T/2I` topology as
> the distributed RL template. The replacement multi-learner design is in
> [DECOUPLED_DILOCO_SECRLENV_PLAN.md](DECOUPLED_DILOCO_SECRLENV_PLAN.md).

## Authoritative identities

- Yeto branch: `review/wandb-v28-fullrun-20260818`
- Yeto source commit used by the run: `f5bcef352509f48373c096946363a1c11dcac5fa`
- Model: `Qwen/Qwen3.8-27B` BF16
- Model revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Miles image ID: `sha256:80c20538b63f76defde06ad5d4cfa564ae6f261110696eb1864470cb835e1590`
- SecRLEnv task-pack SHA-256: `cf1d277e1e5a91d42445c08df67bc1f164e1602211b016ad2ca2f565b0dfb759`
- Task count: 100; unique task-image pins: 112
- SecRLEnv corpus commit represented by the pack: `af1a29f0130b194d2367504d99d398c9ddf32d99`

## Architecture

- One `8xH200` node: `h200-n7`
- GPUs 0-5: full-parameter Megatron DDP training, TP2/PP3 with pipeline layers 22/21/21
- GPUs 6-7: one TP2 SGLang rollout engine
- Context length: 262,144
- Microbatch: 1; `qkv_format=bshd`; packed GDN/dynamic batching disabled
- Codex harness: Qwen3.8 native template, `reasoning_effort=xhigh`
- Each solver receives a private flagless DinD debug copy of its challenge
- Workload: five rounds, 20 prompts/round, three samples/prompt

## Exact code and artifacts

The complete authoritative staged bundle is on n7:

```text
/data/yeto-rl/inputs/qwen38-fullparam-xhigh-flaky100-6t2i-v6
├── run-6t2i.sh
├── run/train-6t2i.py
├── source/                  # Yeto source used by the run
├── miles/                   # exact patched Miles source
├── secrlenv/                # SecRLEnv runtime and corpus tree
├── test-subnet-concurrency.py
└── contract.json
```

Key artifact hashes:

```text
80fd6d5b8f4b8eb5464a30a83630f2c41412d2e2122b7b7efe3bc94c5d454639  run-6t2i.sh
068a0efe67b142c9fcbb08c17cafd3c2cc5e619d9fdafc2704e48631f8067e9b  contract.json
e968aafaed927a8086caca36cf7d9f9f09c7a115258182e3979b0b3454697ee0  secrlenv/secrlenv_rl/runtime.py
964fee5e1187b990c49484139a8407649709caff9ad538261af713b2c8cc259f  test-subnet-concurrency.py
```

The task pack is:

```text
/data/yeto-rl/taskpack-derivations/flaky100-rebuilt-v17-final-v2/cf1d277e1e5a91d42445c08df67bc1f164e1602211b016ad2ca2f565b0dfb759
```

The model is staged at:

```text
/data/yeto-rl/model-cache/huggingface/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
```

Local controller artifacts are under:

```text
/Users/walden/FuzzLand/rl-run-inputs/qwen38-fullparam-xhigh-flaky100-6t2i-v6
/Users/walden/FuzzLand/rl-run-inputs/secrlenv-v24-builder-audit-20260819/secrlenv_rl
```

Do not treat the dirty worktree at `/Users/walden/FuzzLand/yeto-secrlenv-fix`
as the run source. This clean review branch/worktree and the staged n7 bundle are
the authoritative sources.

## Current run

- Run ID: `qwen38-fullparam-xhigh-flaky100-6t2i-full-v125`
- Container: `yeto-rl-qwen38-fullparam-xhigh-flaky100-6t2i-full-v125`
- Daemon unit: `yeto-secrlenv-qwen38-fullparam-xhigh-flaky100-6t2i-full-v125.service`
- SecRLEnv control port: `28869`
- SGLang session port: `31801`
- Run root: `/data/yeto-rl/runs/qwen38-fullparam-xhigh-flaky100-6t2i-full-v125`
- Training log: `/data/yeto-rl/runs/qwen38-fullparam-xhigh-flaky100-6t2i-full-v125/output/train.log`
- Started: `2026-08-22T02:31:24Z`

At the snapshot boundary the container was running, `OOMKilled=false`, the
SecRLEnv daemon was healthy, GPU workers were active, and there were zero
terminal traceback/error matches. Treat these as a timestamped observation,
not a permanent assertion.

Monitor without changing state:

```bash
ssh root@h200-n7 'docker logs -f --tail 200 yeto-rl-qwen38-fullparam-xhigh-flaky100-6t2i-full-v125'
```

Compact status:

```bash
ssh root@h200-n7 '
docker inspect yeto-rl-qwen38-fullparam-xhigh-flaky100-6t2i-full-v125 \
  --format "status={{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}"
curl -fsS --max-time 3 http://127.0.0.1:28869/healthz
'
```

Never relaunch or mutate v124/v125 in place. Any code/config change gets a new
monotonic run ID.

## v124 failure and fix

v124 exited 1, not OOM. Rollout 0 and optimizer step 0 completed, but step 0 had
zero loss and zero gradient. During rollout 1, CVE-2018-11322 and
CVE-2018-12636 each launched three same-task samples while their compose files
declared a fixed IPAM subnet. Only one concurrent project could own each subnet;
the remaining provisions returned Docker `Pool overlaps` errors and exhausted
the single infrastructure replacement budget.

The generic fix is in `secrlenv_rl.runtime.EpisodeRuntime._write_compose_override`:
remove task-authored `ipam` from the per-episode resolved compose network and
let Docker allocate a unique subnet atomically for every compose project. This
fix covers every task with a static authored subnet, not only the two tasks that
first exposed it. The exact patch is in
`docs/patches/secrlenv-concurrent-static-subnet.patch`.

Validation:

- SecRLEnv runtime unit suite: 13/13 passed.
- Production-shaped test: nine affected flaky100 tasks, three concurrent
  episodes each, two waves.
- 26/27 initial creates reached ready with zero subnet-overlap errors.
- The sole miss was a separate private-DinD startup timeout; the exact one
  infrastructure retry reached ready and closed successfully.
- Effective result under the production retry contract: 27/27 ready.
- Final episode containers, networks, GPU processes, and test processes: zero.

The failed v124 run and logs remain preserved. Its daemon was stopped cleanly.
Because its only completed optimizer step had zero gradient and no model
checkpoint was written, v125's clean base-weight restart did not discard a
learned update.

## Ownership boundary

The network fix belongs in the SecRLEnv repository, not Yeto. This Yeto branch
records the exact patch and operational handoff, while n7's v6 input contains
the executable patched SecRLEnv file. Before the next unrelated run, upstream
the patch into SecRLEnv with its regression test and update the pinned runtime
identity deliberately.
