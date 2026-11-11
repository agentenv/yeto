# Sandbox hosts: `yeto sandbox` and dispensing by readiness

`yeto sandbox` prepares the task environments of an agentic RL run on one or more sandbox hosts
(docker daemons that Harbor's `remote-docker` provider drives), and records in the run store which
task is ready where. The store dispenser reads that record and gives a sampler only tasks that are
ready on its host, and only as many as its sandboxes can start now.

## Task sets

A task set is a directory of Harbor task directories:

```
<task>/task.toml               [environment] docker_image = "..." (optional when a Dockerfile is present)
<task>/environment/Dockerfile  build context of the task image (optional when docker_image is pullable)
<task>/tests/  <task>/solution/
<task>/yeto-sidecars.json      optional: long-lived services this task needs on the host
```

A sidecar declaration (per task, or for the whole set in `sidecars_file` as
`{"tasks": {"<task id>": {...}}}`):

```json
{"sidecars": [{"name": "task-server", "image": "ts:task-1", "build": "server/", "port": 9111,
               "host_port": null, "container_name": null, "env": {"AUTH_TOKEN": "..."}, "health": "/health"}],
 "prefetch": ["runner-image:tag"]}
```

- `host_port`: fixed host port. Without it a free port of the host's `port_range` is allocated
  (ports already published on the host are skipped) and kept across re-runs.
- `container_name`: a container of that name that already runs `image` is **adopted**: recorded as
  ready, never restarted, never removed. A container of that name with another image is an error
  for that task, and the container is left alone.
- The published port binds the host's docker bridge gateway, probed per host.
- Text files of the task may contain `{{yeto.sidecar.<name>.url}}`, `{{yeto.sidecar.<name>.host}}`,
  `{{yeto.sidecar.<name>.port}}` and `{{yeto.host.bridge_ip}}`. `prepare` writes a rendered copy of
  each task per host into the stage directory (`--stage-dir`, default
  `~/.yeto/sandbox/<run>/stage/<host>/<task>`), and rsyncs it to `sync_to` when the host has `ssh`.
  The sampler node of that host must use that copy as `HARBOR_TASKS_DIR`.

## Configuration: `agentic.envs`

The same object is read from the sampler spec (`agentic.envs`) or from a standalone JSON file.

| Key | Meaning | Default |
|---|---|---|
| `run` | Label put on every container and image yeto starts or builds (`yeto.sandbox.run`). `teardown` removes these and the run's Harbor trial containers (`ai.yeto.run=<run>`) with their compose networks; nothing else. | required |
| `ready_key` | Manifest path in the run store. | `sandbox/ready.json` |
| `require_validation` | Dispense a task only after oracle and nop passed. | `true` |
| `affinity` | `strict`: a sampler takes only tasks ready on its own host. `prefer`: its own host first, then sidecar-free tasks ready on another host. | `strict` |
| `pull_parallel`, `build_parallel`, `validate_parallel` | Concurrency per host. | 4, 2, 4 |
| `hosts[]` | `name`, `docker_host` (`ssh://user@host`), `ssh` (optional, for disk, health checks and sync), `max_sandboxes` (host capacity, every owner's containers counted), `port_range`, `sync_to`. | |
| `tasksets[]` | `name`, `tasks_dir`, `ids`, `image_source` (`auto`: pull, else build; `pull`; `build`), `image_repo`, `agent_base` (`image`, optional build `context`), `bake_runtime` (`pi`), `sidecars_file`. | |

Bridge address, docker data-root and used ports are probed on each host, never configured.

`bake_runtime: "pi"` builds `FROM <task image>` plus the pinned pi install, tagged
`<image_repo>:<task>-pi-<hash>`, and points the rendered `task.toml` at it, so a trial finds pi
preinstalled and skips the per-trial apt/nvm/npm install. Task sets whose Dockerfile already starts
from a pi base image (`agent_base`) do not need it.

## Commands

```
yeto sandbox prepare  --envs spec.json --store <run store> [--host h] [--task id] [--stage-dir d]
yeto sandbox validate --envs spec.json --store <run store> [--out passed.txt] [--force]
yeto sandbox status   --envs spec.json --store <run store>
yeto sandbox teardown --envs spec.json --store <run store> [--images] [--dry-run]
```

- `prepare` is idempotent: present images and running sidecars with the same declaration are left
  alone. A task that fails is recorded with its error and the others go on; the exit code is 1
  when any task or host failed. Re-running finishes only what is missing.
- `validate` runs the oracle agent (must score 1) and the nop agent (must score 0) through yeto's
  Harbor path on each host. A wrong score marks the task failed; an infrastructure error leaves it
  unchecked, to be retried by the next `validate`. A task whose image digest changed is checked
  again.
- `status` prints, per host: running containers against `max_sandboxes`, this run's containers,
  free disk under the docker data-root, image readiness, sidecar health and how many tasks are
  dispensable.
- `teardown` lists containers with `yeto.sandbox.run=<run>`, checks each one's label again, and
  removes those only. Adopted containers, Harbor trial containers and other people's containers
  are never removed.

## The readiness manifest

`<run store>/sandbox/ready.json` (schema `yeto.sandbox-ready/v1`) holds host facts and, per task
and host: `prepared`, `image`, `digest`, `sidecars` (container, port, URL, adopted), `error`,
`prepared_at`, and `validation` (`passed` true / false / null, `oracle`, `nop`, `reason`, `at`).
Commands read, merge and write it; do not run two of them on the same run at once.

## Dispensing

With `agentic.envs` in the sampler spec and `prompts.dispense.backend: "store"`, each ledger scan:

1. reads the manifest (one read per scan). The manifest is a runtime filter, not part of the
   dispenser plan: every sampler keeps the same plan whatever the manifest says;
2. settles `dropped` every slot of a task whose validation failed, with the reason
   (`validate_failed: <host>:oracle_reward=0.0`);
3. claims only tasks that are ready (on the sampler's host under `strict`); a task that becomes ready
   later is claimed at a later scan;
4. claims at most `free // n` groups (`n` = `sampling.n`), where `free` is the smaller of the
   process cap (`HARBOR_MAX_SANDBOXES` minus held slots) and the host's `max_sandboxes` minus its
   running containers (probed at most every 15 s). With no room it claims nothing and emits
   `rl_dispense_wait_capacity`, so no lease is held by a task waiting for a sandbox.

The sampler's host is `YETO_SANDBOX_HOST` (a host name), else the host whose `docker_host` equals
`HARBOR_ENV_KWARGS.docker_host` or `DOCKER_HOST`, else the only host.
