# Harbor harness for agentic RL

Harbor runs the agent side of agentic RL: task loading, sandbox, agent, verifier.
yeto adds the outer layer (admission, outcome, trust boundary) and the backend
glue for Miles and verl.

Status: Harbor trials with the oracle / nop agents and Codex trajectories with a
real model run on boat sandboxes on both the Miles and the verl agent paths.
A full training run on either backend: not verified.

## Who does what

| Part | Owner | Code |
|---|---|---|
| Task loading, sandbox (provider), agent install/launch, tool exec, verifier, teardown | Harbor | `harbor.trial.Trial` |
| Agent | Harbor's built-in `codex` agent, build pinned to `CODEX_VERSION` (`pins.py`) | Harbor `agents/installed/codex.py` |
| Neutral types `AgentTask` / `TrialVerdict` / `AgentTrajectory` | yeto core | `yeto/rl/agentic/types.py` |
| Outer layer: admission (bounded sandboxes, drain, policy version), OK / POLICY_LIMIT / ABORTED, trust boundary, metrics | yeto core | `yeto/rl/agentic/{admission,outcome,trust,runner}.py` |
| Harbor edge: `AgentTask` -> `TrialConfig`, `TrialResult` -> `TrialVerdict` | yeto | `trial.py` |
| Miles backend: Miles' example + session server; yeto agent function and reward hook | Miles + yeto | `yeto/rl/adapters/miles/harbor_agent.py` |
| verl backend: `HarborAgentLoop(AgentLoopBase)`, registered as `yeto_harbor` | yeto (verl unchanged) | `yeto/rl/adapters/verl/harbor_agent_loop.py` |
| Token recorder in front of the sampler | yeto | port: `yeto/rl/agentic/recorder.py`; records proxy: `yeto/rl/harness/gateway/` |

## Outcome rule

| Harbor result | Outcome | Reward | Trained |
|---|---|---|---|
| verifier result, no exception | OK | verifier `reward` | yes |
| `AgentTimeoutError`, `MaxSeqLenExceededError`, `SingleTurnMaxSeqLenExceededError`, context/output length errors | POLICY_LIMIT | verifier reward if it ran, else 0 | yes |
| any other exception, no verifier result, wall-clock cap, admission draining, token misalignment (verl) | ABORTED | none | no |

Miles' example scores the ABORTED cases 0 (`AgentError`); yeto does not.
Configuration errors and trust violations raise (the run stops).

## Pinned Harbor

`pins.py` pins a commit (`HARBOR_COMMIT`) of the `harbor-miles-v0.20.0` branch
(package version 0.20.0); the branch moves, so the pin is a commit, not the
branch name. Install it in the rollout image:

```bash
uv pip install "harbor[modal] @ git+https://github.com/harbor-framework/harbor@<HARBOR_COMMIT>"
```

`installed_problem()` reports an installed version or commit that differs from
the pin.

### harbor-miles-v0.20.0 vs 0.24.0

- Only on the branch (absent from upstream 0.24.0): terminus-2
  `response_length_exceeded_policy`, `MaxSeqLenExceededError` /
  `SingleTurnMaxSeqLenExceededError`, the `max_seq_len` check in `Chat`, and
  mini-swe-agent `poll_steps`; also the agent server, critic agents, S3 upload,
  a docker egress-control sidecar and Daytona auto-stop. Miles' TITO path needs
  the first four for terminus-2; `codex` does not use them.
- In both: the built-in `codex` agent; `OPENAI_BASE_URL` is copied into
  `$CODEX_HOME/config.toml` as `openai_base_url`; `extra_allowed_hosts`,
  CPU/memory enforcement policies, `override_storage_mb`, `Trial.create` /
  `Trial.run`, `TrialResult.exception_info` / `verifier_result`.
- Only in 0.24.0: a codex agent rewrite (user `config.toml` / inline JSON merged
  with the base URL, ACP mode, rollout-to-ATIF conversion), a larger Modal
  provider, `TrialConfig` and verifier changes, and more `EnvironmentType`
  values (both have `modal`, `docker`, `ack`, `opensandbox`).
- Consequence: stay on the branch commit for Miles (TITO). Moving to 0.24.0
  means re-porting those four patches, or using only agents that do not need
  them (such as codex).

## Run (Miles backend)

1. Image: install the pinned Harbor (above) next to Miles; Miles' example
   directories on the worker PYTHONPATH (`launch_common.agentic_pythonpath_dirs()`).
2. Tasks: one Harbor task directory per `metadata.instance_id`, on a path all
   workers read (`HARBOR_TASKS_DIR`).
3. Environment: `HARBOR_ENV_TYPE` (a Harbor environment type, or `boat`),
   provider credentials as Miles' sandbox-provider guide describes,
   `AGENT_TRIAL_TIMEOUT` (outer cap, default 7200 s) above `AGENT_TIMEOUT`,
   `YETO_HARBOR_MAX_IN_FLIGHT` (sandboxes per worker, default 64).
4. Flags: `harbor_agent.train_args(tito_model=..., session_server_workers=...,
   session_server_external_host=<address the sandboxes reach>)` - Miles'
   `agentic_train_args` with the agent function and reward hook swapped
   (a CPU test checks the rest is identical).
5. Network: the sandbox must reach the session-server ports and the SGLang
   router (31000), as in Miles' example README.

## Run (verl backend)

Data row: `agent_name: yeto_harbor`, `extra_info.agent_task: {task_id, agent,
provider, ...}`. Worker env: `YETO_AGENTIC_RECORDER=module:callable`,
`HARBOR_TASKS_DIR`. Known gap: verl has no per-sample drop, so an ABORTED
sample is a mask-0 placeholder with reward 0; it still counts in its GRPO
group mean. The verl launcher and island entry refuse the Miles agentic flags
(`--custom-agent-function-path`, `--use-session-server`, `--tito-model`, ...).

boat on verl: `HARBOR_ENV_TYPE=boat` (or `provider: boat` in the row; the row
wins), `HARBOR_ENV_KWARGS='{"machine_type": "small"}'` (optional). Each of
`rollout.agent.num_workers` worker processes admits up to
`YETO_HARBOR_MAX_IN_FLIGHT` sandboxes and refuses a value above
`YETO_BOAT_MAX_ACTIVE`; keep `num_workers * YETO_BOAT_MAX_ACTIVE` within the
plan. The admission gate and the boat quota are kept per event loop
(`yeto/rl/agentic/loop_local.py`); verl's `AgentLoopWorker` is a Ray async
actor, so that is one per worker process. verl builds the agent loop once per
sample: `YETO_AGENTIC_RECORDER` should return a shared recorder.

## boat sandbox provider

`provider: boat` maps to `EnvironmentConfig.import_path =
"yeto.rl.harness.harbor.boat_env:BoatEnvironment"` (Harbor unchanged).
Code: `boat_client.py` (official `boat-sdk`, quota guard), `boat_sandbox.py`
(VM + task container, no Harbor import), `boat_env.py` (Harbor
`BaseEnvironment`). Install: `pip install boat-sdk`. Credentials:
`BOAT_API_KEY`, else the boat CLI token in `~/.config/ascii/boat/config.json`.

- One boat VM per environment; the task's prebuilt `docker_image` runs as
  container `main` on the Docker bridge network; exec = `docker exec` through
  boat's detached command API (polled, no 600 s cap). Files go through a
  signed URL to `/tmp/yeto-xfer` on the VM, then `docker cp`; directories as
  one tar.gz.
- The VM is deleted on `stop` (also for `delete=False`), on any `start`
  failure and after cancellation; `ttl_s` (default 3 h) is the backstop.
- Machine type: the smallest that fits the task (`small` = 2 vCPU / 3 GiB).
  Refused: Dockerfile-only or compose tasks, GPU/TPU, Windows, network modes
  other than public.
- Quota per process (`BoatQuota`): `YETO_BOAT_MAX_ACTIVE` (default 100) live
  VMs, `YETO_BOAT_STARTS_PER_MINUTE` (default 12) creates. Several processes
  share one account: split the caps. `check_in_flight_cap()` refuses an
  admission gate above the active cap. Hourly/daily start caps are left to the
  server.
- `docker exec -e` values land in boat's command history: pass no secrets as
  exec env.
- Miles backend: the Miles example maps `HARBOR_ENV_TYPE` to Harbor's
  `EnvironmentType` and rejects `boat`. The yeto agent function
  (`adapters/miles/harbor_agent.py`) therefore builds the `TrialConfig` for
  `CUSTOM_PROVIDERS` with the verl builder (`build_trial_config`,
  `HarborSettings.from_env()`); Harbor's built-in providers keep the
  example's builder. The example's harness bindings (terminus-2 TITO
  kwargs, claude-code, opencode) do not apply on boat: only agents that
  `agent_env` wires (OpenAI base URL: codex, oracle, nop) run there.

`scripts/harbor_boat_trial.py` runs Harbor trials on boat without a model
endpoint (`oracle` scores 1, `nop` scores 0 on a solvable task).

## Codex + records proxy

The pinned Codex build refuses `wire_api = "chat"`: it only speaks the
Responses API, so the records proxy (Responses -> Chat, `yeto/rl/harness/gateway/`)
is its path to a chat-template engine. Codex first tries a WebSocket on
`/v1/responses` (the proxy answers 405), then uses HTTP POST.

What the yeto side sets for Codex: `reasoning_summary=none` and
`web_search=disabled` (`trial.CODEX_KWARGS`); the proxy leaves Codex's
`namespace` (sub-agent) tools out of the prompt; the `qwen38` template merges
Codex's two leading system messages (instructions + developer message).

`scripts/modal_records_endpoint.py` serves a model with SGLang behind the
records proxy on Modal (tunnel plus a secret path prefix) for Codex trials. On
both the Miles path (agent function + reward hook, proxy in place of the
session server) and the verl path (`HarborAgentLoop.run`, HTTP recorder over
the proxy), every recorded turn re-tokenizes to the same ids as the stream.
