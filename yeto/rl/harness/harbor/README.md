# Harbor harness for agentic RL (S25 N10.1-N10.4)

Status: Harbor trials with the oracle/nop agents ran on boat sandboxes (see
"boat sandbox provider"). **No model-driven (codex) trajectory has run yet
(not verified).**

## Who does what

| Part | Owner | Code |
|---|---|---|
| Task loading, sandbox (provider), agent install/launch, tool exec, verifier, teardown | Harbor | `harbor.trial.Trial` |
| Agent | Harbor built-in `codex` agent (N10.7), build pinned to `CODEX_VERSION` | `agents/installed/codex.py` |
| Neutral types `AgentTask` / `TrialVerdict` / `AgentTrajectory` | yeto core | `yeto/rl/agentic/types.py` |
| Outer layer: admission (bounded sandboxes, drain, policy version), OK / POLICY_LIMIT / ABORTED, trust boundary, metrics | yeto core | `yeto/rl/agentic/{admission,outcome,trust,runner}.py` |
| Harbor edge: `AgentTask` -> `TrialConfig`, `TrialResult` -> `TrialVerdict` | yeto | `trial.py` |
| Miles backend: Miles' example + session server; yeto agent function and reward hook | Miles + yeto | `yeto/rl/adapters/miles/harbor_agent.py` |
| verl backend: `HarborAgentLoop(AgentLoopBase)`, registered as `yeto_harbor` | yeto (verl unchanged) | `yeto/rl/adapters/verl/harbor_agent_loop.py` |
| Token recorder in front of the sampler (verl) | yeto, N10.5 (not written) | port: `yeto/rl/agentic/recorder.py` |

The self-written Codex harness (`yeto/rl/harness/codex/`) is not deleted (user
decision pending). New code does not import it.

## Outcome rule

| Harbor result | Outcome | Reward | Trained |
|---|---|---|---|
| verifier result, no exception | OK | verifier `reward` | yes |
| `AgentTimeoutError`, `MaxSeqLenExceededError`, `SingleTurnMaxSeqLenExceededError`, context/output length errors | POLICY_LIMIT | verifier reward if it ran, else 0 | yes |
| any other exception, no verifier result, wall-clock cap, admission draining, token misalignment (verl) | ABORTED | none | no |

Miles' example scores the ABORTED cases 0 (`AgentError`); yeto does not.
Configuration errors and trust violations raise (the run stops).

## Pinned Harbor

`pins.py`: branch `harbor-miles-v0.20.0` at commit
`8f88560ce9daae9b4cc0f6ff7bac8779ae9cfa15` (branch head 2026-10-19; package
version 0.20.0). Install in the rollout image:

```bash
uv pip install "harbor[modal] @ git+https://github.com/harbor-framework/harbor@8f88560ce9daae9b4cc0f6ff7bac8779ae9cfa15"
```

`installed_problem()` reports a version or commit that differs from the pin.

### harbor-miles-v0.20.0 vs 0.24.0

Read from git (`v0.20.0`, `v0.24.0` tags and the branch), 2026-10-19.

- The branch is upstream `v0.20.0` plus 50 commits (33 files under `src/`,
  +4278/-226). Upstream `v0.24.0` is 412 commits after the common base
  (231 files, +47279/-6850).
- Only on the branch (absent from 0.24.0, grep count 0): terminus-2
  `response_length_exceeded_policy`, `MaxSeqLenExceededError` /
  `SingleTurnMaxSeqLenExceededError`, the `max_seq_len` check in `Chat`,
  mini-swe-agent `poll_steps`; also the agent server, critic agents, S3
  upload, docker egress-control sidecar, Daytona auto-stop. Miles' TITO path
  needs the first four for terminus-2; `codex` does not use them.
- In both: built-in `codex` agent; `OPENAI_BASE_URL` is copied into
  `$CODEX_HOME/config.toml` as `openai_base_url`; `extra_allowed_hosts`,
  CPU/memory enforcement policies, `override_storage_mb`, `Trial.create` /
  `Trial.run`, `TrialResult.exception_info` / `verifier_result`.
- Only in 0.24.0: codex agent rewrite (+936 lines: user `config.toml` / inline
  JSON merged with the base URL, ACP mode, rollout-to-ATIF conversion); Modal
  provider +286 lines; `TrialConfig` +128 lines; verifier +34 lines;
  `EnvironmentType` has 31 values vs 23 on the branch (both have `modal`,
  `docker`, `ack`, `opensandbox`).
- Neither version sets Codex `wire_api`. Whether the pinned Codex build can
  talk Chat Completions to the Miles session server (which has no Responses
  API) is **not verified**; if not, the N10.5 recorder must translate
  Responses -> Chat.
- Consequence: stay on the branch commit for Miles (TITO). Moving to 0.24.0
  means re-porting those four patches (or using only agents that do not need
  them, such as codex; not checked against the Miles session server).

## Run (Miles backend, not verified)

1. Image: install the pinned Harbor (above) next to Miles; Miles' example
   directories on the worker PYTHONPATH (`launch_common.agentic_pythonpath_dirs()`).
2. Tasks: one Harbor task directory per `metadata.instance_id`, on a path all
   workers read (`HARBOR_TASKS_DIR`); e.g. `harbor download terminal-bench@2.0`
   (command name not checked).
3. Environment: `HARBOR_ENV_TYPE=modal`, Modal credentials as Miles'
   `docs/user-guide/sandbox-providers.md` describes, `AGENT_TRIAL_TIMEOUT`
   (outer cap, default 7200 s) above `AGENT_TIMEOUT`,
   `YETO_HARBOR_MAX_IN_FLIGHT` (sandboxes per worker, default 64).
4. Flags: `harbor_agent.train_args(tito_model=..., session_server_workers=...,
   session_server_external_host=<address the sandboxes reach>)` - Miles'
   `agentic_train_args` with the agent function and reward hook swapped
   (a CPU test checks the rest is identical).
5. Network: the sandbox must reach the session-server ports and the SGLang
   router (31000), as in Miles' example README.

Acceptance (N10.1, open): one TB2 trajectory end to end, reward equal to the
existing harness on the same task.

## Run (verl backend, not verified)

Data row: `agent_name: yeto_harbor`, `extra_info.agent_task: {task_id, agent,
provider, ...}`. Worker env: `YETO_AGENTIC_RECORDER=module:callable` (N10.5),
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

Verified 2026-10-19 (T-N108C): `HarborAgentLoop.run()` (verl agent-loop
module stubbed, configuration from the env variables above) on boat, fix-git
and regex-log: oracle reward 1 on both, nop reward 0 on both, outcome ok;
41 billable s, `boat list` empty afterwards. The recorder returned one
synthetic turn (oracle/nop call no model). A verl training run: not verified.

## boat sandbox provider (N10.8)

`provider: boat` maps to `EnvironmentConfig.import_path =
"yeto.rl.harness.harbor.boat_env:BoatEnvironment"` (Harbor unchanged).
Code: `boat_client.py` (official `boat-sdk` 1.10.0, quota guard),
`boat_sandbox.py` (VM + task container, no Harbor import), `boat_env.py`
(Harbor `BaseEnvironment`). Install: `pip install boat-sdk==1.10.0`.
Credentials: `BOAT_API_KEY`, else the boat CLI token in
`~/.config/ascii/boat/config.json`.

- One boat VM per environment; the task's prebuilt `docker_image` runs as
  container `main` on the Docker bridge network; exec = `docker exec` through
  boat's detached command API (polled, no 600 s cap). Files go through a
  signed URL to `/tmp/yeto-xfer` on the VM, then `docker cp`; directories as
  one tar.gz.
- The VM is deleted on `stop` (also for `delete=False`), on any `start`
  failure and after cancellation; `ttl_s` (default 3 h) is the backstop.
- Machine type: smallest that fits the task (`small` = 2 vCPU / 3 GiB, half
  rate). Refused: Dockerfile-only or compose tasks, GPU/TPU, Windows,
  network modes other than public.
- Quota per process (`BoatQuota`): `YETO_BOAT_MAX_ACTIVE` (default 100) live
  VMs, `YETO_BOAT_STARTS_PER_MINUTE` (default 12) creates. Several processes
  share one account: split the caps. `check_in_flight_cap()` refuses an
  admission gate above the active cap. Hourly/daily start caps (60/200) are
  left to the server.
- `docker exec -e` values land in boat's command history (90 days): pass
  no secrets as exec env.
- Miles backend: the Miles example maps `HARBOR_ENV_TYPE` to Harbor's
  `EnvironmentType` and rejects `boat`. The yeto agent function
  (`adapters/miles/harbor_agent.py`) therefore builds the `TrialConfig` for
  `CUSTOM_PROVIDERS` with the verl builder (`build_trial_config`,
  `HarborSettings.from_env()`); Harbor's built-in providers keep the
  example's builder. The example's harness bindings (terminus-2 TITO
  kwargs, claude-code, opencode) do not apply on boat: only agents that
  `agent_env` wires (OpenAI base URL: codex, oracle, nop) run there.

Verified 2026-10-19 (`scripts/harbor_boat_trial.py`, Harbor `oracle` and
`nop` agents, no model endpoint): 6 TB2 tasks (fix-git, regex-log,
sqlite-db-truncate, log-summary-date-ranges, openssl-selfsigned-cert,
git-multibranch): oracle reward 1 on all 6, nop reward 0 on all 6, the same
as the earlier Modal record of the self-written harness (official solutions
1, empty 0). 15-36 s per trial; 102 billable s for the final 12 trials.
`boat list` empty afterwards. Codex + model endpoint on boat: not verified.

Miles path, 2026-10-19: the Miles agent function
(`harbor_agent.run`, with the real Miles example module and Harbor, no
rollout worker or session server) plus `reward_func`, `HARBOR_ENV_TYPE=boat`:
fix-git and regex-log, oracle reward 1 on both, nop reward 0 on both
(outcome ok); 30 billable s. A full Miles training run on boat: not verified.
