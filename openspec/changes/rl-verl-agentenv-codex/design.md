# Design

## Context

Verified from code (2026-10-12):

- verl fork `/home/michael/work/verl-fork-acad9875` (HEAD acad9875, 2026-10-08):
  - The only sandbox capability is Sandbox Fusion. The client is `call_sandbox_api` at `verl/utils/reward_score/sandbox_fusion/utils.py:67`. It POSTs JSON to `reward.sandbox_fusion.url` (which must end with `/run_code`). Fields: compile_timeout, run_timeout, code, stdin, memory_limit_MB, language, files, fetch_files. A 504 is retried. The config items are at `verl/trainer/config/reward/reward.yaml:88` (url, max_concurrent, memory_limit_mb).
  - On the tool side, `verl/tools/` has only base_tool, function_tool, schemas and tool_registry. The in-tree `SandboxFusionTool` was deleted upstream. `docs/sglang_multiturn/sandbox_fusion.rst` keeps the integration method. The reference implementation is in upstream `f5e21df6` and earlier.
  - Agent loop: `verl/experimental/agent_loop/agent_loop.py:227 AgentLoopBase`, abstract `run` at :466; `register(agent_name)` at :487; the YAML tool config goes through `rollout_config.agent.agent_loop_config_path` (:551); a sample selects its loop by `non_tensor_batch["agent_name"]` (:627). `tool_agent_loop.py:101` registers `tool_agent`.
  - `AgentLoopOutput` (:88) fields: prompt_ids, response_ids, response_mask, response_logprobs, num_turns, metrics, extra_fields, and others.
  - No hit for agentenv or e2b.
- yeto worktree `/home/michael/work/s22-agentic-sft`:
  - `yeto/rl/adapters/verl/` already has a full backend (entry, verl_main, trainer, patch_verl, publish, reward_fn, rollout_events, pins, and others).
  - `yeto/rl/harness/codex/tb2_provider.py:349-361` has the `SandboxHandle` / `SandboxBackend` protocols and two implementations, LocalProcess and Modal. This is the sandbox abstraction of the Miles-side TB2, not of verl.
  - The spec `rl-harness-sandbox-broker` explicitly delivers only the interface and a fake implementation. `rl-tito-session-service` (`rl-verl-backend` task 1.9) describes the token session service of the verl backend, and it is not implemented.
- AgentENV: Firecracker microVM, one VM per sandbox, E2B compatible SDK. It requires `/dev/kvm` and kernel >= 6.8. Known issues: #20 (ublk ADD_DEV regression), #320, #215, #180, #138. None of the official performance numbers has been measured.
- cherryservers: single-tenant bare metal, outlet line from EUR 49/month, with Ubuntu 24.04 (kernel 6.8, exactly at the limit). The exact models and prices load dynamically on the page, so a person must look.

## Goals / Non-Goals

Goals:
1. verl uses AgentENV with zero change (both the reward side and the tool side use the Sandbox Fusion protocol).
2. One cherryservers host goes from bare metal to "sandbox ready" in a reproducible way.
3. Stage 1 runs one training step of `tool_agent` + sandbox tool + sandbox reward.
4. Stage 2 (Codex in the sandbox) goes only as far as "can measure the token mismatch rate".

Non-Goals: rewriting the session server (the `rl-tito-session-service` spec already exists); changing the Miles backend or the TB2 provider; connecting AgentENV to the yeto `SandboxBackend` protocol (that is the Miles/TB2 path, and this week verl runs natively); Ascend; load test conclusions.

## Decisions

1. **Choose the Sandbox Fusion protocol, not an E2B client written for verl.** Reason: the user requires reuse of the sandbox capability that verl already has. The verl reward path and the tool docs recognize only this one protocol. The gateway is a small HTTP service, and it changes less than adding a new class of client inside verl. Cost: the gateway must map the one-shot `/run_code` semantics of Sandbox Fusion to the "long-lived sandbox" model of AgentENV (see decision 3).
2. **Put the gateway on the cherryservers host, on the same machine as AgentENV.** It uses the E2B Python SDK against the local AgentENV (`E2B_API_URL` points to the local machine). It is a single-file FastAPI/aiohttp service at `tools/agentenv/gateway.py`. It does not go into the yeto package, to avoid the cloud-layer/adapter-layer rules of `test_import_boundaries.py`.
3. **Two sandbox lifetimes**:
   - Reward-side `/run_code`: one sandbox per request, destroyed right after execution (Sandbox Fusion itself has stateless semantics).
   - Tool side: `create` of the `agentenv_sandbox` tool asks the gateway for a sandbox with a `session_id`. `execute` calls `/run_code` with the `session_id`. `release` calls DELETE `/sessions/{id}`. For a request without `session_id`, the gateway falls back to the one-shot semantics, so the verl reward side does not notice the extension. `session_id` is an extension as a request field, and it does not break the protocol.
4. **Tool code location**: first choice is the yeto side, `yeto/rl/adapters/verl/tools/agentenv_sandbox.py`. It is loaded by its fully qualified `class_name` through the verl `agent_loop_config_path` YAML, and the fork is not changed. If the verl tool loading requires the module to be under the `verl.tools` namespace, put it in the fork. At implementation, read `tool_registry.py` to decide.
5. **Host bootstrap is a script, not manual work**: `tools/agentenv/bootstrap.sh` (check kvm/kernel -> install AgentENV -> hello world -> write the ready record `/var/lib/yeto-agentenv/ready.json`). All numbers (kernel, version, latency) go into the record file. Later reports cite the record file, not oral statements.
6. **The Codex stage uses a custom AgentLoop, not BaseTool.** Reason as in section 2.3 of the P1 document: Codex has its own loop, and the model does not call it as a tool. The class `CodexAgentENVLoop(AgentLoopBase)` is registered as `codex_agentenv`. In `run`: create a session sandbox, start Codex in the sandbox pointing at the rollout endpoint, wait for the end, score through the Sandbox Fusion reward path, release, re-tokenize and compare with the tokens of the inference side, then build the `AgentLoopOutput` or mark the episode as dropped. The mismatch rate is written to `metrics`.
7. **Token ownership: option B first, then option A.** B = re-tokenize + check and drop, fast to build. A = connect to `rl-tito-session-service`. A is not in this change. The mismatch rate decides whether to open a new change. Until the rate is measured, B must not be declared usable.
8. **Egress policy**: AgentENV sandboxes have no network by default. In the Codex stage, the gateway allowlist (YAML) opens only the rollout endpoint and an optional auth injection. A malformed allowlist makes startup fail.

## Risks / Trade-offs

- Kernel at the 6.8 limit and issue #20: install and verify first. If it fails, use the HWE kernel. If HWE also fails, change to a cherryservers virtualization-capable line or other bare metal. This is a blocking risk, and it is placed in task 1.
- Sandbox Fusion result format: the fields that the verl client parses follow `_process_single_case` in `utils.py`. At implementation, write the response against it, not from memory.
- Gateway concurrency on one host: the official memory overcommit of 9.6x for AgentENV has not been measured. The gateway limits concurrency with a semaphore and returns 503/504 to verl so that it retries. It does not let the host run out of memory.
- Stage 1 end-to-end needs verl rollout cards: approve the budget separately under the minimum-card, cheap-card rule (Modal by default). Do not mix it into the sandbox chain validation.
- cherryservers model and price are unknown: after a person reads the page, write the model, price and location into the record of tasks 1.1.

## Migration Plan

No existing path changes. This is purely additive. Order: host bootstrap -> gateway + direct reward-side connection -> tool + Stage 1 step -> Codex image + custom loop + mismatch rate. Each step has an artifact that can be checked on its own (ready record, gateway tests, training log, mismatch rate report).

## Open Questions

- Does verl `tool_registry.py` allow loading tool classes outside `verl.tools` (decision 4)?
- The exact form of the AgentENV v0.2.3 "native Codex sandbox session". If it already provides Codex startup and auth injection, the image preinstall work of Stage 2 can shrink.
- Can the requests from Codex to an OpenAI-compatible endpoint carry back the token IDs of the inference side? If the verl rollout endpoint supports a `return_token_ids` style field, the check of option B can use it directly and does not need to re-tokenize.
