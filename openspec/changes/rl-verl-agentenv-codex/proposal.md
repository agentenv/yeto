# Proposal

## Why

On 2026-10-12 the user decided: agentic RL uses verl with a self-hosted AgentENV sandbox on cherryservers bare metal, and the agent is Codex. The user also requires reuse of the sandbox capability that verl already has, with no reinvented wheel. Current state (verified by the main agent from code on 2026-10-12, verl fork `/home/michael/work/verl-fork-acad9875` HEAD acad9875):

- The only sandbox capability built into verl uses one protocol: **Sandbox Fusion**. On the reward side, `reward.sandbox_fusion.url` points to a `/run_code` HTTP endpoint (`verl/utils/reward_score/sandbox_fusion/utils.py:67`). On the tool side, the same endpoint is used through the `BaseTool` create/execute/calc_reward/release lifecycle (the in-tree `SandboxFusionTool` was deleted upstream, and the docs keep the integration method, `docs/sglang_multiturn/sandbox_fusion.rst:9`).
- verl has no AgentENV or E2B integration (grep: 0 hits). AgentENV only declares E2B compatibility to the outside.
- So the correct way to "reuse the verl sandbox capability" is: **make AgentENV look like a Sandbox Fusion endpoint to verl**. No new sandbox client is written on the verl side. Both reward execution and tool execution use the existing verl paths.
- On the yeto side, the `rl-harness-sandbox-broker` spec already exists (it delivers only the interface and a fake implementation). The `rl-tito-session-service` spec also exists (the token session service of the verl backend, `rl-verl-backend` task 1.9, not yet implemented). This change does not redo either of them.

## What Changes

- **Add a cherryservers host bootstrap flow** (ops script + checklist): order an outlet bare-metal server, check `/dev/kvm` and kernel >= 6.8 (use the HWE kernel if needed), install AgentENV, and run a hello world with the E2B SDK. This step does not depend on yeto code.
- **Add a Sandbox Fusion compatible gateway**: a small HTTP service that runs on the cherryservers host. It implements the `/run_code` request semantics (code, language, stdin, compile_timeout, run_timeout, memory_limit_MB, files, fetch_files) and maps them to AgentENV sandboxes (E2B SDK). The verl `reward.sandbox_fusion.url` and the tool side point to it directly.
- **Add the verl tool `agentenv_sandbox`**: it implements the verl `BaseTool` lifecycle, and `execute` calls the gateway above. It is registered in the `tool_agent` AgentLoop. In this way, the case where the model calls the sandbox itself needs no token ownership bridge, because verl records the tokens natively.
- **The path with Codex as the agent has two stages**:
  - Stage 1 (delivered by this change): the model completes tasks by calling the `agentenv_sandbox` tool through verl `tool_agent`. Codex does not take part. The goal is to first make the chain cherryservers + AgentENV + verl work.
  - Stage 2 (this change only starts it and does not accept it): preinstall the Codex CLI in the sandbox image. Write a custom AgentLoop subclass that starts Codex inside the sandbox, pointing at the verl rollout endpoint. For token ownership, first use "re-tokenize + consistency check, drop on mismatch" and **measure the mismatch rate**. If the mismatch rate is not acceptable, connect the existing `rl-tito-session-service` spec (the yeto session service). Do not build another session server.
- Load tests and the known AgentENV issues (image cache is not evicted #320, cold start latency #215, warm pool does not shrink #180, network changes break P2P #138) are observed after Stage 1 works, and the measured results are recorded.

## Capabilities

### New Capabilities
- `rl-verl-agentenv-sandbox`: verl uses the AgentENV sandboxes on cherryservers through the Sandbox Fusion protocol. This covers the host bootstrap check, the gateway semantics, the tool lifecycle, per-episode isolation and destruction, and the token consistency measurement of the Codex stage.

### Modified Capabilities
(None. The requirements of `rl-harness-sandbox-broker`, `rl-tito-session-service` and `rl-verl-engine-backend` do not change. This change is a new integration surface outside them.)

## Impact

- New code: `tools/agentenv/` (host bootstrap script, gateway service), `yeto/rl/adapters/verl/tools/agentenv_sandbox.py` (or in `verl/tools/` of the verl fork; decide at implementation by the boundary rules of `test_import_boundaries.py`), and the verl-side agent loop config YAML.
- Dependencies: the E2B Python SDK (`E2B_API_URL` / `E2B_SANDBOX_URL` point to the self-hosted AgentENV). The verl fork needs no upgrade, but if the tool is placed in the fork, the fork gets one more file.
- Cost: at least one cherryservers outlet server (about EUR 49/month; the exact model and price must be confirmed by a person who opens the pricing page). Stage 1 sandbox chain validation uses no GPU. The Stage 1 end-to-end training validation needs verl rollout cards, which are approved separately under the GPU minimum-card rule.
- No effect on the Miles backend or the existing TB2 provider.
