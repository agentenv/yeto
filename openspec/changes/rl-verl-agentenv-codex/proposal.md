# Proposal

## Why

On 2026-10-12 the user decided: agentic RL uses verl with the AgentENV sandbox that is self-hosted on cherryservers bare metal, and the agent is Codex. The user also requires reusing the sandbox capability that verl already has, with no reinvented wheel. Current state (checked by the main agent by reading code on 2026-10-12, verl fork `/home/michael/work/verl-fork-acad9875` HEAD acad9875):

- verl has only one built-in sandbox protocol: **Sandbox Fusion**. On the reward side, `reward.sandbox_fusion.url` points to a `/run_code` HTTP endpoint (`verl/utils/reward_score/sandbox_fusion/utils.py:67`). On the tool side, the same endpoint is used through the `BaseTool` create/execute/calc_reward/release lifecycle (the in-tree `SandboxFusionTool` was deleted upstream, and the docs keep the wiring, `docs/sglang_multiturn/sandbox_fusion.rst:9`).
- verl has no AgentENV or E2B integration (0 grep hits). AgentENV declares only E2B compatibility to the outside.
- So the correct way to "reuse the verl sandbox capability" is: **make AgentENV look like a Sandbox Fusion endpoint to verl.** verl gets no new sandbox client. Reward execution and tool execution both use the existing verl paths.
- yeto already has the `rl-harness-sandbox-broker` spec (delivers only the interface and a fake implementation) and the `rl-tito-session-service` spec (the token session service of the verl backend, `rl-verl-backend` task 1.9, not implemented yet). This change does not redo these two.

## What Changes

- **Add a cherryservers host bootstrap flow** (ops script + checklist): open an outlet bare-metal server, check `/dev/kvm`, check kernel >= 6.8 (use an HWE kernel if needed), install AgentENV, and run a hello world with the E2B SDK. This step does not depend on yeto code.
- **Add a Sandbox Fusion compatible gateway**: a small HTTP service that runs on the cherryservers host. It implements the `/run_code` request semantics (code, language, stdin, compile_timeout, run_timeout, memory_limit_MB, files, fetch_files) and maps them to AgentENV sandboxes (E2B SDK). The verl `reward.sandbox_fusion.url` and the tool side point to it directly.
- **Add a verl tool `agentenv_sandbox`**: implemented with the verl `BaseTool` lifecycle. `execute` calls the gateway above. It is registered in the `tool_agent` AgentLoop. In this case, where the model calls the sandbox itself, no token ownership bridge is needed. verl records the tokens natively.
- **The Codex-as-agent path has two stages**:
  - Stage 1 (delivered by this change): the model completes tasks by calling the `agentenv_sandbox` tool through verl `tool_agent`. Codex does not take part. The purpose is to get the cherryservers + AgentENV + verl chain working first.
  - Stage 2 (this change only starts it and does not accept it): preinstall the Codex CLI in the sandbox image. Write a custom AgentLoop subclass that starts Codex inside the sandbox and points it at the verl rollout endpoint. Token ownership first uses "re-tokenize + consistency check, drop on mismatch" and **measures the mismatch rate**. If the mismatch rate is not acceptable, connect to the existing `rl-tito-session-service` spec (the yeto session service). Do not build another session server.
- Load testing and the known AgentENV issues (image cache is never evicted #320, cold-start latency #215, warm pool does not shrink #180, network changes interrupt P2P #138) are observed after stage 1 works. The measured results are recorded.

## Capabilities

### New Capabilities
- `rl-verl-agentenv-sandbox`: verl uses AgentENV sandboxes on cherryservers through the Sandbox Fusion protocol. It covers host bootstrap checks, gateway semantics, tool lifecycle, per-episode isolation and destruction, and the token consistency measurement of the Codex stage.

### Modified Capabilities
(None. The requirements of `rl-harness-sandbox-broker`, `rl-tito-session-service` and `rl-verl-engine-backend` do not change. This change is a new integration surface outside them.)

## Impact

- New code: `tools/agentenv/` (host bootstrap script, gateway service), `yeto/rl/adapters/verl/tools/agentenv_sandbox.py` (or in `verl/tools/` of the verl fork; decided at implementation by the boundary rules of `test_import_boundaries.py`), and the verl-side agent loop config YAML.
- Dependencies: the E2B Python SDK (`E2B_API_URL` / `E2B_SANDBOX_URL` point to the self-hosted AgentENV). The verl fork does not need an upgrade. If the tool is put into the fork, the fork has one more file.
- Cost: at least one cherryservers outlet server (in the range of EUR 49/month; the exact model and price need a person to open the pricing page and confirm). Stage 1 sandbox chain verification uses no GPU. The stage 1 end-to-end training check needs a verl rollout card, which is approved separately under the GPU minimum-card rule.
- No impact on the Miles backend or the existing TB2 provider.
