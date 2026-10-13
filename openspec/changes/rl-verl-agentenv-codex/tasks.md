# Tasks

Commit/PR text is English only. Save all measured numbers to a file first, then cite them (memory gpu-run-value-max / no-fabricated-data). Groups 1-3 of stage 1 can run in parallel with the P0 code quality batch.

## 1. cherryservers host bootstrap (does not depend on yeto code)

- [ ] 1.1 A person opens cherryservers `/pricing/server-outlet`, picks a large-memory outlet machine, and records the model, memory, location and monthly price in the new subsection 2.8 of `infra-drafts/SESSION23/P1-sandbox-verl-agentenv.md`
- [ ] 1.2 Write `tools/agentenv/bootstrap.sh`: check `/dev/kvm`, check `uname -r` >= 6.8 (abort and hint HWE if not met), install AgentENV, E2B SDK hello world (create, run `echo`, kill), write `/var/lib/yeto-agentenv/ready.json`
- [ ] 1.3 Run 1.2 on the machine. If AgentENV issue #20 is triggered, install an HWE kernel and rerun. Record the kernel version, the AgentENV version and the first sandbox start time in ready.json, and copy it back to `infra-drafts/SESSION23/evidence/agentenv-ready.json`
- [ ] 1.4 Write a local unit test `tests/test_agentenv_bootstrap.py`: use fake commands to check that the script aborts when kvm is missing or the kernel is too old

## 2. Sandbox Fusion compatible gateway

- [ ] 2.1 Read `call_sandbox_api` and `_process_single_case` in `verl/utils/reward_score/sandbox_fusion/utils.py`, and copy the request fields and the response parse fields into `tools/agentenv/protocol.md`
- [ ] 2.2 Write `tools/agentenv/gateway.py`: `POST /run_code` (no session_id: one-shot sandbox; with session_id: reuse the session sandbox), `POST /sessions`, `DELETE /sessions/{id}` (a lookup that finds nothing is the reference), `GET /sessions/{id}`; a semaphore limits concurrency and returns 503 above the limit; when run_timeout expires, kill the sandbox and return the timeout status; pass memory_limit_MB to the sandbox
- [ ] 2.3 Egress allowlist: a YAML list of targets, empty by default; a malformed file makes startup fail
- [ ] 2.4 Unit test `tests/test_agentenv_gateway.py`: with a fake E2B client, test the one-shot and session semantics, timeout, concurrency limit, destruction confirmation, and a malformed allowlist
- [ ] 2.5 On the host, connect the official verl Sandbox Fusion sample dataset directly to the gateway (`reward.sandbox_fusion.url=http://<host>:<port>/run_code`) without changing verl code. Record the pass rate and time in `infra-drafts/SESSION23/evidence/gateway-reward-smoke.md`

## 3. verl tool and stage 1 end-to-end

- [ ] 3.1 Read `verl/tools/tool_registry.py` and `base_tool.py` and decide whether the tool class can live in `yeto/rl/adapters/verl/tools/`. If not, put it in the fork as `verl/tools/agentenv_sandbox_tool.py` (record the decision)
- [ ] 3.2 Write the `agentenv_sandbox` tool: create opens a session, execute calls `/run_code` with session_id, calc_reward is optional, release deletes the session and confirms. The tool schema is `{code, language}`
- [ ] 3.3 Write the agent loop YAML (`agent_loop_config_path`) that registers the tool. Dataset samples have `agent_name=tool_agent`
- [ ] 3.4 Unit tests: with a fake gateway, check the create/execute/release order, that two executes in the same episode see the same file, and that a failed release is retried
- [ ] 3.5 Stage 1 step: small model, small batch, tool and reward both point to the gateway, run 1 training step. Save the log, the `num_turns` distribution and the sandbox create/destroy counts to `infra-drafts/SESSION23/evidence/stage1-step/`. The GPU is approved separately under the minimum-card and Modal-default rules. Before the run on GPU, write the review conclusion per memory gpu-review-before-launch

## 4. Codex stage start (only up to measuring the mismatch rate)

- [ ] 4.1 Confirm the form of the "native Codex sandbox session" of AgentENV v0.2.3 (docs/source). Record whether it comes with Codex start and auth injection
- [ ] 4.2 Build a sandbox image with the Codex CLI preinstalled. Let the allowlist open only the verl rollout endpoint. Inside the sandbox, start Codex by hand once against a local fake endpoint, and confirm that it reaches that endpoint and that other targets fail
- [ ] 4.3 Check whether the verl rollout endpoint can return token IDs (`return_token_ids` or equivalent). If it can, the check uses them directly. If it cannot, re-tokenize
- [ ] 4.4 Write `CodexAgentENVLoop(AgentLoopBase)` and register it as `codex_agentenv`: create a session, start Codex, wait for the end, score, release, compare tokens. If they match, produce the `AgentLoopOutput`. If they do not match, drop the episode and count it. The mismatch rate goes into `metrics`
- [ ] 4.5 Unit tests: the match and mismatch paths with a fake endpoint, the drop count, the report output
- [ ] 4.6 Small-scale rollout (can be merged into the same run on GPU as 3.5). Write the total, the dropped count and the mismatch rate to `infra-drafts/SESSION23/evidence/codex-token-consistency.md`. **Do not conclude that "option B is usable".** The mismatch rate decides whether to open a new change that connects `rl-tito-session-service`

## 5. Observe the known AgentENV issues (alongside the runs in 3.5/4.6)

- [ ] 5.1 Record the number of image cache entries over time (#320), the first-command latency (#215), the warm pool size (#180) and the number of P2P interruptions (#138) in the evidence directory. Record only. Draw no conclusion

## 6. Wrap-up

- [ ] 6.1 Update `infra-drafts/SESSION23/P1-sandbox-verl-agentenv.md` with three columns: implemented / verified / unverified
- [ ] 6.2 `openspec validate rl-verl-agentenv-codex --strict` passes
