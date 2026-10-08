# 会话服务协议（decoupling 任务 3.8，spec `rl-neutral-rewards-harness`）

codex harness 对"会话服务"的全部依赖。Miles 会话服务是这个协议的一种实现；以后换框架（如 verl）时，只要提供同样的五条路径，harness 不用改。测试替身：`tests/session_service_double.py`。

本协议只写文档与测试替身，**没有**实现 verl 侧。

## 路径

`{router}` 是会话服务地址；harness 拿到的会话地址是 `{router}/sessions/{session_id}`（`codex_openenv_agent_function.split_session_url` 拆分它）。

| 操作 | 请求 | 成功回复 | 失败 |
|---|---|---|---|
| 创建会话 | `POST {router}/sessions`，JSON 体 `{"evaluation": bool, "temperature"?: float, "top_p"?: float, "top_k"?: int}`（空体等于 `{"evaluation": false}`） | 200，`{"session_id": "<id>"}`，id 中不含 `/` | 请求体不合法回 400 |
| 会话内聊天 | `POST {router}/sessions/{id}/v1/chat/completions`，OpenAI 兼容聊天请求体 | 200，OpenAI 兼容回复；服务端**强制** `logprobs=true`，每个 choice 带 `meta_info.output_token_logprobs`（`[logprob, token_id, ...]` 列表，长度 = 回复词元数） | 会话不存在回 404 |
| 取会话样本 | `POST {router}/sessions/{id}/samples`，JSON 体 `{"max_seq_len": int\|null, "metadata"?: object}` | 200，服务端拼好的训练样本（Miles 用自有编码；harness 只经 `collect_segment_session` 读取，不解析字节） | 会话不存在回 404；拼装校验失败回 422 |
| 查看会话 | `GET {router}/sessions/{id}` | 200，`{"session_id", "records": [...], "metadata": {...}}`（调试用；`metadata` 可含 `tito_session_mismatch`） | 404 |
| 删除会话 | `DELETE {router}/sessions/{id}` | 204（harness 也接受 200） | 已收走或已删除回 404，harness 视为成功 |

harness 的约定：

- 创建会话在可信层（`create_segment_sessions`）完成，不可信的 worker 只能用本轮次自己的会话；中途失败把已建的会话删掉。
- 取样本后会话即删除（Miles 的 `OpenAIEndpointTracer.collect_samples` 在 `finally` 里发 DELETE）；未被取走的会话由 `delete_segment_sessions` / `release_unreturned_segments` 尽力删除，删除失败不覆盖原错误。
- harness 只依赖本协议、核心的轮次元数据接口与奖励结果（`yeto.rl.rewards.types.RewardResult`）。

## 核实情况

以 Miles ports 路径 pin `MILES_NEXT_COMMIT = c35702eefcf2862cee155e46870e6ad30568d2c6`（`yeto/rl/__init__.py`）的源码核实（本地仓库 `/home/michael/work/miles-2b` 的 git 对象，2026-10-08）：

| 项 | 文件:行（pin 提交） | 状态 |
|---|---|---|
| `POST /sessions` 读 `CreateSessionRequest`（evaluation/temperature/top_p/top_k），体不合法 400 | `miles/rollout/session/sessions.py:104-113`，`types.py:6-11` | 已核实 |
| 创建回 `{"session_id"}`，200 | `core.py:248-254` | 已核实 |
| `POST /sessions/{id}/v1/chat/completions` | `sessions.py:123` | 已核实 |
| 强制 `logprobs=True`，回复须带 `meta_info.output_token_logprobs` 且长度等于回复词元数 | `request_args.py:121`，`core.py:178-199` | 已核实 |
| `POST /sessions/{id}/samples`，体 `max_seq_len`/`metadata` | `sessions.py:225-235` | 已核实 |
| 拼装校验失败 422 | `core.py:281-285`（文档字符串） | 已核实（只读了文档字符串） |
| `GET /sessions/{id}` 回 `GetSessionResponse{session_id, records, metadata}` | `sessions.py:115`，`types.py:57-60` | 已核实 |
| `DELETE /sessions/{id}` 回 204；会话不存在回 404 | `core.py:318-326`，`errors.py:21-24` | 已核实 |
| 取样本后客户端发 DELETE | `generate_utils/openai_endpoint_utils.py:82-100` | 已核实 |
| 另有 `POST /sessions/{id}/v1/messages`（Anthropic 兼容）与 `/sessions/{id}/{path}` 透传 | `sessions.py:135,237` | 已核实存在；codex harness 不用，不属本协议 |
| 流式聊天（SSE）回复格式 | — | **未核实**；替身只实现非流式 JSON |

注意：legacy pin `MILES_COMMIT = ae475060…` 的会话服务较旧（`POST /sessions` 不读请求体、无 `/samples` 路径，`miles/rollout/session/sessions.py:51-60`）；本协议以 ports pin 为准，legacy 路径是否用到会话服务未核实。
