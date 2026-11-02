# Session service protocol

The Codex harness this file was written for has been removed. The protocol is
kept as the contract of the test double `tests/session_service_double.py`.

This is everything an agent harness needs from a "session service". The Miles
session service is one implementation; another framework (for example verl) can
replace it by serving the same five paths, with no harness change. Only the
documentation and the test double exist; there is **no** verl-side
implementation.

## Paths

`{router}` is the session service address. The session URL the harness receives
is `{router}/sessions/{session_id}`.

| Operation | Request | Success | Failure |
|---|---|---|---|
| Create session | `POST {router}/sessions`, JSON body `{"evaluation": bool, "temperature"?: float, "top_p"?: float, "top_k"?: int}` (an empty body equals `{"evaluation": false}`) | 200, `{"session_id": "<id>"}`; the id contains no `/` | 400 on an invalid body |
| Chat in session | `POST {router}/sessions/{id}/v1/chat/completions`, OpenAI-compatible chat body | 200, OpenAI-compatible response. The server **forces** `logprobs=true`; every choice carries `meta_info.output_token_logprobs` (a `[logprob, token_id, ...]` list whose length equals the number of response tokens) | 404 if the session does not exist |
| Collect samples | `POST {router}/sessions/{id}/samples`, JSON body `{"max_seq_len": int\|null, "metadata"?: object}` | 200, the training sample assembled by the server (Miles uses its own encoding; the harness passes the bytes through without parsing them) | 404 if the session does not exist; 422 if assembly validation fails |
| Inspect session | `GET {router}/sessions/{id}` | 200, `{"session_id", "records": [...], "metadata": {...}}` (for debugging; `metadata` may include `tito_session_mismatch`) | 404 |
| Delete session | `DELETE {router}/sessions/{id}` | 204 (the harness also accepts 200) | 404 if already collected or deleted; the harness treats this as success |

Harness conventions:

- Sessions are created in the trusted layer; an untrusted worker can only use
  the sessions of its own round. A failure part-way deletes the sessions already
  created.
- A session is deleted after its samples are collected (Miles'
  `OpenAIEndpointTracer.collect_samples` sends the DELETE in a `finally`).
  Sessions that were never collected are deleted on a best-effort basis; a failed
  delete never masks the original error.
- The harness depends only on this protocol, the core's round metadata interface
  and the reward result (`yeto.rl.rewards.types.RewardResult`).

## Verification against Miles

Checked against the Miles session service source at an earlier ports pin
(the current pin is `MILES_NEXT_COMMIT` in `yeto/rl/adapters/miles/pins.py`);
re-check when the pin moves.

Verified:

- `POST /sessions` reads `CreateSessionRequest` (evaluation / temperature /
  top_p / top_k) and answers 400 on an invalid body; creation returns
  `{"session_id"}` with 200.
- `POST /sessions/{id}/v1/chat/completions` forces `logprobs=True`; the response
  carries `meta_info.output_token_logprobs` with one entry per response token.
- `POST /sessions/{id}/samples` takes `max_seq_len` / `metadata`; assembly
  validation failures return 422 (read from the docstring only).
- `GET /sessions/{id}` returns `GetSessionResponse{session_id, records, metadata}`.
- `DELETE /sessions/{id}` returns 204, and 404 for an unknown session.
- The client sends DELETE after collecting samples.
- Miles also serves `POST /sessions/{id}/v1/messages` (Anthropic-compatible) and a
  `/sessions/{id}/{path}` passthrough; they are not part of this protocol.

Not verified:

- The streaming chat (SSE) response format. The test double implements only
  non-streaming JSON.
