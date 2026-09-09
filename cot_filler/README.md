# Synthetic rationale filler and local review

This is a local pilot implementation of the agreed replacement filler. It builds each request from **all original visible events before an assistant action, a `<cot>` insertion slot, and the next five original visible events including that action**. It never consumes a previous generated candidate as context. Outputs are synthetic, lookahead-conditioned explanations—not recovered original reasoning.

The current corpus worker supports resumable bulk generation, separate prefix-only review, and bounded prefix-only regeneration after a grounding rejection. See [the bulk handoff](BULK_HANDOFF_20260909.md) for the dated live run, [corpus operations](CORPUS_RUN.md) for the worker, and [regeneration](REGENERATION.md) for the later review/retry stage. The running generation job uses an immutable deployed snapshot; adding these local review tools does not activate them or change that job.

The sections below retain the original local pilot interface and its implementation history. Their original “not yet run” statements are historical; current execution and validation status is in the handoff. The included demo remains handwritten fixture data and cannot establish model quality.

## Run the local demo

Requires Python 3.9+; import, fake generation, review, and tests use the standard library only. Run these commands from `/Users/walden/Documents/ChatGPT/Yeta Labs`:

```sh
python3 -m cot_filler --db cot_filler/demo-review.sqlite3 import cot_filler/examples/demo.json
python3 -m cot_filler --db cot_filler/demo-review.sqlite3 generate --fake
python3 -m cot_filler --db cot_filler/demo-review.sqlite3 serve --port 8765
```

Open <http://127.0.0.1:8765>. The server binds only loopback. It has no automatic generation route. Select a gap, expand the complete prefix/tool payloads, hide future events to check factual claims, then show them to assess action fit. Approve, save an edited approval, reject with tags/notes, or defer. Decisions survive refresh and server restart in SQLite. Prior model text and review edits remain in the revision history. Alt+left/right navigates the queue.

“Queue regeneration” persists a request without calling a model. It excludes that gap from export until its replacement is generated and approved. A subsequent explicit CLI `generate` processes ungenerated and queued gaps. The demo can use `--fake` again to exercise revision behavior.

```sh
python3 -m cot_filler --db cot_filler/demo-review.sqlite3 status
python3 -m cot_filler --db cot_filler/demo-review.sqlite3 export --arm visible --output reviewed-visible.jsonl
python3 -m cot_filler --db cot_filler/demo-review.sqlite3 export --arm masked --output reviewed-masked.jsonl
python3 -m cot_filler --db cot_filler/demo-review.sqlite3 export --arm none --output reviewed-none.jsonl
python3 -m cot_filler --db cot_filler/demo-review.sqlite3 export --arm visible --format messages --output reviewed-messages.jsonl
```

Export paths must not exist. Source input paths and the database cannot be overwritten. The browser also provides the three download links. Keep the original input at its imported path: generation, approval, and export check its file hash and refuse changed or missing input. Copy a pilot and its database together when moving it; path relocation currently requires reimport.

## Input contract and adapters

The preferred input is JSON or one trajectory per JSONL row in `cot.trace.v1`:

```json
{
  "schema": "cot.trace.v1",
  "trace_id": "sealed-capture-or-session-id",
  "source_format": "normalized-full-visible/v1",
  "metadata": {"source_seal": "upstream seal", "branch_id": "one resolved branch"},
  "events": [
    {"event_id": "e1", "role": "user", "kind": "message", "content": "Original request"},
    {"event_id": "e3", "role": "assistant", "kind": "tool_call", "content": "", "tool_calls": [{"id":"call-1","name":"read_file","arguments":{"path":"app.py"}}]},
    {"event_id": "e4", "role": "tool", "kind": "tool_result", "tool_call_id":"call-1", "content":"Original full observation"}
  ],
  "gap_targets": [{"event_id":"e3", "reason":"encrypted_marker", "source_marker_ids":["e2"]}]
}
```

The adapter upstream must supply one fully resolved, original branch in canonical visible order. Preserve all original event IDs and tool calls, arguments, results, definitions, and visible metadata. **Do not feed compact labeling projections, model-generated sidecars, encrypted reasoning text, old appended `[thinking]` outputs, or unresolved database references.** Original reasoning is omitted; retain only its presence and marker-to-action mapping. Unknown source event structures fail instead of being silently dropped. Visible compaction/media placeholders from upstream remain placeholders; this text-only filler cannot infer omitted earlier history or image contents.

The current production `trace_labeling.archive.project_visible` removes original reasoning and keeps visible events, but the inspected version removes `data` and renders tool payloads as text. A production bridge must carry the cleaned structured tool payloads alongside that projection if the structured originals are needed; this module deliberately does not claim a turnkey RocksDB/corpus reader. Construct the canonical envelope after source resolution and before compact label projection. Carry the frozen seal and branch/capture identity in metadata. The source file hash, normalized trajectory digest, prefix/lookahead hashes, insertion action ID, prompt hash, and adapter version are persisted in the sidecar.

An included **already-resolved archive bridge** accepts `{"trace_id":"...", "resolved":true, "archive_events":[...], "metadata":{"source_seal":"..."}}`. Each archive event must contain the inspected `event_id`, `role`, `kind`, `content`, and `data` fields. It preserves structured `data` on visible events, removes standalone reasoning/analysis events, cleans recognized assistant message reasoning fields/blocks, and maps explicit encrypted-marker slots to the immediately following action. It can instead accept `gap_targets` supplied by an upstream reasoning-free projector. Tool payload dictionaries are not recursively scrubbed merely because a task-data key is named `reasoning` or `thinking`. Unsupported assistant message block shapes, unresolved envelopes, inline reasoning requiring an upstream projector, and ambiguous boundaries fail explicitly. This adapter was tested with invented examples matching the inspected schema; no real corpus row has been imported or validated yet.

Additional supported inputs are ATIF documents (`steps`), supported Codex/Puffer session event JSON/JSONL, and clean legacy Yeto `messages` rows. ATIF preserves full tool observations without the old 4,000-character crop. Codex response events take precedence over mirrored UI `event_msg` messages. An encrypted marker must map immediately to an assistant action; a marker followed by a tool result or other ambiguous boundary is rejected, not shifted forward. Unsupported event/ATIF observation shapes require an explicit canonical adapter. There is no attempt to decrypt or reconstruct protected source reasoning.

Default eligibility is explicit encrypted-marker targets. `--gap-policy missing-assistant` additionally selects assistant actions with no known original plaintext reasoning. Adapters retain a boolean `original_reasoning_present` and omit the text; canonical importers must provide that boolean where relevant. A clean converted Yeto row often has already lost marker positions, so supply explicit `gap_targets` or intentionally choose the missing-assistant policy. Default marker matching recognizes tagged encrypted/redacted markers, not arbitrary strings that merely look like ciphertext. Resolve other encodings explicitly in the canonical envelope.

## Real pilot generation: explicit configuration and shared capacity

Do **not** run a competing filler against the current labeling fleet. The existing labeling service holds `trace_labeling/runs/fleet_medium/inference.lock` on the manager host. The real generation CLI requires an existing lock file and obtains an exclusive nonblocking `flock` for the entire pilot. It refuses a busy lock. This protects capacity only when all clients use the **same underlying lock on the serving coordinator/shared filesystem**. A new Mac-local file cannot protect the manager's remote labeling run. Deploying this worker beside that lock, scheduling a labeling pause or dedicated capacity, and authorizing the pilot remain operational steps outside this local implementation. No lock or production reservation has been changed.

Real generation additionally needs the serving model's tokenizer files already present locally and an installed `transformers` package. No dependency or tokenizer download is automatic; custom remote tokenizer code is disabled. The actual server chat template and its thinking controls must be verified to match the local tokenizer. Example config (copy and fill; values are illustrative, not an approved production configuration):

```json
{
  "base_url": "http://127.0.0.1:30100/v1",
  "model": "EXACT_SERVING_MODEL",
  "tokenizer_path": "/existing/local/model-tokenizer",
  "chat_template_matches_server": true,
  "chat_template_kwargs": {"enable_thinking": true},
  "context_limit": 262144,
  "max_output_tokens": 4096,
  "safety_margin": 256,
  "temperature": 0.3,
  "timeout_seconds": 300,
  "api_key_env": "COT_TEACHER_API_KEY"
}
```

```sh
python3 -m cot_filler --db pilot.sqlite3 generate --config teacher.json --inference-lock /existing/shared/inference.lock --limit 30
```

The full rendered chat prompt is tokenized with the configured template kwargs and generation prefix. Prompt tokens + output ceiling + safety margin must fit; overflow records a failure and sends no request. There is no silent tail crop or summary substitution. If using `reasoning_effort`, include the same setting in the explicitly verified template kwargs. The output ceiling includes any server reasoning tokens where that API applies it; tune only after observing the actual server contract. Tokenizer identity and counting options are recorded. Real providers use an explicit OpenAI-compatible chat completions endpoint; credentials come only from the named environment variable and are never persisted. Loopback HTTP is accepted for SSH tunnels; remote access requires `allow_remote_endpoint: true` and HTTPS. Redirects and environment proxies are disabled. No Responses API or streaming adapter is implemented.

Only final message `content` becomes a candidate; provider `reasoning_content` is not saved. Empty/malformed/truncated/incomplete outputs cannot be approved unchanged. Edits retain the original candidate and can fix a structurally blocked result before approval. Other flags are advisory, and semantic grounding is a human judgment: a regex cannot certify action fit, absence of hindsight, or “reasonable CoT.” The pilot is sequential, bounded to 1–50 gaps per invocation, resumable, and has no automatic retries that could silently repeat billed work. Failures are recorded in SQLite; a manual later invocation retries still-unfilled/queued gaps.

## External Codex-authored pilot artifacts

A bounded import path accepts finished rationale artifacts written by explicitly assigned Codex tasks from the **exact complete sanitized gap prompt**. This does not call or compete with the DeepSeek labeling fleet. Such candidates are honestly labeled `codex-session`, with the exact backend model and token usage unavailable. They are not FakeProvider fixtures and do not establish DeepSeek teacher quality. This route imports artifacts only; it neither generates candidates nor authorizes their creation.

```sh
python3 -B -m cot_filler --db real-pilot.sqlite3 import-candidates finished-candidates.jsonl
```

Each JSONL record must contain exactly the following fields. Copy identities and hashes from the existing gap; never substitute a shortened prefix or continuation while claiming its full-window hash. `expected_revision` is 0 for a gap with no candidate and otherwise the exact current candidate revision. The authoring task is its actual canonical collaboration task path.

```json
{
  "schema": "cot.external-candidate.v1",
  "gap_id": "existing gap ID",
  "source_digest": "existing source digest",
  "prefix_hash": "existing prefix hash",
  "lookahead_hash": "existing lookahead hash",
  "prompt_hash": "existing full prompt hash",
  "expected_revision": 0,
  "text": "Finished plain-text synthetic rationale, without CoT wrappers.",
  "generator": {
    "provider": "codex-session",
    "model": "not_attested",
    "authoring_task": "/root/assigned_author",
    "prompt_version": "synthetic-prefix-lookahead5/v1",
    "token_usage": null
  }
}
```

The importer accepts 1–10 records per batch, at most 1 MiB from the CLI, in JSONL or a JSON object/list. It checks the current underlying source-file hash, reconstructs each immutable gap and prompt, checks every supplied hash and candidate revision, and rejects unknown/duplicate gaps, malformed text, false model/token provenance, and approval fields. The batch is one SQLite transaction: an error in a later record rolls back earlier records. Finished text and provenance are preserved, and every new revision starts **pending**, with manual-review and external-authorship flags. Completion is recorded as `assistant_authored_finished_artifact`, without inventing an API finish reason or token count. No candidate becomes exportable until manually approved.

The same operation is available through `Store.import_candidates(records)` and local `POST /api/candidates/import` with body `{"candidates":[...]}`. The HTTP route uses the review server's existing session-token/origin protection and 128-KiB request bound. Import does not acquire or bypass the production inference lock because it performs no inference at all.

## Export and training integration limits

The primary export is `cot.reviewed-events.v1`, with approved synthetic rationale events placed **before** the associated original action. Original source event objects remain unchanged and in order. Currently approved latest revisions only are selected; rejection/regeneration cannot silently fall back to an older approved candidate. All three arms share the same selected trajectories and approved gap revision references. The `none` arm removes synthetic events while preserving this common selection.

Exports include source/gap/prompt hashes, generator and review provenance, and structural `loss_segments` identifying training versus masking intent. They explicitly set `training_ready: false`. These are **not token-aligned loss masks**. Before SFT, add the target model's actual chat-template renderer and verify rationale placement and token boundaries for visible/masked/no-rationale arms.

Use `export --format messages` for the bounded compatibility adapter to the old converter's `messages`/`metadata` row shape. It prepends `<cot>\n...\n</cot>\n\n` to the associated assistant message while retaining that message's action/tool fields. Standard function calls are preserved; Codex/ATIF/Puffer calls with explicit call ID, function name and arguments are mapped to function-call shape. Original event objects remain in `metadata.original_events` so adapter normalization loses no source provenance. Tool results retain their call IDs. String message content is supported; tool definitions, custom tool calls, multimodal/structured content, missing call IDs and unknown event kinds fail clearly rather than being omitted. Tool-only assistant actions remain tool-only apart from the prefixed rationale. All three arms are available in both formats.

The messages export also records character-offset rationale/content segments and mask intent. `cot_filler.masking.align_char_segments` converts boundaries into a token mask only when the caller supplies the exact target-rendered string and a tokenizer that exposes offsets; `attach_token_masks` additionally requires explicit template/tokenizer attestations and adapter-produced offsets in that rendered string (it rejects per-message offsets). Special tokens are masked and conflicting overlaps fail closed. The canonical event export remains available for unsupported shapes. A legacy append-after-answer layout is intentionally absent.

Full-corpus generation is available as `generate-bulk`. It requires an existing shared inference lock, an explicit `--confirm-bulk`, a finite `--max-gaps` bound, and writes an optional append-only checkpoint manifest. Each gap is committed independently, so an interruption can resume from the remaining queue. The command is intentionally never launched implicitly by export or review operations.

`cot_filler.deepseek_adapter.events_to_messages` is the explicit canonical-event mapping for DeepSeek V4. Complete `tool_definition` events become OpenAI function schemas attached to the following user/developer message; calls and results retain IDs and normalize arguments to JSON strings. Incomplete definitions, orphaned definitions, missing call IDs, and unknown event kinds fail closed. `render_events` then delegates to the bundled native `encoding_dsv4.py` renderer.

`render_events_with_segments` inserts synthetic rationale before its action, renders through the native encoder, and locates unchanged text plus generated DSML tool-call/tool-result blocks in order for token alignment. It fails closed if any expected block is transformed unexpectedly.

The upstream DeepSeek V4 Jinja template is pinned at `assets/deepseek-v4-flash-0731/chat_template.jinja` (commit `e13c04e`). `DeepSeekJinjaRenderer` uses it with a plain JSON filter to avoid HTML escaping DSML markup and normalizes wire-format JSON arguments to mappings. Its output is tested byte-for-byte and token-for-token against `encoding_dsv4.py` on the demo tool trace.

## Verification

```sh
python3 -m unittest discover -s cot_filler/tests -v
```

The tests exercise exact full-prefix/five-event windows, nested tool payload preservation, marker alignment, no synthetic feedback, reasoning presence without text retention, context overflow before network access, template-option matching, truncated completions, immutable sources, stale candidate revisions, durable edits/regeneration, three-arm exports, and localhost API origin/token protection. The HTTP integration test needs permission to bind a temporary loopback socket in a sandbox. Demo/browser QA cannot establish model quality; real manual pilot review remains required.
