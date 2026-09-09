# Selected-corpus CoT worker

This worker fills missing rationale at original reasoning boundaries. It keeps
the no-CoT SFT baseline separate. Generation sees original full prefix plus up to
five original events; the second semantic evaluator sees **only prefix and
candidate**. Unknown/uncertain/unsupported judgments are excluded. A model review
pass is evidence of those checks, not a proof that semantic leakage is impossible.

Current operating mode is **generation only, reviewer paused**. See
[the live handoff](BULK_HANDOFF_20260909.md) before starting any process. The
worker's `--generation-only` flag saves candidates as `review_pending` without
calling review or approving them. Later grounding rejections can feed the
[bounded prefix-only regeneration worker](REGENERATION.md). Nothing here
automatically activates that later phase.

The source and journal are a new versioned run. Do not point this worker at the
old pilot sidecar, rewrite its prompt hashes, or mix earlier candidate revisions.
No top-level API `tools` or `tool_choice` is sent by either stage.

## Prepare source on the manager coordinator (CPU only)

Copy the current `cot_filler/` package there first. The following paths describe
the data agent's frozen selection. Output filenames must not already exist.

```sh
python3 -B -m cot_filler.corpus_source \
  --manifest /mnt/lvm_data/sft_analysis/sft_baseline_20260908/codex_manifest.jsonl \
  --manifest-sha256 34d67323db93e40a33f9039f022f3a956ac14169c5ba47278493559b79d2e3e1 \
  --output /mnt/lvm_data/sft_analysis/sft_baseline_20260908/cot-source-v4-projection-v2.jsonl \
  --report /mnt/lvm_data/sft_analysis/sft_baseline_20260908/cot-source-v4-projection-v2-report.json \
  --max-traces 20000
```

The report states retained/excluded trace counts, exact source SHA, and gap count.
Each archive SHA is checked. Scores must satisfy quality>=4 and
overall_confidence>=3 on the original 5-point scale. Tool calls/results retain
original payload and order; original reasoning payloads are removed and only
their event IDs/positions establish gap boundaries. Unsupported event kinds or
ambiguous boundaries are reported, not silently converted or moved.
Projection v2 retains `agent_message.content` lists of explicit typed text or
refusal blocks, cleaning their text fields without flattening their structure.
The original v1 sources remain immutable historical inputs.

## Configure generation and review

Use the existing `OpenAICompatibleProvider` configuration with the exact copied
DeepSeek tokenizer and native renderer. Both configs need a real endpoint and
the verified actual context ceiling. Prefer a loopback tunnel to one reserved
DeepSeek host; do not use the SFT host or a fleet router that can send it traffic.

Generator: keep the pinned v5 prompt and tested generation settings. Reviewer: same
compatible endpoint/tokenizer is supported, but supply a separate explicit
config with enough output for the structured review. The paused accuracy trial
used v4 review with 32,768 thinking tokens and 49,152 total output tokens; this
requires actual strict-thinking enforcement and is not a completed quality
qualification. Local context counting includes the complete output allowance.
The model name and
thinking/template controls must match the server. Credentials remain in the
configured environment variable, never in config or command-line arguments.

Review v2 assigns candidate units and exact prefix snippets deterministic IDs.
The evaluator assesses all candidate IDs in order and selects evidence IDs;
code resolves them to original text before enforcing the complete support,
coverage, and finish checks. Every prefix string is retained across consecutive
snippets. The evaluator does not transcribe quotations or candidate sentences.
The initial v1 real pilot exposed quotation/coverage transcription errors and
4096-token review truncation; its failed rows are not retroactively approved.

## Bounded actual pilot

Acquire the **existing actual manager coordinator shared inference lock**. A
new dummy local lock does not reserve fleet capacity. Supply the source hash
from the successful preparation report; no arbitrary example hash is accepted.

```sh
python3 -B -m cot_filler.corpus_worker \
  --source /path/to/cot-source-v4.jsonl \
  --source-sha256 ACTUAL_HASH_FROM_PREPARATION_REPORT \
  --journal /path/to/cot-corpus-v4.sqlite3 \
  --config /path/to/generation.json \
  --review-config /path/to/prefix-review.json \
  --inference-lock /actual/shared/coordinator/inference.lock \
  --max-gaps 12 --workers 2 --token-budget 524288 \
  --retries 2 --provider-failure-limit 3 --validation-failure-limit 3 \
  --confirm-run
```

This is a real request command; it is not executed by importing the package or
running tests. Source preparation, indexing, rendering, and counting use CPU.
Generation and semantic review use the selected teacher endpoint's GPUs.

Inspect actual raw response receipts, candidate text, structured reviewer
evidence, accepted/rejected/uncertain counts, context-overflow count and health.
Then extend the same journal with a finite larger `--max-gaps`, concurrency and
token budget suitable for that single host. Operational limits can change on
resume; source, code, prompt, model, tokenizer and generation/review configuration
identities cannot silently change. New identities require a new journal.

## Recovery and export contract

- SQLite stores one offset/hash/digest per source trace and compact event IDs per
  gap. Windows are reconstructed just in time, independent of prior candidates.
- Sources are hashed once at startup. Every loaded source row is hash checked,
  and the frozen source file identity is checked before scheduling and commit.
- A candidate commits before its review is scheduled. Interrupted reviews reuse
  that candidate. An in-flight generation without a receipt can be repeated after
  process death; exact-once network execution is not claimed.
- SIGINT/SIGTERM stops new work and admission, drains admitted HTTP requests, and
  leaves deferred stages resumable. Configure a supervisor stop timeout longer
  than the HTTP request timeout plus retry handling; do not use immediate kill.
- All complete raw model responses, including invalid review JSON, are kept in
  `responses`. Accepted structured judgments are in `reviews`; candidates are
  in `candidates`; `gaps.state` governs inclusion. Only `approved` joins are usable
  as model-reviewed CoT. `uncertain`, `rejected`, `failed`, `generation_invalid`,
  `context_overflow`, and `admission_overflow` stay excluded.
- Repeated provider-unavailable or invalid-output results stop new requests and
  drain. Terminal failures remain recorded; a configuration or prompt repair
  uses a fresh journal instead of relabeling old receipts. Ordinary interruption
  resumes queued generation or durable-candidate review.
- `review_done` records `automatic_approval:true` only when an actual configured
  compatible reviewer returns a structurally valid pass with all required checks
  and prefix evidence resolved from exact snippet IDs. It records the user-authorized model-review policy
  and explicitly makes no semantic-certainty claim.
- A CoT journal approval does not mark training masks or training examples ready.

## Verification

```sh
python3 -B -m unittest cot_filler.tests.test_corpus_worker -v
python3 -B -m unittest discover -s cot_filler/tests -v
```

Tests use deterministic provider doubles or patched HTTP responses, never actual
teacher inference. They verify exact v4 gap equivalence, finite scheduling,
durable stage recovery, immutable identity rejection, token admission, source
mutation, context rejection, retries/circuit stop, structured review exclusion,
source marker/tool preservation and absence of executable API tools.
