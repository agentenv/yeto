# Later reviewer-requested regeneration

Implemented locally; **not deployed or running**. The current six-node bulk run
continues generation only. The user stopped the reviewer, so this feature does
not authorize restarting it. No real regeneration quality evaluation has run.

`regeneration.py` defines the bounded protocol, prefix-only request, provider and
append-only ledger. `regeneration_worker.py` connects those pieces to the same
configured prefix reviewer. A coordinator can call `run_regenerations` after an
original review, or explicitly activate the finite CLI later.

## Eligibility and acceptance

Only a complete, correctly bound model rejection with an explicit
`no_future_observation_claims:false` or
`all_factual_claims_supported_by_prefix:false` can request a replacement. An
approval, uncertainty, style-only rejection, missing/malformed review, transport
failure or context overflow does not qualify.

The replacement prompt contains the **original prefix only** and bounded machine
failure codes/unit IDs. It contains no following events, target action, rejected
candidate text, reviewer prose or evidence quotations. Previous generated
rationales never replace original prefix events. Gap IDs, original event IDs and
source/prefix hashes preserve the insertion boundary.

Every new candidate needs a fresh review under the exact configured policy,
model, tokenizer, output limit and decoding settings. Complete statement/evidence
validation and exact local/server prompt-token parity remain required. There are
at most **two replacement attempts per original gap in the ledger**; a second
attempt requires the first replacement's actual grounding rejection. There are
no automatic retries for uncertainty, malformed responses, transport failures or
context overflow. Never create another ledger merely to bypass this limit.

The intended quality criterion is prefix-grounded facts without hindsight or
future-only information. Reasonable prospective plans are allowed. Model review
is evidence, not a guarantee of zero leakage; protocol tests do not establish
semantic quality. Existing same-policy review checks are preserved; style alone
cannot request regeneration.

## Storage and recovery

The original source JSONL and original SQLite journal are read only. A separate
SQLite ledger records each attempt's immutable lineage, request start, raw
response, validated generation, actual review and any terminal failure. Failed
and malformed raw responses remain auditable. No original candidate, review or
export is overwritten.

- A saved generation resumes at review rather than generating again.
- A saved raw response resumes validation without sending the request again.
- A request with a persisted start but no response is recorded as interrupted
  and excluded. Its outcome is unknown; it is never silently replayed.
- A graceful stop finishes the current call, records its response, and leaves
  the next stage pending. No new call starts after the stop flag is checked.
- `accepted_selections(ledger)` yields separate gap/attempt/candidate-hash
  references after actual acceptance. It does not export training data.

Each run pins the exact source journal identity, finite selected queue, configs,
tokenizers and implementation. Changed source bytes, changed original rejection,
changed code/config or a changed finite queue prevent implicit resume. An
intentional migration requires a new explicitly audited artifact retaining all
old attempt histories and counts; the worker does not perform that migration.

## Explicit future CLI

Do not run this while the user's reviewer pause remains in effect. The live
generation supervisor currently owns the shared inference lock; this CLI will
refuse to run concurrently with it. A later coordinator may instead call the
Python API while retaining that same lock across coordinated stages.

The CLI uses configs stored in the original journal, with no endpoint/model
override. It requires the original journal's renderer/reviewer implementation to
match the loaded code. Stage the correct immutable code and tokenizer assets;
do not rewrite a journal identity to force compatibility.

Example for a future fleet run, after review is explicitly resumed:

```bash
python3 -m cot_filler.regeneration_worker \
  --original-journal /path/to/reviewed-original.sqlite3 \
  --ledger /path/to/new-prefix-regeneration.sqlite3 \
  --inference-lock /mnt/lvm_data/sft_analysis/extracted/trace_labeling/runs/fleet_medium/inference.lock \
  --max-gaps 32 \
  --after-ordinal 0 \
  --live-readiness-hook /path/to/audited_live_fleet_check.py \
  --live-readiness-hook-sha256 EXACT_SHA256_OF_THAT_CODE \
  --confirm-run
```

The worker deliberately admits **one model request at a time** for this optional
retry queue. `--max-gaps` is mandatory and finite; attempts are capped separately
at two. The final aggregate result includes `next_after_ordinal`: use that cursor
with a new ledger for the next disjoint cohort **only after `cohort_complete:true`**.
An interrupted or pending cohort returns a null cursor and
`resume_same_ledger_required:true`. Resume an interrupted cohort with
the same ledger and its original cursor/limit; do not advance past pending work.
Both the actual shared inference lock and the new ledger's worker lock
must be free. Credentials are read only through the configured environment
variable and are never written to receipts or docs.

For a genuinely direct SGLang endpoint, `--direct-strict-server` replaces the
hook arguments. It queries that configured endpoint's `/get_server_info` before
requests and verifies actual `enable_strict_thinking`, served model and context
length, plus `grammar_backend=xgrammar`. **Do not use that direct option with a router.**

## Live strict-backend activation hook

If the configured reviewer requires strict thinking, a trusted coordinator must
check actual live backend settings before any generation/review request. A
configuration boolean or previously written readiness file is insufficient.

For fleet routing, the hash-pinned local hook implements `verify_live(config)`.
On **each call**, it must inspect the exact configured router's current eligible
backend inventory and freshly query every routable backend. It must abort if
identity, routing or settings cannot be verified. The worker checks this return
contract and stores only server-argument hashes:

```python
{
    "schema": "cot.live-strict-backends/v1",
    "config_hash": digest(config),
    "checked_at_unix": time.time(),
    "routed_backend_ids": ["node-id"],
    "backends": [{
        "id": "node-id",
        "server_args": actual_fresh_get_server_info["server_args"],
    }],
}
```

All listed routes must have exactly one verified backend, strict thinking must
actually be enabled, served model must match, and context must accommodate the
configured limit. The grammar backend must be `xgrammar`; any model path/revision
explicitly present in the config must match too. Results older than 30 seconds
are refused. The callback is a
trusted activation boundary; the return schema alone cannot prove that its code
queried the correct fleet. The caller must audit/pin the hook and maintain
exclusive scheduling ownership so routing cannot change behind it. A production
fleet hook has **not** been deployed as part of this local feature.

## Tests

```bash
python3 -m unittest cot_filler.tests.test_regeneration cot_filler.tests.test_regeneration_worker
```

Fixtures are explicit synthetic model doubles. Tests cover prefix isolation,
grounding-only eligibility, two-attempt lineage, receipt/config/code bindings,
no overwrite, interruption recovery, saved-response replay prevention, finite
activation and actual-readiness contract rejection. They make no model quality
or full-corpus readiness claim.
