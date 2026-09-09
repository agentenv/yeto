# CoT bulk handoff — September 9, 2026 UTC

**Generation is running across all six inference nodes. The reviewer is stopped at the user's request; review will happen later.** All generated candidates remain `review_pending`, with no bulk approvals or exports. n7 remains reserved for SFT and unavailable to the inference router.

Latest verified state at **01:24:15 UTC**: 516 committed candidates (56 newly completed after resume), 38 active generation tasks, 292 excluded context overflows, 642,667 queued gaps, zero bulk review records. The paused trial remains unchanged at19 review responses. At the01:22:43 fleet check, all six inference backends were healthy and loaded (7/6/7/7/7/6 requests); n7 was unhealthy/load0. Generation-only resumed at01:22:13. These are startup snapshots, not a sustained throughput forecast.

## Current owner and durable artifacts

Manager is `c@65.19.161.135`. All relative manager paths below are under:

```
/mnt/lvm_data/sft_analysis/sft_baseline_20260908
```

- **Active generation-only supervisor PID1488373, startticks37556563.**
- Run directory: `cot-generation-only-20260909/`.
- Durable executable: `cot-generation-only-20260909/supervisor.py`, SHA256 `6a79ca26cc8cfbf758d83174eedb51d5810b3d4321aa1ff2b262ff0dee895062`.
- Local source copy: `cot_filler/pilots/20260908-real/generation_only_supervisor.py`.
- Local actual launch receipt: `cot_filler/pilots/20260908-real/generation-only-launch-20260909.json`.
- Local actual stop receipt: `cot_filler/pilots/20260908-real/reviewer-stopped-20260909.json`.
- Log: `cot-generation-only-20260909/supervisor.log`.
- `launch.json`, `reservation-acquired.json`, `generation-started.json`, and `reviewer-stopped.json` are in the active run directory.
- The process owns both the journal worker lock and the actual shared lock: `/mnt/lvm_data/sft_analysis/extracted/trace_labeling/runs/fleet_medium/inference.lock`.
- Immutable code: `cot-code-v5-accuracy-bulk/`, manifest SHA256 `da05d85a7873f2f15fda36f9f33444732bc1d344be07a150552c56a31046d7d4`.

The supervisor was launched detached with `python3 -u` against the durable script, stdin disconnected, stdout/stderr to its log, and a new process session. Do not launch a second copy. Existing one-shot receipts intentionally prevent accidental replay. For any restart, use a new versioned supervisor/run directory with the same stored journal identity and pinned implementation.

## What is generating

- Source: `cot-source-v4-projection-v2.jsonl`.
- SHA256: `ed833db681fd8fd1e3c2315d96503016e8753bf7b4036ed6fbda4f88981c8bed`.
- 15,368,261,626 bytes; 14,146 safely representable traces from the quality/confidence-filtered20k selection; **643,513 original reasoning boundaries**.
- Bulk journal: `cot-bulk-v5-review-v4-accuracy.sqlite3`.
- Generator configuration remains exactly the journal's immutable `identity.generator_config`, using manager router `127.0.0.1:30100/v1`.
- 48 workers; token admission6,291,456; finite max643,513 gaps; one transient retry.
- Explicit `generation_only=True`; the supplied `NoReview` object raises if called. Successful outputs become `review_pending`. Failed/incomplete/overflow candidates remain excluded.
- Established generator v5 still sees the complete original prefix plus next five original events, as the user previously chose. Original encrypted reasoning is removed, and previous generated rationales never contaminate subsequent prefixes. The source and exact insertion boundaries remain immutable.

The fresh bulk journal reused only verified index rows from untouched `cot-bulk-v5-review-v3-thinking.sqlite3`. Checks cover the full source hash, original core and index/gap-construction code, contiguous source/ordinal coverage, every gap ID, complete/unclaimed/empty parent state, and an unchanged parent artifact. Each actual request still checks source row SHA, trace digest and original boundary. Independent tests/review found no index or generation-only blocker.

## Reviewer stopped; preserved results

The former joint supervisor PID1483495/start37483037 received SIGTERM at01:20:18. It stopped new admission. **Generation drained normally by01:20:43**, committing all460 candidates with no generating rows left. Only then, after the grace interval, the old controller was terminated at01:22:02 because two review HTTP calls remained. No inference engine, router, model cache or other job was stopped. The new generation-only owner acquired the lock at01:22:03 and reused the exact journal identity; no committed candidate was regenerated or overwritten.

`cot-generation-only-20260909/reviewer-stopped.json` records the shutdown and the two possibly unfinished review IDs. No outcome was fabricated for those requests.

Preserved accuracy experiment:

- `cot-review-v5-v4-strict32k-48.sqlite3`: 19 actual review responses; 9 provisional passes, 9 rejections, 1 uncertain, 2 context overflows, 25 not yet reviewed and 2 interrupted/uncommitted reviews still recorded as `reviewing` in that historical journal.
- The planned latest128 regression subset13/64 **never started**.
- The earlier proposed8k thinking experiment also **never sent an inference request**.
- Former joint artifacts remain under `cot-joint-accuracy-v4-bulk/`; local copies are `joint_accuracy_supervisor.py` and `bulk-launch-20260909.json` in the pilots directory.

**Do not restart the reviewer now. There is no review ETA or pending automatic activation.** The user said to generate first and review later.

## Later review scope

The user's latest requirement is **no future-information leakage**. Later review should check that factual observations and claimed prior results are supported by the prefix, and reject hindsight/future-only facts. Reasonable prospective plans are allowed. Do not add a style-matching gate or restart general model-quality optimization.

Because the generator intentionally sees next five events, generation alone cannot prove no leakage. Pending candidates are not accepted training output. Existing prefix-only review remains relevant, but its finite passes do not guarantee semantic correctness.

A **future** bounded reviewer-requested regeneration feature is implemented locally in `regeneration.py` and `regeneration_worker.py`: prefix-only regeneration, no rejected text or future-derived feedback in its prompt, at most two attempts, and fresh review on every attempt. The original journal stays read only; a separate append-only ledger preserves replacements and actual receipts. The finite CLI has explicit activation, shared/journal locks, exact source/code/config bindings and safe resume/pagination.31 focused protocol tests and an independent code audit passed. No real regeneration quality test has run. See [REGENERATION.md](REGENERATION.md) for later activation and live backend verification requirements. **It is not deployed and does not authorize resuming review.**

Current stored intended reviewer configuration uses v4, thinking32768 and total output49152 with strict JSON; it points at router30100. Only n5 currently has strict thinking enabled. The field `require_strict_thinking_server:true` is a requirement, not proof of fleet readiness. If that reviewer is later used, verify every routed backend's actual settings and preserve exact prompt/response control receipts. Full-prefix context overflow stays excluded; never truncate the prefix merely to fit an output allowance. Any later prompt/config change needs an explicit new identity/migration preserving candidates and old receipts.

## Read-only status command

```sh
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile=/private/tmp/yeta-labeling-known-hosts \
  c@65.19.161.135 'python3 -' <<'PY'
from pathlib import Path
import json,sqlite3,urllib.request
b=Path('/mnt/lvm_data/sft_analysis/sft_baseline_20260908')
r=b/'cot-generation-only-20260909'
p=Path('/proc/1488373/stat')
print('same_generator_alive',p.exists() and p.read_text().split()[21]=='37556563')
with sqlite3.connect((b/'cot-bulk-v5-review-v4-accuracy.sqlite3').as_uri()+'?mode=ro',uri=True) as d:
    print('states',dict(d.execute('select state,count(*) from gaps group by state')))
    print('responses',dict(d.execute('select stage,count(*) from responses group by stage')))
    print('bulk_reviews',d.execute('select count(*) from reviews').fetchone()[0])
for n in ['failed.json','completed.json']:
    if (r/n).exists():print(n,(r/n).read_text())
w=json.load(urllib.request.urlopen('http://127.0.0.1:30100/workers',timeout=10))
print('workers',[{'url':x['url'],'healthy':x['is_healthy'],'load':x['load']} for x in w['workers']])
PY
```

## Safe generation-only resume

The active durable `generation_only_supervisor.py` is the reference implementation. It reads the stored complete journal identity, checks its implementation hashes against the pinned stage, acquires both locks, opens `Journal` with that exact identity, and calls:

```python
run_corpus(journal, generator, NoReview(), max_gaps=643513,
           workers=48, token_budget=6291456, retries=1,
           stop=stop_event, generation_only=True)
```

It preserves committed candidates and requeues only uncommitted `generating` entries if an interrupted process left any. The current transition required no generation requeue because all active generator work drained and committed. The generic corpus CLI does not automatically reconstruct the additional derived-index/creator identity fields: do not replace them or edit the original journal metadata. Use the existing stored identity.

For a future authorized restart, first verify the current PID/startticks, send SIGTERM only to that owner, and let active generation drain. Keep its receipts. Create a new run directory and durable supervisor copy, with a new actual launch receipt, preserving the pinned stage, journal identity and generation-only mode. Do not stop model engines as part of a client restart.

Review activation is a separate future user-authorized operation. No current command or supervisor will enable it automatically.

## n5 and connection state

- n5: `ubuntu@100.79.82.11`; existing `dsv4-eng` container unchanged.
- Strict model PID4129424/start33904940 is healthy. Its argv is exactly the original plus `--enable-strict-thinking`; model/TP8/EP8/context262144/memory0.8/prefill settings are unchanged.
- On n5 only, `/data/rl/launch/engines/cot-strict-n5-20260909/` contains `strict-launch.json`, `strict.log`, `restart-plan.json`, `ROLLBACK.txt` and the private0600 original launch/environment snapshot. Do not copy that environment into project context.
- Restore the original launch under the shared inference lock before future labeling if its original defaults are required. No rollback or other engine change is currently requested.
- Existing Mac forward PID56886: Mac33120→n5 localhost31200.
- Reverse forward PID65279: manager33122→Mac33120; receipt `/private/tmp/yeta-cot-n5-strict-reverse-33122.json`. It is no longer used by a running reviewer and has been left intact.
- The stale manager33120 listener was left untouched. No SSH keys or access controls were created/changed.
- Current generation uses the existing router30100 connection across all six nodes. It does not depend on the paused direct-review workflow.

Generation-only is detached and running. Reviewer is stopped. No bulk approval, export or future-leakage guarantee is claimed.
