"""Resumable, explicitly bounded full-corpus generation orchestration."""
from __future__ import annotations

import json
from pathlib import Path


def generate_bulk(store, provider, *, max_gaps, batch_size=25, checkpoint=None, confirm=False):
    """Generate queued gaps in durable bounded batches.

    This is an execution primitive; callers must explicitly confirm and provide a
    finite max. Each gap is committed independently, so interruption is resumable.
    """
    if not confirm:
        raise ValueError("Full-corpus generation requires explicit confirmation")
    if not isinstance(max_gaps, int) or max_gaps < 1:
        raise ValueError("max_gaps must be a positive finite limit")
    if not isinstance(batch_size, int) or not 1 <= batch_size <= 100:
        raise ValueError("batch_size must be between 1 and 100")
    path = Path(checkpoint) if checkpoint else None
    done = failures = 0
    queue = store.generation_queue()[:max_gaps]
    for offset in range(0, len(queue), batch_size):
        for row in queue[offset:offset + batch_size]:
            try:
                store.verify_source(row["id"])
                gap = store.detail(row["id"])["gap"]
                text, metadata = provider.generate(gap)
                revision = store.add_candidate(row["id"], text, metadata, row["revision"])
                done += 1
                status = "generated"
                detail = {"gap_id": row["id"], "revision": revision, "status": status}
            except (ValueError, OSError) as exc:
                store.record_failure(row["id"], type(exc).__name__ + ": " + str(exc))
                failures += 1
                detail = {"gap_id": row["id"], "status": "failed", "error_type": type(exc).__name__}
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(detail, sort_keys=True) + "\n")
    return {"generated": done, "failed": failures, "selected": len(queue)}
