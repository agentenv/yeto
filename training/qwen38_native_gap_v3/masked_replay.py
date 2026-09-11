"""Re-render verified original replay sources; never reuse baseline token arrays."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from training.qwen38_no_cot import prepare_data as baseline
from . import source_adapters

VERSION = "original-replay-to-native-xhigh-turn-boundary/v2"
_renderer = None


def load_jobs(manifests):
    """Freeze existing source inventory identities, deduplicating exact files."""
    jobs, seen, inventories = [], set(), []
    for supplied in manifests:
        path = Path(supplied).resolve(strict=True)
        raw = path.read_bytes()
        signature = hashlib.sha256(raw).hexdigest()
        inventories.append({"path": str(path), "sha256": signature})
        for lineno, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            job = json.loads(line)
            if (job.get("source") != "replay" or job.get("format") not in {"rollout", "atif"}
                    or not isinstance(job.get("sha256"), str) or len(job["sha256"]) != 64
                    or job.get("identity") != "replay:" + job["sha256"]):
                raise ValueError("Only original baseline replay source jobs are supported")
            original = Path(job["path"]).resolve(strict=True)
            if not original.is_file():
                raise ValueError("Original replay source file is missing")
            if job["sha256"] in seen:
                continue
            seen.add(job["sha256"])
            jobs.append({**job, "path": str(original), "source_jobs_sha256": signature,
                         "source_jobs_line": lineno})
    return jobs, inventories


def normalize_job(job):
    if job.get("source") != "replay" or job.get("format") not in {"rollout", "atif"}:
        raise baseline.UnsupportedTrace("not_an_original_replay_job")
    raw = Path(job["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != job["sha256"]:
        raise baseline.UnsupportedTrace("source_file_hash_mismatch")
    if job["format"] == "rollout":
        messages, counts, group = source_adapters.rollout_messages(
            [json.loads(line) for line in raw.splitlines() if line.strip()])
    else:
        messages, counts, group = source_adapters.atif_messages(json.loads(raw))
    return messages, counts, group


def init_worker(tokenizer_dir):
    global _renderer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from .masked import Qwen38TurnBoundaryMaskedRenderer
    _renderer = Qwen38TurnBoundaryMaskedRenderer(tokenizer_dir)


def convert_job(job, *, renderer=None):
    from .masked import training_row
    renderer = renderer or _renderer
    if renderer is None:
        raise RuntimeError("Replay worker renderer is not initialized")
    try:
        messages, counts, group = normalize_job(job)
        if job.get('original_baseline_group_id') is not None and group != job['original_baseline_group_id']:
            raise ValueError('Original replay session grouping changed')
        rendered = renderer.render_replay_messages(
            messages, source_digest=job["sha256"], trace_id=job["identity"],
            boundaries=[{"barrier_before": True} for _ in messages],
            source_tool_audit=counts["turn_boundary_audit"]["raw_tool_cardinality"])
        provenance = {**job, "replay_adapter_version": VERSION,
            "replay_adapter_sha256": baseline.sha_file(__file__),
            "normalization_version": source_adapters.VERSION,
            "normalizer_sha256": baseline.sha_file(source_adapters.__file__),
            "source_normalization_counts": counts, "messages_sha256": baseline.digest(messages),
            "original_reasoning_used": False, "session_identity_verified": True,
            "group_basis": "original_replay_session", "capability_schemas_included": False,
            "baseline_token_arrays_reused": False,
            "custom_adapter": "raw-input-as-function-input-string/v1"}
        row = training_row(rendered, group_id=group, source="replay", provenance=provenance)
        return {"ok": True, "job": job, "row": row,
                "messages_sha256": baseline.digest(messages)}
    except Exception as exc:
        # Never put original replay text or arbitrary exception content in logs.
        parity=getattr(exc,'audit',None)
        return {"ok": False, "identity": job["identity"], "source": "replay",
                "reason": "cross_arm_no_cot_parity_failed" if parity is not None else
                    str(exc) if isinstance(exc, baseline.UnsupportedTrace) else type(exc).__name__,
                **({"cross_arm_parity_failure":True,"cross_arm_parity_audit":parity} if parity is not None else {})}
