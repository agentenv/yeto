"""SQLite sidecar: immutable sources, append-only candidates and review history."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .core import BLOCKING_FLAGS, canonical, digest, gaps_for, validate_candidate
from .external import validate_external


def now():
    return datetime.now(timezone.utc).isoformat()


class Conflict(ValueError):
    pass


class Store:
    def __init__(self, path):
        self.path = str(path)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS traces(digest TEXT PRIMARY KEY, trace_id TEXT, data TEXT, provenance TEXT);
        CREATE TABLE IF NOT EXISTS gaps(id TEXT PRIMARY KEY, source_digest TEXT REFERENCES traces(digest), data TEXT);
        CREATE TABLE IF NOT EXISTS candidates(gap_id TEXT REFERENCES gaps(id), revision INTEGER, text TEXT, data TEXT, created TEXT, PRIMARY KEY(gap_id,revision));
        CREATE TABLE IF NOT EXISTS reviews(id INTEGER PRIMARY KEY, gap_id TEXT, revision INTEGER, status TEXT, text TEXT, note TEXT, tags TEXT, created TEXT, FOREIGN KEY(gap_id,revision) REFERENCES candidates(gap_id,revision));
        CREATE TABLE IF NOT EXISTS regeneration(id INTEGER PRIMARY KEY, gap_id TEXT REFERENCES gaps(id), revision INTEGER, created TEXT, fulfilled INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS failures(id INTEGER PRIMARY KEY, gap_id TEXT REFERENCES gaps(id), error TEXT, created TEXT);
        """)

    def close(self):
        self.db.close()

    def import_trace(self, trace, provenance, policy="markers"):
        dg = digest(trace)
        gaps = list(gaps_for(trace, policy))
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO traces VALUES(?,?,?,?)", (dg, trace["trace_id"], canonical(trace), canonical(provenance)))
            for gap in gaps:
                self.db.execute("INSERT OR IGNORE INTO gaps VALUES(?,?,?)", (gap["id"], dg, canonical(gap)))
        return len(gaps)

    def verify_source(self, gap_id):
        row = self.db.execute("SELECT traces.* FROM traces JOIN gaps ON traces.digest=gaps.source_digest WHERE gaps.id=?", (gap_id,)).fetchone()
        if row is None:
            raise KeyError(gap_id)
        trace, provenance = json.loads(row["data"]), json.loads(row["provenance"])
        if digest(trace) != row["digest"]:
            raise Conflict("Stored source digest mismatch")
        stored_gap = json.loads(self.db.execute("SELECT data FROM gaps WHERE id=?", (gap_id,)).fetchone()[0])
        expected = next((g for g in gaps_for(trace, stored_gap["gap_policy"]) if g["id"] == gap_id), None)
        if expected != stored_gap:
            raise Conflict("Stored gap does not match its immutable source window and prompt")
        path = provenance.get("path")
        if path and (not Path(path).is_file() or hashlib.sha256(Path(path).read_bytes()).hexdigest() != provenance["file_sha256"]):
            raise Conflict("Original input changed or disappeared; restore it or import as a new source")
        return trace

    def detail(self, gap_id):
        row = self.db.execute("SELECT data FROM gaps WHERE id=?", (gap_id,)).fetchone()
        if row is None:
            raise KeyError(gap_id)
        gap = json.loads(row["data"])
        candidates = self.db.execute("SELECT * FROM candidates WHERE gap_id=? ORDER BY revision", (gap_id,)).fetchall()
        history = []
        for c in candidates:
            data = json.loads(c["data"])
            reviews = self.db.execute("SELECT * FROM reviews WHERE gap_id=? AND revision=? ORDER BY id", (gap_id, c["revision"])).fetchall()
            current = dict(reviews[-1]) if reviews else {}
            history.append({**data, "revision": c["revision"], "original_text": c["text"], "text": current.get("text", c["text"]),
                            "status": current.get("status", "pending"), "created": c["created"],
                            "review_note": current.get("note", ""), "review_tags": json.loads(current.get("tags", "[]")),
                            "reviews": [{**dict(r), "tags": json.loads(r["tags"])} for r in reviews]})
        requests = [dict(r) for r in self.db.execute("SELECT * FROM regeneration WHERE gap_id=? ORDER BY id", (gap_id,))]
        return {"gap": gap, "candidate": history[-1] if history else None, "history": history, "regeneration_requests": requests}

    def list_gaps(self):
        result = []
        for row in self.db.execute("SELECT id FROM gaps ORDER BY rowid").fetchall():
            d = self.detail(row["id"])
            g, c = d["gap"], d["candidate"]
            queued = any(not r["fulfilled"] for r in d["regeneration_requests"])
            result.append({k: g[k] for k in ("id", "trace_id", "event_index", "source_format")} | {
                "status": c["status"] if c else "ungenerated", "revision": c["revision"] if c else 0,
                "candidate": c["text"] if c else "", "flags": c.get("flags", []) if c else [], "regeneration_queued": queued})
        return result

    def add_candidate(self, gap_id, text, data, expected_revision):
        self.verify_source(gap_id)
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            return self._append_candidate(gap_id, text, data, expected_revision)

    def _append_candidate(self, gap_id, text, data, expected_revision):
        """Caller owns one SQLite transaction; candidate history is append-only."""
        current = self.db.execute("SELECT COALESCE(MAX(revision),0) FROM candidates WHERE gap_id=?", (gap_id,)).fetchone()[0]
        if current != expected_revision:
            raise Conflict("Candidate changed while generation was running")
        flags = sorted(set(data.get("flags", []) + validate_candidate(text)))
        self.db.execute("INSERT INTO candidates VALUES(?,?,?,?,?)", (gap_id, current + 1, text, canonical({**data, "flags": flags}), now()))
        self.db.execute("UPDATE regeneration SET fulfilled=1 WHERE gap_id=? AND revision<=?", (gap_id, current))
        return current + 1

    def import_candidates(self, records, import_provenance=None):
        """Atomic local import only; no model calls, reviews, or implicit approval."""
        if not isinstance(records, list) or not 1 <= len(records) <= 10:
            raise ValueError("External import accepts 1–10 candidate records per batch")
        ids = [r.get("gap_id") if isinstance(r, dict) else None for r in records]
        if any(not isinstance(gap_id, str) for gap_id in ids) or len(set(ids)) != len(ids):
            raise ValueError("Each external candidate requires a unique gap_id in this batch")
        imported = []
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            for record in records:
                gap_id = record["gap_id"]
                self.verify_source(gap_id)  # Includes current source file bytes and reconstructed window/prompt.
                gap = self.detail(gap_id)["gap"]
                metadata = validate_external(record, gap)
                if import_provenance is not None:
                    metadata["import_provenance"] = import_provenance
                revision = self._append_candidate(gap_id, record["text"], metadata, record["expected_revision"])
                imported.append({"gap_id": gap_id, "revision": revision, "status": "pending"})
        return imported

    def review(self, gap_id, revision, status, text=None, review_note="", review_tags=None):
        if status not in {"approved", "rejected", "deferred"}:
            raise ValueError("Invalid review status")
        self.verify_source(gap_id)
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            d = self.detail(gap_id)
            c = d["candidate"]
            if not c or c["revision"] != revision:
                raise Conflict("Stale candidate revision; reload before reviewing")
            text = c["text"] if text is None else text
            if not isinstance(text, str):
                raise ValueError("Review text must be a string")
            # Truncated generations can be deliberately edited to a complete candidate.
            flags = validate_candidate(text)
            if text == c["original_text"]:
                flags += c.get("flags", [])
            if status == "approved" and BLOCKING_FLAGS.intersection(flags):
                raise ValueError("Candidate has blocking validation flags; edit or regenerate before approval")
            self.db.execute("INSERT INTO reviews(gap_id,revision,status,text,note,tags,created) VALUES(?,?,?,?,?,?,?)", (gap_id, revision, status, text, review_note, canonical(review_tags or []), now()))
        return self.detail(gap_id)

    def request_regeneration(self, gap_id, revision):
        self.verify_source(gap_id)
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            c = self.detail(gap_id)["candidate"]
            if (c["revision"] if c else 0) != revision:
                raise Conflict("Stale candidate revision")
            if not self.db.execute("SELECT 1 FROM regeneration WHERE gap_id=? AND fulfilled=0", (gap_id,)).fetchone():
                self.db.execute("INSERT INTO regeneration(gap_id,revision,created) VALUES(?,?,?)", (gap_id, revision, now()))

    def generation_queue(self):
        return [g for g in self.list_gaps() if g["revision"] == 0 or g["regeneration_queued"]]

    def record_failure(self, gap_id, error):
        with self.db:
            self.db.execute("INSERT INTO failures(gap_id,error,created) VALUES(?,?,?)", (gap_id, error, now()))

    def export(self, arm="visible"):
        if arm not in {"visible", "masked", "none"}:
            raise ValueError("Unknown export arm")
        for row in self.db.execute("SELECT * FROM traces ORDER BY rowid").fetchall():
            trace = json.loads(row["data"])
            approved = {}
            for g in self.db.execute("SELECT id FROM gaps WHERE source_digest=?", (row["digest"],)).fetchall():
                d = self.detail(g["id"])
                c = d["candidate"]
                if c and c["status"] == "approved" and not any(not r["fulfilled"] for r in d["regeneration_requests"]):
                    self.verify_source(g["id"])
                    if BLOCKING_FLAGS.intersection(validate_candidate(c["text"])):
                        raise ValueError("Approved text failed validation")
                    approved[d["gap"]["event_index"]] = (d["gap"], c)
            if not approved:
                continue
            events, segments, references = [], [], []
            for i, event in enumerate(trace["events"]):
                if i in approved:
                    g, c = approved[i]
                    references.append({"gap_id": g["id"], "revision": c["revision"], "text_sha256": digest(c["text"]), "prompt_hash": g["prompt_hash"],
                                       "generator": c.get("generator", {}), "flags": c.get("flags", []), "review_note": c.get("review_note", ""), "review_tags": c.get("review_tags", [])})
                    if arm != "none":
                        rationale = {"event_id": f"synthetic:{g['id']}:{c['revision']}", "role": "assistant", "kind": "synthetic_rationale", "content": c["text"], "synthetic": True, "lookahead_conditioned": True, "before_event_id": event["event_id"]}
                        events.append(rationale)
                        segments.append({"event_id": rationale["event_id"], "loss_intent": "mask" if arm == "masked" else "train"})
                events.append(event)
                segments.append({"event_id": event["event_id"], "loss_intent": "train" if event["role"] == "assistant" else "mask"})
            yield {"schema": "cot.reviewed-events.v1", "trace_id": trace["trace_id"], "events": events,
                   "metadata": {**trace.get("metadata", {}), "source_digest": row["digest"], "synthetic_reasoning": arm != "none", "arm": arm,
                                "approved_gaps": references, "training_ready": False, "loss_segments": segments,
                                "required_next_step": "Render target chat template and verify token-aligned loss masks; structural event spans are not tokenizer masks"}}
