"""Reuse an untouched completed source index in a new immutable run identity."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sqlite3

from .core import canonical, digest
from .corpus_source import file_sha256
from .corpus_worker import Journal, now


def _functions(path):
    module = ast.parse(Path(path).read_text())
    gap = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "gap_at")
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Journal")
    index = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_index")
    return {"gap_at": ast.dump(gap, include_attributes=False), "Journal._index": ast.dump(index, include_attributes=False)}


class DerivedIndexJournal(Journal):
    def __init__(self, path, source, source_sha256, identity, *, index_template,
                 template_worker_path, expected_sources, expected_gaps):
        self.template = Path(index_template).resolve(strict=True)
        self.template_worker = Path(template_worker_path).resolve(strict=True)
        if Path(path).resolve() == self.template:
            raise ValueError("A derived journal must not overwrite the index template")
        self.expected_sources, self.expected_gaps = expected_sources, expected_gaps
        for count in (expected_sources, expected_gaps):
            if type(count) is not int or count < 1:
                raise ValueError("Derived index needs exact positive source and gap counts")
        parent_sha = file_sha256(self.template)
        self.template_sha256 = parent_sha
        identity = {**identity, "index_parent": {"path": str(self.template),
                    "file_sha256": parent_sha, "template_worker_sha256": file_sha256(self.template_worker),
                    "policy": "copy-verified-empty-original-index/v1",
                    "expected_sources": expected_sources, "expected_gaps": expected_gaps},
                    "derived_index_implementation_sha256": file_sha256(__file__)}
        try:
            super().__init__(path, source, source_sha256, identity)
        except Exception:
            if hasattr(self, "db"):
                self.db.close()
            raise

    def _index(self):
        if self.db.execute("SELECT 1 FROM meta WHERE key='index_complete'").fetchone():
            return
        old = sqlite3.connect(self.template.as_uri() + "?mode=ro", uri=True)
        old.row_factory = sqlite3.Row
        try:
            old.execute("BEGIN")
            parent = json.loads(old.execute("SELECT value FROM meta WHERE key='identity'").fetchone()[0])
            if (parent.get("source_path") != str(self.source)
                    or parent.get("source_sha256") != self.identity["source_sha256"]
                    or parent.get("prompt_version") != self.identity["prompt_version"]):
                raise ValueError("Index source or generator prompt identity changed")
            if not old.execute("SELECT 1 FROM meta WHERE key='index_complete'").fetchone():
                raise ValueError("Index template is not complete")
            for table in ("responses", "candidates", "reviews", "failures"):
                if old.execute("SELECT count(*) FROM " + table).fetchone()[0]:
                    raise ValueError("Only an untouched source index can be derived")
            if old.execute("SELECT count(*) FROM gaps WHERE state!='queued' OR invocation IS NOT NULL").fetchone()[0]:
                raise ValueError("Index template has already been claimed")
            from . import corpus_worker
            current_worker = Path(corpus_worker.__file__)
            impl = parent.get("implementation_sha256", {})
            if (file_sha256(self.template_worker) != impl.get("corpus_worker.py")
                    or file_sha256(current_worker.with_name("core.py")) != impl.get("core.py")
                    or _functions(self.template_worker) != _functions(current_worker)):
                raise ValueError("Original index/gap construction implementation does not match")
            sources = [dict(row) for row in old.execute("SELECT * FROM sources ORDER BY id")]
            if len(sources) != self.expected_sources:
                raise ValueError("Unexpected original source count")
            position = 0
            source_digests = {}
            logical = hashlib.sha256(canonical(parent).encode())
            for index, source in enumerate(sources, 1):
                if source["id"] != index or source["offset"] != position or source["length"] < 1:
                    raise ValueError("Original source byte coverage is not contiguous")
                position += source["length"]
                source_digests[index] = source["source_digest"]
                logical.update(canonical(source).encode())
            if position != self.source.stat().st_size:
                raise ValueError("Original source index does not cover the complete file")
            count = 0
            with self.db:
                self.db.executemany("INSERT INTO sources VALUES(?,?,?,?,?,?)",
                    [tuple(row[key] for key in ("id", "trace_id", "offset", "length", "row_sha256", "source_digest")) for row in sources])
                cursor = old.execute("SELECT * FROM gaps ORDER BY ordinal")
                while True:
                    batch = cursor.fetchmany(4096)
                    if not batch:
                        break
                    values = []
                    for raw in batch:
                        row = dict(raw);count += 1
                        if row["ordinal"] != count or row["source_id"] not in source_digests or row["event_index"] < 0:
                            raise ValueError("Original gap coverage or source reference is invalid")
                        expected = digest({"source_digest": source_digests[row["source_id"]],
                                           "event_id": row["event_id"], "window_version": "original-visible/v1"})
                        if row["id"] != expected:
                            raise ValueError("Original gap identity is inconsistent")
                        logical.update(canonical(row).encode())
                        values.append(tuple(row[key] for key in ("ordinal", "id", "source_id", "event_index", "event_id", "state", "invocation", "updated")))
                    self.db.executemany("INSERT INTO gaps VALUES(?,?,?,?,?,?,?,?)", values)
                if count != self.expected_gaps:
                    raise ValueError("Unexpected original gap count")
                self.verify_file()
                if file_sha256(self.template) != self.template_sha256:
                    raise ValueError("Index template changed during derivation")
                self.db.execute("INSERT INTO meta VALUES('index_derivation',?)", (canonical({
                    "parent_logical_sha256": logical.hexdigest(), "source_rows": len(sources),
                    "gap_rows": count, "actual_source_sha256_verified": True,
                    "every_gap_id_verified": True, "original_functions_identical": True}),))
                self.db.execute("INSERT INTO meta VALUES('index_complete',?)", (now(),))
        finally:
            old.close()
