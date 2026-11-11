"""Island tapes uploaded to the run store (``yeto.rl.engine.tape_upload``).

``--store s3://bucket/runs/<run>`` (or a local / shared store directory) makes
serve and export read every ``tapes/<island>/<incarnation>/<seq>.jsonl`` chunk:
islands on leased nodes appear without a per-node ``dashboard mirror``, and a
replacement node shows up with its first upload. Each (island, incarnation) is one
source; chunks are read once, in sequence order, and the offsets passed to the
reducer keep a re-read from double counting.
"""

from __future__ import annotations

import json
import os
from typing import Any

PREFIX = "tapes/"
_SEQ_SPAN = 1 << 32  # offset = seq * span + byte position inside the chunk


def open_objects(store: str, environ: Any = None) -> Any:
    """The object API for ``store``: an ``s3://`` URI (credentials from the
    environment, as for the launcher) or a directory path."""
    if "://" in store:
        from yeto.cloud.s3_store import head_objects  # noqa: PLC0415

        return head_objects(store, environ=environ)
    from yeto.rl.engine.object_store import DirectoryObjectStore  # noqa: PLC0415

    return DirectoryObjectStore(os.path.expanduser(store))


class StoreTapes:
    def __init__(self, objects: Any, *, label: str = "store") -> None:
        self.objects = objects
        self.label = label
        self.seen: set[str] = set()
        self.sources: set[str] = set()
        self.errors = 0
        self.last_error: str | None = None

    def pump(self, reducer) -> int:
        try:
            keys = sorted(k for k in self.objects.list(PREFIX) if k.endswith(".jsonl") and k not in self.seen)
        except Exception as error:  # noqa: BLE001 - a store hiccup must not stop the page
            self.errors += 1
            self.last_error = f"{type(error).__name__}: {error}"[:300]
            return 0
        n, blocked = 0, set()
        for key in keys:
            parts = key[len(PREFIX):].split("/")
            if len(parts) != 3:
                self.seen.add(key)
                continue
            island, inc, chunk = parts
            if (island, inc) in blocked:
                continue  # an earlier chunk of this tape failed: keep the order
            try:
                seq = int(chunk.split(".")[0])
                data = self.objects.read(key)
            except Exception as error:  # noqa: BLE001
                self.errors += 1
                self.last_error = f"{type(error).__name__}: {error}"[:300]
                blocked.add((island, inc))
                continue  # retried at the next poll
            self.seen.add(key)
            if data is None:
                continue
            source = f"{self.label}:{island}/{inc}"
            self.sources.add(source)
            pos = 0
            for line in data.splitlines(keepends=True):
                pos += len(line)
                text = line.strip()
                if not text:
                    continue
                try:
                    rec = json.loads(text)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(rec, dict) and reducer.feed(rec, source=source, offset=seq * _SEQ_SPAN + pos):
                    n += 1
        return n
