"""``--data-mix``: several prompt data sets with weights (S27 P3).

``--data-mix 'a.jsonl=0.7,org/ds@rev=0.3'`` names each source (a local path
or an HF dataset id with an optional ``@revision``) and its weight. Before
anything pins ``--data``, the launcher reads every source and writes the rows,
source by source, into one prompt file in the run directory; each row records
its source index in ``metadata["yeto_data_source"]``. That file becomes the
run's ``--data`` (so provenance, head staging and the island file mounts treat
it as any local prompt file), and the weights reach the islands as
``--rl-dispense-mix``. The task dispenser keeps one stream per source (per
source and task label with ``--rl-island-task-labels``) and interleaves them
by weight (``yeto.rl.engine.task_dispense`` module doc), so a mix needs
``--rl-data-split dispense``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

MIX_FILE = "data-mix.jsonl"


@dataclass(frozen=True)
class MixSource:
    source: str
    revision: str | None
    weight: str  # the decimal the user wrote (the islands parse it the same way)

    @property
    def label(self) -> str:
        return self.source if self.revision is None else f"{self.source}@{self.revision}"


def parse_data_mix(spec: str) -> list[MixSource]:
    """``'a.jsonl=0.7,org/ds@rev=0.3'`` -> sources in order. Raises ValueError."""
    out: list[MixSource] = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        source, sep, weight = item.rpartition("=")
        if not sep or not source.strip() or not weight.strip():
            raise ValueError(f"--data-mix item {item!r}: expected SOURCE=WEIGHT")
        try:
            value = Fraction(weight.strip())
        except (ValueError, ZeroDivisionError):
            raise ValueError(f"--data-mix item {item!r}: weight {weight!r} is not a number") from None
        if value <= 0:
            raise ValueError(f"--data-mix item {item!r}: the weight must be positive")
        source = source.strip()
        revision = None
        local = os.path.exists(os.path.expanduser(source))
        if not local and "@" in source:
            source, _, revision = source.rpartition("@")
            if not source or not revision:
                raise ValueError(f"--data-mix item {item!r}: expected org/name@revision")
        out.append(MixSource(source, revision, weight.strip()))
    if len(out) < 2:
        raise ValueError("--data-mix needs two or more SOURCE=WEIGHT items (one data set: use --data)")
    if len({s.label for s in out}) != len(out):
        raise ValueError("--data-mix names one source twice")
    return out


def _load(source: MixSource) -> list[dict[str, Any]]:
    from ..data import load_rows

    if source.revision is None and not os.path.exists(os.path.expanduser(source.source)):
        rows = load_rows(source.source)
    else:
        rows = load_rows(source.source, revision=source.revision)
    out = [dict(row) for row in rows]
    if not out:
        raise ValueError(f"--data-mix source {source.label} has no rows")
    return out


def write_mix(sources: list[MixSource], path: Path, *, load=_load) -> list[dict[str, Any]]:
    """Write the rows of ``sources`` into ``path`` (module doc); returns one
    record per source (rows, weight, SHA-256 of the rows written)."""
    from ..rl.engine.task_dispense import DATA_SOURCE_KEY

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    records = []
    with temporary.open("w", encoding="utf-8") as handle:
        for index, source in enumerate(sources):
            digest, count = hashlib.sha256(), 0
            for row in load(source):
                metadata = dict(row.get("metadata") or {})
                metadata[DATA_SOURCE_KEY] = index
                row["metadata"] = metadata
                line = json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str) + "\n"
                digest.update(line.encode("utf-8"))
                handle.write(line)
                count += 1
            records.append({"source": source.label, "weight": source.weight, "rows": count,
                            "sha256": digest.hexdigest()})
    os.replace(temporary, path)
    return records


def materialize_data_mix(args, *, load=_load) -> Path | None:
    """``--data-mix`` -> the run's ``--data`` file and ``--rl-dispense-mix``
    (module doc); None without ``--data-mix``. Idempotent: a run whose mix was
    written already (``args.data_mix`` consumed) is left alone."""
    spec = getattr(args, "data_mix", None)
    if not spec:
        return None
    if getattr(args, "training_mode", "rl") != "rl":
        raise ValueError("--data-mix is an RL option (--training-mode rl)")
    if getattr(args, "data", None):
        raise ValueError("--data-mix replaces --data; give one of them")
    if getattr(args, "data_revision", None):
        raise ValueError("--data-mix: give each HF source its revision as org/name@revision, "
                         "not --data-revision")
    from .. import runs

    sources = parse_data_mix(spec)
    path = runs.run_dir(args.cluster_prefix) / MIX_FILE
    records = write_mix(sources, path, load=load)
    args.data = str(path)
    args.rl_dispense_mix = ",".join(s.weight for s in sources)
    args.data_mix_record = {"spec": str(spec), "file": str(path), "sources": records}
    args.data_mix = None  # consumed: the head and the worker see --data
    total = sum(r["rows"] for r in records)
    weights = [Fraction(s.weight) for s in sources]
    share = ", ".join(f"{r['source']} {float(w / sum(weights)):.3f} ({r['rows']} rows)"
                      for r, w in zip(records, weights))
    print(f"[launcher] data mix: {total} rows from {len(records)} sources into {path}: {share}",
          file=sys.stderr, flush=True)
    return path
