"""The launcher's ``fleet.jsonl`` in the run store.

``fleet.jsonl`` (island lifecycle, cost ticks, spare-node and node-health events) is a
file in the run directory of the host that runs the launch. When the control plane
moves to another host the file is split over two hosts, and a host that is lost takes
its part with it. A launch that keeps its island records in the run store (an island
fleet: ``island_lifecycle``) therefore also copies the file there, the way an island's
event tape goes (:class:`yeto.rl.engine.tape_upload.TapeUploader`): every ``interval_s``
the complete lines added since the last upload become one new object

    ``fleet/log/<lane>/<host>-p<pid>-<nonce>-<UTC start>/<seq>.jsonl``

``lane`` is ``l<k>`` for an island launch of a fleet launched island by island (learner
id ``k``) and ``fleet`` for a launch of the whole fleet; each launcher process adds its
own folder (a random nonce: two launchers of one host never share one, even with the same
pid in the same second), so no two write the same key. The copy covers this run's part of
the local file (:func:`copy_start`: a run directory reused by another run starts at the
file's end); a launcher restarted for the same run uploads its earlier lines again, and the
readers fold each record once. ``yeto down`` copies the ``island_stop`` lines it appends
(:func:`upload_appended`).

Observation only: a failed upload is a WARN on stderr and is retried with the next
chunk; it never reaches the launcher. A launch without island records in a store (one
trainer island, or no ``--rl-checkpoint-store``) uploads nothing: as before.

Readers merge every folder: ``yeto dashboard serve|export --store`` reads them next to
the island tapes (:func:`store_readers`; a record that is also in a local ``fleet.jsonl``
given with ``--tapes`` is folded once), and ``python -m yeto.dashboard.fleet_store merge
--store URI [--fleet-log PATH]...`` writes one time-ordered ``fleet.jsonl`` without
duplicates for the tools that read the file.
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

LOG_PREFIX = "fleet/log"  # chunk keys: <LOG_PREFIX>/<lane>/<incarnation>/<seq>.jsonl
FLEET_LANE = "fleet"  # the lane of a launch that runs the whole fleet
WARN_EVERY = 10  # a failing store: the first failure and then every 10th is printed
FINAL_UPLOAD_S = 15.0  # the bounded final upload at the launcher's end


def lane(learner_id: Any = None) -> str:
    """``l<k>`` for the island launch of learner ``k``, else :data:`FLEET_LANE`."""
    if isinstance(learner_id, int) and not isinstance(learner_id, bool) and learner_id >= 0:
        return f"l{learner_id}"
    return FLEET_LANE


def incarnation(*, host: str | None = None, pid: int | None = None, now: float | None = None,
                nonce: str | None = None) -> str:
    """This launcher's folder: host, pid, a random nonce and UTC start. The nonce keeps two
    launchers of one host apart even with the same pid in the same second (containers); the
    start ends the name, as the trainer tape folders do, so readers order a lane's folders
    by start."""
    host = host if host is not None else socket.gethostname()
    pid = os.getpid() if pid is None else int(pid)
    nonce = secrets.token_hex(4) if nonce is None else str(nonce)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(time.time() if now is None else now))
    return f"{host}-p{pid}-{nonce}-{stamp}"


# -- which part of the local file belongs to this store ---------------------------------------
#
# A run directory (``~/.yeto/runs/<name>/``) outlives its run: a new run of the same name
# appends to the same ``fleet.jsonl``. ``fleet.store.json`` next to it records the run (the
# run registry's ``started_at``, new for each run of the name: :func:`run_identity`), the store
# the file is copied to and the byte offset the copy starts at. A launch of the same run for
# the same store goes on from that offset (a restarted launcher of the run uploads its earlier
# lines again: the readers fold each record once); any other launch, or a file without the
# record (written before this copy existed), starts at the file's end, so an earlier run's
# lines never reach this run's store (the rule of ``tape_upload.syncer_uploaders(from_end=True)``).
# A launch that does not copy the file voids the record (:func:`copy_off`): no store, the file's
# end. ``yeto down`` copies its lines only for the run the record names.

COPY_RECORD = "fleet.store.json"
COPY_SCHEMA = "yeto.fleet.log-copy/v1"


def copy_record_path(path: str | os.PathLike) -> Path:
    return Path(path).with_name(COPY_RECORD)


def read_copy_record(path: str | os.PathLike) -> dict | None:
    try:
        doc = json.loads(copy_record_path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) and doc.get("schema") == COPY_SCHEMA and doc.get("store") else None


def run_identity(name: str) -> float | None:
    """The run of ``name`` the copy record names: its run registry ``started_at`` (set anew by
    every run of the name, kept by its launcher restarts and its ``yeto down``), or None."""
    try:
        from yeto import runs  # noqa: PLC0415

        started = (runs.load_run(name) or {}).get("started_at")
        return float(started) if started is not None else None
    except Exception:  # noqa: BLE001 - telemetry never stops a launch or a down
        return None


def _size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


def _write_record(p: Path, doc: dict) -> None:
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        copy_record_path(p).write_text(json.dumps(doc, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as exc:
        _stderr(f"[launcher] WARN: fleet.jsonl store copy: {COPY_RECORD} not written ({type(exc).__name__})")


def copy_start(path: str | os.PathLike, store: str, *, learner_id: Any = None, run: float | None = None) -> int:
    """The byte offset of ``path`` this launch's copy into ``store`` starts at (see above),
    recorded for the next launch and for ``yeto down``. ``run``: :func:`run_identity` (None:
    the file's end). Never raises (a record that cannot be written: the offset still holds for
    this launch)."""
    p = Path(path)
    size = _size(p)
    known = read_copy_record(p)
    same = known is not None and known.get("store") == str(store) and run is not None and known.get("run") == run
    start = min(int(known.get("start") or 0), size) if same else size
    _write_record(p, {"schema": COPY_SCHEMA, "store": str(store), "start": start, "run": run,
                      "learner_id": learner_id if lane(learner_id) != FLEET_LANE else None})
    return start


def copy_off(path: str | os.PathLike, *, run: float | None = None) -> None:
    """A launch that does not copy ``path`` (no store, no island records, the copy turned
    off): its record, if any, names no store from the file's end on, so neither ``yeto down``
    nor a later launch of another run copies this launch's lines into an earlier run's
    store. A file without a record keeps none (one-island launches write nothing). Never raises."""
    p = Path(path)
    if copy_record_path(p).exists():
        _write_record(p, {"schema": COPY_SCHEMA, "store": None, "start": _size(p), "run": run, "learner_id": None})


class _Warn:
    """The uploader's log: a WARN line for the first failure and every :data:`WARN_EVERY`-th."""

    def __init__(self, out: Callable[[str], None]) -> None:
        self.out = out
        self.count = 0

    def __call__(self, message: str) -> None:
        self.count += 1
        if self.count == 1 or self.count % WARN_EVERY == 0:
            self.out(f"[launcher] WARN: fleet.jsonl store copy ({self.count} failure(s)): "
                     f"{message.removeprefix('[tape-upload] ')}"[:300])


def _stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def start_uploader(objects: Any, path: str | os.PathLike, *, learner_id: Any = None, environ: Any = None,
                   log: Callable[[str], None] = _stderr, interval_s: float | None = None, offset: int = 0) -> Any:
    """The started copy of ``path`` from byte ``offset`` into ``objects`` (module doc), or
    None: the upload interval is 0 (``YETO_RL_TAPE_UPLOAD_S``), or it could not start (a
    WARN). An explicit ``interval_s`` must be positive (ValueError: a caller's bug); otherwise
    never raises."""
    if interval_s is not None and not interval_s > 0:
        raise ValueError(f"fleet.jsonl store copy: interval_s must be > 0 (got {interval_s!r}); "
                         "turn the copy off with YETO_RL_TAPE_UPLOAD_S=0")
    try:
        from yeto.rl.engine.tape_upload import TapeUploader, upload_interval_s  # noqa: PLC0415

        interval = upload_interval_s(None, environ) if interval_s is None else interval_s
        if interval is None:
            return None
        uploader = TapeUploader(objects, path, island=lane(learner_id), incarnation=incarnation(), interval_s=interval,
                                log=_Warn(log), prefix=LOG_PREFIX)
        uploader.offset = max(0, int(offset))
        return uploader.start()
    except Exception as exc:  # noqa: BLE001 - telemetry never stops a launch
        log(f"[launcher] WARN: fleet.jsonl store copy not started ({type(exc).__name__}: {exc})"[:300])
        return None


def upload_appended(path: str | os.PathLike, offset: int, *, run: float | None = None, objects: Any = None,
                    timeout_s: float | None = None, log: Callable[[str], None] = _stderr) -> bool:
    """``yeto down``: the lines appended to ``path`` from ``offset`` on (the ``island_stop``
    of the islands a killed launcher left open) into the store its copy record names, in the
    record's lane, as one more folder; bounded. Nothing without a copy record of this ``run``
    (:func:`run_identity`; a run whose fleet.jsonl stays local, or a record another run left)
    or without new lines. A failure is a WARN; never raises. Returns True when the store has
    every new line (or there is nothing to copy)."""
    try:
        record = read_copy_record(path)
        if record is None or run is None or record.get("run") != run:
            return True
        size = Path(path).stat().st_size
        if size <= offset:
            return True
        if objects is None:
            from yeto.launch.island_lifecycle import store_objects  # noqa: PLC0415

            objects = store_objects(record["store"])
        if objects is None:
            return True
        from yeto.rl.engine.tape_upload import TapeUploader  # noqa: PLC0415

        uploader = TapeUploader(objects, path, island=lane(record.get("learner_id")), incarnation=incarnation(),
                                interval_s=FINAL_UPLOAD_S, log=_Warn(log), prefix=LOG_PREFIX)
        uploader.offset = int(offset)
        return bool(uploader.stop(FINAL_UPLOAD_S if timeout_s is None else timeout_s))
    except Exception as exc:  # noqa: BLE001 - telemetry never fails a down
        log(f"[yeto] WARN: fleet.jsonl store copy: island_stop lines not copied ({type(exc).__name__}: {exc})"[:300])
        return False


def stop_uploader(uploader: Any, timeout_s: float = FINAL_UPLOAD_S) -> bool:
    """The final upload, bounded; True when the store has every complete line. Never raises."""
    if uploader is None:
        return True
    try:
        return bool(uploader.stop(timeout_s))
    except Exception as exc:  # noqa: BLE001
        _stderr(f"[launcher] WARN: fleet.jsonl store copy: final upload failed ({type(exc).__name__})")
        return False


# -- readers ------------------------------------------------------------------------------------

def log_reader(objects: Any, *, label: str = "store-fleet", interval_s: float = 0.0) -> Any:
    """A :class:`~yeto.dashboard.store_tapes.StoreTapes` over the store's fleet logs: each
    launcher folder is one source, its chunks read once and in order, the folders of a lane
    in the order they started."""
    from .store_tapes import StoreTapes  # noqa: PLC0415

    return StoreTapes(objects, label=label, interval_s=interval_s, prefix=LOG_PREFIX + "/")


class StoreReaders:
    """Several store readers as one (``serve`` / ``export`` take one ``store``): the island
    tapes first, then the fleet logs. ``objects`` and ``label`` are the first reader's."""

    def __init__(self, *readers: Any) -> None:
        self.readers = [r for r in readers if r is not None]
        self.objects = self.readers[0].objects
        self.label = getattr(self.readers[0], "label", "store")

    def pump(self, reducer: Any) -> int:
        return sum(int(r.pump(reducer) or 0) for r in self.readers)

    @property
    def sources(self) -> set[str]:
        return set().union(*(set(getattr(r, "sources", ())) for r in self.readers))

    @property
    def errors(self) -> int:
        return sum(int(getattr(r, "errors", 0) or 0) for r in self.readers)

    def status(self) -> dict:
        """The first reader's status; failures, errors, requests and sources of all."""
        rows = [r.status() for r in self.readers]
        out = dict(rows[0])
        for key in ("sources", "requests", "errors", "failing"):
            out[key] = sum(int(row.get(key) or 0) for row in rows)
        failed = [row for row in rows if row.get("last_error_at") is not None]
        if failed:
            last = max(failed, key=lambda row: row["last_error_at"])
            out["last_error"], out["last_error_at"] = last.get("last_error"), last["last_error_at"]
        oks = [row.get("last_ok_at") for row in rows]
        out["last_ok_at"] = None if any(v is None for v in oks) else min(oks)
        out["polled"] = all(bool(row.get("polled")) for row in rows)
        return out


def store_readers(tapes: Any, *, interval_s: float = 0.0) -> StoreReaders:
    """The dashboard's store: the island tapes ``tapes`` and the fleet logs of its store."""
    return StoreReaders(tapes, log_reader(tapes.objects, interval_s=interval_s))


def _jsonl(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _time(rec: dict) -> float | None:
    v = rec.get("time_unix")
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def merge_records(groups: Iterable[Iterable[dict]]) -> list[dict]:
    """Every record of ``groups`` once (equal JSON is one record), in time order; a record
    without ``time_unix`` keeps the time of the record before it in its group."""
    seen: set[str] = set()
    rows: list[tuple[float, int, dict]] = []
    for group in groups:
        last = float("-inf")
        for rec in group:
            key = json.dumps(rec, sort_keys=True, separators=(",", ":"), default=str)
            t = _time(rec)
            last = last if t is None else t
            if key in seen:
                continue
            seen.add(key)
            rows.append((last, len(rows), rec))
    rows.sort(key=lambda row: (row[0], row[1]))
    return [rec for _, _, rec in rows]


def read_store(objects: Any) -> list[list[dict]]:
    """The records of every fleet log folder in the store, one list per folder."""
    reader = log_reader(objects)
    by_source: dict[str, list[dict]] = {}

    class _BySource:  # a reducer stand-in: every record, by source
        def feed(self, record: Any, *, source: str = "-", offset: int | None = None, **_: Any) -> bool:
            if isinstance(record, dict):
                by_source.setdefault(source, []).append(record)
                return True
            return False

    reader.pump(_BySource())
    if reader.failing:
        raise OSError(f"fleet logs not read from the store: {reader.last_error}")
    return list(by_source.values())


def main(argv: Any = None) -> int:
    """``merge --store URI [--fleet-log PATH]... [--out PATH]``: one ``fleet.jsonl``."""
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(prog="python -m yeto.dashboard.fleet_store")
    sub = parser.add_subparsers(dest="cmd", required=True)
    mg = sub.add_parser("merge", help="the run's fleet.jsonl from the store (and local copies), merged")
    mg.add_argument("--store", default=None, help="the run store (s3://, oss:// or a directory)")
    mg.add_argument("--fleet-log", action="append", default=[], help="a local fleet.jsonl (repeatable)")
    mg.add_argument("--out", default="-", help="output file (default: stdout)")
    args = parser.parse_args(argv)
    if not args.store and not args.fleet_log:
        parser.error("merge needs --store and/or --fleet-log")
    groups: list[list[dict]] = []
    if args.store:
        from .store_tapes import open_objects  # noqa: PLC0415

        try:
            groups += read_store(open_objects(args.store))
        except OSError as exc:
            print(f"[fleet-log] {exc}", file=sys.stderr)
            return 2
    for path in args.fleet_log:
        groups.append(_jsonl(Path(path).expanduser().read_text(encoding="utf-8", errors="replace")))
    merged = merge_records(groups)
    text = "".join(json.dumps(rec, ensure_ascii=False, default=str) + "\n" for rec in merged)
    if args.out == "-":
        sys.stdout.write(text)
    else:
        Path(args.out).expanduser().write_text(text, encoding="utf-8")
    print(f"[fleet-log] {len(merged)} record(s) from {len(groups)} log(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
