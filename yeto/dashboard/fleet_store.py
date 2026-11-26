"""The launcher's ``fleet.jsonl`` in the run store.

``fleet.jsonl`` (island lifecycle, cost ticks, spare-node and node-health events) is a
file in the run directory of the host that runs the launch. When the control plane
moves to another host the file is split over two hosts, and a host that is lost takes
its part with it. A launch that keeps its island records in the run store (an island
fleet: ``island_lifecycle``) therefore also copies the file there, the way an island's
event tape goes (:class:`yeto.rl.engine.tape_upload.TapeUploader`): every ``interval_s``
the complete lines added since the last upload become one new object

    ``fleet/log/<lane>/<host>-p<pid>-<UTC start>/<seq>.jsonl``

``lane`` is ``l<k>`` for an island launch of a fleet launched island by island (learner
id ``k``) and ``fleet`` for a launch of the whole fleet; each launcher process adds its
own folder, so two hosts (or two launches of one host) never write the same key. The
copy starts at the beginning of the local file: a launcher restarted on the same host
uploads the earlier records again, and the readers fold each record once.

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
import socket
import sys
import time
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


def incarnation(*, host: str | None = None, pid: int | None = None, now: float | None = None) -> str:
    """This launcher's folder: host, pid and UTC start (the start ends the name, as the
    trainer tape folders do, so readers order a lane's folders by start)."""
    host = host if host is not None else socket.gethostname()
    pid = os.getpid() if pid is None else int(pid)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(time.time() if now is None else now))
    return f"{host}-p{pid}-{stamp}"


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
                   log: Callable[[str], None] = _stderr, interval_s: float | None = None) -> Any:
    """The started copy of ``path`` into ``objects`` (module doc), or None: the upload
    interval is 0 (``YETO_RL_TAPE_UPLOAD_S``), or it could not start (a WARN). Never raises."""
    try:
        from yeto.rl.engine.tape_upload import TapeUploader, upload_interval_s  # noqa: PLC0415

        interval = upload_interval_s(None, environ) if interval_s is None else interval_s
        if interval is None or interval <= 0:
            return None
        uploader = TapeUploader(objects, path, island=lane(learner_id), incarnation=incarnation(), interval_s=interval,
                                log=_Warn(log), prefix=LOG_PREFIX)
        return uploader.start()
    except Exception as exc:  # noqa: BLE001 - telemetry never stops a launch
        log(f"[launcher] WARN: fleet.jsonl store copy not started ({type(exc).__name__}: {exc})"[:300])
        return None


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
    from pathlib import Path  # noqa: PLC0415

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
