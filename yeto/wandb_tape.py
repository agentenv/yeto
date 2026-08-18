"""Syncer event tape -> W&B, without touching the Rust syncer.

The syncer already writes one JSONL record per merge (``--event-tape``,
see ``append_tape`` in ``syncer/src/server.rs``), and that record carries
exactly the merge-side signals an async DiLoCo fleet is debugged with:
how long quorum took, who missed the grace window, how stale each
island's base was, and how much each island actually contributed to the
outer step.

So the syncer needs no telemetry code of its own. In head-controller mode
the tape sits on the same VM as the controller, and a daemon thread here
tails it into a W&B run with ``job_type="syncer"``. The same module
replays a finished tape after the fact::

    python -m yeto.wandb_tape ~/yeto-tape.jsonl --wandb-project yeto \
        --wandb-group my-run

Both paths share ``tape_metrics``, which is a pure record -> metrics dict
function and is what the tests pin.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path

from .wandb_logger import NullRun, WandbRun

log = logging.getLogger("wandb-tape")

# Poll interval while the tape has no new complete line.
POLL_SECONDS = 1.0


def tape_metrics(rec: dict) -> dict:
    """One merge record -> the metrics logged for it.

    Keys the syncer emits as null (``quorum_ms``/``grace_ms`` before a
    round has waited on either) are omitted rather than logged as zero: a
    zero wait and "never waited" are different states, and conflating
    them makes the grace-window curve unreadable.
    """
    responders = rec.get("responders") or []
    expected = rec.get("expected") or []
    responded = rec.get("responded") or []
    missed = rec.get("missed_grace") or []

    metrics: dict[str, float | int] = {
        "global_step": int(rec.get("step", 0)),
        "sync/fragment": int(rec.get("fragment", 0)),
        "sync/attempt": int(rec.get("attempt", 0)),
        "sync/gnorm": float(rec.get("gnorm", 0.0)),
        "sync/merge_ms": float(rec.get("ms", 0.0)),
        "sync/sync_ms": float(rec.get("sync_ms", 0.0)),
        "sync/quorum": int(rec.get("quorum", 0)),
        "sync/expected": len(expected),
        "sync/responded": len(responded),
        "sync/missed": len(missed),
        "sync/launch_base_version": int(rec.get("launch_base_version", 0)),
    }
    if expected:
        metrics["sync/participation"] = len(responded) / len(expected)
    for optional in ("quorum_ms", "grace_ms"):
        value = rec.get(optional)
        if value is not None:
            metrics[f"sync/{optional}"] = float(value)

    stalenesses = [float(r.get("staleness", 0)) for r in responders]
    if stalenesses:
        metrics["sync/staleness_max"] = max(stalenesses)
        metrics["sync/staleness_mean"] = sum(stalenesses) / len(stalenesses)

    for responder in responders:
        if "id" not in responder:
            continue
        learner = f"learner/{int(responder['id'])}"
        metrics[f"{learner}/staleness"] = float(responder.get("staleness", 0))
        metrics[f"{learner}/contribution"] = float(responder.get("contribution", 0.0))
        metrics[f"{learner}/weight"] = float(responder.get("weight", 0.0))
        metrics[f"{learner}/c_steps"] = float(responder.get("c_steps", 0))
        metrics[f"{learner}/c_tokens"] = float(responder.get("c_tokens", 0))
        metrics[f"{learner}/base_version"] = float(responder.get("base_version", 0))
    # A missing island logs an explicit 0 so its participation curve shows
    # the dropout instead of flatlining at its last contributed value.
    for learner_id in missed:
        metrics[f"learner/{int(learner_id)}/contribution"] = 0.0
    return metrics


def follow_jsonl(
    path: str | os.PathLike,
    *,
    from_start: bool = True,
    stop: threading.Event | None = None,
    poll_seconds: float = POLL_SECONDS,
) -> Iterator[dict]:
    """Yield JSON objects from a growing JSONL file.

    Only complete, newline-terminated lines are parsed: the syncer
    appends with a single ``write_all`` per record, but a reader can
    still catch a short write, and half a record is not a merge.
    Returns when ``stop`` is set; without one it follows forever.
    """
    path = Path(os.path.expanduser(str(path)))
    while not path.exists():
        if stop is not None and stop.wait(poll_seconds):
            return
        if stop is None:
            time.sleep(poll_seconds)
    with open(path, encoding="utf-8", errors="replace") as f:
        if not from_start:
            f.seek(0, os.SEEK_END)
        while True:
            position = f.tell()
            line = f.readline()
            if not line.endswith("\n"):
                # Incomplete tail (or EOF): rewind and wait for the rest.
                f.seek(position)
                if stop is not None and stop.is_set():
                    return
                if stop is not None:
                    if stop.wait(poll_seconds):
                        return
                else:
                    time.sleep(poll_seconds)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                log.warning("skipping unparsable event-tape line: %s", e)


class TapeForwarder:
    """Daemon thread tailing an event tape into a W&B run."""

    def __init__(self, run: NullRun | WandbRun, tape_path: str | os.PathLike):
        self.run = run
        self.tape_path = tape_path
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.records = 0

    def start(self) -> None:
        if not getattr(self.run, "enabled", False):
            return
        self._thread = threading.Thread(
            target=self._forward, name="yeto-wandb-tape", daemon=True
        )
        self._thread.start()
        log.info("forwarding syncer event tape %s to W&B", self.tape_path)

    def _forward(self) -> None:
        try:
            for rec in follow_jsonl(self.tape_path, stop=self._stop):
                self.run.log(tape_metrics(rec))
                self.records += 1
        except Exception as e:  # noqa: BLE001 - telemetry never breaks the run
            log.warning("event-tape forwarding stopped: %s", e)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self.run.summary({"sync/tape_records": self.records})


def replay(
    tape_path: str | os.PathLike, run: NullRun | WandbRun
) -> int:
    """Log every record of a finished tape; returns the record count."""
    count = 0
    with open(os.path.expanduser(str(tape_path)), encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning("skipping unparsable event-tape line: %s", e)
                continue
            run.log(tape_metrics(rec))
            count += 1
    run.summary({"sync/tape_records": count})
    return count


def main(argv=None) -> int:
    """Replay a finished event tape into a W&B run (offline analysis)."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("tape", help="path to a syncer --event-tape JSONL file")
    p.add_argument("--wandb-project", default="yeto")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-group", default=None, help="run name / --cluster-prefix")
    p.add_argument("--wandb-mode", choices=["online", "offline"], default="online")
    p.add_argument("--follow", action="store_true", help="tail a live tape instead")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s wandb-tape %(levelname)s %(message)s")
    from .wandb_logger import init

    args.wandb = True
    run = init(
        args,
        job_type="syncer",
        name="syncer",
        group=args.wandb_group,
        step_metrics={"sync/*": "global_step", "learner/*": "global_step"},
    )
    if not run.enabled:
        print("[yeto] W&B is unavailable; nothing was logged.")
        return 1
    if args.follow:
        forwarder = TapeForwarder(run, args.tape)
        forwarder.start()
        try:
            while True:
                time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            forwarder.stop()
    else:
        count = replay(args.tape, run)
        print(f"[yeto] replayed {count} merge records into W&B")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
