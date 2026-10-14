"""Event-tape collection, recovery and island-departure accounting."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

from ..rl.event_echo import TapeCollector as EventCollector


def wait_for_tapes(collectors: dict, names, threads, limit: float, *,
                   clock=time.monotonic, sleep=time.sleep, poll: float = 0.2) -> str:
    """Bounded, event-based wait before the tapes are judged complete.

    Returns ``"finalized"`` as soon as every expected island's collector holds its
    ``rl_learner_finalized`` record, ``"streams_ended"`` when every log stream
    ended (nothing more can arrive), else ``"deadline"`` after ``limit`` s.
    """
    deadline = clock() + limit
    while True:
        if set(collectors) >= set(names) and all(c.finalized for c in collectors.values()):
            return "finalized"
        if not any(t.is_alive() for t in threads):
            return "streams_ended"
        if clock() >= deadline:
            return "deadline"
        sleep(poll)


ECHO_TAPE_RECOVER_TIMEOUT_S = 180.0


def _recover_echo_tape(cluster: str, collector, *, run=None, timeout: float = ECHO_TAPE_RECOVER_TIMEOUT_S) -> bool:
    """A no-sync sky island whose echoed tape stayed incomplete: fetch the
    island's own tape files (``~/yeto-output/rl-island-*.jsonl``, the source
    the echo mirrors) over the ssh alias sky wrote and complete the local tape
    from them. The S1 2x1 L40S runs (2026-10-03/04) showed ``sky.tail_logs``
    delivering the setup lines but no run-phase line at all (root cause: the
    heartbeat ``None`` ended ``_tail``'s loop, see there; fixed, kept as the
    fallback), so the stream alone cannot be the only source. Returns whether the tape is finalized;
    any fetch failure leaves it incomplete (fail closed) and is printed."""
    import tempfile

    run = run or subprocess.run
    with tempfile.TemporaryDirectory(prefix="yeto-tape-") as tmp:
        cmd = ["rsync", "-az", f"{cluster}:yeto-output/rl-island-*.jsonl", f"{tmp}/"]
        try:
            run(cmd, check=True, timeout=timeout, capture_output=True)
        except Exception as e:  # noqa: BLE001 - subprocess/timeout: report, stay incomplete
            print(f"[launcher] {cluster}: tape recovery fetch failed ({e})", file=sys.stderr)
            return False
        files = sorted(Path(tmp).glob("rl-island-*.jsonl"))
        if not files:
            print(f"[launcher] {cluster}: tape recovery found no rl-island-*.jsonl", file=sys.stderr)
            return False
        before = collector.count
        complete = False
        for f in files:
            complete = collector.recover_from_file(f) or complete
        print(f"[launcher] {cluster}: tape recovered from {[f.name for f in files]}: "
              f"+{collector.count - before} record(s), finalized={complete}")
        return complete


def settle_echo_tapes(collectors: dict, names, modal_cfgs, events_dir, *,
                      recover_sky=None, modal_tape_dir=None) -> list[str]:
    """Judge every expected island's echoed tape after the run: close it (fail
    closed: nothing is written afterwards) and, when the log stream did not
    deliver the ``rl_learner_finalized`` record, complete it from the island's
    own tape file before calling it incomplete. Returns the names whose tape is
    still incomplete (``.incomplete`` marker written) -- the launcher's exit 3.

    S14/A19 (FINAL-REPORT-S7 §6.1, -5r1 r7): a SUCCEEDED job ended with exit 3
    because ``sky.tail_logs`` stopped delivering lines part-way (58 of 146
    records, stream silent after a learner restart) and that run predated the
    island-file recovery. The island file is now consulted for every unfinalized
    tape: a streamed-but-cut one, one that never streamed at all (previously
    marked incomplete without looking), and a Modal island's tape pulled from
    its Volume (``modal_tape_dir/<island>/**/rl-island-*.jsonl``). A tape whose
    island file also lacks the finalized record stays incomplete."""
    recover_sky = recover_sky or _recover_echo_tape
    incomplete: list[str] = []
    for name in sorted(set(names) | set(collectors)):
        collector = collectors.get(name)
        streamed = collector is not None
        if collector is None:  # never streamed: nothing received over the log stream
            collector = EventCollector(Path(events_dir) / f"{name}.jsonl", fresh=False)
        complete = collector.close()
        if streamed:
            print(
                f"[launcher] {name}: {collector.count} event(s) -> {collector.path}; "
                f"{collector.discarded} malformed prefixed line(s) discarded"
            )
        else:
            print(f"[launcher] {name}: no log stream line received; tape judged from the "
                  f"island's own tape file", file=sys.stderr)
        if not complete:
            if name in modal_cfgs:
                files = (sorted(Path(modal_tape_dir, name).rglob("rl-island-*.jsonl"))
                         if modal_tape_dir is not None else [])
                if not files:
                    print(f"[launcher] {name}: no pulled Modal tape file to recover from",
                          file=sys.stderr)
                before = collector.count
                for f in files:
                    try:
                        complete = collector.recover_from_file(f) or complete
                    except Exception as e:  # noqa: BLE001 - unreadable file: stay incomplete
                        print(f"[launcher] {name}: tape recovery from {f} failed ({e})",
                              file=sys.stderr)
                if files:
                    print(f"[launcher] {name}: tape recovered from {[f.name for f in files]}: "
                          f"+{collector.count - before} record(s), finalized={complete}")
            else:
                # sky island: the tape file itself is reachable, complete from it
                complete = recover_sky(name, collector)
        if not complete:
            incomplete.append(name)
            print(
                f"[launcher] WARN: {name} event tape has no rl_learner_finalized "
                f"record; marked {collector.incomplete_marker}",
                file=sys.stderr,
            )
    return incomplete


_ISLAND_NAME_RE = re.compile(r"-l(\d+)-")


# LEAVE reason names (syncer leave_reason_name) whose island owes no
# rl_learner_finalized. INFRA_FAILED, TRAINING_FAILED, CONFIG_MISMATCH, unknown
# codes and the pre-reason-code "requested" stay failures (syncer and launcher
# ship from one build; older runs are not resumed).
EXCUSED_LEAVE_REASONS = frozenset({"finished", "stopped", "preempted", "lease_expired"})


def syncer_pool_departures(tape_path, offset: int = 0) -> dict[int, str]:
    """Elastic islands the syncer removed from the pool for an excused reason and
    never re-admitted.

    Reads the syncer event tape (from ``offset``: this run's part only) and returns
    ``{island_id: reason}`` for every island whose last pool record is ``pool_leave``
    with a reason in ``EXCUSED_LEAVE_REASONS``. User ruling S16 §6.6: such an island
    owes no ``rl_learner_finalized`` record; the run summary reports it as a mid-run
    departure instead of the launcher's exit 3. An island that left with a failure
    code is not returned, so its missing record still fails the run. Unreadable
    tape -> {} (fail closed: exit 3 stays)."""
    last: dict[int, tuple[str, str]] = {}
    try:
        with open(tape_path, encoding="utf-8") as handle:
            handle.seek(offset)
            for line in handle:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(rec, dict):
                    continue
                kind = rec.get("kind")
                if kind in ("pool_join", "pool_leave") and isinstance(rec.get("island_id"), int):
                    last[rec["island_id"]] = (kind, str(rec.get("reason") or "unknown"))
    except OSError:
        return {}
    return {i: reason for i, (kind, reason) in last.items()
            if kind == "pool_leave" and reason in EXCUSED_LEAVE_REASONS}


def excuse_departed_islands(incomplete: list[str], departures: dict[int, str]) -> tuple[list[str], list[dict]]:
    """Split the unfinalized island tapes into those still owing ``rl_learner_finalized``
    (exit 3) and those the syncer removed from the pool (reported, not an error)."""
    still, left = [], []
    for name in incomplete:
        m = _ISLAND_NAME_RE.search(name)
        island = int(m.group(1)) if m else None
        if island is not None and island in departures:
            left.append({"island": island, "name": name, "reason": departures[island]})
        else:
            still.append(name)
    return still, left
