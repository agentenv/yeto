"""SFT (ms-swift Megatron + decoupled DiLoCo) adapter for the dashboard.

ms-swift islands and the syncer do not write the RL learner tape. This module
turns their native files into ``sft_*`` records that the one dashboard reducer
folds (``Reducer._feed_sft``), so the live server and the export keep using a
single reducer and a single view.

Files understood (the island id comes from an ``island-<N>`` path component,
default ``0``; the syncer's files live in a ``syncer`` directory):

==========================  =========================================================
``logging.jsonl``           ms-swift per-step log: loss / grad_norm / learning_rate /
                            ``memory(GiB)`` / ``iteration "k/K"`` / ``train_speed(s/it)``
``yeto_steps.jsonl``        yeto plugin per step: time / target_tokens / sync_s /
                            export_s / apply_s / pushes_sent / applies
``bench_metrics.jsonl``     per step: tokens / wall_s / tokens_per_s / mfu_6N_pct /
                            world_size (``step`` counts from 1 in each launch)
``health.rank<R>.jsonl``    per rank and step: ``*_peak_gib`` / grad_norm
``args.json``               ms-swift args: TP / EP / PP / world size / train_iters /
                            cached_dataset (``_v2`` in a path -> data v2)
``island.json``             optional side file: ``name`` / ``nodes`` / ``dataset``
``syncer.log``              syncer text log: rejections, timeouts, not-carried pushes,
                            stale bases, late pushes, disconnects, checkpoints; round
                            timing from ``grace window armed`` (``quorum_secs``) and
                            ``outer step`` (``ms``) lines
``cmdline.txt``             the syncer's argv (``--total-steps`` and friends)
syncer ``tape.jsonl``       merge records (already understood by the reducer) and
                            ``kind`` records: outer_step / delta_* / pool_*
``eval/rows.jsonl``         held-out eval per syncer checkpoint: step / loss / ppl /
                            ``vs_<reference>`` (e.g. ``vs_base``) / ``vs_prev`` (re-read whole
                            on change; one row per step, the last line of a step wins)
==========================  =========================================================

Eval comparisons: ``dloss`` is this checkpoint's loss minus the reference's loss
(negative is better), so the reference loss is ``loss - dloss``. ``mean`` / ``se``
are the per-sample paired difference and its standard error (a different weighting
than ``dloss``), and ``better`` the share of samples where this checkpoint wins.

Side samples written by an external, read-only collector are already ``sft_*``
records: ``sft_syncer_rss`` (``rss_bytes``) and ``sft_host_mem``
(``node`` / ``avail_bytes`` / ``total_bytes``).

Syncer round timing (one merge row per outer step): ``ms`` is the round's
open-to-done wall time, ``quorum_ms`` the wait until the quorum arrived and
``sync_ms`` (``sync/merge_seconds``) the merge itself; what is left
(``ms - quorum_ms - sync_ms``) is the grace window spent waiting for stragglers.
The wait of a round is ``ms - sync_ms``. When the tape lacks ``ms`` / ``quorum_ms``
they come from the syncer.log lines of the same step.

Missing values stay ``None``; nothing is interpolated here (the page resamples
the stacked tokens/s on a shared time grid).
"""

from __future__ import annotations

import json
import math
import re
import shlex
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SFT_FILES = ("logging.jsonl", "yeto_steps.jsonl", "bench_metrics.jsonl", "args.json", "island.json",
             "syncer.log", "cmdline.txt")
SIBLINGS = ("logging.jsonl", "bench_metrics.jsonl", "yeto_steps.jsonl")
SYNCER_TAPE_KINDS = frozenset({"outer_step", "delta_accepted", "delta_rejected", "delta_carried_over",
                               "pool_join", "pool_leave"})


def is_syncer_tape(record: dict) -> bool:
    """A syncer event-tape record (outer steps, deltas, pool changes) the SFT page reads."""
    return record.get("event") is None and record.get("kind") in SYNCER_TAPE_KINDS and "seq" not in record


# Reference lines drawn on the page; set with --thresholds (0 = no line).
DEFAULT_REFS = {"sft_ref_step_s": 0.0, "sft_hbm_limit_gib": 0.0, "sft_syncer_rss_limit_bytes": 0.0,
                "sft_slow_factor": 1.1}
# Stale-island check when neither a step time nor a reference is known yet.
FALLBACK_STEP_S = 600.0
STEP_COLS = ("step", "ts", "loss", "grad_norm", "lr", "step_s", "tps", "tokens", "target_tokens",
             "sync_s", "export_s", "apply_s", "hbm_gib", "mfu", "pushes", "applies")
MERGE_COLS = ("ts", "step", "fragment", "merge_s", "gnorm", "round_s", "quorum_s", "grace_s", "wait_s")
EVAL_REF_PREFIX = "vs_"
EVAL_FIELDS = ("dloss", "mean", "se", "better")
SAMPLES_MAX = 20000
EVENTS_MAX = 20000

_ISLAND_DIR = re.compile(r"^island-?(\d+)$")
_HEALTH = re.compile(r"^health\.rank(\d+)\.jsonl$")
_LOG_LINE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?)Z?\s+(TRACE|DEBUG|INFO|WARN|ERROR)\s+(\S+?):?\s+(.*)$")
_LEARNER = re.compile(r"learner_id=(\d+)")
_SECONDS = re.compile(r"seconds=([0-9.]+)")
_STEP_KV = re.compile(r"\bstep=(\d+)")
_ARMED = re.compile(r"^grace window armed step=(\d+) fragment=(\d+).*?\bquorum_secs=([0-9.]+)")
_OUTER = re.compile(r"^outer step step=(\d+) fragment=(\d+).*?\bms=(\d+)")
# syncer.log message -> category, first match wins
LOG_CATEGORIES = (
    ("rejected", re.compile(r"reject", re.I)),
    ("timeout", re.compile(r"time(?:d)?[ _-]?out", re.I)),
    ("not_carried", re.compile(r"not carried|left out of the final cut", re.I)),
    ("stale", re.compile(r"stale base", re.I)),
    ("late", re.compile(r"late push", re.I)),
    ("disconnect", re.compile(r"disconnect|connection ended|broken pipe|reset by peer", re.I)),
    ("checkpoint", re.compile(r"checkpoint written", re.I)),
    ("joined", re.compile(r"learner connected", re.I)),
)
# categories the page counts as problems (KPI "rejected / timed out / not carried")
PROBLEM_CATEGORIES = ("rejected", "timeout", "not_carried")
# Records read from syncer.log: an RL run's syncer writes the same log (e.g. next to its
# event tape), so these alone do not make a run an SFT run.
SHARED_SYNCER_EVENTS = frozenset({"sft_syncer_log", "sft_syncer_round"})


def finite(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if math.isfinite(v) else None


# -- path roles ---------------------------------------------------------------

def island_of(path: Path) -> str:
    for part in reversed(path.parent.parts):
        m = _ISLAND_DIR.match(part)
        if m:
            return m.group(1)
    return "0"


def is_syncer_path(path: Path) -> bool:
    return any(p == "syncer" or p.startswith("syncer-") for p in path.parent.parts[-2:])


def role(path: Path) -> str | None:
    """Which SFT file this is, or None for a file the generic tape reader handles."""
    name = path.name
    if name == "syncer.log" or (name == "cmdline.txt" and is_syncer_path(path)):
        return name
    if name in ("logging.jsonl", "yeto_steps.jsonl", "bench_metrics.jsonl"):
        return name
    if _HEALTH.match(name):
        return "health"
    if name == "island.json":
        return name
    if name == "rows.jsonl" and path.parent.name == "eval":
        return "eval"
    if name == "args.json" and any((path.parent / f).exists() for f in SIBLINGS):
        return name
    return None


# -- record converters (pure) ---------------------------------------------------

def _hms(text: Any) -> float | None:
    """ms-swift ``elapsed_time`` "1d 2h 17m 53s" -> seconds."""
    if not isinstance(text, str):
        return None
    total, seen = 0.0, False
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([dhms])", text):
        total += float(num) * {"d": 86400, "h": 3600, "m": 60, "s": 1}[unit]
        seen = True
    return total if seen else None


def _secs(ms: Any) -> float | None:
    v = finite(ms)
    return None if v is None else v / 1000.0


def merge_seconds(m: dict) -> float | None:
    """Time the syncer spent merging one round (``sync/merge_seconds``, else
    ``sync_ms``). The record's ``ms`` is the round's open-to-done wall time,
    which includes waiting for the quorum, and is not a merge time."""
    v = finite(m.get("sync/merge_seconds"))
    return v if v is not None else _secs(m.get("sync_ms"))


def from_swift_log(rec: dict, island: str) -> dict | None:
    it = rec.get("iteration")
    step = total = None
    if isinstance(it, str) and "/" in it:
        a, _, b = it.partition("/")
        step, total = (int(a) if a.strip().isdigit() else None), (int(b) if b.strip().isdigit() else None)
    elif isinstance(it, int):
        step = it
    if step is None:
        return None
    return {"event": "sft_train_log", "island_id": island, "step": step, "steps_total": total,
            "loss": rec.get("loss"), "grad_norm": rec.get("grad_norm"), "lr": rec.get("learning_rate"),
            "memory_gib": rec.get("memory(GiB)"), "s_per_it": rec.get("train_speed(s/it)"),
            "elapsed_s": _hms(rec.get("elapsed_time"))}


def from_yeto_step(rec: dict, island: str) -> dict | None:
    step = rec.get("iteration", rec.get("step"))
    if not isinstance(step, int):
        return None
    out = {"event": "sft_sync_step", "island_id": island, "step": step, "time_unix": rec.get("time")}
    for k in ("target_tokens", "sync_s", "export_s", "apply_s", "pushes_sent", "applies"):
        out[k] = rec.get(k)
    return out


def from_bench(rec: dict, island: str) -> dict | None:
    if not isinstance(rec.get("step"), int):
        return None
    out = {"event": "sft_bench", "island_id": island, "seg_step": rec["step"]}
    for k in ("tokens", "padded_tokens", "wall_s", "tokens_per_s", "mfu_6N_pct", "world_size", "optimizer_max_s"):
        out[k] = rec.get(k)
    return out


def from_health(rec: dict, island: str, rank: str) -> dict | None:
    if not isinstance(rec.get("step"), int):
        return None
    return {"event": "sft_health", "island_id": island, "seg_step": rec["step"],
            "rank": rec.get("rank", int(rank) if rank.isdigit() else rank),
            "reserved_peak_gib": rec.get("reserved_peak_gib"), "fwd_bwd_peak_gib": rec.get("fwd_bwd_peak_gib"),
            "optimizer_peak_gib": rec.get("optimizer_peak_gib"), "grad_norm": rec.get("grad_norm"),
            "update_successful": rec.get("update_successful")}


def dataset_tag(paths: Any) -> str | None:
    items = paths if isinstance(paths, list) else [paths] if isinstance(paths, str) else []
    if not items:
        return None
    m = re.search(r"_v(\d+)(?:/|$|_)", " ".join(str(p) for p in items) + "/")
    return f"v{m.group(1)}" if m else "v1"


def from_args(rec: dict, island: str) -> dict:
    g = rec.get
    tp, ep, pp = g("tensor_model_parallel_size"), g("expert_model_parallel_size"), g("pipeline_model_parallel_size")
    layout = "/".join(f"{k}{v}" for k, v in (("TP", tp), ("EP", ep), ("PP", pp)) if isinstance(v, int))
    return {"event": "sft_island_meta", "island_id": island, "layout": layout or None,
            "cards": g("world_size"), "steps_total": g("train_iters"), "gbs": g("global_batch_size"),
            "max_length": g("max_length"), "dataset": dataset_tag(g("cached_dataset") or g("dataset"))}


def from_island_json(rec: dict, island: str) -> dict:
    """island.json is written by hand or by a launcher: keep only well-typed fields
    (strings, a list of node names, positive integers) so the page can trust them."""
    out = {"event": "sft_island_meta", "island_id": str(rec.get("island", island))}
    for k in ("name", "dataset", "layout"):
        if isinstance(rec.get(k), str):
            out[k] = rec[k]
    nodes = rec.get("nodes")
    if isinstance(nodes, str):
        nodes = [nodes]
    if isinstance(nodes, list) and all(isinstance(n, str) for n in nodes):
        out["nodes"] = nodes
    for k in ("cards", "steps_total"):
        v = rec.get(k)
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            out[k] = v
    return out


def _iso(text: str) -> float | None:
    try:
        return datetime.strptime(text[:26], "%Y-%m-%dT%H:%M:%S.%f" if "." in text else "%Y-%m-%dT%H:%M:%S") \
            .replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def from_syncer_log_line(line: str) -> dict | None:
    """One syncer.log line -> ``sft_syncer_log`` when it falls in a category, else None."""
    m = _LOG_LINE.match(line.strip())
    if not m:
        return None
    ts, level, _target, msg = m.groups()
    rm = _ARMED.match(msg) or _OUTER.match(msg)
    if rm:
        armed = rm.re is _ARMED
        return {"event": "sft_syncer_round", "time_unix": _iso(ts), "phase": "armed" if armed else "done",
                "step": int(rm.group(1)), "fragment": int(rm.group(2)),
                ("quorum_s" if armed else "round_s"): float(rm.group(3)) if armed else int(rm.group(3)) / 1000.0}
    cat = next((c for c, rx in LOG_CATEGORIES if rx.search(msg)), None)
    if cat is None and level not in ("WARN", "ERROR"):
        return None
    lm, sm, st = _LEARNER.search(msg), _SECONDS.search(msg), _STEP_KV.search(msg)
    return {"event": "sft_syncer_log", "time_unix": _iso(ts), "level": level, "category": cat or level.lower(),
            "learner": int(lm.group(1)) if lm else None, "seconds": float(sm.group(1)) if sm else None,
            "step": int(st.group(1)) if st else None, "message": msg[:240]}


def from_cmdline(text: str) -> dict:
    try:
        argv = shlex.split(text)
    except ValueError:
        argv = text.split()
    out: dict[str, Any] = {"event": "sft_syncer_meta"}
    keys = {"--total-steps": "total_steps", "--sync-interval-steps": "sync_interval", "--learners": "learners",
            "--max-learners": "max_learners", "--quorum": "quorum", "--pipeline": "pipeline"}
    for i, a in enumerate(argv):
        k, _, v = a.partition("=")
        if k in keys:
            val = v or (argv[i + 1] if i + 1 < len(argv) else "")
            out[keys[k]] = int(val) if val.isdigit() else val
    return out


def eval_row(rec: dict) -> dict | None:
    """One ``eval/rows.jsonl`` row -> the fields the page draws, or None without a step."""
    step = rec.get("step")
    if isinstance(step, bool) or not isinstance(step, int):
        return None
    loss = finite(rec.get("loss"))
    out: dict[str, Any] = {"step": step, "loss": loss, "ppl": finite(rec.get("ppl")),
                           "ckpt_written": rec.get("ckpt_written") if isinstance(rec.get("ckpt_written"), str) else None,
                           "prev_step": rec.get("prev_step") if isinstance(rec.get("prev_step"), int) else None,
                           "merges_total": finite(rec.get("merges_total"))}
    # every ``vs_<name>`` comparison is a reference (``vs_prev``: the previous row)
    for ref in sorted(k for k in rec if isinstance(k, str) and k.startswith(EVAL_REF_PREFIX)):
        cmp_ = rec.get(ref)
        if not isinstance(cmp_, dict):
            continue
        c = {f: finite(cmp_.get(f)) for f in EVAL_FIELDS}
        c["ref_loss"] = loss - c["dloss"] if loss is not None and c["dloss"] is not None else None
        out[ref] = c
    return out


def from_eval_rows(text: str) -> tuple[dict, int]:
    """Whole ``eval/rows.jsonl`` -> (``sft_eval`` record with rows sorted by step, bad lines)."""
    by_step: dict[int, dict] = {}
    bad = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        row = eval_row(rec) if isinstance(rec, dict) else None
        if row is None:
            bad += 1
            continue
        by_step[row["step"]] = row
    return {"event": "sft_eval", "rows": [by_step[s] for s in sorted(by_step)]}, bad


# -- source ---------------------------------------------------------------------

class SftSource:
    """Tails one SFT file and feeds converted records (same contract as TapeSource)."""

    def __init__(self, path: str | Path, kind: str | None = None):
        self.path = Path(path)
        self.name = str(self.path)
        self.kind = kind or role(self.path)
        self.island = island_of(self.path)
        self.offset = 0
        self.bad_lines = 0
        self._sig: tuple | None = None

    def _whole(self) -> str | None:
        try:
            st = self.path.stat()
        except OSError:
            return None
        sig = (st.st_size, st.st_mtime_ns)
        if sig == self._sig:
            return None
        self._sig = sig
        try:
            return self.path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    def _lines(self) -> Iterator[tuple[int, bytes]]:
        try:
            size = self.path.stat().st_size
        except OSError:
            return
        if size < self.offset:
            self.offset = 0
        if size == self.offset:
            return
        with open(self.path, "rb") as fh:
            fh.seek(self.offset)
            data = fh.read(size - self.offset)
        pos = self.offset
        for line in data.splitlines(keepends=True):
            if not line.endswith(b"\n"):
                break
            pos += len(line)
            yield pos, line
        self.offset = pos

    def records(self) -> Iterator[tuple[int | None, dict]]:
        k = self.kind
        if k in ("args.json", "island.json", "cmdline.txt", "eval"):
            text = self._whole()
            if text is None:
                return
            if k == "eval":
                rec, bad = from_eval_rows(text)
                self.bad_lines += bad
                yield None, rec
                return
            if k == "cmdline.txt":
                yield None, from_cmdline(text)
                return
            try:
                rec = json.loads(text)
            except json.JSONDecodeError:
                self.bad_lines += 1
                return
            if isinstance(rec, dict):
                yield None, (from_args if k == "args.json" else from_island_json)(rec, self.island)
            return
        rank = _HEALTH.match(self.path.name).group(1) if k == "health" else ""
        for pos, line in self._lines():
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            if k == "syncer.log":
                out = from_syncer_log_line(text)
            else:
                try:
                    rec = json.loads(text)
                except json.JSONDecodeError:
                    self.bad_lines += 1
                    continue
                if not isinstance(rec, dict):
                    continue
                out = {"logging.jsonl": lambda: from_swift_log(rec, self.island),
                       "yeto_steps.jsonl": lambda: from_yeto_step(rec, self.island),
                       "bench_metrics.jsonl": lambda: from_bench(rec, self.island),
                       "health": lambda: from_health(rec, self.island, rank)}[k]()
            if out is not None:
                yield pos, out

    def pump(self, reducer) -> int:
        n = 0
        for off, rec in self.records():
            if reducer.feed(rec, source=self.name, offset=off):
                n += 1
        return n


# -- folding --------------------------------------------------------------------

def _new_island(iid: str) -> dict:
    # health: (launch, seg_step) -> {rank: peak GiB}; health_last: rank -> (launch, last seg_step)
    return {"id": iid, "meta": {}, "log": {}, "sync": {}, "bench": [], "health": {}, "health_last": {},
            "host": {}, "steps_total": None, "log_starts": [], "log_last": None}


class SftState:
    def __init__(self) -> None:
        self.islands: dict[str, dict] = {}
        self.syncer_meta: dict = {}
        self.rss: deque = deque(maxlen=SAMPLES_MAX)
        self.outer: list[list] = []
        self.events: deque = deque(maxlen=EVENTS_MAX)  # [ts, category, learner, origin]
        self.eval: list[dict] = []  # held-out eval rows by step (eval/rows.jsonl, replaced whole)
        self.log_rounds: dict[int, dict] = {}  # syncer.log round timing by outer step
        self.log_seen = False
        self.records = 0
        self.sft_records = 0  # records only an SFT run writes (not the shared syncer tape / log)

    def island(self, iid: Any) -> dict:
        iid = str(iid if iid is not None else "0")
        if iid not in self.islands:
            self.islands[iid] = _new_island(iid)
        return self.islands[iid]

    def fold(self, r: dict, ts: float | None) -> None:
        self.records += 1
        ev = r.get("event")
        if ev is None and r.get("kind") in SYNCER_TAPE_KINDS:
            return self._tape(r, ts)
        if isinstance(ev, str) and ev.startswith("sft_") and ev not in SHARED_SYNCER_EVENTS:
            self.sft_records += 1
        if ev == "sft_syncer_log":
            self.log_seen = True
            self.events.append([ts, r.get("category"), r.get("learner"), "log", r.get("seconds")])
            return
        if ev == "sft_syncer_round":
            rd = self.log_rounds.setdefault(r["step"], {"fragment": r.get("fragment")})
            rd["armed" if r["phase"] == "armed" else "done"] = ts
            for k in ("quorum_s", "round_s"):
                if r.get(k) is not None:
                    rd[k] = r[k]
            return
        if ev == "sft_syncer_meta":
            self.syncer_meta.update({k: v for k, v in r.items() if k != "event"})
            return
        if ev == "sft_eval":
            rows = r.get("rows")
            self.eval = list(rows) if isinstance(rows, list) else []
            return
        if ev == "sft_syncer_rss":
            v = finite(r.get("rss_bytes"))
            if ts is not None and v is not None:
                self.rss.append([ts, v])
            return
        isl = self.island(r.get("island_id"))
        if ev == "sft_island_meta":
            isl["meta"].update({k: v for k, v in r.items() if k not in ("event", "island_id") and v is not None})
        elif ev == "sft_train_log":
            # ms-swift appends to logging.jsonl: a step not above the last one
            # starts a new launch (a resume from an earlier checkpoint).
            if isl["log_last"] is None or r["step"] <= isl["log_last"]:
                isl["log_starts"].append(r["step"])
            isl["log_last"] = r["step"]
            isl["log"][r["step"]] = r
            if r.get("steps_total"):
                isl["steps_total"] = r["steps_total"]
        elif ev == "sft_sync_step":
            isl["sync"][r["step"]] = r
        elif ev == "sft_bench":
            isl["bench"].append(r)
        elif ev == "sft_health":
            # Each rank's health file counts from 1 in every launch, like the
            # bench: a step not above that rank's previous one starts a launch.
            rank, step = str(r.get("rank")), r.get("seg_step")
            launch, last = isl["health_last"].get(rank, (0, None))
            if last is not None and isinstance(step, int) and step <= last:
                launch += 1
            isl["health_last"][rank] = (launch, step)
            v = finite(r.get("reserved_peak_gib"))
            if v is not None:
                isl["health"].setdefault((launch, step), {})[rank] = v
        elif ev == "sft_host_mem":
            node = str(r.get("node") or "?")
            avail, total = finite(r.get("avail_bytes")), finite(r.get("total_bytes"))
            if ts is not None and avail is not None:
                isl["host"].setdefault(node, deque(maxlen=SAMPLES_MAX)).append([ts, avail, total])

    def _tape(self, r: dict, ts: float | None) -> None:
        kind = r["kind"]
        if kind == "outer_step":
            self.outer.append([ts, r.get("step"), r.get("fragment"), finite(r.get("step_seconds")),
                               bool(r.get("timed_out")), r.get("arrived")])
            if r.get("timed_out"):
                self.events.append([ts, "timeout", None, "tape", None])
        elif kind == "delta_rejected":
            self.events.append([ts, "rejected", r.get("island_id", r.get("learner_id")), "tape", None])
        elif kind == "delta_carried_over":
            self.events.append([ts, "carried", r.get("island_id", r.get("learner_id")), "tape", None])
        elif kind in ("pool_join", "pool_leave"):
            self.events.append([ts, kind, r.get("island_id"), "tape", None])

    # -- view -------------------------------------------------------------------
    @staticmethod
    def _bench_by_step(isl: dict) -> tuple[dict[int, dict], list[int]]:
        """bench / health ``step`` restarts at 1 in each launch: map each launch
        segment onto absolute steps. Launch k starts at the first step of the
        k-th ms-swift log segment (a resume from checkpoint N logs N+1 again
        after a crash at a later step); without one it continues after the
        last mapped step. The first launch falls back to the first step logged
        by ms-swift or the yeto plugin. Returns the mapped bench records and
        each launch's step offset."""
        known = sorted(set(isl["log"]) | set(isl["sync"]))
        starts = isl.get("log_starts") or []
        base = (starts[0] - 1) if starts else ((known[0] - 1) if known else 0)
        out, prev, offsets = {}, None, [base]
        for b in isl["bench"]:
            s = b["seg_step"]
            if prev is not None and s <= prev:  # a new launch
                seg = len(offsets)
                offsets.append(starts[seg] - 1 if seg < len(starts) else max(out))
            out[s + offsets[-1]] = b
            prev = s
        return out, offsets

    @staticmethod
    def _health_by_step(isl: dict, offsets: list[int]) -> dict[int, dict]:
        """Health per absolute step, with the launch offsets of the bench (a
        launch the bench has not seen starts at its ms-swift log segment)."""
        starts = isl.get("log_starts") or []
        out: dict[int, dict] = {}
        for (launch, step), ranks in isl["health"].items():
            if not isinstance(step, int):
                continue
            if launch < len(offsets):
                off = offsets[launch]
            elif launch < len(starts):
                off = starts[launch] - 1
            else:
                continue  # no anchor for this launch: better no value than a wrong step
            out.setdefault(step + off, {}).update(ranks)
        return out

    def island_rows(self, isl: dict) -> list[list]:
        bench, offsets = self._bench_by_step(isl)
        health = self._health_by_step(isl, offsets)
        steps = sorted(set(isl["log"]) | set(isl["sync"]) | set(bench))
        rows, prev_ts = [], None
        for s in steps:
            lg, sy, bn = isl["log"].get(s, {}), isl["sync"].get(s, {}), bench.get(s, {})
            ts = finite(sy.get("time_unix"))
            hp = health.get(s)
            hbm = max(hp.values()) if hp else finite(lg.get("memory_gib"))
            step_s = finite(bn.get("wall_s"))
            if step_s is None and ts is not None and prev_ts is not None and s - 1 in isl["sync"]:
                step_s = ts - prev_ts
            tps = finite(bn.get("tokens_per_s"))
            if tps is None and finite(bn.get("tokens")) and step_s:
                tps = bn["tokens"] / step_s
            rows.append([s, ts, finite(lg.get("loss")), finite(lg.get("grad_norm")), finite(lg.get("lr")), step_s,
                         tps, finite(bn.get("tokens")), finite(sy.get("target_tokens")), finite(sy.get("sync_s")),
                         finite(sy.get("export_s")), finite(sy.get("apply_s")), hbm, finite(bn.get("mfu_6N_pct")),
                         finite(sy.get("pushes_sent")), finite(sy.get("applies"))])
            prev_ts = ts if ts is not None else prev_ts
        return rows

    def merge_rows(self, merges: list[dict]) -> list[list]:
        """One row per merge: [ts, step, fragment, merge_s, gnorm, round_s, quorum_s, grace_s, wait_s].

        ``round_s`` / ``quorum_s`` come from the tape (``ms`` / ``quorum_ms``), else from
        the syncer.log lines of the step (``outer step ... ms=`` and ``grace window armed
        ... quorum_secs=``); a round only in the log still gets a row (merge_s unknown).
        ``wait_s = round_s - merge_s``; ``grace_s = wait_s - quorum_s`` (clamped at 0)."""
        rows, seen = [], set()
        for m in merges:
            step = m.get("step")
            lg = self.log_rounds.get(step, {}) if isinstance(step, int) else {}
            rnd = _secs(m.get("ms"))
            q = _secs(m.get("quorum_ms"))
            rows.append(self._round_row(finite(m.get("time_unix")), step, m.get("fragment"), merge_seconds(m),
                                        finite(m.get("gnorm", m.get("sync/global_delta_norm"))),
                                        rnd if rnd is not None else lg.get("round_s"),
                                        q if q is not None else lg.get("quorum_s")))
            seen.add(step)
        for step, lg in self.log_rounds.items():
            if step not in seen and lg.get("done") is not None:
                rows.append(self._round_row(lg["done"], step, lg.get("fragment"), None, None, lg.get("round_s"),
                                            lg.get("quorum_s")))
        rows.sort(key=lambda x: (x[0] is None, x[0] or 0))
        return rows

    @staticmethod
    def _round_row(ts, step, frag, merge, gnorm, rnd, quorum) -> list:
        wait = max(0.0, rnd - merge) if rnd is not None and merge is not None else None
        grace = max(0.0, wait - quorum) if wait is not None and quorum is not None else None
        return [ts, step, frag, merge, gnorm, rnd, quorum, grace, wait]

    def view(self, *, now: float, merges: list[dict], refs: dict) -> dict:
        islands = []
        for iid in sorted(self.islands, key=lambda i: (0, int(i)) if i.isdigit() else (1, i)):
            isl = self.islands[iid]
            rows = self.island_rows(isl)
            meta = dict(isl["meta"])
            total = isl["steps_total"] or meta.get("steps_total")
            last = rows[-1] if rows else None
            last_ts = max((r[1] for r in rows if r[1] is not None), default=None)
            steps_s = sorted(r[5] for r in rows[-20:] if r[5] is not None)
            p50 = steps_s[max(0, math.ceil(0.5 * len(steps_s)) - 1)] if steps_s else None  # nearest rank, as the page
            age = (now - last_ts) if last_ts is not None else None
            if last and total and last[0] >= total:
                status = "done"
            elif age is None:
                status = "unknown"
            else:
                expect = p50 or refs.get("sft_ref_step_s") or FALLBACK_STEP_S
                status = "ok" if age < 2 * expect + 300 else ("warn" if age < 4 * expect + 600 else "stale")
            # the plugin pushes from a background thread, so ``pushes_sent`` (pushes
            # finished inside the step) is almost always 0; a push step is one that
            # spent time exporting the fragment
            push = next((r for r in reversed(rows) if r[14] or (r[10] or 0) > 0), None)
            apply = next((r for r in reversed(rows) if r[15]), None)
            host = {n: [[x[0], x[1], x[2]] for x in list(ser)[-2000:]] for n, ser in isl["host"].items()}
            islands.append({
                "id": iid, "name": meta.get("name") or f"island {iid}", "nodes": meta.get("nodes"),
                "layout": meta.get("layout"), "cards": meta.get("cards"), "dataset": meta.get("dataset"),
                "gbs": meta.get("gbs"), "steps_total": total, "status": status, "last_ts": last_ts,
                "age_s": None if age is None else round(age, 1),
                "last_push": None if push is None else {"step": push[0], "ts": push[1], "export_s": push[10],
                                                         "sync_s": push[9]},
                "last_apply": None if apply is None else {"step": apply[0], "ts": apply[1], "apply_s": apply[11]},
                "rows": rows, "host": host})
        rej_origin = "log" if self.log_seen else "tape"
        events = [e for e in self.events
                  if e[1] not in ("rejected", "timeout") or e[3] == rej_origin]
        merge_rows = self.merge_rows(merges)
        outer = sorted(self.outer, key=lambda x: (x[0] or 0))
        cur = max([x[1] for x in outer if isinstance(x[1], int)] +
                  [x[1] for x in merge_rows if isinstance(x[1], int)], default=None)
        total = self.syncer_meta.get("total_steps")
        eta = None
        timed = [x for x in outer if x[0] is not None and isinstance(x[1], int)]
        if isinstance(total, int) and len(timed) >= 2 and cur is not None:
            a, b = timed[max(0, len(timed) - 32)], timed[-1]
            if b[1] > a[1] and b[0] > a[0]:
                eta = b[0] + (total - cur) * (b[0] - a[0]) / (b[1] - a[1])
        return {
            "now": now, "refs": dict(refs), "cols": list(STEP_COLS), "islands": islands,
            "syncer": {"meta": dict(self.syncer_meta), "total_steps": total, "outer_step": cur, "eta_ts": eta,
                       "outer": outer[-5000:], "merges": merge_rows[-5000:], "merge_cols": list(MERGE_COLS), "rss": list(self.rss)[-5000:],
                       "events": events[-5000:], "log_seen": self.log_seen,
                       "problem_categories": list(PROBLEM_CATEGORIES)},
            "eval": {"rows": list(self.eval)},
        }
