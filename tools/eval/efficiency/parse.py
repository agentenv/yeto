"""Framework logs and yeto tapes -> one per-round record format.

Every parser returns a :class:`Parsed`: ``rounds`` (one dict per optimizer
step, the fields of :data:`ROUND_FIELDS`), ``sources`` (per field: the log key
it came from, ``"derived: ..."`` when it is computed from logged values by an
exact identity, ``MISSING`` when no source in the raw data carries it, or
``NA`` when the field does not apply to the arm), and ``segments_raw`` (every
native timing per round, for the overhead table).

A field that a log does not carry stays ``None``; nothing is estimated.

Formats read (each one is cited where it is parsed):

* verl ``FileLogger`` JSONL (``VERL_FILE_LOGGER_PATH``): one
  ``{"step": N, "data": {...}}`` per step (fork acad9875 ``verl/utils/tracking.py``
  ``FileLogger.log``); keys from ``metric_utils.compute_timing_metrics`` /
  ``compute_throughout_metrics`` (``timing_s/*``, ``perf/*``) and the fully_async
  trainer (``fully_async/*``, ``timing_s/param_sync``, ``timing_s/wait_for_enough_samples``);
* verl console lines ``step:N - key:value - ...`` (``LocalLogger``,
  ``concat_dict_to_str``) behind the harness's ``<unix time> `` line prefix, for the
  step end time;
* Miles ``perf N: {python dict}`` log lines: rollout side
  (``miles/ray/rollout/metrics.py:93``, keys ``rollout/*``, ``perf/rollout_time``,
  ``perf/tokens_per_gpu_per_sec``) and train side
  (``miles/utils/train_metric_utils.py:56``, keys ``perf/<timer>_time``,
  ``perf/actor_train_tok_per_s``); the line's ``[YYYY-mm-dd HH:MM:SS.mmm`` stamp
  (Miles' logger format) gives the time;
* yeto event tapes (JSONL ``{"event": ...}``): ``rl_round_wall``,
  ``rl_local_round``, ``rl_round_trained``, ``rl_timeline_span``,
  ``rl_driver_phase``, ``rl_rollout_carry_over``, ``rl_resource_sample``;
* NVML CSV (``nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,
  memory.used,memory.total,power.draw --format=csv``), with or without a
  ``<container>,`` prefix per line (``miles_local.sh`` polls every 20 s).
"""

from __future__ import annotations

import ast
import csv
import datetime as _dt
import io
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

MISSING = "missing"
NA = "n/a"
ROUND_FIELDS = (
    "round", "t_end_unix", "round_s", "gen_s", "train_s", "weight_sync_s", "sampler_update_s", "wait_s",
    "gen_tokens", "train_tokens", "samples", "reward_mean", "is_clip_frac", "dropped_samples",
    "generated_samples", "version_lag",
)
ANSI = re.compile(r"\x1b\[[0-9;]*m")


@dataclass
class Parsed:
    rounds: list[dict[str, Any]] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)
    segments_raw: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"rounds": self.rounds, "sources": self.sources, "segments_raw": self.segments_raw}


def _first(data: dict, keys: Iterable[str]) -> tuple[Any, str | None]:
    for key in keys:
        if data.get(key) is not None:
            return data[key], key
    return None, None


def _finish(parsed: Parsed, na: Iterable[str] = ()) -> Parsed:
    """Fill ``sources`` for fields no round carries (missing or not applicable)."""
    na = set(na)
    parsed.sources["round"] = "step index (verl global step; Miles / yeto rollout id)"
    for name in ROUND_FIELDS[1:]:
        if name in na:
            parsed.sources[name] = NA
            for r in parsed.rounds:
                r[name] = None
        elif name not in parsed.sources or all(r.get(name) is None for r in parsed.rounds):
            parsed.sources[name] = MISSING
    return parsed


def _record(rnd: int) -> dict[str, Any]:
    return {name: None for name in ROUND_FIELDS} | {"round": rnd}


def _set(parsed: Parsed, rec: dict, name: str, value: Any, source: str | None) -> None:
    if value is None or source is None:
        return
    rec[name] = value
    parsed.sources.setdefault(name, source)


def _step_ends(log_text: str, pattern: re.Pattern) -> dict[int, float]:
    """Round -> unix time of its log line (lines carry a ``<unix> `` prefix)."""
    out: dict[int, float] = {}
    for line in log_text.splitlines():
        m = pattern.search(ANSI.sub("", line))
        if m:
            out[int(m.group(2))] = float(m.group(1))
    return out


def _round_s_from_ends(parsed: Parsed) -> None:
    """round_s = time between consecutive step ends (one definition for every arm)."""
    prev = None
    for r in parsed.rounds:
        t = r.get("t_end_unix")
        if t is not None and prev is not None:
            r["round_s"] = round(t - prev, 3)
            parsed.sources.setdefault("round_s", "derived: step end - previous step end")
        prev = t


# ====================================================================== verl
VERL_STEP_LINE = re.compile(r"^(\d+(?:\.\d+)?) .*?\bstep:(\d+) - ")


def parse_verl(metrics_jsonl: str, log_text: str = "", *, samples_per_round: int, trainer_cards: int,
               mode: str) -> Parsed:
    """verl FileLogger rows (+ console step lines for end times)."""
    p = Parsed()
    ends = _step_ends(log_text, VERL_STEP_LINE)
    recs: dict[int, dict[str, Any]] = {}
    segs: dict[int, dict[str, float]] = {}
    for line in metrics_jsonl.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        d, step = row.get("data") or {}, int(row["step"])
        rnd = step
        if mode == "async" and "training/global_step" in d:
            # bare fully_async logs a step's training metrics one parameter version
            # later (row step N+1, training/global_step N) and never logs the last
            # step's; the rows without it carry the weight sync that ends version N.
            rnd = int(d["training/global_step"])
        elif mode == "async" and step == 0:
            continue  # the weight sync before the first training step
        rec = recs.setdefault(rnd, _record(rnd))
        if rnd == step:
            _set(p, rec, "t_end_unix", ends.get(step), "console step line (harness time prefix)")
        # fully_async: the trainer's timing_s/gen only pulls finished samples from the
        # queue; the rollouter's active time in the version window is the generation
        gen, key = _first(d, ("fully_async/rollouter/active_time",) if mode == "async" else ("timing_s/gen",))
        _set(p, rec, "gen_s", gen, key)
        parts = [k for k in ("timing_s/old_log_prob", "timing_s/update_actor") if d.get(k) is not None]
        if parts:
            _set(p, rec, "train_s", sum(float(d[k]) for k in parts), " + ".join(parts))
        sync, key = _first(d, ("timing_s/update_weights", "timing_s/param_sync"))
        _set(p, rec, "weight_sync_s", sync, key)
        wait, key = _first(d, ("timing_s/wait_for_enough_samples",))
        _set(p, rec, "wait_s", wait, key)
        if d.get("response_length/mean") is not None:  # every step trains exactly samples_per_round responses
            _set(p, rec, "gen_tokens", float(d["response_length/mean"]) * samples_per_round,
                 f"derived: response_length/mean x {samples_per_round} responses")
            _set(p, rec, "samples", samples_per_round, "config: samples per step (verl trains a fixed batch)")
        tokens, key = _first(d, ("perf/total_num_tokens",))
        if tokens is None and d.get("global_seqlen/mean") is not None:
            tokens, key = float(d["global_seqlen/mean"]) * trainer_cards, \
                f"derived: global_seqlen/mean x {trainer_cards} trainer ranks"
        _set(p, rec, "train_tokens", tokens, key)
        reward, key = _first(d, ("critic/score/mean", "critic/rewards/mean"))
        _set(p, rec, "reward_mean", reward, key)
        clip, key = _first(d, ("rollout_corr/rollout_is_ratio_fraction_high", "yeto_mismatch/tis_clipfrac"))
        _set(p, rec, "is_clip_frac", clip, key)
        dropped, key = _first(d, ("fully_async/count/dropped_stale_samples",))
        _set(p, rec, "dropped_samples", dropped, key)
        generated, key = _first(d, ("fully_async/count/total_generated_samples",))
        _set(p, rec, "generated_samples", generated, key)
        segs.setdefault(rnd, {}).update({k: float(v) for k, v in d.items() if k.startswith("timing_s/")
                                         and isinstance(v, (int, float))})
    for rec in recs.values():  # a step whose metrics verl never logged still trained one fixed batch
        _set(p, rec, "samples", samples_per_round if rec["samples"] is None else rec["samples"],
             "config: samples per step (verl trains a fixed batch)")
    p.rounds = [recs[r] for r in sorted(recs)]
    p.segments_raw = [segs[r] for r in sorted(recs)]
    _round_s_from_ends(p)
    na = {"sampler_update_s"}
    if mode == "sync":
        na |= {"wait_s", "dropped_samples", "generated_samples", "version_lag"}
    return _finish(p, na)


# ====================================================================== Miles
MILES_DICT_LINE = re.compile(r"\b(perf|step|rollout) (\d+): (\{.*\})\s*$")
MILES_STAMP = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)")


def _miles_stamp(line: str) -> float | None:
    m = MILES_STAMP.search(line)
    if not m:
        return None
    return _dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=_dt.timezone.utc).timestamp()


def miles_dicts(log_text: str) -> dict[int, dict[str, Any]]:
    """Rollout id -> {"rollout": {...}, "train": {...}, "step": {...}, "train_t": unix}."""
    out: dict[int, dict[str, Any]] = defaultdict(dict)
    for raw in log_text.splitlines():
        line = ANSI.sub("", raw)
        m = MILES_DICT_LINE.search(line)
        if not m:
            continue
        try:
            data = ast.literal_eval(m.group(3))
        except (ValueError, SyntaxError):
            continue
        rid = int(m.group(2))
        if m.group(1) == "perf" and ("perf/rollout_time" in data or any(k.startswith("rollout/") for k in data)):
            out[rid].setdefault("rollout", {}).update(data)
        elif m.group(1) == "perf":
            out[rid].setdefault("train", {}).update(data)
            out[rid]["train_t"] = _miles_stamp(line)
        else:
            out[rid].setdefault("step", {}).update(data)
    return dict(out)


def parse_miles(log_text: str, *, rollout_cards: int, mode: str) -> Parsed:
    p = Parsed()
    for rid, parts in sorted(miles_dicts(log_text).items()):
        ro, tr, st = parts.get("rollout", {}), parts.get("train", {}), parts.get("step", {})
        merged = {**ro, **tr, **st}
        rec = _record(rid)
        _set(p, rec, "t_end_unix", parts.get("train_t"), "Miles train perf line time stamp")
        gen, key = _first(ro, ("perf/rollout_time",))
        _set(p, rec, "gen_s", gen, key)
        train, key = _first(tr, ("perf/train_time",))
        _set(p, rec, "train_s", train, key)
        sync, key = _first(tr, ("perf/update_weights_time",))
        _set(p, rec, "weight_sync_s", sync, key)
        if mode == "async":
            wait, key = _first(tr, ("perf/train_wait_time",))
            _set(p, rec, "wait_s", wait, key)
        if ro.get("perf/tokens_per_gpu_per_sec") is not None and gen is not None:
            _set(p, rec, "gen_tokens", float(ro["perf/tokens_per_gpu_per_sec"]) * float(gen) * rollout_cards,
                 f"derived: perf/tokens_per_gpu_per_sec x perf/rollout_time x {rollout_cards} rollout cards")
        if tr.get("perf/actor_train_tok_per_s") is not None and tr.get("perf/actor_train_time") is not None:
            _set(p, rec, "train_tokens", float(tr["perf/actor_train_tok_per_s"]) * float(tr["perf/actor_train_time"]),
                 "derived: perf/actor_train_tok_per_s x perf/actor_train_time")
        samples, key = _first(merged, ("rollout/num_training_samples", "num_training_samples"))
        _set(p, rec, "samples", samples, key)
        reward, key = _first(merged, ("rollout/raw_reward", "rollout/episode_raw_reward", "rollout/rewards"))
        _set(p, rec, "reward_mean", reward, key)
        clip, key = _first(merged, ("train/tis_clipfrac", "train/tis_clip_frac", "tis_clipfrac"))
        _set(p, rec, "is_clip_frac", clip, key)
        lag = {k: v for k, v in merged.items() if "staleness" in k or "weight_version" in k}
        _set(p, rec, "version_lag", lag or None, "Miles keys containing 'staleness' / 'weight_version'")
        p.rounds.append(rec)
        p.segments_raw.append({k: float(v) for k, v in merged.items()
                               if k.startswith("perf/") and k.endswith("_time") and isinstance(v, (int, float))})
    _round_s_from_ends(p)
    na = {"sampler_update_s"}
    if mode == "sync":
        na |= {"wait_s", "dropped_samples", "generated_samples", "version_lag"}
    return _finish(p, na)


# ====================================================================== yeto tape
def tape_rows(text: str) -> list[dict[str, Any]]:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        start = line.find("{")
        if start < 0:
            continue
        try:
            row = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("event"):
            rows.append(row)
    return rows


# yeto task / phase names -> canonical segment (others stay in segments_raw only)
YETO_SEGMENT = {"generate": "gen_s", "train": "train_s", "publish": "weight_sync_s"}


def parse_yeto(events: list[dict[str, Any]], *, mode: str, framework_parsed: Parsed | None = None,
               framework_round_offset: int = 0, skip: frozenset[str] = frozenset(),
               rollouter_gen: bool = False) -> Parsed:
    """yeto tape -> rounds; ``framework_parsed`` (the framework's own log in the
    same run, e.g. Miles perf lines) fills fields the tape does not carry. Its
    round ``rollout_id + framework_round_offset`` is the tape's ``rollout_id``:
    0 for Miles (perf ids are rollout ids), 1 for verl (FileLogger step =
    global step, 1-based; T-EVAL eval-e1-verl-c: steps 1..4, rollout ids 0..3).

    ``rollouter_gen`` (yeto+verl fully_async): ``gen_s`` is the rollouter's
    ``fully_async/rollouter/active_time`` from ``rl_publication_check.rollouter_timing``;
    the check of policy version v closes round v - 1, the same window as the bare
    arm's per-step value."""
    p = Parsed()
    by = defaultdict(list)
    for e in events:
        by[e["event"]].append(e)
    walls = {int(e["rollout_id"]): e for e in by["rl_round_wall"] if e.get("rollout_id") is not None}
    trained = {int(e["rollout_id"]): e for e in by["rl_round_trained"] if e.get("rollout_id") is not None}
    local = sorted(by["rl_local_round"], key=lambda e: float(e.get("time_unix") or 0))
    spans: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for s in by["rl_timeline_span"]:
        if s.get("rollout_id") is None or s.get("start") is None or s.get("end") is None:
            continue
        name = "wait" if s.get("kind") == "wait" else str(s.get("task"))
        spans[int(s["rollout_id"])][name] += float(s["end"]) - float(s["start"])
    phases = _driver_phase_durations(by["rl_driver_phase"])
    drops = defaultdict(float)
    for e in by["rl_rollout_carry_over"]:
        if e.get("rollout_id") is not None and e.get("over_age_discarded_groups") is not None:
            drops[int(e["rollout_id"])] += float(e["over_age_discarded_groups"])
    rollouter = _rollouter_active(by["rl_publication_check"]) if rollouter_gen else {}
    rounds = sorted(set(walls) | set(trained))
    fw = {r["round"]: r for r in (framework_parsed.rounds if framework_parsed else [])}
    for i, rid in enumerate(rounds):
        rec = _record(rid)
        w, t = walls.get(rid, {}), trained.get(rid, {})
        lr = local[i] if i < len(local) else {}
        _set(p, rec, "t_end_unix", w.get("time_unix"), "rl_round_wall.time_unix")
        seg = dict(spans.get(rid) or phases.get(rid) or {})
        src = "rl_timeline_span" if rid in spans else "rl_driver_phase (marker to next marker)"
        for task, name in YETO_SEGMENT.items():
            if name not in skip:
                _set(p, rec, name, seg.get(task), f"{src} {task}")
        if rid in rollouter:
            _set(p, rec, "gen_s", rollouter[rid], f"rl_publication_check.rollouter_timing {ROLLOUTER_ACTIVE}")
        if mode == "async":
            _set(p, rec, "wait_s", seg.get("wait"), f"{src} kind=wait")
        _set(p, rec, "gen_tokens", lr.get("action_tokens"), "rl_local_round.action_tokens")
        _set(p, rec, "samples", t.get("trained_samples"), "rl_round_trained.trained_samples")
        _set(p, rec, "reward_mean", lr.get("reward_mean"), "rl_local_round.reward_mean")
        _set(p, rec, "is_clip_frac", (t.get("mismatch") or {}).get("tis_clipfrac"), "rl_round_trained.mismatch.tis_clipfrac")
        if mode == "async":
            _set(p, rec, "dropped_samples", drops.get(rid), "rl_rollout_carry_over.over_age_discarded_groups (groups)")
        f = fw.get(rid + framework_round_offset, {})
        for name in ROUND_FIELDS:  # what the tape lacks, from the framework's own log of the same run
            if rec[name] is None and f.get(name) is not None and name not in ("round", "t_end_unix", "round_s") \
                    and name not in skip:
                _set(p, rec, name, f[name], "framework log: " + framework_parsed.sources[name])
        p.rounds.append(rec)
        p.segments_raw.append({k: round(v, 3) for k, v in seg.items()})
    _round_s_from_ends(p)
    na = {"sampler_update_s"}  # single-island arms: no separate sampler island
    if mode == "sync":
        na |= {"wait_s", "dropped_samples", "generated_samples", "version_lag"}
    return _finish(p, na)


ROLLOUTER_ACTIVE = "fully_async/rollouter/active_time"


def _rollouter_active(checks: list[dict[str, Any]]) -> dict[int, float]:
    """rl_publication_check events -> rollout id (policy version - 1) -> rollouter active s."""
    out: dict[int, float] = {}
    for e in checks:
        value = (e.get("rollouter_timing") or {}).get(ROLLOUTER_ACTIVE)
        version = e.get("policy_version")
        if isinstance(value, (int, float)) and version is not None and int(version) >= 1:
            out[int(version) - 1] = float(value)
    return out


def _driver_phase_durations(markers: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    """rl_driver_phase markers (one per phase start) -> rollout id -> phase -> seconds
    (marker to the next marker; a marker without rollout_id belongs to the round in progress)."""
    out: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    rows = sorted(markers, key=lambda e: float(e.get("time_unix") or 0))
    current = 0
    for a, b in zip(rows, rows[1:]):
        if a.get("rollout_id") is not None:
            current = int(a["rollout_id"])
        out[current][str(a.get("phase"))] += float(b["time_unix"]) - float(a["time_unix"])
    return out


# ====================================================================== NVML
def parse_nvml(text: str) -> list[dict[str, Any]]:
    """Rows {t_unix, key (container:index), name, util, mem_mb, mem_total_mb, power_w}; header lines skipped."""
    rows = []
    for parts in csv.reader(io.StringIO(text)):
        parts = [x.strip() for x in parts]
        if not parts or parts[0].startswith("timestamp") or "timestamp" in parts:
            continue
        container = ""
        if not re.match(r"\d{4}/\d\d/\d\d", parts[0]):
            container, parts = parts[0], parts[1:]
        try:
            t = _dt.datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=_dt.timezone.utc).timestamp()
        except (ValueError, IndexError):
            continue
        num = [re.sub(r"[^\d.]", "", x) for x in parts]
        if len(parts) >= 7:  # timestamp,index,name,util,mem.used,mem.total,power
            index, util, mem, total, power = parts[1], num[3], num[4], num[5], num[6]
        else:  # timestamp,name,util,mem.used,power (T-EVAL format, no index)
            index, util, mem, total, power = "0", num[2], num[3], "", num[4] if len(num) > 4 else ""
        name = parts[2] if len(parts) >= 7 else parts[1]
        rows.append({"t_unix": t, "key": f"{container}:{index}", "name": name, "util": float(util) if util else None,
                     "mem_mb": float(mem) if mem else None, "mem_total_mb": float(total) if total else None,
                     "power_w": float(power) if power else None})
    return rows


# ====================================================================== run directory
def _read(path: Path) -> str:
    return path.read_text(errors="replace") if path.is_file() else ""


NOT_RUN_TAPES = ("yeto", "store")


def _tape_events(raw: Path) -> list[dict[str, Any]]:
    seen, rows = set(), []
    for path in sorted(raw.rglob("*.jsonl")):
        # miles_local.sh pulls a copy of the repo (raw/yeto, whose test fixtures are
        # tapes of other runs) and the checkpoint store next to the run's own tapes
        if "verl-" in path.name or path.relative_to(raw).parts[0] in NOT_RUN_TAPES:
            continue
        for row in tape_rows(_read(path)):
            key = json.dumps(row, sort_keys=True, default=str)  # the same event pulled from two sources
            if key not in seen:
                seen.add(key)
                rows.append(row)
    return rows


def parse_run(run_dir: Path) -> dict[str, Any]:
    """Parse a run directory (``arm.json`` + ``raw/``) into ``parsed.json`` content."""
    run_dir = Path(run_dir)
    plan = json.loads((run_dir / "arm.json").read_text())
    raw = run_dir / "raw"
    mode, spec, c = plan["mode"], plan["spec"], plan["cards"]
    arm_dir = next(iter(sorted(raw.glob("*/arm"))), raw / "arm")
    nvml_text = _read(arm_dir / "nvml.csv") or _read(raw / "nvml.csv")
    meta = json.loads(_read(arm_dir / "meta.json") or "{}")
    if plan["framework"] == "verl":
        metrics = next((p for p in sorted(raw.rglob("verl-metrics.jsonl"))), None) \
            or next((p for p in sorted(raw.rglob("verl-file-logger-*.jsonl"))), None)
        fw = parse_verl(_read(metrics) if metrics else "", _read(arm_dir / "main.log"),
                        samples_per_round=spec["groups_per_round"] * spec["samples_per_group"],
                        trainer_cards=c["trainer"], mode=mode)
    else:
        log = _read(arm_dir / "main.log") or _read(raw / "launch.ts.log")
        fw = parse_miles(log, rollout_cards=c["rollout"], mode=mode)
    # yeto+verl fully_async: the driver's generate phase (and verl's timing_s/gen) only
    # pull finished samples from the rollouter queue; generation time is the
    # rollouter's own active time, carried by the tape's publication checks
    verl_async = plan["framework"] == "verl" and mode == "async"
    skip = frozenset({"gen_s"}) if verl_async else frozenset()
    parsed = parse_yeto(_tape_events(raw), mode=mode, framework_parsed=fw, skip=skip,
                        framework_round_offset=1 if plan["framework"] == "verl" else 0,
                        rollouter_gen=verl_async) \
        if plan["system"] == "yeto" else fw
    if meta.get("rc") is None and _read(raw / "rc").strip().lstrip("-").isdigit():
        meta["rc"] = int(_read(raw / "rc"))  # miles_local.sh: the launcher's exit code
    start = meta.get("start_unix") or _utc_file(raw / "start_utc.txt")
    end = meta.get("end_unix") or _utc_file(raw / "end_utc.txt")
    return {"arm": plan["arm"], "run_id": plan["run_id"], "fingerprint": plan["fingerprint"], "mode": mode,
            "framework": plan["framework"], "system": plan["system"], "cards": c, "gpu": plan["gpu"],
            "spec": spec, "container_start_unix": start, "container_end_unix": end, "rc": meta.get("rc"),
            "framework_log": fw.to_dict() if plan["system"] == "yeto" else None,
            "nvml": (nvml := parse_nvml(nvml_text)),
            # E1: Modal "A10G" containers are NVIDIA A10 or (sometimes) A10G, which train slower
            "gpu_names": sorted({r["name"] for r in nvml if r.get("name")}),
            "nvml_interval_s": 20.0 if plan["harness"] == "miles_local" else 1.0,
            **parsed.to_dict()}


def _utc_file(path: Path) -> float | None:
    text = _read(path).strip()
    if not text:
        return None
    return _dt.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_dt.timezone.utc).timestamp()
