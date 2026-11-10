"""Run health breakers over a run's event tapes (``yeto breaker``).

Reads the trainer's events (``YETO_RL_EVENT`` lines of a launcher log, or a JSONL tape)
and the sampler event tapes, prints each breaker with its window value and limit,
and exits 1 when one trips (0: all ok or only warnings). Read-only: it never stops
anything; stopping is the operator's call (``yeto stop-run RUN``), or the in-run
guard's (``YETO_RL_XHW_GUARD``, :mod:`yeto.rl.engine.xhw_guard`).

The train/rollout log-prob mismatch (``train_rollout_logprob_abs_diff``) depends on the
sampler hardware: the same weights sampled on another card and engine differ from the
trainer's recompute by their numerics alone. Its limits come per hardware class
(:data:`LOGPROB_LIMITS`): ``amd`` uses the in-run guard's defaults
(:class:`~yeto.rl.engine.xhw_guard.XhwGuard`, set above the measured MI355X vLLM band),
``nvidia`` the B300 SGLang baseline. The class is read from the sampler tapes
(``rl_sampler_engine_round.source``, ``rl_xhw_mismatch`` rows), or given with
``--sampler-hardware``. A run with any AMD sampler is judged by the ``amd`` limits
(the step value mixes every source's tokens); an unknown one too. Per-source rows of
the in-run guard (``rl_xhw_mismatch``) are reported with the level that guard gave.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from yeto.rl.engine.xhw_guard import XhwGuard, step_value

EVENT_PREFIX = "YETO_RL_EVENT "

# fixed limits: (value, "max" = trips above, "min" = trips below)
LIMITS: dict[str, tuple[float, str]] = {
    "is_clipped": (0.05, "max"),        # sc_is_clipped, mean over the window
    "is_weight": (0.9, "min"),          # sc_importance_weight
    "stale": (0.30, "max"),             # stale groups / (stale + trained)
    "flips": (0.01, "max"),             # changed / total weights of the last release (--flips)
    "aborted": (0.20, "max"),           # agentic trials lost / trials
    "zero_call": (0.05, "max"),         # agentic trials retried before any model call / trials
    "nan_skip_consecutive": (2, "max"),
    "nan_skip_fraction": (0.05, "max"),  # judged from NAN_SKIP_MIN_ROUNDS trained rounds on
}
NAN_SKIP_MIN_ROUNDS = 20

_GUARD = XhwGuard()
# (warn, act) of train_rollout_logprob_abs_diff per sampler hardware class.
# amd: the in-run guard's defaults (MI355X vLLM -> B300 Megatron measured 0.076-0.099).
# nvidia: B300 SGLang -> B300 Megatron measured 0.023-0.028; act at the 0.052 baseline.
LOGPROB_LIMITS: dict[str, tuple[float, float]] = {
    "amd": (_GUARD.warn.abs_diff, _GUARD.act.abs_diff),
    "nvidia": (0.04, 0.052),
}
HARDWARE = ("auto", *LOGPROB_LIMITS)
NOISE_FLOOR_HEADROOM = 1.5  # --noise-floor: act at least this many times the measured floor


def hardware_class(device_family: str | None = None, card_model: str | None = None) -> str | None:
    """``amd`` / ``nvidia`` of a sampler, None when neither field tells."""
    family = (device_family or "").strip().lower()
    card = (card_model or "").strip().upper()
    if family == "rocm" or card.startswith("MI"):
        return "amd"
    if family == "cuda" or card[:1] in {"A", "B", "H", "L"} and card[1:2].isdigit():
        return "nvidia"
    return None


def source_hardware(source: str) -> str | None:
    """Hardware class of an engine-round source key, ``family/card/engine@version/...``."""
    parts = str(source or "").split("/")
    return hardware_class(parts[0] if parts else None, parts[1] if len(parts) > 1 else None)


def events(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    """Event records of one file: JSONL lines, or ``YETO_RL_EVENT {json}`` lines of a log."""
    with open(path, errors="replace") as fh:
        for line in fh:
            i = line.find(EVENT_PREFIX)
            text = line[i + len(EVENT_PREFIX):] if i >= 0 else line
            try:
                record = json.loads(text)
            except ValueError:
                continue
            if isinstance(record, dict) and "event" in record:
                yield record


def tape_files(paths: Iterable[str]) -> list[str]:
    """Every ``*.jsonl`` under each directory, each file as is, each glob expanded."""
    out: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            out.extend(sorted(glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True)))
        elif any(c in p for c in "*?["):
            out.extend(sorted(glob.glob(p, recursive=True)))
        else:
            out.append(p)
    return out


def _mean(xs: Iterable[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


@dataclass
class Check:
    name: str
    value: float | None
    limit: float
    direction: str = "max"
    warn: float | None = None
    note: str = ""

    @property
    def level(self) -> str:
        v = self.value
        if v is None:
            return "ok"
        if (v < self.limit) if self.direction == "min" else (v > self.limit):
            return "trip"
        if self.warn is not None and v > self.warn:
            return "warn"
        return "ok"

    def line(self) -> str:
        tag = {"trip": "TRIP", "warn": "WARN", "ok": "ok  "}[self.level]
        shown = "-" if self.value is None else f"{round(self.value, 4)}"
        op = "<" if self.direction == "min" else ">"
        warn = f", warn > {self.warn}" if self.warn is not None else ""
        return f"{tag} {self.name:20s} {shown:>8} ({op} {self.limit}{warn}){'  ' + self.note if self.note else ''}"

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value, "limit": self.limit, "direction": self.direction,
                "warn": self.warn, "level": self.level, **({"note": self.note} if self.note else {})}


def logprob_limits(hardware: str, *, noise_floor: float | None = None, warn: float | None = None,
                   act: float | None = None) -> tuple[float, float]:
    w, a = LOGPROB_LIMITS[hardware]
    if noise_floor is not None:
        a = max(a, noise_floor * NOISE_FLOOR_HEADROOM)
        w = max(w, noise_floor)
    w, a = (w if warn is None else warn), (a if act is None else act)
    if not 0 < w <= a:
        raise ValueError(f"logprob limits need 0 < warn ({w}) <= act ({a})")
    return w, a


def evaluate(trainer: Sequence[Mapping[str, Any]], sampler: Sequence[Mapping[str, Any]], *,
             window: int = 3, hardware: str = "auto", noise_floor: float | None = None,
             logprob_warn: float | None = None, logprob_act: float | None = None,
             flips: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """The breakers over the given events; ``{"checks", "xhw", "tripped", "hardware", ...}``."""
    if window < 1:
        raise ValueError("window must be >= 1")
    if hardware not in HARDWARE:
        raise ValueError(f"sampler hardware must be one of {HARDWARE}")
    all_trained = [d for d in trainer if d.get("event") == "rl_round_trained"]
    trained = all_trained[-window:]
    stale = [d for d in trainer if d.get("event") == "rl_buffer_stale_skipped"][-window:]
    xhw = [d for d in trainer if d.get("event") == "rl_xhw_mismatch"]
    rounds = [d for d in sampler if d.get("event") == "rl_sampler_engine_round"]

    seen = {source_hardware(d.get("source", "")) for d in rounds}
    for d in xhw[-1:]:
        for row in (d.get("sources") or {}).values():
            seen.add(hardware_class(None, row.get("card_model")))
    known = sorted(h for h in seen if h)
    note = ""
    if hardware == "auto":
        if "amd" in known or not known:
            hw = "amd"
            if not known:
                note = "sampler hardware unknown: amd limits (pass --sampler-hardware)"
        else:
            hw = "nvidia"
    else:
        hw = hardware
    warn, act = logprob_limits(hw, noise_floor=noise_floor, warn=logprob_warn, act=logprob_act)

    def metric(name: str) -> float | None:
        return _mean(step_value(d.get("train_metrics") or {}, name) for d in trained)

    checks = [
        Check("is_clipped", metric("sc_is_clipped"), *LIMITS["is_clipped"]),
        Check("is_weight", metric("sc_importance_weight"), *LIMITS["is_weight"]),
        Check("logprob_diff", metric("train_rollout_logprob_abs_diff"), act, "max", warn,
              note or f"{hw} sampler limits"),
    ]
    n_stale = sum(int(d.get("stale_groups") or 0) for d in stale)
    n_trained = sum(int(d.get("trained_groups") or 0) for d in trained)
    checks.append(Check("stale", n_stale / (n_stale + n_trained) if n_stale + n_trained else None,
                        *LIMITS["stale"]))
    agentic = sorted((d for d in rounds if d.get("agentic")), key=lambda d: d.get("time_unix", 0))
    agentic = agentic[-window * 3:]
    trials = sum(int(d["agentic"].get("trials") or 0) for d in agentic)
    lost = sum(sum((d["agentic"].get("lost") or {}).values()) for d in agentic)
    # infra retries run only for a trial aborted before the policy produced a turn
    zero = sum(sum((d["agentic"].get("infra_retries") or {}).values()) for d in agentic)
    checks.append(Check("aborted", lost / trials if trials else None, *LIMITS["aborted"]))
    checks.append(Check("zero_call", zero / trials if trials else None, *LIMITS["zero_call"]))
    flip = None
    if flips:
        last = list(flips.values())[-1]
        flip = float(last["changed"]) / float(last["total"])
    checks.append(Check("flips", flip, *LIMITS["flips"]))

    skips = [d for d in trainer if d.get("event") == "rl_nan_skip"]
    consecutive = int(skips[-1].get("nan_consecutive") or 0) if skips else 0
    if skips and all_trained and all_trained[-1].get("rollout_id", -1) > skips[-1].get("rollout_id", -1):
        consecutive = 0
    fraction = len(skips) / len(all_trained) if len(all_trained) >= NAN_SKIP_MIN_ROUNDS else 0.0
    checks.append(Check("nan_skip_consecutive", float(consecutive), *LIMITS["nan_skip_consecutive"]))
    checks.append(Check("nan_skip_fraction", fraction, *LIMITS["nan_skip_fraction"]))

    r3_errors = sum(1 for d in sampler if "routed_experts has" in json.dumps(d, default=str))
    checks.append(Check("r3_rows", float(r3_errors), 0.0))

    xhw_rows = []
    tripped = [c.name for c in checks if c.level == "trip"]
    if xhw:
        last = xhw[-1]
        for src, row in sorted((last.get("sources") or {}).items()):
            bad = row.get("level") == "act" or "stop" in last
            xhw_rows.append({"source": src, **{k: row.get(k) for k in (
                "sampler_id", "card_model", "abs_diff", "kl", "clip", "level", "action")}, "trip": bad})
            if bad:
                tripped.append(f"xhw:{row.get('sampler_id')}")
    return {"checks": [c.to_json() for c in checks], "xhw": xhw_rows, "tripped": tripped,
            "hardware": hw, "hardware_seen": known, "logprob_limits": {"warn": warn, "act": act},
            "trained_steps": len(trained), "sampler_rounds": len(rounds), "_checks": checks}


def add_parser(sub: Any) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "breaker", help="check a run's event tapes against health limits (exit 1 on a trip)",
        description="Read-only health breakers over a run's event tapes; exit 1 when one trips. "
                    "See docs/BREAKER.md.")
    p.add_argument("--trainer-log", required=True, nargs="+",
                   help="trainer events: launcher log(s) with YETO_RL_EVENT lines, JSONL tapes or directories")
    p.add_argument("--sampler-tapes", nargs="*", default=[],
                   help="sampler event tapes: files, globs or directories (every *.jsonl under them)")
    p.add_argument("--flips", help="JSON {release: {changed, total}} of weight flips per release")
    p.add_argument("--window", type=int, default=3, help="trailing trained steps (sampler: 3x rounds)")
    p.add_argument("--sampler-hardware", choices=HARDWARE, default="auto",
                   help="hardware class whose log-prob mismatch limits apply (auto: from the sampler tapes)")
    p.add_argument("--noise-floor", type=float,
                   help=f"measured repeat-run logprob abs_diff; act at least {NOISE_FLOOR_HEADROOM}x it")
    p.add_argument("--logprob-warn", type=float, help="override the logprob_diff warn limit")
    p.add_argument("--logprob-act", type=float, help="override the logprob_diff trip limit")
    p.add_argument("--json", action="store_true", help="print one JSON object instead of the table")
    return p


def cmd_breaker(args: argparse.Namespace) -> int:
    try:
        trainer = [e for f in tape_files(args.trainer_log) for e in events(f)]
        sampler = [e for f in tape_files(args.sampler_tapes) for e in events(f)]
        flips = None
        if args.flips:
            with open(args.flips) as fh:
                flips = json.load(fh)
        out = evaluate(trainer, sampler, window=args.window, hardware=args.sampler_hardware,
                       noise_floor=args.noise_floor, logprob_warn=args.logprob_warn,
                       logprob_act=args.logprob_act, flips=flips)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[breaker] {exc}", file=sys.stderr)
        return 2
    checks = out.pop("_checks")
    if args.json:
        print(json.dumps(out, sort_keys=True))
    else:
        for c in checks:
            print(c.line())
        for row in out["xhw"]:
            print(f"{'TRIP' if row['trip'] else 'ok  '} xhw {row['sampler_id']}/{row['card_model']} "
                  f"abs_diff={row['abs_diff']} kl={row['kl']} clip={row['clip']} "
                  f"level={row['level']} action={row['action']}")
        print(json.dumps({"tripped": out["tripped"], "hardware": out["hardware"],
                          "trained_steps": out["trained_steps"], "sampler_rounds": out["sampler_rounds"]}))
    return 1 if out["tripped"] else 0
