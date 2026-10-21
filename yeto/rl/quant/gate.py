"""Gate between rollout quantization tiers (S26 E6): a quantized sampler against BF16.

A tier may replace the previous one only when, on one fixed prompt set, its
samples stay close to what the BF16 policy would assign them. For each
completion the quantized engine generates (the behaviour policy of the rollout),
the BF16 engine scores the same text teacher-forced (the policy the trainer
holds). Per token:

* ``diff = logprob_quant - logprob_bf16`` (signed mean, and the mean and
  percentiles p50 / p90 / p99 / max of ``|diff|``);
* the TIS weight ``exp(logprob_bf16 - logprob_quant)`` and the share of tokens
  it truncates (above ``tis_upper``, or below ``tis_lower`` when that is > 0),
  the same bounds as the run's TIS;
* generation speed: output tokens per second of each engine on the same prompts
  (wall clock of one ``generate`` call over the whole set) and their ratio.

The criteria (:class:`GateCriteria`) are read from a JSON file, so each step
(tier 0 -> 1, 0 -> 2, ...) states its own; a criterion left ``None`` is
reported but not applied. The report is written to disk before the verdict is
printed; the exit code is 0 only when every applied criterion passes.

Two ways in:

* ``collect``: two engine sections (``yeto.rl.adapters.sampler_engines.from_spec``)
  and a prompt file; the engines run one after the other (one GPU is enough);
* ``records``: per-completion records that already hold both log-prob lists
  (e.g. ``rollout_log_probs`` and the trainer's ``log_probs`` of an RL round).

A completion whose BF16 score has a different token count than the generated
completion (the text tokenized differently) is not compared; the share of such
completions is a criterion of its own (``max_unaligned_share``).

Status: CPU tests with the fake engine (tests/test_rl_quant_gate.py); not run
on a GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

SCHEMA = "yeto.rl.quant.gate/v1"


@dataclass(frozen=True)
class GateCriteria:
    """Pass rules of one tier step; ``None`` = reported only."""

    tis_upper: float = 2.0
    tis_lower: float = 0.0
    max_mean_abs_diff: float | None = None
    max_p99_abs_diff: float | None = None
    max_tis_clip_share: float | None = None
    min_speedup: float | None = None  # quant tokens/s / bf16 tokens/s
    max_unaligned_share: float | None = 0.05

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "GateCriteria":
        known = {f.name for f in fields(cls)}
        if unknown := set(raw) - known:
            raise ValueError(f"unknown gate criteria {sorted(unknown)}; known: {sorted(known)}")
        criteria = cls(**raw)
        if not criteria.tis_upper > 1.0 or not 0.0 <= criteria.tis_lower < 1.0:
            raise ValueError(f"TIS bounds need 0 <= lower < 1 < upper, got {criteria.tis_lower}, {criteria.tis_upper}")
        return criteria


@dataclass(frozen=True)
class Pair:
    """One completion: the quantized engine's log-probs and the BF16 scores of the same tokens."""

    quant: Sequence[float]
    bf16: Sequence[float]


def quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile of ``values`` (sorted here), ``0 <= q <= 1``."""
    if not values:
        return math.nan
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    low = math.floor(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def compare(pairs: Iterable[Pair], criteria: GateCriteria) -> dict[str, Any]:
    """Token statistics over the aligned pairs (module doc)."""
    diffs: list[float] = []
    clipped_high = clipped_low = completions = unaligned = 0
    for pair in pairs:
        completions += 1
        if len(pair.quant) != len(pair.bf16):
            unaligned += 1
            continue
        for lq, lb in zip(pair.quant, pair.bf16):
            lq, lb = float(lq), float(lb)
            if not (math.isfinite(lq) and math.isfinite(lb)):
                raise ValueError(f"non-finite log-prob (quant {lq}, bf16 {lb})")
            diffs.append(lq - lb)
            log_ratio = lb - lq
            if log_ratio > math.log(criteria.tis_upper):
                clipped_high += 1
            elif criteria.tis_lower > 0 and log_ratio < math.log(criteria.tis_lower):
                clipped_low += 1
    absolute = [abs(d) for d in diffs]
    tokens = len(diffs)
    return {
        "completions": completions,
        "unaligned_completions": unaligned,
        "unaligned_share": unaligned / completions if completions else math.nan,
        "tokens": tokens,
        "mean_diff": sum(diffs) / tokens if tokens else math.nan,
        "mean_abs_diff": sum(absolute) / tokens if tokens else math.nan,
        "p50_abs_diff": quantile(absolute, 0.50),
        "p90_abs_diff": quantile(absolute, 0.90),
        "p99_abs_diff": quantile(absolute, 0.99),
        "max_abs_diff": max(absolute) if absolute else math.nan,
        "tis_clip_high_share": clipped_high / tokens if tokens else math.nan,
        "tis_clip_low_share": clipped_low / tokens if tokens else math.nan,
        "tis_clip_share": (clipped_high + clipped_low) / tokens if tokens else math.nan,
    }


def verdict(stats: Mapping[str, Any], speed: Mapping[str, Any] | None, criteria: GateCriteria) -> dict[str, Any]:
    """Each applied criterion with its value and result; ``passed`` is their conjunction.
    A criterion whose value is missing or NaN fails (never a silent pass)."""
    checks: dict[str, dict[str, Any]] = {}

    def at_most(name: str, value: Any, limit: float | None) -> None:
        if limit is not None:
            ok = isinstance(value, (int, float)) and math.isfinite(value) and value <= limit
            checks[name] = {"value": value, "limit": limit, "passed": bool(ok)}

    at_most("max_mean_abs_diff", stats.get("mean_abs_diff"), criteria.max_mean_abs_diff)
    at_most("max_p99_abs_diff", stats.get("p99_abs_diff"), criteria.max_p99_abs_diff)
    at_most("max_tis_clip_share", stats.get("tis_clip_share"), criteria.max_tis_clip_share)
    at_most("max_unaligned_share", stats.get("unaligned_share"), criteria.max_unaligned_share)
    if criteria.min_speedup is not None:
        value = None if speed is None else speed.get("speedup")
        ok = isinstance(value, (int, float)) and math.isfinite(value) and value >= criteria.min_speedup
        checks["min_speedup"] = {"value": value, "limit": criteria.min_speedup, "passed": bool(ok)}
    if not stats.get("tokens"):
        checks["tokens"] = {"value": stats.get("tokens"), "limit": "> 0", "passed": False}
    return {"passed": all(c["passed"] for c in checks.values()), "checks": checks}


def report(pairs: Sequence[Pair], criteria: GateCriteria, *, speed: Mapping[str, Any] | None = None,
           meta: Mapping[str, Any] | None = None) -> dict[str, Any]:
    stats = compare(pairs, criteria)
    return {"schema": SCHEMA, "criteria": asdict(criteria), "stats": stats, "speed": dict(speed or {}) or None,
            "verdict": verdict(stats, speed, criteria), "meta": dict(meta or {})}


# ---------------------------------------------------------------- collection over engines
def _tokens_per_s(completions: Sequence[Sequence[Any]], seconds: float) -> dict[str, Any]:
    tokens = sum(len(c.token_ids) for group in completions for c in group)
    return {"output_tokens": tokens, "seconds": seconds, "tokens_per_s": tokens / seconds if seconds > 0 else math.nan}


def collect(quant: Any, bf16: Any, prompts: Sequence[str], sampling: Any,
            clock: Any = time.monotonic) -> tuple[list[Pair], dict[str, Any], list[dict[str, Any]]]:
    """Run the quantized engine, then the BF16 engine, on ``prompts`` (module doc).

    Returns the pairs, the speed record and one raw record per completion (to save
    before anything is computed from them). Each engine is started on its base weights
    (``start(None, 0)``) and stopped afterwards, so one GPU holds one engine at a time.
    """
    quant.start(None, 0)
    try:
        t0 = clock()
        generated = quant.generate(list(prompts), sampling)
        quant_speed = _tokens_per_s(generated, clock() - t0)
    finally:
        quant.stop()
    bf16.start(None, 0)
    try:
        t0 = clock()
        bf16_generated = bf16.generate(list(prompts), sampling)
        bf16_speed = _tokens_per_s(bf16_generated, clock() - t0)
        pairs: list[Pair] = []
        raw: list[dict[str, Any]] = []
        for prompt_index, (prompt, group) in enumerate(zip(prompts, generated)):
            for sample_index, completion in enumerate(group):
                # by token ids when the engine can (a trailing EOS has no text to re-tokenize)
                scores = (list(bf16.score_tokens(prompt, completion.token_ids)) if hasattr(bf16, "score_tokens")
                          else list(bf16.score(prompt, completion.text)))
                pairs.append(Pair(tuple(completion.logprobs), tuple(scores)))
                raw.append({"prompt_index": prompt_index, "sample": sample_index,
                            "token_ids": list(completion.token_ids), "quant_logprobs": list(completion.logprobs),
                            "bf16_logprobs": scores, "finish_reason": completion.finish_reason})
    finally:
        bf16.stop()
    speedup = (quant_speed["tokens_per_s"] / bf16_speed["tokens_per_s"]
               if bf16_speed["tokens_per_s"] and math.isfinite(bf16_speed["tokens_per_s"]) else math.nan)
    speed = {"quant": quant_speed, "bf16": bf16_speed, "speedup": speedup}
    return pairs, speed, raw


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _read_records(path: Path, quant_key: str, bf16_key: str) -> list[Pair]:
    pairs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            pairs.append(Pair(tuple(row[quant_key]), tuple(row[bf16_key])))
    return pairs


def _load_criteria(path: str | None) -> GateCriteria:
    return GateCriteria.from_mapping(json.loads(Path(path).read_text())) if path else GateCriteria()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m yeto.rl.quant.gate", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="run both engines on a prompt file and gate")
    c.add_argument("--engines", required=True, help='JSON file {"quant": <engine section>, "bf16": <engine section>}')
    c.add_argument("--prompts", required=True, help="JSONL with a 'prompt' field per line")
    c.add_argument("--max-tokens", type=int, required=True)
    c.add_argument("--temperature", type=float, default=1.0)
    c.add_argument("--n", type=int, default=1)
    c.add_argument("--seed", type=int, default=1)
    r = sub.add_parser("records", help="gate per-completion records that hold both log-prob lists")
    r.add_argument("--records", required=True)
    r.add_argument("--quant-key", default="rollout_log_probs")
    r.add_argument("--bf16-key", default="log_probs")
    for q in (c, r):
        q.add_argument("--criteria", help="JSON file of GateCriteria fields")
        q.add_argument("--out", required=True, help="report JSON (raw records next to it for collect)")
        q.add_argument("--label", default="", help="free text stored in the report (e.g. 'tier0->tier1 Qwen3-4B')")
    a = p.parse_args(argv)
    criteria = _load_criteria(a.criteria)
    out = Path(a.out)
    meta: dict[str, Any] = {"label": a.label, "cmd": a.cmd, "created_unix": time.time()}
    if a.cmd == "collect":
        from yeto.rl.adapters.sampler_engines.from_spec import build
        from yeto.rl.engine.sampler_engine import SamplingParams

        engines = json.loads(Path(a.engines).read_text())
        prompts = [json.loads(line)["prompt"] for line in Path(a.prompts).read_text().splitlines() if line.strip()]
        sampling = SamplingParams(max_tokens=a.max_tokens, temperature=a.temperature, n=a.n, seed=a.seed).check()
        pairs, speed, raw = collect(build(engines["quant"]), build(engines["bf16"]), prompts, sampling)
        _write_json(out.with_name(out.stem + "-raw.json"), raw)  # raw data first
        meta.update(prompts=len(prompts), sampling=sampling.to_dict(), engines=engines)
        result = report(pairs, criteria, speed=speed, meta=meta)
    else:
        meta["records"] = a.records
        result = report(_read_records(Path(a.records), a.quant_key, a.bf16_key), criteria, meta=meta)
    _write_json(out, result)
    print(json.dumps({"out": str(out), "verdict": result["verdict"], "stats": result["stats"]}, indent=1, default=str))
    return 0 if result["verdict"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
