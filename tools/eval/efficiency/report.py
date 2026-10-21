"""Markdown report of a set of efficiency runs (``parsed.json`` in each run dir).

Sections: the shared spec and its fingerprint; one row per run; per-arm
aggregates (median and min-max over repeats); yeto vs bare per framework and
mode, metric by metric, with a range verdict; yeto's own segments; reward
curves; and the fields each arm could not measure.

Range verdict (repeats are not bit-reproducible on H100, so no single-run
comparison): with >= :data:`MIN_REPEATS` runs on both sides, yeto is "worse"
or "better" only when its min-max range and bare's do not overlap; otherwise
"within run-to-run range". Fewer runs, or a missing value, is said as such.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .metrics import summarize
from .spec import ARMS, FRAMEWORKS, MODES, EffSpec, verl_knob_diff

MIN_REPEATS = 2
# metric -> (label, True when lower is better)
COMPARED = {
    "round_s_median": ("round time (s)", True),
    "gen_s_median": ("generation (s/round)", True),
    "train_s_median": ("training (s/round)", True),
    "weight_sync_s_median": ("weight sync trainer->rollout (s/round)", True),
    "sampler_update_s_median": ("update sampler weights (s/round)", True),
    "wait_s_median": ("wait (s/round)", True),
    "unaccounted_s_median": ("other, not in the segments above (s/round)", True),
    "gen_tok_per_card_s": ("generated tokens / card / s", False),
    "train_tok_per_card_s": ("trained tokens / card / s", False),
    "samples_per_s": ("trained samples / s", False),
    "gpu_util_mean": ("GPU utilization mean (%)", False),
    "bubble_frac": ("idle bubble share (util < 5 %)", True),
    "gpu_mem_peak_mb": ("GPU memory peak (MiB)", True),
    "usd_per_1k_samples_steady": ("USD / 1k samples (steady)", True),
    "usd_per_1k_samples_run": ("USD / 1k samples (whole container run)", True),
}


def load(run_dirs: list[Path]) -> list[dict[str, Any]]:
    out = []
    for d in run_dirs:
        path = Path(d) / "parsed.json"
        if path.is_file():
            out.append(json.loads(path.read_text()))
    return out


def _fmt(v: Any) -> str:
    if v is None:
        return "missing"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _range(values: list[float]) -> str:
    return "missing" if not values else f"{_fmt(statistics.median(values))} [{_fmt(min(values))}, {_fmt(max(values))}]"


def verdict(bare: list[float | None], yeto: list[float | None], lower_is_better: bool) -> str:
    if any(v is None for v in bare + yeto) or not bare or not yeto:
        return "missing value"
    if len(bare) < MIN_REPEATS or len(yeto) < MIN_REPEATS:
        return f"insufficient repeats (bare {len(bare)}, yeto {len(yeto)}; need {MIN_REPEATS})"
    if max(yeto) < min(bare):
        return "yeto better" if lower_is_better else "yeto worse"
    if min(yeto) > max(bare):
        return "yeto worse" if lower_is_better else "yeto better"
    return "within run-to-run range"


def _delta(bare: list[float], yeto: list[float]) -> str:
    if not bare or not yeto or any(v is None for v in bare + yeto):
        return "missing"
    b, y = statistics.median(bare), statistics.median(yeto)
    if b == 0:
        return f"{y - b:+.4g} (abs)"
    return f"{y - b:+.4g} ({(y - b) / abs(b) * 100:+.1f} %)"


def markdown(parsed_runs: list[dict[str, Any]], *, title: str) -> str:
    if not parsed_runs:
        return f"# {title}\n\nNo parsed runs.\n"
    prints = {p["fingerprint"] for p in parsed_runs}
    if len(prints) > 1:
        raise ValueError(f"runs come from different specs (fingerprints {sorted(prints)}); report them separately")
    models = {tuple(p.get("gpu_names") or ()) for p in parsed_runs if p.get("gpu_names")}
    if len(models) > 1:
        # E1: e1a9 ran on an NVIDIA A10G, the others on an A10; its train step was ~12 % slower
        raise ValueError(f"runs ran on different GPU models {sorted(models)}; report them separately")
    spec = parsed_runs[0]["spec"]
    summaries = [summarize(p) for p in parsed_runs]
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for s in summaries:
        by_arm[s["arm"]].append(s)
    lines = [f"# {title}", "",
             f"Spec `{spec['name']}`, fingerprint `{parsed_runs[0]['fingerprint'][:16]}`: model `{spec['model']}`@"
             f"`{spec['model_revision'][:10]}`, data `{spec['data']}`@`{spec['data_revision'][:10]}`, "
             f"{spec['rounds']} rounds x {spec['groups_per_round']} prompts x {spec['samples_per_group']} samples, "
             f"max_tokens {spec['max_response_len']}, lr {spec['lr']}, LoRA r {spec['lora_r']}, seed {spec['seed']}, "
             f"{spec['gpu']} (sync 1 card colocated, async 1 rollout + 1 trainer card), "
             f"max_policy_age {spec['max_policy_age']} (async), TIS upper {spec['tis_upper']}.", "",
             "Steady window = rounds after the first. Throughput counts every card of the arm. "
             "Costs are list-price estimates (`prices.json`), not a bill.", ""]
    lines += _runs_table(summaries) + _arms_table(by_arm) + _overhead(by_arm) + _yeto_segments(by_arm)
    lines += _knobs(spec) + _rewards(summaries) + _missing(by_arm)
    return "\n".join(lines) + "\n"


def _runs_table(summaries: list[dict]) -> list[str]:
    head = ("arm", "run", "rc", "rounds", "steady wall s", "gen tok/card/s", "train tok/card/s", "round s med",
            "util %", "bubble", "mem peak MiB", "$/1k steady", "$/1k run", "reward last3")
    rows = ["## Runs", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for s in sorted(summaries, key=lambda s: (ARMS.index(s["arm"]), s["run_id"])):
        vals = (s["arm"], f"`{s['run_id']}`", s["rc"], f"{s['rounds']}/{s['rounds_expected']}", s["steady_wall_s"],
                s["gen_tok_per_card_s"], s["train_tok_per_card_s"], s["round_s_median"], s["gpu_util_mean"],
                s["bubble_frac"], s["gpu_mem_peak_mb"], s["usd_per_1k_samples_steady"], s["usd_per_1k_samples_run"],
                s["reward_last3_mean"])
        rows.append("| " + " | ".join(_fmt(v) for v in vals) + " |")
    return rows + [""]


def _arms_table(by_arm: dict[str, list[dict]]) -> list[str]:
    rows = ["## Per arm (median [min, max] over repeats)", "",
            "| metric | " + " | ".join(a for a in ARMS if a in by_arm) + " |", "|---|" + "---|" * len(by_arm)]
    for key, (label, _) in COMPARED.items():
        cells = []
        for a in ARMS:
            if a in by_arm:
                vals = [s[key] for s in by_arm[a]]
                cells.append(_range([v for v in vals if v is not None]) if any(v is not None for v in vals)
                             else ("n/a" if all(s["sources"].get(key.removesuffix("_median")) == "n/a"
                                                for s in by_arm[a]) else "missing"))
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return rows + [""]


def _overhead(by_arm: dict[str, list[dict]]) -> list[str]:
    rows = ["## yeto vs bare (same framework and mode)", "",
            "Delta = median(yeto) - median(bare); positive time = yeto slower.", ""]
    for fw in FRAMEWORKS:
        for mode in MODES:
            bare, yeto = by_arm.get(f"bare-{fw}-{mode}", []), by_arm.get(f"yeto-{fw}-{mode}", [])
            if not bare and not yeto:
                continue
            rows += [f"### {fw} {mode} (bare n={len(bare)}, yeto n={len(yeto)})", "",
                     "| metric | bare | yeto | delta | verdict |", "|---|---|---|---|---|"]
            for key, (label, lower) in COMPARED.items():
                bv, yv = [s[key] for s in bare], [s[key] for s in yeto]
                if all(s["sources"].get(key.removesuffix("_median")) == "n/a" for s in bare + yeto):
                    rows.append(f"| {label} | n/a | n/a | n/a | not applicable |")
                    continue
                rows.append(f"| {label} | {_range([v for v in bv if v is not None])} | "
                            f"{_range([v for v in yv if v is not None])} | {_delta(bv, yv)} | "
                            f"{verdict(bv, yv, lower)} |")
            rows.append("")
    return rows


def _yeto_segments(by_arm: dict[str, list[dict]]) -> list[str]:
    rows = ["## Native segments per round (median, steady window)", "",
            "yeto: tape task/phase names; bare: the framework's own timers. Where yeto adds work "
            "(offload/onload, outer sync, publish), it shows here.", ""]
    for a in ARMS:
        for s in by_arm.get(a, []):
            raw = s.get("segments_raw_median") or {}
            rows.append(f"- {a} `{s['run_id']}`: " + (", ".join(f"{k} {_fmt(v)}" for k, v in raw.items())
                                                       or "missing"))
    return rows + [""]


def _knobs(spec: dict) -> list[str]:
    rows = ["## verl overrides that differ between bare and yeto (same spec)", ""]
    try:
        for mode in MODES:
            diff = verl_knob_diff(EffSpec(**spec), mode)
            rows.append(f"- {mode}: " + ("; ".join(f"`{k}` bare={v['bare']} yeto={v['yeto']}"
                                                  for k, v in diff.items()) or "none"))
    except ImportError as exc:
        rows.append(f"- not computed ({exc})")
    rows += ["- Miles: the yeto launcher builds the Miles argv at run time; compare the launch log's argv with "
             "the bare script (`arm.json`) by hand (not automated in E0.1).", ""]
    return rows


def _rewards(summaries: list[dict]) -> list[str]:
    rows = ["## Reward against wall time and samples", "", "Per run: round: t_s / samples / reward_mean.", ""]
    for s in sorted(summaries, key=lambda s: (ARMS.index(s["arm"]), s["run_id"])):
        pts = " ".join(f"{c['round']}:{_fmt(c['t_s'])}/{_fmt(c['samples'])}/{_fmt(c['reward_mean'])}"
                       for c in s["reward_curve"])
        rows.append(f"- {s['arm']} `{s['run_id']}`: {pts or 'missing'}")
    return rows + [""]


def _missing(by_arm: dict[str, list[dict]]) -> list[str]:
    rows = ["## Field sources and gaps", "", "Per arm (first run): each round field's source; `missing` = no "
            "source in the raw data, `n/a` = does not apply.", ""]
    for a in ARMS:
        if a in by_arm:
            src = by_arm[a][0]["sources"]
            rows.append(f"- {a}: " + "; ".join(f"{k}: {v}" for k, v in src.items()))
            partial = {k: v for s in by_arm[a] for k, v in (s.get("rate_rounds") or {}).items()}
            if partial:
                rows.append(f"  - {a}: rates over the steady rounds that report the value: "
                            + "; ".join(f"{k} {v}" for k, v in partial.items()))
    return rows + [""]
