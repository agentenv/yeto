#!/usr/bin/env python3
"""Build reusable group-local feature rows from candidate and policy replay artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_group_local():
    path = REPO_ROOT / "scripts" / "replay_group_local_probecommit.py"
    spec = importlib.util.spec_from_file_location("_group_local_probecommit", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_group_local_probecommit"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


gl = _load_group_local()


def _jsonable(value):
    return gl.jsonable(value)


def _score_entropy(scores: list[float]) -> float:
    if len(scores) <= 1:
        return 0.0
    scale = gl.std(scores)
    if not math.isfinite(scale) or scale <= 1e-12:
        return 1.0
    mean = gl.mean(scores)
    zs = [(s - mean) / scale for s in scores]
    max_z = max(zs)
    exps = [math.exp(min(30.0, z - max_z)) for z in zs]
    total = sum(exps)
    if total <= 0.0:
        return 1.0
    probs = [v / total for v in exps]
    entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
    return entropy / math.log(len(probs))


def _field_stats(candidates: list[dict], field: str) -> dict:
    stats = dict(gl.group_stats(candidates, field))
    ordered = sorted(candidates, key=lambda r: gl.score_value(r, field), reverse=True)
    scores = [gl.score_value(r, field) for r in ordered]
    score_q50 = gl.quantile(scores, 0.50)
    stats.update(
        {
            "score_q10": gl.quantile(scores, 0.10),
            "score_q25": gl.quantile(scores, 0.25),
            "score_q50": score_q50,
            "score_q75": gl.quantile(scores, 0.75),
            "score_q90": gl.quantile(scores, 0.90),
            "score_entropy": _score_entropy(scores),
            "top1_learner": int(ordered[0].get("learner_id", -1)),
            "bottom1_learner": int(ordered[-1].get("learner_id", -1)),
            "top2_gap": scores[0] - scores[1] if len(scores) > 1 else 0.0,
            "top3_gap": scores[0] - scores[2] if len(scores) > 2 else 0.0,
            "above_median_mass": gl.selected_mass(
                [r for r in candidates if gl.score_value(r, field) >= score_q50],
                candidates,
            ),
            "positive_score_mass": gl.selected_mass(
                [r for r in candidates if gl.score_value(r, field) > 0.0],
                candidates,
            ),
        }
    )
    return stats


def _actions(replay: dict, candidate_count: int) -> dict:
    out = {}
    for policy in gl.BASE_POLICIES:
        utility_key = f"{policy}_utility"
        if utility_key not in replay:
            continue
        out[policy] = {
            "utility": float(replay[utility_key]),
            "negative": gl.safe_bool(replay.get(f"{policy}_negative")),
            "strict_negative": gl.safe_bool(replay.get(f"{policy}_strict_negative")),
            "selected_mass": float(replay.get(f"{policy}_selected_mass", 1.0)),
            "selected_count": float(replay.get(f"{policy}_selected_count", candidate_count)),
        }
    return out


def _agreement(score_stats: dict[str, dict]) -> dict:
    fields = sorted(score_stats)
    out = {}
    for i, left in enumerate(fields):
        for right in fields[i + 1 :]:
            ltop = score_stats[left]["top1_learner"]
            rtop = score_stats[right]["top1_learner"]
            out[f"top_agree:{left}:{right}"] = ltop == rtop
            spread_min = min(score_stats[left]["score_spread"], score_stats[right]["score_spread"])
            gap_min = min(score_stats[left]["top_gap_z"], score_stats[right]["top_gap_z"])
            out[f"spread_min:{left}:{right}"] = spread_min
            out[f"top_gap_z_min:{left}:{right}"] = gap_min
    return out


def build_rows(args) -> tuple[list[dict], dict]:
    score_fields = [s.strip() for s in args.score_fields.split(",") if s.strip()]
    groups = gl.load_groups(args.features, args.policy_replay, score_fields)
    rows = []
    per_seed = Counter()
    for group in groups:
        per_seed[int(group["seed"])] += 1
        candidates = group["candidates"]
        score_stats = {field: _field_stats(candidates, field) for field in score_fields}
        actions = _actions(group["replay"], group["candidate_count"])
        token = actions["token_weighted"]["utility"]
        oracle = actions.get("oracle_positive", {}).get("utility", float("nan"))
        row = {
            "schema": "group_local_features_v1",
            "seed": int(group["seed"]),
            "step": int(group["step"]),
            "fragment": int(group["fragment"]),
            "candidate_count": int(group["candidate_count"]),
            "scores": score_stats,
            "agreement": _agreement(score_stats),
            "actions": actions,
            "token_weighted_utility": token,
            "token_weighted_negative": actions["token_weighted"]["negative"],
            "token_weighted_strict_negative": actions["token_weighted"]["strict_negative"],
            "oracle_positive_utility": oracle,
            "oracle_positive_headroom": oracle - token if math.isfinite(oracle) else None,
        }
        if args.include_candidates:
            row["candidates"] = [
                {
                    "learner_id": int(c.get("learner_id", -1)),
                    "utility": float(c["utility"]),
                    "bad": gl.safe_bool(c.get("bad")),
                    "bad_strict": gl.safe_bool(c.get("bad_strict")),
                    "weight": gl.token_weight(c),
                    **{field: gl.score_value(c, field) for field in score_fields},
                }
                for c in candidates
            ]
        rows.append(row)

    summary = {
        "schema": "group_local_feature_summary_v1",
        "records": len(rows),
        "seeds": sorted(per_seed),
        "groups_per_seed": {str(k): v for k, v in sorted(per_seed.items())},
        "score_fields": score_fields,
        "score_diagnostics": gl.summarize_score_fields(groups, score_fields),
        "token_weighted_negative_rate": gl.mean(
            [1.0 if r["token_weighted_negative"] else 0.0 for r in rows]
        ),
        "token_weighted_strict_negative_rate": gl.mean(
            [1.0 if r["token_weighted_strict_negative"] else 0.0 for r in rows]
        ),
        "oracle_positive_headroom_mean": gl.mean(
            [float(r["oracle_positive_headroom"]) for r in rows if r["oracle_positive_headroom"] is not None]
        ),
    }
    return rows, summary


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return "n/a"
    if abs(v) < 0.001 and v != 0.0:
        return f"{v:.2e}"
    return f"{v:.{digits}f}"


def to_markdown(summary: dict) -> str:
    lines = ["# Group-Local Feature Summary", ""]
    lines.append(f"- Records: `{summary['records']}`")
    lines.append(f"- Seeds: `{summary['seeds']}`")
    lines.append(f"- Groups per seed: `{summary['groups_per_seed']}`")
    lines.append(
        f"- Token-weighted negative rate: `{_fmt(summary['token_weighted_negative_rate'])}`"
    )
    lines.append(
        f"- Oracle-positive headroom mean: `{_fmt(summary['oracle_positive_headroom_mean'], 6)}`"
    )
    lines.append("")
    lines.append("| Score | Candidate good AUROC | Pairwise concordance | Top1 bad | Bottom1 bad | Linear drop25 gain |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for field, stats in summary["score_diagnostics"].items():
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} |".format(
                field,
                _fmt(stats["candidate_good_auc"]),
                _fmt(stats["pairwise_concordance"]),
                _fmt(stats["top1_bad_rate"]),
                _fmt(stats["bottom1_bad_rate"]),
                _fmt(stats["linear_drop25_gain_mean"], 6),
            )
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", nargs="+", required=True, type=Path)
    p.add_argument("--policy-replay", nargs="+", required=True, type=Path)
    p.add_argument(
        "--score-fields",
        default="probe_grad_dot,probe_grad_cosine,calibrated_score,consensus_cosine,freshness,combined_score",
    )
    p.add_argument("--include-candidates", action="store_true")
    p.add_argument("--out-jsonl", required=True, type=Path)
    p.add_argument("--out-summary", required=True, type=Path)
    p.add_argument("--out-md", required=True, type=Path)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rows, summary = build_rows(args)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    gl.write_jsonl(args.out_jsonl, rows)
    args.out_summary.write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    args.out_md.write_text(to_markdown(summary))
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
