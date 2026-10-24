"""S26 E0.3: per-round metrics of the bias-stacking ablation runs (run.sh).

Reads YETO_RL_EVENT lines (rl_local_round, rl_round_trained) from a run's launcher log
(single island: launch.ts.log; multi-island: launch.log pulled from the head) and prints one
JSON summary: per layer and run, per round reward (mean over islands, each island has the
same sample count), samples, wall seconds since the first round started, rollout/trainer
mismatch (train_rollout_kl, logprob |diff|, TIS mean ratio, mean |ratio - 1|, clip fraction),
truncation fraction and generation tokens/s; then per layer the [min, max] over runs of the
last-3-round reward mean (range criterion, gpu-review/E0.3.md §3).

    python tools/probes/s26_e03/parse.py /home/michael/work/s1-runs/s26-e03/s26-e03-L*-? > summary.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from statistics import mean

EVENT = re.compile(r"YETO_RL_EVENT (\{.*\})\s*$")
LOGS = ("launch.ts.log", "final-launch.log", "launch.log")


def events(run: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {"rl_local_round": [], "rl_round_trained": []}
    seen = set()
    for name in LOGS:
        log = run / name
        if not log.is_file():
            continue
        for line in log.read_text(errors="replace").splitlines():
            m = EVENT.search(line)
            if not m:
                continue
            try:
                e = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            kind = e.get("event")
            if kind not in out:
                continue
            key = (kind, e.get("island_id"), e.get("local_round_id", e.get("rollout_id")))
            if key in seen:  # the same event in two log copies
                continue
            seen.add(key)
            out[kind].append(e)
    return out


def _avg(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return mean(values) if values else None


def run_rounds(run: Path) -> list[dict]:
    ev = events(run)
    local = {}
    for e in ev["rl_local_round"]:
        local.setdefault(e["local_round_id"] - 1, []).append(e)
    trained = {}
    for e in ev["rl_round_trained"]:
        trained.setdefault(e.get("rollout_id"), []).append(e)
    if not local:
        return []
    first = local[min(local)]
    t0 = min(e["time_unix"] - e.get("rollout_seconds", 0) - e.get("train_seconds", 0) for e in first)
    rounds = []
    for r in sorted(local):
        ls, ts = local[r], trained.get(r, [])
        mm = [t.get("mismatch") or {} for t in ts]
        tm = [t.get("train_metrics") or {} for t in ts]
        tokens = sum(e.get("action_tokens") or 0 for e in ls)
        rollout_s = _avg([e.get("rollout_seconds") for e in ls])
        rounds.append({
            "round": r,
            "islands": len(ls),
            "reward": _avg([e.get("reward_mean") for e in ls]),
            "samples_cum": None,
            "samples": sum(e.get("completed_trajectories") or 0 for e in ls),
            "wall_s": max(e["time_unix"] for e in ls) - t0,
            "train_rollout_kl": _avg([m.get("train_rollout_kl") for m in mm]),
            "logprob_abs_diff": _avg([m.get("train_rollout_logprob_abs_diff") for m in tm]),
            "tis_mean": _avg([m.get("tis") for m in mm]),
            "tis_abs": _avg([m.get("tis_abs") for m in mm]),
            "tis_clipfrac": _avg([m.get("tis_clipfrac") for m in mm]),
            "truncated_frac": _avg([t.get("truncated_frac") for t in ts]),
            "gen_tok_per_s": (tokens / len(ls) / rollout_s) if rollout_s else None,
            "rollout_s": rollout_s,
            "train_s": _avg([e.get("train_seconds") for e in ls]),
            "resp_len_mean": _avg([t.get("resp_len_mean") for t in ts]),
        })
    total = 0
    for r in rounds:
        total += r["samples"]
        r["samples_cum"] = total
    return rounds


def layer_of(run: Path) -> str:
    m = re.match(r"s26-e03-(?:(v\d+\w*)-)?(L\w+?)-", run.name)
    return "-".join(g for g in m.groups() if g) if m else run.name


def summarize(runs: list[Path]) -> dict:
    out = {"runs": {}, "layers": {}}
    for run in runs:
        rounds = run_rounds(run)
        if not rounds:
            continue
        last3 = _avg([r["reward"] for r in rounds[-3:]])
        steady = rounds[1:]
        out["runs"][run.name] = {
            "layer": layer_of(run),
            "rounds_done": len(rounds),
            "last3_reward": last3,
            "mean_reward": _avg([r["reward"] for r in rounds]),
            "wall_s": rounds[-1]["wall_s"],
            "steady_round_s": _avg([b["wall_s"] - a["wall_s"] for a, b in zip(rounds, rounds[1:])]),
            "kl_mean": _avg([r["train_rollout_kl"] for r in steady]),
            "logprob_abs_diff_mean": _avg([r["logprob_abs_diff"] for r in steady]),
            "tis_abs_mean": _avg([r["tis_abs"] for r in steady]),
            "tis_abs_max": max((r["tis_abs"] or 0) for r in rounds),
            "tis_clipfrac_mean": _avg([r["tis_clipfrac"] for r in steady]),
            "tis_clipfrac_max": max((r["tis_clipfrac"] or 0) for r in rounds),
            "truncated_frac_mean": _avg([r["truncated_frac"] for r in rounds]),
            "gen_tok_per_s_mean": _avg([r["gen_tok_per_s"] for r in steady]),
            "rounds": rounds,
        }
    for name, s in out["runs"].items():
        layer = out["layers"].setdefault(s["layer"], {"runs": [], "last3": []})
        layer["runs"].append(name)
        if s["rounds_done"] >= 10:
            layer["last3"].append(s["last3_reward"])
    for layer in out["layers"].values():
        v = layer["last3"]
        layer["range"] = [min(v), max(v)] if v else None
    return out


if __name__ == "__main__":
    print(json.dumps(summarize([Path(p) for p in sys.argv[1:]]), indent=1))
