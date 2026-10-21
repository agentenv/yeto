"""S26 E6: per-round metrics of Miles RL runs (rl_miles.sh) from the launcher logs.

For each run directory: reward mean, rollout/trainer mismatch (Miles ``mismatch``:
train_rollout_kl, tis_abs = mean |ratio - 1|, tis_clipfrac), rollout seconds and
generated tokens (colocated runs), round wall seconds. Then per group the
[min, max] over its runs of the last-3-round reward mean (the range criterion,
gpu-review/E6.md §4).

    python tools/probes/s26_e6/parse_miles.py /home/michael/work/s1-runs/s26-e6 > summary.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from statistics import mean

EVENT = re.compile(r"YETO_RL_EVENT (\{.*\})\s*$")


def events(log: Path, name: str) -> list[dict]:
    out = []
    if not log.is_file():
        return out
    for line in log.read_text(errors="replace").splitlines():
        m = EVENT.search(line)
        if m and f'"event":"{name}"' in m.group(1):
            try:
                out.append(json.loads(m.group(1)))
            except json.JSONDecodeError:
                pass
    return out


def run_rounds(run: Path) -> list[dict]:
    log = run / "launch.ts.log"
    trained = {e.get("rollout_id", i): e for i, e in enumerate(events(log, "rl_round_trained"))}
    local = {e["local_round_id"] - 1: e for e in events(log, "rl_local_round") if "local_round_id" in e}
    rows = []
    for rid in sorted(trained):
        t, lr = trained[rid], local.get(rid, {})
        mm = t.get("mismatch") or {}
        rows.append({
            "round": rid,
            "reward_mean": lr.get("reward_mean", t.get("reward_mean")),
            "train_rollout_kl": mm.get("train_rollout_kl"),
            "tis_abs": mm.get("tis_abs"),
            "tis_clipfrac": mm.get("tis_clipfrac"),
            "rollout_seconds": lr.get("rollout_seconds"),
            "action_tokens": lr.get("action_tokens"),
            "train_seconds": lr.get("train_seconds"),
            "resp_len_mean": t.get("resp_len_mean"),
        })
    return rows


def summary(rows: list[dict]) -> dict:
    def col(key):
        return [r[key] for r in rows if r.get(key) is not None]

    rewards = col("reward_mean")
    tok, sec = col("action_tokens"), col("rollout_seconds")
    return {
        "rounds": len(rows),
        "reward_last3_mean": mean(rewards[-3:]) if len(rewards) >= 3 else None,
        "reward_all": [round(x, 4) for x in rewards],
        "kl_mean": mean(col("train_rollout_kl")) if col("train_rollout_kl") else None,
        "tis_abs_mean": mean(col("tis_abs")) if col("tis_abs") else None,
        "tis_clipfrac_mean": mean(col("tis_clipfrac")) if col("tis_clipfrac") else None,
        "tis_clipfrac_max": max(col("tis_clipfrac")) if col("tis_clipfrac") else None,
        "rollout_tokens_per_s": (sum(tok) / sum(sec)) if tok and sec and len(tok) == len(sec) else None,
    }


def main(base: str) -> None:
    runs = {}
    for run in sorted(Path(base).glob("e6-*")):
        if run.is_dir() and "fail" not in run.name and "dry" not in run.name:
            rows = run_rounds(run)
            runs[run.name] = {"rc": (run / "rc").read_text().strip() if (run / "rc").is_file() else None,
                              "summary": summary(rows), "rounds": rows}
    groups: dict[str, list[float]] = {}
    for name, doc in runs.items():
        value = doc["summary"]["reward_last3_mean"]
        if value is not None:
            groups.setdefault(name.rsplit("-", 1)[0], []).append(value)
    print(json.dumps({"groups_last3_reward_range": {g: [min(v), max(v)] for g, v in groups.items()},
                      "runs": runs}, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1])
