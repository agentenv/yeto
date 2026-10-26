#!/usr/bin/env python3
"""Read-only health check of a P4 run (two verl full-parameter islands + the syncer).

Usage:  python -I tools/probes/s27_verl_full/health_p4.py RUN_DIR [--json OUT.json]

Reads RUN_DIR/raw/<run>/{island0,island1,syncer}/ as pulled by ``modal_p4.py``
(standard library only).  Each check is PASS / WARN / FAIL / N/A; the run is
healthy when no check FAILs.  Thresholds follow ``tools/eval/health_gr.py``
(s27-fixes c8757a83, basis GAD-REPORT.md section 4) for the quantities a verl
island reports; health_gr itself reads Miles log lines and does not apply.

Checks:
  exit        both islands and the syncer exited 0
  cards       every island container saw only NVIDIA H200
  merges      outer merges every island applied (sweeps whose fragments all
              moved), >= 3 PASS; and the syncer's outer steps whose responders
              were both islands
  final_ack   both islands acknowledged the final cut
  finite      no NaN/Inf in grad norm, policy loss, entropy, rewards
  lp_abs      trainer-vs-rollout mean |logprob diff| (verl criteria abs_diff):
              WARN > 0.10, FAIL > 0.20
  tis_cf      TIS clip fraction: WARN > 0.01, FAIL > 0.05
  gn_spike    grad norm max / island median: WARN > 15, FAIL > 50
  ogn_spike   syncer outer gnorm max / run median: WARN > 6, FAIL > 20
  publish     one publication per policy version per island (status recorded)
  reward      mean reward per round; WARN when every round is 0
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

OUTER = re.compile(r"outer step (.*)")
RESPONDERS = re.compile(r"responders=\[([^\]]*)\]")
GNORM = re.compile(r'gnorm="?([-+0-9.eE]+|NaN|nan|inf)"?')
STEP = re.compile(r"\bstep=(\d+)")
LEARNER = re.compile(r"learner_id: (\d+)|\b(\d+)\b")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def _meta(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def _finite(values) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def _grade(value, warn, fail) -> str:
    if value is None:
        return "N/A"
    return "FAIL" if value > fail else "WARN" if value > warn else "PASS"


def island_facts(root: Path, learner_id: int) -> dict:
    base = root / f"island{learner_id}"
    out = base / "yeto-output"
    tape = _jsonl(out / f"rl-island-{learner_id}.jsonl")
    metrics = [record.get("metrics", {}) for record in _jsonl(out / f"verl-metrics-{learner_id}.jsonl")]
    criteria = _jsonl(out / f"verl-criteria-{learner_id}.jsonl")
    publish = _jsonl(out / f"verl-publish-{learner_id}.jsonl")
    rounds = [event for event in tape if event.get("event") == "rl_full_parameter_round"]
    versions = [tuple(event.get("sync/fragment_versions") or ()) for event in rounds]
    merges, previous = 0, None
    for current in versions:
        if current and previous is not None and all(a > b for a, b in zip(current, previous, strict=True)):
            merges += 1
        elif current and previous is None and all(v > 0 for v in current):
            merges += 1
        previous = current

    def series(key):
        return [float(m[key]) for m in metrics if isinstance(m.get(key), (int, float))]

    return {
        "meta": _meta(base / "meta.json"),
        "rounds": len(rounds),
        "fragment_versions": [list(v) for v in versions],
        "merges": merges,
        "final_ack": [e for e in tape if e.get("event") == "rl_full_parameter_final_ack"],
        "done": [e for e in tape if e.get("event") == "rl_verl_island_done"],
        "grad_norm": series("actor/grad_norm"),
        "pg_loss": series("actor/pg_loss"),
        "entropy": series("actor/entropy"),
        "reward": series("critic/score/mean"),
        "abs_diff": [float(c["abs_diff"]) for c in criteria if isinstance(c.get("abs_diff"), (int, float))],
        "tis_clipfrac": [float(c["tis_clipfrac"]) for c in criteria
                         if isinstance(c.get("tis_clipfrac"), (int, float))],
        "publish_versions": [p.get("policy_version") for p in publish],
        "publish_status": sorted({str(p.get("status")) for p in publish}),
    }


def syncer_facts(root: Path, learners: int) -> dict:
    log = root / "syncer" / "syncer.log"
    steps, gnorms = {}, []
    for line in (log.read_text(errors="replace").splitlines() if log.is_file() else []):
        match = OUTER.search(ANSI.sub("", line))
        if not match:
            continue
        body = match.group(1)
        step = STEP.search(body)
        responders = RESPONDERS.search(body)
        gnorm = GNORM.search(body)
        ids = set()
        if responders:
            for a, b in LEARNER.findall(responders.group(1)):
                ids.add(int(a or b))
        if step:
            steps[int(step.group(1))] = ids
        if gnorm:
            try:
                gnorms.append(float(gnorm.group(1)))
            except ValueError:
                gnorms.append(float("nan"))
    full = sorted(step for step, ids in steps.items() if ids >= set(range(learners)))
    return {"meta": _meta(root / "syncer" / "meta.json"), "outer_steps": len(steps),
            "steps_with_every_island": len(full), "outer_gnorm": gnorms}


def check(run_dir: Path, learners: int = 2) -> dict:
    raws = [p for p in (run_dir / "raw").iterdir() if p.is_dir()] if (run_dir / "raw").is_dir() else []
    root = raws[0] if len(raws) == 1 else run_dir / "raw"
    islands = [island_facts(root, i) for i in range(learners)]
    syncer = syncer_facts(root, learners)
    rows = []

    def row(name, status, detail):
        rows.append({"check": name, "status": status, "detail": detail})

    rcs = [i["meta"].get("rc") for i in islands] + [syncer["meta"].get("rc")]
    row("exit", "PASS" if rcs == [0] * (learners + 1) else "FAIL", f"islands+syncer rc={rcs}")
    cards = [i["meta"].get("gpu") for i in islands]
    row("cards", "PASS" if all(c and set(c) == {"NVIDIA H200"} for c in cards) else "FAIL", f"{cards}")
    merges = [i["merges"] for i in islands]
    fragments = max((len(v) for i in islands for v in i["fragment_versions"]), default=0)
    sweeps_every = syncer["steps_with_every_island"] // fragments if fragments else 0
    row("merges", "PASS" if min(merges, default=0) >= 3 and sweeps_every >= 3 else "FAIL",
        f"applied per island {merges}; syncer outer steps {syncer['outer_steps']}, with every island "
        f"{syncer['steps_with_every_island']} (= {sweeps_every} sweeps of {fragments})")
    acks = [bool(i["final_ack"]) and all(e.get("acknowledged") for e in i["final_ack"]) for i in islands]
    row("final_ack", "PASS" if all(acks) else "FAIL", f"{acks}")
    values = [v for i in islands for k in ("grad_norm", "pg_loss", "entropy", "reward") for v in i[k]]
    row("finite", "PASS" if values and _finite(values) else ("N/A" if not values else "FAIL"),
        f"{len(values)} values")
    lp = max((v for i in islands for v in i["abs_diff"]), default=None)
    row("lp_abs", _grade(lp, 0.10, 0.20), f"max abs_diff {lp}")
    cf = max((v for i in islands for v in i["tis_clipfrac"]), default=None)
    row("tis_cf", _grade(cf, 0.01, 0.05), f"max tis_clipfrac {cf}")
    spikes = []
    for island in islands:
        gn = [g for g in island["grad_norm"] if math.isfinite(g)]
        if gn and statistics.median(gn) > 0:
            spikes.append(max(gn) / statistics.median(gn))
    row("gn_spike", _grade(max(spikes) if spikes else None, 15.0, 50.0), f"max/median per island {spikes}")
    ogn = [g for g in syncer["outer_gnorm"] if math.isfinite(g)]
    ratio = max(ogn) / statistics.median(ogn) if ogn and statistics.median(ogn) > 0 else None
    bad_ogn = len(syncer["outer_gnorm"]) - len(ogn)
    row("ogn_spike", "FAIL" if bad_ogn else _grade(ratio, 6.0, 20.0),
        f"max/median {ratio}; non-finite {bad_ogn}; n={len(syncer['outer_gnorm'])}")
    publish_ok = all(sorted(set(i["publish_versions"])) == list(range(len(set(i["publish_versions"]))))
                     and i["publish_versions"] for i in islands)
    row("publish", "PASS" if publish_ok else "FAIL",
        f"versions {[i['publish_versions'] for i in islands]}; status {[i['publish_status'] for i in islands]}")
    rewards = [v for i in islands for v in i["reward"]]
    row("reward", "N/A" if not rewards else "WARN" if not any(rewards) else "PASS",
        f"per island {[[round(v, 3) for v in i['reward']] for i in islands]}")
    counts = {status: sum(r["status"] == status for r in rows) for status in ("PASS", "WARN", "FAIL", "N/A")}
    return {"rows": rows, "counts": counts, "islands": islands, "syncer": {k: v for k, v in syncer.items()
                                                                            if k != "meta"}}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    result = check(Path(args.run_dir))
    for row in result["rows"]:
        print(f"{row['status']:5} {row['check']:10} {row['detail']}")
    print(" ".join(f"{k}={v}" for k, v in result["counts"].items()))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1, default=str))
    return 1 if result["counts"]["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
