#!/usr/bin/env python3
"""Read-only health check of a `yeto launch` run of verl full-parameter islands (S27 G-DN).

Usage:  python -I tools/probes/s27_verl_full/health_launch.py RUN_DIR [--json OUT.json]
                  [--learners 2] [--total-steps 32]

Reads what s1-runs/s27-dn/run-dn.sh pulls (standard library only):
  RUN_DIR/tape-direct/<app>/l<k>/rank0/   island ~/yeto-output mirror (tape, verl metrics,
                                          criteria, publications, modal-node.json)
  RUN_DIR/head/yeto-syncer.log            the head syncer's log
  RUN_DIR/head/yeto-output/yeto-tape.jsonl  the head syncer's event tape
  RUN_DIR/rc.txt                          the head job's result (rc=0: SUCCEEDED)

Each check is PASS / WARN / FAIL / N/A; the run is healthy when no check FAILs.
The quantity checks and thresholds are those of ``health_p4.py`` (from
``tools/eval/health_gr.py``).  Async-round checks (pre-registered in
SESSION27/VERL-SYNC.md, G-DN section):

  merges      the syncer committed every fragment step 1..total-steps (each with
              >= 1 responder and a finite gnorm), and every island applied the
              broadcasts in order (no fragment version ever went down)
  rejected    pushes the syncer rejected (stale or duplicate; allowed by the async
              profile): 0 PASS, more WARN
  final_ack   both islands acknowledged the final cut, and the syncer logged that
              every learner acknowledged it
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from health_p4 import ANSI, _finite, _grade, _jsonl  # noqa: E402

REJECTED = re.compile(r"push rejected|duplicate push rejected")
ALL_ACK = re.compile(r"all learners acknowledged final cut learners=(\d+) absent=(\d+)")


def _island_dir(run_dir: Path, learner_id: int) -> Path | None:
    hits = sorted(run_dir.glob(f"tape-direct/*/l{learner_id}/rank0"))
    return hits[0] if hits else None


def _latest(base: Path, name: str) -> Path:
    """``name`` of the island's last container: a relaunched island's k-th
    container mirrors its files as ``<stem>.inc<k><suffix>`` (modal_runner
    TapeSync); its tape also holds the earlier containers' records (S27 G-EC,
    s27-ec-a). Without relaunches this is ``base / name``."""
    stem, dot, suffix = name.rpartition(".")
    relaunched = sorted(base.glob(f"{stem}.inc*{dot}{suffix}"),
                        key=lambda p: int(re.search(r"\.inc(\d+)\.", p.name).group(1))
                        if re.search(r"\.inc(\d+)\.", p.name) else 0)
    return relaunched[-1] if relaunched else base / name


def island_facts(run_dir: Path, learner_id: int) -> dict:
    base = _island_dir(run_dir, learner_id)
    if base is None:
        return {"present": False}
    tape = _jsonl(_latest(base, f"rl-island-{learner_id}.jsonl"))
    metrics = [r.get("metrics", {}) for r in _jsonl(_latest(base, f"verl-metrics-{learner_id}.jsonl"))]
    criteria = _jsonl(_latest(base, f"verl-criteria-{learner_id}.jsonl"))
    publish = _jsonl(_latest(base, f"verl-publish-{learner_id}.jsonl"))
    node = {}
    if _latest(base, "modal-node.json").is_file():
        node = json.loads(_latest(base, "modal-node.json").read_text())
    rounds = [e for e in tape if e.get("event") == "rl_full_parameter_round"]
    versions = [tuple(e.get("sync/fragment_versions") or ()) for e in rounds]
    went_down = [i for i in range(1, len(versions))
                 if any(b < a for a, b in zip(versions[i - 1], versions[i], strict=True))]
    applied = sum(1 for i in range(1, len(versions)) if versions[i] != versions[i - 1])

    def series(key):
        return [float(m[key]) for m in metrics if isinstance(m.get(key), (int, float))]

    return {
        "present": True, "dir": str(base), "gpu_names": node.get("gpu_names"),
        "rounds": len(rounds), "first_versions": list(versions[0]) if versions else None,
        "last_versions": list(versions[-1]) if versions else None,
        "boundaries_with_new_versions": applied, "versions_went_down_at": went_down,
        "final_ack": [e for e in tape if e.get("event") == "rl_full_parameter_final_ack"],
        "done": [e for e in tape if e.get("event") == "rl_verl_island_done"],
        "grad_norm": series("actor/grad_norm"), "pg_loss": series("actor/pg_loss"),
        "entropy": series("actor/entropy"), "reward": series("critic/score/mean"),
        "abs_diff": [float(c["abs_diff"]) for c in criteria if isinstance(c.get("abs_diff"), (int, float))],
        "tis_clipfrac": [float(c["tis_clipfrac"]) for c in criteria
                         if isinstance(c.get("tis_clipfrac"), (int, float))],
        "publish_versions": [p.get("policy_version") for p in publish],
        "publish_status": sorted({str(p.get("status")) for p in publish}),
    }


def syncer_facts(run_dir: Path) -> dict:
    tape = [r for r in _jsonl(run_dir / "head/yeto-output/yeto-tape.jsonl") if "step" in r and "fragment" in r]
    log_path = run_dir / "head/yeto-syncer.log"
    log = ANSI.sub("", log_path.read_text(errors="replace")) if log_path.is_file() else ""
    steps = {int(r["step"]): r for r in tape}
    acks = [tuple(int(x) for x in m) for m in ALL_ACK.findall(log)]
    # A late push of the async profile is carried into a later round of the same
    # fragment (``carried``); it contributes there.  One queued and never carried
    # missed the end of the run.
    carried = {s: [int(c["id"]) for c in (r.get("carried") or [])] for s, r in steps.items()}
    return {
        "steps": sorted(steps), "responders": {s: list(r.get("responded") or []) for s, r in steps.items()},
        "contributors": {s: sorted(set(r.get("responded") or []) | set(carried[s])) for s, r in steps.items()},
        "carried_pushes": sum(len(v) for v in carried.values()),
        "late_queued": log.count("late push queued"),
        "stale_admitted": log.count("stale base-relative delta admitted"),
        "gnorm": [float(r["gnorm"]) if isinstance(r.get("gnorm"), (int, float)) else float("nan")
                  for _s, r in sorted(steps.items())],
        "rejected_lines": len(REJECTED.findall(log)), "all_ack": acks, "log_bytes": len(log),
    }


def check(run_dir: Path, learners: int = 2, total_steps: int = 32) -> dict:
    islands = [island_facts(run_dir, i) for i in range(learners)]
    syncer = syncer_facts(run_dir)
    rows = []

    def row(name, status, detail):
        rows.append({"check": name, "status": status, "detail": detail})

    rc = (run_dir / "rc.txt").read_text().strip() if (run_dir / "rc.txt").is_file() else "missing"
    done = [bool(i.get("done")) for i in islands]
    row("exit", "PASS" if rc == "rc=0" and all(done) else "FAIL", f"head job {rc}; island done events {done}")
    cards = [i.get("gpu_names") for i in islands]
    row("cards", "PASS" if all(c and set(c) == {"NVIDIA H200"} for c in cards) else "FAIL", f"{cards}")
    expected = list(range(1, total_steps + 1))
    every = [s for s, ids in syncer["responders"].items() if set(ids) >= set(range(learners))]
    every_merged = [s for s, ids in syncer["contributors"].items() if set(ids) >= set(range(learners))]
    empty = [s for s, ids in syncer["responders"].items() if not ids]
    gn_ok = _finite(syncer["gnorm"])
    down = [i.get("versions_went_down_at") for i in islands]
    merges_ok = (syncer["steps"] == expected and not empty and gn_ok and all(i.get("present") for i in islands)
                 and not any(down))
    row("merges", "PASS" if merges_ok else "FAIL",
        f"syncer steps {len(syncer['steps'])} (expected 1..{total_steps}: {syncer['steps'] == expected}), "
        f"with every island in time {len(every)}, with every island incl. carried late pushes "
        f"{len(every_merged)}; late pushes queued {syncer['late_queued']}, carried {syncer['carried_pushes']}, "
        f"stale admitted {syncer['stale_admitted']}; without responder {empty}, gnorm finite {gn_ok}; islands' last "
        f"versions {[i.get('last_versions') for i in islands]}, versions went down at {down}")
    row("rejected", "PASS" if syncer["rejected_lines"] == 0 else "WARN",
        f"{syncer['rejected_lines']} 'push rejected' lines in the syncer log")
    acks = [bool(i.get("final_ack")) and all(e.get("acknowledged") for e in i["final_ack"]) for i in islands]
    all_ack = (learners, 0) in syncer["all_ack"]
    row("final_ack", "PASS" if all(acks) and all_ack else "FAIL", f"islands {acks}; syncer {syncer['all_ack']}")
    values = [v for i in islands for k in ("grad_norm", "pg_loss", "entropy", "reward") for v in i.get(k, [])]
    row("finite", "PASS" if values and _finite(values) else ("N/A" if not values else "FAIL"), f"{len(values)} values")
    lp = max((v for i in islands for v in i.get("abs_diff", [])), default=None)
    row("lp_abs", _grade(lp, 0.10, 0.20), f"max abs_diff {lp}")
    cf = max((v for i in islands for v in i.get("tis_clipfrac", [])), default=None)
    row("tis_cf", _grade(cf, 0.01, 0.05), f"max tis_clipfrac {cf}")
    spikes = []
    for island in islands:
        gn = [g for g in island.get("grad_norm", []) if math.isfinite(g)]
        if gn and statistics.median(gn) > 0:
            spikes.append(max(gn) / statistics.median(gn))
    row("gn_spike", _grade(max(spikes) if spikes else None, 15.0, 50.0), f"max/median per island {spikes}")
    ogn = [g for g in syncer["gnorm"] if math.isfinite(g)]
    ratio = max(ogn) / statistics.median(ogn) if ogn and statistics.median(ogn) > 0 else None
    bad = len(syncer["gnorm"]) - len(ogn)
    row("ogn_spike", "FAIL" if bad else _grade(ratio, 6.0, 20.0),
        f"max/median {ratio}; non-finite {bad}; n={len(syncer['gnorm'])}")
    publish_ok = all(i.get("publish_versions") and sorted(set(i["publish_versions"]))
                     == list(range(len(set(i["publish_versions"])))) for i in islands)
    row("publish", "PASS" if publish_ok else "FAIL",
        f"versions {[i.get('publish_versions') for i in islands]}; status {[i.get('publish_status') for i in islands]}")
    rewards = [v for i in islands for v in i.get("reward", [])]
    row("reward", "N/A" if not rewards else "WARN" if not any(rewards) else "PASS",
        f"per island {[[round(v, 3) for v in i.get('reward', [])] for i in islands]}")
    counts = {s: sum(r["status"] == s for r in rows) for s in ("PASS", "WARN", "FAIL", "N/A")}
    return {"rows": rows, "counts": counts, "islands": islands,
            "syncer": {k: v for k, v in syncer.items() if k not in ("responders", "contributors")}}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--json", default=None)
    parser.add_argument("--learners", type=int, default=2)
    parser.add_argument("--total-steps", type=int, default=32)
    args = parser.parse_args(argv)
    result = check(Path(args.run_dir), args.learners, args.total_steps)
    for r in result["rows"]:
        print(f"{r['status']:5} {r['check']:10} {r['detail']}")
    print(" ".join(f"{k}={v}" for k, v in result["counts"].items()))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1, default=str))
    return 1 if result["counts"]["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
