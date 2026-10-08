#!/usr/bin/env python3
"""Judge fix-decoupled-lr-schedule 3.2 (head-mode two-island decoupled rerun) from a s14-dlr-remote.sh run dir.

Evidence: island events (``YETO_RL_EVENT`` lines in launch.log, ports only) merged with the island tape
files mirrored to the Modal tape volume (tape-direct/**/*.jsonl, both engines), the syncer tape on the head
(head/yeto-output/yeto-tape.jsonl: one record per outer step with sync/global_delta_norm), and launch.log text.

Checks (3.2 + the regression scenario the 2026-09-29 benchmark could not reproduce):
 1 each island trained MORE than global_rounds x optimizer_steps (= 4) local rounds (run-until-stop; expected 12),
   every rl_local_round carries applied_lrs, and every applied lr == --inner-lr exactly (1e-05);
 2 every syncer outer step 1..total_steps*fragments (=16) has sync/global_delta_norm != 0 (the pre-fix run had
   steps 7..16 at 0);
 3 both islands' last rl_policy_apply are at the same policy_version with the same sync/global_policy_hash;
 4 rl_learner_finalized from both islands, no zero-lr invariant failure text, rc=0, containers report EXPECT_GPU.
Optional: --compare <other run dir> checks per-island applied_lrs sequences are repr-identical (legacy vs ports).
Missing data -> INCOMPLETE (never filled in).
usage: s14-dlr-judge.py <run dir> [expected GPU=H100] [--compare <run dir>]
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

INNER_LR = 1e-05
GLOBAL_ROUNDS, OPT_STEPS, TOTAL_STEPS, FRAGMENTS = 4, 1, 4, 4


def load_events(R: Path):
    events, seen = [], set()
    srcs = [R / "launch.log"] + [Path(p) for p in sorted(glob.glob(str(R / "tape-direct" / "**" / "*.jsonl"), recursive=True))]
    for p in srcs:
        if not p.is_file():
            continue
        for line in p.read_text(errors="replace").splitlines():
            at = line.find("YETO_RL_EVENT ") if p.name == "launch.log" else (0 if line.startswith("{") else -1)
            if at < 0:
                continue
            raw = line[at + 14:].strip() if p.name == "launch.log" else line.strip()
            if raw in seen:
                continue
            seen.add(raw)
            try:
                events.append(json.loads(raw))
            except ValueError:
                pass
    return events


def island_view(R: Path):
    events = load_events(R)
    rounds, applies, finalized = defaultdict(dict), defaultdict(list), set()
    for e in events:
        i = e.get("island_id")
        if e.get("event") == "rl_local_round" and isinstance(e.get("local_round_id"), int):
            rounds[i][e["local_round_id"]] = e
        elif e.get("event") == "rl_policy_apply" and isinstance(e.get("policy_version"), int):
            applies[i].append((e["policy_version"], e.get("time_unix") or 0, e.get("sync/global_policy_hash")))
        elif e.get("event") == "rl_learner_finalized":
            finalized.add(i)
    view = {}
    for i in sorted(set(rounds) | set(applies)):
        ids = sorted(rounds[i])
        view[i] = {
            "round_ids": ids,
            "applied_lrs": [rounds[i][k].get("applied_lrs") for k in ids],
            "applied_lr": [rounds[i][k].get("applied_lr") for k in ids],
            "delta_l2_norm": [rounds[i][k].get("delta_l2_norm") for k in ids],
            "last_apply": max(applies[i], key=lambda a: (a[0], a[1])) if applies[i] else None,
            "finalized": i in finalized,
        }
    return view, len(events)


def syncer_steps(R: Path):
    out = {}
    p = R / "head" / "yeto-output" / "yeto-tape.jsonl"
    if p.is_file():
        for line in p.read_text(errors="replace").splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if isinstance(e.get("step"), int):
                out[e["step"]] = e.get("sync/global_delta_norm")
    return out


def main():
    argv = sys.argv[1:]
    cmp_dir = None
    if "--compare" in argv:
        k = argv.index("--compare")
        cmp_dir = Path(argv[k + 1])
        argv = argv[:k] + argv[k + 2:]
    R = Path(argv[0])
    expect_gpu = argv[1] if len(argv) > 1 else "H100"
    log = (R / "launch.log").read_text(errors="replace") if (R / "launch.log").is_file() else ""
    view, n_events = island_view(R)
    steps = syncer_steps(R)
    problems, missing, notes = [], [], []
    horizon = GLOBAL_ROUNDS * OPT_STEPS
    for i in (0, 1):
        v = view.get(i)
        if v is None:
            missing.append(f"island {i}: no events")
            continue
        n = len(v["round_ids"])
        if n == 0:
            missing.append(f"island {i}: no rl_local_round")
            continue
        if v["round_ids"] != list(range(1, n + 1)):
            problems.append(f"island {i}: round ids not contiguous {v['round_ids']}")
        if n <= horizon:
            missing.append(f"island {i}: only {n} local rounds, not beyond global_rounds*optimizer_steps={horizon}")
        else:
            notes.append(f"island {i}: {n} local rounds (> {horizon}: run-until-stop scenario reproduced)")
        for k, lrs in zip(v["round_ids"], v["applied_lrs"]):
            if not lrs:
                missing.append(f"island {i} round {k}: applied_lrs absent")
            elif any(x != INNER_LR for x in lrs):
                problems.append(f"island {i} round {k}: applied_lrs {lrs!r} != {INNER_LR!r}")
        if not v["finalized"]:
            missing.append(f"island {i}: no rl_learner_finalized")
        if v["last_apply"] is None:
            missing.append(f"island {i}: no rl_policy_apply")
    if all(view.get(i, {}).get("last_apply") for i in (0, 1)):
        a0, a1 = view[0]["last_apply"], view[1]["last_apply"]
        if a0[0] != a1[0] or a0[2] != a1[2]:
            problems.append(f"final apply differs: island0 v{a0[0]} {a0[2]} vs island1 v{a1[0]} {a1[2]}")
        else:
            notes.append(f"final hash equal at v{a0[0]}: {a0[2]}")
    want = TOTAL_STEPS * FRAGMENTS
    if not steps:
        missing.append("syncer tape (head/yeto-output/yeto-tape.jsonl) absent or empty")
    else:
        for s in range(1, want + 1):
            if s not in steps:
                missing.append(f"syncer outer step {s} missing")
            elif steps[s] in (None, 0, 0.0):
                problems.append(f"syncer outer step {s}: global_delta_norm {steps[s]!r}")
    for pat in ("applied learning rate", "zero learning rate", "lr_zero", "ZeroLearningRate"):
        if pat.lower() in log.lower():
            problems.append(f"launch.log mentions {pat!r} (zero-lr invariant?)")
    if "Traceback" in log:
        notes.append(f"launch.log has {log.count('Traceback')} Traceback(s)")
    gpu = [l for l in log.splitlines() if "requested H100" in l or "got ['NVIDIA" in l]
    if len([l for l in gpu if expect_gpu in l]) < 2:
        missing.append(f"fewer than 2 container GPU confirmations containing {expect_gpu!r}")
    rc = (R / "rc.txt").read_text().strip() if (R / "rc.txt").is_file() else None
    if rc != "rc=0":
        problems.append(f"rc: {rc}")
    compare = None
    if cmp_dir is not None:
        other, _ = island_view(cmp_dir)
        compare = {}
        for i in (0, 1):
            a = [repr(x) for x in view.get(i, {}).get("applied_lrs", [])]
            b = [repr(x) for x in other.get(i, {}).get("applied_lrs", [])]
            compare[i] = {"equal": a == b and bool(a), "this": a, "other": b}
            if not compare[i]["equal"]:
                problems.append(f"island {i}: applied_lrs differ from {cmp_dir.name}")
    verdict = "INCOMPLETE" if missing else ("FAIL" if problems else "PASS")
    print(json.dumps({
        "verdict": verdict, "run": R.name, "events": n_events, "rc": rc,
        "islands": {str(i): {k: v[k] for k in ("round_ids", "applied_lrs", "applied_lr", "delta_l2_norm", "finalized")} | {"last_apply": v["last_apply"]} for i, v in view.items()},
        "syncer_global_delta_norm": {str(k): steps[k] for k in sorted(steps)},
        "gpu_lines": gpu, "problems": problems, "missing": missing, "notes": notes, "compare": compare,
    }, indent=1))
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
