#!/usr/bin/env python3
"""Judge S18 LPG 4.3 (launch-preflight-guards N12/N14 negative islands). Criteria are
pre-registered in openspec/changes/launch-preflight-guards/evidence/n12-n14-plan.md.

Evidence (never filled in; missing -> INCOMPLETE):
  launch.stream.log  local submit log ([preflight] lines of cmd_launch_head)
  launch.log         head job log (YETO_RL_EVENT lines, [preflight] lines of launcher.run, island errors)
  modal-app-logs.txt island container logs; head/yeto-syncer.log syncer log; tape-direct/** island tapes
usage: s18-lpg-judge.py <run dir> <legacy|elastic> <steps>
"""
import glob
import json
import re
import sys
from pathlib import Path

R = Path(sys.argv[1]); SCHED = sys.argv[2]; N = int(sys.argv[3])


def read(p):
    p = R / p
    return p.read_text(errors="replace") if p.is_file() else ""


stream, log, mlog, slog = read("launch.stream.log"), read("launch.log"), read("modal-app-logs.txt"), read("head/yeto-syncer.log")
events, seen = [], set()
for text in [log, mlog, stream] + [Path(f).read_text(errors="replace") for f in glob.glob(str(R / "tape-direct/**/*.jsonl*"), recursive=True)]:
    for line in text.splitlines():
        at = line.find("YETO_RL_EVENT ")
        raw = line[at + 14:].strip() if at >= 0 else line.strip()
        if not raw.startswith("{") or raw in seen:
            continue
        seen.add(raw)
        try:
            e = json.loads(raw)
        except ValueError:
            continue
        if isinstance(e, dict) and "event" in e:
            events.append(e)

out = {"run": R.name, "sched": SCHED, "steps": N, "criteria": {}, "facts": {}}
C, F = out["criteria"], out["facts"]
alltext = "\n".join((stream, log, mlog, slog))

# P1 preflight ran for real (local cmd_launch_head and head launcher.run)
# v2 (10-09): only the LOCAL part of the stream (before the head job was submitted);
# v1 also counted the head job's lines streamed back and required exactly 2 memory lines.
local = stream.split("submitted: job")[0]
thr = re.findall(r"\[preflight\] threads OK: (\d+) < (\d+)", local)
mem = re.findall(r"\[preflight\] island (\d) (\S+): estimated peak ([\d.]+) GiB / limit ([\d.]+) GiB \((\w+)\)", local)
F["preflight_threads_local"] = thr; F["preflight_memory_local"] = mem
F["preflight_threads_head"] = re.findall(r"\[preflight\] threads OK: (\d+) < (\d+)", log)
F["negative_banner"] = "NEGATIVE-TEST RUN" in stream
C["P1_preflight_ran_before_cloud"] = bool(thr) and int(thr[0][0]) < 2800 and len(mem) == 2 and F["negative_banner"]

# P2 override event at the head of island 1's tape; none on island 0
ov = [e for e in events if e.get("event") == "rl_island_override"]
F["override_events"] = [{"island": e.get("island_id"), "overrides": e.get("overrides")} for e in ov]
C["P2_override_event_island1_only"] = bool(ov) and {e.get("island_id") for e in ov} == {1}

# P3 island 1 refused, reason matches the parameter
if SCHED == "elastic":
    pat = r"backend identity mismatch, JOIN refused: island 1 declares"
else:
    pat = r"session mismatch \(HELLO refused, session keeps running\)"
refusals = sorted(set(m.group(0) for m in re.finditer(pat + r".{0,400}", alltext)))
F["refusal_lines"] = [r[:500] for r in refusals[:4]]
C["P3_island1_refused_with_both_hashes"] = bool(refusals) and all(
    len(re.findall(r"[0-9a-f]{64}", r)) >= 2 for r in refusals[:1])

# P4 only that connection: syncer kept running (no fatal exit before the run ended), island 0 saw no refusal
# v2 (10-09): look for a syncer exit only where the syncer and the launcher's syncer
# supervision write (syncer log, "[syncer]"/"syncer subprocess" lines); v1 also matched
# the refused island's own rl_strict_failure metric name "layout_hash_mismatch".
syncer_text = slog + "\n" + "\n".join(l for l in log.splitlines() if "[syncer]" in l or "syncer subprocess" in l)
fatal = re.findall(r"(?i)(syncer subprocess exited[^\n]{0,120}|panicked at[^\n]{0,120}|ERROR yeto_syncer[^\n]{0,160}|layout_hash_mismatch[^\n]{0,80})", syncer_text)
after = [l for l in slog.splitlines() if l[:27] > (re.search(r"^(\S+) .*(HELLO refused|JOIN refused)", slog, re.M).group(1) if re.search(r"^(\S+) .*(HELLO refused|JOIN refused)", slog, re.M) else "9")]
F["syncer_lines_after_refusal"] = len(after)
F["syncer_fatal_lines"] = [f[0] for f in fatal[:5]]
island0_refused = re.search(r"island 0 declares|learner_id=0[^\n]{0,80}rejected HELLO", alltext) is not None
F["island0_refused"] = island0_refused
C["P4_only_that_connection"] = not fatal and not island0_refused and bool(slog)

# P5 (elastic) island 0 finished every round and finalized; (legacy) by design the run stops
rounds0 = sorted({e.get("local_round_id") for e in events if e.get("event") == "rl_local_round" and e.get("island_id") == 0})
fin0 = any(e.get("event") == "rl_learner_finalized" and e.get("island_id") == 0 for e in events)
F["island0_local_rounds"] = rounds0; F["island0_finalized"] = fin0
F["island1_local_rounds"] = sorted({e.get("local_round_id") for e in events if e.get("event") == "rl_local_round" and e.get("island_id") == 1})
if SCHED == "elastic":
    C["P5_island0_completes_all_rounds"] = fin0 and len(rounds0) >= N
else:
    C["P5_legacy_n_a"] = "not applicable: fixed roster stops the run by design (N14 report)"

# P6 dashboard marks island 1
try:
    sys.path.insert(0, str(R / "yeto"))
    from yeto.dashboard.reducer import Reducer
    red = Reducer(run=R.name)
    for e in events:
        red.feed(e)
    cards = {c["id"]: c for c in red.overview(now=max([e.get("time_unix") or 0 for e in events] + [0]))["islands"]}
    F["dashboard_cards"] = {k: v.get("negative_test") for k, v in cards.items()}
    C["P6_dashboard_marks_island1"] = bool((cards.get("1") or {}).get("negative_test")) and not (cards.get("0") or {}).get("negative_test")
except Exception as exc:  # noqa: BLE001
    F["dashboard_error"] = repr(exc)
    C["P6_dashboard_marks_island1"] = False

# GPU assertion
F["gpu_names"] = sorted(set(re.findall(r"NVIDIA H\d+[^\n,\"]{0,20}", mlog)))[:4]
need = [k for k, v in C.items() if not k.startswith("P5_legacy")]
missing_evidence = [n for n, t in (("launch.stream.log", stream), ("launch.log", log), ("modal-app-logs.txt", mlog), ("head/yeto-syncer.log", slog)) if not t]
out["missing_evidence"] = missing_evidence
out["verdict"] = "PASS" if all(C[k] is True for k in need) else ("INCOMPLETE" if missing_evidence else "FAIL")
print(json.dumps(out, indent=1, ensure_ascii=False))
