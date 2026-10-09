#!/usr/bin/env python3
"""S19 critic G3 judge (copy of s13-g3-judge.py; NOKILL=1 skips check 2 for runs without kill/resume). Judge S13 GPU G3 (rl-algo-critic-family 4.5) from a s13-g3-modal.sh run dir.

Evidence = the island tapes as echoed into <run dir>/launch.log (``YETO_RL_EVENT <json>``
lines; this keeps the killed container's pre-kill records, which a per-island tape
file of the relaunched container would overwrite), plus kill.json from the killer.
Hash口径: actor = ``rl_policy_apply`` ``sync/global_policy_hash`` (canonical policy
tensor hash of the committed average, checked equal to the trainer's export after
apply); critic = ``rl_critic_apply`` ``sync/global_critic_hash`` (canonical FP32
critic state from the critic syncer) and ``rl/critic/applied_weights_sha256`` (the
critic FP32 masters re-hashed after write-back; the import plugin refuses the apply
unless every bf16 model param equals its master cast, so a recorded apply implies
"bf16 regenerated from masters").

Checks:
 1 every committed version 1..N has an actor and a critic apply on BOTH islands, and
   for each version all records (both islands, every container generation) agree;
 2 kill happened (kill.json), island 1 has a post-kill container (a second
   rl_driver_start after the kill time), and its first post-kill applies are at the
   last version island 1 had committed before the kill, with the same actor and
   critic hashes as recorded for that version before the kill;
 3 training reached N on both islands (applies at v=N, rl_learner_finalized x2 after kill);
 4 GPU names reported by the containers contain the expected name; no write-back refusal text.
Missing data -> INCOMPLETE (never filled in).
usage: s13-g3-judge.py <run dir> [expected GPU, default H100] [rounds, default 3]
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

R = Path(sys.argv[1])
EXPECT_GPU = sys.argv[2] if len(sys.argv) > 2 else "H100"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 3
log = (R / "launch.log").read_text(errors="replace") if (R / "launch.log").is_file() else ""
events, seen = [], set()
for line in log.splitlines():
    at = line.find("YETO_RL_EVENT ")
    if at < 0:
        continue
    raw = line[at + 14:].strip()
    if raw in seen:
        continue
    seen.add(raw)
    try:
        events.append(json.loads(raw))
    except ValueError:
        pass
problems, missing, notes = [], [], []
kill = json.loads((R / "kill.json").read_text()) if (R / "kill.json").is_file() else None
kill_t = kill["kill_unix"] if kill else None


def applies(kind, field):
    out = defaultdict(list)  # version -> [(island, time, hash, post_kill)]
    for e in events:
        if e.get("event") == kind and isinstance(e.get("policy_version"), int):
            t = e.get("time_unix") or 0
            out[e["policy_version"]].append((e.get("island_id"), t, e.get(field),
                                             kill_t is not None and t > kill_t))
    return out


actor = applies("rl_policy_apply", "sync/global_policy_hash")
critic = applies("rl_critic_apply", "sync/global_critic_hash")
critic_w = applies("rl_critic_apply", "rl/critic/applied_weights_sha256")
table = {}
for v in range(0, N + 1):
    row = {}
    for label, data in (("actor", actor), ("critic", critic), ("critic_masters", critic_w)):
        recs = data.get(v, [])
        islands = {r[0] for r in recs}
        hashes = {r[2] for r in recs}
        row[label] = {"islands": sorted(i for i in islands if i is not None), "records": len(recs),
                      "hashes": sorted(map(str, hashes))}
        if v >= 1 and not {0, 1} <= islands:
            missing.append(f"v{v} {label}: applies only from islands {sorted(islands)}")
        if len(hashes) > 1:
            problems.append(f"v{v} {label}: hashes differ across records {sorted(map(str, hashes))}")
        if None in hashes and recs:
            missing.append(f"v{v} {label}: record without hash")
    table[v] = row

import os
NOKILL = os.environ.get("NOKILL") == "1"
resume = None
if NOKILL:
    notes.append("NOKILL: check 2 (kill/resume) not part of this run")
    if kill is not None:
        problems.append("kill.json present in a NOKILL run")
elif kill is None:
    missing.append("no kill.json (kill not performed)")
else:
    starts = [e for e in events if e.get("event") == "rl_driver_start" and e.get("island_id") == 1
              and (e.get("time_unix") or 0) > kill_t]
    if not starts:
        missing.append("island 1: no rl_driver_start after the kill (no relaunched container seen)")
    pre = [v for v, xs in actor.items() for x in xs if x[0] == 1 and not x[3]]
    last_committed = max(pre, default=None)
    post_a = sorted(((x[1], v, x[2]) for v, xs in actor.items() for x in xs if x[0] == 1 and x[3]))
    post_c = sorted(((x[1], v, x[2]) for v, xs in critic.items() for x in xs if x[0] == 1 and x[3]))
    resume = {"last_committed_before_kill": last_committed,
              "first_post_kill_actor_apply": post_a[0][1:] if post_a else None,
              "first_post_kill_critic_apply": post_c[0][1:] if post_c else None}
    if last_committed is None:
        missing.append("island 1 had no committed apply before the kill")
    elif not post_a or not post_c:
        missing.append("island 1: no actor/critic apply after the kill")
    else:
        pre_a = {x[2] for x in actor[last_committed] if not x[3]}
        pre_c = {x[2] for x in critic[last_committed] if not x[3]}
        if post_a[0][1] != last_committed or post_c[0][1] != last_committed:
            problems.append(f"resume at actor v{post_a[0][1]} / critic v{post_c[0][1]} != last committed "
                            f"v{last_committed}")
        elif {post_a[0][2]} != pre_a or {post_c[0][2]} != pre_c:
            problems.append("post-resume hashes differ from the last committed round's")
finals = [e for e in events if e.get("event") == "rl_learner_finalized"]
fin_islands = {e.get("island_id") for e in finals if kill_t is None or (e.get("time_unix") or 0) > kill_t}
if not {0, 1} <= fin_islands:
    missing.append(f"rl_learner_finalized after the kill only from islands {sorted(map(str, fin_islands))}")
gpu_lines = [l for l in log.splitlines() if re.search(r"\[modal-island \d+\] requested", l)]
if not gpu_lines or any(EXPECT_GPU not in l.split("got", 1)[-1] for l in gpu_lines):
    missing.append(f"GPU identity ({EXPECT_GPU}) not confirmed in every container line")
for pat in ("not regenerated from masters", "written critic hash", "CrossChannelCommitError",
            "[yeto-rl-strict-failure]"):
    if pat in log:
        problems.append(f"log contains {pat!r}")
rc = (R / "rc.txt").read_text().strip() if (R / "rc.txt").is_file() else None
verdict = "FAIL" if problems else ("INCOMPLETE" if missing else "PASS")
print(json.dumps({"verdict": verdict, "problems": problems, "missing": sorted(set(missing)), "notes": notes,
                  "rc": rc, "kill": kill, "resume": resume, "per_version": table,
                  "gpu_lines": gpu_lines[:8], "events": len(events)}, indent=1, sort_keys=True))
