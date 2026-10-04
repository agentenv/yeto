# in-container: watch the island journal; at the N-th terminal phase run the fork probes locally (seconds, no ssh/pull
# latency): status -> ~/yeto-rl/probe_after.txt; with mode e1b also stale <first member_engines url> -> probe_stale.txt
# and oldepoch -> probe_oldepoch.txt.  usage: term_probe.py <n_terminal> [e1b|rec]
# mode rec (E1-D ⑤⑦, 3.7 restart recovery): count `recovery status=verified` records instead of terminal phases.
import json, os, subprocess, sys, time
N = int(sys.argv[1]); MODE = sys.argv[2] if len(sys.argv) > 2 else ""
J = os.path.expanduser("~/yeto-rl/elastic-state/reconfig/journal.jsonl"); H = os.path.expanduser("~/yeto-rl")
log = open(os.path.join(H, "term_probe.log"), "a")
def L(**k): k["t"] = time.time(); log.write(json.dumps(k) + "\n"); log.flush()
def probe(out, *args):
    with open(os.path.join(H, out), "w") as f:
        rc = subprocess.call(["bash", os.path.join(H, "probe_remote.sh"), *args], stdout=f, stderr=subprocess.STDOUT)
    L(event="probe", out=out, args=list(args), rc=rc)
pos = 0; term = 0; url = ""; done = False
while not done:
    if os.path.exists(J):
        with open(J) as f:
            f.seek(pos); data = f.read(); pos = f.tell()
        for l in data.splitlines():
            try: r = json.loads(l)
            except Exception: continue
            if r.get("kind") == "member_engines" and r.get("engine_urls") and not url: url = sorted(r["engine_urls"])[0]
            hit = (r.get("kind") == "recovery" and r.get("status") == "verified") if MODE == "rec" else (
                r.get("kind") == "phase" and r.get("phase") in ("SUCCEEDED", "REBUILT_OLD", "CANCELLED", "RECOVERY_REQUIRED"))
            if hit:
                term += 1; L(event="terminal", n=term, phase=r.get("phase") or r.get("status"))
                if term >= N:
                    # all probes in parallel: each needs ~20 s to start a Ray client and the run may end ~50 s after the terminal phase
                    import threading
                    jobs = [("probe_after.txt", ("status",))]
                    if MODE == "e1b":
                        if url: jobs.append(("probe_stale.txt", ("stale", url)))
                        jobs.append(("probe_oldepoch.txt", ("oldepoch",)))
                    ts = [threading.Thread(target=probe, args=(o, *a)) for o, a in jobs]
                    for t in ts: t.start()
                    for t in ts: t.join()
                    L(event="done"); done = True; break
    time.sleep(1)
