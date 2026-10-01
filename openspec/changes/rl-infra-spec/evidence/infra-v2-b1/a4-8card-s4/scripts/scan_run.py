#!/usr/bin/env python3
"""scan_run.py <run dir> [--out scan]  -> <run>/scan.json + <run>/scan.md
1. stage timings (launch.ts.log markers, tape rl_driver_*, journal phases)  2. anomaly scan of every text log (WARNING/ERROR/Traceback/timeout/retry/OOM/Xid/...),
grouped by normalised template with source file, count, first/last line and a heuristic class (product|test|platform|upstream|?) that the human reviewer must confirm."""
import json, re, sys, glob, os, collections, base64, io, tarfile
from datetime import datetime, timezone

R = sys.argv[1].rstrip("/")
ANSI = re.compile(r"\x1b\[[0-9;]*m|\[\d+(;\d+)*m")
def ts(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
def rd(p):
    try: return open(p, errors="replace").read()
    except Exception: return ""

# ---------------- timings ----------------
T = {}
t0 = rd(R + "/start_utc.txt").strip(); t1 = rd(R + "/end_utc.txt").strip()
T["launch_start_utc"] = t0; T["launch_end_utc"] = t1
if t0 and t1: T["launch_wall_s"] = round(ts(t1) - ts(t0), 1)
marks = [("sky_launching", r"Launching on Nebius"), ("sky_cluster_launched", r"Cluster launched"), ("job_submitted", r"Job submitted"),
         ("first_island_log_line", r"\(yeto-rl-island"), ("first_YETO_RL_EVENT", r"YETO_RL_EVENT"), ("job_finished", r"Job finished"),
         ("teardown_start", r"tearing down"), ("run_finished", r"\[yeto\] run .* finished")]
lines = []
for l in rd(R + "/launch.ts.log").splitlines():
    m = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) (.*)", l)
    if m: lines.append((ts(m.group(1)), ANSI.sub("", m.group(2))))
first = {}
for t, l in lines:
    for k, rx in marks:
        if k not in first and re.search(rx, l): first[k] = t
if t0:
    T["marks_s_since_launch_start"] = {k: round(v - ts(t0), 1) for k, v in first.items()}

def load_jsonl(p):
    out = []
    for l in rd(p).splitlines():
        try: out.append(json.loads(l))
        except Exception: pass
    return out
tape = []
for f in sorted(glob.glob(R + "/runs/*/events/*.jsonl")): tape += load_jsonl(f)
tape_src = "runs/*/events"
if not tape:
    tape = load_jsonl(R + "/pulled/rl-island-0.final.jsonl") or load_jsonl(R + "/pulled/rl-island-0.jsonl"); tape_src = "pulled tape"
T["tape_source"] = tape_src; T["tape_events"] = len(tape)
ds = [e for e in tape if e.get("event") == "rl_driver_start"]
ph = [e for e in tape if e.get("event") == "rl_driver_phase" and "time_unix" in e]
if t0 and ph:
    T["first_driver_phase_s_since_launch_start"] = round(min(e["time_unix"] for e in ph) - ts(t0), 1)
gens = [e for e in ph if e.get("phase") == "generate"]
if gens:
    g = sorted(gens, key=lambda e: e["time_unix"])
    T["generate_start_times_s_rel_first"] = [round(e["time_unix"] - g[0]["time_unix"], 1) for e in g]
    T["round_interval_s"] = [round(b["time_unix"] - a["time_unix"], 1) for a, b in zip(g, g[1:])]
    if t0: T["first_generate_s_since_launch_start"] = round(g[0]["time_unix"] - ts(t0), 1)
# journal
jr = []
for cand in (R + "/.j/elastic-state/reconfig/journal.jsonl",):
    jr = load_jsonl(cand)
if not jr:
    for b in (R + "/pulled/elastic-state-final.b64", R + "/pulled/elastic-state.tgz.b64"):
        try:
            raw = base64.b64decode(rd(b).strip() or "")
            tf = tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz")
            m = [x for x in tf.getmembers() if x.name.endswith("reconfig/journal.jsonl")]
            if m:
                jr = [json.loads(l) for l in tf.extractfile(m[0]).read().decode().splitlines() if l.strip().startswith("{")]
                break
        except Exception:
            pass
T["journal_records"] = len(jr)
tx = collections.OrderedDict()
for r in jr:
    t = r.get("tx_id")
    if not t: continue
    d = tx.setdefault(t, {"phases": [], "fork_ops": []})
    w = r.get("wall_time")
    if r.get("kind") == "phase": d["phases"].append((r.get("phase"), w))
    if r.get("kind") == "fork_op": d["fork_ops"].append((r.get("op"), r.get("status"), w))
T["transactions"] = {}
for t, d in tx.items():
    ph_ = [(p, w) for p, w in d["phases"] if w]
    ent = {"phases_s_rel": {}, "total_s": None, "fork_ops": []}
    if ph_:
        b = ph_[0][1]
        for p, w in ph_: ent["phases_s_rel"].setdefault(p, round(w - b, 1))
        ent["total_s"] = round(ph_[-1][1] - b, 1); ent["terminal"] = ph_[-1][0]
    for op, st, w in d["fork_ops"]:
        ent["fork_ops"].append((op, st, round(w - ph_[0][1], 1) if (w and ph_) else None))
    # start_cells duration = fork_op start issued -> next phase after it
    iss = [w for op, st, w in d["fork_ops"] if op == "start" and st == "issued" and w]
    ver = [w for p, w in ph_ if p == "VERIFYING"]
    if iss and ver: ent["start_cells_s"] = round(ver[0] - iss[0], 1)
    T["transactions"][t] = ent

# ---------------- anomaly scan ----------------
SRC = ["launch.log", "n2run.out", "selfcheck.txt", "cleanup.out", "puller.log", "final_stop.txt", "nstop.out", "judge.out",
       "pulled/inwatch.final.log", "pulled/dkill.log", "pulled/dctl.log", "pulled/sampler.out", "pulled/diag/diag.err", "pulled/diag/dmesg.txt",
       "pulled/diag/stuck.txt", "autostop.out", "watchdog.out", "inwatch-arm.txt", "progress_stop.txt", "probe_after.txt", "selfcheck_probe.txt"]
extra = glob.glob(R + "/pulled/diag/pyspy-*.txt") + glob.glob(R + "/home/sky_logs/**/*.log", recursive=True) + glob.glob(R + "/arm-*.txt") + glob.glob(R + "/../" + os.path.basename(R) + ".n2run.out")
files = [R + "/" + s for s in SRC] + extra
PAT = re.compile(r"WARNING|\bWARN\b|\bERROR\b|Traceback|Exception|FATAL|Fatal|Error\b|error:|timed out|timeout|Timeout|retry|retrying|OOM|out of memory|Xid|NVRM|segfault|SIGKILL|SIGQUIT|crashed|did not exit|failed|Failed|unavailable|refused|Killed|denied", re.I)
# lines that are routine, not anomalies
SKIP = re.compile(r"^\s*$|Permanently added|INFO:|\"event\"|retry_timeout|--modal-retries|no-island-relaunch|YETO_RL_EVENT|\s\.{5,}\s|server_args=|\[WeightChecker\]|process_trampoline|Streaming logs|calc_ft")
def norm(l):
    l = ANSI.sub("", l)
    l = re.sub(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ ", "", l)
    l = re.sub(r"^\[[\w.-]+-l\d+-eu-north1\] ", "", l)
    l = re.sub(r"\(yeto-rl-island-\d+, pid=\d+\)", "(island)", l)
    l = re.sub(r"\((\w+) pid=\d+\)", r"(\1)", l)
    l = re.sub(r"\[\d{4}-\d\d-\d\d[ T][\d:.]+[^\]]*\]", "[ts]", l)
    l = re.sub(r"0x[0-9a-f]+|[0-9a-f]{12,}", "<hex>", l)
    l = re.sub(r"\d+(\.\d+)?", "N", l)
    return l.strip()[:200]
CLS = [  # (regex on normalised line, class, short reason) -- heuristic only; the reviewer confirms
    (r"freeze_gc|MaxRetryError|NewConnectionError|ConnectionError|ConnectionRefusedError|During handling of the above|direct cause of the following|^raise |^File \"|^Traceback|Traceback \(most recent", "upstream", "post-warmup freeze_gc / HTTP call racing engine teardown (Miles/SGLang); verify against the injection timeline"),
    (r"retry_utils.*(wait_init_expected_num_cells|retry_until_deadline)", "platform", "engine start wait (expected during start_cells; sleeps until the new SGLang engines answer)"),
    (r"ft op=check errored|decision=no_retry", "upstream", "Miles FT check, survivors normal (benign)"),
    (r"could not confirm termination|leaving the head up", "product", "launcher teardown could not confirm cluster termination (launcher/yeto teardown path) -- check cleanup evidence"),
    (r"Cluster\(s\) failed|sky down --purge", "platform", "sky autostop/status against an already-removed cluster"),
    (r"unauthenticated requests to the HF Hub|HF_TOKEN", "platform", "HF Hub unauthenticated warning (env)"),
    (r"muse_glimmer|qwen3_vl|megatron-bridge|trust_remote_code=True allows|Inferring the appropriate argparse|FutureWarning|DeprecationWarning|UserWarning", "upstream", "Miles/Megatron/HF import-time warning"),
    (r"user_agent_prefix", "platform", "sky/nebius SDK FutureWarning"),
    (r"instance\(s\) still live after down|retrying teardown|Cluster '.*' does not exist", "platform", "launcher teardown retry vs Nebius deletion latency / sky state"),
    (r"did not exit within .*SIGTERM|detokenizer .*crashed|SIGQUIT received|Sleeping .* before crash diagnostics|Killed actor", "product", "engine stop path (fork stop_cells / Miles process teardown) -- expected side effect of kill; confirm"),
    (r"event loop is already running|PublicationError|RECOVERY_REQUIRED|DriverError", "product", "yeto elastic path"),
    (r"router_sampler|sampler|inwatch|dkill|dctl|probe|selfcheck|n2inwatch|n2arm", "test", "test tooling"),
    (r"Failed to wait for instances|Reconciling|quota|ResourceExhausted|capacity", "platform", "Nebius provisioning"),
    (r"Connection (refused|reset|closed)|ssh|Could not resolve|banner exchange", "platform", "ssh/network to island"),
]
def klass(n):
    for rx, c, why in CLS:
        if re.search(rx, n, re.I): return c, why
    return "?", "unclassified"
groups = collections.OrderedDict()
for f in files:
    if not os.path.isfile(f): continue
    rel = os.path.relpath(f, R)
    for i, l in enumerate(rd(f).splitlines(), 1):
        if SKIP.search(l) or not PAT.search(l): continue
        n = norm(l); key = (rel, n)
        g = groups.setdefault(key, {"source": rel, "template": n, "count": 0, "first": i, "last": i, "example": ANSI.sub("", l)[:300]})
        g["count"] += 1; g["last"] = i
rows = []
for g in groups.values():
    g["class"], g["why"] = klass(g["template"]); rows.append(g)
rows.sort(key=lambda g: (g["class"] == "?", -g["count"]))
# also cross-file merge summary by template
byt = collections.defaultdict(lambda: {"count": 0, "sources": set()})
for g in rows: byt[g["template"]]["count"] += g["count"]; byt[g["template"]]["sources"].add(g["source"])
out = {"timings": T, "anomalies": rows, "n_templates": len(rows), "n_lines": sum(g["count"] for g in rows)}
json.dump(out, open(R + "/scan.json", "w"), indent=1, ensure_ascii=False, default=list)
md = ["# scan %s" % os.path.basename(R), "", "## timings", "```", json.dumps(T, indent=1, ensure_ascii=False, default=list), "```", "",
      "## anomalies (%d templates, %d lines)" % (len(rows), out["n_lines"]), "", "| class | count | source | template | why |", "|---|---|---|---|---|"]
for g in rows: md.append("| %s | %d | %s | `%s` | %s |" % (g["class"], g["count"], g["source"], g["template"].replace("|", "/")[:160], g["why"]))
open(R + "/scan.md", "w").write("\n".join(md) + "\n")
print("scan ok: %d templates %d lines -> %s/scan.md" % (len(rows), out["n_lines"], R))
