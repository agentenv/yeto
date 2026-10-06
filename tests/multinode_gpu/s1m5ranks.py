#!/usr/bin/env python3
"""m5 in-container rank probe (one island node): one JSON line per live sglang scheduler process
{ts, node, tp_rank, pid, title, gpu_uuid, method}. tp_rank from the sglang process title
(sglang::scheduler_TP<r>[...]); gpu_uuid from `nvidia-smi --query-compute-apps` (pid match), else the
process' single open /dev/nvidia<minor> mapped through /proc/driver/nvidia/gpus/*/information.
usage: python3 - <node index> < s1m5ranks.py"""
import glob, json, os, re, subprocess, sys, time

node = int(sys.argv[1]) if len(sys.argv) > 1 else -1
by_pid = {}
try:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=30).stdout
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            by_pid.setdefault(int(parts[0]), set()).add(parts[1])
except Exception:
    pass
minor = {}
for info in glob.glob("/proc/driver/nvidia/gpus/*/information"):
    txt = open(info, errors="replace").read()
    m, u = re.search(r"Device Minor:\s*(\d+)", txt), re.search(r"GPU UUID:\s*(\S+)", txt)
    if m and u:
        minor[int(m.group(1))] = u.group(1)
for d in glob.glob("/proc/[0-9]*"):
    pid = int(d.rsplit("/", 1)[1])
    try:
        title = open(f"{d}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace").strip()
    except Exception:
        continue
    m = re.match(r"sglang::scheduler\S*?TP(\d+)", title)
    if not m:
        continue
    uuid, method = None, None
    if len(by_pid.get(pid, ())) == 1:
        uuid, method = next(iter(by_pid[pid])), "nvidia-smi-pid"
    else:
        devs = set()
        for fd in glob.glob(f"{d}/fd/*"):
            try:
                t = os.readlink(fd)
            except OSError:
                continue
            mm = re.fullmatch(r"/dev/nvidia(\d+)", t)
            if mm:
                devs.add(int(mm.group(1)))
        if len(devs) == 1 and next(iter(devs)) in minor:
            uuid, method = minor[next(iter(devs))], "dev-minor"
        else:
            method = f"unresolved(devs={sorted(devs)})"
    print(json.dumps({"ts": time.time(), "node": node, "tp_rank": int(m.group(1)), "pid": pid,
                      "title": title[:80], "gpu_uuid": uuid, "method": method}))
