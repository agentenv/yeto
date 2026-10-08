# S17 N7 A5 in-container sidecar. usage: python3 sidecar.py '<json {"triggers": [[phase, rollout_id, req_id, body, verb?], ...]}>'
# Writes only into /root/yeto-output (mirrored to the Modal tape volume by TapeSync):
#   s17-gpu-<island>.jsonl     every 2 s: per-GPU util/mem/power + compute apps
#   s17-router-<island>.jsonl  every 0.5 s: Miles router /worker_inflight (once discovered)
#   s17-inwatch-<island>.jsonl trigger submissions (island 0 only)
#   s17-elastic-state-<island>.b64.txt  every 15 s: tar.gz(base64) of ~/yeto-rl/elastic-state (journal, inbox, status)
import base64, glob, io, json, os, re, subprocess, sys, tarfile, time, urllib.request

OUT = "/root/yeto-output"
cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
trig = cfg.get("triggers", [])
STATE = os.path.expanduser("~/yeto-rl/elastic-state")


def island_id():
    while True:
        tapes = sorted(glob.glob(f"{OUT}/rl-island-*.jsonl"))
        if tapes:
            return int(re.search(r"rl-island-(\d+)\.jsonl", tapes[0]).group(1)), tapes[0]
        time.sleep(2)


def ports():
    out = set()
    for f in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            for line in open(f).read().splitlines()[1:]:
                c = line.split()
                if c[3] == "0A":
                    out.add(int(c[1].split(":")[1], 16))
        except OSError:
            pass
    return sorted(out)


def get(url):
    with urllib.request.urlopen(url, timeout=2) as r:
        return json.loads(r.read())


iid, tape = island_id()
gout = open(f"{OUT}/s17-gpu-{iid}.jsonl", "a")
rout = open(f"{OUT}/s17-router-{iid}.jsonl", "a")
wlog = open(f"{OUT}/s17-inwatch-{iid}.jsonl", "a")
wlog.write(json.dumps({"start": time.time(), "island": iid, "triggers": trig}) + "\n"); wlog.flush()
router, lastg, lasts, lastr, pos, done = None, 0.0, 0.0, 0.0, 0, set()
while True:
    now = time.time()
    if now - lastg >= 2:
        lastg = now
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,power.draw",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        a = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True).stdout
        gout.write(json.dumps({"t": now, "gpus": [l.split(", ") for l in q.splitlines() if l],
                               "apps": [l.split(", ") for l in a.splitlines() if l]}) + "\n"); gout.flush()
    if now - lasts >= 15:
        lasts = now
        if os.path.isdir(STATE):
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tf:
                tf.add(STATE, arcname="elastic-state")
            tmp = f"{OUT}/.s17-es.tmp"
            open(tmp, "w").write(base64.b64encode(buf.getvalue()).decode())
            os.replace(tmp, f"{OUT}/s17-elastic-state-{iid}.b64.txt")
    if router is None and now - lastr >= 5:
        lastr = now
        try:
            ips = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=5).stdout.split()
        except Exception:  # noqa: BLE001
            ips = []
        for p in ports():
            if not 20000 <= p <= 20400:
                continue
            for ip in ips + ["127.0.0.1"]:
                try:
                    d = get(f"http://{ip}:{p}/worker_inflight")
                    if isinstance(d, dict):
                        router = f"http://{ip}:{p}"
                        rout.write(json.dumps({"t": now, "router": router}) + "\n"); break
                except Exception:  # noqa: BLE001
                    pass
            if router:
                break
    elif router is not None:
        try:
            rout.write(json.dumps({"t": now, "data": get(router + "/worker_inflight")}) + "\n")
        except Exception as e:  # noqa: BLE001
            rout.write(json.dumps({"t": now, "error": repr(e)[:200]}) + "\n")
        rout.flush()
    if iid == 0 and len(done) < len(trig):
        try:
            with open(tape) as f:
                f.seek(pos); data = f.read(); pos = f.tell()
        except FileNotFoundError:
            data = ""
        for line in data.splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("event") != "rl_driver_phase":
                continue
            for i, t in enumerate(trig):
                ph, rid, req, body = t[:4]; verb = t[4] if len(t) > 4 else "request"
                if i in done or e.get("phase") != ph or e.get("rollout_id") != rid:
                    continue
                os.makedirs(f"{STATE}/inbox", exist_ok=True)
                p = f"{STATE}/inbox/{req}.{verb}.json"
                open(p + ".tmp", "w").write(json.dumps(body)); os.replace(p + ".tmp", p)
                done.add(i)
                wlog.write(json.dumps({"submitted": req, "trigger": [ph, rid], "body": body,
                                       "event_time": e.get("time_unix"), "wall": time.time()}) + "\n"); wlog.flush()
    time.sleep(0.5)
