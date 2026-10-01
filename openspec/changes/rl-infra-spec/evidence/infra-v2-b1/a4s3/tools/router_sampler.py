# in-container: discover the Miles router (the local HTTP port whose /worker_inflight returns JSON
# with "inflight"), then poll it every 0.5 s -> ~/yeto-rl/router_samples.jsonl; also samples
# nvidia-smi compute-apps every 2 s -> ~/yeto-rl/gpu_samples.jsonl.
import json, os, re, subprocess, time, urllib.request
out = open(os.path.expanduser("~/yeto-rl/router_samples.jsonl"), "a")
gout = open(os.path.expanduser("~/yeto-rl/gpu_samples.jsonl"), "a")
def ports():
    out = set()
    for f in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            for l in open(f).read().splitlines()[1:]:
                c = l.split()
                if c[3] == "0A":  # LISTEN
                    p = int(c[1].split(":")[1], 16)
                    if 20000 <= p <= 20400:
                        out.add(p)
        except OSError:
            pass
    return sorted(out)
def get(url):
    with urllib.request.urlopen(url, timeout=2) as r:
        return json.loads(r.read())
ip = subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.split()[0]
router, lastg = None, 0.0
while True:
    now = time.time()
    if now - lastg >= 2:
        lastg = now
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout
        gout.write(json.dumps({"t": now, "apps": [l.split(", ") for l in apps.splitlines() if l]}) + "\n"); gout.flush()
    if router is None:
        for p in ports():
            try:
                d = get(f"http://{ip}:{p}/worker_inflight")
                if isinstance(d, dict) and "inflight" in d:
                    router = f"http://{ip}:{p}"
                    out.write(json.dumps({"t": now, "router": router}) + "\n"); break
            except Exception:  # noqa: BLE001
                pass
    else:
        try:
            out.write(json.dumps({"t": now, "data": get(router + "/worker_inflight")}) + "\n")
        except Exception as e:  # noqa: BLE001
            out.write(json.dumps({"t": now, "error": repr(e)[:200]}) + "\n")
    out.flush(); time.sleep(0.5)
