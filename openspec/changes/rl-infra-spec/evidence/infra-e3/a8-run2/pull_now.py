import hashlib, json, sys, time
import modal
R = "/home/michael/work/infra-e3-gpu/b3a8r"
sb = modal.Sandbox.from_id("sb-EyrZEOiYfyCeMyWkyEjkwb")
fs = sb.filesystem
dest = f"{R}/out/work/packed"
index = json.loads(fs.read_text("/work/e3/packed/index.json"))
open(f"{dest}/index.json", "w").write(json.dumps(index, indent=1, sort_keys=True))
report = {"ok": [], "bad": []}
for name, meta in sorted(index["files"].items()):
    t = time.time()
    fs.copy_to_local(f"/work/e3/packed/{name}", f"{dest}/{name}")
    h = hashlib.sha256(open(f"{dest}/{name}", "rb").read()).hexdigest()
    (report["ok"] if h == meta["sha256"] else report["bad"]).append(name)
    print(name, "ok" if h == meta["sha256"] else "BAD", f"{time.time()-t:.1f}s", flush=True)
open(f"{dest}/pull_report_fsapi.json", "w").write(json.dumps(report, indent=1))
fs.write_text("pulled\n", "/work/pulled")
print("released", len(report["ok"]), "ok", len(report["bad"]), "bad", flush=True)
