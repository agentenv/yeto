"""G1 driver on Modal. usage: g1_sbx.py <yeto.tgz> <localdir> "<run names>"
Sandbox: app algo2a-g1, GPU H100! (exact, asserted in run_all.sh), timeout 5400 s."""
import base64, json, os, sys, time
import modal

APP = "algo2a-g1"
IMAGE = ("ghcr.io/michaellchung/yeto-miles-ports@sha256:"
         "5da40a07dabb3ea3fcf921efb4b2a21ca1178220bde40dc178c79b734fdaa540")  # = MILES_NEXT_IMAGE (rl-integ f6194da)
tgz, out, runs = sys.argv[1:4]
auth = json.load(open(os.path.expanduser("~/.docker/config.json")))["auths"]["ghcr.io"]["auth"]
user, token = base64.b64decode(auth).decode().split(":", 1)
secret = modal.Secret.from_dict({"REGISTRY_USERNAME": user, "REGISTRY_PASSWORD": token})
image = modal.Image.from_registry(IMAGE, secret=secret).entrypoint([])
app = modal.App.lookup(APP, create_if_missing=True)
t0 = time.time()
sb = modal.Sandbox.create("sleep", "infinity", app=app, image=image, gpu="H100!",
                          cpu=16, memory=131072, timeout=5400)
open(os.path.join(out, "sandbox_id.txt"), "a").write(sb.object_id + " " + time.strftime("%FT%TZ", time.gmtime()) + "\n")
print("sandbox", sb.object_id, "ready after", round(time.time() - t0), "s", flush=True)
rc = 99
try:
    sb.filesystem.write_bytes(open(tgz, "rb").read(), "/tmp/yeto.tgz")
    p = sb.exec("bash", "-lc",
                "mkdir -p /work/yeto /work/g1 /work/out && tar xzf /tmp/yeto.tgz -C /work/yeto && "
                "cp /work/yeto/openspec/changes/rl-algo-seq-and-adv/evidence/g1/*.py /work/g1/ && "
                f"bash /work/yeto/openspec/changes/rl-algo-seq-and-adv/evidence/g1/run_all.sh '{runs}'",
                timeout=5200)
    with open(os.path.join(out, "driver.log"), "a") as log:
        for line in p.stdout:
            print(line, end="", flush=True); log.write(line)
        log.write("\n--- stderr ---\n" + p.stderr.read())
    rc = p.wait()
    print("sandbox script rc", rc, flush=True)
    try:
        open(os.path.join(out, "out.tgz"), "wb").write(sb.filesystem.read_bytes("/tmp/out.tgz"))
        print("fetched out.tgz", flush=True)
    except Exception as exc:  # noqa: BLE001
        print("fetch failed", exc, flush=True)
finally:
    sb.terminate()
    print("terminated", sb.object_id, "total", round(time.time() - t0), "s", flush=True)
sys.exit(rc)
