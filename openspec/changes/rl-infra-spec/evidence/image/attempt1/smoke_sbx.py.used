"""IMG smoke: private ports image on Modal (see plan.md). Usage:
   smoke_sbx.py <image_ref> <yeto.tgz> <outdir>"""
import base64, json, os, sys, time
import modal

APP = "img-smoke"
ref, tgz, out = sys.argv[1:4]
auth = json.load(open(os.path.expanduser("~/.docker/config.json")))["auths"]["ghcr.io"]["auth"]
user, token = base64.b64decode(auth).decode().split(":", 1)
secret = modal.Secret.from_dict({"REGISTRY_USERNAME": user, "REGISTRY_PASSWORD": token})
image = modal.Image.from_registry(ref, secret=secret).entrypoint([])
app = modal.App.lookup(APP, create_if_missing=True)
t0 = time.time()
sb = modal.Sandbox.create("sleep", "infinity", app=app, image=image, gpu="L40S",
                          cpu=8, memory=65536, timeout=2400)
open(os.path.join(out, "sandbox_id.txt"), "w").write(sb.object_id + "\n")
print("sandbox", sb.object_id, "ready after", round(time.time() - t0), "s", flush=True)
try:
    sb.filesystem.write_bytes(open(tgz, "rb").read(), "/tmp/yeto.tgz")
    p = sb.exec("bash", "-lc", "mkdir -p /work/yeto && tar xzf /tmp/yeto.tgz -C /work/yeto && bash /work/yeto/openspec/changes/rl-infra-spec/evidence/image/smoke_in_image.sh", timeout=2000)
    log = open(os.path.join(out, "smoke.log"), "w")
    for line in p.stdout:
        print(line, end="", flush=True); log.write(line)
    err = p.stderr.read(); log.write("\n--- stderr ---\n" + err); log.close()
    rc = p.wait()
    print("smoke rc", rc, flush=True)
finally:
    sb.terminate()
    print("terminated", sb.object_id, "total", round(time.time() - t0), "s", flush=True)
sys.exit(rc)
