"""1.1 manifest in the ports image on Modal (see plan.md). Usage: run.py <image_ref> <yeto.tgz> <outdir>"""
import base64, json, os, sys, time
import modal

APP = "infra-a-manifest"
ref, tgz, out = sys.argv[1:4]
auth = json.load(open(os.path.expanduser("~/.docker/config.json")))["auths"]["ghcr.io"]["auth"]
user, token = base64.b64decode(auth).decode().split(":", 1)
secret = modal.Secret.from_dict({"REGISTRY_USERNAME": user, "REGISTRY_PASSWORD": token})
image = modal.Image.from_registry(ref.removeprefix("docker:"), secret=secret).entrypoint([])
app = modal.App.lookup(APP, create_if_missing=True)
t0 = time.time()
sb = modal.Sandbox.create("sleep", "infinity", app=app, image=image, gpu="L40S", cpu=8, memory=65536, timeout=1500)
open(os.path.join(out, "sandbox_id.txt"), "w").write(sb.object_id + "\n")
print("sandbox", sb.object_id, "ready after", round(time.time() - t0), "s", flush=True)
rc = 1
try:
    sb.filesystem.write_bytes(open(tgz, "rb").read(), "/tmp/yeto.tgz")
    cmd = f"""set -u; mkdir -p /work/yeto && tar xzf /tmp/yeto.tgz -C /work/yeto && cd /work/yeto
nvidia-smi -L
export PYTHONPATH=/work/yeto:/root/miles
python3 -m yeto.rl.engine.runtime_manifest --image {ref} --out /tmp/manifest.json --capability ports-colocated-serial --capability ports-partitioned-serial; echo "RC_BASE=$?"
python3 -m yeto.rl.engine.runtime_manifest --check /tmp/manifest.json --capability fixed-partition-standby; echo "RC_STANDBY=$?"
python3 -m yeto.rl.engine.runtime_manifest --check /tmp/manifest.json --capability RolloutPool.add_engines; echo "RC_ADD=$?"
echo ===MANIFEST; cat /tmp/manifest.json"""
    p = sb.exec("bash", "-lc", cmd, timeout=1200)
    text = p.stdout.read(); err = p.stderr.read(); rc = p.wait()
    open(os.path.join(out, "run.log"), "w").write(text + "\n--- stderr ---\n" + err)
    if "===MANIFEST" in text:
        open(os.path.join(out, "manifest.json"), "w").write(text.split("===MANIFEST", 1)[1].strip() + "\n")
    print(text[-3000:], "\nSTDERR:", err[-2000:], "\nrc", rc, flush=True)
finally:
    sb.terminate()
    print("terminated", sb.object_id, "total", round(time.time() - t0), "s", flush=True)
sys.exit(rc)
