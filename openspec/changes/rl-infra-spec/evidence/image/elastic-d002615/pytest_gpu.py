import base64, json, os, sys, modal
auth = json.load(open(os.path.expanduser("~/.docker/config.json")))["auths"]["ghcr.io"]["auth"]
u, t = base64.b64decode(auth).decode().split(":", 1)
app = modal.App.lookup("img-smoke", create_if_missing=True)
img = modal.Image.from_registry(sys.argv[1], secret=modal.Secret.from_dict({"REGISTRY_USERNAME": u, "REGISTRY_PASSWORD": t})).entrypoint([])
sb = modal.Sandbox.create("sleep", "infinity", app=app, image=img, gpu="L40S", cpu=8, memory=65536, timeout=1500)
try:
    sb.filesystem.write_bytes(open(sys.argv[2], "rb").read(), "/tmp/y.tgz")
    p = sb.exec("bash", "-lc", "mkdir -p /work/yeto && tar xzf /tmp/y.tgz -C /work/yeto && cd /work/yeto && (python3 -c 'import pytest' 2>/dev/null || pip install -q pytest) && PYTHONPATH=/root/miles:/work/yeto python3 -m pytest -q -rfs --tb=short -p no:cacheprovider tests/test_rl_miles_adapter_config.py tests/test_rl_argv_snapshot.py 2>&1 | grep -v -iE 'warn|flax' | tail -80", timeout=900)
    print(p.stdout.read(), flush=True); print("rc", p.wait())
finally:
    sb.terminate(); print("terminated", sb.object_id)
