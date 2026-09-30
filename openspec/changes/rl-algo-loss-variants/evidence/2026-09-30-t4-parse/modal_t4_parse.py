"""algo2b-t4-parse (run 2): Modal Sandbox driver, runs locally only.
Registry credentials are read from ~/.docker/config.json and passed only as the
from_registry pull secret (never printed, never in the task env).
usage: python modal_t4_parse.py <outdir>"""
import base64, json, os, sys, time
import modal

APP = "algo2b-t4-parse"
IMAGE = ("ghcr.io/michaellchung/yeto-miles-ports@sha256:"
         "17d428a2e955a1d43525b59b8785bb786b8e48852fe00c6e3e90dad798f0bcef")  # miles 5c1b49eb
HERE = os.path.dirname(os.path.abspath(__file__))
out = sys.argv[1]
auth = json.load(open(os.path.expanduser("~/.docker/config.json")))["auths"]["ghcr.io"]["auth"]
user, token = base64.b64decode(auth).decode().split(":", 1)
secret = modal.Secret.from_dict({"REGISTRY_USERNAME": user, "REGISTRY_PASSWORD": token})
image = (modal.Image.from_registry(IMAGE, secret=secret).entrypoint([])
         .add_local_dir("/home/michael/work/algo-2b", "/yeto", copy=False,
                        ignore=[".git", "**/__pycache__", "openspec/**"])
         .add_local_file(os.path.join(HERE, "full_parse.py"), "/work/full_parse.py", copy=False))
SCRIPT = r"""
set -x
nvidia-smi --query-gpu=name,driver_version --format=csv
git --git-dir=/root/miles/.git rev-parse HEAD
cd /tmp
PYTHONPATH=/yeto:${PYTHONPATH} python /work/full_parse.py /tmp/result.json 2>&1 | grep -v "Warning\|warn(" | tail -80
echo "=== RESULT_JSON ==="
cat /tmp/result.json
"""
app = modal.App.lookup(APP, create_if_missing=True)
print("app", app.app_id, flush=True)
sb = modal.Sandbox.create("bash", "-c", SCRIPT, app=app, image=image, gpu="T4",
                          cpu=2.0, memory=8192, timeout=840)
open(os.path.join(out, "sandbox_id.txt"), "a").write(f"{app.app_id} {sb.object_id} {time.strftime('%FT%TZ', time.gmtime())}\n")
print("sandbox", sb.object_id, flush=True)
sb.wait(raise_on_termination=False)
stdout, stderr = sb.stdout.read(), sb.stderr.read()
print(stdout); print("--- stderr (tail) ---"); print(stderr[-6000:])
print("sandbox returncode", sb.returncode)
if "=== RESULT_JSON ===" in stdout:
    open(os.path.join(out, "result.json"), "w").write(stdout.split("=== RESULT_JSON ===", 1)[1].strip() + "\n")
