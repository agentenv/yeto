"""algo2b-t4-parse: full Miles parse_args + validate_parsed_args in the pinned image (4.4).
Registry credentials are read from ~/.docker/config.json and passed only as the
from_registry pull secret (never printed, never in the task env)."""
import base64, json, os
import modal

APP = "algo2b-t4-parse"
IMAGE = ("ghcr.io/michaellchung/yeto-miles-ports@sha256:"
         "17d428a2e955a1d43525b59b8785bb786b8e48852fe00c6e3e90dad798f0bcef")  # miles 5c1b49eb
_auth = json.load(open(os.path.expanduser("~/.docker/config.json")))["auths"]["ghcr.io"]["auth"]
_user, _token = base64.b64decode(_auth).decode().split(":", 1)
_secret = modal.Secret.from_dict({"REGISTRY_USERNAME": _user, "REGISTRY_PASSWORD": _token})
image = (modal.Image.from_registry(IMAGE, secret=_secret).entrypoint([])
         .add_local_dir("/home/michael/work/algo-2b", "/yeto", copy=False,
                        ignore=[".git", "**/__pycache__", "openspec/**"])
         .add_local_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_parse.py"),
                         "/work/full_parse.py", copy=False))
app = modal.App(APP, image=image)

SCRIPT = r"""
set -x
nvidia-smi --query-gpu=name,driver_version --format=csv
git --git-dir=/root/miles/.git rev-parse HEAD
cd /tmp
PYTHONPATH=/yeto:${PYTHONPATH} python /work/full_parse.py /tmp/result.json 2>&1 | grep -v "Warning\|warn(" | tail -80
echo "=== RESULT_JSON ==="
cat /tmp/result.json
"""


@app.function(gpu="T4", cpu=2.0, memory=8192, timeout=840)
def run() -> str:
    import subprocess
    out = subprocess.run(["bash", "-c", SCRIPT], capture_output=True, text=True)
    return out.stdout + "\n--- stderr (tail) ---\n" + out.stderr[-6000:]


@app.local_entrypoint()
def main():
    print(run.remote())
