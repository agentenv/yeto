"""Run the E2 learner preflight in the pinned image on a Modal CPU sandbox (approved: B2, cap $1, 20 min).

usage: python e2_modal_preflight.py <snapshot-repo> <dry-run-root> <out-dir> <run> [<run> ...]
Registry credentials are read from ~/.docker/config.json and used only as the pull secret.
"""
import base64
import json
import os
import shlex
import sys
import time

import modal

APP = os.environ.get("E2PF_APP", "infra-v2-b2-e2pf-20260930-1")
GPU = os.environ.get("E2PF_GPU") or None  # e.g. "T4": libcuda must dlopen for Megatron/TE imports
repo, dry, out, runs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
sys.path.insert(0, repo)
from yeto.rl import MILES_NEXT_IMAGE  # noqa: E402  the checkout's pin

IMAGE = MILES_NEXT_IMAGE.removeprefix("docker:")
os.makedirs(out, exist_ok=True)
auth = json.load(open(os.path.expanduser("~/.docker/config.json")))["auths"]["ghcr.io"]["auth"]
user, token = base64.b64decode(auth).decode().split(":", 1)
secret = modal.Secret.from_dict({"REGISTRY_USERNAME": user, "REGISTRY_PASSWORD": token})
image = (modal.Image.from_registry(IMAGE, secret=secret).entrypoint([])
         .add_local_dir(os.path.join(repo, "yeto"), "/yeto/yeto", copy=False, ignore=["**/__pycache__"])
         .add_local_file(os.path.join(repo, "gsm8k_reward.py"), "/yeto/gsm8k_reward.py", copy=False)
         .add_local_file(os.path.join(os.path.dirname(__file__), "e2_container_preflight.py"),
                         "/work/e2_container_preflight.py", copy=False)
         .add_local_dir(dry, "/dry", copy=False))
steps = []
for run in runs:
    plan = json.load(open(os.path.join(dry, "plan.json")))
    spec = next(r for r in plan["runs"] if r["run"] == run)
    d = "/dry/" + os.path.basename(spec["dir"])
    text = open(os.path.join(spec["dir"], "launcher-dry-run.txt")).read()
    launcher = json.loads(text[text.index("{"):text.rindex("}") + 1])
    cmd = launcher["island_requests"][0]["learner_command"]
    learner_args = cmd.split("python3 -m yeto.rl.adapters.miles.island_entry", 1)[1].replace("$LEARNER_ID", "0")
    with open(os.path.join(spec["dir"], "learner-args.txt"), "w") as fh:
        fh.write(learner_args)
    harness = f"{d}/harness.json" if os.path.exists(os.path.join(spec["dir"], "harness.json")) else "-"
    res = f"{d}/resources.json"
    steps.append(
        f"mkdir -p ~/yeto-rl ~/yeto-output; [ -f {res} ] && cp {res} ~/yeto-rl/elastic_resources.json; "
        f"cd /yeto && PYTHONPATH=/root/miles:/root/sglang/python:/sgl-workspace/sglang/python:/yeto "
        f"timeout 480 python3 /work/e2_container_preflight.py {run} {d}/learner-args.txt {harness} /tmp/{run}.json "
        f"> /tmp/{run}.log 2>&1; echo rc_{run}=$?; tail -3 /tmp/{run}.log; "
        f"echo '=== RESULT {run} ==='; cat /tmp/{run}.json; echo '=== END ==='")
SCRIPT = "\n".join([
    "nproc; git --git-dir=/root/miles/.git rev-parse HEAD; nvidia-smi --query-gpu=name,driver_version --format=csv 2>&1 | head -3",
    *steps,
    "cd /yeto && PYTHONPATH=/root/miles:/root/sglang/python:/yeto timeout 240 python3 -m "
    f"yeto.rl.engine.runtime_manifest --image {IMAGE} --out /tmp/m.json > /tmp/m.log 2>&1; echo rc_manifest=$?; "
    "tail -5 /tmp/m.log; echo '=== RESULT manifest ==='; cat /tmp/m.json; echo '=== END ==='",
])
app = modal.App.lookup(APP, create_if_missing=True)
print("app", app.app_id, flush=True)
sb = modal.Sandbox.create("bash", "-c", SCRIPT, app=app, image=image, cpu=2.0, memory=8192, timeout=1080,
                          **({"gpu": GPU} if GPU else {}))
with open(os.path.join(out, "sandbox_id.txt"), "a") as fh:
    fh.write(f"{app.app_id} {sb.object_id} {time.strftime('%FT%TZ', time.gmtime())}\n")
print("sandbox", sb.object_id, flush=True)
sb.wait(raise_on_termination=False)
stdout, stderr = sb.stdout.read(), sb.stderr.read()
open(os.path.join(out, "stdout.txt"), "w").write(stdout)
open(os.path.join(out, "stderr_tail.txt"), "w").write(stderr[-20000:])
print(stdout[-6000:])
print("sandbox returncode", sb.returncode)
for chunk in stdout.split("=== RESULT ")[1:]:
    name, body = chunk.split(" ===\n", 1)
    open(os.path.join(out, f"result-{name}.json"), "w").write(body.split("=== END ===", 1)[0].strip() + "\n")
