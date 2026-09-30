"""Local launcher for DEV-GATHER (2xA10G) and A8 (Modal ``H100!:2``) -- plan-v3 §1/§2.

NOT run by tests or by INFRA-E3 (no GPU was started). Usage, only after the
plan and this code are committed and the main agent approves the run:

    python modal_run.py {dev-gather|a8} <outdir> <learner_flags.txt> <app-name> <frozen repo snapshot>

``learner_flags.txt``: the flags of the ``python3 -m yeto.rl.learner`` line of
``yeto launch ... --rl-single-island-no-sync --controller local --dry-run``
(``build_flags.py`` extracts them). One Sandbox runs ``container_script`` with a
hard ``timeout``; the app id / sandbox id go to ``<outdir>/resources.txt`` for
the independent watchdog (``modal app stop <id>``). Registry credentials are
only the image pull secret (never printed, never in the task env).
"""

from __future__ import annotations

import os
import shlex
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
IMAGE = ("ghcr.io/michaellchung/yeto-miles-ports@sha256:"
         "17d428a2e955a1d43525b59b8785bb786b8e48852fe00c6e3e90dad798f0bcef")  # tag 5c1b49e-9f29303
MILES_COMMIT = "5c1b49ebccbc7508c1d9ef89eacc2db3e448b6ba"
ARMS = ("A1", "A2", "B1", "B1p", "B2", "RT")
# plan-v3: GPU, expected nvidia-smi name, hard timeout (s), whether determinism env is mandatory
PROFILES = {
    "dev-gather": {"gpu": "A10G:2", "expect": ("NVIDIA A10G", "NVIDIA A10"), "timeout": 5400, "deterministic": False},
    "a8": {"gpu": "H100!:2", "expect": ("NVIDIA H100 80GB HBM3",), "timeout": 7200, "deterministic": True},
}
# plan-v3 §0 profile: every dropout 0 (Megatron defaults hidden/attention to 0.1); A8 adds deterministic mode.
# balance_data: the ports translation (miles_adapter/config.py) always emits --balance-data, which the
# first-round DP certification refuses (review H1); the harness profile turns it off.
OVERRIDES = {"dev-gather": ["lora_dropout=0.0", "hidden_dropout=0.0", "attention_dropout=0.0",
                            "balance_data=false"]}
OVERRIDES["a8"] = OVERRIDES["dev-gather"] + ["deterministic_mode=true"]
DETERMINISM_ENV = {"NCCL_ALGO": "Ring", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "NVIDIA_TF32_OVERRIDE": "0"}


def container_script(profile: str, *, work: str = "/work/e3", flags_file: str = "/work/learner_flags.txt") -> str:
    p = PROFILES[profile]
    env = " ".join(f"{k}={shlex.quote(v)}" for k, v in DETERMINISM_ENV.items()) if p["deterministic"] else ""
    shim = "/yeto/tools/probes/e3_reshard/learner_shim.py"
    sets = " ".join(f"--set {o}" for o in OVERRIDES[profile])
    run = (f'env {env} PYTHONPATH=/root/miles:/sgl-workspace/sglang/python:/yeto:${{PYTHONPATH}} '
           f'bash -c "python {shim} --work {work} {sets} %s -- $(cat {flags_file})"')
    lines = [
        "set -euo pipefail",
        "cd /yeto",  # the reward module (gsm8k_reward.py) is imported from the working directory
        "export LEARNER_ID=0",  # the dry-run learner line reads $LEARNER_ID
        f"mkdir -p {work}",
        # GPU name assertion before anything else (plan-v3 §0)
        f"nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee {work}/gpus.txt",
        # DEV-GATHER (debug, no cross-model bitwise comparison) accepts A10G or A10 (main agent ruling);
        # A8 stays strict. The actual name and driver are in gpus.txt.
        "n=$(grep -c . {w}/gpus.txt); bad=$(grep -vcE '^({names}),' {w}/gpus.txt || true)".format(
            w=work, names="|".join(p["expect"])),
        f'if [ "$n" != 2 ] || [ "$bad" != 0 ]; then echo "GPU assertion failed"; exit 3; fi',
        f"test \"$(git --git-dir=/root/miles/.git rev-parse HEAD)\" = {MILES_COMMIT} || {{ echo 'miles pin mismatch'; exit 4; }}",
        f"PYTHONPATH=/root/miles:/yeto python -m yeto.rl.engine.runtime_manifest --image {IMAGE} --out {work}/runtime_manifest.json",
        "ray start --head --port=6379 --num-gpus=2 --disable-usage-stats",
        "export RAY_ADDRESS=127.0.0.1:6379",
        run % "--phase dry",
        run % "--phase gen",
    ]
    lines += [run % f"--phase arm --arm {arm}" for arm in ARMS]
    lines += [
        f"PYTHONPATH=/root/miles:/yeto python /yeto/tools/probes/e3_reshard/compare.py {work} "
        f"|| echo compare-failed",
        f"tar czf /work/e3-evidence.tgz -C {work} --exclude=cuts --exclude='arms/*/state' --exclude=frozen .",
        "echo === EVIDENCE_B64 ===; base64 -w0 /work/e3-evidence.tgz; echo",
    ]
    return "\n".join(lines)


def main(argv: list[str]) -> int:  # pragma: no cover - needs Modal credentials and GPUs
    import base64
    import json

    import modal

    import subprocess

    profile, out, flags, app_name, repo = argv[0], Path(argv[1]), Path(argv[2]), argv[3], Path(argv[4])
    out.mkdir(parents=True, exist_ok=True)
    # Identical check as the container's first step, run with the yeto environment
    # (E3_DRY_PYTHON; the Modal client venv has no yeto dependencies) on the uploaded snapshot.
    dry_python = os.environ.get("E3_DRY_PYTHON", sys.executable)
    proc = subprocess.run([dry_python, str(repo / "tools/probes/e3_reshard/local_dry.py"), profile, str(flags),
                           str(out / "local_dry.json")], cwd=repo, env={**os.environ, "PYTHONPATH": str(repo)},
                          capture_output=True, text=True)
    (out / "local_dry.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        print("local dry-run refused or failed; no Sandbox started (see local_dry.log)")
        return 2
    p = PROFILES[profile]
    auth = json.load(open(os.path.expanduser("~/.docker/config.json")))["auths"]["ghcr.io"]["auth"]
    user, token = base64.b64decode(auth).decode().split(":", 1)
    secret = modal.Secret.from_dict({"REGISTRY_USERNAME": user, "REGISTRY_PASSWORD": token})
    image = (modal.Image.from_registry(IMAGE, secret=secret).entrypoint([])
             .add_local_dir(str(repo), "/yeto", copy=False, ignore=[".git", "**/__pycache__", "openspec/**"])
             .add_local_file(str(flags), "/work/learner_flags.txt", copy=False))
    app = modal.App.lookup(app_name, create_if_missing=True)
    try:
        return _run(app, image, profile, p, out)
    finally:
        # Stop the app on every exit path (not only by the watchdog), then list it.
        subprocess.run([str(Path(sys.executable).with_name("modal")), "app", "stop", "-y", app.app_id],
                       check=False)
        (out / "app_stopped.txt").write_text(f"{app.app_id} {time.strftime('%FT%TZ', time.gmtime())}\n")


def _run(app, image, profile, p, out):  # pragma: no cover - needs Modal
    import base64

    import modal

    sb = modal.Sandbox.create("bash", "-c", container_script(profile), app=app, image=image, gpu=p["gpu"],
                              cpu=8.0, memory=65536, timeout=p["timeout"])
    (out / "resources.txt").open("a").write(
        f"{app.app_id} {sb.object_id} {profile} {time.strftime('%FT%TZ', time.gmtime())}\n")
    sb.wait(raise_on_termination=False)
    stdout, stderr = sb.stdout.read(), sb.stderr.read()
    (out / "stdout.txt").write_text(stdout.split("=== EVIDENCE_B64 ===")[0])
    (out / "stderr_tail.txt").write_text(stderr[-20000:])
    if "=== EVIDENCE_B64 ===" in stdout:
        (out / "evidence.tgz").write_bytes(base64.b64decode(stdout.split("=== EVIDENCE_B64 ===", 1)[1].strip()))
    (out / "returncode.txt").write_text(str(sb.returncode))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
