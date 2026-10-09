"""Local launcher for DEV-GATHER (2xA10G) and A8 (Modal ``H100!:2``) -- plan-v3 §1/§2.

NOT run by tests or by INFRA-E3 (no GPU was started). Usage, only after the
plan and this code are committed and the main agent approves the run:

    python modal_run.py {dev-gather|a8} <outdir> <learner_flags.txt> <app-name> <frozen repo snapshot>

``learner_flags.txt``: the flags of the ``python3 -m yeto.rl.adapters.miles.island_entry`` line of
``yeto launch ... --rl-single-island-no-sync --controller local --dry-run``
(``build_flags.py`` extracts them). One Sandbox runs ``container_script`` with a
hard ``timeout``; the app id / sandbox id go to ``<outdir>/resources.txt`` for
the independent watchdog (``modal app stop <id>``). Registry credentials are
only the image pull secret (never printed, never in the task env).
"""

from __future__ import annotations

import os
import re
import shlex
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
def _pins() -> tuple[str, str]:
    """MILES_NEXT_IMAGE digest and MILES_NEXT_COMMIT of this checkout (read as text: no yeto import)."""
    text = (REPO / "yeto" / "rl" / "adapters" / "miles" / "pins.py").read_text()
    image = re.search(r'MILES_NEXT_IMAGE = \(\s*"docker:([^"]+)"\s*"([^"]+)"', text)
    commit = re.search(r'MILES_NEXT_COMMIT = "([0-9a-f]{40})"', text)
    return image.group(1) + image.group(2), commit.group(1)


IMAGE, MILES_COMMIT = _pins()
ARMS = ("A1", "A2", "B1", "B1p", "B2", "RT")
# plan-v3: GPU, expected nvidia-smi name, hard timeout (s), whether determinism env is mandatory
PROFILES = {
    "dev-gather": {"gpu": "A10G:2", "expect": ("NVIDIA A10G", "NVIDIA A10"), "timeout": 5400, "deterministic": False},
    "a8": {"gpu": "H100!:2", "expect": ("NVIDIA H100 80GB HBM3",), "timeout": 7200, "deterministic": True},
}
# plan-v3 §0 profile: every dropout 0 (Megatron defaults hidden/attention to 0.1); A8 adds deterministic mode.
# Profile overrides applied after parse (recorded in miles_args.*.json). --balance-data is NOT
# overridden: the harness uses the production trainer-edge translation (RLRunConfig.trainer_dp_edges,
# the field --rl-elastic-trainer-edges sets), which omits it. A8's determinism comes from the learner flag
# --rl-deterministic-trainer (Megatron --deterministic-mode + NCCL/cuBLAS/TF32 env), checked in local_dry.
# Megatron hidden/attention dropout have no yeto flag (default 0.1): set to 0 here.
OVERRIDES = {"dev-gather": ["hidden_dropout=0.0", "attention_dropout=0.0"]}
OVERRIDES["a8"] = list(OVERRIDES["dev-gather"])
REQUIRED_ARGV = {"dev-gather": (), "a8": ("--deterministic-mode",)}
DETERMINISM_ENV = {"NCCL_ALGO": "Ring", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "NVIDIA_TF32_OVERRIDE": "0",
                   "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0"}  # = entry.DETERMINISM_ENV (checked by a test)


PACKED_MARKER = "=== PACKED READY ==="
PULLED_FLAG = "/work/pulled"
PULL_WAIT_S = 1200
STALL_MINUTES = 20  # a phase with no new progress line for this long is killed (fail)
MAX_SERVER_ERRORS = 200  # 5xx / "request failed with server error" lines in one phase -> fail
LOCAL_SILENCE_MINUTES = 25  # the local launcher stops the Sandbox when no output arrives for this long


def container_script(profile: str, *, work: str = "/work/e3", flags_file: str = "/work/learner_flags.txt",
                     stall_minutes: int = STALL_MINUTES, max_server_errors: int = MAX_SERVER_ERRORS) -> str:
    p = PROFILES[profile]
    env = " ".join(f"{k}={shlex.quote(v)}" for k, v in DETERMINISM_ENV.items()) if p["deterministic"] else ""
    shim = "/yeto/tools/probes/e3_reshard/learner_shim.py"
    sets = " ".join(f"--set {o}" for o in OVERRIDES[profile])
    # bash -c re-parses the shell-quoted learner flags (e.g. the chat-template JSON).
    cmd = (f'env {env} PYTHONPATH=/root/miles:/sgl-workspace/sglang/python:/yeto:${{PYTHONPATH:-}} '
           f'bash -c "python {shim} --work {work} {sets} $2 -- $(cat {flags_file})"')
    lines = [
        "set -uo pipefail",
        "exec 2>&1",  # every command's stderr into the mirrored stream (run 7 lost compare's traceback)
        "cd /yeto",  # the reward module (gsm8k_reward.py) is imported from the working directory
        "export LEARNER_ID=0",  # the dry-run learner line reads $LEARNER_ID
        f"mkdir -p {work}/logs",
        f"export E3_PROGRESS_FILE={work}/progress.log",
        # Evidence is packed on EVERY exit (success, failure, stall kill).
        "pack() {",
        # Data safety net (also after a failure): pack gathered states and wait until the launcher
        # pulled them (<= PULL_WAIT_S), so G1-G6 can be recomputed offline if compare fails.
        f'  if ls {work}/arms/*/state >/dev/null 2>&1; then',
        f'    echo "$(date -u +%FT%TZ) pack start" >> {work}/progress.log',
        f"    PYTHONPATH=/root/miles:/yeto python /yeto/tools/probes/e3_reshard/pack_states.py {work} "
        f"|| echo pack-failed >> {work}/progress.log",
        f'    echo "{PACKED_MARKER}"',
        f"    for i in $(seq 1 {PULL_WAIT_S // 5}); do [ -f {PULLED_FLAG} ] && break; sleep 5; done",
        f'    [ -f {PULLED_FLAG} ] && echo "packed pulled" >> {work}/progress.log',
        "  fi",
        f"  tar czf /work/e3-evidence.tgz -C {work} --exclude=cuts --exclude='arms/*/state' --exclude=frozen "
        "--exclude=packed . "
        "2>/dev/null",
        '  echo "=== EVIDENCE_B64 ==="; base64 -w0 /work/e3-evidence.tgz; echo',
        "}",
        "trap pack EXIT",
        "progress() { echo \"$(date -u +%FT%TZ) $*\" | tee -a $E3_PROGRESS_FILE; }",
        # run_phase NAME ARGS: stall / server-error watchdog per phase (not only the Sandbox timeout).
        "run_phase() {",
        f'  local log={work}/logs/$1.log; progress "phase $1 start"',
        f'  ( {cmd} ) > "$log" 2>&1 &',
        "  local pid=$!",
        "  while kill -0 $pid 2>/dev/null; do",
        "    sleep 30",
        "    local age=$(( $(date +%s) - $(stat -c %Y $E3_PROGRESS_FILE) ))",
        "    local errs=$(grep -cE '503 Service Unavailable|request failed with server error' \"$log\" || true)",
        f'    if [ $age -gt {stall_minutes * 60} ]; then progress "phase $1 STALLED ${{age}}s"; '
        'kill -TERM $pid; sleep 10; kill -KILL $pid 2>/dev/null; tail -200 "$log"; exit 5; fi',
        f'    if [ "$errs" -gt {max_server_errors} ]; then progress "phase $1 SERVER-ERRORS $errs"; '
        'kill -TERM $pid; sleep 10; kill -KILL $pid 2>/dev/null; tail -200 "$log"; exit 6; fi',
        "  done",
        "  wait $pid; local rc=$?",
        '  tail -40 "$log"; progress "phase $1 rc=$rc"',
        '  if [ $rc -ne 0 ]; then exit 7; fi',
        "}",
        # GPU name assertion before anything else (plan-v3 §0)
        f"nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee {work}/gpus.txt",
        # DEV-GATHER (debug, no cross-model bitwise comparison) accepts A10G or A10 (main agent ruling);
        # A8 stays strict. The actual name and driver are in gpus.txt.
        "n=$(grep -c . {w}/gpus.txt); bad=$(grep -vcE '^({names}),' {w}/gpus.txt || true)".format(
            w=work, names="|".join(p["expect"])),
        'if [ "$n" != 2 ] || [ "$bad" != 0 ]; then echo "GPU assertion failed"; exit 3; fi',
        f"test \"$(git --git-dir=/root/miles/.git rev-parse HEAD)\" = {MILES_COMMIT} || {{ echo 'miles pin mismatch'; exit 4; }}",
        f"PYTHONPATH=/root/miles:/yeto python -m yeto.rl.engine.runtime_manifest --image {IMAGE} "
        f"--out {work}/runtime_manifest.json || exit 8",
        # determinism env before the raylet starts, so every Ray worker inherits it (A8)
        *([f"export {k}={shlex.quote(v)}" for k, v in DETERMINISM_ENV.items()] if p["deterministic"] else []),
        "ray start --head --port=6379 --num-gpus=2 --disable-usage-stats > /dev/null || exit 9",
        "export RAY_ADDRESS=127.0.0.1:6379",
        'run_phase dry "--phase dry"',
        'run_phase gen "--phase gen"',
    ]
    lines += [f'run_phase {arm} "--phase arm --arm {arm}"' for arm in ARMS]
    lines += [
        'progress "compare start"',
        f"PYTHONPATH=/root/miles:/yeto python /yeto/tools/probes/e3_reshard/compare.py {work} "
        f"|| progress compare-failed",
        f"cat {work}/RESULT.json 2>/dev/null || true",
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


def pull_packed(sb, dest: Path, *, touch, log, work: str = "/work/e3") -> dict:
    """Copy <work>/packed/* out of the running Sandbox (Modal filesystem API), verify sha256, release.

    A8 run 2: the legacy ``Sandbox.open`` API is refused by Modal ("legacy Sandbox filesystem API is no
    longer supported"); ``sb.filesystem`` (read_text / copy_to_local / write_text) is used instead.
    """
    import hashlib
    import json as _json

    dest.mkdir(parents=True, exist_ok=True)
    report = {"ok": [], "bad": []}
    fs = sb.filesystem
    try:
        index = _json.loads(fs.read_text(f"{work}/packed/index.json"))
        (dest / "index.json").write_text(_json.dumps(index, indent=1, sort_keys=True))
        for name, meta in sorted(index["files"].items()):
            fs.copy_to_local(f"{work}/packed/{name}", str(dest / name))
            touch()
            digest = hashlib.sha256((dest / name).read_bytes()).hexdigest()
            (report["ok"] if digest == meta["sha256"] else report["bad"]).append(name)
    except Exception as exc:  # noqa: BLE001 - report, never leave the container waiting
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            fs.write_text("pulled\n", PULLED_FLAG)
        except Exception as exc:  # noqa: BLE001
            report["release_error"] = repr(exc)
    (dest / "pull_report.json").write_text(_json.dumps(report, indent=1))
    log.write(f"[local] pulled packed states: {len(report['ok'])} ok, {len(report['bad'])} bad, "
              f"error={report.get('error')}\n")
    log.flush()
    return report


def _run(app, image, profile, p, out):  # pragma: no cover - needs Modal
    import base64
    import threading

    import modal

    # 2>&1: one stream, mirrored line by line into <out>/container.log while it runs.
    sb = modal.Sandbox.create("bash", "-c", container_script(profile), app=app, image=image,
                              gpu=p["gpu"], cpu=8.0, memory=65536, timeout=p["timeout"])
    (out / "resources.txt").open("a").write(
        f"{app.app_id} {sb.object_id} {profile} {time.strftime('%FT%TZ', time.gmtime())}\n")
    last = [time.monotonic()]
    stop = threading.Event()

    def silence_guard():
        while not stop.wait(30):
            if time.monotonic() - last[0] > LOCAL_SILENCE_MINUTES * 60:
                (out / "local_silence_kill.txt").write_text(time.strftime("%FT%TZ", time.gmtime()) + "\n")
                sb.terminate()
                return

    threading.Thread(target=silence_guard, daemon=True).start()
    chunks = []
    with open(out / "container.log", "a", encoding="utf-8") as log:
        for line in sb.stdout:
            last[0] = time.monotonic()
            if line.strip() == PACKED_MARKER:
                log.write(line)
                log.flush()
                pull_packed(sb, out / "work" / "packed", touch=lambda: last.__setitem__(0, time.monotonic()),
                            log=log)
                continue
            chunks.append(line)
            if "=== EVIDENCE_B64 ===" not in line and len(line) < 100000:
                log.write(line)
                log.flush()
    stop.set()
    sb.wait(raise_on_termination=False)
    stdout = "".join(chunks)
    if "=== EVIDENCE_B64 ===" in stdout:
        payload = stdout.split("=== EVIDENCE_B64 ===", 1)[1].strip().split()[0]
        (out / "evidence.tgz").write_bytes(base64.b64decode(payload))
    (out / "returncode.txt").write_text(str(sb.returncode))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
