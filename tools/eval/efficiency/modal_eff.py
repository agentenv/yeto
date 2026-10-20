"""Modal harness of the efficiency arms that run in one container: the four bare
arms and yeto+verl (yeto+Miles runs through the yeto launcher, ``miles_local.sh``).

One GPU container per run, ``<gpu>:<cards>``, in the framework's image (the
same image the yeto arm of that framework uses). The container:

1. checks every visible GPU is the asked type (``modal_verl._gpu``);
2. samples NVML every second into ``nvml.csv`` (index, util, memory, power);
3. bare: writes the spec, ``eff_reward.py`` and ``eff_prep.py`` under
   ``/root/eff`` and runs the arm's bash script (prep, then the framework entry);
   yeto+verl: starts the island Ray and runs ``island_entry`` (as T-EVAL E1);
4. records ``meta.json`` (start/end unix time, rc, GPU names, argv/script) and
   copies its outputs into the results Volume ``<run>/arm/`` before returning.

The local side spawns the call in a detached app, waits, pulls ``<run>/`` into
``<run dir>/raw/`` and stops only its own app (``yeto-eff-<run>``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .. import modal_verl as mv
from .spec import EFF_DIR, OUT_DIR

HERE = Path(__file__).resolve().parent
APP_PREFIX = "yeto-eff-"
NVML_QUERY = "timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw"
NVML_INTERVAL_MS = 1000


# ====================================================================== in-container
def _nvml(path: Path) -> subprocess.Popen:
    return subprocess.Popen(["bash", "-c", f"nvidia-smi --query-gpu={NVML_QUERY} --format=csv "
                             f"-lms {NVML_INTERVAL_MS} >> {path}"])


def _run_logged(argv: list[str], log: Path, timeout: float, **kwargs) -> int:
    """Run ``argv``; every output line is written to ``log`` behind ``<unix time> ``
    (the step end times the parsers read). The process is killed at ``timeout``."""
    deadline = time.time() + timeout
    with log.open("a") as fh, subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                               errors="replace", **kwargs) as proc:
        for line in proc.stdout:
            fh.write(f"{time.time():.3f} {line}")
            fh.flush()
            if time.time() > deadline:
                proc.kill()
                fh.write(f"{time.time():.3f} [eff harness] killed at the {timeout:.0f} s timeout\n")
                break
        return proc.wait()


def arm(run: str, plan: dict) -> dict:
    """One run of one arm (see the module doc)."""
    out = Path(mv.RESULTS) / run / "arm"
    out.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {"arm": plan["arm"], "container": os.environ.get("MODAL_TASK_ID"),
                            "cards": plan["cards"]["total"]}
    meta["gpu"] = mv._gpu(out, {"gpu": plan["gpu"]})
    nvml = _nvml(out / "nvml.csv")
    meta["start_unix"] = time.time()
    copies = {"eff-out": Path(OUT_DIR), "yeto-output": Path("/root/yeto-output")}
    budget = plan["spec"]["timeout_s"] - 300  # leaves time to copy and commit
    log = out / "main.log"
    try:
        if plan["system"] == "bare":
            eff = Path(EFF_DIR)
            eff.mkdir(parents=True, exist_ok=True)
            for name, text in plan["files"].items():
                (eff / name).write_text(text)
            (eff / "spec.json").write_text(json.dumps(plan["spec"]))
            (out / "script.sh").write_text(plan["script"])
            meta["rc"] = _run_logged(["bash", str(out / "script.sh")], log, budget)
        else:
            sys.path.insert(0, mv.WORKDIR)
            env = mv._island_env({"gpu": plan["gpu"]}, 0)
            ray = str(Path(mv.VERL_PY).with_name("ray"))
            _run_logged([ray, "start", "--head", "--node-ip-address=127.0.0.1", "--port=6379", "--num-cpus=32",
                         f"--num-gpus={plan['cards']['total']}", "--include-dashboard=false",
                         "--temp-dir=/root/eval-ray"], log, 300, env=env)
            meta["argv"] = [mv.VERL_PY, *plan["argv"]]
            meta["rc"] = _run_logged(meta["argv"], log, budget, env=env, cwd="/root")
    except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
        meta["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        meta["end_unix"] = time.time()
        nvml.terminate()
        mv._finish(out, meta, meta["start_unix"], copies)
    return meta


# ====================================================================== app
def build_app(run: str, plan: dict):
    import modal

    sys.path.insert(0, str(mv.REPO))
    from modal._vendor import cloudpickle

    # the container functions travel by value: `tools` is not importable in the images
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    cloudpickle.register_pickle_by_value(mv)
    from yeto.rl.adapters.miles.pins import MILES_NEXT_IMAGE
    from yeto.rl.adapters.verl.image import modal_image

    if plan["image"] == "verl":
        image = modal_image(modal).env({"PYTHONUNBUFFERED": "1"})
        if plan["system"] == "yeto":
            image = image.add_local_dir(str(mv.REPO), mv.WORKDIR, copy=False, ignore=MOUNT_IGNORE)
    else:
        image = (modal.Image.from_registry(MILES_NEXT_IMAGE.removeprefix("docker:"))
                 .env({"HOME": "/root", "PYTHONUNBUFFERED": "1"}))
    app = modal.App(APP_PREFIX + run)
    results = modal.Volume.from_name(mv.RESULTS_VOLUME, create_if_missing=True, version=2)
    gpu = f"{plan['gpu']}:{plan['cards']['total']}"
    fn = app.function(image=image, gpu=gpu, cpu=8.0 * plan["cards"]["total"],
                      memory=49152 * plan["cards"]["total"], timeout=plan["spec"]["timeout_s"], retries=0,
                      volumes={mv.RESULTS: results}, single_use_containers=True, name="arm", serialized=True)(arm)
    return app, fn


MOUNT_IGNORE = [".git", ".venv", ".claude", "**/__pycache__", "**/*.pyc", "s1-runs", "syncer/target",
                "evidence*", "openspec", "docs", "tutorial", "data"]


def bare_files() -> dict[str, str]:
    return {name: (HERE / name).read_text() for name in ("eff_reward.py", "eff_prep.py")}


def run(plan: dict, run_dir: Path) -> dict[str, Any]:
    """Run one arm end to end (spawn, wait, pull, stop own app)."""
    import modal

    run_id = plan["run_id"]
    raw = run_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    log = run_dir / "driver.log"
    if plan["system"] == "bare":
        plan = {**plan, "files": bare_files()}
    app, fn = build_app(run_id, plan)
    began = time.time()
    with modal.enable_output(), app.run(detach=True):  # a killed driver never cancels a spawned call
        app_id = app.app_id
        (run_dir / "app_id.txt").write_text(f"{app_id}\n")
        call = fn.spawn(run_id, plan)
        mv._log(log, f"spawned arm {plan['arm']} app {APP_PREFIX + run_id}")
        results = mv._wait_calls({"arm": call}, began + plan["spec"]["timeout_s"] + 600, log)
    pulled = mv._modal_cli("volume", "get", "--force", mv.RESULTS_VOLUME, run_id, str(raw), timeout=1800)
    stopped = mv._modal_cli("app", "stop", "-y", app_id, timeout=120)
    summary = {"calls": results, "wall_s": round(time.time() - began, 1), "pull_rc": pulled.returncode,
               "stop_rc": stopped.returncode}
    (run_dir / "driver.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


# ====================================================================== yeto+Miles (launcher)
def run_miles_launcher(plan: dict, run_dir: Path, *, dry_run: bool = False) -> dict[str, Any]:
    """The yeto+Miles arm: ``miles_local.sh`` with the plan's launcher arguments
    (the T-EVAL Miles path; it pulls launch log, tapes and NVML into ``raw/``)."""
    from .. import miles

    raw = run_dir / "raw"
    env = dict(os.environ, P=plan["run_id"], R=str(raw), REPO=str(miles.REPO), PY=miles.PY, SKY=miles.SKY,
               STOREVOL=miles.STORE_VOLUME, TAPEVOL=miles.TAPE_VOLUME, HARD=str(plan["spec"]["timeout_s"]),
               HOME="/home/michael", ARGS=plan["launcher_args"], ISLANDS="1")
    if dry_run:
        env["DRY"] = "1"
    log = run_dir / ("dry-run.log" if dry_run else "driver.log")
    with log.open("w") as fh:
        rc = subprocess.run(["bash", str(miles.HERE / "miles_local.sh")], env=env, stdout=fh,
                            stderr=subprocess.STDOUT, cwd="/tmp").returncode
    return {"script": "miles_local.sh", "rc": rc, "log": str(log)}
