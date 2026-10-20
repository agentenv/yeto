"""S25 T-G1: stage-1 mini rehearsal on Modal (examples/stage1/glm5next-mini-modal.yaml).

The containers run the commands ``yeto-stage1 plan`` generates for that plan
(computed inside each container, so ``--source-sha256`` matches the mounted
tree): the Miles trainer-only island with the MXFP4 re-quantizer side process,
and the engine sampler island. Only the nodes (Modal containers instead of SSH
hosts) and the store (Modal Volume through the yeto store hook instead of OSS)
differ from the PPU plan.

    modal run tools/probes/s25_g1/modal_app.py::prep            # CPU: checkpoints + data
    modal run --detach tools/probes/s25_g1/modal_app.py::smoke  # 1 x H100: SGLang + post-load checks
    modal run --detach tools/probes/s25_g1/modal_app.py::launch # 3 rounds, cut after 2, resume

``smoke`` and ``launch`` spawn and return; the local client may exit
(never kill a ``.remote()`` caller: memory modal-detach-kill-cancels).
Raw logs go to the store Volume under ``logs/<tag>/``.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve()
REPO = HERE.parents[3] if len(HERE.parents) > 3 else Path("/")  # in the container the module sits at /root
WORKDIR = "/root/sky_workdir"
PLAN = "examples/stage1/glm5next-mini-modal.yaml"
IMAGE = ("ghcr.io/michaellchung/yeto-miles-ports@sha256:"
         "5b9dc2a9e6e315d73dabfbc6faae428b59cc01aabec6189645a5c8c80d6e3721")
MILES_COMMIT = "1618b5c38db80e17ec928cfa47fe832cb18565fc"  # s25-distopt
MODELS_VOLUME, STORE_VOLUME = "yeto-s25-g1-models", "yeto-s25-g1-store"
GPU_NAME = "H100"

app = modal.App("yeto-s25-g1-stage1mini")
models = modal.Volume.from_name(MODELS_VOLUME, create_if_missing=True, version=2)
store = modal.Volume.from_name(STORE_VOLUME, create_if_missing=True, version=2)
VOLUMES = {"/data/models": models, "/data/store": store}

_PATCH_SGLANG = (
    "import importlib.util, sys; sys.path.insert(0, '/tmp/yeto-slr');"
    "import sglang_load_report as r;"
    "root = importlib.util.find_spec('sglang').submodule_search_locations[0];"
    "f = root + '/srt/models/glm5_next.py'; print('patched', f, r.patch_model_file(f))"
)
image = (
    modal.Image.from_registry(IMAGE)
    .add_local_file(str(REPO / ".g1-build/miles-s25-distopt.bundle"), "/tmp/miles.bundle", copy=True)
    .add_local_file(str(REPO / "yeto/rl/adapters/sampler_engines/sglang_load_report.py"),
                    "/tmp/yeto-slr/sglang_load_report.py", copy=True)
    .run_commands(
        "git -C /root/miles fetch /tmp/miles.bundle s25-distopt",
        f"git -C /root/miles checkout --detach {MILES_COMMIT}",
        "git -C /root/miles status --porcelain --untracked-files=all | head -5",
        f"python3 -c \"{_PATCH_SGLANG}\"",
    )
    # not PYTHONPATH: Modal's own runtime path must stay intact; subprocesses get it (_env)
    .env({"HOME": "/root", "PYTHONUNBUFFERED": "1", "HF_HOME": "/data/models/hf"})
    .add_local_dir(str(REPO), WORKDIR, copy=False,
                   ignore=[".git", ".g1-build", ".venv", ".claude", "syncer/target", "**/__pycache__",
                           "**/*.pyc", "evidence*", "openspec", "data"])
)


def _hf_secret() -> modal.Secret:
    token = Path("~/.cache/huggingface/token").expanduser()
    return modal.Secret.from_dict({"HF_TOKEN": token.read_text().strip()} if token.is_file() else {})


LOCAL_LOGS = Path("/tmp/g1-logs")  # container disk: an open file on the Volume blocks every reload


def _log_dir(tag: str) -> Path:
    path = LOCAL_LOGS / tag
    path.mkdir(parents=True, exist_ok=True)
    return path


def _commit(tag: str | None = None) -> None:
    """Copy the local logs of ``tag`` to the store Volume (closed files only), then commit."""
    try:
        if tag is not None:
            shutil.copytree(LOCAL_LOGS / tag, Path("/data/store/logs") / tag, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("*.sock", "*.pid"))
        store.commit()
    except Exception as error:  # noqa: BLE001 - evidence copy only
        print(f"[g1] log mirror/commit failed: {error}", flush=True)


def _mirror_every(tag: str, seconds: float = 180.0) -> None:
    import threading

    def loop() -> None:
        while True:
            time.sleep(seconds)
            _commit(tag)

    threading.Thread(target=loop, daemon=True).start()


def _sh(cmd: str, log: Path | None = None, env: dict | None = None, check: bool = True) -> str:
    print(f"[g1] $ {cmd}", flush=True)
    r = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, env=env or _env({}), cwd=WORKDIR)
    out = r.stdout + r.stderr
    if log is not None:
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"$ {cmd}\n{out}\n")
    if check and r.returncode:
        raise RuntimeError(f"{cmd} -> {r.returncode}: {out[-2000:]}")
    return out


def _gpu_facts(log_dir: Path, want: int) -> dict:
    log = log_dir / "gpu.txt"
    out = _sh("nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader", log)
    names = [line.split(",")[1].strip() for line in out.strip().splitlines()]
    if len(names) != want or not all(GPU_NAME in n for n in names):
        raise RuntimeError(f"GPU assert failed: want {want} x {GPU_NAME}, got {names}")
    _sh("nvidia-smi; free -g; nproc; df -h /data/models /data/store /root", log, check=False)
    return {"gpus": names}


def _plan(log: Path) -> dict:
    out = _sh(f"python3 -m yeto.rl.stage1 plan {PLAN} --json 2>/dev/null; true", None, check=False)
    start = out.index("{")
    plan = json.loads(out[start: out.rindex("}") + 1])
    (log / "plan.json").write_text(json.dumps(plan, indent=1))
    if plan["errors"] or plan["gaps"]:
        raise RuntimeError(f"plan errors {plan['errors']} gaps {plan['gaps']}")
    return plan


def _nvml_sampler(path: Path) -> subprocess.Popen:
    return subprocess.Popen(
        ["bash", "-c", "nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw "
                       f"--format=csv,noheader -l 5 >> {shlex.quote(str(path))}"])


def _env(extra: dict) -> dict:
    path = ":".join(p for p in (WORKDIR, os.environ.get("PYTHONPATH", "")) if p)
    # The island's Python cannot import the Modal client; the Volume store commits and
    # reloads through this interpreter (yeto.cloud.modal_ckpt_store, as yeto.modal_runner does).
    runner = {"YETO_MODAL_PYTHON": sys.executable,
              "YETO_MODAL_SYSPATH": os.pathsep.join(p for p in sys.path if p)}
    return {**os.environ, "PYTHONPATH": path, **runner, **{k: str(v) for k, v in extra.items()}}


# --------------------------------------------------------------------- prep (CPU)
@app.function(image=image, volumes={"/data/models": models}, cpu=8, memory=65536, timeout=7200,
              secrets=[_hf_secret()])
def prep() -> str:
    out = subprocess.run(["python3", f"{WORKDIR}/tools/probes/s25_g1/make_mini.py", "/data/models"],
                         capture_output=True, text=True, cwd=WORKDIR, env=_env({}))
    text = out.stdout[-6000:] + "\nSTDERR:\n" + out.stderr[-4000:]
    Path("/data/models/prep.log").write_text(text)
    models.commit()
    return text


@app.local_entrypoint()
def prep_main() -> None:
    print(prep.remote())


# --------------------------------------------------------------------- sampler smoke
@app.function(image=image, volumes=VOLUMES, gpu=f"{GPU_NAME}!:1", cpu=8, memory=65536, timeout=2400)
def smoke_run(tag: str) -> None:
    """Start SGLang on round 0 through the adapter and run the post-load checks once."""
    log = _log_dir(tag)
    facts = _gpu_facts(log, 1)
    nvml = _nvml_sampler(log / "nvml.csv")
    try:
        plan = _plan(log)
        run = plan["sampler"]["engines"][0]
        os.environ.update({k: str(v) for k, v in run["env"].items()})
        sys.path.insert(0, WORKDIR)
        from yeto.rl.adapters.sampler_engines.from_spec import build
        from yeto.rl.engine.sampler_checks import PostLoadChecks

        engine = build(run["spec"]["engine"])
        checks = PostLoadChecks(**run["spec"]["checks"])
        began = time.time()
        report = engine.start(None, 0)
        started = time.time() - began
        results = checks.run(engine, expected_experts=int(run["spec"]["checks"]["expected_expert_matrices"]))
        from yeto.rl.engine.sampler_engine import SamplingParams

        sample = engine.generate(["Question: What is 2 + 3?\nAnswer:"],
                                 SamplingParams(max_tokens=32, temperature=1.0, n=2, seed=1).check())
        from yeto.rl.adapters.sampler_engines.sglang_load_report import read_reports

        record = {**facts, "start_s": round(started, 1), "load_report": report.detail, "checks": results,
                  "hook_report": read_reports(run["spec"]["engine"]["options"]["load_report_dir"]),
                  "sample": [[c.text for c in g] for g in sample]}
        engine.stop()
    except Exception as error:  # noqa: BLE001 - the record is the evidence
        record = {**facts, "error": f"{type(error).__name__}: {error}"}
    finally:
        nvml.terminate()
    (log / "smoke.json").write_text(json.dumps(record, indent=1, default=str))
    print(json.dumps(record, default=str)[:4000], flush=True)
    _commit(tag)


@app.local_entrypoint()
def smoke(tag: str = "smoke-1") -> None:
    call = smoke_run.spawn(tag)
    print(f"spawned smoke_run {call.object_id}")


# --------------------------------------------------------------------- the islands
@app.function(image=image, volumes=VOLUMES, gpu=f"{GPU_NAME}!:1", cpu=8, memory=65536, timeout=7200)
def sampler_island(tag: str) -> int:
    log = _log_dir(tag)
    _gpu_facts(log, 1)
    nvml = _nvml_sampler(log / "nvml.csv")
    _mirror_every(tag)
    plan = _plan(log)
    run = plan["sampler"]["engines"][0]
    spec_path = Path(run["spec_path"])
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(run["spec"], indent=1))
    began = time.time()
    with open(log / "sampler.log", "w", encoding="utf-8") as f:
        rc = subprocess.run(run["argv"], env=_env(run["env"]), cwd=WORKDIR, stdout=f, stderr=subprocess.STDOUT).returncode
    nvml.terminate()
    (log / "rc.json").write_text(json.dumps({"rc": rc, "wall_s": round(time.time() - began, 1)}))
    state = Path(run["spec"]["event_tape"]).parent
    if state.is_dir():
        shutil.copytree(state, log / "state", dirs_exist_ok=True)
    _commit(tag)
    return rc


@app.function(image=image, volumes=VOLUMES, gpu=f"{GPU_NAME}!:2", cpu=12, memory=163840, timeout=3600)
def trainer_island(tag: str, stop_after_rounds: int | None = None, convert_only: bool = False) -> int:
    log = _log_dir(tag)
    _gpu_facts(log, 2)
    nvml = _nvml_sampler(log / "nvml.csv")
    _mirror_every(tag)
    plan = _plan(log)
    trainer = plan["trainer"]
    env = _env(trainer["env"])
    island = list(trainer["miles"]["island_argv"])
    if stop_after_rounds is not None:
        island += ["--rl-stop-after-rounds", str(int(stop_after_rounds))]
    (log / "argv.json").write_text(json.dumps({"island": island, "requant": plan["requant"]["loop_argv"],
                                               "env": trainer["env"]}, indent=1))
    Path("/data/yeto", plan["run_id"]).mkdir(parents=True, exist_ok=True)
    marker = Path(str(trainer["torch_dist"])) / "latest_checkpointed_iteration.txt"
    if trainer.get("convert_argv") and not (marker.is_file() and marker.read_text().strip() == "release"):
        began_convert = time.time()
        with open(log / "convert.log", "w", encoding="utf-8") as f:
            path = f"{trainer['convert_env']['PYTHONPATH']}:{env['PYTHONPATH']}"
            crc = subprocess.run(trainer["convert_argv"], env={**env, "PYTHONPATH": path},
                                 cwd=WORKDIR, stdout=f, stderr=subprocess.STDOUT).returncode
        (log / "convert.json").write_text(json.dumps({"rc": crc, "wall_s": round(time.time() - began_convert, 1)}))
        models.commit()  # trainer B (a new container) loads the same torch_dist
        if crc != 0:
            _commit(tag)
            return 90
    if convert_only:
        _sh(f"ls -la {shlex.quote(str(trainer['torch_dist']))}; du -sh {shlex.quote(str(trainer['torch_dist']))}",
            log / "convert-out.txt", check=False)
        nvml.terminate()
        _commit(tag)
        return 0
    _sh("ray start --head --port=6379 --node-ip-address=127.0.0.1 --num-gpus=2 --disable-usage-stats",
        log / "ray.log", env=env)
    began = time.time()
    with open(log / "requant.log", "w", encoding="utf-8") as rq, open(log / "trainer.log", "w", encoding="utf-8") as f:
        requant = subprocess.Popen(plan["requant"]["loop_argv"], env=env, cwd=WORKDIR, stdout=rq,
                                   stderr=subprocess.STDOUT)
        # as the stage-1 node script does: the island forwards its env to the Ray actors
        rc = subprocess.run(island, env={**env, "RAY_ADDRESS": "127.0.0.1:6379"}, cwd=WORKDIR, stdout=f,
                            stderr=subprocess.STDOUT).returncode
        requant.terminate()
        try:
            requant.wait(timeout=60)
        except subprocess.TimeoutExpired:
            requant.kill()
        final = subprocess.run(plan["requant"]["argv"], env=env, cwd=WORKDIR, stdout=rq,
                               stderr=subprocess.STDOUT).returncode
    nvml.terminate()
    (log / "rc.json").write_text(json.dumps({"rc": rc, "final_requant_rc": final,
                                             "wall_s": round(time.time() - began, 1)}))
    shutil.copytree(f"/data/yeto/{plan['run_id']}", log / "state", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("*.pt", "*.safetensors"))
    _sh("cp -r /tmp/ray/session_latest/logs " + shlex.quote(str(log / "ray-logs")) + " || true", check=False)
    _commit(tag)
    return rc


@app.function(image=image, volumes=VOLUMES, cpu=1, memory=1024, timeout=3 * 3600)
def orchestrate(tag: str, cut_after: int = 2) -> dict:
    """Sampler for the whole run; trainer A stops after ``cut_after`` rounds (cut),
    trainer B resumes in a new container. A trainer that fails cancels the sampler;
    a sampler that fails cancels the running trainer (no card waits for nothing)."""
    sampler = sampler_island.spawn(f"{tag}/sampler")
    sampler_rc: object = None

    def sampler_done() -> bool:
        nonlocal sampler_rc
        try:
            sampler_rc = sampler.get(timeout=0)
            return True
        except TimeoutError:
            return False
        except Exception as error:  # noqa: BLE001 - the sampler call failed
            sampler_rc = f"error: {error}"
            return True

    def run_trainer(sub: str, stop: int | None) -> object:
        call = trainer_island.spawn(f"{tag}/{sub}", stop)
        while True:
            try:
                return call.get(timeout=30)
            except TimeoutError:
                pass
            except Exception as error:  # noqa: BLE001
                return f"error: {error}"
            if sampler_done() and sampler_rc != 0:
                call.cancel()
                return "cancelled: sampler ended with " + str(sampler_rc)

    first = run_trainer("trainer-a", cut_after)
    second = run_trainer("trainer-b", None) if first == 0 else None
    if first != 0 or second != 0:
        if not sampler_done():
            sampler.cancel()
            sampler_rc = "cancelled"
    else:
        while not sampler_done():
            time.sleep(30)
    result = {"trainer_a": first, "trainer_b": second, "sampler": sampler_rc}
    print(json.dumps(result, default=str), flush=True)
    out = Path(f"/data/store/logs/{tag}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "orchestrate.json").write_text(json.dumps(result, default=str))
    _commit()
    return result


@app.local_entrypoint()
def convert(tag: str = "convert-1") -> None:
    call = trainer_island.spawn(tag, None, True)
    print(f"spawned convert {call.object_id}")


@app.local_entrypoint()
def launch(tag: str = "run-1", cut_after: int = 2) -> None:
    call = orchestrate.spawn(tag, cut_after)
    print(f"spawned orchestrate {call.object_id}")


@app.function(image=image, volumes={"/data/models": models}, cpu=2, memory=8192, timeout=900)
def cpu_shell(cmd: str) -> str:
    """CPU-only inspection of the image (no GPU)."""
    r = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, cwd=WORKDIR, env=_env({}))
    return r.stdout[-20000:] + "\nSTDERR:\n" + r.stderr[-4000:]


@app.local_entrypoint()
def shell(cmd: str) -> None:
    print(cpu_shell.remote(cmd))
