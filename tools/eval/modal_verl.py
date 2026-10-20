"""verl cells: a direct Modal harness (no yeto launcher; the S25 C2/C3V/C4 path).

Container functions (one GPU each unless noted), all writing their raw data to
the results Volume under ``<run>/<component>/`` before they return:

* ``syncer`` (CPU): the elastic yeto-syncer built from this checkout's
  ``syncer/``, reachable through a Modal TCP tunnel (address in the control Dict);
* ``sampler``: the neutral engine sampler island (SGLang server adapter, TP1) in
  the Miles ports image; optionally reclaimed once by the local driver
  (``modal container stop``) after it wrote round ``reclaim_after_round``;
* ``trainer``: ``yeto.rl.adapters.verl.island_entry --rl-sample-role trainer``
  (a buffer trainer; with ``learners`` >= 2 synced through the syncer, lane m);
* ``island``: one plain verl island (rollout + trainer, ``--rl-single-island-no-sync``);
  phase ``a`` is commanded (``run 2`` then ``stop STOPPED``: SCHEDULED cut into
  the store), ``b`` resumes from the store, ``c`` is the continuous control;
* ``collect`` (CPU): copies the store's records (no weights) into the results.

The local side (:func:`run`) checks the preflight, spawns the calls in a
detached app (a killed client never cancels a spawned call), triggers the
reclaim, waits, collects, pulls ``<run>/`` from the results Volume into
``<run dir>/raw/`` and stops only its own app (``yeto-eval-<run>``).
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
APP_PREFIX = "yeto-eval-"
STORE_VOLUME = "yeto-eval-store"
RESULTS_VOLUME = "yeto-eval-results"
CONTROL = "yeto-eval-control"
STORE = "/mnt/store"
RESULTS = "/mnt/results"
WORKDIR = "/root/sky_workdir"
VERL_PY = "/workspace/verl/.venv/bin/python"
MODEL = "Qwen/Qwen3-0.6B"
MODEL_REV = "c1899de289a04d12100db370d81485cdf75e47ca"
DATA = "zhuzilin/gsm8k"
DATA_REV = "0cbd9f31d91ac21a7613dcbc7fef992adac459ae"
SYNCER_PORT = 29400
GPU_NAMES = {"A10G": "NVIDIA A10", "L4": "L4", "H100!": "H100"}  # Modal A10G reports "NVIDIA A10"
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".distcp")
MAX_COLLECT_BYTES = 32 << 20
# runs in the verl venv: the head's commands for trainer incarnation 0 (run N rounds,
# then stop STOPPED), printed island environment as JSON (S25 C4 modal_c4.py)
COMMAND_WRITER = (
    "import json, sys\n"
    "from yeto.rl.elastic_client import LeaveReason\n"
    "from yeto.rl.engine import trainer_remote as t\n"
    "from yeto.rl.engine.resume import store_for\n"
    "w = t.CommandWriter(t.LocalStoreFiles(store_for(sys.argv[1])), 0)\n"
    "w.send(t.RUN, rounds=int(sys.argv[2]))\n"
    "w.send(t.STOP, reason=LeaveReason.STOPPED)\n"
    "print(json.dumps(t.island_env(0, None, poll_s=2.0)))\n")


# ====================================================================== in-container helpers
def _gpu(out: Path, cfg: dict) -> list[str]:
    names = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=60).stdout.split("\n")
    names = [n.strip() for n in names if n.strip()]
    (out / "nvidia-smi.txt").write_text(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    want = GPU_NAMES[cfg["gpu"]]
    if not names or not all(want in n for n in names):
        raise RuntimeError(f"asked for {want}, got {names}")
    return names


def _nvml(path: Path) -> subprocess.Popen:
    return subprocess.Popen(["bash", "-c", "nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,"
                             f"power.draw --format=csv -l 5 >> {path}"])


def _commit_results() -> None:
    import modal

    modal.Volume.from_name(RESULTS_VOLUME).commit()


def _store_env() -> dict[str, str]:
    from yeto.cloud.modal_ckpt_store import MODAL_VOLUME_ENV, STORE_IMPL
    from yeto.rl.engine.resume import STORE_IMPL_ENV

    return {STORE_IMPL_ENV: STORE_IMPL, MODAL_VOLUME_ENV: STORE_VOLUME,
            # the islands' interpreters lack the Modal client; commit/reload run here
            "YETO_MODAL_PYTHON": sys.executable, "YETO_MODAL_SYSPATH": ":".join(p for p in sys.path if p)}


def _finish(out: Path, meta: dict, began: float, copies: dict[str, Path]) -> None:
    meta["seconds"] = round(time.time() - began, 1)
    for name, src in copies.items():
        if src.is_dir():
            shutil.copytree(src, out / name, dirs_exist_ok=True)
        elif src.is_file():
            shutil.copy2(src, out / name)
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    _commit_results()


def _incarnation(control, key: str) -> int:
    """0 for the first container of an input, 1 for the next (Modal re-runs an
    input in a new container after a reclaim)."""
    n = int(control.get(key, -1)) + 1
    control[key] = n
    return n


# ---------------------------------------------------------------------- syncer
def syncer(run: str, cfg: dict) -> dict:
    import modal

    out = Path(RESULTS) / run / "syncer"
    out.mkdir(parents=True, exist_ok=True)
    control = modal.Dict.from_name(CONTROL, create_if_missing=True)
    began = time.time()
    argv = ["/opt/yeto-syncer", "--port", str(SYNCER_PORT), "--learners", str(cfg["learners"]),
            "--total-steps", str(cfg["rounds"]), "--event-tape", "/tmp/syncer-tape.jsonl",
            "--island-scheduling-mode", "elastic", "--quorum-theta", "0.75", "--carry-gamma", "0.5",
            "--soft-deadline-s", "600", "--q-min", "1", "--max-carry-lag", "2", "--syncer-epoch", "0",
            "--island-lease-s", "150.0", "--final-grace-s", "300"]
    meta: dict[str, Any] = {"argv": argv, "container": os.environ.get("MODAL_TASK_ID")}
    try:
        with modal.forward(SYNCER_PORT, unencrypted=True) as tunnel, (out / "syncer.log").open("w") as log:
            host, port = tunnel.tcp_socket
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
            time.sleep(2.0)
            control[f"{run}/syncer"] = meta["address"] = f"{host}:{port}"
            meta["rc"] = proc.wait(timeout=cfg["timeout_s"] - 300)
    except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
        meta["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _finish(out, meta, began, {"syncer-tape.jsonl": Path("/tmp/syncer-tape.jsonl")})
    return meta


# ---------------------------------------------------------------------- sampler
def sampler(run: str, cfg: dict) -> dict:
    import modal

    sys.path.insert(0, WORKDIR)
    from yeto.cloud.modal_eval_island import _image_paths

    _image_paths()
    control = modal.Dict.from_name(CONTROL, create_if_missing=True)
    incarnation = _incarnation(control, f"{run}/sampler/incarnation")
    out = Path(RESULTS) / run / f"sampler-{incarnation}"
    out.mkdir(parents=True, exist_ok=True)
    control[f"{run}/sampler/container"] = os.environ.get("MODAL_TASK_ID")
    names = _gpu(out, cfg)
    nvml = _nvml(out / "nvml.csv")
    os.environ.update(_store_env())
    began = time.time()
    meta: dict[str, Any] = {"gpu": names, "container": os.environ.get("MODAL_TASK_ID"), "cfg": cfg,
                            "incarnation": incarnation}
    from yeto.rl.engine import engine_sampler_entry as entry
    from yeto.rl.engine.sampler_resume import (
        DEFAULT_RECLAIM_MARKER,
        RECLAIM_MARKER_ENV,
        new_incarnation,
        reclaim_requested,
    )

    os.environ[RECLAIM_MARKER_ENV] = DEFAULT_RECLAIM_MARKER  # the reclaim signal writes the marker
    island_incarnation = new_incarnation()
    tape = Path("/tmp/sampler-tape.jsonl")  # the entry writes sampler-tape.<incarnation>.jsonl
    tape = entry.incarnation_tape(tape, island_incarnation)
    try:
        import pandas as pd
        import sglang
        from huggingface_hub import hf_hub_download, snapshot_download
        from transformers import AutoTokenizer

        base = snapshot_download(MODEL, revision=MODEL_REV, local_dir="/tmp/qwen3-0.6b")
        tok = AutoTokenizer.from_pretrained(base)
        frame = pd.read_parquet(hf_hub_download(DATA, "train.parquet", repo_type="dataset", revision=DATA_REV))
        prompts = Path("/tmp/prompts.jsonl")
        with prompts.open("w") as handle:
            for _, row in frame.head(512).iterrows():
                messages = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
                text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
                handle.write(json.dumps({"text": text, "label": str(row["label"])}) + "\n")
        spec = {
            "sampler_id": "eval-sglang-0",
            "engine": {"factory": "yeto.rl.adapters.sampler_engines.from_spec:build", "adapter": "sglang-server",
                       "server": {"base_dir": base, "card_model": names[0], "device_family": "cuda",
                                  "engine_version": sglang.__version__, "precision": "bf16", "quant_scheme": "bf16",
                                  "topology": "tp1", "max_model_len": cfg["max_model_len"], "port": 30000,
                                  "python": sys.executable, "start_timeout_s": 1200.0, "in_place_reload": True,
                                  "extra_args": ["--mem-fraction-static", str(cfg["sglang_mem"])]},
                       "options": {"tp": 1, "served_model_name": "qwen3-0.6b"}},
            "sampling": {"max_tokens": cfg["max_tokens"], "temperature": 1.0, "top_p": 1.0, "top_k": -1,
                         "n": cfg["samples_per_group"]},
            "num_rollouts": cfg["sampler_rounds"], "max_backlog_groups": cfg["max_backlog_groups"],
            "max_prompt_tokens": 512, "base_version": 0, "base_dir": base, "poll_s": 2.0,
            "store": f"{STORE}/{run}",
            "prompts": {"path": str(prompts), "input_key": "text", "label_key": "label",
                        "groups_per_round": cfg["groups_per_round"], "format": "text"},
            "reward": "yeto.rl.gsm8k_reward:score", "reward_kind": "miles",
            "checks": {"expected_expert_matrices": None, "fixed_prompts": []},
            "event_tape": "/tmp/sampler-tape.jsonl",
        }
        (out / "spec.json").write_text(json.dumps(spec, indent=1))
        meta["island_incarnation"] = island_incarnation

        def on_event(event: str, **fields: Any) -> None:
            if event == "rl_sampler_round":  # the local driver's reclaim trigger
                control[f"{run}/sampler/round"] = {"incarnation": incarnation, **{
                    k: fields.get(k) for k in ("rollout_id", "groups")}}

        done_keys = [f"{run}/done/{i}" for i in range(cfg["learners"])]
        deadline = began + cfg["timeout_s"] - 300

        def trainers_done() -> bool:
            return all(control.get(k, False) for k in done_keys) or time.time() > deadline

        # the production entry: resumes after the rounds earlier incarnations wrote,
        # turns the reclaim signal into the marker and leaves PREEMPTED (S25 G1C)
        meta["rounds"] = entry.run(spec, incarnation=island_incarnation, stop=trainers_done, on_event=on_event)
        meta["reclaimed"] = reclaim_requested(os.environ)
    except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
        meta["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        nvml.terminate()
        _finish(out, meta, began, {"sampler-tape.jsonl": tape})
    # reclaimed: stay up until Modal stops the container, so Modal runs the input again
    entry.hold_after_reclaim(hold_s=900.0)
    return meta


# ---------------------------------------------------------------------- trainer
def _ray_start(env: dict, log) -> None:
    subprocess.run(["/workspace/verl/.venv/bin/ray", "start", "--head", "--node-ip-address=127.0.0.1",
                    "--port=6379", "--num-cpus=32", "--num-gpus=1", "--include-dashboard=false",
                    "--temp-dir=/root/eval-ray"], env=env, check=True, timeout=300, stdout=log, stderr=log)


def _island_env(cfg: dict, learner_id: int) -> dict[str, str]:
    from yeto.hw.catalog import COMPAT_GROUP_ENV, compat_group

    # the launcher's island environment (rl_island_task): compat group and learner id
    return dict(os.environ, PYTHONPATH=WORKDIR, HF_HUB_DISABLE_TELEMETRY="1", NCCL_DEBUG="WARN",
                RAY_ADDRESS="127.0.0.1:6379", LEARNER_ID=str(learner_id),
                **{COMPAT_GROUP_ENV: compat_group(cfg["gpu"].rstrip("!"))})


def _common_island_argv(cfg: dict, learner_id: int, tape: Path) -> list[str]:
    return [VERL_PY, "-m", "yeto.rl.adapters.verl.island_entry", "--model", MODEL, "--model-revision", MODEL_REV,
            "--data", DATA, "--data-revision", DATA_REV, "--learner-id", str(learner_id),
            "--reward-function", "gsm8k_reward:score", "--global-rounds", str(cfg["rounds"]),
            "--groups-per-round", str(cfg["groups_per_round"]), "--samples-per-group", str(cfg["samples_per_group"]),
            "--rollout-max-response-len", str(cfg["max_tokens"]), "--event-tape", str(tape),
            "--inner-lr", str(cfg["lr"]), "--lora-r", str(cfg["lora_r"]), "--seq-len", str(cfg["seq_len"]),
            "--seed", str(cfg["seed"]), "--apply-chat-template-kwargs", '{"enable_thinking": false}']


def trainer(run: str, cfg: dict, learner_id: int) -> dict:
    import modal

    sys.path.insert(0, WORKDIR)
    out = Path(RESULTS) / run / f"trainer{learner_id}"
    out.mkdir(parents=True, exist_ok=True)
    names = _gpu(out, cfg)
    nvml = _nvml(out / "nvml.csv")
    control = modal.Dict.from_name(CONTROL, create_if_missing=True)
    began = time.time()
    meta: dict[str, Any] = {"gpu": names, "container": os.environ.get("MODAL_TASK_ID"), "cfg": cfg,
                            "learner_id": learner_id}
    from yeto.rl.engine.algorithm import AlgorithmSpec, CorrectionSpec, ExecutionSpec

    spec = AlgorithmSpec(execution=ExecutionSpec(max_policy_staleness=cfg["max_policy_age"]),
                         correction=CorrectionSpec(method="tis", tis_clip=2.0, tis_clip_low=0.0))
    Path("/tmp/algo.json").write_text(json.dumps(spec.to_dict()))
    env = {**_island_env(cfg, learner_id), **_store_env()}
    tape = out / "trainer-tape.jsonl"
    try:
        argv = _common_island_argv(cfg, learner_id, tape) + [
            "--rl-max-policy-age", str(cfg["max_policy_age"]), "--rl-algorithm-spec", "/tmp/algo.json",
            "--rl-sample-role", "trainer", "--rl-sample-wait-s", "1200"]
        if cfg["learners"] > 1:
            address = None
            for _ in range(600):  # the syncer container publishes its tunnel address
                address = control.get(f"{run}/syncer")
                if address:
                    break
                time.sleep(2.0)
            if not address:
                raise RuntimeError("no syncer address within 1200 s")
            argv += ["--syncer", address, "--rl-island-scheduling", "elastic", "--rl-syncer-epoch", "0",
                     "--rl-island-lease-s", "150.0", "--rl-sample-store", f"{STORE}/{run}",
                     "--rl-sample-lanes", str(cfg["learners"])]
        else:
            argv += ["--rl-single-island-no-sync", "--rl-resume-store", f"{STORE}/{run}"]
        meta["argv"] = argv
        with (out / "island.log").open("w") as log:
            _ray_start(env, log)
            meta["rc"] = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        cwd="/workspace/verl", timeout=cfg["timeout_s"] - 300).returncode
    except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
        meta["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        control[f"{run}/done/{learner_id}"] = True
        nvml.terminate()
        _finish(out, meta, began, {"yeto-output": Path("~/yeto-output").expanduser()})
    return meta


# ---------------------------------------------------------------------- plain island (E1, E5)
def island(run: str, cfg: dict, phase: str) -> dict:
    """One verl island (rollout + trainer); phases as in the module doc."""
    import modal

    sys.path.insert(0, WORKDIR)
    modal.Volume.from_name(STORE_VOLUME).reload()  # see what the previous phase committed
    out = Path(RESULTS) / run / f"island-{phase}"
    out.mkdir(parents=True, exist_ok=True)
    done = out / "meta.json"
    if done.is_file() and json.loads(done.read_text()).get("rc") == 0:
        return {**json.loads(done.read_text()), "reused_after_restart": True}  # never rerun a finished phase
    names = _gpu(out, cfg)
    nvml = _nvml(out / "nvml.csv")
    began = time.time()
    meta: dict[str, Any] = {"gpu": names, "container": os.environ.get("MODAL_TASK_ID"), "cfg": cfg, "phase": phase}
    env = {**_island_env(cfg, 0)}
    store = f"{STORE}/{run}/cuts"
    state = Path(f"/root/state-{phase}")
    try:
        argv = _common_island_argv(cfg, 0, Path("/root/yeto-output/rl-island-0.jsonl")) + [
            "--rl-single-island-no-sync", "--lora-targets", "all-linear"]
        if phase in ("a", "b"):
            argv += ["--rl-resume-store", store, "--rl-resume-state-dir", str(state)]
        if phase == "a":  # the head's commands for incarnation 0, through yeto's own writer
            write = subprocess.run([VERL_PY, "-c", COMMAND_WRITER, store, str(cfg["stop_after_rounds"])],
                                   env=env, capture_output=True, text=True, timeout=300)
            if write.returncode != 0:
                raise RuntimeError(f"command file not written: {write.stderr[-2000:]}")
            modal.Volume.from_name(STORE_VOLUME).commit()
            env.update(json.loads(write.stdout.strip().splitlines()[-1]))
        meta["argv"] = argv
        with (out / "island.log").open("w") as log:
            _ray_start(env, log)
            meta["rc"] = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT, cwd="/root",
                                        timeout=cfg["timeout_s"] - 300).returncode
            subprocess.run(["/workspace/verl/.venv/bin/ray", "stop", "--force"], stdout=log, stderr=log)
        modal.Volume.from_name(STORE_VOLUME).commit()
    except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
        meta["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        nvml.terminate()
        _finish(out, meta, began, {"yeto-output": Path("/root/yeto-output"), "state-dir": state})
    return meta


# ---------------------------------------------------------------------- collect
def collect(run: str) -> dict:
    """Copy the store's records under ``<run>/`` (no weights, no large files)
    into the results Volume ``<run>/store/``."""
    import modal

    modal.Volume.from_name(STORE_VOLUME).reload()
    src, dst = Path(STORE) / run, Path(RESULTS) / run / "store"
    copied, skipped = 0, []
    for path in sorted(src.rglob("*")) if src.is_dir() else []:
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        if path.suffix in WEIGHT_SUFFIXES or path.stat().st_size > MAX_COLLECT_BYTES:
            skipped.append([str(rel), path.stat().st_size])
            continue
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst / rel)
        copied += 1
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "_skipped.json").write_text(json.dumps(skipped, indent=0))
    _commit_results()
    return {"copied": copied, "skipped": len(skipped)}


# ====================================================================== app
def build_app(run: str, cfg: dict):
    import modal

    sys.path.insert(0, str(REPO))
    from modal._vendor import cloudpickle

    # the container functions travel by value: `tools` is not importable in the images
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    from yeto.rl.adapters.miles.pins import MILES_NEXT_IMAGE
    from yeto.rl.adapters.verl.image import modal_image

    ignore = [".git", ".venv", ".claude", "**/__pycache__", "**/*.pyc", "s1-runs", "syncer/target",
              "evidence*", "openspec", "docs", "tutorial", "data"]
    sampler_image = (modal.Image.from_registry(MILES_NEXT_IMAGE.removeprefix("docker:"))
                     .env({"HOME": "/root", "PYTHONUNBUFFERED": "1"})
                     .add_local_dir(str(REPO), WORKDIR, copy=False, ignore=ignore))
    verl_image = (modal_image(modal).env({"PYTHONUNBUFFERED": "1"})
                  .add_local_dir(str(REPO), WORKDIR, copy=False, ignore=ignore))
    cpu_image = modal.Image.debian_slim(python_version="3.12")
    app = modal.App(APP_PREFIX + run)
    store = modal.Volume.from_name(STORE_VOLUME, create_if_missing=True, version=2)
    results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True, version=2)
    vols = {STORE: store, RESULTS: results}
    hmac = modal.Secret.from_dict({"YETO_ISLAND_HMAC_KEY": secrets.token_hex(32)})
    gpu, timeout = cfg["gpu"], cfg["timeout_s"]
    fns = {
        "sampler": app.function(image=sampler_image, gpu=gpu, cpu=8.0, memory=32768, timeout=timeout, retries=0,
                                volumes=vols, name="sampler", serialized=True)(sampler),
        "trainer": app.function(image=verl_image, gpu=gpu, cpu=16.0, memory=65536, timeout=timeout, retries=0,
                                volumes=vols, secrets=[hmac], name="trainer", serialized=True)(trainer),
        # single_use_containers: every phase in a fresh container (a warm one keeps ~/yeto-output: C4 run c)
        "island": app.function(image=verl_image, gpu=gpu, cpu=8.0, memory=49152, timeout=timeout, retries=0,
                               volumes=vols, single_use_containers=True, name="island", serialized=True)(island),
        "collect": app.function(image=cpu_image, cpu=1.0, memory=2048, timeout=900, volumes=vols,
                                name="collect", serialized=True)(collect),
    }
    if cfg.get("learners", 1) > 1:
        syncer_image = (modal.Image.debian_slim(python_version="3.12")
                        .apt_install("curl", "build-essential", "pkg-config")
                        .run_commands("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal")
                        .add_local_dir(str(REPO / "syncer"), "/build/syncer", copy=True, ignore=["target"])
                        .run_commands("cd /build/syncer && /root/.cargo/bin/cargo build --release --quiet"
                                      " && cp target/release/yeto-syncer /opt/yeto-syncer"))
        fns["syncer"] = app.function(image=syncer_image, cpu=2.0, memory=4096, timeout=timeout, retries=0,
                                     volumes={RESULTS: results}, secrets=[hmac], name="syncer",
                                     serialized=True)(syncer)
    return app, fns


# ====================================================================== local driver
def _modal_cli(*args: str, timeout: float = 900) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "modal", *args], capture_output=True, text=True,
                          timeout=timeout, cwd="/tmp")


def _log(path: Path, message: str) -> None:
    with path.open("a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}\n")


def _wait_calls(calls: dict, deadline: float, log: Path, on_poll=None) -> dict[str, Any]:
    import modal

    results: dict[str, Any] = {}
    while len(results) < len(calls) and time.time() < deadline:
        for name, call in calls.items():
            if name in results:
                continue
            try:
                results[name] = call.get(timeout=0)
                _log(log, f"{name} returned {json.dumps(results[name], default=str)[:300]}")
            except (TimeoutError, modal.exception.TimeoutError):
                pass  # still running
            except modal.exception.OutputExpiredError:
                results[name] = {"error": "output expired"}
            except Exception as exc:  # noqa: BLE001 - a failed call is a result
                results[name] = {"error": f"{type(exc).__name__}: {exc}"}
                _log(log, f"{name} failed: {results[name]['error']}")
        if on_poll is not None:
            on_poll()
        time.sleep(10)
    for name in calls:
        results.setdefault(name, {"error": "timeout waiting for the call"})
    return results


def run(cell, run_id: str, run_dir: Path) -> dict[str, Any]:
    """Run one verl cell end to end (spawn, reclaim, wait, collect, pull)."""
    import modal

    cfg = dict(cell.config)
    raw = run_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    log = run_dir / "driver.log"
    control = modal.Dict.from_name(CONTROL, create_if_missing=True)
    if control.get(f"{run_id}/started", None) is not None:
        raise SystemExit(f"run id {run_id} already used (control Dict {CONTROL})")
    control[f"{run_id}/started"] = time.time()
    app, fns = build_app(run_id, cfg)
    began = time.time()
    reclaim: dict[str, Any] = {"done": cfg.get("reclaim_after_round") is None}

    def maybe_reclaim() -> None:
        if reclaim["done"]:
            return
        progress = control.get(f"{run_id}/sampler/round", None) or {}
        if progress.get("incarnation") == 0 and int(progress.get("rollout_id", -1)) >= cfg["reclaim_after_round"]:
            container = control.get(f"{run_id}/sampler/container", None)
            result = _modal_cli("container", "stop", "--yes", str(container), timeout=120)
            reclaim.update(done=True, container=container, rc=result.returncode, at=time.time(),
                           rollout_id=progress.get("rollout_id"))
            _log(run_dir / "reclaim.log", f"stop container={container} rc={result.returncode} "
                 f"after rollout {progress.get('rollout_id')} {result.stdout.strip()} {result.stderr.strip()}")

    # one detached app for the whole run: a killed driver never cancels a spawned call
    with modal.enable_output(), app.run(detach=True):
        app_id = app.app_id  # an ephemeral app is stopped by id (`modal app stop <name>` misses it)
        (run_dir / "app_id.txt").write_text(f"{app_id}\n")
        calls: dict[str, Any] = {}
        if cfg["kind"] == "island":  # c (control) beside a; b after a, from a's store
            for phase in [p for p in cfg["phases"] if p != "b"]:
                calls[f"island-{phase}"] = fns["island"].spawn(run_id, cfg, phase)
        else:
            if cfg["learners"] > 1:
                calls["syncer"] = fns["syncer"].spawn(run_id, cfg)
            calls["sampler"] = fns["sampler"].spawn(run_id, cfg)
            for i in range(cfg["learners"]):
                calls[f"trainer{i}"] = fns["trainer"].spawn(run_id, cfg, i)
        _log(log, f"spawned {sorted(calls)} app {APP_PREFIX + run_id}")
        results = _wait_calls(calls, began + cfg["timeout_s"] + 600, log, maybe_reclaim)
        if "b" in cfg.get("phases", ()):
            if (results.get("island-a") or {}).get("rc") == 0:
                call_b = fns["island"].spawn(run_id, cfg, "b")
                results.update(_wait_calls({"island-b": call_b}, time.time() + cfg["timeout_s"] + 600, log))
            else:
                results["island-b"] = {"error": "phase a failed; b not run"}
        results["collect"] = _wait_calls({"collect": fns["collect"].spawn(run_id)}, time.time() + 1200,
                                         log)["collect"]
    wall = time.time() - began
    pulled = _modal_cli("volume", "get", "--force", RESULTS_VOLUME, run_id, str(raw), timeout=1800)
    stopped = _modal_cli("app", "stop", "-y", app_id, timeout=120)  # only our own app
    summary = {"calls": results, "reclaim": reclaim, "wall_s": round(wall, 1),
               "pull_rc": pulled.returncode, "stop_rc": stopped.returncode}
    (run_dir / "driver.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def card_seconds(raw: Path) -> float:
    """GPU container seconds from the components' meta.json (one card each)."""
    total = 0.0
    for meta in sorted(Path(raw).rglob("meta.json")):
        data = json.loads(meta.read_text())
        if data.get("gpu"):
            total += float(data.get("seconds") or 0.0)
    return total

