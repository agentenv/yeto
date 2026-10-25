"""S27 P4 (G-DB): two verl full-parameter islands + the unified Rust syncer on Modal.

Usage (local, yeto-test-venv):
  python tools/probes/s27_verl_full/modal_p4.py plan RUN_ID   # PLAN_ONLY: no Modal call
  python tools/probes/s27_verl_full/modal_p4.py run  RUN_ID   # needs the plan; spawns, waits, pulls, stops
  python tools/probes/s27_verl_full/modal_p4.py pull RUN_ID   # results Volume -> run dir (files < 50 MB)

What runs (one detached app ``yeto-gdb-<RUN_ID>``):

* ``syncer`` (CPU): the Rust syncer built from this checkout's ``syncer/``,
  started with the exact actor command the launcher builds for
  ``--rl-sync-preset full-parameter`` (``plan`` derives it, and its semantic
  profile hash, from the launcher functions), reachable through a Modal TCP
  tunnel whose address goes to the control Dict;
* ``island`` x 2 (one H200 each): ``yeto.rl.adapters.verl.island_entry`` with
  ``--parameter-mode full``, the strict sweep (or async rounds), each island
  on its own half of the data (``--rl-data-slot k/2``).

The island commands are this harness's, not the launcher's (the launcher's
full-parameter preset is Miles-shaped: fixed-partition engines; the verl
island is colocated).  The layout hash comes from the model's HF config on
the meta device (``plan``); an island whose layout differs refuses at start
and prints its observed layout (``[yeto-verl-layout-probe]``).

Guards: every container checks its card names (all ``NVIDIA H200``) before
loading a model; the local driver stops the app at ``hard_stop_s`` wall
seconds after the spawn; ``run`` counts S27's H200 containers first and
refuses above the 80-card cap.  Raw data is written to the results Volume
before a call returns; ``pull`` never fetches a file of 50 MB or more.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
RUNS = Path("/home/michael/work/s1-runs")
APP_PREFIX = "yeto-gdb-"
RESULTS_VOLUME = "yeto-eval-results"
CONTROL = "yeto-gdb-control"
RESULTS = "/mnt/results"
WORKDIR = "/root/sky_workdir"
VERL_PY = "/workspace/verl/.venv/bin/python"
SYNCER_PORT = 29400
MAX_FILE = 50 << 20
GPU_NAME = "NVIDIA H200"
S27_H200_CAP = 80

CFG: dict[str, Any] = {
    "model": "Qwen/Qwen3-0.6B",
    "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
    "data": "zhuzilin/gsm8k",
    "data_revision": "0cbd9f31d91ac21a7613dcbc7fef992adac459ae",
    "gpu": "H200",
    "islands": 2,
    "rounds": 4,                # global rounds = outer merges of the strict sweep
    "fragments": 4,             # minimum fragment count of the layout
    "full_parameter_rounds": "sweep",
    "final_cut": "ack",
    "push_dtype": "f32",
    "groups_per_round": 16,
    "samples_per_group": 4,
    "max_tokens": 512,
    "seq_len": 1024,
    "lr": 1e-6,
    "seed": 1,
    "gpu_mem": 0.5,             # vLLM share of the card (YETO_VERL_GPU_MEM)
    "wait_timeout_s": 900,
    "timeout_s": 3600,          # per container
    "hard_stop_s": 4200,        # local watchdog: stop the app this long after the spawn
    "rate_per_island_h": 6.0,   # $ per island container hour (H200 + 64 GiB + 16 CPU, estimate)
    "rate_syncer_h": 0.3,
}


# ====================================================================== plan (local, no Modal)
def model_shapes(model: str, revision: str) -> dict[str, tuple[int, ...]]:
    """Trainable parameter shapes of the HF model, built on the meta device
    (tied weights appear once, as in the FSDP2 actor's ``named_parameters``)."""
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(str(Path(hf_hub_download(model, "config.json", revision=revision)).parent))
    with torch.device("meta"):
        net = AutoModelForCausalLM.from_config(config)
    return {name: tuple(parameter.shape) for name, parameter in net.named_parameters() if parameter.requires_grad}


def plan_layout(cfg: dict) -> dict:
    import hashlib

    from huggingface_hub import hf_hub_download

    from yeto.rl.adapters.verl import full_parameter as fp
    from yeto.rl.local_learner import ComponentIdentity

    config_path = hf_hub_download(cfg["model"], "config.json", revision=cfg["model_revision"])
    config_sha = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    shapes = model_shapes(cfg["model"], cfg["model_revision"])
    component = ComponentIdentity(role="actor", model_revision=cfg["model_revision"].lower(), config_hash=config_sha)
    layout = fp.plan_layout(fp.island_specs(shapes), component=component, minimum_fragments=cfg["fragments"],
                            matrix_merge="avg" if cfg["full_parameter_rounds"] == "sweep" else "rda")
    return {"model_config_sha256": config_sha, "parameter_layout_sha256": layout.layout_hash,
            "fragments": layout.fragments.num_fragments, "tensors": len(shapes),
            "scalars": sum(spec.numel for spec in layout.specs),
            "fragment_bytes": [fragment.numel * 4 for fragment in layout.fragments.fragments]}


def launcher_syncer(cfg: dict, layout: dict) -> dict:
    """The launcher's actor syncer command and semantic profile hash for this run."""
    import yeto.launch.modal_islands as modal_islands
    from yeto import launcher
    from yeto.cli import parse_args
    from yeto.launcher import _prepare_rl_args

    modal_islands._list_modal_apps = lambda: []  # PLAN_ONLY: never list the account's apps
    gpu = ",".join(["modal:2xh200"] * cfg["islands"])
    args = parse_args([
        "--gpu", gpu, "--model", cfg["model"], "--data", cfg["data"], "--training-mode", "rl",
        "--total-steps", str(cfg["rounds"]), "--rollout-batch-size", str(cfg["groups_per_round"]),
        "--n-samples-per-prompt", str(cfg["samples_per_group"]),
        "--rollout-max-response-len", str(cfg["max_tokens"]), "--local-rl-rounds-per-sync", "1",
        "--reward-function", "yeto.rl.gsm8k_reward:score", "--rl-engine", "ports", "--trust-remote-code",
        "--rl-sync-preset", "full-parameter", "--rl-placement", "fixed-partition", "--rl-rollout-gpus", "1",
        "--rl-parameter-layout-sha256", layout["parameter_layout_sha256"],
        "--fragments", str(layout["fragments"]),
        "--rl-sync-full-parameter-rounds", cfg["full_parameter_rounds"],
        "--rl-sync-final-cut", cfg["final_cut"],
        "--rl-sync-wait-timeout-s", str(float(cfg["wait_timeout_s"])),
    ])
    args.model_revision, args.data_revision = cfg["model_revision"], cfg["data_revision"]
    args.source_sha256, args.reward_sha256 = "0" * 64, "0" * 64  # not part of the syncer command
    args.controller = "local"
    _prepare_rl_args(args)
    command = launcher.role_syncer_commands(args, cfg["islands"], binary="/opt/yeto-syncer")["actor"]
    command = (command.replace(f"--port {launcher.SYNCER_PORT}", f"--port {SYNCER_PORT}")
               .replace(launcher.RL_SYNCER_CHECKPOINT, "/tmp/syncer-state.ckpt")
               .replace(launcher.RL_SYNCER_EVENT_TAPE, "/tmp/syncer-tape.jsonl"))
    return {"command": command, "profile_sha256": launcher.rl_syncer_profile(args, cfg["islands"]).sha256,
            "matrix_merge": getattr(args, "rl_sync_matrix_merge", "avg"),
            "pipeline": int(getattr(args, "pipeline", 1))}


def island_argv(cfg: dict, layout: dict, syncer: dict, learner_id: int) -> list[str]:
    rounds, fragments = cfg["rounds"], layout["fragments"]
    argv = [VERL_PY, "-m", "yeto.rl.adapters.verl.island_entry",
            "--model", cfg["model"], "--model-revision", cfg["model_revision"],
            "--data", cfg["data"], "--data-revision", cfg["data_revision"],
            "--learner-id", str(learner_id), "--reward-function", "gsm8k_reward:score",
            "--global-rounds", str(rounds), "--groups-per-round", str(cfg["groups_per_round"]),
            "--samples-per-group", str(cfg["samples_per_group"]),
            "--rollout-max-response-len", str(cfg["max_tokens"]),
            "--event-tape", f"/root/yeto-output/rl-island-{learner_id}.jsonl",
            "--inner-lr", str(cfg["lr"]), "--seq-len", str(cfg["seq_len"]),
            "--seed", str(cfg["seed"] + learner_id),
            "--apply-chat-template-kwargs", '{"enable_thinking": false}',
            "--rl-data-slot", f"{learner_id}/{cfg['islands']}", "--rl-echo-events",
            "--parameter-mode", "full", "--num-learners", str(cfg["islands"]),
            "--fragments", str(fragments), "--total-fragment-steps", str(rounds * fragments),
            "--parameter-layout-sha256", layout["parameter_layout_sha256"],
            "--syncer-profile-sha256", syncer["profile_sha256"],
            "--full-parameter-rounds", cfg["full_parameter_rounds"], "--sweep-final-cut", cfg["final_cut"],
            "--sweep-push-dtype", cfg["push_dtype"], "--sweep-wait-timeout-s", str(float(cfg["wait_timeout_s"])),
            "--rl-sync-matrix-merge", syncer["matrix_merge"]]
    if cfg["full_parameter_rounds"] == "async":
        argv += ["--pipeline", str(syncer["pipeline"]), "--local-horizon", "1"]
    return argv


def plan(run: str) -> dict:
    sys.path.insert(0, str(REPO))
    cfg = dict(CFG)
    layout = plan_layout(cfg)
    syncer = launcher_syncer(cfg, layout)
    islands = [island_argv(cfg, layout, syncer, i) for i in range(cfg["islands"])]
    hours = cfg["hard_stop_s"] / 3600
    estimate = round(hours * (cfg["islands"] * cfg["rate_per_island_h"] + cfg["rate_syncer_h"]), 2)
    out = {"run": run, "cfg": cfg, "layout": layout, "syncer": syncer, "island_argv": islands,
           "cap_estimate_usd": estimate, "planned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                                      text=True).stdout.strip()}
    run_dir = RUNS / run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plan.json").write_text(json.dumps(out, indent=1))
    return out


# ====================================================================== in-container helpers
def _gpu_names(out: Path) -> list[str]:
    names = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=60).stdout.split("\n")
    names = [name.strip() for name in names if name.strip()]
    (out / "nvidia-smi.txt").write_text(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    if not names or any(name != GPU_NAME for name in names):
        raise RuntimeError(f"asked for {GPU_NAME}, got {names}")
    return names


def _commit() -> None:
    import modal

    modal.Volume.from_name(RESULTS_VOLUME).commit()


def _copy_small(src: Path, dst: Path) -> list[list]:
    """Copy ``src`` (file or tree) into ``dst``; files of MAX_FILE or more are listed, not copied."""
    skipped = []
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.is_file()) if src.is_dir() else []
    for path in files:
        rel = path.name if src.is_file() else path.relative_to(src)
        if path.stat().st_size >= MAX_FILE:
            skipped.append([str(rel), path.stat().st_size])
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return skipped


def _finish(out: Path, meta: dict, began: float, copies: dict[str, Path]) -> None:
    meta["seconds"] = round(time.time() - began, 1)
    meta["skipped_large"] = {name: _copy_small(src, out / name) for name, src in copies.items()}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    _commit()


def syncer(run: str, plan_doc: dict) -> dict:
    import modal

    out = Path(RESULTS) / run / "syncer"
    out.mkdir(parents=True, exist_ok=True)
    control = modal.Dict.from_name(CONTROL, create_if_missing=True)
    began = time.time()
    command = plan_doc["syncer"]["command"]
    meta: dict[str, Any] = {"command": command, "container": os.environ.get("MODAL_TASK_ID")}
    try:
        with modal.forward(SYNCER_PORT, unencrypted=True) as tunnel, (out / "syncer.log").open("w") as log:
            host, port = tunnel.tcp_socket
            proc = subprocess.Popen(["bash", "-c", command], stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "TOKIO_WORKER_THREADS": "8"})
            time.sleep(2.0)
            control[f"{run}/syncer"] = meta["address"] = f"{host}:{port}"
            meta["rc"] = proc.wait(timeout=plan_doc["cfg"]["timeout_s"] - 300)
    except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
        meta["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _finish(out, meta, began, {"syncer-tape.jsonl": Path("/tmp/syncer-tape.jsonl")})
    return meta


def island(run: str, plan_doc: dict, learner_id: int) -> dict:
    import modal

    sys.path.insert(0, WORKDIR)
    cfg = plan_doc["cfg"]
    out = Path(RESULTS) / run / f"island{learner_id}"
    out.mkdir(parents=True, exist_ok=True)
    began = time.time()
    meta: dict[str, Any] = {"container": os.environ.get("MODAL_TASK_ID"), "learner_id": learner_id}
    nvml = None
    try:
        meta["gpu"] = _gpu_names(out)  # before any model load
        nvml = subprocess.Popen(["bash", "-c", "nvidia-smi --query-gpu=timestamp,name,utilization.gpu,"
                                 f"memory.used,power.draw --format=csv -l 5 >> {out}/nvml.csv"])
        control = modal.Dict.from_name(CONTROL, create_if_missing=True)
        address = None
        for _ in range(600):  # the syncer container publishes its tunnel address
            address = control.get(f"{run}/syncer")
            if address:
                break
            time.sleep(2.0)
        if not address:
            raise RuntimeError("no syncer address within 1200 s")
        from yeto.hw.catalog import COMPAT_GROUP_ENV, compat_group

        env = dict(os.environ, PYTHONPATH=WORKDIR, HF_HUB_DISABLE_TELEMETRY="1", NCCL_DEBUG="WARN",
                   RAY_ADDRESS="127.0.0.1:6379", LEARNER_ID=str(learner_id), TOKIO_WORKER_THREADS="8",
                   YETO_VERL_GPU_MEM=str(cfg["gpu_mem"]), **{COMPAT_GROUP_ENV: compat_group(cfg["gpu"])})
        argv = list(plan_doc["island_argv"][learner_id]) + ["--syncer", address]
        meta["argv"] = argv
        with (out / "island.log").open("w") as log:
            subprocess.run(["/workspace/verl/.venv/bin/ray", "start", "--head", "--node-ip-address=127.0.0.1",
                            "--port=6379", "--num-cpus=32", "--num-gpus=1", "--include-dashboard=false",
                            "--temp-dir=/root/p4-ray"], env=env, check=True, timeout=300, stdout=log, stderr=log)
            meta["rc"] = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT, cwd="/root",
                                        timeout=cfg["timeout_s"] - 300).returncode
            subprocess.run(["/workspace/verl/.venv/bin/ray", "stop", "--force"], stdout=log, stderr=log)
    except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
        meta["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if nvml is not None:
            nvml.terminate()
        _finish(out, meta, began, {"yeto-output": Path("/root/yeto-output")})
    return meta


# ====================================================================== app
def build_app(run: str, plan_doc: dict):
    import modal

    sys.path.insert(0, str(REPO))
    from modal._vendor import cloudpickle

    cloudpickle.register_pickle_by_value(sys.modules[__name__])  # `tools` is not in the images
    from yeto.rl.adapters.verl.image import modal_image

    cfg = plan_doc["cfg"]
    ignore = [".git", ".venv", ".claude", "**/__pycache__", "**/*.pyc", "s1-runs", "syncer/target",
              "evidence*", "openspec", "docs", "tutorial", "data"]
    verl_image = modal_image(modal).env({"PYTHONUNBUFFERED": "1"}).add_local_dir(
        str(REPO), WORKDIR, copy=False, ignore=ignore)
    syncer_image = (modal.Image.debian_slim(python_version="3.12")
                    .apt_install("curl", "build-essential", "pkg-config")
                    .run_commands("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal")
                    .add_local_dir(str(REPO / "syncer"), "/build/syncer", copy=True, ignore=["target"])
                    .run_commands("cd /build/syncer && /root/.cargo/bin/cargo build --release --quiet"
                                  " && cp target/release/yeto-syncer /opt/yeto-syncer"))
    app = modal.App(APP_PREFIX + run)
    results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True, version=2)
    hmac = modal.Secret.from_dict({"YETO_ISLAND_HMAC_KEY": secrets.token_hex(32)})
    timeout = cfg["timeout_s"]
    fns = {
        "island": app.function(image=verl_image, gpu=f"{cfg['gpu']}:1", cpu=16.0, memory=65536,
                               timeout=timeout, retries=0, volumes={RESULTS: results}, secrets=[hmac],
                               name="island", serialized=True)(island),
        "syncer": app.function(image=syncer_image, cpu=2.0, memory=8192, timeout=timeout, retries=0,
                               volumes={RESULTS: results}, secrets=[hmac], name="syncer",
                               serialized=True)(syncer),
    }
    return app, fns


# ====================================================================== local driver
def _modal_cli(*args: str, timeout: float = 900) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "modal", *args], capture_output=True, text=True,
                          timeout=timeout, cwd="/tmp")


def _log(path: Path, message: str) -> None:
    with path.open("a") as handle:
        handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}\n")


def s27_h200_containers() -> tuple[int, str]:
    """Running Modal containers of S27 apps that hold H200s, in cards (estimate:
    the app's GPU count is not listed, so every yeto-* GPU container counts as 8
    unless its app name says otherwise)."""
    listing = _modal_cli("container", "list", "--json", timeout=120)
    if listing.returncode != 0:
        raise RuntimeError(f"modal container list failed: {listing.stderr[-500:]}")
    rows = json.loads(listing.stdout or "[]")
    cards = 0
    for row in rows:
        name = str(row.get("App Name") or row.get("app_name") or row)
        if "glm53-uncensored" in name or not name.startswith("yeto-"):
            continue
        cards += 1 if name.startswith(APP_PREFIX) else 8
    return cards, listing.stdout


def _wait_calls(calls: dict, deadline: float, log: Path) -> dict[str, Any]:
    import modal

    results: dict[str, Any] = {}
    while len(results) < len(calls) and time.time() < deadline:
        for name, call in calls.items():
            if name in results:
                continue
            try:
                results[name] = call.get(timeout=0)
                _log(log, f"{name} returned {json.dumps(results[name], default=str)[:400]}")
            except (TimeoutError, modal.exception.TimeoutError):
                pass
            except Exception as exc:  # noqa: BLE001 - a failed call is a result
                results[name] = {"error": f"{type(exc).__name__}: {exc}"}
                _log(log, f"{name} failed: {results[name]['error']}")
        time.sleep(10)
    for name in calls:
        results.setdefault(name, {"error": "watchdog: hard stop before the call returned"})
    return results


def run(run_id: str) -> dict[str, Any]:
    import modal

    run_dir = RUNS / run_id
    plan_doc = json.loads((run_dir / "plan.json").read_text())
    log = run_dir / "driver.log"
    cards, listing = s27_h200_containers()
    (run_dir / "containers-before.json").write_text(listing)
    _log(log, f"S27 H200 cards running (estimate): {cards}; this run adds {plan_doc['cfg']['islands']}")
    if cards + plan_doc["cfg"]["islands"] > S27_H200_CAP:
        raise SystemExit(f"S27 H200 cap: {cards} running + {plan_doc['cfg']['islands']} > {S27_H200_CAP}")
    control = modal.Dict.from_name(CONTROL, create_if_missing=True)
    if control.get(f"{run_id}/started", None) is not None:
        raise SystemExit(f"run id {run_id} already used")
    control[f"{run_id}/started"] = time.time()
    app, fns = build_app(run_id, plan_doc)
    with modal.enable_output(), app.run(detach=True):
        app_id = app.app_id
        (run_dir / "app_id.txt").write_text(f"{app_id}\n")
        began = time.time()
        calls = {"syncer": fns["syncer"].spawn(run_id, plan_doc)}
        for learner_id in range(plan_doc["cfg"]["islands"]):
            calls[f"island{learner_id}"] = fns["island"].spawn(run_id, plan_doc, learner_id)
        _log(log, f"spawned {sorted(calls)} app {APP_PREFIX + run_id} ({app_id})")
        results = _wait_calls(calls, began + plan_doc["cfg"]["hard_stop_s"], log)
    wall = time.time() - began
    stopped = _modal_cli("app", "stop", "-y", app_id, timeout=120)
    _log(log, f"app stop rc={stopped.returncode} after {wall:.0f}s")
    summary = {"calls": results, "wall_s": round(wall, 1), "stop_rc": stopped.returncode, "app_id": app_id}
    (run_dir / "driver.json").write_text(json.dumps(summary, indent=1, default=str))
    summary["pull"] = pull(run_id)
    return summary


def pull(run_id: str) -> dict:
    raw = RUNS / run_id / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    got = _modal_cli("volume", "get", "--force", RESULTS_VOLUME, run_id, str(raw), timeout=1800)
    return {"rc": got.returncode, "stderr": got.stderr[-500:]}


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] not in ("plan", "run", "pull") or not re.fullmatch(r"[a-z0-9-]+", argv[1]):
        print(__doc__, file=sys.stderr)
        return 2
    command, run_id = argv
    if command == "plan":
        print(json.dumps(plan(run_id), indent=1))
    elif command == "run":
        print(json.dumps(run(run_id), indent=1, default=str))
    else:
        print(json.dumps(pull(run_id), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
