"""S28 G-120: the Modal agentic sampler island of the cross-cloud 120B RL run
(npu1 Ascend 910B3 trainer, Modal sampler; SESSION28/G-120.md, G-120-W2.md).

Same layout as G-ES's sampler (``tools/probes/s27_xcloud/sampler_app.py``: the
Volume ``yeto-s27-xc`` holds ``policy-releases/`` and ``sample-buffer/``; a
mirror on the dev machine moves releases from npu1 and groups to npu1). The
island is the unchanged engine sampler (``python -m yeto.rl.engine.engine_sampler``)
with its ``agentic`` section: each group is one Harbor task x ``n`` Codex trials
in boat sandboxes; Codex calls the records proxy, which samples this
container's SGLang engine and records ids and log-probs
(``yeto.rl.engine.agentic_round``).

* model: Nemotron-3.5-Super 120B-A12B, the text-only checkpoint the trainer
  trains (``nemotron_h``; G-EZ's materialization, 61 shards, 241,337,456,640
  bytes), placed on the Volume at ``--base-dir`` as release version 0 (the
  delta releases rebuild against it, so its files must equal the trainer's
  base byte for byte); 8 x H200, SGLang TP8;
* tasks: Harbor task directories at ``<store>/harbor-tasks`` (TB2.1 train
  tasks), rows at ``<store>/prompts/agentic.jsonl``
  (``{"prompt": ..., "metadata": {"instance_id", "agent_name": "codex", ...}}``);
* records proxy: port ``PROXY_PORT`` forwarded by ``modal.forward``; the path
  secret is made in the container and only its file path is in the
  environment (as ``modal_runner.harbor_relay_tunnel``);
* boat: ``BOAT_API_KEY`` from the launching machine (``boat_client.load_token``)
  as a Modal secret; the pool is closed at the island's end
  (``AgenticRound.close`` -> ``boat_pool.close_pools_now``).

Stop rules as G-ES's (``backlog`` idle, ``version_groups``, ``wall``).

    PLAN_ONLY=1 python tools/probes/s28_g120/agentic_sampler_app.py
    YETO_G120_APP=yeto-s28-g120-s1 modal run tools/probes/s28_g120/agentic_sampler_app.py --tag s28-g120/s1 ...

The entry spawns and returns (a killed local client does not cancel the
spawned call). Logs: Volume ``yeto-s28-g120-logs`` under ``<tag>/``.
"""
from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3] if len(HERE.parents) > 3 else Path("/")
RECLAIM_MARKER = "/tmp/yeto-reclaim-requested"  # sampler_resume.DEFAULT_RECLAIM_MARKER

WORKDIR = "/root/yeto"
IMAGE = ("ghcr.io/michaellchung/yeto-miles-ports@sha256:"
         "d639e258709b9d4ae4a980c63818562daf62137c9e364f779010bc0b753e867a")
XC_VOLUME = "yeto-s27-xc"
STORE_MOUNT = "/root/yeto-checkpoint-store"
SHARD = "modal-0"
GPU_NAME = "NVIDIA H200"
TP = 8
PROXY_PORT = 31600
SECRET_PATH = "/root/.yeto-records-proxy-secret"
MIRROR_DIR = "/dev/shm/yeto-release-mirror"
MEMORY_MIB = 1_048_576  # the release mirror (2 x 241 GB) lives in /dev/shm
TIMEOUT_S = int(os.environ.get("YETO_G120_TIMEOUT_S", "14400"))
DEFAULTS = {
    "store": f"{STORE_MOUNT}/xc120", "base_dir": f"{STORE_MOUNT}/xc120/base/nemotron-120b-text",
    "groups_per_round": 8, "samples_per_group": 4, "max_backlog_groups": 16, "num_rollouts": 64,
    "turn_max_tokens": 4096, "max_model_len": 65536, "max_response_len": 61440, "max_prompt_len": 16384,
    "max_in_flight": 32, "min_samples": 2, "agent_timeout": 900, "trial_timeout": 1500, "boat_ttl": 2700,
    "idle_s": 600.0, "stop_version": -1, "stop_groups": 0, "max_s": 10800.0, "poll_s": 20.0,
    "mem_fraction": 0.8, "temperature": 1.0, "top_p": 1.0, "model_name": "nemotron-3.5-super",
}
PROBE_PAIRS = [["The capital of France is", " Paris. It is known for the Eiffel Tower and the Louvre museum."]]


def spec(store: str, out: str, *, engine_version: str, cfg: dict) -> dict:
    base = str(cfg["base_dir"])
    server = {"base_dir": base, "card_model": "H200", "device_family": "cuda", "engine_version": engine_version,
              "precision": "bf16", "quant_scheme": "bf16", "topology": f"tp{TP}",
              "max_model_len": int(cfg["max_model_len"]), "python": sys.executable, "in_place_reload": True,
              "start_timeout_s": 2400.0, "request_timeout_s": 3600.0,
              # the adapter passes --context-length from max_model_len
              "extra_args": ["--mem-fraction-static", str(cfg["mem_fraction"])]}
    return {
        "sampler_id": SHARD,
        "engine": {"factory": "yeto.rl.adapters.sampler_engines.from_spec:build", "adapter": "sglang-server",
                   "server": server, "options": {"tp": TP}},
        # n = trials per task; max_tokens is the per-call cap (the proxy signs the same)
        "sampling": {"max_tokens": int(cfg["turn_max_tokens"]), "temperature": float(cfg["temperature"]),
                     "top_p": float(cfg["top_p"]), "n": int(cfg["samples_per_group"])},
        "num_rollouts": int(cfg["num_rollouts"]), "max_backlog_groups": int(cfg["max_backlog_groups"]),
        "max_prompt_tokens": int(cfg["max_model_len"]) - int(cfg["turn_max_tokens"]),
        "base_version": 0, "base_dir": base, "poll_s": float(cfg["poll_s"]),
        "store": str(cfg["store"]), "buffer_shards": [SHARD],
        "prompts": {"path": f"{cfg['store']}/prompts/agentic.jsonl", "input_key": "prompt",
                    "groups_per_round": int(cfg["groups_per_round"]), "format": "agentic", "task_key": "metadata"},
        "agentic": {"template": "nemotron3", "tokenizer": base, "model": str(cfg["model_name"]),
                    "turn_max_tokens": int(cfg["turn_max_tokens"]), "max_response_len": int(cfg["max_response_len"]),
                    "max_prompt_len": int(cfg["max_prompt_len"]), "max_in_flight": int(cfg["max_in_flight"]),
                    "min_samples": int(cfg["min_samples"]), "proxy": {"host": "0.0.0.0", "port": PROXY_PORT},
                    "task_defaults": {"agent_timeout_s": float(cfg["agent_timeout"]),
                                      "wall_clock_s": float(cfg["trial_timeout"]) + 600.0}},
        "checks": {"expected_expert_matrices": 0, "require_fixed_answers": False, "fixed_max_tokens": 256,
                   "probe_pairs": PROBE_PAIRS, "max_probe_jump": 0.5},
        "event_tape": f"{out}/events.jsonl",
    }


def harbor_env(cfg: dict) -> dict[str, str]:
    store = str(cfg["store"])
    n = int(cfg["max_in_flight"])
    return {
        "HARBOR_ENV_TYPE": "boat", "HARBOR_TASKS_DIR": f"{store}/harbor-tasks",
        "HARBOR_TRIALS_DIR": "/tmp/harbor-trials", "HARBOR_SETUP_RETRIES": "2",
        "HARBOR_ENV_KWARGS": json.dumps({"ttl_s": int(cfg["boat_ttl"])}, separators=(",", ":")),
        "AGENT_MODEL_NAME": str(cfg["model_name"]), "AGENT_TIMEOUT": str(int(cfg["agent_timeout"])),
        "AGENT_TRIAL_TIMEOUT": str(int(cfg["trial_timeout"])),
        "YETO_BOAT_MAX_ACTIVE": str(n + 8), "YETO_BOAT_STARTS_PER_MINUTE": "10", "YETO_BOAT_STARTS_PER_HOUR": "50",
        "YETO_BOAT_POOL": "1", "YETO_BOAT_POOL_IDLE_TTL_S": "1500", "YETO_BOAT_POOL_IDLE_VM_TTL_S": "1800",
        "YETO_BOAT_CODEX_CACHE": "1",
    }


def plan(**overrides: object) -> dict:
    cfg = {**DEFAULTS, **overrides}
    return {"image": IMAGE, "gpu": f"H200:{TP}", "memory_mib": MEMORY_MIB, "timeout_s": TIMEOUT_S,
            "volume": XC_VOLUME, "proxy_port": PROXY_PORT, "harbor_env": harbor_env(cfg),
            "spec": spec(str(cfg["store"]), "/data/logs/<tag>", engine_version="<sglang>", cfg=cfg),
            "stop": {k: cfg[k] for k in ("idle_s", "stop_version", "stop_groups", "max_s")}}


if os.environ.get("PLAN_ONLY") == "1" and __name__ == "__main__":
    print(json.dumps(plan(), indent=1))
    raise SystemExit(0)

import modal  # noqa: E402 - PLAN_ONLY runs without the Modal client

# The module-scope import below runs in the container too (Modal imports this
# file to find the function). There the repo is the mounted WORKDIR, and the
# image's own `yeto` package has no `rl`, so WORKDIR must come first on the path.
for _root in (WORKDIR, str(REPO)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

from yeto.rl.harness.harbor.image_layer import modal_harbor_layer  # noqa: E402

app = modal.App(os.environ.get("YETO_G120_APP", "yeto-s28-g120-sampler"))
xc = modal.Volume.from_name(XC_VOLUME, create_if_missing=False)
logs = modal.Volume.from_name("yeto-s28-g120-logs", create_if_missing=True, version=2)
image = modal_harbor_layer(
    modal.Image.from_registry(IMAGE).env({"HOME": "/root", "PYTHONUNBUFFERED": "1"}),
    "miles", "$(command -v python)",
).add_local_dir(str(REPO / "yeto"), f"{WORKDIR}/yeto", copy=False, ignore=["**/__pycache__", "**/*.pyc"]
).add_local_dir(str(REPO / "docker" / "harbor-layer"), f"{WORKDIR}/docker/harbor-layer", copy=False,
                ignore=["**/__pycache__", "**/*.pyc"]
).add_local_file(str(REPO / "docker" / "image-assert" / "assert_image_versions.py"),
                 f"{WORKDIR}/docker/image-assert/assert_image_versions.py", copy=False)
# modal_harbor_layer reads docker/harbor-layer/versions.json relative to the yeto
# package it is imported from. In the container that package is the mounted
# WORKDIR, so those two paths have to be mounted there as well or the container's
# own import of this file fails before the function is reached.


# G-ES's stop rules and sampler wait (tools/probes/s27_xcloud/sampler_app.py), copied:
# that module builds its own Modal app at import, so it cannot be imported here.
class StopRules:
    """Reads the tape lines and decides when to write the reclaim marker (module doc)."""

    def __init__(self, cfg: dict, began: float) -> None:
        self.cfg, self.began = cfg, began
        self.full_since: float | None = None
        self.version_groups = 0
        self.reason: str | None = None

    def see(self, record: dict) -> None:
        event = record.get("event")
        if event == "rl_sampler_backlog_full":
            self.full_since = time.time()
        elif event in ("rl_sampler_admitted", "rl_sampler_round"):
            self.full_since = None
        if event == "rl_sampler_round" and int(record.get("policy_version", -1)) >= int(self.cfg["stop_version"]):
            self.version_groups += int(record.get("groups") or 0)

    def check(self) -> str | None:
        now = time.time()
        if self.full_since is not None and now - self.full_since >= float(self.cfg["idle_s"]):
            return "backlog"
        if int(self.cfg["stop_groups"]) > 0 and self.version_groups >= int(self.cfg["stop_groups"]):
            return "version_groups"
        if now - self.began >= float(self.cfg["max_s"]):
            return "wall"
        return None


def _watch(out: Path, rules: StopRules, stop: threading.Event) -> None:
    """Print new tape lines, apply the stop rules, commit the logs Volume every 30 s."""
    offsets: dict[Path, int] = {}
    last_commit = 0.0
    while True:
        for tape in sorted(out.glob("events.*.jsonl")):
            try:
                with tape.open("rb") as fh:
                    fh.seek(offsets.get(tape, 0))
                    data = fh.read()
            except OSError:
                continue
            end = data.rfind(b"\n")
            if end < 0:
                continue
            offsets[tape] = offsets.get(tape, 0) + end + 1
            for line in data[:end].split(b"\n"):
                text = line.decode("utf-8", "replace")
                print("[g120-tape] " + text[:1500], flush=True)
                try:
                    rules.see(json.loads(text))
                except ValueError:
                    pass
        if rules.reason is None:
            reason = rules.check()
            if reason is not None:
                rules.reason = reason
                Path(RECLAIM_MARKER).write_text(json.dumps({"by": "wrapper", "reason": reason,
                                                            "time": time.time()}) + "\n")
                print(f"[g120] stop rule {reason}: reclaim marker written", flush=True)
        if time.monotonic() - last_commit >= 30.0:
            try:
                logs.commit()
            except Exception as error:  # noqa: BLE001 - retried on the next pass
                print(f"[g120] logs commit failed: {error}", flush=True)
            last_commit = time.monotonic()
        if stop.wait(5.0):
            return


def _mark_on_signal() -> None:
    """SIGINT/SIGTERM of this function (``modal app stop``) write the reclaim marker."""
    import signal

    def mark(signum: int, _frame: object) -> None:
        try:
            Path(RECLAIM_MARKER).write_text(json.dumps({"signal": int(signum), "time": time.time()}) + "\n")
        except OSError:
            pass
        print(f"[g120] signal {signum}: reclaim marker written", flush=True)

    for name in ("SIGINT", "SIGTERM"):
        try:
            signal.signal(getattr(signal, name), mark)
        except ValueError:
            print(f"[g120] {name} handler not installed (not the main thread)", flush=True)


def _wait_sampler(child: subprocess.Popen, log: Path, own_stop) -> int:
    """Wait for the sampler. After a reclaim the engine sampler holds the process for
    ``engine_sampler_entry.RECLAIM_HOLD_S`` (300 s) so a spot cloud can stop the
    container and run the input again. When the stop is this wrapper's own rule
    (``own_stop()``), once the sampler printed its ``done`` line (its tape, buffer
    commits and leave record are written by then) the hold is cut short with SIGKILL
    (a signal only writes the marker). A marker written by a signal (``modal app
    stop``, a preemption) keeps the hold, as designed for a reclaim."""
    while True:
        try:
            return child.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pass
        if own_stop() and b"[engine-sampler] done:" in log.read_bytes()[-4096:]:
            time.sleep(2.0)
            child.kill()
            print("[g120] sampler done after the stop marker: hold cut short", flush=True)
            return child.wait()


def _boat_secret() -> modal.Secret:
    from yeto.rl.harness.harbor.boat_client import load_token

    return modal.Secret.from_dict({"BOAT_API_KEY": load_token()})


@app.function(image=image, gpu=f"H200:{TP}", memory=MEMORY_MIB, cpu=64, timeout=TIMEOUT_S,
              volumes={STORE_MOUNT: xc, "/data/logs": logs}, secrets=[_boat_secret()])
def sampler(tag: str, cfg: dict) -> int:
    began = time.time()
    out = Path("/data/logs") / tag
    out.mkdir(parents=True, exist_ok=True)
    names = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, check=True).stdout.strip()
    (out / "gpu.txt").write_text(names + "\n")
    lines = names.splitlines()
    if len(lines) != TP or any(line.strip() != GPU_NAME for line in lines):
        logs.commit()
        raise SystemExit(f"wrong GPUs: {names}")
    for need in ("config.json", "model.safetensors.index.json"):
        if not (Path(cfg["base_dir"]) / need).is_file():
            logs.commit()
            raise SystemExit(f"base checkpoint missing {need} at {cfg['base_dir']}")
    import sglang

    sp = spec(str(cfg["store"]), str(out), engine_version=sglang.__version__, cfg=cfg)
    (out / "spec.json").write_text(json.dumps(sp, indent=1))
    rules = StopRules(cfg, began)
    stop = threading.Event()
    watcher = threading.Thread(target=_watch, args=(out, rules, stop), daemon=True)
    watcher.start()
    rc = 1
    with modal.forward(PROXY_PORT) as tunnel:
        fd = os.open(SECRET_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(secrets.token_urlsafe(24))
        env = {**os.environ, **harbor_env(cfg), "PYTHONPATH": WORKDIR, "YETO_SPOT_RECLAIM_MARKER": RECLAIM_MARKER,
               "YETO_RL_STORE_IMPL": "yeto.cloud.modal_ckpt_store:store_from_env",
               "YETO_RL_STORE_MODAL_VOLUME": XC_VOLUME, "YETO_RL_STORE_MODAL_MOUNT": STORE_MOUNT,
               "YETO_RL_RELEASE_MIRROR": MIRROR_DIR, "YETO_MODAL_PYTHON": sys.executable,
               "YETO_MODAL_SYSPATH": os.pathsep.join(p for p in sys.path if p),
               "YETO_RECORDS_PROXY_PUBLIC_BASE": tunnel.url.rstrip("/"),
               "YETO_RECORDS_PROXY_SECRET_FILE": SECRET_PATH}
        meta = {"tag": tag, "gpu": names, "sglang": sglang.__version__, "cfg": cfg, "tunnel": "set",
                "container_started_unix": began, "modal_region": os.environ.get("MODAL_REGION")}
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        logs.commit()
        print(f"[g120] {json.dumps(meta)}", flush=True)
        Path(RECLAIM_MARKER).unlink(missing_ok=True)
        _mark_on_signal()
        with open(out / "sampler.log", "ab") as log:
            child = subprocess.Popen([sys.executable, "-m", "yeto.rl.engine.engine_sampler", "--spec",
                                      str(out / "spec.json")], cwd=WORKDIR, env=env, stdout=log,
                                     stderr=subprocess.STDOUT)
            rc = _wait_sampler(child, out / "sampler.log", lambda: rules.reason is not None)
    stop.set()
    watcher.join(timeout=30)
    result = {**meta, "rc": rc, "stop_rule": rules.reason, "version_groups": rules.version_groups,
              "wall_s": round(time.time() - began, 1)}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    logs.commit()
    print(f"[g120] sampler exited {json.dumps(result)}", flush=True)
    return rc


@app.local_entrypoint()
def main(tag: str, store: str = DEFAULTS["store"], base_dir: str = DEFAULTS["base_dir"],
         groups_per_round: int = 8, samples_per_group: int = 4, max_in_flight: int = 32,
         stop_version: int = -1, stop_groups: int = 0, idle_s: float = 600.0, max_s: float = 10800.0) -> None:
    for path in (store, base_dir):
        if not path.startswith(STORE_MOUNT + "/"):
            raise SystemExit(f"{path} must lie under {STORE_MOUNT}")
    cfg = {**DEFAULTS, "store": store, "base_dir": base_dir, "groups_per_round": groups_per_round,
           "samples_per_group": samples_per_group, "max_in_flight": max_in_flight, "stop_version": stop_version,
           "stop_groups": stop_groups, "idle_s": idle_s, "max_s": max_s}
    call = sampler.spawn(tag, cfg)
    print(f"spawned sampler {call.object_id}")
