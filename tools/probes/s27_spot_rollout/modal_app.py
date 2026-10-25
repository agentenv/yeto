"""S27 spot rollout channel check: one spot sampler island on 1 x H100 (SPOT-ROLLOUT.md 7).

One container runs ``python -m yeto.rl.spot_rollout.sampler`` (the engine sampler
island with the ``sglang-server`` adapter on Qwen3-0.6B) against the head of a
running training launch: it reads the layout board, follows the merged weights
as a receive-only subscriber, loads each cut through the post-load checks and
writes its groups to relay channel ``--channel``. An on-demand Modal H100 stands
in for a spot node; ``modal app stop`` of this app stands in for a reclaim.

Reclaim (S27 G-EM): Modal stops a container with SIGINT, for a preemption and
for ``modal app stop`` alike (G-EE saw ``KeyboardInterrupt`` in the sampler).
The sampler runs with ``YETO_SPOT_RECLAIM_MARKER``: SIGINT/SIGTERM write the
marker, and the sampler leaves PREEMPTED (``engine_sampler_entry``). This
function also writes the marker on its own SIGINT/SIGTERM and keeps waiting
for the sampler instead of killing it, and commits the Volume once the marker
appears, so the tape keeps the leave record.

    YETO_SPOT_APP=yeto-<run>-spot1 YETO_ISLAND_HMAC_KEY=... \\
      modal run --detach tools/probes/s27_spot_rollout/modal_app.py \\
      --tag <run>/spot1 --syncer HOST:PORT --channel 0 --sampler-id spot-0

The entry spawns and returns (never kill a ``.remote()`` caller). The HMAC key
reaches the container as a Modal secret built from the caller's environment
(never printed). Results go to the Volume ``yeto-s27-ea-logs`` under ``<tag>/``:
``spec.json``, ``gpu.txt``, ``sampler.log``, ``events.<incarnation>.jsonl`` (the
sampler tape), ``nvml.csv``, ``result.json``. Every tape line is also printed as
``[spot-tape] <json>`` so ``modal app logs`` shows the island live.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve()
REPO = HERE.parents[3] if len(HERE.parents) > 3 else Path("/")
WORKDIR = "/root/yeto"
# the image the training islands run (yeto.rl.adapters.miles.pins.MILES_NEXT_IMAGE, 9708070-2fa8801)
IMAGE = ("ghcr.io/michaellchung/yeto-miles-ports@sha256:"
         "d639e258709b9d4ae4a980c63818562daf62137c9e364f779010bc0b753e867a")
MODEL, MODEL_REVISION = "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"
GPU_NAME = "H100"
TASKS = "e11-tasks.jsonl"  # the training launch's prompt file (the run snapshot root)
RECLAIM_MARKER = "/tmp/yeto-reclaim-requested"  # sampler_resume.DEFAULT_RECLAIM_MARKER
TIMEOUT_S = int(os.environ.get("YETO_SPOT_TIMEOUT_S", "2400"))
# Post-load probe: teacher-forced continuations; a wrong Megatron -> HF conversion moves
# their mean log-prob by several nats, a few RL steps at lr 1e-5 by far less than 0.5.
PROBE_PAIRS = [
    ["Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many "
     "clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer:",
     " In May she sold 48 / 2 = 24 clips. Altogether she sold 48 + 24 = 72 clips. The answer is 72."],
    ["The capital of France is", " Paris. It is known for the Eiffel Tower and the Louvre museum."],
]

app = modal.App(os.environ.get("YETO_SPOT_APP", "yeto-s27-spot-sampler"))
logs = modal.Volume.from_name("yeto-s27-ea-logs", create_if_missing=True, version=2)
_KEY = os.environ.get("YETO_ISLAND_HMAC_KEY", "")
secrets = [modal.Secret.from_dict({"YETO_ISLAND_HMAC_KEY": _KEY})] if _KEY else []
image = (
    modal.Image.from_registry(IMAGE)
    .env({"HOME": "/root", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(str(REPO / "yeto"), f"{WORKDIR}/yeto", copy=False,
                   ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_file(str(REPO / TASKS), f"{WORKDIR}/{TASKS}", copy=False)
)


def _render_prompts(base: Path, out: Path) -> int:
    """The training islands' prompts as text: the chat template with thinking off
    (the launch's ``--apply-chat-template-kwargs {"enable_thinking": false}``)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(base))
    rows = 0
    with open(f"{WORKDIR}/{TASKS}", encoding="utf-8") as src, out.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            text = tok.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
            dst.write(json.dumps({"prompt": text, "label": row["label"],
                                  "task_id": (row.get("metadata") or {}).get("task_id")}) + "\n")
            rows += 1
    return rows


def _spec(base: Path, prompts: Path, out: Path, *, syncer: str, channel: int, sampler_id: str,
          groups_per_round: int, samples_per_group: int, max_backlog_groups: int, engine_version: str,
          syncer_lost_s: float = 600.0, data_slot: str = "") -> dict:
    server = {"base_dir": str(base), "card_model": GPU_NAME, "device_family": "cuda",
              "engine_version": engine_version, "precision": "bf16", "quant_scheme": "bf16",
              "topology": "tp1", "max_model_len": 4096, "python": sys.executable, "in_place_reload": True,
              "start_timeout_s": 900.0, "request_timeout_s": 1800.0,
              "extra_args": ["--mem-fraction-static", "0.8"]}
    return {
        "sampler_id": sampler_id,
        "engine": {"factory": "yeto.rl.adapters.sampler_engines.from_spec:build", "adapter": "sglang-server",
                   "server": server, "options": {"tp": 1}},
        # the launch's recipe: 8 samples per prompt, 1024 response tokens, temperature 1
        "sampling": {"max_tokens": 1024, "temperature": 1.0, "top_p": 1.0, "n": samples_per_group},
        "max_backlog_groups": max_backlog_groups, "max_prompt_tokens": 512,
        "prompts": {"path": str(prompts), "input_key": "prompt", "label_key": "label",
                    "groups_per_round": groups_per_round, "format": "text",
                    # G-EM: samplers on one run draw disjoint prompt rows (I/N)
                    **({"data_slot": data_slot} if data_slot else {})},
        "reward": "yeto.rl.gsm8k_reward:score", "reward_kind": "miles",
        # bf16 dense model: no expert count; Qwen3-0.6B answers raw GSM8K text unreliably, so a
        # missing answer is recorded only; empty / unreadable output and the probe jump still fail
        "checks": {"expected_expert_matrices": 0, "require_fixed_answers": False, "fixed_max_tokens": 512,
                   "probe_pairs": PROBE_PAIRS, "max_probe_jump": 0.5},
        "event_tape": str(out / "events.jsonl"),
        "base_dir": str(base), "base_model": str(base), "model_name": "qwen3",
        "syncer": syncer, "channel": int(channel), "work_dir": "/tmp/spot-work",
        "reload_every": 1, "layout_timeout_s": 1500.0, "syncer_lost_s": float(syncer_lost_s),
        "relay_gone_s": float(syncer_lost_s),
    }


def _mirror(out: Path, stop: threading.Event) -> None:
    """Print every new tape line (``[spot-tape]``) and commit the Volume every 30 s."""
    offsets: dict[Path, int] = {}
    last_commit = 0.0
    reclaim_committed = False
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
                print("[spot-tape] " + line.decode("utf-8", "replace")[:800], flush=True)
        reclaimed = not reclaim_committed and Path(RECLAIM_MARKER).exists()
        if reclaimed:
            reclaim_committed = True
            time.sleep(3.0)  # the sampler's leave record (reclaim watcher polls every 0.5 s)
        if reclaimed or time.monotonic() - last_commit >= 30.0:
            try:
                logs.commit()
            except Exception as error:  # noqa: BLE001 - a failed commit is retried
                print(f"[spot] volume commit failed: {error}", flush=True)
            last_commit = time.monotonic()
        if stop.wait(5.0):
            return


@app.function(image=image, volumes={"/data/logs": logs}, gpu=f"{GPU_NAME}!:1", cpu=8, memory=65536,
              timeout=TIMEOUT_S, secrets=secrets)
def run(tag: str, syncer: str, channel: int, sampler_id: str, groups_per_round: int = 16,
        samples_per_group: int = 8, max_backlog_groups: int = 32, syncer_lost_s: float = 600.0,
        data_slot: str = "") -> int:
    from huggingface_hub import snapshot_download

    out = Path("/data/logs") / tag
    out.mkdir(parents=True, exist_ok=True)
    began = time.time()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout.strip()
    (out / "gpu.txt").write_text(gpu + "\n")
    print(f"[spot] gpu {gpu}", flush=True)
    if not gpu or any(GPU_NAME not in line for line in gpu.splitlines()):
        logs.commit()
        raise SystemExit(f"wrong GPU: {gpu}")
    if not os.environ.get("YETO_ISLAND_HMAC_KEY"):
        raise SystemExit("no YETO_ISLAND_HMAC_KEY secret")
    import sglang

    base = Path(snapshot_download(MODEL, revision=MODEL_REVISION))
    prompts = Path("/tmp/spot-prompts.jsonl")
    rows = _render_prompts(base, prompts)
    spec = _spec(base, prompts, out, syncer=syncer, channel=channel, sampler_id=sampler_id,
                 groups_per_round=groups_per_round, samples_per_group=samples_per_group,
                 max_backlog_groups=max_backlog_groups, engine_version=sglang.__version__,
                 syncer_lost_s=syncer_lost_s, data_slot=data_slot)
    (out / "spec.json").write_text(json.dumps(spec, indent=1))
    meta = {"tag": tag, "gpu": gpu, "sglang": sglang.__version__, "model": MODEL, "revision": MODEL_REVISION,
            "prompt_rows": rows, "syncer": syncer, "channel": channel, "sampler_id": sampler_id,
            "container_started_unix": began, "setup_s": round(time.time() - began, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    logs.commit()
    print(f"[spot] {json.dumps(meta)}", flush=True)
    nvml = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw",
                             "--format=csv", "-l", "5"], stdout=open(out / "nvml.csv", "w"),
                            stderr=subprocess.STDOUT)
    stop = threading.Event()
    mirror = threading.Thread(target=_mirror, args=(out, stop), daemon=True)
    mirror.start()
    env = {**os.environ, "PYTHONPATH": WORKDIR, "YETO_SPOT_RECLAIM_MARKER": RECLAIM_MARKER}
    Path(RECLAIM_MARKER).unlink(missing_ok=True)
    _mark_on_signal()
    with open(out / "sampler.log", "ab") as log:
        child = subprocess.Popen([sys.executable, "-m", "yeto.rl.spot_rollout.sampler", "--spec",
                                  str(out / "spec.json")], cwd=WORKDIR, env=env, stdout=log,
                                 stderr=subprocess.STDOUT)
        rc = child.wait()  # a signal only writes the marker (never kills the sampler)
    nvml.terminate()
    stop.set()
    mirror.join(timeout=30)
    (out / "result.json").write_text(json.dumps({**meta, "rc": rc, "wall_s": round(time.time() - began, 1)}))
    logs.commit()
    print(f"[spot] sampler exited rc={rc}", flush=True)
    return rc


def _mark_on_signal() -> None:
    """SIGINT/SIGTERM of this function write the reclaim marker (module doc)."""
    import signal

    def mark(signum: int, _frame: object) -> None:
        try:
            Path(RECLAIM_MARKER).write_text(json.dumps({"signal": int(signum), "time": time.time()}) + "\n")
        except OSError:
            pass
        print(f"[spot] signal {signum}: reclaim marker written", flush=True)

    for name in ("SIGINT", "SIGTERM"):
        try:
            signal.signal(getattr(signal, name), mark)
        except ValueError:  # not the main thread: the sampler's own handler still marks
            print(f"[spot] {name} handler not installed (not the main thread)", flush=True)


@app.function(image=image, cpu=2, memory=4096, timeout=1800)
def prebuild() -> dict:
    """CPU only: build and cache the image before a run (a cold image pull would delay
    the sampler's first round) and check the imports the sampler needs."""
    import sglang
    from miles.backends.megatron_utils.megatron_to_hf import _convert_to_hf_core  # noqa: F401

    sys.path.insert(0, WORKDIR)
    from yeto.rl.spot_rollout import sampler  # noqa: F401

    return {"sglang": sglang.__version__, "python": sys.version.split()[0]}


@app.local_entrypoint()
def main(tag: str, syncer: str, channel: int = 0, sampler_id: str = "spot-0", groups_per_round: int = 16,
         samples_per_group: int = 8, max_backlog_groups: int = 32, data_slot: str = "") -> None:
    call = run.spawn(tag, syncer, channel, sampler_id, groups_per_round, samples_per_group, max_backlog_groups,
                     600.0, data_slot)
    print(f"spawned run {call.object_id}")
