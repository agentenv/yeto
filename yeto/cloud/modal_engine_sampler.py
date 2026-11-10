"""Cloud layer: one engine sampler island on Modal (``yeto engine-sampler-modal``).

One Modal GPU function runs ``python -m yeto.rl.engine.engine_sampler --spec``
(the sglang-server adapter) against the run's object store: it writes its own
buffer shard (``SamplerConfig.shard``) and follows the trainer's policy releases.
The trainer reads it with ``--rl-buffer-shards`` naming this shard. Several
samplers split the prompts with ``data_slot`` ``"I/N"`` (``JsonlPrompts``).

The configuration is read on the launching machine (:meth:`SamplerConfig.from_env`
takes the same variable names the K3.1 ops launcher used: RUN, SHARD, RELOAD,
DROP_ZV, N, MAX_TOKENS, THINKING_EFFORT, ...; flags override) and travels to the
container as an argument: Modal re-imports this module in the container without
the launcher's environment.

Store credentials come from a Modal Secret (``secret``): either yeto's own names
(``YETO_RL_STORE_S3_ACCESS_KEY_ID`` / ``..._SECRET_ACCESS_KEY`` / ``..._ENDPOINT_URL``)
or the AWS names (``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` / ``AWS_ENDPOINT_URL``),
read only because the island env says ``YETO_RL_STORE_S3_CREDENTIALS=aws-env``.

Stop a sampler: ``modal.FunctionCall.from_id(id).cancel(terminate_containers=True)``
(the id is printed at spawn), then ``modal app stop --yes <app>``.

Not the Kimi-K3.1 production path: K3.1 samplers run on A-card nodes with the
vLLM ROCm image (``yeto.rl.sampler_pins.VLLM_SAMPLER_IMAGE_ROCM``). This module has
no default image (its former default was a stale trainer image) and refuses to
start until ``SAMPLER_IMAGE`` / ``--image`` names one; the bucket
(``STORE_BUCKET``) and the Modal Secret (``SAMPLER_SECRET``) are required too.

Imports ``modal`` lazily; :func:`build_spec` and :func:`store_env` are CPU-testable.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from yeto.rl.engine.data_split import DataSlot

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_MOUNT = "/root/run/prompts.jsonl"
CODE_ROOT = "/root"  # add_local_python_source("yeto") puts the package at /root/yeto
LOG_MOUNT = "/vol"
STAGE = "/tmp/stage"
DEFAULT_IMAGE = None  # no default: see the module doc (not the K3.1 production path)
NO_IMAGE = ("the Modal engine sampler has no default image: Kimi-K3.1 samplers run on A-card "
            "nodes with the vLLM ROCm image (yeto.rl.sampler_pins.VLLM_SAMPLER_IMAGE_ROCM), not "
            "on Modal. To run this path anyway, name an SGLang sampler image with "
            "SAMPLER_IMAGE=<registry>/<image>@sha256:<digest> (or --image)")
CODE_IGNORE = ["**/__pycache__", "**/*.pyc"]
RELOADS = ("hot", "restart")


@dataclass(frozen=True)
class SamplerConfig:
    run: str
    shard: str
    bucket: str
    selection_sha256: str
    prompts: str
    data_slot: str | None = None
    image: str | None = DEFAULT_IMAGE
    engine_version: str = "sglang-miles-kimi-k31@35708f76"
    model_volume: str = "k31-weights"
    model_mount: str = "/mnt/yeto-models"
    base_dir: str = "/mnt/yeto-models/hf/Kimi-K3.1-Pro/d9e99b07"
    log_volume: str = "yeto-k31-sampler"
    secret: str = ""
    app_name: str = "yeto-k31-sampler-b300"
    python: str = "/opt/sglang/bin/python3"
    gpu: str = "B300"
    tp: int = 8
    ep: int = 8
    mem_fraction: str = "0.85"
    max_model_len: int = 4096
    max_tokens: int = 1024
    n: int = 4
    groups: int = 8
    backlog: int | None = None  # None: 4 x groups
    rounds: int = 1_000_000
    reload: str = "hot"
    drop_zero_variance: bool = False
    thinking_effort: str = "low"
    reward: str = "yeto.rl.gsm8k_reward:score"
    timeout_h: float = 6.0
    memory_gib: int = 2048
    cpu: int = 64
    prewarm: bool = True
    extra_shards: tuple[str, ...] = field(default_factory=tuple)  # informational: the trainer's list

    def __post_init__(self) -> None:
        if self.reload not in RELOADS:
            raise ValueError(f"reload {self.reload!r} is not one of {RELOADS}")
        if self.data_slot:
            DataSlot.parse(self.data_slot)  # "I/N", refused early
        if self.max_tokens + 1024 > self.max_model_len:
            raise ValueError(f"max_tokens {self.max_tokens} + 1024 prompt tokens exceed max_model_len "
                             f"{self.max_model_len}")
        if not self.shard or not self.run or not self.bucket:
            raise ValueError("run, shard and bucket (STORE_BUCKET / --bucket) are required")
        if not self.image:
            raise ValueError(NO_IMAGE)
        if not self.secret:
            raise ValueError("the Modal Secret with the store credentials is required "
                             "(SAMPLER_SECRET / --secret)")

    @property
    def max_backlog_groups(self) -> int:
        return 4 * self.groups if self.backlog is None else int(self.backlog)

    @property
    def trainer_buffer_shards(self) -> str:
        """The ``--rl-buffer-shards`` value the trainer needs (with ``extra_shards``)."""
        return ",".join([*self.extra_shards, self.shard])

    def record(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["extra_shards"] = list(self.extra_shards)
        return out

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "SamplerConfig":
        data = dict(record)
        data["extra_shards"] = tuple(data.get("extra_shards") or ())
        return cls(**data)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None, **overrides: Any) -> "SamplerConfig":
        """The K3.1 ops launcher's variables (RUN, SHARD, SELECTION, PROMPTS, RELOAD, ...);
        ``overrides`` (not None) win."""
        env = os.environ if environ is None else environ
        selection = overrides.pop("selection", None) or env.get("SELECTION")
        sha = overrides.pop("selection_sha256", None) or env.get("SELECTION_SHA256")
        if not sha:
            if not selection:
                raise ValueError("SELECTION (expert selection file) or SELECTION_SHA256 is required")
            sha = json.loads(Path(selection).read_text(encoding="utf-8"))["sha256"]
        got = {
            "run": env.get("RUN"), "shard": env.get("SHARD"), "bucket": env.get("STORE_BUCKET") or env.get("R2_BUCKET"),
            "selection_sha256": sha,
            "prompts": env.get("PROMPTS") or str(REPO_ROOT / "data" / "gsm8k" / "gsm8k-tasks512.jsonl"),
            "data_slot": env.get("DATA_SLOT") or None,
            "image": env.get("SAMPLER_IMAGE"), "engine_version": env.get("ENGINE_VERSION"),
            "ep": _int(env.get("EP")), "mem_fraction": env.get("MEM_FRACTION"),
            "max_model_len": _int(env.get("MAX_MODEL_LEN")), "max_tokens": _int(env.get("MAX_TOKENS")),
            "n": _int(env.get("N")), "groups": _int(env.get("GROUPS")), "backlog": _int(env.get("BACKLOG")),
            "rounds": _int(env.get("ROUNDS")), "reload": env.get("RELOAD"),
            "drop_zero_variance": None if env.get("DROP_ZV") is None else env.get("DROP_ZV") == "1",
            "thinking_effort": env.get("THINKING_EFFORT"), "reward": env.get("REWARD"),
            "timeout_h": None if env.get("FN_TIMEOUT_H") is None else float(env["FN_TIMEOUT_H"]),
            "memory_gib": _int(env.get("MEMORY_GIB")),
            "prewarm": None if env.get("PREWARM") is None else env.get("PREWARM") == "1",
            "secret": env.get("SAMPLER_SECRET"), "app_name": env.get("SAMPLER_APP"),
            "extra_shards": tuple(s for s in (env.get("EXTRA_SHARDS") or "").split(",") if s) or None,
        }
        got.update({k: v for k, v in overrides.items() if v is not None})
        got.setdefault("bucket", None)
        got["bucket"] = got["bucket"] or ""  # refused in __post_init__ with the variable name
        return cls(**{k: v for k, v in got.items() if v is not None})


def _int(value: str | None) -> int | None:
    return None if value in (None, "") else int(value)


def build_spec(c: SamplerConfig, *, out: str, stage: str = STAGE) -> dict[str, Any]:
    """The engine sampler spec (``yeto.rl.engine.engine_sampler_entry``) of one island."""
    extra = ["--mem-fraction-static", c.mem_fraction, "--disable-flashinfer-autotune"]
    if c.ep > 1:
        extra += ["--ep-size", str(c.ep)]
    hot = c.reload == "hot"
    prompts: dict[str, Any] = {
        "path": PROMPTS_MOUNT, "input_key": "messages", "label_key": "label",
        "groups_per_round": c.groups, "format": "chat", "tokenizer": c.base_dir, "trust_remote_code": True,
        "chat_template_kwargs": {"thinking_effort": c.thinking_effort}}
    if c.data_slot:
        prompts["data_slot"] = c.data_slot
    return {
        "sampler_id": c.shard,
        "platform": "cuda",
        "rollout_fields": {"routed_experts": True, "top_logprobs_num": 128, "fp32_lm_head": True},
        "expert_selection_sha256": c.selection_sha256,
        "engine": {
            "factory": "yeto.rl.adapters.sampler_engines.from_spec:build",
            "adapter": "sglang-server",
            "server": {"base_dir": c.base_dir, "port": 30000, "python": c.python, "card_model": c.gpu,
                       "device_family": "nvidia", "engine_version": c.engine_version, "precision": "mxfp4",
                       "quant_scheme": "MXFP4_EXPERTS_RTN", "topology": f"tp{c.tp}",
                       "max_model_len": c.max_model_len, "in_place_reload": hot, "start_timeout_s": 5400,
                       "extra_args": extra},
            "options": {"tp": c.tp, "expert_file_update": "auto" if hot else "off", "expert_update_probe": hot,
                        "live_update": False, "dp_attention_flag": "attn-dp-size",
                        "load_report_dir": f"{out}/load-reports"},
        },
        "sampling": {"max_tokens": c.max_tokens, "temperature": 1.0, "top_p": 1.0, "top_k": -1, "n": c.n},
        "num_rollouts": c.rounds, "max_backlog_groups": c.max_backlog_groups, "max_prompt_tokens": 1024,
        "drop_zero_variance_groups": bool(c.drop_zero_variance),
        "base_version": 0, "poll_s": 5.0,
        "store": f"{stage}/{c.bucket}/runs/{c.run}",
        "release_dir": "policy-releases",
        "buffer_shards": [c.shard],
        "prompts": prompts,
        "reward": c.reward,
        "reward_kind": "miles",
        "checks": {"expected_expert_matrices": None, "allow_unverified_count": True, "require_fixed_answers": False},
        "incarnation_lease": False,
        "max_consecutive_load_failures": 3,
        "event_tape": f"{out}/events.jsonl",
    }


def store_env(c: SamplerConfig, environ: Mapping[str, str], *, stage: str = STAGE) -> dict[str, str]:
    """The store variables of the sampler process, from the container environment
    (the Secret): yeto-named store credentials as they are, else the AWS names."""
    from yeto.cloud.s3_store import (CREDENTIALS_AWS_ENV, CREDENTIALS_ENV, ENDPOINT_ENV, KEY_ID_ENV,
                                     REGION_ENV, STAGE_ENV, URI_ENV)

    env = {"YETO_RL_STORE_IMPL": "yeto.cloud.s3_store:store_from_env",
           URI_ENV: f"s3://{c.bucket}/runs/{c.run}", STAGE_ENV: stage, "PYTHONPATH": CODE_ROOT}
    endpoint = environ.get(ENDPOINT_ENV) or environ.get("AWS_ENDPOINT_URL")
    if not endpoint:
        raise ValueError(f"the store Secret has no endpoint ({ENDPOINT_ENV} or AWS_ENDPOINT_URL)")
    env[ENDPOINT_ENV] = endpoint
    env[REGION_ENV] = environ.get(REGION_ENV) or "auto"
    if not environ.get(KEY_ID_ENV):
        env[CREDENTIALS_ENV] = CREDENTIALS_AWS_ENV
    return env


def dry_run(c: SamplerConfig) -> dict[str, Any]:
    """What a launch would run (CPU only, no Modal call): the spec and the trainer flag."""
    rows = [line for line in Path(c.prompts).read_text(encoding="utf-8").splitlines() if line.strip()]
    slot = DataSlot.parse(c.data_slot)
    owned = sum(1 for i in range(len(rows)) if slot.owns(i))
    problems = []
    if c.groups > owned:
        problems.append(f"groups_per_round {c.groups} exceeds the {owned} prompt rows of slot {slot.env()}")
    return {"app": c.app_name, "store": f"s3://{c.bucket}/runs/{c.run}", "shard": c.shard,
            "data_slot": slot.env(), "prompt_rows": owned,
            "trainer_buffer_shards": c.trainer_buffer_shards, "image": c.image, "secret": c.secret,
            "spec": build_spec(c, out=f"{LOG_MOUNT}/sampler/{c.run}/{c.shard}"), "problems": problems}


# ---------------------------------------------------------------------------- container
def sampler_body(record: Mapping[str, Any]) -> dict[str, Any]:
    """Runs INSIDE the Modal container: write the spec, run the engine sampler."""
    import subprocess
    import threading

    import modal

    c = SamplerConfig.from_record(record)
    t0 = time.time()
    logs = modal.Volume.from_name(c.log_volume)
    out = f"{LOG_MOUNT}/sampler/{c.run}/{c.shard}-{time.strftime('%Y%m%d-%H%M%S')}"
    Path(out).mkdir(parents=True, exist_ok=True)
    spec = build_spec(c, out=out)
    Path(out, "spec.json").write_text(json.dumps(spec, indent=2))
    env = dict(os.environ, **store_env(c, os.environ))
    pre = subprocess.run(f"nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader; "
                         f"{c.python} -c 'import sglang;print(sglang.__version__)'; "
                         f"head -c 300 {c.base_dir}/MANIFEST.json", shell=True, capture_output=True, text=True)
    Path(out, "preflight.log").write_text(pre.stdout + pre.stderr)
    print(pre.stdout, pre.stderr, flush=True)
    stop = threading.Event()

    def smi() -> None:
        with open(f"{out}/nvidia-smi.csv", "a", buffering=1) as fh:
            while not stop.is_set():
                fh.write(subprocess.run(["nvidia-smi", "--query-gpu=timestamp,index,memory.used,utilization.gpu",
                                         "--format=csv,noheader"], capture_output=True, text=True).stdout)
                stop.wait(15)

    def commit_loop() -> None:
        while not stop.wait(300):
            logs.commit()

    def prewarm() -> None:
        # read every weight file once, 32 at a time, so SGLang's own load hits page cache
        t = time.time()
        r = subprocess.run(f"ls {c.base_dir}/*.safetensors | xargs -P 32 -I{{}} cat {{}} > /dev/null",
                           shell=True, capture_output=True, text=True)
        Path(out, "prewarm.log").write_text(f"rc={r.returncode} secs={time.time() - t:.1f}\n{r.stderr[-2000:]}")

    if c.prewarm:
        threading.Thread(target=prewarm, daemon=True).start()
    threading.Thread(target=smi, daemon=True).start()
    threading.Thread(target=commit_loop, daemon=True).start()
    with open(f"{out}/engine_sampler.log", "w") as log:
        rc = subprocess.run([c.python, "-m", "yeto.rl.engine.engine_sampler", "--spec", f"{out}/spec.json"],
                            env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    stop.set()
    logs.commit()
    return {"rc": rc, "out": out, "secs": round(time.time() - t0, 1)}


def build_app(c: SamplerConfig) -> tuple[Any, Any]:
    """(app, sampler GPU function running :func:`sampler_body`)."""
    import modal

    image = (modal.Image.from_registry(c.image).entrypoint([])  # the image's version-assert entrypoint off
             .env({"HOME": "/root", "PYTHONUNBUFFERED": "1"})   # no PYTHONPATH: it shadows Modal's runtime
             .add_local_file(c.prompts, PROMPTS_MOUNT, copy=False)
             .add_local_python_source("yeto", copy=False, ignore=CODE_IGNORE))
    app = modal.App(c.app_name)
    models = modal.Volume.from_name(c.model_volume)
    logs = modal.Volume.from_name(c.log_volume, create_if_missing=True)
    fn = app.function(image=image, gpu=f"{c.gpu}:{c.tp}", cpu=c.cpu, memory=c.memory_gib * 1024,
                      timeout=int(c.timeout_h * 3600),
                      volumes={c.model_mount: models.read_only(), LOG_MOUNT: logs},
                      secrets=[modal.Secret.from_name(c.secret)], name="sampler")(sampler_body)
    return app, fn


def launch(c: SamplerConfig, *, wait: bool = True, log=print) -> dict[str, Any] | str:
    """Spawn the sampler; returns its result (``wait``) or the FunctionCall id."""
    from ..cost.ledger import record_safely

    app, fn = build_app(c)
    record_safely("intent", "modal_app", c.app_name, gpu=c.gpu, gpus=c.tp)
    with app.run(detach=True):
        # a detached ephemeral app per launch: its own app id ($YETO_RESOURCE_LEDGER records it,
        # so `yeto cleanup --run` stops exactly this app; the sampler outlives this process)
        if getattr(app, "app_id", None):
            record_safely("created", "modal_app", app.app_id, c.app_name, gpu=c.gpu, gpus=c.tp,
                          role="sampler", shard=c.shard)
        call = fn.spawn(c.record())
        record_safely("created", "modal_call", str(call.object_id), "sampler",
                      app_id=getattr(app, "app_id", None), app_name=c.app_name)
        log(f"[sampler] app {c.app_name} FunctionCall {call.object_id}; trainer --rl-buffer-shards "
            f"{c.trainer_buffer_shards}; stop: modal.FunctionCall.from_id(id).cancel(terminate_containers=True) "
            f"then modal app stop --yes {c.app_name}")
        return call.get() if wait else call.object_id
