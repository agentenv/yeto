"""sky tasks on a National Compute node pool (``infra: ssh/nc-<ticket>``).

Each builder returns the task as a sky YAML dict (``sky.Task.from_yaml_config``
takes it), so the task is plain data the tests can read:

* ``hello_task``: the GPUs the pod sees (``/dev/kfd``, render nodes) and the NVMe
  host path;
* ``model_pull_task``: the base model from the object store onto
  ``/mnt/nvme/models`` with the bandwidth cap (``aws s3`` ``max_bandwidth``,
  default 2500 MB/s: at ~7 GB/s an MI355X node drops off the network within two
  minutes and does not come back, GPU-B report section 4.1);
* ``sampler_task``: ``python -m yeto.rl.engine.engine_sampler --spec`` in the
  serving image (vLLM ``kimi-k31`` with the ``amd-article`` preset in the spec),
  writing its own buffer shard (``sampler_id``) of the run prefix.

Object store credentials go in as sky ``secrets`` read from an env file
(``--r2-env-file``) at submit time; they are not in the task file or the repo.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from yeto.rl.sampler_pins import VLLM_SAMPLER_IMAGE_ROCM

from .node_pool import SKY_ACCELERATOR

NVME = "/mnt/nvme"
SERVING_IMAGE = VLLM_SAMPLER_IMAGE_ROCM.removeprefix("docker:")  # pinned digest, not the tag
MAX_PULL_MB_S = 2500
HARD_PULL_LIMIT_MB_S = 3000
SECRET_KEYS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_ENDPOINT_URL", "AWS_REGION", "R2_BUCKET")


def read_env_file(path: str | Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in Path(path).expanduser().read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        out[key.strip()] = value.strip().strip("'\"")
    return out


def _nvme_pod_config(mount: str = NVME) -> dict[str, Any]:
    return {"ssh": {"pod_config": {"spec": {
        "containers": [{"volumeMounts": [{"name": "nvme", "mountPath": mount}]}],
        "volumes": [{"name": "nvme", "hostPath": {"path": NVME, "type": "Directory"}}],
    }}}}


def _resources(pool: str, gpus: int, image: str | None = None) -> dict[str, Any]:
    res: dict[str, Any] = {"infra": f"ssh/{pool}", "accelerators": f"{SKY_ACCELERATOR}:{gpus}"}
    if image:
        res["image_id"] = f"docker:{image}"
    return res


def hello_task(pool: str, *, gpus: int = 8) -> dict[str, Any]:
    run = (
        "set -u; echo YETO_NC_HELLO host=$(hostname) cpus=$(nproc) "
        "mem_gb=$(awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo); "
        "test -e /dev/kfd && echo kfd=yes || echo kfd=no; "
        "echo render_nodes=$(ls /dev/dri 2>/dev/null | grep -c renderD); "
        f"df -h {NVME} | tail -1; ls {NVME}"
    )
    return {"name": "yeto-nc-hello", "resources": _resources(pool, gpus), "run": run,
            "config": _nvme_pod_config()}


def model_pull_task(pool: str, *, src: str, dest: str, max_mb_s: int = MAX_PULL_MB_S,
                    include: str | None = None, gpus: int = 8) -> dict[str, Any]:
    if max_mb_s > HARD_PULL_LIMIT_MB_S:
        raise ValueError(f"pull cap {max_mb_s} MB/s exceeds the node limit {HARD_PULL_LIMIT_MB_S} MB/s")
    if not dest.startswith(f"{NVME}/"):
        raise ValueError(f"dest must be on {NVME}")
    sync = ["aws", "s3", "sync", src, dest, "--only-show-errors"]
    if include:
        sync += ["--exclude", "*", "--include", include]
    setup = "command -v aws >/dev/null || python3 -m pip install -q awscli"
    run = "\n".join([
        "set -eu",
        'export AWS_CONFIG_FILE="$HOME/.yeto-nc-aws-config"',
        ("printf '[default]\\nregion = %s\\ns3 =\\n  max_concurrent_requests = 32\\n"
        f"  max_bandwidth = {max_mb_s}MB/s\\n' \"${{AWS_REGION:-auto}}\" > \"$AWS_CONFIG_FILE\""),
        f"mkdir -p {shlex.quote(dest)}",
        'echo "YETO_NC_PULL start $(date -u +%FT%TZ) cap=' + f'{max_mb_s}MB/s"',
        shlex.join(sync),
        f'echo "YETO_NC_PULL done $(date -u +%FT%TZ) $(du -sh {shlex.quote(dest)} | cut -f1)"',
    ])
    return {"name": "yeto-nc-model-pull", "resources": _resources(pool, gpus), "setup": setup,
            "run": run, "config": _nvme_pod_config()}


def sampler_spec(base: Mapping[str, Any], *, sampler_id: str, shards: list[str]) -> dict[str, Any]:
    """The base engine-sampler spec with this sampler's shard and the run's shard list."""
    spec = dict(base)
    spec["sampler_id"] = sampler_id
    spec["buffer_shards"] = sorted(set(shards) | {sampler_id})
    return spec


RELOADS = ("hot", "restart")


def dispense_config(namespace: str, *, lease_s: float | None = None, max_redo: int | None = None,
                    passes: int | None = None, seed: int | None = None, holdout_file: str | None = None,
                    mix: str | None = None, labels: str | None = None,
                    by_label: bool = False) -> dict[str, Any]:
    """``prompts.dispense`` of a store-ledger sampler (``yeto.rl.engine.store_dispense``):
    every sampler of the run names the same ``namespace`` under the run's store, so
    samplers that share only the bucket share the tasks. ``holdout_file`` is read
    here (a local path) and goes into the spec as ``holdout_ids``: the sampler
    refuses rows with one of them, and the plan pins the list's digest."""
    out: dict[str, Any] = {"backend": "store", "namespace": str(namespace).strip("/")}
    if not out["namespace"] or ".." in out["namespace"].split("/"):
        raise ValueError(f"dispense namespace {namespace!r} is not a relative store path")
    for key, value in (("lease_s", lease_s), ("max_redo", max_redo), ("passes", passes), ("seed", seed)):
        if value is not None:
            out[key] = value
    if holdout_file:
        from yeto.rl.engine.store_dispense import load_holdout  # noqa: PLC0415

        ids, _digest = load_holdout(holdout_file)
        out["holdout_ids"] = sorted(ids)
        out["require_holdout"] = True
    if mix:
        out["mix"] = [w.strip() for w in mix.split(",") if w.strip()]
    if labels:
        out["labels"] = [x.strip() for x in labels.split(",") if x.strip()]
    if by_label:
        out["by_label"] = True
    return out


def apply_sampler_overrides(spec: Mapping[str, Any], *, data_slot: str | None = None,
                            drop_zero_variance: bool | None = None, thinking_effort: str | None = None,
                            reload: str | None = None, n: int | None = None,
                            max_tokens: int | None = None, continues_shard: str | None = None,
                            dispense: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The run knobs of one sampler laid over a base spec (None: keep the base value),
    with the same meaning as ``yeto engine-sampler-modal``: ``data_slot`` "I/N" (this
    sampler's prompt rows), ``drop_zero_variance`` (``drop_zero_variance_groups``),
    ``thinking_effort`` (the chat template kwarg), ``reload`` ``hot`` (in-place reload
    with expert-file updates and the probe) or ``restart``, ``n`` and ``max_tokens``
    (sampling); ``continues_shard``: the shard this replacement sampler continues;
    ``dispense``: ``prompts.dispense`` (:func:`dispense_config`; excludes ``data_slot``). Refuses a ``max_tokens`` that does not fit ``max_model_len`` with the
    spec's ``max_prompt_tokens``."""
    from yeto.rl.engine.data_split import DataSlot

    out = json.loads(json.dumps(spec))
    prompts = out.setdefault("prompts", {})
    sampling = out.setdefault("sampling", {})
    if data_slot:
        DataSlot.parse(data_slot)
        prompts["data_slot"] = data_slot
    if dispense:  # the tasks come from the store ledger, not a slot
        if data_slot or prompts.get("data_slot"):
            raise ValueError("a dispensed sampler takes no data slot")
        prompts["dispense"] = dict(dispense)
    if continues_shard:  # a replacement: continue that shard's prompts (sampler_resume handoff)
        if continues_shard == out.get("sampler_id"):
            raise ValueError("continues_shard names the sampler's own shard")
        out["continues_shard"] = continues_shard
    if drop_zero_variance is not None:
        out["drop_zero_variance_groups"] = bool(drop_zero_variance)
    if thinking_effort is not None:
        prompts["chat_template_kwargs"] = dict(prompts.get("chat_template_kwargs") or {},
                                               thinking_effort=thinking_effort)
        if out.get("agentic"):  # an agentic round renders with agentic.template_kwargs only
            out["agentic"]["template_kwargs"] = dict(out["agentic"].get("template_kwargs") or {},
                                                     thinking_effort=thinking_effort)
    if reload is not None:
        if reload not in RELOADS:
            raise ValueError(f"reload {reload!r} is not one of {RELOADS}")
        hot = reload == "hot"
        engine = out.setdefault("engine", {})
        engine.setdefault("server", {})["in_place_reload"] = hot
        if engine.get("adapter") != "vllm-server":  # SGLang's expert-file update and probe; VllmOptions has none
            options = engine.setdefault("options", {})
            options["expert_file_update"] = "auto" if hot else "off"
            options["expert_update_probe"] = hot
    if n is not None:
        if int(n) < 1:
            raise ValueError("n must be >= 1")
        sampling["n"] = int(n)
    if max_tokens is not None:
        sampling["max_tokens"] = int(max_tokens)
    limit = ((out.get("engine") or {}).get("server") or {}).get("max_model_len")
    if limit and "max_tokens" in sampling and int(sampling["max_tokens"]) + int(out.get("max_prompt_tokens") or 0) > int(limit):
        raise ValueError(f"max_tokens {sampling['max_tokens']} + max_prompt_tokens "
                         f"{out.get('max_prompt_tokens')} exceed max_model_len {limit}")
    return out


def sampler_task(pool: str, *, spec_path: str, run_prefix: str, workdir: str,
                 image: str = SERVING_IMAGE, python: str = "/opt/runtime-venv/bin/python3",
                 gpus: int = 8, protected: frozenset[str] = frozenset()) -> dict[str, Any]:
    run_prefix = run_prefix.strip("/")
    if not run_prefix:
        raise ValueError("run_prefix is required")
    if run_prefix in protected:
        raise ValueError(f"run prefix {run_prefix!r} is protected (a buffer in use)")
    envs = {
        "PYTHONUNBUFFERED": "1",
        "VLLM_USE_BREAKABLE_CUDAGRAPH": "1",
        "YETO_RL_STORE_IMPL": "yeto.cloud.s3_store:store_from_env",
        "YETO_RL_STORE_S3_STAGE": f"{NVME}/work/stage",
        "YETO_RL_STORE_S3_REGION": "auto",
        "YETO_RL_STORE_S3_CREDENTIALS": "aws-env",
        "YETO_NC_RUN_PREFIX": run_prefix,
    }
    run = "\n".join([
        "set -eu",
        f"mkdir -p {NVME}/work/stage",
        'export YETO_RL_STORE_S3_URI="s3://${R2_BUCKET}/runs/${YETO_NC_RUN_PREFIX}"',
        'export YETO_RL_STORE_S3_ENDPOINT_URL="$AWS_ENDPOINT_URL"',
        'export PYTHONPATH="$HOME/sky_workdir${PYTHONPATH:+:$PYTHONPATH}"',
        f"exec {shlex.quote(python)} -m yeto.rl.engine.engine_sampler --spec ~/yeto-nc-spec.json",
    ])
    return {"name": "yeto-nc-sampler", "resources": _resources(pool, gpus, image), "workdir": workdir,
            "file_mounts": {"~/yeto-nc-spec.json": spec_path}, "envs": envs, "run": run,
            "config": _nvme_pod_config()}


def with_secrets(task: dict[str, Any], env_file: str | Path | None) -> dict[str, Any]:
    if not env_file:
        return task
    env = read_env_file(env_file)
    missing = [k for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_ENDPOINT_URL") if not env.get(k)]
    if missing:
        raise ValueError(f"{env_file}: missing {missing}")
    return {**task, "secrets": {k: env[k] for k in SECRET_KEYS if env.get(k)}}


def write_spec(spec: Mapping[str, Any], path: str | Path) -> str:
    Path(path).write_text(json.dumps(spec, indent=1))
    return str(path)


def backlog_groups(*, groups_per_round: int, synced: bool, trainers: int = 1, samplers: int = 1,
                   fragments: int = 1, local_rounds: int = 1, pipeline: int = 0,
                   explicit_rounds: int | None = None) -> int:
    """``max_backlog_groups`` of the sampler, sized by :mod:`yeto.rl.engine.sample_roles`
    as the launcher sizes it: a plain (no-sync) sampler takes
    ``sampler_backlog_rounds`` (the operator's value or the default); a sampler of a
    full-parameter synced group takes the stall-free minimum
    (``stall_free_backlog_rounds``) plus the default margin, and an explicit value
    below the minimum is refused."""
    from yeto.rl.engine import sample_roles

    if not synced:
        return sample_roles.sampler_backlog_rounds(explicit_rounds) * groups_per_round
    need = sample_roles.stall_free_backlog_rounds(trainers_fed=trainers, sharing=samplers, fragments=fragments,
                                                  local_rounds=local_rounds, pipeline=pipeline)
    if explicit_rounds is None:
        return (need + sample_roles.DEFAULT_MAX_BACKLOG_ROUNDS) * groups_per_round
    if explicit_rounds < need:
        raise ValueError(f"backlog {explicit_rounds} rounds stalls the group: at least {need} rounds")
    return explicit_rounds * groups_per_round
