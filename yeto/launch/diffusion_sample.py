"""Diffusion sampling: the sampler task and ``run_diffusion_sample``."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import threading

from .. import delivery
from ..gpu_spec import ClusterSpec, parse_gpu_spec
from . import REPO_ROOT
from .remote_setup import HF_TOKEN_ENV, HF_TOKEN_PATH, NVME_ENV, NVME_SETUP, TORCH_SETUP, WAN_TUNING
from .credentials import split_secret_envs
from .island_tasks import _refuse_island_tasks
from .images import learner_image_for
from .sky_ops import SkySDKOps, _tail
from .teardown import terminate_and_verify


DIFFUSION_SAMPLE_ADAPTER_DIR = "~/yeto-adapter"
DIFFUSION_SAMPLE_OUTPUT_DIR = "~/yeto-output"


def _diffusion_sample_adapter_mount(adapter_dir: str) -> tuple[str, dict[str, str]]:
    from ..datasource import _is_cloud_url

    if _is_cloud_url(adapter_dir):
        return DIFFUSION_SAMPLE_ADAPTER_DIR, {DIFFUSION_SAMPLE_ADAPTER_DIR: adapter_dir}
    path = os.path.expanduser(adapter_dir)
    if os.path.exists(path):
        return DIFFUSION_SAMPLE_ADAPTER_DIR, {DIFFUSION_SAMPLE_ADAPTER_DIR: path}
    raise ValueError("--adapter-dir must be an existing local path or a cloud URI")


def _add_flag(cmd: str, name: str, value) -> str:
    if value is None:
        return cmd
    return f"{cmd} --{name} {shlex.quote(str(value))}"


def make_diffusion_sample_task(args, spec: ClusterSpec):
    _refuse_island_tasks(args, "diffusion sample")
    import sky

    from ..datasource import learner_data_arg, learner_file_mounts

    adapter_arg, file_mounts = _diffusion_sample_adapter_mount(args.adapter_dir)
    sample_cmd = (
        "python3 -m yeto.diffusion.sample"
        f" --adapter-dir {shlex.quote(adapter_arg)}"
        f" --dtype {shlex.quote(args.dtype)}"
        f" --num-inference-steps {int(args.num_inference_steps)}"
        f" --fps {int(args.fps)}"
    )
    if args.data:
        sample_cmd += (
            f" --data {shlex.quote(learner_data_arg(args.data))}"
            f" --output-dir {shlex.quote(DIFFUSION_SAMPLE_OUTPUT_DIR)}"
            f" --prompt-column {shlex.quote(args.prompt_column)}"
        )
        if args.seed_column:
            sample_cmd += f" --seed-column {shlex.quote(args.seed_column)}"
        if args.max_rows is not None:
            sample_cmd += f" --max-rows {int(args.max_rows)}"
        file_mounts.update(learner_file_mounts(args.data))
    else:
        sample_cmd += (
            f" --prompt {shlex.quote(args.prompt)}"
            f" --output {shlex.quote(DIFFUSION_SAMPLE_OUTPUT_DIR + '/sample.png')}"
        )
    for name in (
        "model",
        "model_revision",
        "source_sha256",
        "diffusion_adapter",
        "diffusion_adapter_sha256",
        "model_requested_identifier",
        "model_requested_revision",
        "data_requested_identifier",
        "data_requested_revision",
        "guidance_scale",
        "height",
        "width",
        "num_frames",
        "seed",
    ):
        sample_cmd = _add_flag(sample_cmd, name.replace("_", "-"), getattr(args, name, None))
    if getattr(args, "data_revision", None):
        sample_cmd = _add_flag(sample_cmd, "data-revision", args.data_revision)
    if getattr(args, "trust_remote_code", False):
        sample_cmd += " --trust-remote-code"
    if getattr(args, "allow_unattested_legacy_adapter", False):
        sample_cmd += " --allow-unattested-legacy-adapter"

    local_token = os.path.expanduser(HF_TOKEN_PATH)
    if os.path.isfile(local_token):
        file_mounts[HF_TOKEN_PATH] = local_token
    setup_steps = [
        WAN_TUNING,
        NVME_SETUP,
        NVME_ENV,
        HF_TOKEN_ENV,
        TORCH_SETUP,
        "pip install -q -r requirements.txt",
        "pip install -q 'diffusers>=0.35' safetensors pillow 'imageio[ffmpeg]' 'bitsandbytes>=0.46.1'",
    ]
    run = f"{NVME_ENV}\n{HF_TOKEN_ENV}\nmkdir -p {DIFFUSION_SAMPLE_OUTPUT_DIR}\n{sample_cmd}"
    envs = {"HF_HUB_ENABLE_HF_TRANSFER": "1"}
    if os.environ.get("HF_TOKEN"):
        envs["HF_TOKEN"] = os.environ["HF_TOKEN"]
    envs, env_secrets = split_secret_envs(envs)
    task = sky.Task(
        name="yeto-diffusion-sample",
        setup="\n".join(setup_steps),
        run=run,
        envs=envs,
        **({"secrets": env_secrets} if env_secrets else {}),
        workdir=str(REPO_ROOT),
        file_mounts=file_mounts or None,
    )
    infra = f"{spec.cloud}/{spec.region}" if spec.region else spec.cloud
    resources_kwargs = {}
    image = learner_image_for(args, spec)
    if image is not None:
        resources_kwargs["image_id"] = image
    task.set_resources(
        sky.Resources(
            infra=infra,
            accelerators=spec.accelerators,
            cpus=getattr(args, "learner_cpus", None),
            instance_type=getattr(args, "learner_instance_type", None),
            use_spot=args.spot,
            disk_size=args.disk_size,
            **resources_kwargs,
        )
    )
    return task


def _status_terminal(status) -> bool:
    if status is None:
        return False
    is_terminal = getattr(status, "is_terminal", None)
    if callable(is_terminal):
        return bool(is_terminal())
    text = str(status)
    return any(s in text for s in ("SUCCEEDED", "FAILED", "CANCELLED", "STOPPED"))


def _status_succeeded(status) -> bool:
    return status is not None and "SUCCEEDED" in str(status)


def _wait_for_terminal_job(cluster: str, job_id: int, poll_interval: int = 30):
    ops = SkySDKOps()
    while True:
        status = ops.job_status(cluster, job_id)
        if _status_terminal(status):
            return status
        ops.sleep(max(1, poll_interval))


def run_diffusion_sample(args) -> int:
    from ..datasource import kind as data_kind
    from ..models import resolve
    from ..provenance import (
        file_sha256,
        python_spec_path,
        python_spec_sha256,
        resolve_reference,
        verify_source_tree_sha256,
    )

    args.source_sha256 = verify_source_tree_sha256(
        getattr(args, "source_sha256", None)
    )
    if getattr(args, "diffusion_adapter", None):
        expected_adapter_sha256 = getattr(
            args, "diffusion_adapter_sha256", None
        )
        adapter_path = python_spec_path(args.diffusion_adapter, base_dir=REPO_ROOT)
        try:
            relative_adapter_path = adapter_path.relative_to(REPO_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(
                f"diffusion adapter source {adapter_path} is outside the synced "
                "Yeto workdir; copy it into the repository before sampling"
            ) from exc
        target, separator, factory_name = args.diffusion_adapter.partition(":")
        if target.endswith(".py") or os.path.sep in target:
            args.diffusion_adapter = (
                f"{relative_adapter_path.as_posix()}{separator}{factory_name}"
            )
            actual_adapter_sha256 = file_sha256(adapter_path)
        else:
            actual_adapter_sha256 = python_spec_sha256(
                args.diffusion_adapter
            )
        if (
            expected_adapter_sha256 is not None
            and actual_adapter_sha256 != expected_adapter_sha256.lower()
        ):
            raise ValueError(
                "diffusion adapter SHA256 does not match the expected sampler "
                "attestation"
            )
        args.diffusion_adapter_sha256 = actual_adapter_sha256

    if getattr(args, "model", None):
        model_record = resolve_reference(
            resolve(args.model),
            getattr(args, "model_revision", None),
            repo_type="model",
            original_identifier=args.model,
        )
        args.model_revision = model_record["resolved_revision"]
        args.model_requested_identifier = model_record["requested_identifier"]
        args.model_requested_revision = model_record["requested_revision"]
    if getattr(args, "data", None):
        kind = data_kind(args.data)
        if kind == "hf":
            data_record = resolve_reference(
                args.data,
                getattr(args, "data_revision", None),
                repo_type="dataset",
            )
            args.data_revision = data_record["resolved_revision"]
            args.data_requested_identifier = data_record["requested_identifier"]
            args.data_requested_revision = data_record["requested_revision"]
        elif getattr(args, "data_revision", None) is not None:
            raise ValueError("--data-revision applies only to a Hugging Face prompt dataset")
        else:
            args.data_requested_identifier = args.data
            args.data_requested_revision = None
    specs = parse_gpu_spec(args.gpu)
    if len(specs) != 1:
        raise ValueError("diffusion sampling expects exactly one --gpu cluster")
    spec = specs[0]
    if spec.num_nodes != 1 or spec.total_gpus != 1:
        raise ValueError("diffusion sampling currently expects one single-GPU node")

    import sky

    cluster = f"{args.cluster_prefix}-sample"
    task = make_diffusion_sample_task(args, spec)
    output = getattr(args, "output", None)
    local_dest = (
        os.path.expanduser(output)
        if output and delivery.kind(output) == "local"
        else os.path.expanduser(DIFFUSION_SAMPLE_OUTPUT_DIR)
    )
    try:
        print(f"[launcher] launching diffusion sampler on {spec} as {cluster}")
        job_id, _handle = sky.stream_and_get(
            sky.launch(task, cluster_name=cluster, retry_until_up=args.retry_until_up)
        )
        threading.Thread(target=_tail, args=(cluster, job_id, "sample"), daemon=True).start()
        status = _wait_for_terminal_job(
            cluster, job_id, getattr(args, "controller_poll", 30)
        )
        if not _status_succeeded(status):
            print(f"[launcher] diffusion sample job ended as {status}", file=sys.stderr)
            return 1
        os.makedirs(local_dest, exist_ok=True)
        subprocess.run(delivery.fetch_cmd(cluster, local_dest), check=True)
        print(f"[launcher] diffusion samples fetched to {local_dest}")
        if delivery.is_remote(output):
            delivery.deliver(output, local_dest)
            print(f"[launcher] diffusion samples uploaded to {output}")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"[launcher] fetching {cluster}:~/yeto-output failed ({e})", file=sys.stderr)
        return 2
    finally:
        if args.keep:
            print(f"[launcher] keeping cluster: {cluster}")
        else:
            print(f"[launcher] tearing down {cluster}")
            terminate_and_verify(sky, cluster)
