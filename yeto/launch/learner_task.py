"""The SkyPilot task of one non-RL learner island."""

from __future__ import annotations

import os
import shlex

from ..gpu_spec import ClusterSpec
from . import REPO_ROOT
from .remote_setup import (
    HF_TOKEN_ENV, HF_TOKEN_PATH, HUNYUAN3D_ROOT, HUNYUAN3D_SETUP, NVME_ENV, NVME_SETUP,
    PROTENIX_SETUP, TORCH_SETUP, WAN_TUNING, causal_kernel_setup_steps,
)
from .topology import multinode_network_tier
from .loss import pickled_loss_path
from .credentials import split_secret_envs
from .island_tasks import _refuse_island_tasks
from .images import learner_image_for


PROTENIX_DIFFUSION_ADAPTER = "yeto.diffusion.adapters.protenix:make_adapter"
HUNYUAN3D_DIFFUSION_ADAPTER = "yeto.diffusion.adapters.hunyuan3d:make_adapter"
ALPHAFOLD3_DIFFUSION_ADAPTER = "yeto.diffusion.adapters.alphafold3:make_adapter"


def _is_protenix_request(args) -> bool:
    model = getattr(args, "model", "")
    adapter = getattr(args, "diffusion_adapter", "") or ""
    return model in {"protenix", "protenix-v2"} or "protenix" in adapter


def _is_hunyuan3d_request(args) -> bool:
    model = getattr(args, "model", "")
    adapter = getattr(args, "diffusion_adapter", "") or ""
    return model in {"hunyuan3d-21"} or "hunyuan3d" in adapter


def _is_alphafold3_request(args) -> bool:
    model = getattr(args, "model", "")
    adapter = getattr(args, "diffusion_adapter", "") or ""
    return model == "alphafold3" or "alphafold3" in adapter


def make_learner_task(args, spec: ClusterSpec, learner_id: int, num_learners: int, syncer_addr: str):
    _refuse_island_tasks(args, "SFT learner")
    import sky

    from ..adapter_lifecycle import (
        learner_parent_arg,
        learner_parent_mounts,
        selected_parent,
    )
    from ..datasource import learner_data_arg, learner_file_mounts
    from ..models import resolve_model_kind

    model_kind = resolve_model_kind(args.model, getattr(args, "model_kind", "auto"))
    if model_kind != "diffusion" and getattr(args, "diffusion_adapter", None):
        raise ValueError("--diffusion-adapter applies only to diffusion models")
    loss_function = args.loss_function
    if model_kind == "diffusion" and loss_function == "cross_entropy":
        loss_function = "flow_matching"

    backend = getattr(args, "island_backend", "torch")
    attention_backend = getattr(args, "attention_backend", "auto")
    kernel_backend = getattr(args, "kernel_backend", "native")
    data_format = getattr(args, "data_format", "auto")
    base_quantization = getattr(args, "base_quantization", "none")
    parent_mode, parent_source = selected_parent(args)
    if parent_source is not None:
        if model_kind != "causal-lm":
            raise ValueError(
                "--resume-from/--branch-from apply only to causal-LM models"
            )
        if backend != "torch":
            raise ValueError(
                "--resume-from/--branch-from require --island-backend torch"
            )
        if args.tuning != "lora":
            raise ValueError(
                "--resume-from/--branch-from require --tuning lora"
            )
        if not getattr(args, "adapter_sha256", None):
            raise ValueError(
                "--resume-from/--branch-from must be attested before learner launch"
            )
    if model_kind != "causal-lm" and data_format != "auto":
        raise ValueError("--data-format applies only to causal-LM models")
    if model_kind != "causal-lm" and (
        attention_backend != "auto" or kernel_backend != "native"
    ):
        raise ValueError(
            "--attention-backend and --kernel-backend apply only to causal-LM models"
        )
    if model_kind == "causal-lm" and backend != "torch":
        if attention_backend != "auto" or kernel_backend != "native":
            raise ValueError(
                "--attention-backend and --kernel-backend are supported only "
                "by the torch causal-LM island backend"
            )
    if base_quantization != "none":
        if model_kind != "causal-lm":
            raise ValueError("--base-quantization applies only to causal-LM models")
        if backend != "torch":
            raise ValueError(
                "--base-quantization nf4 is supported only by the torch "
                "causal-LM island backend"
            )
        if args.tuning != "lora":
            raise ValueError("--base-quantization nf4 requires --tuning lora")
        if args.shard != "ddp":
            raise ValueError("--base-quantization nf4 requires --shard ddp")
        if kernel_backend != "native":
            raise ValueError(
                "--base-quantization nf4 requires --kernel-backend native"
            )
    if kernel_backend == "liger" and loss_function != "cross_entropy":
        raise ValueError(
            "--kernel-backend liger fused-linear-CE supports only the built-in "
            "cross_entropy loss"
        )
    if kernel_backend == "liger" and args.tuning != "lora":
        raise ValueError(
            "--kernel-backend liger fused-linear-CE is production-approved only "
            "for --tuning lora"
        )
    if kernel_backend == "liger" and args.shard != "ddp":
        raise ValueError(
            "--kernel-backend liger fused-linear-CE is production-approved only "
            "for --shard ddp until FSDP has separate CUDA parity evidence"
        )

    # Flags shared by all learners. The DiLoCo sync, LoRA, and data source
    # shape are identical; the per-task forward/loss loop differs.
    common_flags = (
        f" --model {shlex.quote(args.model)}"
        f" --data {shlex.quote(learner_data_arg(args.data))}"
        f" --syncer $SYNCER_ADDR"
        f" --learner-id $LEARNER_ID"
        f" --num-learners {num_learners}"
        f" --loss-function {shlex.quote(loss_function)}"
        f" --tuning {args.tuning}"
        f" --lora-r {args.lora_r}"
        f" --lora-alpha {getattr(args, 'lora_alpha', 32)}"
        f" --lora-targets {getattr(args, 'lora_targets', 'auto')}"
        f" --micro-batch-size {args.micro_batch_size}"
        f" --grad-accum {args.grad_accum}"
        f" --inner-lr {args.inner_lr}"
        f" --fragments {args.fragments}"
        f" --fragment-pattern {args.fragment_pattern}"
        f" --matrix-merge {getattr(args, 'matrix_merge', 'rda')}"
        f" --merge-alpha {args.merge_alpha}"
        f" --wire-dtype {args.wire_dtype}"
        f" --wan-streams {args.wan_streams}"
        f" --output-dir ~/yeto-output"
    )
    parent_arg = learner_parent_arg(args)
    if parent_mode is not None and parent_arg is not None:
        common_flags += f" --{parent_mode}-from {shlex.quote(parent_arg)}"
        common_flags += (
            f" --adapter-sha256 {shlex.quote(args.adapter_sha256)}"
        )
    if getattr(args, "model_revision", None):
        common_flags += f" --model-revision {shlex.quote(args.model_revision)}"
    if getattr(args, "data_revision", None):
        common_flags += f" --data-revision {shlex.quote(args.data_revision)}"
    if getattr(args, "trust_remote_code", False):
        common_flags += " --trust-remote-code"
    if getattr(args, "allow_unsafe_pickled_loss", False):
        common_flags += " --allow-unsafe-pickled-loss"
    if getattr(args, "loss_sha256", None):
        common_flags += f" --loss-sha256 {shlex.quote(args.loss_sha256)}"
    if getattr(args, "source_sha256", None):
        common_flags += f" --source-sha256 {shlex.quote(args.source_sha256)}"
    if getattr(args, "wandb", False):
        common_flags += " --wandb"
        common_flags += f" --wandb-project {shlex.quote(args.wandb_project)}"
        common_flags += f" --wandb-mode {args.wandb_mode}"
        if getattr(args, "wandb_entity", None):
            common_flags += f" --wandb-entity {shlex.quote(args.wandb_entity)}"
    for name in (
        "model_requested_identifier",
        "model_requested_revision",
        "data_requested_identifier",
        "data_requested_revision",
    ):
        value = getattr(args, name, None)
        if value is not None:
            common_flags += (
                f" --{name.replace('_', '-')} {shlex.quote(str(value))}"
            )
    learner_flags = common_flags
    if model_kind == "causal-lm":
        learner_flags += (
            f" --train-on {args.train_on}"
            f" --assistant-mask-mode {getattr(args, 'assistant_mask_mode', 'native')}"
            f" --data-format {getattr(args, 'data_format', 'auto')}"
            f" --seed {getattr(args, 'seed', 0)}"
            f" --seq-len {args.seq_len}"
            f" --tokenize {args.tokenize}"
            f" --stream-workers {args.stream_workers}"
        )
        if backend == "torch":
            learner_flags += (
                f" --base-quantization {base_quantization}"
                f" --attention-backend {attention_backend}"
                f" --kernel-backend {kernel_backend}"
            )
    else:
        if getattr(args, "island_backend", "torch") != "torch":
            raise ValueError("diffusion model-kind uses the torch island backend, not megatron")
        learner_flags += (
            f" --shard {args.shard}"
            f" --image-column {shlex.quote(args.image_column)}"
            f" --video-column {shlex.quote(args.video_column)}"
            f" --prompt-column {shlex.quote(args.prompt_column)}"
            f" --latent-column {shlex.quote(args.latent_column)}"
            f" --text-embeds-column {shlex.quote(args.text_embeds_column)}"
            f" --text-attention-mask-column {shlex.quote(args.text_attention_mask_column)}"
            f" --pooled-text-embeds-column {shlex.quote(args.pooled_text_embeds_column)}"
            f" --resize-mode {shlex.quote(getattr(args, 'resize_mode', 'stretch'))}"
            f" --stream-workers {args.stream_workers}"
        )
        diffusion_adapter = getattr(args, "diffusion_adapter", None)
        if _is_protenix_request(args) and not diffusion_adapter:
            diffusion_adapter = PROTENIX_DIFFUSION_ADAPTER
        if _is_hunyuan3d_request(args) and not diffusion_adapter:
            diffusion_adapter = HUNYUAN3D_DIFFUSION_ADAPTER
        if _is_alphafold3_request(args) and not diffusion_adapter:
            diffusion_adapter = ALPHAFOLD3_DIFFUSION_ADAPTER
        if diffusion_adapter:
            learner_flags += f" --diffusion-adapter {shlex.quote(diffusion_adapter)}"
            if getattr(args, "diffusion_adapter_sha256", None):
                learner_flags += (
                    " --diffusion-adapter-sha256 "
                    f"{shlex.quote(args.diffusion_adapter_sha256)}"
                )
        if getattr(args, "diffusion_seed", None) is not None:
            learner_flags += f" --seed {args.diffusion_seed}"
        if getattr(args, "cache_latents", False):
            learner_flags += " --cache-latents"
        if getattr(args, "cache_text_embeds", False):
            learner_flags += " --cache-text-embeds"
        if getattr(args, "bucket_by_shape", False):
            learner_flags += " --bucket-by-shape"
        if getattr(args, "diffusion_loss_weighting", "none") != "none":
            learner_flags += f" --diffusion-loss-weighting {args.diffusion_loss_weighting}"
            learner_flags += f" --diffusion-min-snr-gamma {args.diffusion_min_snr_gamma}"
        if args.height:
            learner_flags += f" --height {args.height}"
        if args.width:
            learner_flags += f" --width {args.width}"
        if args.num_frames:
            learner_flags += f" --num-frames {args.num_frames}"
        if getattr(args, "fps", None):
            learner_flags += f" --fps {args.fps}"
    if args.max_rows:
        learner_flags += f" --max-rows {args.max_rows}"

    if model_kind == "diffusion":
        entrypoint = "yeto.diffusion.learner"
        setup_steps = [
            WAN_TUNING,
            NVME_SETUP,
            NVME_ENV,
            HF_TOKEN_ENV,
            TORCH_SETUP,
            "pip install -q -r requirements.txt",
            "pip install -q 'diffusers>=0.35' safetensors pillow 'imageio[ffmpeg]' 'bitsandbytes>=0.46.1'",
        ]
        if _is_protenix_request(args):
            setup_steps.append(PROTENIX_SETUP)
        if _is_hunyuan3d_request(args):
            setup_steps.append(HUNYUAN3D_SETUP)
    elif backend == "megatron":
        gpus = spec.num_nodes * spec.gpus_per_node
        tp = max(1, getattr(args, "tensor_parallel", 1))
        pp = max(1, getattr(args, "pipeline_parallel", 1))
        ep = getattr(args, "expert_parallel", None) or max(1, gpus // (tp * pp))
        learner_flags += (
            f" --island-backend megatron"
            f" --expert-parallel {ep}"
            f" --tensor-parallel {tp}"
            f" --pipeline-parallel {pp}"
        )
        entrypoint = "yeto.megatron.learner"
        # Inside the NGC container the whole training stack (torch, TE,
        # megatron-core, bridge) is already present, so skip TORCH_SETUP and
        # MEGATRON_SETUP. NVME_SETUP is a host RAID operation that can't run in
        # a container, so it's skipped too (HF cache lands on the container's
        # disk — slower download, but correct; wiring the host instance-store
        # through to the container is a follow-up). Only yeto's pure-python
        # deps that the container may lack are added, --no-deps so they never
        # perturb the container's pinned torch/TE/transformers.
        setup_steps = [
            WAN_TUNING,
            HF_TOKEN_ENV,
            "pip install -q --no-deps datasets peft hf_transfer cloudpickle sentencepiece",
        ]
    else:
        learner_flags += f" --shard {args.shard}"
        entrypoint = "yeto.learner"
        setup_steps = [WAN_TUNING, NVME_SETUP, NVME_ENV, HF_TOKEN_ENV, TORCH_SETUP,
                       "pip install -q -r requirements.txt"]
        setup_steps.extend(causal_kernel_setup_steps(args))

    if getattr(args, "wandb", False):
        # Telemetry is never worth losing an island over: a failed install
        # is reported and the learner trains on (yeto.wandb_logger falls
        # back to a no-op sink when the import fails).
        setup_steps.append(
            "pip install -q wandb || echo '[yeto-setup] wandb install failed; "
            "the island will train without telemetry' >&2"
        )

    run = (
        f"{NVME_ENV}\n"
        f"{HF_TOKEN_ENV}\n"
        'MASTER_ADDR=$(echo "$SKYPILOT_NODE_IPS" | head -n1)\n'
        "torchrun --nnodes=$SKYPILOT_NUM_NODES --node_rank=$SKYPILOT_NODE_RANK "
        "--nproc_per_node=$SKYPILOT_NUM_GPUS_PER_NODE "
        "--master_addr=$MASTER_ADDR --master_port=29500 "
        f"-m {entrypoint}{learner_flags}"
    )
    envs = {
        "SYNCER_ADDR": syncer_addr,
        "LEARNER_ID": str(learner_id),
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }
    if os.environ.get("HF_TOKEN"):
        envs["HF_TOKEN"] = os.environ["HF_TOKEN"]
    if getattr(args, "wandb", False):
        # Every island joins the fleet's W&B group under the run's name, so
        # the islands and the syncer's tape run land on one comparison view.
        envs["YETO_RUN_GROUP"] = args.cluster_prefix
        if os.environ.get("WANDB_API_KEY"):
            envs["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    if spec.num_nodes > 1:
        # Surface NCCL's chosen transport in the job logs so an EFA-less
        # fallback to TCP sockets is visible, not silent.
        envs["NCCL_DEBUG"] = "INFO"
    if _is_hunyuan3d_request(args):
        envs["YETO_HUNYUAN3D_ROOT"] = HUNYUAN3D_ROOT
    # Non-HF --data sources (local paths, s3://, gs://, ...) ride sky's
    # file_mounts onto every learner; see yeto/datasource.py.
    file_mounts = dict(learner_file_mounts(args.data))
    file_mounts.update(learner_parent_mounts(args))
    file_mounts = file_mounts or None
    if args.loss_function.startswith("pickle:"):
        # The pickled loss is gitignored, so the workdir sync skips it;
        # mount it into the workdir explicitly.
        file_mounts = file_mounts or {}
        loss_path = pickled_loss_path(args.loss_function)
        file_mounts[f"~/sky_workdir/{loss_path.name}"] = str(loss_path)
    # Ride the launching machine's HF token onto every learner: anonymous
    # Hub quota is half the authenticated one and shared per-IP, and a
    # gated/private --model needs the token outright. HF_TOKEN_ENV then
    # copies it wherever NVME_ENV points HF_HOME.
    local_token = os.path.expanduser(HF_TOKEN_PATH)
    if os.path.isfile(local_token):
        file_mounts = file_mounts or {}
        file_mounts[HF_TOKEN_PATH] = local_token
    # Kick the weight download off in the background at the END of setup:
    # it overlaps sky's remaining bookkeeping and races ahead of the run
    # command, which then finds a warm (or warming — hf resumes) cache.
    # hf_transfer multi-streams the download; NVMe absorbs it at GB/s.
    from ..models import resolve
    from ..provenance import is_local_reference

    repo = resolve(args.model)
    if _is_protenix_request(args):
        prefetch = "true  # Protenix uses native model names/checkpoints, not HF prefetch"
    elif _is_alphafold3_request(args):
        prefetch = "true  # AlphaFold3 requires local authorized parameters, not HF prefetch"
    elif is_local_reference(repo):
        prefetch = ": # local model; no Hub prefetch"
    else:
        revision_flag = (
            f" --revision {shlex.quote(args.model_revision)}"
            if getattr(args, "model_revision", None)
            else ""
        )
        prefetch = (
            f"(nohup huggingface-cli download {shlex.quote(repo)}{revision_flag} "
            ">/tmp/hf-prefetch.log 2>&1 &) || true"
        )
    run = (
        f"{NVME_ENV}\n"
        f"{HF_TOKEN_ENV}\n"
        'MASTER_ADDR=$(echo "$SKYPILOT_NODE_IPS" | head -n1)\n'
        "torchrun --nnodes=$SKYPILOT_NUM_NODES --node_rank=$SKYPILOT_NODE_RANK "
        "--nproc_per_node=$SKYPILOT_NUM_GPUS_PER_NODE "
        "--master_addr=$MASTER_ADDR --master_port=29500 "
        f"-m {entrypoint}{learner_flags}"
    )
    envs, env_secrets = split_secret_envs(envs)
    task = sky.Task(
        name=f"yeto-learner-{learner_id}",
        setup="\n".join(setup_steps + [prefetch]),
        run=run,
        envs=envs,
        **({"secrets": env_secrets} if env_secrets else {}),
        num_nodes=spec.num_nodes,
        workdir=str(REPO_ROOT),
        file_mounts=file_mounts,
    )
    infra = f"{spec.cloud}/{spec.region}" if spec.region else spec.cloud
    resources_kwargs = {}
    image = learner_image_for(args, spec, learner_id)
    if image is not None:
        resources_kwargs["image_id"] = image
    if spec.num_nodes > 1:
        # Multi-node learner: inner DDP all-reduce crosses the node fabric,
        # so request the cloud's RDMA-class interconnect (EFA on AWS,
        # GPUDirect on GCP). Single-node clusters stay on NVLink and don't
        # need it. On AWS this also swaps in the EFA-ready DLAMI; SkyPilot
        # installs no EFA software itself, and if the pinned AMI is missing
        # NCCL silently falls back to TCP — NCCL_DEBUG below makes the
        # chosen transport visible in the job logs (look for
        # "NET/OFI Selected Provider is efa").
        tier = multinode_network_tier(spec.cloud, spec.gpu, spec.gpus_per_node)
        if tier:
            resources_kwargs["network_tier"] = tier
    if spec.cloud == "modal":
        # Not a sky cloud: build_modal_island_config reads only this task's
        # run script and envs; the Modal runner sizes the container itself.
        return task
    task.set_resources(
        sky.Resources(
            infra=infra,
            accelerators=spec.accelerators,
            cpus=args.learner_cpus,
            instance_type=args.learner_instance_type,
            use_spot=args.spot,
            disk_size=args.disk_size,
            **resources_kwargs,
        )
    )
    return task
