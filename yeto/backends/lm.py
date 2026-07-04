"""The ``lm`` task backend: fine-tune text language models.

This is the reference backend and yeto's default. It carries the two island
engines — ``torch`` (FSDP2/DDP) and ``megatron`` (Megatron-Core expert/tensor/
pipeline parallelism) — which share every DiLoCo/LoRA/data/loss flag and differ
only in the intra-island trainer.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys

from .. import launcher
from .base import IslandEngine, TaskBackend, engine_names, get_engine, register_backend, register_engine

LOSS_FUNCTIONS = ("cross_entropy", "importance_sampling", "ppo", "cispo", "dro")


# ---------------------------------------------------------------------------
# island engines


class TorchEngine(IslandEngine):
    name = "torch"

    def entrypoint(self) -> str:
        return "yeto.learner"

    def extra_learner_flags(self, args, spec) -> str:
        return f" --shard {args.shard}"

    def setup_steps(self, args) -> list[str]:
        return [
            launcher.WAN_TUNING,
            launcher.NVME_SETUP,
            launcher.NVME_ENV,
            launcher.HF_TOKEN_ENV,
            launcher.TORCH_SETUP,
            "pip install -q -r requirements.txt",
        ]


class MegatronEngine(IslandEngine):
    name = "megatron"

    def entrypoint(self) -> str:
        return "yeto.megatron.learner"

    def extra_learner_flags(self, args, spec) -> str:
        gpus = spec.num_nodes * spec.gpus_per_node
        tp = max(1, getattr(args, "tensor_parallel", 1))
        pp = max(1, getattr(args, "pipeline_parallel", 1))
        ep = getattr(args, "expert_parallel", None) or max(1, gpus // (tp * pp))
        return (
            f" --island-backend megatron"
            f" --expert-parallel {ep}"
            f" --tensor-parallel {tp}"
            f" --pipeline-parallel {pp}"
        )

    def setup_steps(self, args) -> list[str]:
        # Inside the NGC container the whole training stack (torch, TE,
        # megatron-core, bridge) is already present, so skip TORCH_SETUP and
        # MEGATRON_SETUP. NVME_SETUP is a host RAID operation that can't run in
        # a container, so it's skipped too (HF cache lands on the container's
        # disk — slower download, but correct). Only yeto's pure-python deps
        # that the container may lack are added, --no-deps so they never
        # perturb the container's pinned torch/TE/transformers.
        return [
            launcher.WAN_TUNING,
            launcher.HF_TOKEN_ENV,
            "pip install -q --no-deps datasets peft hf_transfer cloudpickle sentencepiece",
        ]

    def image(self, args, spec):
        return launcher.MEGATRON_IMAGE


register_engine(TorchEngine())
register_engine(MegatronEngine())


def _int_or_auto(value: str):
    # duplicated from yeto/autobatch.py: importing it would pull torch into the
    # CLI path.
    return value if value == "auto" else int(value)


def _loss_spec(value: str) -> str:
    if value in LOSS_FUNCTIONS or value.startswith(("custom:", "pickle:")):
        return value
    raise argparse.ArgumentTypeError(
        f"expected one of {LOSS_FUNCTIONS} or custom:<file.py>[:<fn>]"
    )


# ---------------------------------------------------------------------------
# backend


class LMBackend(TaskBackend):
    name = "lm"
    supports_auto_fleet = True
    output_dir = "yeto-output"

    def add_launch_cli_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model",
            required=False,
            help="LM model alias (see yeto/models.py: gemma4, qwen35-9b, "
            "llama31-8b, gptoss-120b, ...) or any HF id; required for --task lm",
        )
        parser.add_argument(
            "--data",
            required=False,
            help="fine-tuning data: HF dataset id, local path (dir/file of "
            "jsonl/json/parquet or a save_to_disk dir), or a cloud URI "
            "(s3://, gs://, r2://, ...) — non-HF sources are shipped to learners "
            "via SkyPilot file mounts; rows are messages-format chat traces; "
            "required for --task lm",
        )
        parser.add_argument(
            "--loss-function",
            type=_loss_spec,
            default="cross_entropy",
            help=f"one of {'|'.join(LOSS_FUNCTIONS)}, or custom:<file.py>[:<fn>] "
            "defining fn(logits, input_ids, weights) -> (loss, num_tokens); the "
            "callable is pickled by value and shipped to all learners",
        )

        tune = parser.add_argument_group("LM fine-tuning")
        tune.add_argument("--tuning", choices=["lora", "full"], default="lora")
        tune.add_argument(
            "--island-backend",
            choices=engine_names(),
            default="torch",
            help="per-island trainer: 'torch' (FSDP2/DDP; any bf16 model that "
            "fits the island) or 'megatron' (Megatron-Core expert/tensor/pipeline "
            "parallelism, for large MoE whose experts must be sharded across the "
            "island). Both speak the same DiLoCo adapter sync to the syncer",
        )
        tune.add_argument(
            "--expert-parallel",
            type=int,
            default=None,
            help="megatron backend: expert-parallel degree (default fills the "
            "island: gpus_per_island / (tensor-parallel x pipeline-parallel))",
        )
        tune.add_argument("--tensor-parallel", type=int, default=1, help="megatron backend: TP degree")
        tune.add_argument("--pipeline-parallel", type=int, default=1, help="megatron backend: PP degree")
        tune.add_argument(
            "--train-on",
            choices=["assistant", "all"],
            default="assistant",
            help="which tokens carry loss: assistant-message tokens only (default) or every token",
        )
        tune.add_argument("--lora-r", type=int, default=16)
        tune.add_argument(
            "--lora-targets",
            choices=["auto", "attention", "all-linear"],
            default="auto",
            help="adapter placement: attention-only, every linear, or auto "
            "(attention for MoE — router and routed experts stay frozen)",
        )
        tune.add_argument("--seq-len", type=int, default=2048)
        tune.add_argument(
            "--micro-batch-size",
            type=_int_or_auto,
            default="auto",
            help="per-GPU micro batch; 'auto' (default) probes the largest size "
            "that fits each learner's VRAM at startup and shrinks --grad-accum "
            "to keep the effective batch constant",
        )
        tune.add_argument("--grad-accum", type=int, default=4)
        tune.add_argument("--inner-lr", type=float, default=3e-4)
        tune.add_argument("--max-rows", type=int, default=None, help="cap dataset rows per learner")
        tune.add_argument(
            "--tokenize",
            choices=["stream", "preload"],
            default="stream",
            help="stream: async tokenization in DataLoader workers (default); preload: all upfront",
        )
        tune.add_argument(
            "--stream-workers",
            type=int,
            default=2,
            help="tokenizer worker processes per learner rank (stream mode)",
        )

    def add_export_cli_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model",
            required=False,
            help="HF model id or an alias from yeto/models.py; required for --task lm",
        )
        parser.add_argument("--tuning", choices=["lora", "full"], default="lora")
        parser.add_argument("--lora-r", type=int, default=16)
        parser.add_argument("--lora-alpha", type=int, default=32)
        parser.add_argument(
            "--lora-targets",
            choices=["auto", "attention", "all-linear"],
            default="auto",
            help="adapter placement used during training (layout must match)",
        )

    def validate(self, args) -> list[str]:
        missing = [name for name in ("model", "data") if not getattr(args, name, None)]
        if missing:
            flags = ", ".join("--" + name for name in missing)
            return [f"--task lm requires: {flags}"]
        return []

    def image_override(self, args, spec):
        return get_engine(getattr(args, "island_backend", "torch")).image(args, spec)

    def build_learner_task(self, args, spec, learner_id: int, num_learners: int, syncer_addr: str):
        import sky

        from ..datasource import learner_data_arg, learner_file_mounts
        from ..models import resolve

        engine = get_engine(getattr(args, "island_backend", "torch"))

        # Flags shared by every island engine. The DiLoCo sync, LoRA, data, and
        # loss are identical; only the intra-island parallelism differs.
        learner_flags = (
            f" --model {shlex.quote(args.model)}"
            f" --data {shlex.quote(learner_data_arg(args.data))}"
            f" --syncer $SYNCER_ADDR"
            f" --learner-id $LEARNER_ID"
            f" --num-learners {num_learners}"
            f" --loss-function {args.loss_function}"
            f" --train-on {args.train_on}"
            f" --tuning {args.tuning}"
            f" --lora-r {args.lora_r}"
            f" --lora-targets {getattr(args, 'lora_targets', 'auto')}"
            f" --seq-len {args.seq_len}"
            f" --micro-batch-size {args.micro_batch_size}"
            f" --grad-accum {args.grad_accum}"
            f" --inner-lr {args.inner_lr}"
            f" --fragments {args.fragments}"
            f" --fragment-pattern {args.fragment_pattern}"
            f" --merge-alpha {args.merge_alpha}"
            f" --tokenize {args.tokenize}"
            f" --stream-workers {args.stream_workers}"
            f" --wire-dtype {args.wire_dtype}"
            f" --wan-streams {args.wan_streams}"
            f" --output-dir ~/{self.output_dir}"
        )
        if args.max_rows:
            learner_flags += f" --max-rows {args.max_rows}"
        learner_flags += engine.extra_learner_flags(args, spec)
        setup_steps = engine.setup_steps(args)
        entrypoint = engine.entrypoint()

        run = (
            f"{launcher.NVME_ENV}\n"
            f"{launcher.HF_TOKEN_ENV}\n"
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
        if spec.num_nodes > 1:
            # Surface NCCL's chosen transport in the job logs so an EFA-less
            # fallback to TCP sockets is visible, not silent.
            envs["NCCL_DEBUG"] = "INFO"
        # Non-HF --data sources (local paths, s3://, gs://, ...) ride sky's
        # file_mounts onto every learner; see yeto/datasource.py.
        file_mounts = dict(learner_file_mounts(args.data)) or None
        if args.loss_function.startswith("pickle:"):
            # The pickled loss is gitignored, so the workdir sync skips it;
            # mount it into the workdir explicitly.
            file_mounts = file_mounts or {}
            file_mounts[f"~/sky_workdir/{launcher.PICKLED_LOSS_FILE}"] = str(
                launcher.REPO_ROOT / launcher.PICKLED_LOSS_FILE
            )
        # Ride the launching machine's HF token onto every learner: anonymous
        # Hub quota is half the authenticated one and shared per-IP, and a
        # gated/private --model needs the token outright. HF_TOKEN_ENV then
        # copies it wherever NVME_ENV points HF_HOME.
        local_token = os.path.expanduser(launcher.HF_TOKEN_PATH)
        if os.path.isfile(local_token):
            file_mounts = file_mounts or {}
            file_mounts[launcher.HF_TOKEN_PATH] = local_token
        # Kick the weight download off in the background at the END of setup:
        # it overlaps sky's remaining bookkeeping and races ahead of the run
        # command, which then finds a warm (or warming — hf resumes) cache.
        repo = resolve(args.model)
        prefetch = (
            f"(nohup huggingface-cli download {shlex.quote(repo)} "
            ">/tmp/hf-prefetch.log 2>&1 &) || true"
        )
        task = sky.Task(
            name=f"yeto-learner-{learner_id}",
            setup="\n".join(setup_steps + [prefetch]),
            run=run,
            envs=envs,
            num_nodes=spec.num_nodes,
            workdir=str(launcher.REPO_ROOT),
            file_mounts=file_mounts,
        )
        infra = f"{spec.cloud}/{spec.region}" if spec.region else spec.cloud
        resources_kwargs = {}
        image = launcher.learner_image_for(args, spec)
        if image is not None:
            resources_kwargs["image_id"] = image
        if spec.num_nodes > 1:
            # Multi-node learner: inner DDP all-reduce crosses the node fabric,
            # so request the cloud's RDMA-class interconnect (EFA on AWS,
            # GPUDirect on GCP).
            resources_kwargs["network_tier"] = "best"
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

    def warn_if_wont_fit(self, args, specs) -> None:
        weight_gb = launcher.MODEL_WEIGHT_GB.get(args.model)
        if weight_gb is None:
            return
        for spec in specs:
            vram = launcher.GPU_MEM_GB.get(spec.gpu, 0) * spec.total_gpus
            if vram < weight_gb:
                print(
                    f"[launcher] WARNING: {spec} has ~{vram} GB VRAM but {args.model} "
                    f"needs ~{weight_gb} GB for frozen bf16 weights alone — expect OOM.",
                    file=sys.stderr,
                )

    def export(self, args) -> None:
        import argparse as _argparse
        from pathlib import Path

        import torch

        from ..export import parse_checkpoint, validate_against_layout
        from ..fragments import build_layout
        from ..learner import load_model_and_tokenizer, trainable_params
        from ..tensor_io import apply_fragment

        if not args.model:
            raise SystemExit("--model is required for --task lm")

        ckpt = parse_checkpoint(args.checkpoint)
        device = torch.device(args.device)
        # Mirror the learner's model construction exactly so the trainable
        # tensor set (and therefore the fragment layout) is identical.
        # shard="ddp" keeps the single-process (non-FSDP) dtype path.
        learner_args = _argparse.Namespace(
            model=args.model,
            tuning=args.tuning,
            shard="ddp",
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_targets=args.lora_targets,
        )
        model, tokenizer = load_model_and_tokenizer(learner_args, device)
        params = trainable_params(model)
        layout = build_layout(
            [(n, p.numel()) for n, p in params.items()], args.fragments, args.fragment_pattern
        )
        validate_against_layout(ckpt, layout)

        for frag, (_, flat_params, _) in zip(layout.fragments, ckpt.fragments):
            apply_fragment(frag, flat_params, params)

        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out)
        tokenizer.save_pretrained(out)

        print(f"exported checkpoint at global_step={ckpt.global_step} to {out}")
        print(
            f"applied {layout.num_fragments} fragments "
            f"({sum(p.numel() for _, p, _ in ckpt.fragments)} params) "
            f"onto {len(params)} trainable tensors ({args.tuning})"
        )
        if ckpt.ledger:
            total_merges = sum(m for m, _, _ in ckpt.ledger.values())
            total_steps = sum(s for _, s, _ in ckpt.ledger.values())
            total_tokens = sum(t for _, _, t in ckpt.ledger.values())
            print(
                f"ledger: {len(ckpt.ledger)} learners, {total_merges} merges, "
                f"{total_steps} steps, {total_tokens} tokens"
            )
            for lid in sorted(ckpt.ledger):
                merges, steps, tokens = ckpt.ledger[lid]
                print(f"  learner {lid}: merges={merges} steps={steps} tokens={tokens}")
        else:
            print("ledger: empty")


register_backend(LMBackend(), default=True)
