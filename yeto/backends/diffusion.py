"""Generic diffusion-family task backend.

This backend is the diffusion analogue of the LM backend: it owns SkyPilot
provisioning and Yeto's async fragment sync, while concrete frameworks are
plugged in as components (for example ``--component nava``). NAVA is therefore
not a top-level task special case; it is one adapter behind the diffusion task.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

from .. import launcher
from ..components import component_names, default_component, get_component
from ..models import get_model_spec
from .base import TaskBackend, register_backend


REMOTE_PREFIXES = ("http://", "https://")


def _quote_flag(name: str, value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return f" {name}" if value else ""
    return f" {name} {shlex.quote(str(value))}"


def _is_local_path(value: str | None) -> bool:
    if not value or value.startswith(REMOTE_PREFIXES):
        return False
    if value.startswith(("/", "./", "../", "~")):
        return True
    return os.path.exists(os.path.expanduser(value))


def _is_mountable_checkpoint(value: str | None) -> bool:
    if not value or value.startswith(REMOTE_PREFIXES):
        return False
    try:
        from ..datasource import kind

        return kind(value) in ("local", "cloud")
    except Exception:
        return _is_local_path(value)


def _checkpoint_mount_source(value: str) -> str:
    try:
        from ..datasource import kind

        return os.path.abspath(os.path.expanduser(value)) if kind(value) == "local" else value
    except Exception:
        return os.path.abspath(os.path.expanduser(value)) if _is_local_path(value) else value


def _mounted_checkpoint_arg(args) -> str:
    if _is_mountable_checkpoint(args.base_checkpoint):
        return "~/yeto-base-checkpoint"
    return args.base_checkpoint


class DiffusionBackend(TaskBackend):
    name = "diffusion"
    supports_auto_fleet = False
    output_dir = "yeto-diffusion-output"

    def add_launch_cli_args(self, parser: argparse.ArgumentParser) -> None:
        comp_names = component_names()
        default = default_component() or (comp_names[0] if comp_names else None)
        d = parser.add_argument_group("diffusion fine-tuning")
        d.add_argument("--component", choices=comp_names, default=default)
        d.add_argument(
            "--component-root",
            default=None,
            help="local checkout/root for the selected diffusion component; required when the component is not installed on the image",
        )
        d.add_argument(
            "--component-config",
            default=None,
            help="component training config path (relative to --component-root unless absolute)",
        )
        d.add_argument("--base-checkpoint", default=None, help="base diffusion checkpoint to load before training")
        d.add_argument(
            "--data-format",
            choices=["jsonl", "list"],
            default="jsonl",
            help="generic manifest format passed to the component",
        )
        d.add_argument(
            "--modality",
            choices=["text_to_av", "text_to_audio", "text_to_video", "text_to_image"],
            default="text_to_av",
            help="diffusion conditioning/output modality when supported by the component",
        )
        d.add_argument("--adapter", choices=["lora", "full", "regex"], default="lora")
        d.add_argument("--trainable-regex", default=None)
        d.add_argument("--lora-r", type=int, default=16)
        d.add_argument("--lora-alpha", type=int, default=32)
        d.add_argument("--lora-dropout", type=float, default=0.0)
        d.add_argument(
            "--lora-targets",
            default=None,
            help="component-defined LoRA target preset/regex; default comes from the component",
        )
        d.add_argument(
            "--merge-avg-regex",
            default=r"(^|\.)(bias|norm|modulation|scale|shift)(\.|$)",
            help="trainable tensor names merged by direct averaging instead of RDA",
        )
        d.add_argument("--init-timeout", type=float, default=1800.0)
        d.add_argument("--batch-size", type=int, default=None)
        d.add_argument("--grad-accum", type=int, default=None)
        d.add_argument("--lr", type=float, default=None)
        d.add_argument("--weight-decay", type=float, default=None)
        d.add_argument("--warmup-steps", type=int, default=None)
        d.add_argument("--max-local-steps", type=int, default=None)
        d.add_argument("--num-workers", type=int, default=None)
        d.add_argument("--io-workers", type=int, default=None)
        d.add_argument("--disable-ema", action="store_true")
        d.add_argument("--save-every", type=int, default=100)
        d.add_argument("--learner-state-dir", default=None)
        if default:
            get_component(default).add_launch_cli_args(parser)

    def add_export_cli_args(self, parser: argparse.ArgumentParser) -> None:
        comp_names = component_names()
        default = default_component() or (comp_names[0] if comp_names else None)
        parser.add_argument("--component", choices=comp_names, default=default)
        parser.add_argument("--component-root", default=None)
        parser.add_argument("--component-config", default=None)
        parser.add_argument("--base-checkpoint", default=None)
        parser.add_argument("--adapter", choices=["lora", "full", "regex"], default="lora")
        parser.add_argument("--format", choices=["lora", "merged", "trainable", "both"], default="lora")
        parser.add_argument("--trainable-regex", default=None)
        parser.add_argument("--lora-r", type=int, default=16)
        parser.add_argument("--lora-alpha", type=int, default=32)
        parser.add_argument("--lora-dropout", type=float, default=0.0)
        parser.add_argument("--lora-targets", default=None)
        parser.add_argument("--merge-avg-regex", default=r"(^|\.)(bias|norm|modulation|scale|shift)(\.|$)")
        parser.add_argument("--allow-base-mismatch", action="store_true")
        if default:
            get_component(default).add_export_cli_args(parser)

    def _component(self, args):
        return get_component(getattr(args, "component", None))

    @staticmethod
    def _env_default(value: str | None, env_name: str | None) -> str | None:
        return value or (os.environ.get(env_name) if env_name else None)

    def normalize_args(self, args) -> None:
        spec = get_model_spec(getattr(args, "model", None))
        if spec is None or spec.task != self.name:
            return
        args.component = args.component or spec.component
        args.component_config = args.component_config or spec.component_config
        if not args.component_root:
            args.component_root = self._env_default(
                spec.component_root, spec.component_root_env
            )
        if not args.base_checkpoint:
            args.base_checkpoint = self._env_default(
                spec.base_checkpoint, spec.base_checkpoint_env
            )
        args.data_format = args.data_format or spec.data_format or "jsonl"
        args.adapter = args.adapter or spec.adapter or "lora"
        args.lora_targets = args.lora_targets or spec.lora_targets
        if not getattr(args, "learner_image", None):
            args.learner_image = self._env_default(spec.learner_image, spec.learner_image_env)

    def validate(self, args) -> list[str]:
        errors: list[str] = []
        missing = [
            name
            for name in ("component", "base_checkpoint", "data")
            if not getattr(args, name, None)
        ]
        if missing:
            flags = ", ".join("--" + name.replace("_", "-") for name in missing)
            errors.append(f"--task diffusion requires: {flags}")
            return errors
        if getattr(args, "external_learners", 0):
            errors.append("--external-learners currently supports LM/MLX learners only")
        comp = self._component(args)
        errors.extend(comp.validate(args))
        if args.shard == "fsdp" and args.adapter != "lora":
            errors.append("diffusion FSDP sync currently supports --adapter lora; use --shard ddp or LoRA")
        return errors

    def warnings(self, args) -> list[str]:
        warns = self._component(args).warnings(args)
        if not getattr(args, "learner_image", None):
            warns.append(
                "WARNING: diffusion component runtimes can be heavy; production runs should use --learner-image with required CUDA/runtime deps preinstalled."
            )
        if not getattr(args, "component_root", None) and not getattr(args, "learner_image", None):
            warns.append(
                "WARNING: no --component-root/--learner-image was provided; the learner image must already provide the selected diffusion component."
            )
        return warns

    def head_file_mounts(self, args) -> dict[str, str]:
        return self._component(args).head_file_mounts(args)

    def rewrite_for_head(self, args) -> None:
        self._component(args).rewrite_for_head(args)

    def build_learner_task(self, args, spec, learner_id: int, num_learners: int, syncer_addr: str):
        import sky

        from ..datasource import learner_data_arg, learner_file_mounts

        comp = self._component(args)
        data_arg = learner_data_arg(args.data)
        ckpt_arg = _mounted_checkpoint_arg(args)
        config_arg = comp.resolve_component_config(args)
        targets = args.lora_targets or comp.default_lora_targets
        root_flag = " --component-root ~/yeto-component" if args.component_root else ""

        flags = (
            f" --component {args.component}"
            f" --syncer $SYNCER_ADDR"
            f" --learner-id $LEARNER_ID"
            f" --num-learners {num_learners}"
            f"{root_flag}"
            f" --component-config {shlex.quote(config_arg)}"
            f" --base-checkpoint {shlex.quote(ckpt_arg)}"
            f" --data {shlex.quote(data_arg)}"
            f" --data-format {args.data_format}"
            f" --modality {args.modality}"
            f" --adapter {args.adapter}"
            f" --lora-r {args.lora_r}"
            f" --lora-alpha {args.lora_alpha}"
            f" --lora-dropout {args.lora_dropout}"
            f" --lora-targets {shlex.quote(targets)}"
            f" --merge-avg-regex {shlex.quote(args.merge_avg_regex)}"
            f" --init-timeout {args.init_timeout}"
            f" --shard {args.shard}"
            f" --fragments {args.fragments}"
            f" --fragment-pattern {args.fragment_pattern}"
            f" --merge-alpha {args.merge_alpha}"
            f" --wire-dtype {args.wire_dtype}"
            f" --wan-streams {args.wan_streams}"
            f" --output-dir ~/{self.output_dir}"
        )
        for name, value in (
            ("--trainable-regex", args.trainable_regex),
            ("--batch-size", args.batch_size),
            ("--grad-accum", args.grad_accum),
            ("--lr", args.lr),
            ("--weight-decay", args.weight_decay),
            ("--warmup-steps", args.warmup_steps),
            ("--max-local-steps", args.max_local_steps),
            ("--num-workers", args.num_workers),
            ("--io-workers", args.io_workers),
            ("--save-every", args.save_every),
            ("--learner-state-dir", args.learner_state_dir),
        ):
            flags += _quote_flag(name, value)
        flags += _quote_flag("--disable-ema", args.disable_ema)

        run = (
            'MASTER_ADDR=$(echo "$SKYPILOT_NODE_IPS" | head -n1)\n'
            "torchrun --nnodes=$SKYPILOT_NUM_NODES --node_rank=$SKYPILOT_NODE_RANK "
            "--nproc_per_node=$SKYPILOT_NUM_GPUS_PER_NODE "
            "--master_addr=$MASTER_ADDR --master_port=29500 "
            f"-m yeto.diffusion.learner{flags}"
        )
        setup_steps = [
            launcher.WAN_TUNING,
            launcher.NVME_SETUP,
            launcher.NVME_ENV,
            launcher.HF_TOKEN_ENV,
            "pip install -q -r requirements.txt",
            "pip install -q PyYAML safetensors",
            *comp.setup_steps(args),
            *(["pip install -q -e ~/yeto-component"] if args.component_root else []),
        ]
        envs = {
            "SYNCER_ADDR": syncer_addr,
            "LEARNER_ID": str(learner_id),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            **comp.learner_env(args),
        }
        if os.environ.get("HF_TOKEN"):
            envs["HF_TOKEN"] = os.environ["HF_TOKEN"]
        if spec.num_nodes > 1:
            envs["NCCL_DEBUG"] = "INFO"

        file_mounts = dict(learner_file_mounts(args.data))
        file_mounts.update(comp.build_learner_file_mounts(args))
        if _is_mountable_checkpoint(args.base_checkpoint):
            file_mounts["~/yeto-base-checkpoint"] = _checkpoint_mount_source(args.base_checkpoint)
        task = sky.Task(
            name=f"yeto-diffusion-learner-{learner_id}",
            setup="\n".join(setup_steps),
            run=run,
            envs=envs,
            num_nodes=spec.num_nodes,
            workdir=str(launcher.REPO_ROOT),
            file_mounts=file_mounts or None,
        )
        infra = f"{spec.cloud}/{spec.region}" if spec.region else spec.cloud
        resources_kwargs = {}
        image = launcher.learner_image_for(args, spec)
        if image is not None:
            resources_kwargs["image_id"] = image
        if spec.num_nodes > 1:
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
        self._component(args).warn_if_wont_fit(args, specs)

    def export(self, args) -> None:
        if not args.base_checkpoint:
            raise SystemExit("--task diffusion requires --base-checkpoint")
        from ..export import parse_checkpoint

        return self._component(args).export(args, parse_checkpoint(args.checkpoint))


register_backend(DiffusionBackend())
