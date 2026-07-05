"""The ``nava`` task backend: fine-tune NAVA multimodal diffusion models.

NAVA keeps its own pipeline/dataset/loss code as the source of truth; this
backend wraps yeto's asynchronous fragment sync around NAVA's optimizer step,
provisions the NAVA runtime on each learner, and mounts the local checkout.
Auto-fleet planning is LM-only, so NAVA runs require an explicit ``--gpu``.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

from .. import launcher
from ..backends.base import TaskBackend, register_backend


def _int_or_auto(value: str):
    # duplicated from yeto/autobatch.py: importing it would pull torch into the
    # CLI path (see yeto/backends/lm.py for the same guard).
    return value if value == "auto" else int(value)


class NavaBackend(TaskBackend):
    name = "nava"
    supports_auto_fleet = False
    output_dir = "yeto-nava-output"

    def add_launch_cli_args(self, parser: argparse.ArgumentParser) -> None:
        nava = parser.add_argument_group("NAVA fine-tuning")
        nava.add_argument("--nava-root", default=None, help="local NAVA checkout to mount/sync")
        nava.add_argument("--nava-config", default="configs/nava.yaml")
        nava.add_argument("--nava-ckpt", default=None, help="NAVA base checkpoint, local or s3://")
        nava.add_argument("--nava-assets-uri", default=None, help="optional s3:// asset root copied to learners")
        nava.add_argument("--nava-data", default=None, help="NAVA JSONL/list or Gemini label s3:// input")
        nava.add_argument(
            "--nava-data-format",
            choices=["nava-jsonl", "nava-list", "s3-label-array", "s3-label-prefix"],
            default="nava-jsonl",
        )
        nava.add_argument("--nava-data-weights", default=None)
        nava.add_argument("--nava-data-cache", default="/local_nvme/yeto-nava-cache")
        nava.add_argument(
            "--nava-predownload-uris",
            default=None,
            help="s3:// or local newline list of clip URIs to pre-fetch into "
            "--nava-data-cache during learner setup (before torchrun), so "
            "training reads clips from local disk instead of streaming",
        )
        nava.add_argument(
            "--nava-modality",
            choices=["text_to_av", "text_to_audio", "text_to_video", "text_to_image"],
            default="text_to_av",
        )
        nava.add_argument(
            "--nava-caption-field",
            choices=["composed", "dense_lora_caption", "short_lora_caption", "lora_tags"],
            default="composed",
        )
        nava.add_argument("--nava-min-duration", type=float, default=None)
        nava.add_argument("--nava-max-duration", type=float, default=None)
        nava.add_argument("--nava-probe-labels", action="store_true")
        nava.add_argument("--nava-tuning", choices=["lora", "full", "attention", "regex"], default="lora")
        nava.add_argument("--nava-trainable-regex", default=None)
        nava.add_argument("--nava-lora-r", type=int, default=16)
        nava.add_argument("--nava-lora-alpha", type=int, default=32)
        nava.add_argument("--nava-lora-dropout", type=float, default=0.0)
        nava.add_argument("--nava-lora-targets", default="mmdit-all-linear")
        nava.add_argument("--nava-full-sync", choices=["unsupported", "gather"], default="unsupported")
        nava.add_argument("--nava-merge-avg-regex", default=r"(^|\.)(bias|norm|modulation|scale|shift)(\.|$)")
        nava.add_argument("--nava-init-timeout", type=float, default=1800.0)
        nava.add_argument(
            "--nava-install-flash-attn",
            action="store_true",
            help="build/install flash-attn during learner setup; prefer --learner-image in production",
        )
        nava.add_argument("--nava-batch-size", type=_int_or_auto, default=None,
                          help="per-GPU micro-batch, or 'auto' to probe the largest that fits")
        nava.add_argument("--nava-grad-accum", type=int, default=None)
        nava.add_argument("--nava-lr", type=float, default=None)
        nava.add_argument("--nava-weight-decay", type=float, default=None)
        nava.add_argument("--nava-warmup-steps", type=int, default=None)
        nava.add_argument("--nava-max-local-steps", type=int, default=None)
        nava.add_argument("--nava-num-workers", type=int, default=None)
        nava.add_argument("--nava-io-workers", type=int, default=None)
        nava.add_argument("--nava-disable-ema", action="store_true")
        nava.add_argument("--nava-save-every", type=int, default=100)
        nava.add_argument("--nava-learner-state-dir", default=None)

    def add_export_cli_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--nava-root", default=None)
        parser.add_argument("--nava-config", default="configs/nava.yaml")
        parser.add_argument("--base-ckpt", default=None, help="NAVA base checkpoint for --task nava")
        parser.add_argument("--format", choices=["lora", "merged", "trainable", "both"], default="lora")
        parser.add_argument("--nava-tuning", choices=["lora", "full", "attention", "regex"], default="lora")
        parser.add_argument("--nava-lora-targets", default="mmdit-all-linear")
        parser.add_argument("--nava-trainable-regex", default=None)
        parser.add_argument("--nava-merge-avg-regex", default=r"(^|\.)(bias|norm|modulation|scale|shift)(\.|$)")
        parser.add_argument("--allow-base-mismatch", action="store_true")

    def validate(self, args) -> list[str]:
        errors: list[str] = []
        missing = [
            name for name in ("nava_root", "nava_ckpt", "nava_data") if not getattr(args, name, None)
        ]
        if missing:
            flags = ", ".join("--" + name.replace("_", "-") for name in missing)
            errors.append(f"--task nava requires: {flags}")
            return errors
        if args.shard == "fsdp" and args.nava_tuning != "lora" and args.nava_full_sync != "gather":
            errors.append(
                "NAVA FSDP with non-LoRA trainables requires --nava-full-sync gather; "
                "use --nava-tuning lora for the low-cost default path."
            )
        return errors

    def warnings(self, args) -> list[str]:
        warns: list[str] = []
        # Only meaningful once the required flags are present; a missing-flag
        # run is rejected by validate() before it gets here.
        if not getattr(args, "nava_root", None):
            return warns
        if args.nava_tuning == "full" and args.syncer_memory < 96:
            warns.append(
                "WARNING: NAVA full fine-tune syncer memory is likely too low; "
                "use --syncer-memory 128 or LoRA unless this is a tiny model."
            )
        if not getattr(args, "learner_image", getattr(args, "runtime_image", None)):
            warns.append(
                "WARNING: NAVA runtime setup is heavy; production runs should use "
                "--learner-image with CUDA torch and flash-attn preinstalled."
            )
        return warns

    def head_file_mounts(self, args) -> dict:
        if getattr(args, "nava_root", None):
            return {"~/NAVA": os.path.abspath(os.path.expanduser(args.nava_root))}
        return {}

    def rewrite_for_head(self, args) -> None:
        if getattr(args, "nava_root", None):
            # The submitting CLI mounted the local checkout at ~/NAVA on the
            # head; learner tasks must be built from the head's path.
            args.nava_root = os.path.expanduser("~/NAVA")

    def build_learner_task(self, args, spec, learner_id: int, num_learners: int, syncer_addr: str):
        import sky

        nava_root = os.path.abspath(os.path.expanduser(args.nava_root))
        nava_mount = "~/NAVA"
        nava_config = launcher._remote_nava_path(args.nava_config, nava_mount)
        nava_ckpt = args.nava_ckpt
        nava_data = args.nava_data
        file_mounts = {nava_mount: nava_root}
        if launcher._is_local_existing_path(args.nava_ckpt):
            nava_ckpt = f"~/nava_inputs/{Path(args.nava_ckpt).name}"
            file_mounts[nava_ckpt] = os.path.abspath(os.path.expanduser(args.nava_ckpt))
        if launcher._is_local_existing_path(args.nava_data):
            nava_data = f"~/nava_inputs/{Path(args.nava_data).name}"
            file_mounts[nava_data] = os.path.abspath(os.path.expanduser(args.nava_data))
        assets_cmd = launcher._s3_sync_command(args.nava_assets_uri, "/local_nvme/nava_assets")
        setup_parts = [
            launcher.WAN_TUNING,
            "sudo apt-get update -y >/dev/null 2>&1 && sudo apt-get install -y ffmpeg libsndfile1 >/dev/null 2>&1 || true",
            "pip install -q -e '.[nava]'",
            f"pip install -q -e {nava_mount}",
            "mkdir -p /local_nvme/yeto-nava-cache /local_nvme/nava_assets",
        ]
        if args.nava_install_flash_attn:
            setup_parts.append("pip install -q flash-attn --no-build-isolation")
        if assets_cmd:
            setup_parts.append(assets_cmd)
        if args.nava_predownload_uris:
            # Pull the selected clips into the resolver cache before torchrun, so
            # the first training step reads from local disk (no S3 streaming).
            setup_parts.append(
                "python -m yeto.nava.pre_download_clips"
                f" --uri-list {shlex.quote(args.nava_predownload_uris)}"
                f" --cache-dir {shlex.quote(args.nava_data_cache)}"
                " --workers 48"
            )
        setup = "; ".join(part for part in setup_parts if part)

        flags = (
            f" --syncer $SYNCER_ADDR"
            f" --learner-id $LEARNER_ID"
            f" --num-learners {num_learners}"
            f" --nava-root {nava_mount}"
            f" --nava-config {shlex.quote(nava_config)}"
            f" --nava-ckpt {shlex.quote(nava_ckpt)}"
            f" --nava-data {shlex.quote(nava_data)}"
            f" --nava-data-format {args.nava_data_format}"
            f" --nava-data-cache {shlex.quote(args.nava_data_cache)}"
            f" --nava-assets-dir /local_nvme/nava_assets"
            f" --nava-modality {args.nava_modality}"
            f" --nava-caption-field {args.nava_caption_field}"
            f" --nava-tuning {args.nava_tuning}"
            f" --nava-lora-r {args.nava_lora_r}"
            f" --nava-lora-alpha {args.nava_lora_alpha}"
            f" --nava-lora-dropout {args.nava_lora_dropout}"
            f" --nava-lora-targets {shlex.quote(args.nava_lora_targets)}"
            f" --nava-full-sync {args.nava_full_sync}"
            f" --nava-merge-avg-regex {shlex.quote(args.nava_merge_avg_regex)}"
            f" --nava-init-timeout {args.nava_init_timeout}"
            f" --shard {args.shard}"
            f" --total-steps {args.total_steps}"
            f" --fragments {args.fragments}"
            f" --wire-dtype {args.wire_dtype}"
            f" --wan-streams {args.wan_streams}"
            f" --output-dir ~/{self.output_dir}"
        )
        opt = launcher._quote_optional_flag
        flags += opt("--nava-data-weights", args.nava_data_weights)
        flags += opt("--nava-min-duration", args.nava_min_duration)
        flags += opt("--nava-max-duration", args.nava_max_duration)
        flags += opt("--nava-trainable-regex", args.nava_trainable_regex)
        flags += opt("--nava-batch-size", args.nava_batch_size)
        flags += opt("--nava-grad-accum", args.nava_grad_accum)
        flags += opt("--nava-lr", args.nava_lr)
        flags += opt("--nava-weight-decay", args.nava_weight_decay)
        flags += opt("--nava-warmup-steps", args.nava_warmup_steps)
        flags += opt("--nava-max-local-steps", args.nava_max_local_steps)
        flags += opt("--nava-num-workers", args.nava_num_workers)
        flags += opt("--nava-io-workers", args.nava_io_workers)
        flags += opt("--nava-save-every", args.nava_save_every)
        flags += opt("--nava-learner-state-dir", args.nava_learner_state_dir)
        flags += opt("--nava-probe-labels", args.nava_probe_labels)
        flags += opt("--nava-disable-ema", args.nava_disable_ema)

        run = (
            'MASTER_ADDR=$(echo "$SKYPILOT_NODE_IPS" | head -n1)\n'
            "torchrun --nnodes=$SKYPILOT_NUM_NODES --node_rank=$SKYPILOT_NODE_RANK "
            "--nproc_per_node=$SKYPILOT_NUM_GPUS_PER_NODE "
            "--master_addr=$MASTER_ADDR --master_port=29500 "
            f"-m yeto.nava.learner{flags}"
        )
        envs = {
            "SYNCER_ADDR": syncer_addr,
            "LEARNER_ID": str(learner_id),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "NAVA_S3_REGION": os.environ.get("NAVA_S3_REGION", "us-west-2"),
            "YETO_NAVA_ASSET_CACHE": "/local_nvme/nava_assets",
            "YETO_NAVA_DATA_CACHE": args.nava_data_cache,
        }
        if os.environ.get("HF_TOKEN"):
            envs["HF_TOKEN"] = os.environ["HF_TOKEN"]
        # Forward auto-batch probe tuning (VRAM headroom + batch ceiling) so the
        # submitter's values reach the learner; the probe falls back to safe
        # defaults (0.80 / 8) when unset.
        for var in ("YETO_NAVA_PROBE_MEM_FRACTION", "YETO_NAVA_MAX_MICRO_BATCH"):
            if os.environ.get(var):
                envs[var] = os.environ[var]
        if spec.num_nodes > 1:
            envs["NCCL_DEBUG"] = "INFO"

        task = sky.Task(
            name=f"yeto-nava-learner-{learner_id}",
            setup=setup,
            run=run,
            envs=envs,
            num_nodes=spec.num_nodes,
            workdir=str(launcher.REPO_ROOT),
            file_mounts=file_mounts,
        )
        infra = f"{spec.cloud}/{spec.region}" if spec.region else spec.cloud
        resources_kwargs = {}
        if spec.num_nodes > 1:
            resources_kwargs["network_tier"] = "best"
        image = launcher.learner_image_for(args, spec)
        if image is not None:
            resources_kwargs["image_id"] = image
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
        if args.nava_tuning == "lora":
            min_vram = 64 if args.shard == "ddp" else 40
            mode = "LoRA"
        else:
            min_vram = 80
            mode = f"{args.nava_tuning}/{args.shard}"
        for spec in specs:
            vram_per_gpu = launcher.GPU_MEM_GB.get(spec.gpu, 0)
            if vram_per_gpu and vram_per_gpu < min_vram:
                print(
                    f"[launcher] WARNING: NAVA {mode} on {spec.gpu} (~{vram_per_gpu}GB/GPU) "
                    "is likely to OOM unless the config is tiny; prefer H100/A100-80GB "
                    "or --shard fsdp with LoRA.",
                    file=sys.stderr,
                )
        if args.nava_tuning == "full" and args.syncer_memory < 96:
            print(
                "[launcher] WARNING: NAVA full sync keeps f32 params+momentum on the syncer; "
                "use --syncer-memory 128 for production-scale NAVA.",
                file=sys.stderr,
            )

    def export(self, args) -> None:
        import argparse as _argparse
        import json

        import torch
        import yaml

        from ..export import parse_checkpoint, validate_against_layout
        from ..fragments import build_layout
        from ..layout_metadata import validate_layout_metadata
        from .learner import build_pipeline, configure_trainables, resolve_in_nava_root, trainable_params
        from .lora import LoRAConfig, merge_lora_inplace, save_lora_adapter
        from .utils import sha256_uri
        from ..tensor_io import apply_fragment

        if not args.nava_root or not args.base_ckpt:
            raise SystemExit("--task nava requires --nava-root and --base-ckpt")

        sys.path.insert(0, os.path.abspath(args.nava_root))
        ckpt = parse_checkpoint(args.checkpoint)
        device = torch.device(args.device)
        args.nava_root = os.path.abspath(os.path.expanduser(args.nava_root))
        args.nava_config = resolve_in_nava_root(args.nava_config, args.nava_root)
        cfg = yaml.safe_load(open(args.nava_config, "r", encoding="utf-8"))
        meta = ckpt.layout_meta or {}
        meta_lora = meta.get("lora") or {}
        nava_tuning = meta.get("trainable_policy") or args.nava_tuning
        ns = _argparse.Namespace(
            nava_root=args.nava_root,
            nava_config=args.nava_config,
            nava_ckpt=args.base_ckpt,
            nava_tuning=nava_tuning,
            nava_lora_r=int(meta_lora.get("r", args.lora_r)),
            nava_lora_alpha=int(meta_lora.get("alpha", args.lora_alpha)),
            nava_lora_dropout=float(meta_lora.get("dropout", 0.0)),
            nava_lora_targets=meta_lora.get("targets", args.nava_lora_targets),
            nava_trainable_regex=meta.get("trainable_regex") or args.nava_trainable_regex,
            nava_data_cache=None,
            nava_assets_dir=None,
            nava_disable_ema=True,
        )
        pipe = build_pipeline(ns, cfg, device)
        lora_cfg = configure_trainables(pipe, ns)
        params = trainable_params(pipe.model)
        layout = build_layout(
            [(n, p.numel()) for n, p in params.items()],
            args.fragments,
            avg_name_regex=meta.get("merge_avg_regex") or args.nava_merge_avg_regex,
        )
        validate_against_layout(ckpt, layout)
        base_sha = sha256_uri(args.base_ckpt, os.environ.get("YETO_NAVA_ASSET_CACHE")) if ckpt.layout_meta else None
        validate_layout_metadata(
            ckpt.layout_meta,
            layout,
            params,
            expected_task="nava",
            base_checkpoint_sha256=base_sha,
            allow_base_mismatch=args.allow_base_mismatch,
        )
        for frag, (_, flat_params, _) in zip(layout.fragments, ckpt.fragments):
            apply_fragment(frag, flat_params, params)

        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if nava_tuning == "lora" and args.format in ("lora", "merged", "both"):
            if args.format in ("lora", "both"):
                adapter_out = out if args.format == "lora" else out / "adapter"
                save_lora_adapter(pipe.model, lora_cfg or LoRAConfig(ns.nava_lora_r, ns.nava_lora_alpha), adapter_out)
            if args.format in ("merged", "both"):
                merge_lora_inplace(pipe.model)
                merged_state = {k: v.detach().cpu().contiguous() for k, v in pipe.model.state_dict().items()}
                try:
                    from safetensors.torch import save_file

                    save_file(merged_state, str(out / "NAVA_merged.safetensors"))
                except Exception:
                    torch.save({"state_dict": merged_state}, out / "NAVA_merged.ckpt")
        else:
            torch.save(
                {"state_dict": {n: p.detach().cpu() for n, p in params.items()}, "global_step": ckpt.global_step},
                out / "trainable_state.pt",
            )
        (out / "yeto_export_meta.json").write_text(
            json.dumps(
                {
                    "task": "nava",
                    "global_step": ckpt.global_step,
                    "format": args.format,
                    "tuning": nava_tuning,
                    "fragments": layout.num_fragments,
                    "trainable_tensors": len(params),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if ckpt.layout_meta:
            (out / "layout_manifest.json").write_text(json.dumps(ckpt.layout_meta, indent=2), encoding="utf-8")
        print(f"exported NAVA {args.format} at global_step={ckpt.global_step} to {out}")


register_backend(NavaBackend())
