"""Precompute optional diffusion latents/text embeddings.

Writes a local JSONL manifest whose tensor columns point at ``.pt`` files.
Training can consume the manifest with ``--cache-latents`` and/or
``--cache-text-embeds``; these cache flags are opt-in, not the learner default.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..data import load_rows
from . import learner


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Precompute diffusion latent/text caches")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True, help="local output directory for manifest + tensors")
    p.add_argument("--diffusion-family", choices=["auto", "generic", "ltx", "wan", "nava"], default="auto")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--cache-latents", action="store_true", default=False)
    p.add_argument("--cache-text-embeds", action="store_true", default=False)
    p.add_argument("--image-column", default="image")
    p.add_argument("--video-column", default="video")
    p.add_argument("--prompt-column", default="prompt")
    p.add_argument("--latent-column", default="latents")
    p.add_argument("--text-embeds-column", default="prompt_embeds")
    p.add_argument("--text-attention-mask-column", default="prompt_attention_mask")
    p.add_argument("--pooled-text-embeds-column", default="pooled_prompt_embeds")
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--frame-rate", type=int, default=25)
    p.add_argument("--nava-root", default=None)
    p.add_argument("--nava-config", default="configs/nava.yaml")
    p.add_argument("--nava-checkpoint", default=None)
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def _collate(rows):
    return [dict(r) for r in rows]


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.cache_latents and not args.cache_text_embeds:
        raise SystemExit("precompute needs --cache-latents and/or --cache-text-embeds")
    out = Path(os.path.expanduser(args.output))
    out.mkdir(parents=True, exist_ok=True)
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Reuse the learner's diffusers loading path without attaching adapters.
    args.tuning = "full"
    args.lora_r = 16
    args.lora_alpha = 32
    args.lora_targets = "auto"
    pipe = learner.load_pipeline(args, device)
    ds = load_rows(args.data)
    data_root = None
    data_path = Path(os.path.expanduser(args.data))
    if data_path.exists():
        data_root = data_path if data_path.is_dir() else data_path.parent
    if args.max_rows is not None:
        ds = ds.select(range(min(args.max_rows, len(ds)))) if hasattr(ds, "select") else ds[: args.max_rows]
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=_collate)

    lat_dir = out / "latents"
    txt_dir = out / "text_embeds"
    if args.cache_latents:
        lat_dir.mkdir(exist_ok=True)
    if args.cache_text_embeds:
        txt_dir.mkdir(exist_ok=True)

    manifest = out / "data.jsonl"
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    index = 0
    with manifest.open("w", encoding="utf-8") as f:
        for rows in loader:
            if data_root is not None:
                for row in rows:
                    row["__yeto_data_root__"] = str(data_root)
            latents = latent_batch = None
            prompt_embeds = pooled = mask = None
            if args.cache_latents:
                old = args.cache_latents
                args.cache_latents = False
                latent_batch = learner.encode_latents(pipe, rows, args, device, dtype)
                latents = latent_batch.latents.detach().cpu()
                args.cache_latents = old
            if args.cache_text_embeds:
                old = args.cache_text_embeds
                args.cache_text_embeds = False
                cond = learner.encode_prompt_embeds(pipe, rows, args, device, dtype)
                prompt_embeds = cond.prompt_embeds.detach().cpu() if cond.prompt_embeds is not None else None
                pooled = cond.pooled_prompt_embeds.detach().cpu() if cond.pooled_prompt_embeds is not None else None
                mask = cond.attention_mask.detach().cpu() if cond.attention_mask is not None else None
                args.cache_text_embeds = old
            for b, row in enumerate(rows):
                item = {args.prompt_column: row.get(args.prompt_column, "")}
                if latents is not None:
                    path = lat_dir / f"{index:08d}.pt"
                    torch.save(latents[b], path)
                    item[args.latent_column] = str(path.relative_to(out))
                    if latents[b].ndim >= 4:
                        item["latent_height"] = int(latents[b].shape[-2])
                        item["latent_width"] = int(latents[b].shape[-1])
                    if latents[b].ndim == 4:
                        item["latent_num_frames"] = int(latents[b].shape[-3])
                    elif latent_batch is not None:
                        item["latent_num_frames"] = latent_batch.latent_num_frames
                        item["latent_height"] = latent_batch.latent_height
                        item["latent_width"] = latent_batch.latent_width
                if prompt_embeds is not None:
                    path = txt_dir / f"{index:08d}_prompt.pt"
                    torch.save(prompt_embeds[b], path)
                    item[args.text_embeds_column] = str(path.relative_to(out))
                if mask is not None:
                    path = txt_dir / f"{index:08d}_mask.pt"
                    torch.save(mask[b], path)
                    item[args.text_attention_mask_column] = str(path.relative_to(out))
                if pooled is not None:
                    path = txt_dir / f"{index:08d}_pooled.pt"
                    torch.save(pooled[b], path)
                    item[args.pooled_text_embeds_column] = str(path.relative_to(out))
                f.write(json.dumps(item) + "\n")
                index += 1
    print(f"[yeto] wrote {index} cached diffusion rows to {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
