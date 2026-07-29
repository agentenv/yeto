#!/usr/bin/env python3
"""Held-out HVP/Lanczos spectrum probe for a full-parameter syncer checkpoint.

This is a thin checkpoint/data adapter around repository components that are
already validated independently:

* ``yeto.export.parse_checkpoint`` and the production fragment layout;
* ``yeto.cttn_sidecar.make_hvp`` for panel-averaged double-backward HVPs;
* ``yeto.cttn.block_lanczos`` for the bounded-rank float64 Ritz sketch.

The probe is read-only with respect to its inputs.  It writes one JSON result
and does not optimize or fit any quantity to the observed kappa target.

This discriminator-local copy preserves the validated Lane-E default path and
adds one opt-in mode, ``--randomized-start``.  That mode replaces only the
normally deterministic transverse-buffer Lanczos start vector with a
seed-controlled Gaussian direction orthogonal to the held-out gradient.  It
provides genuinely independent Krylov starts for the lambda-max stability
check while leaving checkpoint parsing, panels, HVPs, orthogonalization, and
block Lanczos unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from yeto.cttn import block_lanczos, orth as numpy_orth  # noqa: E402
from yeto.cttn_sidecar import make_hvp  # noqa: E402
from yeto.data import build_packed_dataset  # noqa: E402
from yeto.export import parse_checkpoint, validate_against_layout  # noqa: E402
from yeto.fragments import build_layout  # noqa: E402
from yeto.losses import sft_loss  # noqa: E402
from yeto.models import resolve  # noqa: E402
from yeto.tensor_io import apply_fragment  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(model_value: str, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = resolve(model_value)
    common = {"trust_remote_code": True, "local_files_only": True}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **common)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        **common,
    ).to(device)
    model.config.use_cache = False
    model.eval()
    return model, tokenizer


def build_panels(args, tokenizer, device: torch.device):
    dataset = build_packed_dataset(
        str(args.data),
        tokenizer,
        learner_id=0,
        num_learners=1,
        seq_len=args.seq_len,
        max_rows=args.probe_max_rows,
        train_on=args.train_on,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.probe_batch_size,
        shuffle=False,
        drop_last=False,
    )
    panels = []
    for input_ids, weights in loader:
        panels.append((input_ids.to(device), weights.to(device)))
        if len(panels) >= args.probe_panels:
            break
    if len(panels) != args.probe_panels:
        raise ValueError(
            f"probe data produced {len(panels)} panels, expected {args.probe_panels}"
        )
    return tuple(panels)


def apply_checkpoint(checkpoint, layout, named_params, device: torch.device) -> None:
    validate_against_layout(checkpoint, layout)
    for fragment, (_, flat_params, _) in zip(layout.fragments, checkpoint.fragments):
        apply_fragment(fragment, flat_params.to(device), named_params)


def ordered_params(layout, named_params) -> tuple[torch.Tensor, ...]:
    names = [name for fragment in layout.fragments for name, _ in fragment.tensors]
    if len(names) != len(named_params) or set(names) != set(named_params):
        raise ValueError("layout does not cover the trainable parameters exactly once")
    return tuple(named_params[name] for name in names)


def flatten_tensors(values) -> torch.Tensor:
    return torch.cat([value.reshape(-1).float() for value in values])


def panel_gradient(model, params, panel, loss_function: str) -> tuple[torch.Tensor, float]:
    input_ids, weights = panel
    output = model(input_ids=input_ids, use_cache=False)
    loss_sum, ntok = sft_loss(output.logits, input_ids, loss_function, weights)
    loss = loss_sum / torch.clamp(ntok.float(), min=1.0)
    grads = torch.autograd.grad(loss, params, create_graph=False)
    return flatten_tensors(grads).detach(), float(loss.detach())


def finite_list(values) -> list[float]:
    values = np.asarray(values, dtype=np.float64).tolist()
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("probe produced a non-finite statistic")
    return [float(value) for value in values]


def run(args) -> dict[str, object]:
    started = time.monotonic()
    device = torch.device(args.device)
    if device.type != "cpu" and not torch.cuda.is_available():
        raise ValueError(f"requested unavailable device: {device}")
    if args.threads is not None:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(max(1, min(4, args.threads)))
    torch.manual_seed(args.seed)

    checkpoint = parse_checkpoint(args.checkpoint)
    model, tokenizer = load_model(args.model, device)
    named = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    layout = build_layout(
        [(name, parameter.numel()) for name, parameter in named.items()],
        args.fragments,
        args.fragment_pattern,
        matrix_merge="rda",
        named_shapes={name: tuple(parameter.shape) for name, parameter in named.items()},
    )
    apply_checkpoint(checkpoint, layout, named, device)
    params = ordered_params(layout, named)
    panels = build_panels(args, tokenizer, device)

    panel_grads = []
    panel_losses = []
    for panel in panels:
        gradient, loss = panel_gradient(model, params, panel, args.loss_function)
        panel_grads.append(gradient)
        panel_losses.append(loss)
    gradient = torch.stack(panel_grads, dim=1).mean(dim=1)
    gradient_norm = torch.linalg.vector_norm(gradient)
    if float(gradient_norm) == 0.0:
        raise ValueError("held-out gradient is exactly zero")

    buffer = torch.cat(
        [fragment[2].to(device=device, dtype=torch.float32) for fragment in checkpoint.fragments]
    )
    if buffer.shape != gradient.shape:
        raise ValueError("checkpoint buffer and held-out gradient dimensions differ")
    q = gradient / gradient_norm
    transverse = buffer - q * torch.dot(q, buffer)
    transverse_norm = torch.linalg.vector_norm(transverse)
    if float(transverse_norm) > 0.0 and not args.randomized_start:
        seed_vectors = [
            q.detach().cpu().numpy(),
            (transverse / transverse_norm).detach().cpu().numpy(),
        ]
    else:
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)
        random_direction = torch.randn(gradient.numel(), generator=generator, device=device)
        random_direction -= q * torch.dot(q, random_direction)
        random_direction /= torch.linalg.vector_norm(random_direction)
        seed_vectors = [q.detach().cpu().numpy(), random_direction.detach().cpu().numpy()]
    # Use the same float64 two-pass orthogonalizer as the validated CTTN
    # sidecar.  At 135M dimensions, a nominal fp32 projection leaves enough
    # accumulated roundoff to violate the strict Lanczos Gram postcondition.
    q0 = numpy_orth(seed_vectors)

    hvp, heldout_loss, release = make_hvp(
        model,
        params,
        panels,
        loss_function=args.loss_function,
    )
    try:
        def hvp_numpy(values: np.ndarray) -> np.ndarray:
            tensor = torch.from_numpy(values).to(device=device, dtype=torch.float32)
            return hvp(tensor).detach().cpu().numpy()

        # The torch-native CTTN basis intentionally uses fp32 p-dimensional
        # algebra and is validated for small LoRA parameter spaces.  At 135M
        # dimensions, fp32 reductions accumulate too much Gram error.  The
        # equally validated NumPy core keeps the tiny-rank orthogonalization
        # and Rayleigh assembly in float64 while HVP evaluation remains fp32.
        basis, rayleigh = block_lanczos(hvp_numpy, q0, args.block_steps)
    finally:
        release()

    evals, evecs = np.linalg.eigh(rayleigh)
    q_numpy = q.detach().cpu().numpy().astype(np.float64, copy=False)
    transverse_numpy = transverse.detach().cpu().numpy().astype(np.float64, copy=False)
    q_coords = evecs.T @ (basis.T @ q_numpy)
    r_coords = evecs.T @ (basis.T @ transverse_numpy)
    panel_mode_gradients = []
    for panel_gradient_value in panel_grads:
        panel_numpy = panel_gradient_value.detach().cpu().numpy().astype(np.float64, copy=False)
        coords = evecs.T @ (basis.T @ panel_numpy)
        panel_mode_gradients.append(coords)
    panel_mode_matrix = np.stack(panel_mode_gradients, axis=0)
    mode_noise_variance = panel_mode_matrix.var(axis=0, ddof=1 if args.probe_panels > 1 else 0)

    return {
        "schema": "yeto_checkpoint_spectrum_probe_v1",
        "status": "COMPLETE",
        "provenance": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "model": args.model,
            "data": str(args.data.resolve()),
            "data_sha256": sha256_file(args.data),
            "seed": args.seed,
            "device": str(device),
            "torch_version": torch.__version__,
        },
        "checkpoint": {
            "global_step": int(checkpoint.global_step),
            "fragment_versions": [int(fragment[0]) for fragment in checkpoint.fragments],
            "fragment_numel": [int(fragment[1].numel()) for fragment in checkpoint.fragments],
        },
        "probe": {
            "seq_len": args.seq_len,
            "panels": args.probe_panels,
            "batch_size": args.probe_batch_size,
            "max_rows": args.probe_max_rows,
            "train_on": args.train_on,
            "block_steps": args.block_steps,
            **(
                {"start_vector_mode": "gradient_plus_seeded_random"}
                if args.randomized_start
                else {}
            ),
            "krylov_rank": int(basis.shape[1]),
            "heldout_loss": float(heldout_loss),
            "panel_losses": panel_losses,
            "gradient_norm": float(gradient_norm),
            "buffer_norm": float(torch.linalg.vector_norm(buffer)),
            "transverse_buffer_norm": float(transverse_norm),
            "buffer_gradient_cosine": float(
                torch.dot(buffer, gradient)
                / torch.clamp(torch.linalg.vector_norm(buffer) * gradient_norm, min=1e-30)
            ),
            "ritz_values": finite_list(evals),
            "gradient_ritz_coordinates": finite_list(q_coords),
            "transverse_buffer_ritz_coordinates": finite_list(r_coords),
            "per_panel_gradient_ritz_coordinates": [
                finite_list(row) for row in panel_mode_matrix
            ],
            "mode_gradient_noise_variance": finite_list(mode_noise_variance),
            "rayleigh_matrix": [finite_list(row) for row in rayleigh],
        },
        "runtime": {
            "seconds": time.monotonic() - started,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "torch_threads": torch.get_num_threads(),
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=96)
    parser.add_argument("--fragments", type=int, default=4)
    parser.add_argument("--fragment-pattern", choices=["binpack", "strided"], default="binpack")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--train-on", choices=["assistant", "all"], default="assistant")
    parser.add_argument("--loss-function", default="cross_entropy")
    parser.add_argument("--probe-panels", type=int, default=4)
    parser.add_argument("--probe-batch-size", type=int, default=1)
    parser.add_argument("--probe-max-rows", type=int, default=128)
    parser.add_argument("--block-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--randomized-start",
        action="store_true",
        help=(
            "replace the transverse-buffer Lanczos start with a seeded "
            "Gaussian direction orthogonal to the held-out gradient"
        ),
    )
    args = parser.parse_args(argv)
    if args.threads is not None and args.threads <= 0:
        parser.error("--threads must be positive")
    if args.fragments <= 0 or args.probe_panels <= 0 or args.block_steps <= 0:
        parser.error("fragment, panel, and block counts must be positive")
    if not args.checkpoint.is_file() or args.checkpoint.is_symlink():
        parser.error("--checkpoint must be a regular non-symlink file")
    if not args.data.is_file() or args.data.is_symlink():
        parser.error("--data must be a regular non-symlink file")
    if args.output.exists():
        parser.error(f"refusing to overwrite --output: {args.output}")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"status": "COMPLETE", "output": str(args.output), **result["runtime"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
