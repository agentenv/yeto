"""Critic warm-up as its own stage (change ``rl-algo-critic-family``, design D5).

Stage W is one plain Miles launch (``train.py``, the mode of Miles' own PPO
example), not the ports driver loop: the critic is built from the initial
actor checkpoint (``--critic-load`` = actor checkpoint; the 1-output value
head is new, Miles model_provider.py:340-341), trains ``warmup_steps``
critic-only rollout steps (``--num-critic-only-steps`` = ``--num-rollout`` =
``warmup_steps``, so the actor never trains and is never saved, Miles
train.py:87-88/123) and saves only the critic (``--critic-save``). The main
stage (ports) then runs with ``--num-critic-only-steps 0`` and
``--critic-load`` = the stage-W product, whose content hash enters the
receipts (``CriticRunConfig.init_sha256``).

This module builds the stage-W argv, hashes checkpoints, writes / validates
the product manifest (actor unchanged across the warm-up, critic hash) and
reuses one product for every island (keyed by algorithm + initial actor).
Launching stage W on a cluster is not wired here (GPU G1, task 5.3).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .engine.algorithm import AlgorithmSpec, AlgorithmSpecError
from .engine.run_config import CriticRunConfig

MANIFEST = "critic-warmup.json"
SCHEMA = "yeto-rl-critic-warmup-v1"

# Main-stage flags stage W replaces or must not carry: the critic schedule and
# checkpoint flags it sets itself, and the ports driver hooks / eval that only
# run under the ports driver (stage W has no driver, publisher or eval).
_STAGE_W_VALUE_FLAGS = (
    "--num-rollout", "--num-critic-only-steps", "--critic-load", "--critic-save",
    "--save", "--save-interval", "--rollout-sample-filter-path",
    "--rollout-all-samples-process-path", "--buffer-filter-path", "--eval-interval",
    "--n-samples-per-eval-prompt", "--eval-max-response-len", "--eval-top-p",
    "--eval-temperature", "--eval-max-context-len", "--eval-max-prompt-len",
)
_STAGE_W_MULTI_VALUE_FLAGS = ("--eval-prompt-data",)  # name + path
_STAGE_W_SWITCHES = ("--skip-eval-before-train",)


class CriticWarmupError(RuntimeError):
    pass


def _require_warmup(spec: AlgorithmSpec) -> int:
    critic = spec.critic
    if not spec.execution.needs_critic or critic.init != "copy_actor_backbone" \
            or not critic.warmup_steps:
        raise AlgorithmSpecError(
            "stage W needs a critic algorithm with critic.init='copy_actor_backbone' and "
            "critic.warmup_steps > 0"
        )
    return int(critic.warmup_steps)


def _strip(argv: Sequence[str]) -> list[str]:
    out: list[str] = []
    tokens = list(argv)
    i = 0
    while i < len(tokens):
        token = tokens[i]
        name = token.split("=", 1)[0]
        if name in _STAGE_W_SWITCHES:
            i += 1
        elif name in _STAGE_W_VALUE_FLAGS:
            i += 1 if "=" in token else 2
        elif name in _STAGE_W_MULTI_VALUE_FLAGS:
            i += 1 if "=" in token else 3
        else:
            out.append(token)
            i += 1
    return out


def warmup_stage_argv(main_argv: Sequence[str], spec: AlgorithmSpec, *,
                      actor_checkpoint: str, critic_save: str) -> list[str]:
    """Stage-W Miles argv derived from the main-stage (ports) argv.

    Same model, data, batch and algorithm flags as the main stage; the critic
    schedule and checkpoint flags are stage W's own.
    """

    steps = _require_warmup(spec)
    return _strip(main_argv) + [
        "--num-rollout", str(steps),
        "--num-critic-only-steps", str(steps),
        "--critic-load", str(actor_checkpoint),
        "--critic-save", str(critic_save),
        "--save-interval", str(steps),
    ]


def main_stage_critic_args(argv: Sequence[str]) -> dict[str, str | None]:
    """``--num-critic-only-steps`` / ``--critic-load`` values of an argv (last wins)."""

    out: dict[str, str | None] = {"--num-critic-only-steps": None, "--critic-load": None}
    tokens = list(argv)
    for i, token in enumerate(tokens):
        name, eq, value = token.partition("=")
        if name in out:
            out[name] = value if eq else (tokens[i + 1] if i + 1 < len(tokens) else None)
    return out


# --------------------------------------------------------------------------
# hashing and the product manifest
# --------------------------------------------------------------------------


def checkpoint_sha256(path: str | os.PathLike) -> str:
    """Content hash of a checkpoint directory (relative paths + bytes, sorted)."""

    root = Path(path)
    if not root.is_dir():
        raise CriticWarmupError(f"checkpoint directory {root} does not exist")
    digest = hashlib.sha256(b"yeto-rl-checkpoint-v1\0")
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != MANIFEST)
    if not files:
        raise CriticWarmupError(f"checkpoint directory {root} is empty")
    for file in files:
        digest.update(str(file.relative_to(root)).encode() + b"\0")
        with open(file, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class WarmupProduct:
    algorithm_sha256: str
    warmup_steps: int
    actor_checkpoint: str
    actor_sha256: str  # before and after the warm-up (must be equal)
    critic_checkpoint: str
    critic_sha256: str
    schema: str = SCHEMA

    def critic_run_config(self) -> CriticRunConfig:
        return CriticRunConfig(critic_load=self.critic_checkpoint, init_sha256=self.critic_sha256)


def reuse_key(spec: AlgorithmSpec, actor_sha256: str) -> str:
    """Two islands with the same algorithm and initial actor share one product."""

    return hashlib.sha256(f"{spec.sha256()}:{actor_sha256}".encode()).hexdigest()


def finish_warmup(spec: AlgorithmSpec, *, actor_checkpoint: str, actor_sha256_before: str,
                  critic_checkpoint: str) -> WarmupProduct:
    """After stage W: verify the actor is unchanged, hash the critic, write the manifest."""

    steps = _require_warmup(spec)
    after = checkpoint_sha256(actor_checkpoint)
    if after != actor_sha256_before:
        raise CriticWarmupError(
            f"actor checkpoint changed during the critic warm-up ({actor_sha256_before} -> "
            f"{after}); stage W must not train or save the actor"
        )
    product = WarmupProduct(
        algorithm_sha256=spec.sha256(), warmup_steps=steps,
        actor_checkpoint=str(actor_checkpoint), actor_sha256=after,
        critic_checkpoint=str(critic_checkpoint),
        critic_sha256=checkpoint_sha256(critic_checkpoint),
    )
    target = Path(critic_checkpoint) / MANIFEST
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(product), sort_keys=True, indent=1))
    os.replace(tmp, target)
    return product


def load_product(critic_checkpoint: str | os.PathLike, spec: AlgorithmSpec, *,
                 actor_sha256: str) -> WarmupProduct:
    """Validate a stage-W product for this algorithm and initial actor."""

    path = Path(critic_checkpoint) / MANIFEST
    try:
        product = WarmupProduct(**json.loads(path.read_text()))
    except (OSError, TypeError, ValueError) as exc:
        raise CriticWarmupError(f"no valid warm-up manifest at {path}: {exc}") from exc
    problems = []
    if product.schema != SCHEMA:
        problems.append(f"schema {product.schema!r}")
    if product.algorithm_sha256 != spec.sha256():
        problems.append("algorithm hash differs")
    if product.warmup_steps != _require_warmup(spec):
        problems.append(f"warmup_steps {product.warmup_steps} != {spec.critic.warmup_steps}")
    if product.actor_sha256 != actor_sha256:
        problems.append("initial actor checkpoint hash differs")
    if checkpoint_sha256(critic_checkpoint) != product.critic_sha256:
        problems.append("critic checkpoint content differs from its manifest")
    if problems:
        raise CriticWarmupError(f"warm-up product {critic_checkpoint} rejected: {problems}")
    return product


def ensure_warmup(spec: AlgorithmSpec, *, actor_checkpoint: str, cache_root: str | os.PathLike,
                  run_stage: Callable[[str], None]) -> WarmupProduct:
    """The product for (algorithm, initial actor): reused when valid, else stage W
    runs once (``run_stage(critic_save_dir)``) and the product is validated."""

    actor_sha = checkpoint_sha256(actor_checkpoint)
    out = Path(cache_root) / reuse_key(spec, actor_sha)
    if (out / MANIFEST).exists():
        return load_product(out, spec, actor_sha256=actor_sha)
    out.mkdir(parents=True, exist_ok=True)
    run_stage(str(out))
    return finish_warmup(spec, actor_checkpoint=actor_checkpoint,
                         actor_sha256_before=actor_sha, critic_checkpoint=str(out))


# --------------------------------------------------------------------------
# dry run
# --------------------------------------------------------------------------


def dry_run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Both stages' algorithm argv for a spec (no engine, no checkpoint I/O)."""

    from .engine.algorithm import resolve_ports_algorithm
    from .engine.miles_adapter.algorithm_flags import absorb_extra_argv, algorithm_argv

    parser = argparse.ArgumentParser(prog="python3 -m yeto.rl.critic_warmup")
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.add_argument("--rl-algorithm-spec", default=None)
    parser.add_argument("--extra", default="")
    parser.add_argument("--actor-checkpoint", required=True)
    parser.add_argument("--critic-save", required=True)
    args = parser.parse_args(argv)
    base = resolve_ports_algorithm(args, rl_engine="ports")
    spec, _, _ = absorb_extra_argv(base, shlex.split(args.extra))
    problems = spec.rejections()
    if problems:
        return {"verdict": "rejected", "error": "; ".join(problems)}
    try:
        main = ["--advantage-estimator", spec.advantage_estimator, *algorithm_argv(spec),
                "--critic-load", "<stage-W product>"]
        stage_w = warmup_stage_argv(main, spec, actor_checkpoint=args.actor_checkpoint,
                                    critic_save=args.critic_save)
    except AlgorithmSpecError as exc:
        return {"verdict": "rejected", "error": str(exc)}
    return {"verdict": "accepted", "algorithm_spec_sha256": spec.sha256(),
            "stage_w_argv": stage_w, "main_argv": main}


def main(argv: Sequence[str] | None = None) -> int:
    result = dry_run(argv)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verdict"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
