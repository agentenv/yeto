"""Run a small, pinned Miles/FSDP CyberGym baseline.

This is intentionally a subprocess launcher rather than a second training
loop.  It starts a local Ray head when needed and submits the documented Miles
command with Yeto's reward module on ``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence


MILES_COMMIT = "dfc66ff38752bfa2c5d325e0037ebc4b537c06de"
MILES_REPOSITORY = "https://github.com/radixark/miles"
REWARD_PATH = "yeto_miles_cybergym.reward.score"


def build_train_command(args: argparse.Namespace) -> list[str]:
    if args.samples_per_iteration <= 0:
        raise ValueError("samples_per_iteration must be positive")
    if args.samples_per_iteration % args.samples_per_prompt:
        raise ValueError("samples_per_iteration must be divisible by samples_per_prompt")
    if args.samples_per_prompt < 2:
        raise ValueError("samples_per_prompt must be at least 2 for GRPO")
    groups = args.samples_per_iteration // args.samples_per_prompt
    return [
        sys.executable,
        str(args.miles_root / "train.py"),
        "--train-backend", "fsdp",
        "--hf-checkpoint", str(args.model),
        "--actor-num-nodes", "1",
        "--actor-num-gpus-per-node", str(args.trainer_gpus),
        "--rollout-num-gpus", str(args.rollout_gpus),
        "--rollout-num-gpus-per-engine", "1",
        "--num-gpus-per-node", str(args.trainer_gpus + args.rollout_gpus),
        "--prompt-data", str(args.prompt_data),
        "--input-key", "messages",
        "--metadata-key", "metadata",
        "--apply-chat-template",
        "--rollout-shuffle",
        "--num-rollout", str(args.iterations),
        "--rollout-batch-size", str(groups),
        "--n-samples-per-prompt", str(args.samples_per_prompt),
        "--num-steps-per-rollout", "1",
        "--global-batch-size", str(args.samples_per_iteration),
        "--rollout-max-context-len", str(args.max_context_len),
        "--rollout-max-prompt-len", str(args.max_context_len // 2),
        "--rollout-max-response-len", str(args.max_response_len),
        "--rollout-temperature", str(args.temperature),
        "--advantage-estimator", "grpo",
        "--eps-clip", "0.2",
        "--eps-clip-high", "0.28",
        "--entropy-coef", "0",
        "--kl-coef", "0",
        "--optimizer", "adam",
        "--lr", str(args.lr),
        "--lr-decay-style", "constant",
        "--weight-decay", "0",
        "--micro-batch-size", "1",
        "--custom-rm-path", REWARD_PATH,
        "--sglang-disable-cuda-graph",
        "--update-weight-buffer-size", "536870912",
    ]


def build_runtime_env(repo_root: Path, miles_root: Path, args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = [str(repo_root), str(miles_root)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(pythonpath),
            "CYBERGYM_URL": args.cybergym_url.rstrip("/"),
            "CYBERGYM_AGENT_ID": args.agent_id,
            "CYBERGYM_TIMEOUT": str(args.cybergym_timeout),
            "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
            "MASTER_ADDR": "127.0.0.1",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    if args.api_key:
        env["CYBERGYM_API_KEY"] = args.api_key
    return env


def _git_output(root: Path, *command: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *command],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_miles_checkout(root: str | Path) -> Path:
    path = Path(root).expanduser().resolve()
    try:
        commit = _git_output(path, "rev-parse", "HEAD")
        origin = _git_output(path, "config", "--get", "remote.origin.url")
        tracked_changes = subprocess.run(
            ["git", "-C", str(path), "diff", "--quiet"], check=False
        ).returncode
        staged_changes = subprocess.run(
            ["git", "-C", str(path), "diff", "--cached", "--quiet"], check=False
        ).returncode
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot verify Miles checkout at {path}") from exc
    if commit != MILES_COMMIT:
        raise RuntimeError(f"Miles revision mismatch: expected {MILES_COMMIT}, got {commit}")
    if origin.removesuffix(".git").rstrip("/") != MILES_REPOSITORY:
        raise RuntimeError(f"Miles origin mismatch: expected {MILES_REPOSITORY}, got {origin}")
    if tracked_changes or staged_changes:
        raise RuntimeError("Miles checkout has tracked source changes; use the pinned clean checkout")
    if not (path / "train.py").is_file():
        raise RuntimeError(f"Miles checkout has no train.py: {path}")
    return path


def _ray_command(args: argparse.Namespace) -> list[str]:
    env_vars = {
        key: value
        for key, value in build_runtime_env(args.repo_root, args.miles_root, args).items()
        if key in {
            "PYTHONPATH", "CYBERGYM_URL", "CYBERGYM_AGENT_ID", "CYBERGYM_API_KEY",
            "CYBERGYM_TIMEOUT", "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR", "MASTER_ADDR", "no_proxy",
        }
    }
    runtime = json.dumps({"env_vars": env_vars}, sort_keys=True)
    return [
        "ray", "job", "submit", "--address=http://127.0.0.1:8265",
        f"--runtime-env-json={runtime}", "--",
        *build_train_command(args),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--miles-root", type=Path, default=Path("/workspace/miles"))
    parser.add_argument("--prompt-data", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--samples-per-iteration", type=int, default=4)
    parser.add_argument("--samples-per-prompt", type=int, default=2)
    parser.add_argument("--trainer-gpus", type=int, default=1)
    parser.add_argument("--rollout-gpus", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-context-len", type=int, default=1024)
    parser.add_argument("--max-response-len", type=int, default=128)
    parser.add_argument("--cybergym-url", default="http://127.0.0.1:8666")
    parser.add_argument("--agent-id", default="yeto_agent")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--cybergym-timeout", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.repo_root = repo_root
    args.miles_root = args.miles_root.expanduser().resolve()
    args.prompt_data = args.prompt_data.expanduser().resolve()
    if args.iterations <= 0 or args.trainer_gpus <= 0 or args.rollout_gpus <= 0:
        parser.error("iterations and GPU counts must be positive")
    if not args.prompt_data.is_file() and not args.dry_run:
        parser.error(f"prompt data does not exist: {args.prompt_data}")
    if not args.dry_run:
        verify_miles_checkout(args.miles_root)
        runtime_env = build_runtime_env(repo_root, args.miles_root, args)
        subprocess.run(
            [sys.executable, "-c", "import miles, sglang"],
            cwd=args.miles_root,
            env=runtime_env,
            check=True,
        )
        ray_status = subprocess.run(["ray", "status"], check=False)
        if ray_status.returncode != 0:
            subprocess.run(
                [
                    "ray", "start", "--head", "--node-ip-address", "127.0.0.1",
                    "--num-gpus", str(args.trainer_gpus + args.rollout_gpus),
                    "--disable-usage-stats",
                ],
                check=True,
                env=runtime_env,
            )
    command = _ray_command(args)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    subprocess.run(command, cwd=args.miles_root, env=build_runtime_env(repo_root, args.miles_root, args), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
