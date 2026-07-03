#!/usr/bin/env python3
"""Yeto: efficient, low-cost post-training across clouds/regions via SkyPilot.

Example:
    python3 train.py \
        --gpu aws:8xa100@us-east-2,aws:8xa100@us-east-1,aws:8xa100@us-west-2 \
        --model deepseek4flash \
        --data armand0e/claude-fable-5-claude-code \
        --loss-function cross_entropy
"""

import argparse
import sys

from .losses import LOSS_FUNCTIONS


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--gpu",
        required=True,
        help="comma-separated learner clusters: cloud:[NODESx]COUNTxGPU[@region], "
        "e.g. aws:4x8xa100@us-east-2,gcp:8xa100@us-central1",
    )
    p.add_argument("--model", required=True, help="model alias (gemma4|deepseek4flash) or HF id")
    p.add_argument("--data", required=True, help="HF dataset id (messages-format chat traces)")
    def loss_spec(value: str) -> str:
        if value in LOSS_FUNCTIONS or value.startswith(("custom:", "pickle:")):
            return value
        raise argparse.ArgumentTypeError(
            f"expected one of {LOSS_FUNCTIONS} or custom:<file.py>[:<fn>]"
        )

    p.add_argument(
        "--loss-function",
        type=loss_spec,
        default="cross_entropy",
        help=f"one of {'|'.join(LOSS_FUNCTIONS)}, or custom:<file.py>[:<fn>] "
        "defining fn(logits, input_ids, weights) -> (loss, num_tokens); the "
        "callable is pickled by value and shipped to all learners",
    )

    tune = p.add_argument_group("fine-tuning")
    tune.add_argument("--tuning", choices=["lora", "full"], default="lora")
    tune.add_argument(
        "--train-on",
        choices=["assistant", "all"],
        default="assistant",
        help="which tokens carry loss: assistant-message tokens only (default) or every token",
    )
    tune.add_argument("--lora-r", type=int, default=16)
    tune.add_argument("--seq-len", type=int, default=2048)
    tune.add_argument("--micro-batch-size", type=int, default=1)
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

    sync = p.add_argument_group("async sync")
    sync.add_argument("--total-steps", type=int, default=64, help="outer steps T (one fragment each)")
    sync.add_argument("--fragments", type=int, default=8, help="fragments P (= sync interval H)")
    sync.add_argument("--quorum", type=int, default=1, help="minimum learners per outer step (K)")
    sync.add_argument("--grace-ms", type=int, default=1000, help="grace window after quorum")
    sync.add_argument("--outer-lr", type=float, default=0.7)
    sync.add_argument("--outer-momentum", type=float, default=0.9)
    sync.add_argument("--wire-dtype", choices=["bf16", "f32"], default="bf16")
    sync.add_argument("--wan-streams", type=int, default=4, help="parallel TCP streams per learner")

    infra = p.add_argument_group("infrastructure")
    infra.add_argument(
        "--spot",
        action="store_true",
        default=True,
        help="use spot instances for learners (default)",
    )
    infra.add_argument(
        "--on-demand",
        dest="spot",
        action="store_false",
        help="use on-demand instances for learners instead of spot",
    )
    infra.add_argument("--disk-size", type=int, default=512, help="learner disk (GB)")
    infra.add_argument(
        "--learner-cpus",
        default=None,
        help="vCPU hint per learner node (e.g. '8+') to steer instance selection",
    )
    infra.add_argument(
        "--learner-instance-type",
        default=None,
        help="pin learner nodes to an exact instance type (e.g. gr6.4xlarge)",
    )
    infra.add_argument(
        "--syncer-region",
        default="us-west-2",
        help="syncer VM placement: 'region' (AWS) or 'cloud/region', e.g. gcp/us-central1",
    )
    infra.add_argument("--syncer-memory", type=int, default=32, help="syncer RAM (GB)")
    infra.add_argument("--cluster-prefix", default="yeto")
    infra.add_argument("--keep", action="store_true", help="do not tear down clusters at the end")
    infra.add_argument(
        "--retry-until-up",
        action="store_true",
        help="keep retrying learner provisioning until capacity is found",
    )
    infra.add_argument(
        "--recover-timeout",
        type=int,
        default=1200,
        help="seconds to keep re-provisioning a failed/preempted learner before "
        "tearing it down and continuing with the remaining fleet (0 tears the "
        "learner down on its first failure; the syncer is always recovered)",
    )
    infra.add_argument(
        "--controller-poll",
        type=int,
        default=30,
        help="fleet-controller health poll interval (seconds)",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from .launcher import run

    return run(args)


if __name__ == "__main__":
    sys.exit(main())
