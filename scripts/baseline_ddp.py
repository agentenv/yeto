#!/usr/bin/env python3
"""Synchronous FSDP baseline for comparison against Yeto.

Mirrors the Yeto arm's topology exactly — 3x gr6.4xlarge (1x L4 each) in
one region — but trains with synchronous multi-node FSDP (torchrun across
nodes, --syncer none) instead of async fragment merging. Same model, data,
LoRA config, and per-step token budget.
"""

import sys
from pathlib import Path

import sky

REPO_ROOT = Path(__file__).resolve().parent.parent
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 64

RUN = (
    'MASTER_ADDR=$(echo "$SKYPILOT_NODE_IPS" | head -n1)\n'
    "torchrun --nnodes=$SKYPILOT_NUM_NODES --node_rank=$SKYPILOT_NODE_RANK "
    "--nproc_per_node=1 --master_addr=$MASTER_ADDR --master_port=29500 "
    "-m yeto.learner "
    "--model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B "
    "--data armand0e/claude-fable-5-claude-code "
    "--syncer none --shard fsdp --learner-id 0 --num-learners 1 --max-rows 12 "
    "--tuning lora --seq-len 1024 --micro-batch-size 1 --grad-accum 2 "
    f"--max-local-steps {STEPS} --output-dir ~/yeto-output"
)


def main() -> int:
    task = sky.Task(
        name="fsdp-baseline",
        setup="pip install -q -r requirements.txt",
        run=RUN,
        envs={"HF_HUB_ENABLE_HF_TRANSFER": "1"},
        num_nodes=3,
        workdir=str(REPO_ROOT),
    )
    task.set_resources(
        sky.Resources(
            infra="aws/us-west-2",
            accelerators="L4:1",
            # Same instance type as the Yeto learners (and the best spot
            # placement score of the G family at the time of writing).
            instance_type="gr6.4xlarge",
            use_spot=True,
            disk_size=256,
        )
    )
    job_id, _handle = sky.stream_and_get(
        sky.launch(task, cluster_name="cmp-fsdp", retry_until_up=True)
    )
    return sky.tail_logs("cmp-fsdp", job_id, follow=True)


if __name__ == "__main__":
    sys.exit(main())
