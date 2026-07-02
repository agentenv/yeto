#!/usr/bin/env python3
"""Standard-DDP baseline for comparison against Decoupled DiLoCo.

Launches one spot node with 4x L4 and runs the same learner code in
--syncer none mode on 3 GPUs (matching 3 DiLoCo learners), same model,
data, LoRA config, and per-step token budget.
"""

import sys
from pathlib import Path

import sky

REPO_ROOT = Path(__file__).resolve().parent.parent
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 64

RUN = (
    "torchrun --nproc_per_node=3 -m decoupled_diloco.learner "
    "--model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B "
    "--data armand0e/claude-fable-5-claude-code "
    "--syncer none --learner-id 0 --num-learners 1 --max-rows 12 "
    f"--tuning lora --seq-len 1024 --micro-batch-size 1 --grad-accum 2 "
    f"--max-local-steps {STEPS} --output-dir ~/diloco-output"
)


def main() -> int:
    task = sky.Task(
        name="ddp-baseline",
        setup="pip install -q -r requirements.txt",
        run=RUN,
        envs={"HF_HUB_ENABLE_HF_TRANSFER": "1"},
        workdir=str(REPO_ROOT),
    )
    task.set_resources(
        sky.Resources(
            infra="aws/us-west-2",
            # Chosen by live spot placement scores (g6.12xlarge scored 3/10
            # at x1 vs 1/10 for all g5/A10G sizes); also the only 4-GPU type
            # under the 64-vCPU G-spot quota.
            accelerators="L4:4",
            instance_type="g6.12xlarge",
            use_spot=True,
            disk_size=256,
        )
    )
    job_id, _handle = sky.stream_and_get(
        sky.launch(task, cluster_name="cmp-ddp-g5", retry_until_up=True)
    )
    return sky.tail_logs("cmp-ddp-g5", job_id, follow=True)


if __name__ == "__main__":
    sys.exit(main())
