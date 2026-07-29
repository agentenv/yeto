"""GPU smoke test: real Rust syncer + two real SmolLM2 learners on CUDA.

Unlike test_integration.py's ToyLearner (quadratic loss, no model), this
drives the actual `yeto.learner` entrypoint end to end on a real model:
HF load, LoRA wrap, CUDA dtype selection, autobatch/OOM handling, tokenize
stream, HELLO layout exchange, pull/push rounds, and terminal finalization.
It exists to exercise the accelerator code paths that CPU CI cannot see.

Both learners share one visible GPU — SmolLM2-135M with r=8 LoRA fits
twice on anything the self-hosted runner would offer. Model weights come
from the HF cache; the first run on a fresh runner downloads ~270 MB.
"""

import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from yeto.export import parse_checkpoint

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]

ROOT = Path(__file__).resolve().parent.parent
# Instruct variant: it ships a chat template, so messages-format rows
# tokenize without any repo-side fixup.
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
TOTAL_STEPS = 3
NUM_LEARNERS = 2


def build_syncer() -> Path:
    binary = ROOT / "syncer/target/debug/yeto-syncer"
    subprocess.run(["cargo", "build", "-q"], cwd=ROOT / "syncer", check=True)
    assert binary.exists()
    return binary


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def write_train_jsonl(path: Path, rows: int = 48) -> Path:
    """Tiny synthetic messages-format dataset — no network dependency."""
    with path.open("w", encoding="utf-8") as handle:
        for i in range(rows):
            row = {
                "messages": [
                    {"role": "user", "content": f"Count from 1 to {i % 9 + 2}."},
                    {
                        "role": "assistant",
                        "content": " ".join(str(j) for j in range(1, i % 9 + 3)),
                    },
                ]
            }
            handle.write(json.dumps(row) + "\n")
    return path


def learner_command(
    learner_id: int, syncer_port: int, data: Path, output_dir: Path
) -> list[str]:
    # Mirrors scripts/compare_diloco.py learner_command: the learner is
    # always launched under torch.distributed.run, nproc=1 per learner.
    return [
        sys.executable,
        "-m", "torch.distributed.run",
        "--nproc_per_node=1",
        "--master_addr=127.0.0.1",
        f"--master_port={free_port()}",
        "-m", "yeto.learner",
        "--model", MODEL,
        "--data", str(data),
        "--syncer", f"127.0.0.1:{syncer_port}",
        "--learner-id", str(learner_id),
        "--num-learners", str(NUM_LEARNERS),
        "--tuning", "lora",
        "--lora-r", "8",
        "--lora-alpha", "16",
        "--seq-len", "256",
        "--micro-batch-size", "1",
        "--grad-accum", "1",
        "--inner-lr", "3e-4",
        "--warmup-steps", "2",
        "--seed", str(17 + learner_id),
        "--max-local-steps", "200",  # safety stop; the syncer finalizes first
        "--tokenize", "stream",
        "--stream-workers", "0",
        "--train-on", "all",
        "--fragments", "4",
        "--wan-streams", "2",
        "--shard", "ddp",
        "--device", "cuda",
        "--output-dir", str(output_dir),
    ]


@pytest.mark.timeout(1800)
def test_two_smollm_learners_finalize_on_cuda(tmp_path):
    binary = build_syncer()
    port = free_port()
    data = write_train_jsonl(tmp_path / "train.jsonl")
    checkpoint = tmp_path / "state.ckpt"
    tape = tmp_path / "tape.jsonl"

    syncer = subprocess.Popen(
        [
            str(binary),
            "--port", str(port),
            "--learners", str(NUM_LEARNERS),
            "--quorum", str(NUM_LEARNERS),
            "--grace-ms", "500",
            "--total-steps", str(TOTAL_STEPS),
            "--outer-lr", "0.7",
            "--outer-momentum", "0.9",
            "--checkpoint-path", str(checkpoint),
            "--checkpoint-every", "1",
            "--event-tape", str(tape),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    learners = []
    try:
        for learner_id in range(NUM_LEARNERS):
            learners.append(
                subprocess.Popen(
                    learner_command(
                        learner_id, port, data, tmp_path / f"learner-{learner_id}"
                    ),
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )
        outputs = []
        for learner_id, proc in enumerate(learners):
            out, _ = proc.communicate(timeout=1500)
            outputs.append(out)
            assert proc.returncode == 0, (
                f"learner {learner_id} exited {proc.returncode}\n{out[-4000:]}"
            )
        syncer_out, _ = syncer.communicate(timeout=60)
        assert syncer.returncode == 0, syncer_out[-4000:]

        assert checkpoint.exists(), "syncer wrote no checkpoint"
        parsed = parse_checkpoint(checkpoint)
        assert parsed.global_step == TOTAL_STEPS

        # The machinery, not just liveness: every outer step must have merged
        # real pushed deltas from BOTH learners, each covering >= 1 inner step
        # over a nonzero token count (test_integration.py checks the toy
        # learners the same way).
        records = [
            json.loads(line) for line in tape.read_text().splitlines()
        ]
        assert records, "syncer wrote an empty event tape"
        assert {record["step"] for record in records} == set(
            range(1, TOTAL_STEPS + 1)
        )
        for record in records:
            responders = record["responders"]
            assert {responder["id"] for responder in responders} == set(
                range(NUM_LEARNERS)
            ), f"step {record['step']} merged without both learners: {record}"
            for responder in responders:
                assert responder["c_steps"] >= 1, record
                assert responder["c_tokens"] > 0, record
    finally:
        for proc in [syncer, *learners]:
            if proc.poll() is None:
                proc.kill()
        for label, out in zip(("syncer",), (locals().get("syncer_out", ""),)):
            if out:
                print(f"--- {label} ---\n{out[-3000:]}")
