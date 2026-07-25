"""Inspect assistant target density for a Yeto chat JSONL file.

Run this before a long training job. It catches bad assistant-only packing where
the model receives mostly zero-weight blocks and the logged loss stays near 0.
"""

from __future__ import annotations

import argparse

from transformers import AutoTokenizer

from yeto.data import StreamingPackedBlocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-12B-it")
    parser.add_argument("--data", required=True)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--blocks", type=int, default=100)
    parser.add_argument("--train-on", choices=["assistant", "all"], default="assistant")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    stream = iter(
        StreamingPackedBlocks(
            args.data,
            tokenizer,
            learner_id=0,
            num_learners=1,
            seq_len=args.seq_len,
            train_on=args.train_on,
        )
    )

    weights = []
    for _ in range(args.blocks):
        _, block_weights = next(stream)
        weights.append(float(block_weights.sum()))

    raw_tokens = args.blocks * args.seq_len
    target_tokens = sum(weights)
    print("blocks", args.blocks)
    print("zero_blocks", sum(value == 0 for value in weights))
    print("target_tokens", target_tokens)
    print("raw_tokens", raw_tokens)
    print("fraction", target_tokens / raw_tokens if raw_tokens else 0.0)
    print("min_max", min(weights), max(weights))


if __name__ == "__main__":
    main()

