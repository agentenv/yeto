"""Weighted next-token CE of a base model (+ optional PEFT adapter) on SFT rows.

Blocks are built with yeto.data._target_packed_blocks (row file order, no
shuffle), i.e. the same _row_tokens + _append_target_blocks path the learner
uses. Loss/token = sum(w[t+1] * CE(t -> t+1)) / sum(w[t+1]) over all blocks,
matching the learner's one-token shift (yeto/learner.py target count).

Usage:
  PYTHONPATH=. python tools/eval_sft_heldout.py --model Qwen/Qwen3-0.6B \
      [--model-revision REV] [--adapter DIR] --data eval.jsonl \
      [--seq-len 4096] [--assistant-mask-mode legacy|native] [--max-blocks N]
"""
from __future__ import annotations

import argparse
import json
import sys

import torch
import torch.nn.functional as F

from yeto.data import _target_packed_blocks, load_rows


def build_blocks(tokenizer, data, seq_len, mask_mode, data_format="auto", max_rows=None):
    rows = load_rows(data)
    n = len(rows) if max_rows is None else min(len(rows), max_rows)
    return _target_packed_blocks(
        rows, tokenizer, list(range(n)), seq_len, "assistant", mask_mode, data_format
    )


def load_model(model_id, revision, adapter, dtype):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype)
    info = None
    if adapter:
        from peft import PeftModel
        from peft.utils import set_peft_model_state_dict
        from safetensors.torch import load_file

        model = PeftModel.from_pretrained(model, adapter)
        # Re-apply explicitly to surface any key mismatch as an error.
        sd = load_file(f"{adapter}/adapter_model.safetensors")
        res = set_peft_model_state_dict(model, sd)
        info = {
            "adapter_tensors": len(sd),
            "missing_lora_keys": [k for k in res.missing_keys if "lora_" in k],
            "unexpected_keys": list(res.unexpected_keys),
        }
        if info["missing_lora_keys"] or info["unexpected_keys"]:
            raise SystemExit(f"adapter key mismatch: {json.dumps(info)[:2000]}")
    model.eval()
    return model, info


@torch.no_grad()
def weighted_ce(model, blocks, weights):
    total_loss, total_w = 0.0, 0.0
    for ids, w in zip(blocks, weights):
        x = torch.tensor([ids], dtype=torch.long)
        wt = torch.tensor(w[1:], dtype=torch.float64)
        logits = model(input_ids=x).logits[0, :-1].float()
        ce = F.cross_entropy(logits, x[0, 1:], reduction="none").double()
        total_loss += float((ce * wt).sum())
        total_w += float(wt.sum())
    return total_loss, total_w


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--model-revision", default=None)
    p.add_argument("--adapter", default=None)
    p.add_argument("--data", required=True)
    p.add_argument("--data-format", default="auto")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--assistant-mask-mode", choices=["legacy", "native"], default="legacy")
    p.add_argument("--max-blocks", type=int, default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--output", default=None)
    a = p.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.adapter or a.model, revision=None if a.adapter else a.model_revision)
    blocks, weights = build_blocks(tok, a.data, a.seq_len, a.assistant_mask_mode, a.data_format, a.max_rows)
    total_blocks = len(blocks)
    if a.max_blocks is not None:
        blocks, weights = blocks[: a.max_blocks], weights[: a.max_blocks]
    model, info = load_model(a.model, a.model_revision, a.adapter, getattr(torch, a.dtype))
    loss, tw = weighted_ce(model, blocks, weights)
    out = {
        "model": a.model, "model_revision": a.model_revision, "adapter": a.adapter,
        "data": a.data, "seq_len": a.seq_len, "assistant_mask_mode": a.assistant_mask_mode,
        "dtype": a.dtype, "blocks_total": total_blocks, "blocks_used": len(blocks),
        "total_tokens": sum(len(b) for b in blocks),
        "target_tokens": tw, "loss_sum": loss,
        "loss_per_token": loss / tw if tw else None,
        "adapter_check": info,
    }
    s = json.dumps(out, indent=2)
    if a.output:
        open(a.output, "w").write(s + "\n")
    print(s)


if __name__ == "__main__":
    sys.exit(main())
