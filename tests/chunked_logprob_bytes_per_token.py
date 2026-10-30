"""Bytes of activation per token that the chunked log-prob removes (S28 FCE).

Run by ``tests/test_s28_chunked_logprob.py`` in an interpreter that has Megatron-Core
(``YETO_MEGATRON_LM_PATH``) and verl's import closure (``YETO_VERL_FORK_PATH``, plus
``YETO_VERL_EXTRA_SITE``); not collected by pytest. Standalone:
``python chunked_logprob_bytes_per_token.py <out.json> [--tokens N N N] [--chunk N]``.

The 120B island's last pipeline stage holds, per token, this rank's ``[1, V/TP]`` logits
on the way to the log-probs. The reference path is verl's own, run for real at TP=1:
``logits.div_(temperature)``, ``logits.clone()`` (verl does it when it asks for entropy),
``vocab_parallel_entropy`` and ``vocab_parallel_log_probs_from_logits``
(``verl/utils/megatron/tensor_parallel.py``), which is what the Megatron engine runs with
``use_fused_kernels`` off (``transformer_impl.py``). The measured path is
:func:`chunked_logprob.linear_cross_entropy` on the same inputs.

Two numbers per path, both as a slope against the token count, so that every term which
does not grow with the sequence (the weight, its gradient, one chunk's buffers) drops out:

* ``retained``  -- bytes still referenced when the forward returns, i.e. the activation
  kept for the backward. This is the quantity the island's activation accounting adds up
  (``SESSION28/G-120C.md`` §9).
* ``peak``      -- bytes at the high-water mark over forward and backward together.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import weakref
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _key in ("YETO_VERL_EXTRA_SITE", "YETO_MEGATRON_LM_PATH", "YETO_VERL_FORK_PATH"):
    _path = os.environ.get(_key)
    if _path and _path not in sys.path:
        sys.path.insert(0, _path)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # verl's entropy wraps a helper in torch.compile

# the 120B island at TP4: 131 072 vocabulary entries over 4 ranks, hidden 4096, bf16
VOCAB_LOCAL = 32768
HIDDEN = 4096
TEMPERATURE = 1.0


def _tracker_class():
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    class LiveBytes(TorchDispatchMode):
        """Bytes of the tensor storages created inside the mode that are still referenced."""

        def __init__(self):
            super().__init__()
            self.live = self.peak = 0
            self.refs: dict[int, list[int]] = {}

        def _track(self, t):
            storage = t.untyped_storage()
            key = storage.data_ptr()
            if key == 0:
                return
            if key not in self.refs:
                self.refs[key] = [0, storage.nbytes()]
                self.live += storage.nbytes()
                self.peak = max(self.peak, self.live)
            self.refs[key][0] += 1
            weakref.finalize(t, self._drop, key)

        def _drop(self, key):
            entry = self.refs.get(key)
            if entry is not None:
                entry[0] -= 1
                if entry[0] == 0:
                    self.live -= entry[1]
                    del self.refs[key]

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for t in out if isinstance(out, (tuple, list)) else (out,):
                if isinstance(t, torch.Tensor):
                    self._track(t)
            return out

    return LiveBytes


def _setup() -> None:
    """One gloo rank with Megatron's TP=1 groups, so verl's reference really runs."""
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as mpu

    if dist.is_initialized():
        return
    torch.cuda.current_device = lambda: None
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29571")
    dist.init_process_group("gloo", rank=0, world_size=1)
    mpu.initialize_model_parallel(tensor_model_parallel_size=1)


def _verl_reference(hidden, weight, labels):
    """verl's non-fused Megatron path: the output layer's logits, then its own log-probs
    and entropy over them (one rank, so the all-reduces are no-ops but still run)."""
    import torch
    from verl.utils.megatron.tensor_parallel import (vocab_parallel_entropy,
                                                     vocab_parallel_log_probs_from_logits)

    logits = torch.matmul(hidden, weight.t())
    logits.div_(torch.tensor(TEMPERATURE, dtype=logits.dtype))
    logits_bak = logits.clone()  # verl clones whenever it asks for entropy
    entropy = vocab_parallel_entropy(logits)
    logprobs = vocab_parallel_log_probs_from_logits(logits_bak, labels)
    return logprobs, entropy


def _chunked(hidden, weight, labels, chunk):
    from yeto.rl.adapters.verl import chunked_logprob as cl

    return cl.ChunkedVocabParallelLogprob.apply(hidden, weight, labels, TEMPERATURE, chunk, None)


def _one(tokens: int, chunk: int, reference: bool) -> tuple[int, int]:
    """(bytes retained when the forward returns, peak bytes over forward and backward)."""
    import torch

    gen = torch.Generator().manual_seed(0)
    hidden = (torch.randn(tokens, HIDDEN, generator=gen) * 0.05).to(torch.bfloat16).requires_grad_()
    weight = (torch.randn(VOCAB_LOCAL, HIDDEN, generator=gen) * 0.05).to(torch.bfloat16).requires_grad_()
    labels = torch.randint(0, VOCAB_LOCAL, (tokens,), generator=gen)
    upstream = torch.randn(tokens, generator=gen)
    with _tracker_class()() as mode:
        logprobs, entropy = _verl_reference(hidden, weight, labels) if reference \
            else _chunked(hidden, weight, labels, chunk)
        retained = mode.live
        ((logprobs * upstream).sum() + entropy.sum()).backward()
        peak = mode.peak
    return retained, peak


def measure(tokens: tuple[int, ...] = (512, 1024, 2048), chunk: int = 256) -> dict:
    _setup()
    out: dict = {"vocab_local": VOCAB_LOCAL, "hidden": HIDDEN, "chunk": chunk,
                 "dtype": "bfloat16", "tokens": list(tokens)}
    for path, reference in (("verl_unfused", True), ("chunked", False)):
        pairs = [_one(t, chunk, reference) for t in tokens]
        out[path] = {"retained": [p[0] for p in pairs], "peak": [p[1] for p in pairs]}
    span = tokens[-1] - tokens[0]
    for kind in ("retained", "peak"):
        for path in ("verl_unfused", "chunked"):
            series = out[path][kind]
            out[f"{kind}_per_token_{path}"] = (series[-1] - series[0]) / span
        out[f"{kind}_per_token_saved"] = (out[f"{kind}_per_token_verl_unfused"]
                                          - out[f"{kind}_per_token_chunked"])
    # the [1, V/TP] rows the chunked path never keeps, 8 bytes per vocabulary entry: verl's
    # bf16 logits, the bf16 ``logits.clone()`` it makes to compute entropy, and the fp32
    # copy Megatron's cross entropy saves for its backward (``vocab_parallel_logits.float()``)
    out["retained_per_token_estimate"] = VOCAB_LOCAL * (2 + 2 + 4)
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path, nargs="?")
    parser.add_argument("--tokens", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--chunk", type=int, default=256)
    args = parser.parse_args(argv)
    result = measure(tuple(args.tokens), args.chunk)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
