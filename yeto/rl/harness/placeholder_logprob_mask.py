"""Mask stop tokens whose generation logprob is an exact 0.0 placeholder (S19 #8).

Problem (s19-agentic5-r-20261010d round 1, 65 tokens): the agentic trajectory
contains ``<|endoftext|>`` (Qwen3.5 id 248044) with generation logprob exactly
``0.0`` and ``loss_mask == 1``; the trainer's current logprob for the same token
is -14 to -19.5.  The 0.0 is not a sampled probability:

- the pinned SGLang fork (``sglang-next`` 2fa880182, ``qwen3_coder_detector.py``
  ``get_auto_tool_call_structural_tag``) builds, for ``tool_choice=auto`` with
  ``parallel_tool_calls=false``, an xgrammar structural tag with
  ``stop_after_first=True``.  After the first ``</tool_call>`` the grammar is
  finished and its vocab mask allows only the grammar's stop tokens (the model
  config ``eos_token_id`` = 248044);
- ``model_runner.py`` applies that mask to the logits (``apply_logits_bias``)
  before ``layers/sampler.py`` takes ``log_softmax``.  With one allowed token
  the returned logprob is exactly ``0.0`` -- the probability under the masked
  distribution, not under the model;
- Miles ``session/samples/merge.py`` copies ``output_token_logprobs`` as is and
  sets ``loss_mask = 1`` for every output token.

So the importance ratio exp(current - rollout) of such a token is ~e^-15: TIS
gives it ~0 weight, a non-TIS loss gets a wrong ratio.  The model never chose
the token, so it must not be trained.  We cannot change SGLang/Miles here, so
the kept samples get ``loss_mask = 0`` on every position whose generated token
is a stop token AND whose rollout logprob is exactly 0.0.  Tokens, logprobs and
lengths are not changed (same contract as :mod:`.reasoning_loss_mask`).

Side effect, accepted: a real stop token sampled with probability that rounds
to exactly 1.0 in float32 is also masked; its policy-gradient term is ~0 anyway.

Neutral (no Miles import): works on any sample with ``tokens`` /
``response_length`` / ``loss_mask`` / ``rollout_log_probs``.
"""

from __future__ import annotations

from typing import Any, Iterable

COUNT_KEY = "placeholder_logprob_tokens"
# Stop-token names checked in the tokenizer besides eos/pad (Qwen family).
STOP_TOKEN_NAMES = ("<|endoftext|>", "<|im_end|>")
_STOP_IDS_ATTR = "_yeto_placeholder_stop_ids"


def stop_token_ids(tokenizer: Any) -> frozenset[int]:
    """eos/pad ids plus the known stop-token names that are single tokens."""
    ids: set[int] = set()
    for attr in ("eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int) and value >= 0:
            ids.add(value)
        elif isinstance(value, (list, tuple)):
            ids.update(int(v) for v in value if isinstance(v, int) and v >= 0)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    unk = getattr(tokenizer, "unk_token_id", None)
    if callable(convert):
        for name in STOP_TOKEN_NAMES:
            token_id = convert(name)
            if isinstance(token_id, int) and token_id >= 0 and token_id != unk:
                ids.add(token_id)
    return frozenset(ids)


def mask_placeholder_in_sample(sample: Any, stop_ids: frozenset[int]) -> int:
    """Set ``loss_mask`` 1 -> 0 on stop tokens with rollout logprob exactly 0.0; return the count."""
    tokens = list(getattr(sample, "tokens", None) or [])
    loss_mask = getattr(sample, "loss_mask", None)
    log_probs = getattr(sample, "rollout_log_probs", None)
    response_length = getattr(sample, "response_length", None)
    if loss_mask is None or log_probs is None or not tokens or not stop_ids:
        return 0
    if not isinstance(response_length, int) or response_length <= 0:
        return 0
    if len(loss_mask) != response_length or len(log_probs) != response_length:
        raise ValueError(
            f"loss_mask ({len(loss_mask)}) / rollout_log_probs ({len(log_probs)}) "
            f"do not match response_length ({response_length})"
        )
    offset = len(tokens) - response_length
    new_mask = list(loss_mask)
    changed = 0
    for index in range(response_length):
        if (
            int(new_mask[index]) == 1
            and tokens[offset + index] in stop_ids
            and float(log_probs[index]) == 0.0
        ):
            new_mask[index] = 0
            changed += 1
    if changed:
        sample.loss_mask = new_mask
    return changed


def _has_candidate(sample: Any) -> bool:
    loss_mask = getattr(sample, "loss_mask", None)
    log_probs = getattr(sample, "rollout_log_probs", None)
    if loss_mask is None or log_probs is None:
        return False
    return any(int(m) == 1 and float(lp) == 0.0 for m, lp in zip(loss_mask, log_probs))


def apply_to_groups(data: Iterable[Any], *, tokenizer_loader: Any, args: Any) -> int:
    """Mask placeholder stop tokens on the kept samples (in place); return the total.

    Per-sample counts are added to ``sample.metadata[COUNT_KEY]`` only when > 0.
    Idempotent: a second call changes nothing more.
    """
    total = 0

    def ids() -> frozenset[int]:
        # Loaded only when some sample has a candidate (no tokenizer load otherwise).
        cached = getattr(args, _STOP_IDS_ATTR, None)
        if cached is None:
            cached = stop_token_ids(tokenizer_loader(args))
            setattr(args, _STOP_IDS_ATTR, cached)
        return cached

    def walk(items: Iterable[Any]) -> None:
        nonlocal total
        for item in items:
            if isinstance(item, (list, tuple)):
                walk(item)
                continue
            if not _has_candidate(item):
                continue
            changed = mask_placeholder_in_sample(item, ids())
            if changed:
                metadata = item.metadata if isinstance(getattr(item, "metadata", None), dict) else {}
                metadata[COUNT_KEY] = int(metadata.get(COUNT_KEY, 0)) + changed
                item.metadata = metadata
            total += changed

    walk(data)
    return total
