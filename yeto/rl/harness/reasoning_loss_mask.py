"""Optional "reasoning tokens do not count toward the loss" switch (S17 WP6).

Default: reasoning tokens DO count (``reasoning_in_loss=True``), i.e. this
module is a no-op unless the run explicitly opts out.

When opted out, every generated token (``loss_mask == 1``) that lies inside a
reasoning block -- between the profile's open marker (e.g. ``<think>``) and
close marker (``</think>``) -- gets ``loss_mask = 0`` on the kept samples, in
the rollout process, from the ports path's ``--rollout-sample-filter-path`` hook
(``rollout_meta_hook.record_trained_groups``).  Miles calls that hook in both
rollout implementations (``sglang_rollout.py``, ``inference_rollout_train.py``)
before returning the samples, i.e. before they become train data; placement
(colocated or disaggregated) does not change that code path.  Only the ports
engine installs the hook, so the learner refuses the switch on legacy.  The
upstream generate function is fixed on the ports path
(``miles_adapter/config.py:243``), so ``codex_openenv_generate`` is not a
usable hook point.  Rules (written into the codex review doc):

- the open marker itself is masked when it was generated;
- the close marker is kept (mask unchanged): it is the decision to stop
  thinking and start acting, which the reward is about;
- the block state is tracked over the WHOLE token sequence, so a generation
  that starts inside a block opened by the chat template's generation prompt
  (Qwen ``<|im_start|>assistant\\n<think>\\n``) is handled, and so is every
  later turn of a merged multi-turn sample;
- a block that never closes (truncated generation) masks to the end;
- tokens, logprobs and lengths are never changed, only ``loss_mask`` entries
  1 -> 0, so the alignment contract (``codex/alignment.py``: mask=1 is a
  subset of generated spans) still holds.

Neutral on purpose (no Miles import): the same function applies to any
backend whose sample exposes ``tokens`` / ``response_length`` / ``loss_mask``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

MASKED_COUNT_KEY = "reasoning_tokens_masked"
IN_LOSS_KEY = "reasoning_in_loss"


class ReasoningMarkerError(ValueError):
    """The profile's reasoning markers do not resolve to single tokenizer ids."""


@dataclass(frozen=True)
class ReasoningMarkers:
    open_id: int
    close_id: int

    @classmethod
    def from_tokenizer(cls, tokenizer: Any, open_text: str, close_text: str) -> "ReasoningMarkers":
        ids = []
        for text in (open_text, close_text):
            token_id = tokenizer.convert_tokens_to_ids(text)
            unk = getattr(tokenizer, "unk_token_id", None)
            if not isinstance(token_id, int) or token_id < 0 or (unk is not None and token_id == unk):
                raise ReasoningMarkerError(f"reasoning marker {text!r} is not a single token of this tokenizer")
            ids.append(token_id)
        if ids[0] == ids[1]:
            raise ReasoningMarkerError("reasoning open and close markers resolve to the same id")
        return cls(open_id=ids[0], close_id=ids[1])


def reasoning_positions(tokens: list[int], markers: ReasoningMarkers) -> list[bool]:
    """``True`` at positions that are reasoning content (open marker included, close marker excluded)."""
    inside = False
    out: list[bool] = []
    for token in tokens:
        if token == markers.open_id:
            inside = True
            out.append(True)
        elif token == markers.close_id:
            inside = False
            out.append(False)
        else:
            out.append(inside)
    return out


def mask_reasoning_in_sample(sample: Any, markers: ReasoningMarkers) -> int:
    """Set ``loss_mask`` to 0 on generated reasoning tokens; return how many were changed."""
    tokens = list(getattr(sample, "tokens", None) or [])
    loss_mask = getattr(sample, "loss_mask", None)
    response_length = getattr(sample, "response_length", None)
    if loss_mask is None or not tokens:
        return 0
    if len(loss_mask) == len(tokens):
        offset = 0
    elif isinstance(response_length, int) and len(loss_mask) == response_length:
        offset = len(tokens) - response_length
    else:
        raise ValueError("loss_mask length matches neither tokens nor response_length")
    flags = reasoning_positions(tokens, markers)
    new_mask = list(loss_mask)
    changed = 0
    for index, value in enumerate(new_mask):
        if int(value) == 1 and flags[offset + index]:
            new_mask[index] = 0
            changed += 1
    sample.loss_mask = new_mask
    return changed


# --- rollout-process entry (ports path: ``--rollout-sample-filter-path``) ----
IN_LOSS_ATTR = "yeto_rl_codex_reasoning_in_loss"  # absent / True == counted (default)
MARKERS_ATTR = "yeto_rl_codex_reasoning_markers"  # [open, close] token strings
_MARKER_CACHE_ATTR = "_yeto_reasoning_marker_ids"


def _load_tokenizer(args: Any) -> Any:
    from miles.utils.processing_utils import load_tokenizer  # cached by Miles

    return load_tokenizer(args.hf_checkpoint, trust_remote_code=True)


def apply_from_args(args: Any, data: Iterable[Any], *, tokenizer_loader: Any = None) -> int | None:
    """Apply the run's reasoning-loss policy to the kept groups (in place).

    Returns ``None`` and touches nothing when the policy is the default
    (``args`` has no ``yeto_rl_codex_reasoning_in_loss`` or it is truthy);
    otherwise the number of mask entries changed 1 -> 0 this call.
    Idempotent: a second call over the same samples changes nothing more.
    """
    if getattr(args, IN_LOSS_ATTR, True) is not False:
        return None
    markers = getattr(args, _MARKER_CACHE_ATTR, None)
    if markers is None:
        names = getattr(args, MARKERS_ATTR, None)
        if not names or len(names) != 2:
            raise ReasoningMarkerError(
                f"args.{IN_LOSS_ATTR}=False but args.{MARKERS_ATTR} is missing; refusing to train "
                "with reasoning silently counted"
            )
        tokenizer = (tokenizer_loader or _load_tokenizer)(args)
        markers = ReasoningMarkers.from_tokenizer(tokenizer, names[0], names[1])
        setattr(args, _MARKER_CACHE_ATTR, markers)
    total = 0

    def walk(items: Iterable[Any]) -> None:
        nonlocal total
        for item in items:
            if isinstance(item, (list, tuple)):
                walk(item)
                continue
            metadata = item.metadata if isinstance(getattr(item, "metadata", None), dict) else {}
            changed = mask_reasoning_in_sample(item, markers)
            metadata[IN_LOSS_KEY] = False
            metadata[MASKED_COUNT_KEY] = int(metadata.get(MASKED_COUNT_KEY, 0)) + changed
            item.metadata = metadata
            total += changed

    walk(data)
    return total
