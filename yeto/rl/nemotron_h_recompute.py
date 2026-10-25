"""Activation recompute for transformers' NemotronH blocks (S27 G-EQ4, npu1).

Why this exists. A full-parameter RL step of Nemotron-3 Nano 30B-A3B did not fit
on a 910B3 (64 GiB, 60.96 GiB usable): the backward asked for 8.5-10 GiB on top
of **54-55 GiB of live tensors** and ran out of memory, so an island only ever
completed one optimizer step (S27 G-EQ3, four attempts, VERL-ASCEND-RL.md section 31).
The live tensors are mostly stored activations, because this island trains with
recompute **off** -- and it does so for a reason:

* ``transformers`` 5.3.0.dev0 ``NemotronHModel`` sets
  ``supports_gradient_checkpointing = False`` and its source mentions
  ``gradient_checkpointing`` **zero** times: it has no recompute hook at all, so
  verl's ``enable_gradient_checkpointing`` cannot help and flipping the class
  attribute would recompute nothing (S27 G-EQ4, read in the island's own image).
  yeto therefore turns the verl flag off for this model
  (``island_entry.gradient_checkpointing_for``).
* the checkpoint's own ``modeling_nemotron_h.py`` *does* support it
  (``supports_gradient_checkpointing = True``, the layer loop calls
  ``self._gradient_checkpointing_func``), but loading it (``--trust-remote-code``)
  trades one problem for three: its ``NemotronHBlock.forward`` wraps every block
  in ``torch.cuda.stream(torch.cuda.default_stream(...))``, which on Ascend raises
  ``AssertionError: Torch not compiled with CUDA enabled`` (measured), it is a
  different implementation than the one whose log-prob gap and export layout are
  already verified, and it loads the weights without transformers' key conversion,
  which the full-parameter export's rename table is written against.

So this module adds the missing hook to the implementation already in use, in the
way that costs the least elsewhere: it wraps ``NemotronHBlock.forward`` so a
training forward runs under ``torch.utils.checkpoint.checkpoint`` with
``use_reentrant=False``. No wrapper **module** is inserted, so the module tree and
every ``state_dict`` key stay exactly as they were -- the full-parameter export,
its rename table and the delta releases are untouched. The Mamba2 scan fix
(:mod:`yeto.rl.nemotron_h_torch_fix`) stays in force; this only changes *when*
the block's internals are materialised.

Scope. :func:`apply` is a no-op unless this node's device family is ``ascend``
(``yeto.hw.catalog.device_family``), because that is the card the 64 GiB ceiling
belongs to; an NVIDIA island keeps storing activations, as before. The env var
``YETO_NEMOTRONH_RECOMPUTE`` forces the decision either way (``1`` / ``0``), which
is what an A/B of the memory saving uses.

Safety. A forward that fills a cache is never checkpointed (``use_cache`` true, or
a cache object passed in): recomputing it would append to that cache twice. The
recompute keeps the RNG state (``preserve_rng_state=True``, the default), so a
dropout-free forward recomputes bit-identically.
"""

from __future__ import annotations

import functools
import os
from typing import Any

MODULE = "transformers.models.nemotron_h.modeling_nemotron_h"
CLASS = "NemotronHBlock"
MARK = "_yeto_nemotron_h_recompute"
ENV = "YETO_NEMOTRONH_RECOMPUTE"


def wanted(*, family: str | None = None) -> bool:
    """Whether this node recomputes NemotronH blocks: the env var when it is set
    to 0/1, else true only on an Ascend node (module doc)."""
    forced = os.environ.get(ENV)
    if forced is not None and forced.strip() != "":
        return forced.strip() not in ("0", "false", "False", "no")
    if family is None:
        try:
            from yeto.hw.catalog import device_family  # noqa: PLC0415

            family = device_family()
        except Exception:  # noqa: BLE001 - a build without yeto.hw, or no torch
            return False
    return family == "ascend"


def _checkpointing_forward(original: Any) -> Any:
    import torch

    @functools.wraps(original)
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if (not self.training) or (not torch.is_grad_enabled()):
            return original(self, *args, **kwargs)
        # a cache-filling forward must not run twice (module doc)
        if kwargs.get("use_cache") or kwargs.get("past_key_values") is not None:
            return original(self, *args, **kwargs)
        return torch.utils.checkpoint.checkpoint(
            original, self, *args, use_reentrant=False, **kwargs)

    return forward


def apply(*, module: Any = None, enabled: bool | None = None) -> dict[str, Any]:
    """Install the recomputing ``forward`` on transformers' ``NemotronHBlock``.

    Returns what was done: ``{"enabled": False, ...}`` when this node does not want
    it, ``{"already": True, ...}`` on a second call. Never raises for a build
    without the model family -- the caller prints the result and goes on.
    """
    if enabled is None:
        enabled = wanted()
    if not enabled:
        return {"enabled": False, "reason": "not an Ascend node" if os.environ.get(ENV) is None
                else f"{ENV}={os.environ.get(ENV)!r}"}
    if module is None:
        import importlib  # noqa: PLC0415

        module = importlib.import_module(MODULE)
    cls = getattr(module, CLASS)
    current = cls.forward
    if getattr(current, MARK, None) is not None:
        return {"already": True, "enabled": True, **getattr(current, MARK)}
    forward = _checkpointing_forward(current)
    info = {"module": module.__name__, "cls": CLASS, "use_reentrant": False}
    setattr(forward, MARK, info)
    forward.__wrapped_original__ = current
    cls.forward = forward
    return {"already": False, "enabled": True, **info}
