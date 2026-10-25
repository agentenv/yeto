"""Fix of transformers' NemotronH Mamba2 torch path (no mamba kernels: Ascend NPU, CPU).

Finding (S27 G-ES, XCLOUD-ROLLOUT.md "Mamba2 naive path"): transformers 5.12.1 / 5.13.0
``NemotronHMamba2Mixer.torch_forward`` computes the inter-chunk SSM states as::

    decay_chunk = torch.exp(segment_sum(nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0))))
    states_permuted = states.permute(0, 2, 1, 3, 4)
    result = (decay_chunk[..., None, None] * states_permuted[:, :, None, ...]).sum(dim=2)

``decay_chunk[..., z, c]`` is the decay from source chunk ``c`` to target chunk ``z``
(rows of ``segment_sum`` are targets), so the new state of chunk ``z`` is the sum over
``c`` -- dimension 3 of the product, not dimension 2 (the reference
``einsum("bhzc,bchpn->bzhpn")``; transformers' own Mamba2 transposes ``decay_chunk``
and sums dimension 1, which is the same). With one chunk the two agree, so short
inputs pass; a sequence longer than ``chunk_size`` (Nano: 128) gets wrong states. On
CUDA with the kernels turned off this put the response log-probs 0.44 nats per token
away from SGLang (kernel path: 0.019), the same gap the Ascend trainer showed.

Second difference, smaller (G-EQ3: about 0.016): the torch path clamps ``dt`` below at
``time_step_min``; the kernel path (and the checkpoint's own code) clamps with
``time_step_limit`` (Nano: (0, inf), i.e. no clamp). ``align_dt_clamp`` makes the
torch path use ``time_step_limit`` like the kernel path.

:func:`apply` rewrites the method's source at the anchors below (each must occur exactly
once, else it raises and changes nothing) and installs the result on the class. It is
backend-neutral (any trainer that builds the model with transformers' built-in
``nemotron_h``); idempotent; it does not touch the CUDA kernel path or the
checkpoint's own ``trust_remote_code`` implementation (that one is already correct).
"""

from __future__ import annotations

import inspect
import textwrap
from typing import Any

MODULE = "transformers.models.nemotron_h.modeling_nemotron_h"
CLASS = "NemotronHMamba2Mixer"
MARK = "_yeto_nemotron_h_torch_fix"

SCAN_ANCHOR = "result = (decay_chunk[..., None, None] * states_permuted[:, :, None, ...]).sum(dim=2)"
SCAN_FIX = ("result = (decay_chunk[..., None, None] * states_permuted[:, :, None, ...]).sum(dim=3)"
            "  # yeto: sum over source chunks")
# the chunked (prefill / training) branch and the one-token (decode) branch
CLAMP_ANCHORS = ("dt = torch.clamp(dt, self.time_step_min) #, self.time_step_max)",
                 "dt = torch.clamp(dt, self.time_step_min)\n")
CLAMP_FIX = ("dt = (dt if self.time_step_limit is None else "
             "torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1]))  # yeto: as the kernel path")


class NemotronHTorchFixError(RuntimeError):
    pass


def _replace_once(source: str, anchor: str, new: str) -> str:
    count = source.count(anchor)
    if count != 1:
        raise NemotronHTorchFixError(f"anchor found {count} times (need exactly 1): {anchor.strip()!r}")
    return source.replace(anchor, new)


def patched_source(source: str, *, align_dt_clamp: bool = True) -> str:
    """The fixed source of ``torch_forward`` (module doc); raises on a missing anchor."""
    out = _replace_once(source, SCAN_ANCHOR, SCAN_FIX)
    if align_dt_clamp:
        out = _replace_once(out, CLAMP_ANCHORS[0], CLAMP_FIX)
        # the second anchor ends with the newline so it does not match the first one's prefix
        out = _replace_once(out, CLAMP_ANCHORS[1], CLAMP_FIX + "\n")
    return out


def apply(*, align_dt_clamp: bool = True, module: Any = None) -> dict[str, Any]:
    """Install the fixed ``torch_forward`` on transformers' ``NemotronHMamba2Mixer``.
    Returns what was done; a second call returns ``{"already": True, ...}``."""
    if module is None:
        import importlib

        module = importlib.import_module(MODULE)
    cls = getattr(module, CLASS)
    current = cls.torch_forward
    if getattr(current, MARK, None) is not None:
        return {"already": True, **getattr(current, MARK)}
    source = textwrap.dedent(inspect.getsource(current))
    fixed = patched_source(source, align_dt_clamp=align_dt_clamp)
    namespace: dict[str, Any] = {}
    exec(compile(fixed, f"<yeto nemotron_h_torch_fix of {module.__name__}>", "exec"),  # noqa: S102
         module.__dict__, namespace)
    function = namespace["torch_forward"]
    info = {"module": module.__name__, "scan_fix": True, "align_dt_clamp": bool(align_dt_clamp)}
    setattr(function, MARK, info)
    function.__wrapped_original__ = current
    cls.torch_forward = function
    return {"already": False, **info}
