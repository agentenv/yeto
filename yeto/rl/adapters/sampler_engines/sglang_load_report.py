"""Expert load report and ``weight_packed`` rename inside an SGLang server process
(S25 N2.3 post-load check 1; A-INF section 2.3 option A, item 1).

SGLang's GLM5.3 ``load_weights`` maps ``experts.<e>.gate_proj.weight`` to the
fused ``experts.w13_weight`` parameter. A compressed-tensors MXFP4 release
stores that matrix as ``...gate_proj.weight_packed``; the mapped name
``experts.w13_weight_packed`` is not a parameter, and SGLang skips the tensor
without an error. :func:`install` wraps the model class's ``load_weights``:

* with ``YETO_SGLANG_RENAME_WEIGHT_PACKED=1`` it renames routed-expert
  ``*.weight_packed`` tensors to ``*.weight`` before SGLang maps them (the
  MXFP4 MoE method stores the packed bytes in ``w13_weight`` / ``w2_weight``);
* with ``YETO_SGLANG_LOAD_REPORT_DIR`` set it counts the calls of the routed
  experts' data-weight loaders (one call = one gate, up or down matrix of one
  expert on this rank) and writes ``rank<tp>-<pid>.json`` into that directory
  after each load. A skipped tensor never reaches a loader, so the count is
  what the engine holds, not what the checkpoint names.

The image applies :data:`MODEL_FILE_HOOK` to SGLang's model file
(:func:`patch_model_file`); the server process imports this module from the
yeto tree (``PYTHONPATH``). :func:`read_reports` is the adapter's side
(``SGLangServerEngine.loaded_expert_matrices``).

This module must not import SGLang at import time (CPU tests use a fake model).
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

REPORT_DIR_ENV = "YETO_SGLANG_LOAD_REPORT_DIR"
RENAME_ENV = "YETO_SGLANG_RENAME_WEIGHT_PACKED"
ROUTED_PACKED = re.compile(r"(\.mlp\.experts\.\d+\.(?:gate|up|down)_proj)\.weight_packed$")
# FusedMoE data parameters (not the scales); the routed experts' loaders.
EXPERT_DATA_PARAMS = re.compile(r"\.experts\.(?:w13_weight|w2_weight)$")
MARKER = "# yeto: sglang_load_report hook"
MODEL_FILE_HOOK = f"""

{MARKER}
import os as _yeto_os
if _yeto_os.environ.get("{REPORT_DIR_ENV}") or _yeto_os.environ.get("{RENAME_ENV}") == "1":
    from yeto.rl.adapters.sampler_engines.sglang_load_report import install as _yeto_install
    for _yeto_cls in EntryClass:
        _yeto_install(_yeto_cls)
"""


def rename_packed(weights: Iterable[tuple[str, Any]]) -> Iterator[tuple[str, Any]]:
    """Routed-expert ``*.weight_packed`` -> ``*.weight``; every other name unchanged."""
    for name, tensor in weights:
        yield ROUTED_PACKED.sub(r"\1.weight", name), tensor


def _tp_rank() -> int:
    try:
        from sglang.srt.distributed import get_tensor_model_parallel_rank  # noqa: PLC0415
        return int(get_tensor_model_parallel_rank())
    except Exception:  # noqa: BLE001 - outside a parallel group (tests)
        return 0


def install(model_cls: type) -> type:
    """Wrap ``model_cls.load_weights`` once (module doc)."""
    original = model_cls.load_weights
    if getattr(original, "_yeto_load_report", False):
        return model_cls

    def load_weights(self: Any, weights: Iterable[tuple[str, Any]], *args: Any, **kwargs: Any) -> Any:
        if os.environ.get(RENAME_ENV) == "1":
            weights = rename_packed(weights)
        report_dir = os.environ.get(REPORT_DIR_ENV)
        if not report_dir:
            return original(self, weights, *args, **kwargs)
        calls = {"n": 0}
        wrapped = []
        for name, param in self.named_parameters():
            loader = getattr(param, "weight_loader", None)
            if EXPERT_DATA_PARAMS.search(name) and loader is not None:
                def counting(*a: Any, _loader: Any = loader, **kw: Any) -> Any:
                    calls["n"] += 1
                    return _loader(*a, **kw)
                param.weight_loader = counting
                wrapped.append((param, loader))
        try:
            result = original(self, weights, *args, **kwargs)
        finally:
            for param, loader in wrapped:
                param.weight_loader = loader
        rank = _tp_rank()
        path = Path(report_dir) / f"rank{rank}-{os.getpid()}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"tp_rank": rank, "pid": os.getpid(), "expert_matrices": calls["n"],
                                   "expert_params": len(wrapped), "time": time.time()}))
        os.replace(tmp, path)
        return result

    load_weights._yeto_load_report = True  # type: ignore[attr-defined]
    model_cls.load_weights = load_weights
    return model_cls


def patch_model_file(path: str | os.PathLike[str]) -> bool:
    """Append :data:`MODEL_FILE_HOOK` to an SGLang model file that defines
    ``EntryClass`` (idempotent; False when already patched)."""
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    if MARKER in text:
        return False
    if "EntryClass" not in text:
        raise ValueError(f"{file} defines no EntryClass")
    file.write_text(text + MODEL_FILE_HOOK, encoding="utf-8")
    return True


def clear_reports(report_dir: str | os.PathLike[str]) -> None:
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for f in directory.glob("rank*.json"):
        f.unlink()


def read_reports(report_dir: str | os.PathLike[str]) -> dict[str, Any] | None:
    """The reports of the current server (None: no rank reported). ``expert_matrices``
    is the smallest rank count, so one rank that skipped experts fails the check."""
    reports = []
    for f in sorted(Path(report_dir).glob("rank*.json")):
        try:
            reports.append(json.loads(f.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    if not reports:
        return None
    counts = [int(r["expert_matrices"]) for r in reports]
    return {"expert_matrices": min(counts), "per_rank": counts, "ranks": len(reports)}
