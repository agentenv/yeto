"""Kimi K3.1 MXFP4 releases on the ``sglang-server`` adapter: tensor naming,
and the file-level sparse expert update (only the trained experts' files).

Naming (checked against ``sglang-miles-kimi-k31@35708f76``
``models/kimi_k3.py``): a release stores each routed expert as
``model.layers.<l>.block_sparse_moe.experts.<e>.w{1,2,3}.weight_packed`` and
``.weight_scale`` (compressed-tensors MXFP4; ``w1`` = gate, ``w3`` = up, ``w2``
= down). SGLang's ``KimiK3*.load_weights`` maps ``block_sparse_moe`` to ``mlp``,
renames ``weight_packed`` to ``weight`` itself (except on NPU), squeezes a 4-D
``weight_scale`` and routes ``experts.<e>.w<n>.`` through the fused expert
mapping, so neither the yeto rename patch (``renames_weight_packed``) nor a
custom loader is needed for a K3.1 release: :func:`native_packed_names` tells
the adapter's preflight to accept ``*.weight_packed`` for these architectures.

File-level sparse update. A partial-experts release (F2: 128 of 896 experts
trained) rewrites only the safetensors files holding the trained experts; the
others are byte-identical to the served release. ``update_weights_from_disk``
has no file filter (it re-reads the whole 500 GB checkpoint on every rank), so
:func:`expert_update_plan` reads the release's change manifest
(``release_delta.read_changes``: the files the mirror rebuilt) and the
safetensors index, and, when every changed file holds only expert tensors,
yields the tensors to send through ``POST /update_weights_from_tensor`` with
no ``load_format`` (SGLang then calls the model's own ``load_weights`` on the
``(name, tensor)`` pairs, the same path the disk load takes for naming). Any
other change (dense weights, the router, a file without an index entry) means
a full reload; nothing is sent.

Risk the plan cannot remove -- the post-load weight transform. SGLang's disk
load runs ``process_weights_after_loading`` after ``load_weights``; the tensor
update does not. On CUDA the MXFP4 Marlin method repacks the expert weights
(``prepare_moe_mxfp4_layer_for_marlin``, with an in-place reload path,
``_mxfp4_marlin_reload_in_place``); on ROCm with aiter the method shuffles
``w13_weight`` / ``w2_weight`` (``shuffle_weight`` / ``shuffle_weight_a16w4``,
``is_shuffled``) and replaces the scale parameters with shuffled copies, with
no reload guard in ``mxfp4.py`` at the pin. A raw tensor written into a packed
or shuffled parameter is silently wrong. The adapter therefore runs
:func:`shuffle_probe_plan` before the first tensor update: it rewrites one
served expert with its own bytes from disk and compares teacher-forced
log-probs before and after; any change turns the file-level path off for the
server and triggers a full disk reload. The same transform question applies to
``update_weights_from_disk`` on ROCm (a second ``shuffle_weight`` over
re-loaded raw bytes is right, the scale parameter replacement may not be); it
can only be settled on an MI355X, see the S1 report.

Dependencies: the plan reads JSON and file names only. Reading tensors needs
``safetensors`` and ``torch`` (:func:`read_tensors`), serializing them for
SGLang needs ``torch`` (:func:`serialize_named_tensors`); both are injected
into the adapter so CPU tests run without them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from yeto.rl.engine.sampler_checks import INDEX_FILE

# architectures whose SGLang model file renames weight_packed itself (kimi_k3.py)
NATIVE_PACKED_ARCHITECTURES = frozenset({"KimiK3ForConditionalGeneration", "KimiK31ForConditionalGeneration",
                                         "KimiK3LinearForCausalLM"})
# ``language_model.model.layers...``: the Kimi-K3 / K3.1 checkpoint names (the trainer's
# release keeps them); ``model.layers...`` / ``layers...``: text-only layouts
EXPERT_TENSOR = re.compile(r"^(?:language_model\.)?(?:model\.)?layers\.(?P<layer>\d+)\.block_sparse_moe\.experts\.(?P<expert>\d+)\."
                           r"(?P<proj>w[123])\.(?P<kind>weight_packed|weight_scale)$")
EXPERT_KINDS = ("weight_packed", "weight_scale")
DEFAULT_BATCH_BYTES = 2 * 1024 ** 3  # tensors per /update_weights_from_tensor call (base64 pickle over HTTP)


class FileSparseUnavailable(RuntimeError):
    """The release cannot be applied as a file-level expert update; ``reason`` is
    the short code of the load record, the message says why."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def release_architectures(model_dir: Path) -> frozenset[str]:
    try:
        config = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return frozenset()
    return frozenset(str(a) for a in (config.get("architectures") or ()))


def native_packed_names(model_dir: Path) -> bool:
    """The served model's SGLang code maps ``*.weight_packed`` itself (module doc)."""
    return bool(release_architectures(model_dir) & NATIVE_PACKED_ARCHITECTURES)


def weight_map(model_dir: Path) -> dict[str, str]:
    """``tensor name -> file`` of the release's safetensors index (empty without one)."""
    try:
        return dict(json.loads((Path(model_dir) / INDEX_FILE).read_text(encoding="utf-8")).get("weight_map") or {})
    except (OSError, ValueError, AttributeError):
        return {}


def parse_expert_tensor(name: str) -> tuple[int, int, str, str] | None:
    """``(layer, expert, proj, kind)`` of a K3.1 routed-expert tensor name, else None."""
    match = EXPERT_TENSOR.match(name)
    if match is None:
        return None
    return int(match["layer"]), int(match["expert"]), match["proj"], match["kind"]


@dataclass(frozen=True)
class ExpertUpdatePlan:
    """What a file-level expert update sends: the changed files and the expert
    tensors in them, grouped per file (one ``/update_weights_from_tensor`` batch
    never splits a file)."""

    release_dir: Path
    files: tuple[str, ...]
    tensors: Mapping[str, tuple[str, ...]]  # file -> tensor names (sorted)
    experts: Mapping[int, tuple[int, ...]] = field(default_factory=dict)  # layer -> expert ids

    @property
    def tensor_count(self) -> int:
        return sum(len(names) for names in self.tensors.values())

    @property
    def expert_count(self) -> int:
        return sum(len(ids) for ids in self.experts.values())


def expert_update_plan(release_dir: Path, changed_files: Iterable[str], *,
                       index: Mapping[str, str] | None = None) -> ExpertUpdatePlan:
    """The plan for ``changed_files`` of ``release_dir`` (the change manifest's
    files), or :class:`FileSparseUnavailable` when a changed file is not a
    safetensors weight file made only of routed-expert tensors, or the index is
    missing / inconsistent."""
    release_dir = Path(release_dir)
    index = weight_map(release_dir) if index is None else dict(index)
    if not index:
        raise FileSparseUnavailable("no_index", f"{release_dir / INDEX_FILE} is missing or has no weight_map")
    changed = sorted(set(changed_files))
    if not changed:
        raise FileSparseUnavailable("no_changes", "the change manifest lists no file")
    by_file: dict[str, list[str]] = {name: [] for name in changed}
    for tensor, file in index.items():
        if file in by_file:
            by_file[file].append(tensor)
    tensors: dict[str, tuple[str, ...]] = {}
    experts: dict[int, set[int]] = {}
    for file in changed:
        if not file.endswith(".safetensors") or "/" in file:
            raise FileSparseUnavailable("non_weight_file_changed", file)
        names = by_file[file]
        if not names:
            raise FileSparseUnavailable("file_not_in_index", file)
        if not (release_dir / file).is_file():
            raise FileSparseUnavailable("file_missing", file)
        for name in names:
            parsed = parse_expert_tensor(name)
            if parsed is None:
                raise FileSparseUnavailable("non_expert_tensor_changed", f"{name} in {file}")
            experts.setdefault(parsed[0], set()).add(parsed[1])
        tensors[file] = tuple(sorted(names))
    return ExpertUpdatePlan(release_dir=release_dir, files=tuple(changed), tensors=tensors,
                            experts={layer: tuple(sorted(ids)) for layer, ids in sorted(experts.items())})


RELEASE_PREFIX = "release-"  # yeto.rl.engine.release_overlay: a partial release's files in an overlay


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.resolve(strict=True) == b.resolve(strict=True)
    except OSError:
        return False


def overlay_update_plan(model_dir: Path, served_dir: Path | None, base_dir: Path | None) -> ExpertUpdatePlan | None:
    """The plan for an overlay of a partial-experts release (``release_overlay``:
    base files plus ``release-*`` files holding only the trained experts) served
    over ``served_dir``; None when ``model_dir`` is not such an overlay.

    The release files are applied as they are: exact when the served weights are
    the overlay's base, or an overlay of the same base with the same released
    tensor names (the same expert selection: every expert outside it is the
    base's in both). Anything else is :class:`FileSparseUnavailable`."""
    model_dir = Path(model_dir)
    index = weight_map(model_dir)
    released = {name for name, file in index.items() if file.startswith(RELEASE_PREFIX)}
    if not released:
        return None
    if served_dir is None or base_dir is None:
        raise FileSparseUnavailable("overlay_no_base", f"serving {served_dir}, base {base_dir}")
    base = Path(base_dir).resolve()
    base_files = {file for name, file in index.items() if name not in released}
    # file by file: the base may itself be a view of symlinks (a Modal Volume HF view),
    # so its files resolve outside ``base``
    if any(not _same_file(model_dir / file, base / file) for file in sorted(base_files)[:4]):
        raise FileSparseUnavailable("overlay_base_mismatch", f"{model_dir} is not laid over {base}")
    served = Path(served_dir)
    if served.resolve() != base:
        served_index = weight_map(served)
        served_released = {name for name, file in served_index.items() if file.startswith(RELEASE_PREFIX)}
        if served_released != released:
            raise FileSparseUnavailable("overlay_selection_changed",
                                        f"{len(served_released)} released tensors served, {len(released)} new")
        some_base_file = next((f for n, f in sorted(served_index.items()) if n not in served_released), None)
        if some_base_file is None or not _same_file(served / some_base_file, base / some_base_file):
            raise FileSparseUnavailable("overlay_base_mismatch", f"{served} is not laid over {base}")
    files = sorted({index[name] for name in released})
    return expert_update_plan(model_dir, files, index=index)


def changed_files_of(release_dir: Path) -> list[str] | None:
    """The files the release's change manifest names (``release_delta.read_changes``),
    None when the release has no manifest (a full release)."""
    from yeto.rl.engine.release_delta import read_changes

    manifest = read_changes(Path(release_dir))
    if manifest is None:
        return None
    return sorted((manifest.get("files") or {}).keys())


def shuffle_probe_plan(served_dir: Path, *, index: Mapping[str, str] | None = None) -> ExpertUpdatePlan:
    """One served expert (the lowest layer and id in the index, both tensor
    kinds of every projection) as a plan: writing it back with its own bytes
    must not change the model (module doc)."""
    served_dir = Path(served_dir)
    index = weight_map(served_dir) if index is None else dict(index)
    experts = sorted((parsed[0], parsed[1], name, file) for name, file in index.items()
                     if (parsed := parse_expert_tensor(name)) is not None)
    if not experts:
        raise FileSparseUnavailable("no_expert_tensors", f"{served_dir}: the index names no K3.1 expert tensor")
    layer, expert = experts[0][0], experts[0][1]
    names = [(name, file) for l, e, name, file in experts if l == layer and e == expert]
    tensors: dict[str, list[str]] = {}
    for name, file in names:
        tensors.setdefault(file, []).append(name)
    return ExpertUpdatePlan(release_dir=served_dir, files=tuple(sorted(tensors)),
                            tensors={f: tuple(sorted(n)) for f, n in tensors.items()}, experts={layer: (expert,)})


# ------------------------------------------------------------------ tensors
ReadTensors = Callable[[Path, Sequence[str]], list[tuple[str, Any]]]
Serialize = Callable[[list[tuple[str, Any]]], str]


def read_tensors(path: Path, names: Sequence[str]) -> list[tuple[str, Any]]:
    """``(name, torch tensor)`` of ``names`` from one safetensors file (CPU)."""
    from safetensors import safe_open  # noqa: PLC0415 - GPU-image dependency

    out = []
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for name in names:
            out.append((name, handle.get_tensor(name)))
    return out


def serialize_named_tensors(named: list[tuple[str, Any]]) -> str:
    """One rank's ``serialized_named_tensors`` entry: the pairs pickled (plain
    pickle of CPU tensors, which SGLang's restricted unpickler accepts) and
    base64 for JSON. Every rank gets the same bytes; each rank's ``load_weights``
    takes its own shard."""
    import base64  # noqa: PLC0415
    import pickle  # noqa: PLC0415

    return base64.b64encode(pickle.dumps(named, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")


def tensor_bytes(tensor: Any) -> int:
    try:
        return int(tensor.numel()) * int(tensor.element_size())
    except AttributeError:
        return int(getattr(tensor, "nbytes", 0))


def batches(plan: ExpertUpdatePlan, read: ReadTensors = read_tensors, *,
            max_bytes: int = DEFAULT_BATCH_BYTES) -> Iterator[list[tuple[str, Any]]]:
    """The plan's tensors in batches of at most ``max_bytes`` (a file's tensors
    may span batches; a tensor never splits)."""
    batch: list[tuple[str, Any]] = []
    size = 0
    for file in plan.files:
        for name, tensor in read(plan.release_dir / file, plan.tensors[file]):
            nbytes = tensor_bytes(tensor)
            if batch and size + nbytes > max_bytes:
                yield batch
                batch, size = [], 0
            batch.append((name, tensor))
            size += nbytes
    if batch:
        yield batch
