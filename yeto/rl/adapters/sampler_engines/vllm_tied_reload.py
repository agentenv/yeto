"""In-place reload of a model whose config ties the word embeddings
(``tie_word_embeddings``, e.g. Qwen3-0.6B), on every line of ``vllm-server``
(cuda / rocm / npu): which path a release takes, the worker extension that
syncs and checks the served ``lm_head``, and the check itself.

How vLLM serves such a model (vLLM 0.29.0 under vllm-ascend 0.29.0rc2, and the
same code in the CUDA pin 266e88c05 and main 083fbac02:
``model_loader/weight_tying.py``, ``config/model.py::maybe_untie_word_embeddings``,
``model_loader/reload/layerwise.py``), measured on a 910B4:

* a checkpoint without ``lm_head.weight``, or whose ``lm_head.weight`` equals
  ``embed_tokens.weight``: ``lm_head.weight`` IS ``embed_tokens.weight`` (one
  Parameter, one storage; "shared");
* a checkpoint whose ``lm_head.weight`` differs from ``embed_tokens.weight``:
  vLLM unties, and serves the two as separate tensors ("split").

That layout is fixed at the cold start. ``reload_weights`` copies every loaded
layer into the served storage (``_copy_and_restore_kernel_tensors``) and never
re-decides it, so a reload is wrong in exactly two cases, silently:

* shared server, release with a ``lm_head`` that differs from its embeddings: both
  are written into the one storage, the last write wins, so the head serves the
  embeddings while a restart serves the release's own head (A11's 0.083 nats:
  its probe release scaled ``embed_tokens`` only and kept the old ``lm_head``);
* split server, release without ``lm_head.weight``: the head layer loads nothing
  and keeps the previous release's head (``finalize_layerwise_processing`` puts
  the old tensor back).

What :func:`plan` does with that: the first case cannot be served in place (the
one storage cannot hold two tensors) and restarts; the second reloads and then
copies the served embeddings into the head (:meth:`TiedEmbeddingExtension.
yeto_tied_sync`, the "full update"); every other release reloads in place as
it is. After every in-place reload of such a server :func:`verify` compares
sampled rows of the served ``embed_tokens`` and ``lm_head`` (all TP ranks) with
the rows a cold start on the release would serve, bit for bit, and fails the
load on any difference.

This module must not import torch / vLLM at import time (CPU tests).
"""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXTENSION = "yeto.rl.adapters.sampler_engines.vllm_tied_reload.TiedEmbeddingExtension"
STATE_METHOD = "yeto_tied_state"
SYNC_METHOD = "yeto_tied_sync"

SHARED, SPLIT = "shared", "split"
ABSENT, EQUAL, DISTINCT = "absent", "equal", "distinct"
IN_PLACE, IN_PLACE_SYNC, RESTART = "in_place", "in_place+sync_lm_head", "restart"
SAMPLE_ROWS = 8
_DTYPES = {"BF16": (2, "bf16"), "F16": (2, "e"), "F32": (4, "f")}
_CHUNK = 64 << 20


class TiedReloadError(RuntimeError):
    """The served word embeddings are not the ones a cold start would serve."""


def config_ties(model_dir: Path) -> bool:
    """``tie_word_embeddings`` of ``model_dir/config.json`` (``text_config`` when
    nested; False when the config does not say or cannot be read)."""
    try:
        config = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    return bool(text.get("tie_word_embeddings"))


def _header(path: Path) -> tuple[dict[str, Any], int]:
    with open(path, "rb") as handle:
        (n,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


def _tensor_files(model_dir: Path) -> dict[str, Path]:
    """Tensor name -> safetensors file of the release."""
    root = Path(model_dir)
    index = root / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map") or {}
        return {name: root / file for name, file in weight_map.items()}
    out: dict[str, Path] = {}
    for path in sorted(root.glob("*.safetensors")):
        out.update(dict.fromkeys(_header(path)[0], path))
    return out


@dataclass(frozen=True)
class _Tensor:
    path: Path
    begin: int  # absolute byte offset of the data
    end: int
    dtype: str
    shape: tuple[int, ...]

    @classmethod
    def find(cls, files: Mapping[str, Path], name: str) -> _Tensor:
        header, base = _header(files[name])
        entry = header[name]
        start, end = entry["data_offsets"]
        return cls(files[name], base + int(start), base + int(end), str(entry["dtype"]),
                   tuple(int(s) for s in entry["shape"]))

    def rows(self, rows: Sequence[int]) -> list[list[float]]:
        """``rows`` of this 2-D tensor as floats (exact for BF16 / F16 / F32)."""
        if self.dtype not in _DTYPES or len(self.shape) != 2:
            raise TiedReloadError(f"{self.path}: cannot read rows of a {self.dtype} {self.shape} tensor")
        size, code = _DTYPES[self.dtype]
        width = self.shape[1]
        out = []
        with open(self.path, "rb") as handle:
            for row in rows:
                handle.seek(self.begin + int(row) * width * size)
                raw = handle.read(width * size)
                if code == "bf16":
                    halves = struct.unpack(f"<{width}H", raw)
                    out.append(list(struct.unpack(f"<{width}f", struct.pack(f"<{width}I",
                                                                            *(h << 16 for h in halves)))))
                else:
                    out.append(list(struct.unpack(f"<{width}{code}", raw)))
        return out


def _same_bytes(a: _Tensor, b: _Tensor) -> bool:
    if (a.dtype, a.shape) != (b.dtype, b.shape):
        return False
    with open(a.path, "rb") as fa, open(b.path, "rb") as fb:
        fa.seek(a.begin)
        fb.seek(b.begin)
        left = a.end - a.begin
        while left > 0:
            n = min(left, _CHUNK)
            if fa.read(n) != fb.read(n):
                return False
            left -= n
    return True


@dataclass(frozen=True)
class ReleaseTie:
    """The word embeddings of a release as a cold start of vLLM would serve them."""

    config_tied: bool
    lm_head: str | None  # ABSENT / EQUAL / DISTINCT; None: not one embed / one head tensor
    embed_name: str | None = None
    head_name: str | None = None

    @property
    def cold_layout(self) -> str | None:
        """SHARED / SPLIT after a cold start on this release (None: unknown)."""
        if self.lm_head is None:
            return None
        if not self.config_tied:
            return SPLIT if self.lm_head != ABSENT else None
        return SPLIT if self.lm_head == DISTINCT else SHARED


def release_tie(model_dir: Path) -> ReleaseTie:
    """Read the release's config and safetensors (only the embeddings are compared,
    byte for byte, when the release carries an ``lm_head``)."""
    tied = config_ties(model_dir)
    try:
        files = _tensor_files(model_dir)
    except (OSError, ValueError, KeyError, struct.error):
        return ReleaseTie(tied, None)
    embeds = [n for n in files if n.endswith("embed_tokens.weight")]
    heads = [n for n in files if n.endswith("lm_head.weight")]
    if len(embeds) != 1 or len(heads) > 1:
        return ReleaseTie(tied, None)
    if not heads:
        return ReleaseTie(tied, ABSENT, embeds[0], None)
    same = _same_bytes(_Tensor.find(files, embeds[0]), _Tensor.find(files, heads[0]))
    return ReleaseTie(tied, EQUAL if same else DISTINCT, embeds[0], heads[0])


def plan(served_layout: str | None, release: ReleaseTie) -> tuple[str, str]:
    """``(path, why)`` of an in-place load of ``release`` into a server whose
    word embeddings are ``served_layout`` (module doc)."""
    if served_layout not in (SHARED, SPLIT) or release.cold_layout is None:
        return RESTART, "embeddings_unknown"
    if served_layout == SHARED:
        if release.lm_head == DISTINCT:
            return RESTART, "lm_head_differs_from_embeddings_on_a_shared_server"
        return IN_PLACE, "shared_storage"
    if release.lm_head == ABSENT:
        return IN_PLACE_SYNC, "split_server_release_without_lm_head"
    return IN_PLACE, "split_server_release_with_lm_head"


def sample_rows(vocab: int, salt: int, n: int = SAMPLE_ROWS) -> list[int]:
    """``n`` distinct rows spread over the vocabulary (first, last and a salted
    stride between), so successive loads look at different rows."""
    if vocab <= n:
        return list(range(vocab))
    rows = {0, vocab - 1}
    step = max(1, vocab // n)
    k = 0
    while len(rows) < n:
        rows.add((int(salt) * 7919 + k * step + k * k) % vocab)
        k += 1
    return sorted(rows)


def describe(release: ReleaseTie, served_layout: str | None, in_place: bool) -> str:
    """The start-up log line: tied or not, layout, and which path a reload takes."""
    if not in_place:
        how = "every release is loaded by restart (in_place_reload off)"
    elif not release.config_tied:
        how = "in-place reload_weights (untied: the release's lm_head is loaded as any other tensor)"
    else:
        how = ("in-place reload_weights, plus the served embeddings copied into lm_head for a release "
               "without one (split server), restart for a release whose lm_head differs from its "
               "embeddings (shared server); sampled embed/lm_head rows verified after every reload")
    return (f"tie_word_embeddings={str(release.config_tied).lower()}; release lm_head={release.lm_head}; "
            f"served embeddings={served_layout or 'n/a'}; reload: {how}")


def verify(model_dir: Path, release: ReleaseTie, states: Sequence[Mapping[str, Any]], rows: Sequence[int]) -> dict:
    """Compare the rows every rank reported (``yeto_tied_state``) with the ones a
    cold start on ``model_dir`` serves; raises :class:`TiedReloadError`."""
    files = _tensor_files(model_dir)
    embed = _Tensor.find(files, release.embed_name)
    head = _Tensor.find(files, release.head_name) if release.lm_head == DISTINCT else embed
    want = {"embed": dict(zip(rows, embed.rows(rows))), "lm_head": dict(zip(rows, head.rows(rows)))}
    seen: dict[str, set[int]] = {"embed": set(), "lm_head": set()}
    for rank, state in enumerate(states):
        for part in ("embed", "lm_head"):
            for key, values in (state.get(part) or {}).items():
                row = int(key)
                if list(values) != want[part][row]:
                    raise TiedReloadError(
                        f"{model_dir}: after the in-place reload rank {rank} serves {part} row {row} != the "
                        f"release's {release.head_name if part == 'lm_head' and head is not embed else release.embed_name}"
                        f" (a cold start serves the release's); the server must not keep serving")
                seen[part].add(row)
    missing = {part: sorted(set(rows) - got) for part, got in seen.items() if set(rows) - got}
    if missing:
        raise TiedReloadError(f"{model_dir}: no rank reported rows {missing} of the served embeddings")
    return {"rows": list(rows), "ranks": len(states), "lm_head_rows_equal_release": True}


class TiedEmbeddingExtension:
    """Mixed into vLLM's worker (``--worker-extension-cls``; vllm-ascend's
    ``NPUWorker`` takes it the same way); called on every rank by ``collective_rpc``."""

    def _yeto_embeddings(self) -> tuple[Any, Any]:
        from vllm.model_executor.layers.vocab_parallel_embedding import (  # noqa: PLC0415
            ParallelLMHead,
            VocabParallelEmbedding,
        )

        model = self.model_runner.get_model()
        heads, embeds = [], []
        for module in model.modules():
            if isinstance(module, ParallelLMHead):
                heads.append(module)
            elif isinstance(module, VocabParallelEmbedding):
                embeds.append(module)
        if len(heads) != 1 or len(embeds) != 1:
            raise TiedReloadError(f"expected one embed_tokens and one lm_head, found {len(embeds)} / {len(heads)}")
        return embeds[0], heads[0]

    def yeto_tied_state(self, rows: str = "[]") -> dict[str, Any]:
        """Whether this rank's ``lm_head.weight`` is ``embed_tokens.weight``, and the
        requested global vocabulary rows of both that this rank holds."""
        embed, head = self._yeto_embeddings()
        out: dict[str, Any] = {"shared": head.weight is embed.weight
                               or head.weight.data_ptr() == embed.weight.data_ptr(),
                               "embed": {}, "lm_head": {}}
        for part, layer in (("embed", embed), ("lm_head", head)):
            idx = layer.shard_indices
            for row in json.loads(rows):
                if idx.org_vocab_start_index <= int(row) < idx.org_vocab_end_index:
                    local = int(row) - idx.org_vocab_start_index
                    out[part][str(row)] = layer.weight[local].float().cpu().tolist()
        return out

    def yeto_tied_sync(self) -> dict[str, Any]:
        """Copy the served embeddings into a separate ``lm_head`` (same storage
        kept, so captured graphs stay valid); nothing to do when they are shared."""
        embed, head = self._yeto_embeddings()
        if head.weight.data_ptr() == embed.weight.data_ptr():
            return {"synced": False, "shared": True}
        if head.weight.shape != embed.weight.shape:
            raise TiedReloadError(f"lm_head {tuple(head.weight.shape)} != embed_tokens {tuple(embed.weight.shape)}")
        head.weight.data.copy_(embed.weight.data)
        return {"synced": True, "shared": False}
