"""The sample-buffer payload: one group of samples as ``Sample.to_dict()``-shaped
mappings, and the writer that stores a group (payload first, then its index entry).

Engine-neutral (moved from ``adapters.miles.buffer_rollout``, which re-exports
these names): the Miles sampler and the engine sampler island
(:mod:`.engine_sampler`) write the same payloads, so one trainer reader serves
both. A sample carries ``response_length``, per-token ``rollout_log_probs`` and
its engine weight version (``weight_versions`` spans or
``metadata.weight_version``); the entry's ``behavior_version`` is the oldest.

Arrays (``numpy.ndarray`` fields) are stored as base64 bytes (little-endian,
C order) with their dtype and shape. An integer array is stored in the
narrowest integer dtype that holds all its values, and ``cast`` names the
dtype the reader returns (the array's original one), so a reader gets the same
values in the same dtype. A payload written before ``cast`` existed decodes as
before. That is format v1, the default; format v2 (opt-in,
:data:`PAYLOAD_CODEC_ENV`) stores the arrays as binary blobs after the JSON
record, raw or zstd-compressed (see the comment above :data:`PAYLOAD_SCHEMA_V2`).

Rollout fields an engine sampler writes per sample (the trainer side reads
them through :func:`rollout_routed_experts`, :func:`rollout_top_logprobs`):

* ``rollout_routed_experts``: ``uint16[T, L_moe, top_k]``, the original ids of
  the experts the engine routed each of the prompt + response - 1 positions to
  (R3; Kimi K3.1 has 896 experts, so ``uint8`` cannot hold the ids). A padded
  position (no routing, Miles writes -1) is :data:`ROUTED_EXPERTS_PAD` (0xFFFF)
  and the field's record declares ``"pad": 65535`` (F2's trainer-side mapping
  passes it through as -1). A payload written by an earlier engine sampler
  holds ``uint8`` (or any integer dtype, -1 pads) and reads as ``uint16`` too;
* ``rollout_top_logprobs_ids`` ``int32[T_resp, K]`` and ``rollout_top_logprobs``
  ``float32[T_resp, K]``: the K most likely next tokens and their log-probs at
  every response position (score_centering; K = the engine's ``top_logprobs_num``);
* ``rollout_log_probs`` ``float32[T_resp]`` (a list in the payload);
* ``metadata.engine``: ``{"version_fields": {"fp32_lm_head", "temperature",
  "top_p", "top_k"}, "routed_experts": bool, "top_logprobs_num": K}``.
"""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .policy_versions import engine_version_number
from .trainer_lifecycle import SampleIndexEntry

PAYLOAD_SCHEMA = "yeto.rl.sample-buffer-payload/v1"
_NDARRAY = "__ndarray__"


class SampleBufferRolloutError(RuntimeError):
    pass


class DamagedPayload(SampleBufferRolloutError):
    """The stored bytes of one payload are not a group (:func:`decode_group`). A
    fault of the data, not of the read: a reader passes the group over and counts
    it; a read that failed (the store did not answer) is never this error."""


class DamagedPayloadLimit(SampleBufferRolloutError):
    """Too many damaged payloads in one take: a fault of the reader or of a whole
    sampler build (another payload schema), not of one object. The take stops
    instead of consuming the backlog group by group; nothing it selected counts as
    consumed."""


# a take that ends without a round (the data ended, no group arrived in time) names
# what it passed over at the end of its error text, after this mark, as one JSON object
PASSED_OVER_MARK = "; passed over "


NONFINITE_LOGPROB = "nonfinite_logprob"  # the drop reason both trainer readers record


def _first_nonfinite(values: Any) -> tuple[int, Any] | None:
    """(index, value) of the first entry of ``values`` that is not a finite number
    (None: all are). Fast path: an array's vectorized ``isfinite``, else one
    ``sum`` (any NaN/inf makes it non-finite; log-probs are bounded, so a finite
    list does not overflow); only a non-finite or non-numeric sum is located entry
    by entry (a 1M-token sample costs one C-level sum, not a Python loop)."""
    import math

    if hasattr(values, "detach"):  # torch.Tensor
        values = values.detach().cpu().numpy()
    if type(values).__module__ == "numpy":
        import numpy

        flat = numpy.asarray(values).reshape(-1)
        if flat.dtype.kind in "fiub":
            bad = numpy.flatnonzero(~numpy.isfinite(flat))
            return None if bad.size == 0 else (int(bad[0]), flat[bad[0]].item())
        values = flat.tolist()
    try:
        if math.isfinite(sum(values)):
            return None
    except (TypeError, ValueError, OverflowError):
        pass
    for t, value in enumerate(values):
        try:
            ok = not isinstance(value, bool) and math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            ok = False
        if not ok:
            return t, value
    return None


def nonfinite_logprob(group: Sequence[Mapping[str, Any]]) -> str | None:
    """Why a decoded group's behavior log-probs cannot be trained (None: they can).

    The trainers take ``exp(old - rollout_log_probs)`` (TIS / off-policy
    correction, the old-log-prob reuse), so one NaN/inf entry makes the loss NaN,
    masked positions included (NaN * 0 is NaN). Shared by the Miles reader and
    the verl feed; a sample without the field is not judged."""
    for i, sample in enumerate(group):
        values = sample.get("rollout_log_probs")
        if values is None:
            continue
        bad = _first_nonfinite(values)
        if bad is not None:
            return (f"sample {sample.get('index', i)} rollout_log_probs[{bad[0]}] = {bad[1]!r} "
                    f"is not a finite number")
    return None


def passed_over_text(counts: Mapping[str, int]) -> str:
    """The tail of such an error's text ("" when nothing was passed over)."""
    counts = {key: int(value) for key, value in counts.items() if value}
    return PASSED_OVER_MARK + json.dumps(counts, sort_keys=True) if counts else ""


def passed_over_of(text: str) -> dict[str, int] | None:
    """The counts :func:`passed_over_text` put into an error's text (None: none)."""
    _, mark, tail = str(text).rpartition(PASSED_OVER_MARK)
    if not mark:
        return None
    try:
        counts, _end = json.JSONDecoder().raw_decode(tail)
    except ValueError:
        return None
    return {str(k): int(v) for k, v in counts.items()} if isinstance(counts, dict) else None


class SampleBufferExhausted(SampleBufferRolloutError):
    """The samplers ended the buffer and no trainable group is left for the round
    The trainer's run ends normally (FINISHED) before this round; it
    is not a failure: e.g. groups were passed over as older than max_policy_age,
    so the data holds fewer rounds than --total-steps."""


def is_buffer_exhausted(error: BaseException | None) -> bool:
    """Whether ``error`` is (or wraps) :class:`SampleBufferExhausted`.

    The reader raises it in the rollout process; it reaches the driver wrapped
    (a Ray task error keeps it as ``cause`` or as a dual subclass), so the cause
    chain is followed and the class is matched by name as well."""
    return data_end_kind(error) == "buffer_exhausted"


# class name -> data end kind; matched by name (the error comes from another process)
_DATA_END_KINDS = {
    SampleBufferExhausted.__name__: "buffer_exhausted",
    # yeto.rl.engine.task_dispense.DispenseEnded (--rl-data-passes used up)
    "DispenseEnded": "dispense_ended",
}


def data_end_kind(error: BaseException | None) -> str | None:
    """``"buffer_exhausted"`` / ``"dispense_ended"`` when ``error`` is (or wraps)
    the rollout side's end of the data, else None. Follows the cause chain as
    :func:`is_buffer_exhausted` does."""
    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        for cls in type(error).__mro__:
            kind = _DATA_END_KINDS.get(cls.__name__)
            if kind is not None:
                return kind
        cause = getattr(error, "cause", None)
        error = cause if isinstance(cause, BaseException) else (error.__cause__ or error.__context__)
    return None


_CAST = "cast"  # the dtype the reader returns when the stored one is narrower
_NARROW_INTS = ("uint8", "int8", "uint16", "int16", "uint32", "int32")


def _narrowest_int(array: Any) -> Any:
    """The narrowest integer dtype that holds every value of the integer ``array``
    (its own dtype when none is narrower)."""
    import numpy

    if array.size == 0:
        return array.dtype
    low, high = int(array.min()), int(array.max())
    for name in _NARROW_INTS:
        candidate = numpy.dtype(name)
        if candidate.itemsize >= array.dtype.itemsize:
            break
        info = numpy.iinfo(candidate)
        if info.min <= low and high <= info.max:
            return candidate
    return array.dtype


class _Field:
    """A sample field with extra record keys (``_FIELD_META``), e.g. the pad id."""

    __slots__ = ("array", "meta")

    def __init__(self, array: Any, meta: Mapping[str, Any]) -> None:
        self.array, self.meta = array, dict(meta)


def _with_field_meta(sample: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(sample)
    for name, meta in _FIELD_META.items():
        value = out.get(name)
        if value is not None and hasattr(value, "dtype"):
            out[name] = _Field(value, meta)
    return out


def _array_record(value: Any) -> tuple[dict[str, Any], Any] | None:
    """``(record without its data, stored array)`` of an array field (the stored
    array in the narrowest integer dtype, ``cast`` naming the original), None for a
    value that is not an array."""
    extra: dict[str, Any] = {}
    if isinstance(value, _Field):
        value, extra = value.array, value.meta
    tolist = getattr(value, "tolist", None)
    if not (callable(tolist) and hasattr(value, "dtype") and hasattr(value, "shape")):
        return None
    import numpy

    array = numpy.ascontiguousarray(value)
    record: dict[str, Any] = {"dtype": str(array.dtype), "shape": list(array.shape), **extra}
    if array.dtype.kind in "iu":
        stored = _narrowest_int(array)
        if stored != array.dtype:
            record = {"dtype": str(stored), _CAST: str(array.dtype), "shape": list(array.shape), **extra}
            array = array.astype(stored)
    return record, array


def _encode_value(value: Any) -> Any:
    found = _array_record(value)
    if found is not None:
        record, array = found
        return {_NDARRAY: base64.b64encode(array.tobytes()).decode("ascii"), **record}
    if hasattr(value, "value") and type(value).__module__ != "builtins":  # enum
        return value.value
    raise TypeError(f"sample field of type {type(value).__name__} is not serializable")


def _decode_hook(obj: dict[str, Any]) -> Any:
    if _NDARRAY in obj:
        import numpy

        data = base64.b64decode(obj[_NDARRAY])
        array = numpy.frombuffer(data, dtype=obj["dtype"]).reshape(obj["shape"])
        cast = obj.get(_CAST)
        return array.astype(cast) if cast else array.copy()
    return obj


ROLLOUT_ROUTED_EXPERTS = "rollout_routed_experts"
ROLLOUT_TOP_LOGPROBS_IDS = "rollout_top_logprobs_ids"
ROLLOUT_TOP_LOGPROBS = "rollout_top_logprobs"
ROLLOUT_LOG_PROBS = "rollout_log_probs"
ENGINE_META = "engine"  # metadata key: version_fields, routed_experts, top_logprobs_num
ROUTED_EXPERTS_DTYPE = "uint16"
ROUTED_EXPERTS_PAD = 65535  # 0xFFFF: a position without routing (-1 in a signed payload)
# per-field record extras (next to dtype / shape) the writer declares
_FIELD_META: dict[str, dict[str, Any]] = {ROLLOUT_ROUTED_EXPERTS: {"pad": ROUTED_EXPERTS_PAD}}
TOP_LOGPROBS_IDS_DTYPE = "int32"
TOP_LOGPROBS_DTYPE = "float32"


def rollout_routed_experts(sample: Mapping[str, Any]) -> Any:
    """The sample's R3 routed experts as ``uint16[T, L_moe, top_k]`` (module
    doc), whatever integer dtype the payload holds (``uint8`` from an earlier
    writer, ``int32`` from a Miles-style writer); None when the sample has none.
    Raises :class:`SampleBufferRolloutError` on a value no expert id can take."""
    import numpy

    value = sample.get(ROLLOUT_ROUTED_EXPERTS)
    if value is None:
        return None
    return routed_experts_uint16(value, where=f"sample {sample.get('index')}")


def routed_experts_uint16(value: Any, *, where: str = "routed_experts") -> Any:
    """``value`` (any integer array ``[T, L_moe, top_k]``) as ``uint16`` with -1
    pads mapped to :data:`ROUTED_EXPERTS_PAD`; an id below -1 or above 65534 is
    an error (65535 is the pad)."""
    import numpy

    array = numpy.asarray(value)
    if array.dtype.kind not in "iu":
        raise SampleBufferRolloutError(f"{where}: {ROLLOUT_ROUTED_EXPERTS} has dtype {array.dtype}, "
                                       "expected an integer dtype")
    if array.ndim != 3:
        raise SampleBufferRolloutError(f"{where}: {ROLLOUT_ROUTED_EXPERTS} has shape {array.shape}, "
                                       "expected [T, L_moe, top_k]")
    if array.size:
        low, high = int(array.min()), int(array.max())
        if low < -1 or high > ROUTED_EXPERTS_PAD:
            raise SampleBufferRolloutError(f"{where}: {ROLLOUT_ROUTED_EXPERTS} holds an id outside uint16 "
                                           f"({low}..{high}; -1 / {ROUTED_EXPERTS_PAD} is the pad)")
        if low == -1:
            array = numpy.where(array == -1, ROUTED_EXPERTS_PAD, array)
    return array.astype(ROUTED_EXPERTS_DTYPE, copy=False)


def rollout_top_logprobs(sample: Mapping[str, Any]) -> tuple[Any, Any] | None:
    """``(ids int32[T_resp, K], logprobs float32[T_resp, K])`` of the sample, or
    None when it carries no top-K log-probs. The two must agree in shape and
    cover every response token."""
    import numpy

    ids, vals = sample.get(ROLLOUT_TOP_LOGPROBS_IDS), sample.get(ROLLOUT_TOP_LOGPROBS)
    if ids is None and vals is None:
        return None
    if ids is None or vals is None:
        raise SampleBufferRolloutError(f"sample {sample.get('index')}: {ROLLOUT_TOP_LOGPROBS_IDS} and "
                                       f"{ROLLOUT_TOP_LOGPROBS} come together")
    ids, vals = numpy.asarray(ids), numpy.asarray(vals)
    length = int(sample.get("response_length") or 0)
    if ids.ndim != 2 or ids.shape != vals.shape or ids.shape[0] != length:
        raise SampleBufferRolloutError(f"sample {sample.get('index')}: top log-probs ids {ids.shape} / values "
                                       f"{vals.shape} do not cover response_length {length} as [T_resp, K]")
    return ids.astype(TOP_LOGPROBS_IDS_DTYPE, copy=False), vals.astype(TOP_LOGPROBS_DTYPE, copy=False)


# Miles ``Sample`` field names of the engine sampler's rollout fields (the trainer reads
# the payload through :func:`miles_sample_dict` before ``Sample.from_dict``).
MILES_TOPK_TOKEN_IDS = "rollout_topk_token_ids"
MILES_TOPK_LOG_PROBS = "rollout_topk_log_probs"


def miles_sample_dict(sample: Mapping[str, Any], *,
                      r3_layers: tuple[int, int] | None = None,
                      r3_num_experts: int | None = None) -> dict[str, Any]:
    """``sample`` with the engine sampler's rollout fields in Miles' ``Sample`` form.

    ``r3_num_experts``: the model's routed expert count (an id at or above it is an
    error, as is a routing row that mixes pads and ids).
    ``r3_layers`` = ``(num_layers, first_dense_layers)`` of the model: the engine's
    R3 holds the MoE layers only (``L_moe = num_layers - first_dense_layers``, the
    layer offset of the selection), Miles' replay indexes R3 by the global layer
    number. An ``L_moe`` array gets ``first_dense_layers`` all ``-1`` rows in front
    (the dense layers have no router, Miles never reads them); a ``num_layers``
    array (a Miles sampler's) is kept; any other layer count is an error.

    * ``rollout_top_logprobs_ids`` / ``rollout_top_logprobs`` become
      ``rollout_topk_token_ids`` (int32) / ``rollout_topk_log_probs`` (float32):
      Miles' score_centering reads only its own names and refuses a sample without
      them. A sample that already holds the Miles names (a Miles sampler wrote it)
      keeps them; holding both forms is an error.
    * ``rollout_routed_experts`` (``uint16``, pad 0xFFFF) becomes ``int32`` with
      -1 pads, the dtype and pad Miles' replay fill (``pad_func``) and the K3.1
      slot mapping (negative = padding) work in; with ``r3_layers`` its layer axis
      is the model's global layer number.
    * a sample without ``weight_versions`` (an engine sampler's: one generate call
      at one version, ``metadata.weight_version``, no in-call update) gets one
      Miles ``WeightVersionsPerCall`` whose single output span covers the response
      tokens at that version, so Miles' per-token version accounting (and the
      full-parameter trajectory evidence) sees the behavior version. A sample whose
      ``final_weight_version`` differs (an in-place update mid-generation: a live
      agentic trial, or a generate call a live reload fell inside) has no per-token
      split; its span covers the response at the oldest of the two, the version the
      sampler names as its behavior version (``metadata.weight_version``: the
      trainer's ``max_policy_age`` gate counts its age from there, and TIS on
      ``rollout_log_probs`` corrects every token for the policy that actually
      sampled it). Without that span the full-parameter evidence found tokens of
      unknown version after the batch was taken and stopped the island.
    """
    import numpy

    out = dict(sample)
    spans = _engine_weight_version_call(out)
    if spans is not None:
        out["weight_versions"] = [spans]
    top = rollout_top_logprobs(out)
    if top is not None:
        if out.get(MILES_TOPK_TOKEN_IDS) is not None or out.get(MILES_TOPK_LOG_PROBS) is not None:
            raise SampleBufferRolloutError(f"sample {out.get('index')}: holds both {ROLLOUT_TOP_LOGPROBS_IDS} "
                                           f"and {MILES_TOPK_TOKEN_IDS}")
        del out[ROLLOUT_TOP_LOGPROBS_IDS], out[ROLLOUT_TOP_LOGPROBS]
        out[MILES_TOPK_TOKEN_IDS] = numpy.ascontiguousarray(top[0], dtype=numpy.int32)
        out[MILES_TOPK_LOG_PROBS] = numpy.ascontiguousarray(top[1], dtype=numpy.float32)
    routed = rollout_routed_experts(out)
    if routed is not None:
        ids = routed.astype(numpy.int32)
        ids[routed == ROUTED_EXPERTS_PAD] = -1
        where = f"sample {out.get('index')}: {ROLLOUT_ROUTED_EXPERTS}"
        pad = ids < 0
        if (pad.any(axis=-1) & ~pad.all(axis=-1)).any():
            raise SampleBufferRolloutError(f"{where} mixes pads and expert ids in one routing row")
        if r3_num_experts is not None and ids.size and int(ids.max()) >= int(r3_num_experts):
            raise SampleBufferRolloutError(f"{where} names expert {int(ids.max())}; the model has "
                                           f"{int(r3_num_experts)}")
        if r3_layers is not None:
            total, dense = r3_layers
            if ids.shape[1] == total - dense and dense:
                rows = numpy.full((ids.shape[0], dense, ids.shape[2]), -1, dtype=numpy.int32)
                ids = numpy.concatenate([rows, ids], axis=1)
            elif ids.shape[1] != total:
                raise SampleBufferRolloutError(
                    f"sample {out.get('index')}: {ROLLOUT_ROUTED_EXPERTS} has {ids.shape[1]} layers; the model "
                    f"has {total} layers, {total - dense} of them MoE")
        out[ROLLOUT_ROUTED_EXPERTS] = ids
    return out


def _engine_weight_version_call(sample: Mapping[str, Any]) -> dict[str, Any] | None:
    """The one-span Miles call of an engine-sampler sample (miles_sample_dict doc)."""
    if sample.get("weight_versions"):
        return None
    metadata = sample.get("metadata") or {}
    version = metadata.get("weight_version")
    if version is None or engine_version_number(version) is None:
        return None
    final = metadata.get("final_weight_version")
    if final is not None:
        newest = engine_version_number(final)
        if newest is None:
            return None
        if newest < engine_version_number(version):
            version = final  # the oldest of the two (miles_sample_dict doc)
    tokens = len(sample.get("tokens") or ())
    response = int(sample.get("response_length") or 0)
    if response <= 0 or response > tokens:
        return None
    start = tokens - response
    return {"spans": [{"version": str(version), "abs_start": start, "abs_end": tokens}],
            "prefill_spans": [], "output_start": start, "prompt_tokens": start}


# Payload format v2 (opt-in, :data:`PAYLOAD_CODEC_ENV`): the same JSON record, but
# every array is a blob after it instead of base64 inside it. A segmented
# (compacted) trial carries R3 and top-K per segment, so a group of long trials is
# tens of GB in v1; v2 drops the base64 third and, with ``zstd``, compresses each
# array by byte planes (K3.1 payloads measured: R3 1.52x, top-K ids 1.93x,
# top-K log-probs 4.36x). Layout: :data:`_V2_MAGIC`, the header length (uint64
# little-endian), the header JSON (``schema`` :data:`PAYLOAD_SCHEMA_V2`), the
# blobs. An array record holds ``__blob__: [offset, nbytes]`` (into the blobs),
# ``codec`` (``raw`` or ``zstd``) and, like v1, ``dtype`` / ``shape`` / ``cast``
# and the field extras. A reader of this module reads v1 and v2; a reader before
# v2 refuses a v2 payload as damaged (``payload does not decode``), so the
# trainers must run this code before any sampler sets the codec.
PAYLOAD_SCHEMA_V2 = "yeto.rl.sample-buffer-payload/v2"
PAYLOAD_CODEC_ENV = "YETO_SAMPLE_BUFFER_PAYLOAD_CODEC"  # json (default, v1) | raw | zstd (v2)
PAYLOAD_CODECS = ("json", "raw", "zstd")
_V2_MAGIC = b"YETO-SBP2\n"
_V2_HEADER_LEN = 8
_BLOB = "__blob__"
_ZSTD_LEVEL = 1  # K3.1 R3 per sample: the ratio of level 3 (1.52x), compressed faster


def payload_codec(environ: Mapping[str, str] | None = None) -> str:
    """The payload codec :data:`PAYLOAD_CODEC_ENV` names (unset: ``json``, format v1)."""
    import os

    raw = str((os.environ if environ is None else environ).get(PAYLOAD_CODEC_ENV) or "json").strip().lower()
    if raw not in PAYLOAD_CODECS:
        raise ValueError(f"{PAYLOAD_CODEC_ENV}={raw!r}: expected one of {', '.join(PAYLOAD_CODECS)}")
    return raw


def _zstd() -> tuple[Callable[[bytes, int], bytes], Callable[[bytes], bytes]]:
    from .release_delta import ReleaseDeltaError, _zstd as zstd_pair  # noqa: PLC0415 - only v2 needs it

    try:
        return zstd_pair()
    except ReleaseDeltaError as error:
        raise SampleBufferRolloutError(f"payload codec zstd needs zstd: Python >= 3.14 or "
                                       f"`pip install zstandard` ({PAYLOAD_CODEC_ENV})") from error


def _byte_planes(array: Any) -> Any:
    """The bytes of ``array`` grouped by byte position (all first bytes, then all
    second bytes, ...; a uint16 expert id < 1024 has a high byte of 0..3), as a
    contiguous ``uint8`` array. One plane at a time: about 3x faster than a
    transposed copy."""
    import numpy

    flat = array.reshape(-1).view(numpy.uint8).reshape(-1, array.itemsize)
    planes = numpy.empty((array.itemsize, flat.shape[0]), dtype=numpy.uint8)
    for byte in range(array.itemsize):
        planes[byte] = flat[:, byte]
    return planes


def _from_byte_planes(data: bytes, dtype: Any, count: int) -> Any:
    """The flat ``dtype`` array whose :func:`_byte_planes` are ``data``."""
    import numpy

    dtype = numpy.dtype(dtype)
    planes = numpy.frombuffer(data, dtype=numpy.uint8).reshape(dtype.itemsize, count)
    out = numpy.empty(count, dtype=dtype)
    flat = out.view(numpy.uint8).reshape(count, dtype.itemsize)
    for byte in range(dtype.itemsize):
        flat[:, byte] = planes[byte]
    return out


class _BlobEncoder:
    """``json.dumps`` ``default`` for v2: an array becomes a record naming its blob."""

    def __init__(self, codec: str) -> None:
        self.codec = codec
        self.blobs: list[bytes] = []
        self.size = 0
        self._compress = _zstd()[0] if codec == "zstd" else None

    def __call__(self, value: Any) -> Any:
        found = _array_record(value)
        if found is None:
            return _encode_value(value)
        record, array = found
        blob = array.tobytes() if self._compress is None else self._compress(_byte_planes(array), _ZSTD_LEVEL)
        record.update({_BLOB: [self.size, len(blob)], "codec": self.codec})
        self.blobs.append(blob)
        self.size += len(blob)
        return record


def encode_group(samples: Sequence[Mapping[str, Any]], *, codec: str | None = None) -> bytes:
    """One payload: the group's samples as ``Sample.to_dict()`` mappings.
    ``codec``: ``json`` (format v1, the bytes every earlier writer wrote), ``raw``
    or ``zstd`` (format v2, the comment above); None reads :data:`PAYLOAD_CODEC_ENV`."""
    codec = payload_codec() if codec is None else payload_codec({PAYLOAD_CODEC_ENV: codec})
    if codec == "json":
        record = {"schema": PAYLOAD_SCHEMA, "samples": [_with_field_meta(s) for s in samples]}
        return json.dumps(record, default=_encode_value, sort_keys=True, allow_nan=True).encode()
    encoder = _BlobEncoder(codec)
    record = {"schema": PAYLOAD_SCHEMA_V2, "samples": [_with_field_meta(s) for s in samples]}
    header = json.dumps(record, default=encoder, sort_keys=True, allow_nan=True).encode()
    return b"".join([_V2_MAGIC, len(header).to_bytes(_V2_HEADER_LEN, "little"), header, *encoder.blobs])


def _v2_record(data: bytes) -> Any:
    """The header record of a v2 payload, its arrays decoded from the blobs."""
    import numpy

    view = memoryview(data)
    start = len(_V2_MAGIC) + _V2_HEADER_LEN
    if len(view) < start:
        raise ValueError("v2 payload is cut short before its header")
    header_end = start + int.from_bytes(view[len(_V2_MAGIC):start], "little")
    if header_end > len(view):
        raise ValueError("v2 payload is cut short inside its header")
    blobs = view[header_end:]
    decompress: list[Callable[[bytes], bytes]] = []

    def hook(obj: dict[str, Any]) -> Any:
        if _BLOB not in obj:
            return obj
        offset, nbytes = (int(x) for x in obj[_BLOB])
        if offset < 0 or nbytes < 0 or offset + nbytes > len(blobs):
            raise ValueError(f"array blob [{offset}, {nbytes}] is outside the payload's {len(blobs)} blob bytes")
        blob, dtype, shape = blobs[offset:offset + nbytes], numpy.dtype(obj["dtype"]), obj["shape"]
        if any(int(n) < 0 for n in shape):
            raise ValueError(f"array shape {shape} has a negative dimension")
        count = math.prod(int(n) for n in shape)
        codec = obj.get("codec")
        if codec == "raw":
            if nbytes != count * dtype.itemsize:
                raise ValueError(f"raw blob holds {nbytes} B, the array {count * dtype.itemsize} B")
            array = numpy.frombuffer(blob, dtype=dtype, count=count)
        elif codec == "zstd":
            if not decompress:
                decompress.append(_zstd()[1])
            try:
                raw = decompress[0](blob)
            except MemoryError:
                raise
            except Exception as error:  # noqa: BLE001 - the zstd module's own error class
                raise ValueError(f"zstd blob does not decompress: {type(error).__name__}: {error}") from error
            if len(raw) != count * dtype.itemsize:
                raise ValueError(f"zstd blob holds {len(raw)} B, the array {count * dtype.itemsize} B")
            array = _from_byte_planes(raw, dtype, count)
        else:
            raise ValueError(f"array codec {codec!r} is unknown")
        array = array.reshape(shape)
        cast = obj.get(_CAST)
        return array.astype(cast) if cast else array.copy() if codec == "raw" else array

    return json.loads(bytes(view[start:header_end]), object_hook=hook)


def decode_group(data: bytes) -> list[dict[str, Any]]:
    """The samples of one payload (format v1 or v2). Bytes that are not a payload
    (not JSON, cut short, not an object, another schema, ``samples`` not a list of
    objects) raise :class:`DamagedPayload`: the fault is in the stored bytes, so
    reading them again gives the same answer."""
    v2 = bytes(data[:len(_V2_MAGIC)]) == _V2_MAGIC
    try:
        record = _v2_record(data) if v2 else json.loads(data, object_hook=_decode_hook)
    except (ValueError, KeyError, TypeError) as error:  # not JSON, not UTF-8, a broken array record
        raise DamagedPayload(f"payload does not decode: {type(error).__name__}: {str(error)[:200]}") from error
    if not isinstance(record, Mapping):
        raise DamagedPayload(f"payload is a {type(record).__name__}, not an object")
    schema = PAYLOAD_SCHEMA_V2 if v2 else PAYLOAD_SCHEMA
    if record.get("schema") != schema:
        raise DamagedPayload(f"payload schema {record.get('schema')!r} != {schema}")
    samples = record.get("samples")
    if not isinstance(samples, list) or not all(isinstance(sample, Mapping) for sample in samples):
        raise DamagedPayload("payload samples are not a list of objects")
    return list(samples)


def payload_is_missing(buffer: Any, ref: str, error: BaseException) -> bool:
    """Whether a failed payload read means the payload does not exist (a fault of
    the data: its index record outlived it), as opposed to a store that did not
    answer (a throttle, a 5xx, a timeout, a refused credential), which must stay an
    error. Only the buffer's own answer counts: ``read_payload`` raised
    ``SampleBufferError`` (an object store answered "no such key") or
    ``FileNotFoundError`` (a shared directory), and a second, independent look
    (``has_payload``) also says it is absent. If that look fails too it raises:
    a store in trouble is never read as a missing payload."""
    from .sample_buffer import SampleBufferError  # noqa: PLC0415 - sample_buffer is the heavier module

    if not isinstance(error, (SampleBufferError, FileNotFoundError)):
        return False
    has_payload = getattr(getattr(buffer, "index", None), "has_payload", None)
    if not callable(has_payload):
        return isinstance(error, SampleBufferError) and "is missing" in str(error)
    return not has_payload(ref)


def cross_version(sample: Mapping[str, Any]) -> bool:
    """``sample`` was served by two policy versions (``metadata.weight_version`` and
    ``final_weight_version`` both numeric and different): :func:`miles_sample_dict`
    names its tokens by the oldest."""
    metadata = sample.get("metadata") or {}
    first, final = metadata.get("weight_version"), metadata.get("final_weight_version")
    if first is None or final is None:
        return False
    first, final = engine_version_number(first), engine_version_number(final)
    return first is not None and final is not None and first != final


def unknown_version_tokens(sample: Mapping[str, Any]) -> int:
    """Response tokens of ``sample`` (a :func:`miles_sample_dict`) that no
    ``weight_versions`` span names a numeric version for: the tokens the
    full-parameter trajectory evidence would find of unknown policy version."""
    response = int(sample.get("response_length") or 0)
    start = len(sample.get("tokens") or ()) - response
    known = [False] * max(0, response)
    for call in sample.get("weight_versions") or ():
        spans = call.get("spans") if isinstance(call, Mapping) else call
        for span in spans or ():
            if engine_version_number(span.get("version")) is None:
                continue
            for pos in range(max(int(span["abs_start"]), start), min(int(span["abs_end"]), start + response)):
                known[pos - start] = True
    return len(known) - sum(known)


def _sample_versions(sample: Mapping[str, Any]) -> list[int]:
    versions = []
    for call in sample.get("weight_versions") or ():
        # Miles Sample.to_dict(): one {"spans": [...], "prefill_spans": ...} per call
        # (WeightVersionsPerCall); older payloads: a bare list of spans per call.
        spans = call.get("spans") if isinstance(call, Mapping) else call
        for span in spans or ():
            versions.append(span["version"])
    if not versions and (sample.get("metadata") or {}).get("weight_version") is not None:
        versions.append(sample["metadata"]["weight_version"])
    numbers = [engine_version_number(v) for v in versions]
    if not numbers or None in numbers:
        raise SampleBufferRolloutError(
            f"sample {sample.get('index')}: no numeric engine weight version ({versions!r}); "
            "the buffer needs the behavior version of every sample")
    return [int(n) for n in numbers]


def behavior_versions(sample: Mapping[str, Any]) -> list[int]:
    """Engine weight versions of one sample's tokens (the buffer's behavior versions)."""
    return _sample_versions(sample)


# ---------------------------------------------------------------------- trials
# A compacted trial (CompactionRL segments, async GRPO) is several samples of one group:
# the same group_index and index, the trial's reward, and ``metadata.compaction``
# ({"schema", "trajectory_id", "segment_index", "num_segments", ...}). A group with such
# samples is counted and numbered by trial; every other group holds one sample per trial
# (its count is ``len``) and is handled exactly as before.
COMPACTION_KEY = "compaction"


def _field(sample: Any, name: str) -> Any:
    return sample.get(name) if isinstance(sample, Mapping) else getattr(sample, name, None)


def compaction_meta(sample: Any) -> Mapping[str, Any] | None:
    """``metadata.compaction`` of a segment sample (a dict or a Miles ``Sample``), else None."""
    meta = _field(sample, "metadata")
    value = meta.get(COMPACTION_KEY) if isinstance(meta, Mapping) else None
    return value if isinstance(value, Mapping) else None


def trial_key(sample: Any, row: int) -> Any:
    """The trial a sample belongs to: Miles' rollout key (``reward_shaping.rollout_segments``:
    ``rollout_id``, else ``index``, else the row)."""
    rollout_id = _field(sample, "rollout_id")
    if rollout_id is not None:
        return rollout_id
    index = _field(sample, "index")
    return ("row", row) if index is None else index


def has_siblings(samples: Sequence[Any]) -> bool:
    """Whether a group holds compaction segments (``metadata.compaction`` on a sample)."""
    return any(compaction_meta(s) is not None for s in samples)


def trial_rows(samples: Sequence[Any]) -> list[list[int]]:
    """The rows of each trial of a group, in first-seen order (by :func:`trial_key`);
    one row per trial in a group without compaction segments (whatever its indices)."""
    if not has_siblings(samples):
        return [[row] for row in range(len(samples))]
    by_trial: dict[Any, list[int]] = {}
    for row, sample in enumerate(samples):
        by_trial.setdefault(trial_key(sample, row), []).append(row)
    return list(by_trial.values())


def trial_count(samples: Sequence[Any]) -> int:
    """The trials of a group (``len(samples)`` without compaction segments)."""
    return len(trial_rows(samples))


def check_trials(samples: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    """The rows of each trial of a group with compaction segments (in segment order),
    checked by the contract (``yeto.rl.contracts.validate_compaction_group``: one shared
    index, reward and ``trajectory_id`` per trial, segments ``0..n-1``, only the last one
    truncated, no ``trajectory_id`` or ``rollout_id`` shared by two trials). Raises
    :class:`SampleBufferRolloutError`."""
    from yeto.rl.contracts import validate_compaction_group  # noqa: PLC0415

    try:
        trials = validate_compaction_group(samples)
    except ValueError as error:
        raise SampleBufferRolloutError(f"compaction group: {error}") from error
    return [list(trial.positions) for trial in trials]


def finite_rewards(group: Sequence[Mapping[str, Any]]) -> bool:
    """Whether every sample of a decoded group has a finite reward. A missing
    (None) or NaN/inf scalar reward (a reward call that failed) makes the group's
    centered advantages NaN in Miles and verl, hence a non-finite loss, or the driver's
    ``nonfinite_reward`` stop; such a group is consumed, never trained. A reward
    dict (a multi-key reward Miles reads by ``reward_key``) is not judged here."""
    for sample in group:
        if "reward" not in sample:
            continue  # a payload without the field (not a Miles Sample.to_dict()): not judged
        reward = sample["reward"]
        if isinstance(reward, Mapping):
            continue
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
            return False
    return True


def payload_oldest_version(samples: Sequence[Mapping[str, Any]]) -> int | None:
    """The oldest engine version of any token of a decoded group, as its payload
    says (what :func:`group_entry_fields` wrote into the index record); None when the
    payload names no numeric version (nothing to compare)."""
    try:
        return min(min(_sample_versions(sample)) for sample in samples)
    except (SampleBufferRolloutError, ValueError, TypeError, KeyError, AttributeError):
        return None


def group_entry_fields(samples: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    """``(behavior_version, tokens)`` of a group: the oldest engine version of any
    token, and the response tokens. Every sample must carry per-token behavior
    log-probs (decoupled PPO needs them)."""
    if not samples:
        raise SampleBufferRolloutError("empty group")
    oldest, tokens = None, 0
    for sample in samples:
        length = int(sample.get("response_length") or 0)
        logprobs = sample.get("rollout_log_probs")
        if logprobs is None or len(logprobs) != length:
            raise SampleBufferRolloutError(
                f"sample {sample.get('index')}: rollout_log_probs "
                f"{'missing' if logprobs is None else len(logprobs)} != response_length {length}")
        version = min(_sample_versions(sample))
        oldest = version if oldest is None else min(oldest, version)
        tokens += length
    return int(oldest), tokens


PROMPT_TOKENIZED_BY = "prompt_tokenized_by"  # metadata: who produced the prompt ids


def complete_prompt_tokens(sample: Mapping[str, Any],
                           tokenize: Callable[[str], Sequence[int]] | None) -> dict[str, Any]:
    """Trainer side of ``metadata.prompt_tokens_known``.

    An engine that does not return the prompt ids (SGLang text input) writes
    ``tokens`` = response only and ``prompt_tokens_known`` False.  The trainer
    then tokenizes the prompt text itself: ``tokens`` becomes
    ``tokenize(prompt) + response`` and ``metadata.prompt_tokenized_by`` is
    ``trainer``.  ``tokenize`` must match the engine's own text tokenization
    (SGLang and vLLM default to the HF tokenizer with special tokens); that
    match is not verified on a real engine.  Without ``tokenize`` such a
    sample is refused: training on response-only ``tokens`` would read the
    response as the prompt.  A sample with known prompt ids is returned as is.
    """
    metadata = sample.get("metadata") or {}
    if metadata.get("prompt_tokens_known", True):
        return dict(sample)
    where = f"sample {sample.get('index')}"
    if tokenize is None:
        raise SampleBufferRolloutError(f"{where}: the engine gave no prompt ids and this reader has no "
                                       "tokenizer to rebuild them")
    response = list(sample.get("tokens") or ())
    if len(response) != int(sample.get("response_length") or 0):
        raise SampleBufferRolloutError(f"{where}: prompt_tokens_known is False but tokens has "
                                       f"{len(response)} ids for response_length {sample.get('response_length')}")
    prompt = [int(t) for t in tokenize(str(sample.get("prompt") or ""))]
    if not prompt:
        raise SampleBufferRolloutError(f"{where}: the prompt text tokenizes to no ids")
    routed = sample.get(ROLLOUT_ROUTED_EXPERTS)
    if routed is not None and len(routed) != len(prompt) + len(response) - 1:
        # R3 has one row per prompt + response - 1 position of the engine's own tokens:
        # a different count means the trainer's prompt ids are not the engine's.
        raise SampleBufferRolloutError(
            f"{where}: the trainer tokenized the prompt to {len(prompt)} ids, but the engine's "
            f"{ROLLOUT_ROUTED_EXPERTS} rows ({len(routed)}) imply {len(routed) + 1 - len(response)} prompt "
            "ids; the routing replay would not line up with the tokens")
    return {**sample, "tokens": prompt + response,
            "metadata": {**metadata, PROMPT_TOKENIZED_BY: "trainer"}}


# One payload is one object written by one PUT (``S3ObjectStore.write``): S3 and OSS
# refuse a single PutObject over 5 GiB (R2: the same order). A compacted trial
# carries R3 and top-K per segment, so its group grows with every compaction.
SINGLE_PUT_LIMIT_BYTES = 5 * 2**30
PAYLOAD_WARN_BYTES = 4 * 2**30  # GroupWriter warns from here (a group near the limit)
# Payload bytes per stream token of a Kimi K3.1 trial (R3 uint16[92, 16] per token,
# top-128 ids + log-probs per response token, the token lists), upper bounds: v1
# base64, v2 raw, v2 zstd (ratios measured on K3.1 payloads; tool tokens compress
# far better than this).
K31_PAYLOAD_BYTES_PER_TOKEN = {"json": 5324, "raw": 4001, "zstd": 2353}


# The raw (v2) K3.1 bytes per token by field: R3 (92 MoE layers x top-16 x uint16), the
# top-128 int32 ids + float32 log-probs, and the rest (token lists, log-probs, masks).
K31_R3_BYTES_PER_TOKEN = 92 * 16 * 2
K31_TOP_LOGPROBS_NUM = 128
TOP_LOGPROB_BYTES = 8
OTHER_BYTES_PER_TOKEN = (K31_PAYLOAD_BYTES_PER_TOKEN["raw"] - K31_R3_BYTES_PER_TOKEN
                         - K31_TOP_LOGPROBS_NUM * TOP_LOGPROB_BYTES)


def k31_group_payload_bytes(trials: int, context_budget: int, *, max_compactions: int = 0,
                            codec: str = "json") -> int:
    """Upper bound of one K3.1 group payload: ``trials`` trials whose
    ``1 + max_compactions`` segments each fill ``context_budget`` tokens (a
    preflight bound; compare with :data:`SINGLE_PUT_LIMIT_BYTES`)."""
    per_token = K31_PAYLOAD_BYTES_PER_TOKEN[payload_codec({PAYLOAD_CODEC_ENV: codec})]
    return int(trials) * (1 + int(max_compactions)) * int(context_budget) * per_token


def group_payload_bytes(trials: int, context_budget: int, *, r3_bytes_per_token: int | None,
                        top_logprobs_num: int = K31_TOP_LOGPROBS_NUM, max_compactions: int = 0,
                        codec: str = "json") -> int:
    """Upper bound of one group payload of any MoE model: the raw bytes per token are the
    model's R3 row (``r3_bytes_per_token``: MoE layers x top-k x 2 bytes), the
    ``top_logprobs_num`` ids and log-probs and the K3.1 rest; the other codecs keep the
    ratio to raw measured on K3.1. ``r3_bytes_per_token`` None: the K3.1 bound
    (:func:`k31_group_payload_bytes`)."""
    if r3_bytes_per_token is None:
        return k31_group_payload_bytes(trials, context_budget, max_compactions=max_compactions, codec=codec)
    name = payload_codec({PAYLOAD_CODEC_ENV: codec})
    raw = int(r3_bytes_per_token) + int(top_logprobs_num) * TOP_LOGPROB_BYTES + OTHER_BYTES_PER_TOKEN
    per_token = raw * K31_PAYLOAD_BYTES_PER_TOKEN[name] / K31_PAYLOAD_BYTES_PER_TOKEN["raw"]
    return int(int(trials) * (1 + int(max_compactions)) * int(context_budget) * per_token)


class GroupWriter:
    """Writes finished groups to the buffer: payload first, then the index entry.
    A payload of :data:`PAYLOAD_WARN_BYTES` or more is logged (one PUT carries it)."""

    def __init__(self, buffer: Any, sampler_id: str) -> None:
        if not sampler_id:
            raise ValueError("sampler_id must be non-empty")
        self.buffer = buffer
        self.sampler_id = str(sampler_id)

    def write(self, samples: Sequence[Mapping[str, Any]], *, group_key: str,
              provenance: Mapping[str, str] | None = None,
              sampler_round: Mapping[str, Any] | None = None) -> SampleIndexEntry:
        """``provenance``: the sampler's source record for the index entry;
        ``sampler_round``: the round record a replacement sampler resumes from
        (``sampler_resume.SamplerRound.record()``)."""
        behavior_version, tokens = group_entry_fields(samples)
        sample_id = f"{self.sampler_id}-{group_key}"
        data = encode_group(samples)
        if len(data) >= PAYLOAD_WARN_BYTES:
            print(f"[sample-buffer] payload {sample_id}: {len(data) / 2**30:.2f} GiB in one object "
                  f"({len(samples)} samples, codec {payload_codec()}); an S3 / R2 / OSS store refuses a "
                  f"single PUT over {SINGLE_PUT_LIMIT_BYTES / 2**30:.0f} GiB: set {PAYLOAD_CODEC_ENV}=zstd "
                  "or lower agentic.compaction (context_budget, max_compactions)", flush=True)
        ref = self.buffer.write_payload(sample_id, data)
        entry = SampleIndexEntry(sample_id=sample_id, behavior_version=behavior_version,
                                 payload_ref=ref, tokens=tokens,
                                 provenance=None if provenance is None else dict(provenance),
                                 sampler_round=None if sampler_round is None else dict(sampler_round))
        self.buffer.append(entry)
        return entry


