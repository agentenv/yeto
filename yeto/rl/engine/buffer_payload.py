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
before.

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


def _encode_value(value: Any) -> Any:
    extra: dict[str, Any] = {}
    if isinstance(value, _Field):
        value, extra = value.array, value.meta
    tolist = getattr(value, "tolist", None)
    if callable(tolist) and hasattr(value, "dtype") and hasattr(value, "shape"):
        import numpy

        array = numpy.ascontiguousarray(value)
        record: dict[str, Any] = {"dtype": str(array.dtype), "shape": list(array.shape), **extra}
        if array.dtype.kind in "iu":
            stored = _narrowest_int(array)
            if stored != array.dtype:
                record = {"dtype": str(stored), _CAST: str(array.dtype), "shape": list(array.shape), **extra}
                array = array.astype(stored)
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
      ``final_weight_version`` differs (an in-place update mid-generation) has no
      per-token split and is left without spans (refused downstream as unknown).
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
    if final is not None and engine_version_number(final) != engine_version_number(version):
        return None
    tokens = len(sample.get("tokens") or ())
    response = int(sample.get("response_length") or 0)
    if response <= 0 or response > tokens:
        return None
    start = tokens - response
    return {"spans": [{"version": str(version), "abs_start": start, "abs_end": tokens}],
            "prefill_spans": [], "output_start": start, "prompt_tokens": start}


def encode_group(samples: Sequence[Mapping[str, Any]]) -> bytes:
    """One payload: the group's samples as ``Sample.to_dict()`` mappings."""
    record = {"schema": PAYLOAD_SCHEMA, "samples": [_with_field_meta(s) for s in samples]}
    return json.dumps(record, default=_encode_value, sort_keys=True, allow_nan=True).encode()


def decode_group(data: bytes) -> list[dict[str, Any]]:
    """The samples of one payload. Bytes that are not a payload (not JSON, cut
    short, not an object, another schema, ``samples`` not a list of objects) raise
    :class:`DamagedPayload`: the fault is in the stored bytes, so reading them
    again gives the same answer."""
    try:
        record = json.loads(data, object_hook=_decode_hook)
    except (ValueError, KeyError, TypeError) as error:  # not JSON, not UTF-8, a broken array record
        raise DamagedPayload(f"payload does not decode: {type(error).__name__}: {str(error)[:200]}") from error
    if not isinstance(record, Mapping):
        raise DamagedPayload(f"payload is a {type(record).__name__}, not an object")
    if record.get("schema") != PAYLOAD_SCHEMA:
        raise DamagedPayload(f"payload schema {record.get('schema')!r} != {PAYLOAD_SCHEMA}")
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


class GroupWriter:
    """Writes finished groups to the buffer: payload first, then the index entry."""

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
        ref = self.buffer.write_payload(sample_id, encode_group(samples))
        entry = SampleIndexEntry(sample_id=sample_id, behavior_version=behavior_version,
                                 payload_ref=ref, tokens=tokens,
                                 provenance=None if provenance is None else dict(provenance),
                                 sampler_round=None if sampler_round is None else dict(sampler_round))
        self.buffer.append(entry)
        return entry


