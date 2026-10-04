"""Metadata extraction that runs *inside* the Miles rollout process (design D3).

Upstream rollout functions call, at the end of every training rollout::

    rollout_sample_filter_path(args, data)                  # the kept groups
    rollout_all_samples_process_path(args, all_samples, data_source)

``record_trained_groups`` remembers which groups were actually kept (the
dynamic-sampling filter plus the batch-size cap decide this, and only ``data``
reflects the cap), without modifying them. ``extract_rollout_metadata`` then
reduces every group to a small JSON-able record (ids, policy token, reward
summary, token counts) and ships it to a sink; tokens, logprobs and tensors
never leave the rollout process.

Sinks (``YETO_ROLLOUT_META_SINK``):

* ``ray:<actor_name>`` (default ``ray:yeto_rollout_meta``) -- a named Ray actor
  created by :class:`~.rollout.RayMetadataSink` in the driver;
* ``dir:<path>`` -- one atomic JSON file per rollout (single-node / tests).

The same sink carries the other direction for one value: the driver-side
``MilesRolloutPool`` stores the published policy token
(``yeto:<rollout_id>:<policy_tensor_hash>``) before each rollout, and
``policy_buffer_filter`` (``--buffer-filter-path``) reads it inside the rollout
process to keep only complete groups of exactly that policy -- the ports
equivalent of legacy ``_complete_group_for_policy`` group-reuse filtering.

Import-light: nothing here imports miles or torch.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

META_SINK_ENV = "YETO_ROLLOUT_META_SINK"
DEFAULT_SINK_ACTOR = "yeto_rollout_meta"
DEFAULT_SINK = f"ray:{DEFAULT_SINK_ACTOR}"
METADATA_SCHEMA = "yeto-rollout-meta-v1"
_TRAINED_ATTR = "_yeto_trained_group_keys"
_BOUNDED_FILTER_STATE_ATTR = "_yeto_bounded_filter_state"
# Per-round algorithm counters reported by rollout-side algorithm code AFTER
# the all-samples hook ran (e.g. the rl-algo-seq-and-adv reward dispatcher's
# non-zero advantage count, computed in reward post-processing). They travel
# as a separate sink record keyed by rollout_id and are merged into that
# rollout's metadata by the driver-side reader, so they are never attributed
# to the next round.
ROUND_META_SCHEMA = "yeto-rl-round-metadata-v1"
ROUND_METADATA_KEYS = frozenset({"nonzero_advantages"})


def current_round_id(samples: Sequence[Any] = (), sink: str | None = None) -> int | None:
    """The training round of the rollout being processed.

    Authoritative source: the policy token the driver published for this
    rollout (``yeto:<rollout_id>:<hash>``). ``Sample.rollout_id`` is NOT used
    when a token exists: in multi-segment agentic rollouts it is a trajectory
    key (Miles ``agentic_tool_call.py``), not the round. Without a token (unit
    fixtures, legacy) the first sample's ``rollout_id`` is the fallback.
    """
    try:
        token = current_policy_token(sink)
    except ImportError:  # no Ray in this process (unit fixtures): no published token
        token = None
    if token:
        from yeto.rl.core import parse_policy_snapshot_token

        return parse_policy_snapshot_token(token)[0]
    return next(
        (s.rollout_id for s in samples if getattr(s, "rollout_id", None) is not None), None
    )


def record_round_metadata(
    args: Any, samples_or_rollout_id: Any, *, sink: str | None = None, **counts: int
) -> None:
    """Rollout side: report this rollout's per-round counters (call once per rollout)."""

    del args
    unknown = sorted(set(counts) - ROUND_METADATA_KEYS)
    if unknown:
        raise RuntimeError(f"unknown per-round metadata keys {unknown}")
    for key, value in counts.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"per-round metadata {key} must be a non-negative int")
    if isinstance(samples_or_rollout_id, int) and not isinstance(samples_or_rollout_id, bool):
        rollout_id = samples_or_rollout_id
    else:
        rollout_id = current_round_id(_flat(samples_or_rollout_id), sink)
    if rollout_id is None:
        raise RuntimeError("per-round metadata needs the rollout id")
    put_to_sink({"schema": ROUND_META_SCHEMA, "rollout_id": int(rollout_id), **counts}, sink)
POLICY_TOKEN_FILE = "policy-token"
_REUSABLE_STATUSES = frozenset({"completed", "truncated"})


def _flat(group: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    for item in group:
        if isinstance(item, (list, tuple)):
            out.extend(item)
        else:
            out.append(item)
    return out


def _group_key(group: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(getattr(s, "index", None) for s in _flat(group))


def record_trained_groups(args: Any, data: Sequence[Sequence[Any]]) -> None:
    """``--rollout-sample-filter-path`` hook: remember the kept groups.

    Shared by rl-algo-grpo-knobs (sample filters, D7) and the rl-infra-spec
    3.6 ledger (alignment A2/F5). Spec-selected sample filters run first
    (``yeto.rl.algos.sample_filters``: overlong filter sets
    ``remove_sample=True`` on truncated samples; default config: untouched).
    Filtered samples are the ledger's terminal ``filtered``. Over-sampling
    leaves no reusable remainder: Miles does not return surplus kept groups
    to its buffer (``sglang_rollout.py:505-510``); only samples aborted under
    ``--partial-rollout`` go back.
    """

    from yeto.rl.algos.sample_filters import apply_sample_filters

    apply_sample_filters(args, data)
    setattr(args, _TRAINED_ATTR, {_group_key(group) for group in data})


def _reward(args: Any, sample: Any) -> float:
    getter = getattr(sample, "get_reward_value", None)
    value = getter(args) if callable(getter) else getattr(sample, "reward", None)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value


def _versions(sample: Any) -> set[str]:
    versions: set[str] = set()
    for call in getattr(sample, "weight_versions", None) or ():
        for span in getattr(call, "spans", None) or ():
            versions.add(str(span.version))
    if not versions:
        meta = getattr(sample, "metadata", None) or {}
        legacy = meta.get("weight_version")
        if legacy is not None:
            versions.add(str(legacy))
    return versions


def _status(sample: Any) -> str:
    status = getattr(sample, "status", None)
    return str(getattr(status, "value", status) or "")


def group_record(args: Any, group: Sequence[Any]) -> dict[str, Any]:
    samples = _flat(group)
    indices = [getattr(s, "index", None) for s in samples]
    group_index = getattr(samples[0], "group_index", None) if samples else None
    if group_index is None:
        # the 3.6 ledger keys groups on Miles' monotonic sample_group_index;
        # a sample index is not a group identity (fail closed)
        raise RuntimeError(
            f"rollout group without group_index (sample indices {indices}); "
            "the yeto ledger needs Miles' data-source group_index"
        )
    versions: set[str] = set()
    for s in samples:
        versions |= _versions(s)
    if len(versions) == 1:
        token = next(iter(versions))
    elif not versions:
        token = ""
    else:  # mixed policies inside one group can never match one snapshot
        token = "mixed:" + "|".join(sorted(versions))
    rewards = [_reward(args, s) for s in samples]
    finite = [r for r in rewards if math.isfinite(r)]
    return {
        "group_id": f"g{group_index}",
        "sample_ids": [f"s{i}" for i in indices],
        "policy_token": token,
        "reward_mean": statistics.fmean(finite) if finite else math.nan,
        "reward_std": statistics.pstdev(finite) if finite else math.nan,
        "token_count": int(
            sum(int(getattr(s, "effective_response_length", None) or getattr(s, "response_length", 0) or 0) for s in samples)
        ),
        "aborted": any(_status(s) == "aborted" for s in samples),
        "_key": list(_group_key(group)),
    }


def build_metadata(
    args: Any, all_samples: Iterable[Sequence[Any]], sink: str | None = None
) -> dict[str, Any]:
    trained = getattr(args, _TRAINED_ATTR, None)
    if trained is None:
        raise RuntimeError(
            "rollout metadata hook ran without record_trained_groups; "
            "--rollout-sample-filter-path must be the yeto recorder"
        )
    from yeto.rl.algos.sample_filters import group_filtered_samples, metadata_fields

    extra = metadata_fields(args)
    sample_filter_counts = extra.get("filtered_samples")
    all_samples = list(all_samples)  # iterated twice (groups, then harness counters)
    tool_wait = 0.0
    rollout_id = None
    groups, filtered, aborted = [], 0, 0
    for group in all_samples:
        samples = _flat(group)
        if not samples:
            continue
        if rollout_id is None:
            rollout_id = current_round_id(samples, sink)
        record = group_record(args, group)
        tool_wait += sum(float(getattr(x, "non_generation_time", 0.0) or 0.0) for x in samples)
        aborted += int(record.pop("aborted"))
        key = tuple(record.pop("_key"))
        if key in trained:
            if sample_filter_counts:
                # terminal ledger state ``filtered`` (alignment A2/F5)
                record["filtered_samples"] = group_filtered_samples(group)
            groups.append(record)
        else:
            filtered += 1
    if len(groups) != len(trained):
        raise RuntimeError(
            f"trained groups ({len(trained)}) not all present in all_samples ({len(groups)} matched)"
        )
    groups.sort(key=lambda g: g["group_id"])
    payload = {
        "schema": METADATA_SCHEMA,
        "rollout_id": rollout_id,
        "groups": groups,
        "completed": len(groups),
        "filtered": filtered,
        "aborted": aborted,
        "trained_sample_indices": sorted(
            int(i[1:]) for g in groups for i in g["sample_ids"] if i[1:].lstrip("-").isdigit()
        ),
        **extra,
    }
    if tool_wait > 0:
        # 1.7: time trajectories spent outside generation (tool calls), summed
        # over every generated sample (Miles Sample.non_generation_time).
        # Absent when no sample reported any: the default key set is unchanged.
        payload["tool_wait_seconds"] = tool_wait
    harness = harness_counters(all_samples)
    if harness:  # IR-3/IR-4: absent when no sample reported any (old key set kept)
        payload.update(harness)
    return payload


# IR-3/IR-4 sample-metadata keys written by agentic generate code (codex-harness
# codex_openenv_generate): summed per rollout into the metadata payload.
EXPECTED_POLICY_VERSION_KEY = "expected_policy_version"
POLICY_AGE_VIOLATION_KEY = "policy_age_violation"
TITO_SESSION_MISMATCH_KEY = "tito_session_mismatch"
TITO_CHAIN_BREAKS_KEY = "tito_chain_breaks"


def expected_policy_version(sample: Any = None, sink: str | None = None) -> str | None:
    """Rollout side (IR-3): the driver's target policy token for this rollout.

    The prompt sample's metadata ``expected_policy_version`` wins when the
    data source carries it; otherwise the token the driver published through
    the metadata sink (``MilesRolloutPool.generate`` -> ``set_policy_token``).
    None = the driver did not publish one (agentic generate must refuse).
    """
    meta = getattr(sample, "metadata", None) if sample is not None else None
    if isinstance(meta, dict) and meta.get(EXPECTED_POLICY_VERSION_KEY):
        return str(meta[EXPECTED_POLICY_VERSION_KEY])
    return current_policy_token(sink)


def harness_counters(all_samples: Iterable[Sequence[Any]]) -> dict[str, Any]:
    """Sum the IR-3/IR-4 per-sample counters; only keys with a non-zero total."""
    age = mismatch = 0
    breaks: dict[str, int] = {}
    for group in all_samples:
        for s in _flat(group):
            meta = getattr(s, "metadata", None)
            if not isinstance(meta, dict):
                continue
            age += int(meta.get(POLICY_AGE_VIOLATION_KEY) or 0)
            mismatch += int(meta.get(TITO_SESSION_MISMATCH_KEY) or 0)
            for reason, n in (meta.get(TITO_CHAIN_BREAKS_KEY) or {}).items():
                breaks[str(reason)] = breaks.get(str(reason), 0) + int(n or 0)
    out: dict[str, Any] = {}
    if age:
        out[POLICY_AGE_VIOLATION_KEY] = age
    if mismatch:
        out[TITO_SESSION_MISMATCH_KEY] = mismatch
    if breaks:
        out[TITO_CHAIN_BREAKS_KEY] = breaks
    return out


# --------------------------------------------------------------------------
# Sinks
# --------------------------------------------------------------------------


def _jsonable(payload: dict[str, Any]) -> str:
    def fix(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: fix(v) for k, v in value.items()}
        if isinstance(value, list):
            return [fix(v) for v in value]
        return value

    return json.dumps(fix(payload), sort_keys=True, separators=(",", ":"))


def put_to_sink(payload: dict[str, Any], sink: str | None = None) -> None:
    sink = sink or os.environ.get(META_SINK_ENV) or DEFAULT_SINK
    kind, _, target = sink.partition(":")
    encoded = _jsonable(payload)
    if kind == "dir":
        directory = Path(target)
        directory.mkdir(parents=True, exist_ok=True)
        rid = payload.get("rollout_id")
        prefix = "round" if payload.get("schema") == ROUND_META_SCHEMA else "rollout"
        name = f"{prefix}-{rid if rid is not None else 'latest'}.json"
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
        os.replace(tmp, directory / name)
        return
    if kind == "ray":
        import ray

        ray.get(ray.get_actor(target or DEFAULT_SINK_ACTOR).put.remote(encoded))
        return
    raise ValueError(f"unknown rollout metadata sink {sink!r}")


def put_policy_token(token: str, sink: str | None = None) -> None:
    """Driver side: publish the token the next rollout must be sampled from."""

    sink = sink or os.environ.get(META_SINK_ENV) or DEFAULT_SINK
    kind, _, target = sink.partition(":")
    if kind == "dir":
        directory = Path(target)
        directory.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(token)
        os.replace(tmp, directory / POLICY_TOKEN_FILE)
        return
    if kind == "ray":
        import ray

        ray.get(ray.get_actor(target or DEFAULT_SINK_ACTOR).set_token.remote(token))
        return
    raise ValueError(f"unknown rollout metadata sink {sink!r}")


def current_policy_token(sink: str | None = None) -> str | None:
    """Rollout side: the token stored by :func:`put_policy_token`, if any."""

    sink = sink or os.environ.get(META_SINK_ENV) or DEFAULT_SINK
    kind, _, target = sink.partition(":")
    if kind == "dir":
        path = Path(target) / POLICY_TOKEN_FILE
        return path.read_text(encoding="utf-8") if path.exists() else None
    if kind == "ray":
        import ray

        return ray.get(ray.get_actor(target or DEFAULT_SINK_ACTOR).get_token.remote())
    raise ValueError(f"unknown rollout metadata sink {sink!r}")


def group_matches_policy(group: Sequence[Any], token: str | None, size: int | None) -> bool:
    """Legacy ``_complete_group_for_policy`` over upstream sample versions."""

    samples = _flat(group)
    if not token or not samples or (size is not None and len(group) != size):
        return False
    for sample in samples:
        if _status(sample) not in _REUSABLE_STATUSES or _versions(sample) != {token}:
            return False
    return True


def policy_buffer_filter(args: Any, _rollout_id: Any, buffer: list, num_samples: int) -> list:
    """``--buffer-filter-path``: reuse only complete groups of the published policy.

    Groups from any other policy (or incomplete ones) are dropped from the
    buffer, exactly as legacy filtered its completed-group queue.
    """

    token = current_policy_token()
    size = getattr(args, "n_samples_per_prompt", None)
    kept = [g for g in buffer if group_matches_policy(g, token, size)]
    selected, buffer[:] = kept[:num_samples], kept[num_samples:]
    return selected


_OFFSET_ATTR = "_yeto_data_source_offset"


def submitted_groups(args: Any, data_source: Any) -> int | None:
    """Prompt groups drawn from the data source this rollout (over-sampling included).

    Miles ``generate_rollout`` submits ``over_sampling_batch_size`` groups at a
    time and aborts the ones still in flight once enough are accepted; those
    never reach ``all_samples``. The drawn count is the advance of the data
    source's ``sample_offset`` since the previous rollout (first rollout:
    unknown; an epoch wrap-around or a buffer source: unknown -> None).
    """
    source = getattr(data_source, "__self__", data_source)
    offset = getattr(source, "sample_offset", None)
    if not isinstance(offset, int) or getattr(source, "buffer", None):
        setattr(args, _OFFSET_ATTR, offset if isinstance(offset, int) else None)
        return None
    previous = getattr(args, _OFFSET_ATTR, None)
    setattr(args, _OFFSET_ATTR, offset)
    if not isinstance(previous, int) or offset < previous:
        return None
    return offset - previous


_CURSOR_FIELDS = ("sample_offset", "epoch_id", "sample_group_index", "sample_index")


def data_cursor(data_source: Any) -> tuple[dict[str, int] | None, int | None]:
    """rl-infra-spec 4.2: data source position and reuse-buffer length (cut-audit §3).

    Read in the rollout process after the rollout drew its prompts (Miles
    ``RolloutDataSource`` attributes; ``get_buffer_length`` on the buffered
    source). Missing or non-integer fields: unknown (None), never guessed.
    """
    source = getattr(data_source, "__self__", data_source)
    if source is None:
        return None, None
    cursor = {f: getattr(source, f, None) for f in _CURSOR_FIELDS}
    known = {k: int(v) for k, v in cursor.items() if isinstance(v, int) and not isinstance(v, bool)}
    length = None
    getter = getattr(source, "get_buffer_length", None)
    try:
        if callable(getter):
            length = int(getter())
        elif isinstance(getattr(source, "buffer", None), list):
            length = len(source.buffer)
    except Exception:  # noqa: BLE001 - unknown, reported as None
        length = None
    return (known if len(known) == len(_CURSOR_FIELDS) else None), length


ELASTIC_METADATA_ENV = "YETO_RL_ELASTIC_METADATA"


def elastic_metadata_enabled(args: Any) -> bool:
    """Report data cursor/buffer length only when E1/E2 (ledger, cut) is on.

    Off by default so the default metadata (and ``carried_over=None``) is
    unchanged. Set ``args.yeto_rl_elastic_metadata`` or the environment
    variable in the rollout process before it starts.
    """
    import os

    return bool(getattr(args, "yeto_rl_elastic_metadata", False)) or os.environ.get(
        ELASTIC_METADATA_ENV) == "1"


def extract_rollout_metadata(args: Any, all_samples: Any, data_source: Any = None) -> None:
    """``--rollout-all-samples-process-path`` hook."""

    try:
        payload = build_metadata(args, all_samples)
        if elastic_metadata_enabled(args):
            cursor, buffer_length = data_cursor(data_source)
            if cursor is not None:
                payload["data_cursor"] = cursor
            if buffer_length is not None:
                payload["buffer_length"] = buffer_length
        submitted = submitted_groups(args, data_source)
        if submitted is not None:
            generated = payload["completed"] + payload["filtered"]
            payload["submitted_groups"] = submitted
            # submitted but not completed when the batch filled: aborted in
            # flight (partial_rollout off -> their prompts are consumed, never
            # trained: terminal, A2/F5 'filtered' with reason aborted_in_flight)
            payload["aborted_in_flight_groups"] = max(0, submitted - generated)
        put_to_sink(payload)
    finally:
        # Reset per-rollout state: the bounded filter keys its memo on
        # ``yeto_rl_policy_version`` which legacy advanced per round; here the
        # rollout boundary is the reset point.
        from yeto.rl.algos.sample_filters import reset as _reset_sample_filters

        _reset_sample_filters(args)
        for attr in (_TRAINED_ATTR, _BOUNDED_FILTER_STATE_ATTR):
            if hasattr(args, attr):
                setattr(args, attr, None)
