"""Miles side of the cut's in-flight section (agentic-rollout-utilization 3.3/5, stage-3 prerequisite).

Under ``--rl-max-policy-age N > 0`` a cut taken at a round boundary may find
unfinished work in the rollout process:

* single-turn (stage 2, ``--partial-rollout``): groups Miles put back into its
  data buffer at the cut-off (``RolloutDataSourceWithBuffer.buffer``), with the
  tokens generated so far, their version segments and generation
  log-probabilities;
* agentic (stage 3, ``--agentic-suspend-between-turns``): trajectories suspended
  between two model turns -- live coroutines with a Codex process and a sandbox,
  known to the cut only by reference (``session_ref``).

:func:`export_in_flight` turns both into :class:`~yeto.rl.engine.version_segments.InFlightTrajectory`
entries (one per trajectory; a buffered sample also carries its full Miles
``Sample`` as ``engine_state`` so a restore can put it back);
:func:`import_in_flight` applies :func:`~yeto.rl.engine.version_segments.restore_in_flight`
on restore: within the limit and not already trained, a buffered group goes
back into the data buffer (continued by the next rollout like any carried
group); everything else is discarded and reported. A suspended agentic
trajectory never survives a restart (its process and sandbox are gone): it is
always discarded (reason ``agentic_session_lost``), never re-run from the start
under the same task id.

Both run INSIDE the rollout executor actor (``__ray_call__``, like the data
cursor reads in :mod:`.rollout`).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from yeto.rl.engine.version_segments import (
    InFlightTrajectory,
    ProvenanceError,
    TokenProvenance,
    restore_in_flight,
)

from .carry_over import response_token_versions

AGENTIC_SESSION_PREFIX = "codex-suspended:"
AGENTIC_SESSION_LOST = "agentic_session_lost"


class InFlightExportError(RuntimeError):
    """A buffered sample cannot be written into the cut (fail closed: no cut)."""


def _jsonable_sample(sample: Any) -> dict[str, Any]:
    raw = sample.to_dict() if callable(getattr(sample, "to_dict", None)) else dict(vars(sample))
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if value is None or isinstance(value, (str, int, float, bool, list, dict)):
            out[key] = value
        else:
            raise InFlightExportError(
                f"buffered sample {getattr(sample, 'index', '?')}: field {key!r} of type "
                f"{type(value).__name__} cannot be carried by a cut")
    return out


def _provenance(sample: Any) -> TokenProvenance:
    versions = response_token_versions(sample)
    logprobs = list(getattr(sample, "rollout_log_probs", None) or ())
    if not versions:
        return TokenProvenance((), ())
    if None in versions or len(logprobs) != len(versions):
        raise InFlightExportError(
            f"buffered sample {getattr(sample, 'index', '?')}: version or generation "
            "log-probability of a response token unknown")
    return TokenProvenance(tuple(int(v) for v in versions),
                           tuple(min(float(p), 0.0) for p in logprobs))


def _flat(group: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    for item in group:
        out.extend(_flat(item) if isinstance(item, list) else [item])
    return out


def buffered_entries(buffer: Iterable[Sequence[Any]]) -> list[dict[str, Any]]:
    """One entry per buffered sample (group order and size kept for restore)."""
    entries = []
    for group in buffer:
        samples = _flat(group)
        for position, sample in enumerate(samples):
            group_index = getattr(sample, "group_index", None)
            if group_index is None:
                raise InFlightExportError("buffered sample without group_index")
            metadata = getattr(sample, "metadata", None) or {}
            entry = InFlightTrajectory(
                trajectory_id=f"s{getattr(sample, 'index', position)}",
                group_id=f"g{group_index}",
                task_id=str(metadata.get("task_id") or metadata.get("sample_id") or f"g{group_index}"),
                provenance=_provenance(sample),
            ).to_dict()
            entry["engine_state"] = {"miles_sample": _jsonable_sample(sample),
                                     "group_position": position, "group_size": len(samples)}
            entries.append(entry)
    return entries


def agentic_entries(gates: Mapping[str, Any], inflight: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One entry per suspended agentic trajectory (reference only)."""
    entries = []
    for trajectory_id in sorted(gates):
        stage = (inflight.get(trajectory_id) or {}).get("stage")
        entry = InFlightTrajectory(
            trajectory_id=str(trajectory_id), group_id=f"agentic:{trajectory_id}",
            task_id=str(trajectory_id), provenance=TokenProvenance((), ()),
            session_ref=f"{AGENTIC_SESSION_PREFIX}{trajectory_id}",
        ).to_dict()
        if stage:
            entry["stage"] = str(stage)
        entries.append(entry)
    return entries


def _codex_suspension() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    import sys

    module = sys.modules.get("yeto.rl.harness.codex.codex_openenv_subprocess_agent_function")
    if module is None:
        return {}, {}
    state = getattr(module, "_SUSPEND_STATE", {}) or {}
    return dict(state.get("gates") or {}), dict(getattr(module, "_INFLIGHT", {}) or {})


def export_in_flight(executor: Any) -> dict[str, Any]:
    """Runs INSIDE the rollout executor actor: the in-flight entries and the
    data buffer length they account for."""
    source = getattr(executor, "data_source", None)
    source = getattr(source, "__self__", source)
    buffer = list(getattr(source, "buffer", None) or [])
    gates, inflight = _codex_suspension()
    entries = buffered_entries(buffer) + agentic_entries(gates, inflight)
    return {"entries": entries, "buffer_groups": len(buffer)}


def import_in_flight(executor: Any, entries: Sequence[Mapping[str, Any]], *, max_policy_age: int,
                     current_version: int, completed_group_ids: Iterable[str] = (),
                     sample_from_dict: Any = None) -> dict[str, Any]:
    """Runs INSIDE the rollout executor actor on restore: resumed buffered groups go
    back into the data buffer; the rest is discarded. Returns the report."""
    agentic = [e for e in entries if str(e.get("session_ref") or "").startswith(AGENTIC_SESSION_PREFIX)]
    buffered = [e for e in entries if e not in agentic]
    decision = restore_in_flight(buffered, max_policy_age=max_policy_age,
                                 current_version=current_version,
                                 completed_group_ids=completed_group_ids)
    report = decision.report()
    resumed_ids = {t.trajectory_id for t in decision.resumed}
    groups: dict[str, list[tuple[int, Any]]] = {}
    sizes: dict[str, int] = {}
    discarded_partial = 0
    for raw in buffered:
        if raw["trajectory_id"] not in resumed_ids:
            continue
        state = raw.get("engine_state") or {}
        groups.setdefault(raw["group_id"], []).append((int(state.get("group_position", 0)), state.get("miles_sample")))
        sizes[raw["group_id"]] = int(state.get("group_size", 0))
    restored: list[list[Any]] = []
    if groups:
        if sample_from_dict is None:
            from miles.utils.types import Sample

            sample_from_dict = Sample.from_dict
        for group_id, members in sorted(groups.items()):
            if len(members) != sizes[group_id] or any(m is None for _, m in members):
                # a group can only continue whole (Miles: n_samples_per_prompt per group)
                discarded_partial += len(members)
                continue
            restored.append([sample_from_dict(m) for _, m in sorted(members, key=lambda pm: pm[0])])
    source = getattr(executor, "data_source", None)
    source = getattr(source, "__self__", source)
    if restored:
        source.add_samples(restored)
    reasons = dict(report["reasons"])
    if agentic:
        reasons[AGENTIC_SESSION_LOST] = len(agentic)
    if discarded_partial:
        reasons["incomplete_group"] = discarded_partial
    return {
        "resumed": report["resumed"] - discarded_partial,
        "resumed_groups": len(restored),
        "discarded": report["discarded"] + len(agentic) + discarded_partial,
        "discarded_tokens": report["discarded_tokens"],
        "reasons": reasons,
    }


def check_export(exported: Mapping[str, Any]) -> list[str]:
    """Consistency of an export (the cut refuses on any problem)."""
    entries = exported.get("entries") or []
    out = []
    buffered_groups = {e["group_id"] for e in entries if e.get("engine_state") is not None}
    if len(buffered_groups) != int(exported.get("buffer_groups") or 0):
        out.append(f"in_flight: {exported.get('buffer_groups')} buffered groups but "
                   f"{len(buffered_groups)} exported")
    for e in entries:
        try:
            InFlightTrajectory.from_dict(e)
        except (ProvenanceError, KeyError, TypeError, ValueError) as exc:
            out.append(f"in_flight: entry {e.get('trajectory_id')!r} invalid: {exc}")
        lp = (e.get("provenance") or {}).get("logprobs")
        if isinstance(lp, list) and any(not math.isfinite(float(x)) for x in lp):
            out.append(f"in_flight: entry {e.get('trajectory_id')!r} has a non-finite log-probability")
    return out
