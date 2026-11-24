"""The sampler spec of one pool member: the base engine-sampler spec with the
member's shard (:func:`sampler_spec`), the run knobs laid over it
(:func:`apply_sampler_overrides`, :func:`dispense_config`, :func:`merge_breakers`)
and its backlog cap (:func:`backlog_groups`). Shared by every pool route (the
SSH + docker route of :mod:`.ssh_sampler`, Daytona, Modal, own SSH nodes) and by
``yeto run up``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def sampler_spec(base: Mapping[str, Any], *, sampler_id: str, shards: list[str]) -> dict[str, Any]:
    """The base engine-sampler spec with this sampler's shard and the run's shard list."""
    spec = dict(base)
    spec["sampler_id"] = sampler_id
    spec["buffer_shards"] = sorted(set(shards) | {sampler_id})
    return spec


RELOADS = ("hot", "restart")


def dispense_config(namespace: str, *, lease_s: float | None = None, max_redo: int | None = None,
                    passes: int | None = None, seed: int | None = None, holdout_file: str | None = None,
                    mix: str | None = None, labels: str | None = None,
                    by_label: bool = False) -> dict[str, Any]:
    """``prompts.dispense`` of a store-ledger sampler (``yeto.rl.engine.store_dispense``):
    every sampler of the run names the same ``namespace`` under the run's store, so
    samplers that share only the bucket share the tasks. ``holdout_file`` is read
    here (a local path) and goes into the spec as ``holdout_ids``: the sampler
    refuses rows with one of them, and the plan pins the list's digest."""
    out: dict[str, Any] = {"backend": "store", "namespace": str(namespace).strip("/")}
    if not out["namespace"] or ".." in out["namespace"].split("/"):
        raise ValueError(f"dispense namespace {namespace!r} is not a relative store path")
    for key, value in (("lease_s", lease_s), ("max_redo", max_redo), ("passes", passes), ("seed", seed)):
        if value is not None:
            out[key] = value
    if holdout_file:
        from yeto.rl.engine.store_dispense import load_holdout  # noqa: PLC0415

        ids, _digest = load_holdout(holdout_file)
        out["holdout_ids"] = sorted(ids)
        out["require_holdout"] = True
    if mix:
        out["mix"] = [w.strip() for w in mix.split(",") if w.strip()]
    if labels:
        out["labels"] = [x.strip() for x in labels.split(",") if x.strip()]
    if by_label:
        out["by_label"] = True
    return out


def merge_breakers(base: Mapping[str, Any], extra: Mapping[str, Any], where: str = "breakers") -> dict[str, Any]:
    """``extra`` merged into the ``breakers`` section ``base``: a value both set has to be the
    same (else ValueError, naming the key); neither input is changed."""
    out = json.loads(json.dumps(dict(base)))
    for key, value in extra.items():
        at = f"{where}.{key}"
        if key not in out:
            out[key] = json.loads(json.dumps(value))
        elif isinstance(out[key], Mapping) and isinstance(value, Mapping):
            out[key] = merge_breakers(out[key], value, at)
        elif out[key] != value or isinstance(out[key], bool) != isinstance(value, bool):
            raise ValueError(f"{at}: {out[key]!r} in the spec, {value!r} from the run config")
    return out


def apply_sampler_overrides(spec: Mapping[str, Any], *, data_slot: str | None = None,
                            drop_zero_variance: bool | None = None, thinking_effort: str | None = None,
                            reload: str | None = None, n: int | None = None,
                            max_tokens: int | None = None, continues_shard: str | None = None,
                            dispense: Mapping[str, Any] | None = None,
                            breakers: Mapping[str, Any] | None = None,
                            island: int | None = None, until_trainers_end: int | None = None,
                            dispense_store: str | None = None,
                            sandbox_store: str | None = None, shard_offset: int | None = None) -> dict[str, Any]:
    """The run knobs of one sampler laid over a base spec (None: keep the base value),
    with the same meaning as ``yeto engine-sampler-modal``: ``data_slot`` "I/N" (this
    sampler's prompt rows), ``drop_zero_variance`` (``drop_zero_variance_groups``),
    ``thinking_effort`` (the chat template kwarg), ``reload`` ``hot`` (in-place reload
    with expert-file updates and the probe) or ``restart``, ``n`` and ``max_tokens``
    (sampling); ``continues_shard``: the shard this replacement sampler continues;
    ``dispense``: ``prompts.dispense`` (:func:`dispense_config`; excludes ``data_slot``);
    ``breakers``: merged into the spec's ``breakers`` section (:func:`merge_breakers`: the
    run's sampler / harness settings, ``yeto run up``). ``island`` (the fleet island this
    sampler feeds), ``until_trainers_end`` (stop after that many trainer end records of its
    store), ``dispense_store`` (the store ledger's root), ``sandbox_store`` (the store
    holding the sandbox readiness manifest, ``yeto.sandbox.gate``) and ``shard_offset``
    (added to the shard index of a records relay / tunnel port range the pool shares) are
    the spec keys of the same name, set by ``yeto run up`` for the pools of an island
    fleet (``yeto.rl.fleet``).
    Refuses a ``max_tokens`` that does not fit ``max_model_len`` with the spec's
    ``max_prompt_tokens``."""
    from yeto.rl.engine.data_split import DataSlot

    out = json.loads(json.dumps(spec))
    prompts = out.setdefault("prompts", {})
    sampling = out.setdefault("sampling", {})
    if data_slot:
        DataSlot.parse(data_slot)
        prompts["data_slot"] = data_slot
    if dispense:  # the tasks come from the store ledger, not a slot
        if data_slot or prompts.get("data_slot"):
            raise ValueError("a dispensed sampler takes no data slot")
        prompts["dispense"] = dict(dispense)
    if breakers:
        out["breakers"] = merge_breakers(out.get("breakers") or {}, breakers)
    for key, value in (("island", island), ("until_trainers_end", until_trainers_end),
                       ("dispense_store", dispense_store), ("sandbox_store", sandbox_store),
                       ("shard_offset", shard_offset)):
        if value is not None:  # an island fleet's pool (yeto run up): unset keeps the spec as it is
            out[key] = value
    if continues_shard:  # a replacement: continue that shard's prompts (sampler_resume handoff)
        if continues_shard == out.get("sampler_id"):
            raise ValueError("continues_shard names the sampler's own shard")
        out["continues_shard"] = continues_shard
    if drop_zero_variance is not None:
        out["drop_zero_variance_groups"] = bool(drop_zero_variance)
    if thinking_effort is not None:
        prompts["chat_template_kwargs"] = dict(prompts.get("chat_template_kwargs") or {},
                                               thinking_effort=thinking_effort)
        if out.get("agentic"):  # an agentic round renders with agentic.template_kwargs only
            out["agentic"]["template_kwargs"] = dict(out["agentic"].get("template_kwargs") or {},
                                                     thinking_effort=thinking_effort)
    if reload is not None:
        if reload not in RELOADS:
            raise ValueError(f"reload {reload!r} is not one of {RELOADS}")
        hot = reload == "hot"
        engine = out.setdefault("engine", {})
        engine.setdefault("server", {})["in_place_reload"] = hot
        if engine.get("adapter") != "vllm-server":  # SGLang's expert-file update and probe; VllmOptions has none
            options = engine.setdefault("options", {})
            options["expert_file_update"] = "auto" if hot else "off"
            options["expert_update_probe"] = hot
    if n is not None:
        if int(n) < 1:
            raise ValueError("n must be >= 1")
        sampling["n"] = int(n)
    if max_tokens is not None:
        sampling["max_tokens"] = int(max_tokens)
    limit = ((out.get("engine") or {}).get("server") or {}).get("max_model_len")
    if limit and "max_tokens" in sampling and int(sampling["max_tokens"]) + int(out.get("max_prompt_tokens") or 0) > int(limit):
        raise ValueError(f"max_tokens {sampling['max_tokens']} + max_prompt_tokens "
                         f"{out.get('max_prompt_tokens')} exceed max_model_len {limit}")
    return out


def backlog_groups(*, groups_per_round: int, synced: bool, trainers: int = 1, samplers: int = 1,
                   fragments: int = 1, local_rounds: int = 1, pipeline: int = 0,
                   explicit_rounds: int | None = None) -> int:
    """``max_backlog_groups`` of the sampler, sized by :mod:`yeto.rl.engine.sample_roles`
    as the launcher sizes it: a plain (no-sync) sampler takes
    ``sampler_backlog_rounds`` (the operator's value or the default); a sampler of a
    full-parameter synced group takes the stall-free minimum
    (``stall_free_backlog_rounds``) plus the default margin, and an explicit value
    below the minimum is refused."""
    from yeto.rl.engine import sample_roles

    if not synced:
        return sample_roles.sampler_backlog_rounds(explicit_rounds) * groups_per_round
    need = sample_roles.stall_free_backlog_rounds(trainers_fed=trainers, sharing=samplers, fragments=fragments,
                                                  local_rounds=local_rounds, pipeline=pipeline)
    if explicit_rounds is None:
        return (need + sample_roles.DEFAULT_MAX_BACKLOG_ROUNDS) * groups_per_round
    if explicit_rounds < need:
        raise ValueError(f"backlog {explicit_rounds} rounds stalls the group: at least {need} rounds")
    return explicit_rounds * groups_per_round
