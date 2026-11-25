"""The sandbox provider each agentic data row really asks for, checked before anything
is ordered (``pool --dry-run``, ``yeto run up``'s preflight).

A row's task mapping (``prompts.task_key``, default ``metadata``) overrides the spec's
``agentic.task_defaults`` (:meth:`yeto.rl.agentic.AgentTask.from_mapping`): a row whose
``provider`` is ``boat`` sends its trial to boat even when the pool runs AgentENV. The
sampler image carries the SDK of the pool's provider only (the Modal image layer and the
NC node setup install that one), so such a trial fails at its first import on every
node. :func:`provider_problems` reads the rows (a local file, or an ``s3://`` object),
sums each row's effective provider and names every row whose provider is not the pool's,
with the module it would need. Rows that cannot be read here (a path on the node or a
Modal volume, with no local copy of the same name) are not checked here, and nothing is
printed for them; a mismatch the check finds is a problem: nothing is ordered.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REMOTE_DOCKER = "remote-docker"
# the module each provider's adapter imports (the SDK the image layer installs for it)
PROVIDER_MODULES = {"boat": "boat_sdk", "agentenv": "e2b", REMOTE_DOCKER: None}
MAX_NAMED_ROWS = 5


def pool_provider(spec: Mapping[str, Any], env: Mapping[str, Any] | None = None) -> str | None:
    """The provider the pool's sandboxes (and its image's SDK) are set up for:
    ``agentic.task_defaults.provider``, else ``agentic.setup.provider``, else the
    provider env's ``HARBOR_ENV_TYPE``, else remote-docker; None for a spec that is not
    agentic."""
    agentic = spec.get("agentic")
    if not isinstance(agentic, Mapping):
        return None
    defaults = agentic.get("task_defaults") if isinstance(agentic.get("task_defaults"), Mapping) else {}
    setup = agentic.get("setup") if isinstance(agentic.get("setup"), Mapping) else {}
    found = defaults.get("provider") or setup.get("provider") or (env or {}).get("HARBOR_ENV_TYPE")
    return str(found or REMOTE_DOCKER).strip().lower()


def read_rows(path: str, *, environ: Mapping[str, str] | None = None) -> list[Any] | None:
    """The JSON rows of ``path`` (a local file, or an ``s3://`` object read with the store
    credentials of ``environ``); None when it cannot be read here (not on this machine,
    no credentials, not JSON lines)."""
    text: str | None = None
    if path.startswith("s3://"):
        try:
            from yeto.cloud.s3_store import head_objects  # noqa: PLC0415

            bucket_path = path[len("s3://"):]
            bucket, _, key = bucket_path.partition("/")
            raw = head_objects(f"s3://{bucket}", environ=environ).read(key)
        except Exception:  # noqa: BLE001 - not readable here: not checked here
            return None
        if raw is None:
            return None
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
    elif "://" not in path:
        try:
            text = Path(os.path.expanduser(path)).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
    if text is None:
        return None
    try:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    except ValueError:
        return None


def row_providers(rows: Iterable[Any], *, task_key: str, default: str | None) -> dict[str, list[int]]:
    """``{effective provider: [row numbers (1-based)]}``: a row's own ``provider`` in its
    task mapping wins over ``default`` (``AgentTask.from_mapping``)."""
    out: dict[str, list[int]] = {}
    for n, row in enumerate(rows, start=1):
        task = row.get(task_key) if isinstance(row, Mapping) else None
        own = task.get("provider") if isinstance(task, Mapping) else None
        provider = str(own if own is not None else (default or "?")).strip().lower()
        out.setdefault(provider, []).append(n)
    return out


def _module(provider: str) -> str:
    module = PROVIDER_MODULES.get(provider)
    return f"module {module}" if module else ("a docker host" if provider == REMOTE_DOCKER else "an unknown SDK")


def provider_problems(spec: Mapping[str, Any], rows: list[Any], *, where: str, expected: str,
                      expected_from: str) -> list[str]:
    """Problems of ``rows`` (read from ``where``) whose effective provider is not
    ``expected`` (named by ``expected_from``); [] when every row agrees."""
    prompts = spec.get("prompts") if isinstance(spec.get("prompts"), Mapping) else {}
    agentic = spec.get("agentic") if isinstance(spec.get("agentic"), Mapping) else {}
    defaults = agentic.get("task_defaults") if isinstance(agentic.get("task_defaults"), Mapping) else {}
    found = row_providers(rows, task_key=str(prompts.get("task_key", "metadata")), default=defaults.get("provider"))
    out = []
    for provider, numbers in sorted(found.items()):
        if provider == expected:
            continue
        named = ", ".join(str(n) for n in numbers[:MAX_NAMED_ROWS]) + (" ..." if len(numbers) > MAX_NAMED_ROWS else "")
        out.append(f"agentic rows: {len(numbers)} of {len(rows)} row(s) of {where} run their trials on sandbox "
                   f"provider {provider!r} (row(s) {named}; the row's {prompts.get('task_key', 'metadata')}.provider "
                   f"overrides agentic.task_defaults.provider), but {expected_from} is {expected!r}: the sampler "
                   f"image carries the {expected} SDK only, so every such trial fails at once (it needs "
                   f"{_module(provider)}); fix the rows' provider or the pool's")
    return out


def pool_rows_problems(cfg: Mapping[str, Any], spec: Mapping[str, Any], *,
                       environ: Mapping[str, str] | None = None) -> list[str]:
    """``pool --dry-run``: the rows the pool's samplers read (its ``prompts_file``, else the
    spec's ``prompts.path`` when it is readable here) against the pool's provider; []
    when the spec is not agentic, the rows are not readable here, or every row agrees."""
    provider_cfg = cfg.get("provider") if isinstance(cfg.get("provider"), Mapping) else {}
    expected = pool_provider(spec, provider_cfg.get("env") if isinstance(provider_cfg.get("env"), Mapping) else None)
    if expected is None:
        return []
    prompts = spec.get("prompts") if isinstance(spec.get("prompts"), Mapping) else {}
    where = str(cfg.get("prompts_file") or prompts.get("path") or "")
    if not where:
        return []
    rows = read_rows(where, environ=environ)
    if not rows:
        return []
    return provider_problems(spec, rows, where=where, expected=expected, expected_from="the pool's provider")



def run_rows_problems(pools: Iterable[tuple[str, Mapping[str, Any], Mapping[str, Any]]], *,
                      data_files: Iterable[str] = (), environ: Mapping[str, str] | None = None) -> list[str]:
    """``yeto run up``'s preflight over every agentic pool ``(name, pool config, spec)``:
    the rows its samplers read -- its ``prompts_file``, the spec's ``prompts.path`` when
    it is readable here, else a run data file (``data_files``: the trainer's local
    ``--data``) of the same file name (the copy a Modal volume or a node holds) -- against
    the pool's provider. [] when nothing is readable here or every row agrees."""
    by_name: dict[str, str] = {}
    for path in data_files:
        by_name.setdefault(os.path.basename(str(path)), str(path))
    out: list[str] = []
    for name, cfg, spec in pools:
        provider_cfg = cfg.get("provider") if isinstance(cfg.get("provider"), Mapping) else {}
        env = provider_cfg.get("env") if isinstance(provider_cfg.get("env"), Mapping) else None
        expected = pool_provider(spec, env)
        if expected is None:
            continue
        prompts = spec.get("prompts") if isinstance(spec.get("prompts"), Mapping) else {}
        remote = str(prompts.get("path") or "")
        candidates = [str(cfg["prompts_file"])] if cfg.get("prompts_file") else []
        candidates += [remote] if remote else []
        if remote and os.path.basename(remote) in by_name:
            candidates.append(by_name[os.path.basename(remote)])
        for where in candidates:
            rows = read_rows(where, environ=environ)
            if not rows:
                continue
            out += [f"{name}: {p}" for p in provider_problems(spec, rows, where=where, expected=expected,
                                                              expected_from=f"pool {name}'s provider")]
            break
    return out
