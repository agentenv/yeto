"""The rollout pools of a run config (``yeto run up`` / ``yeto rollout``).

A run declares its sampler pools either the old way, one pool config file::

    "pool": {"config": "<pool.json>", "interval_s": 30}

or as a list, one entry per pool (any number of providers)::

    "samplers": [
      {"name": "nc", "provider": "nc", "config": "<pool.json>", "target": 2},
      {"name": "modal", "provider": {"kind": "modal", "image": "...", "secret": "...", "bucket": "..."},
       "data_path": "r2", "target": 1, "slots": [1, 2], "shard_base": "k31-modal",
       "spec": "<sampler-spec.json>", "run_prefix": "<run prefix>", "r2_env_file": "<store env>"}
    ]

An entry names a pool config file (``config``) and/or carries pool config keys
inline (``target``, ``spec``, ``slots``, ``shard_base``, ``elastic``, ...: the
``pool.json`` format of docs/NC_POOL.md, unchanged); inline keys override the
file's. ``provider`` is a kind (``nc`` | ``daytona`` | ``sky`` | ``modal``) or a full
provider section. Entry-only keys: ``name`` (default: the provider kind),
``interval_s`` (the pool loop's tick, default 30), ``usd_per_h`` (one node's price
for the run's spend estimate; default ``budget.nc_usd_per_h``), ``data_path``
(``r2``, the default: buffer shards and policy releases in the run store; or
``syncer``: weights from the syncer broadcast and samples through the relay, the
sampler spec says how). An entry that changes nothing of its file uses the file
as it is; any other is written to ``<run state dir>/pools/<name>.json`` and the
pool commands run on that file.

Every pool keeps its own state (``pool-<shard_base>.json``), so ``shard_base`` is
unique across the pools of a run; all pools of a run use one data path (a trainer
fed by both R2 buffer shards and the relay is not verified).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATA_PATHS = ("r2", "syncer")
ENTRY_KEYS = frozenset({"name", "provider", "config", "interval_s", "usd_per_h", "data_path"})
PROVIDER_KINDS = ("nc", "daytona", "sky", "modal")


@dataclass(frozen=True)
class PoolRef:
    name: str
    config: str  # the pool config file the pool commands run on
    interval_s: float = 30.0
    usd_per_h: float | None = None
    data_path: str = "r2"

    def load(self) -> dict[str, Any]:
        return json.loads(Path(self.config).read_text())

    def kind(self) -> str:
        from yeto.launch.nc.provider import provider_kind  # noqa: PLC0415

        try:
            return provider_kind(self.load())
        except (OSError, ValueError):
            return "?"


def _provider_section(value: Any, base: dict[str, Any] | None) -> dict[str, Any]:
    """The pool's provider section: a full section as it is; a kind keeps the pool
    file's section (``base``: the file's config, None without a file) and must match it."""
    if isinstance(value, dict):
        return dict(value)
    kind = str(value)
    if kind not in PROVIDER_KINDS:
        raise ValueError(f"samplers: provider {kind!r} is not one of {', '.join(PROVIDER_KINDS)} "
                         "(or a provider section)")
    if base is not None:
        from yeto.launch.nc.provider import provider_kind  # noqa: PLC0415

        have = provider_kind(base)
        if have != kind:
            raise ValueError(f"samplers: provider {kind!r} but the pool config's provider is {have!r}")
        return dict(base.get("provider") or {})
    return {} if kind == "nc" else {"kind": kind}


def resolve(cfg: dict[str, Any], state_dir: Path) -> list[PoolRef]:
    """The pools of a run config, in order. Raises ValueError on a bad declaration."""
    if "samplers" in cfg and "pool" in cfg:
        raise ValueError("run config: set 'samplers' (a list of pools) or 'pool' (one pool config), not both")
    if "pool" in cfg:
        pool = cfg["pool"] or {}
        if "config" not in pool:
            raise ValueError("run config: pool.config (the pool config file) is required")
        return [PoolRef("pool", os.path.expanduser(str(pool["config"])), float(pool.get("interval_s", 30)))]
    entries = cfg.get("samplers")
    if not isinstance(entries, list) or not entries:
        raise ValueError("run config: 'samplers' must be a non-empty list of pools (or use 'pool')")
    refs: list[PoolRef] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"samplers[{i}] must be an object")
        refs.append(_entry(cfg, entry, i, state_dir))
    names = [r.name for r in refs]
    if len(set(names)) != len(names):
        raise ValueError(f"samplers: pool names repeat {names}; set 'name' on each")
    paths = {r.data_path for r in refs}
    if len(paths) > 1:
        raise ValueError(f"samplers: one data path per run (got {sorted(paths)}): a trainer that reads R2 buffer "
                         "shards and the syncer relay at once is not verified")
    bases: dict[str, str] = {}
    for ref in refs:
        try:
            base = str(ref.load().get("shard_base"))
        except (OSError, ValueError):
            continue
        if base in bases:
            raise ValueError(f"samplers: pools {bases[base]!r} and {ref.name!r} share shard_base {base!r} "
                             "(each pool keeps its state and shards under its own shard_base)")
        bases[base] = ref.name
    return refs


def _entry(cfg: dict[str, Any], entry: dict[str, Any], i: int, state_dir: Path) -> PoolRef:
    inline = {k: v for k, v in entry.items() if k not in ENTRY_KEYS}
    path = os.path.expanduser(str(entry["config"])) if entry.get("config") else None
    base = json.loads(Path(path).read_text()) if path else {}
    doc = dict(base)
    if "provider" in entry:
        doc["provider"] = _provider_section(entry["provider"], base if path else None)
        if doc["provider"] == {} and "provider" not in base:
            doc.pop("provider")
    doc.update(inline)
    if not doc:
        raise ValueError(f"samplers[{i}]: no pool config (set 'config' or the pool keys inline)")
    from yeto.launch.nc.provider import provider_kind  # noqa: PLC0415

    kind = provider_kind(doc)
    name = str(entry.get("name") or kind)
    data_path = str(entry.get("data_path", doc.get("data_path", "r2")))
    if data_path not in DATA_PATHS:
        raise ValueError(f"samplers[{i}]: data_path {data_path!r} is not one of {DATA_PATHS}")
    if data_path != "r2":
        doc["data_path"] = data_path
    if data_path == "syncer":
        if "syncer" not in doc and cfg.get("syncer"):
            doc["syncer"] = dict(cfg["syncer"])  # the run's syncer (address, key_env) for every pool
        if not (doc.get("syncer") or {}).get("address"):
            raise ValueError(f"samplers[{i}]: data_path syncer needs syncer.address (on the pool or the run)")
    if doc == base and path:
        target = path
    else:
        out = state_dir / "pools" / f"{name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1, sort_keys=True))
        target = str(out)
    usd = entry.get("usd_per_h")
    return PoolRef(name, target, float(entry.get("interval_s", 30)), None if usd is None else float(usd), data_path)


def select(refs: list[PoolRef], *, pool: str | None = None, provider: str | None = None) -> list[PoolRef]:
    """The pools named ``pool`` and / or of provider kind ``provider``."""
    out = [r for r in refs if pool is None or r.name == pool]
    if provider is not None:
        out = [r for r in out if r.kind() == provider]
    return out
