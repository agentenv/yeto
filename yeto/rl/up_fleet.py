"""The per-island preflight of an island fleet's ``yeto run up`` (:mod:`yeto.rl.fleet`).

A fleet compiles to one synced-group trainer launch and every island's pools
(:func:`yeto.rl.fleet.compile_fleet`); the single-trainer checks of ``run up`` then
run per pool as for any run. What only a fleet has is checked here, island by
island, from the :class:`~yeto.rl.fleet.FleetPlan` and the pool configs:

* :func:`shard_problem`: the island's trainer reads exactly the shards of its
  ``--rl-island-buffer-shards k:...``; every shard a pool of island k may order
  (``<shard_base>-<n>``, n below its ``shard_limit``) must be in that list;
* :func:`store_problem`: a pool of island k writes ``<store>/trainer<k>`` (the
  store env bucket and its ``run_prefix``), not the run store, not another
  island's sub-directory (the single-trainer store check accepts any
  ``trainer<k>``);
* :func:`rollout_fields_problems`: every pool of an island records the same
  ``rollout_fields`` (one trainer trains on all of them), and what the trainer
  trains on (R3 routed experts, score_centering top-K log-probs) is in them;
* :func:`release_form_lines`: the pool's engine loads releases of the fleet's
  training form (full parameters, and partial experts of the generic recipe: a
  whole HF checkpoint per release, reloaded whole; only the ``kimi_k31`` recipe
  releases the trained experts with a manifest, laid over the engine's base);
* :func:`release_every_warnings`: what a release costs each island's pools in
  sampling (fetch, load and, on a draining engine, the drain), estimated from the
  release size and the engine, and the ``--rl-release-every`` that keeps it
  within one step when the island's step time is known;
* :func:`syncer_lines`: the syncer's region against the islands' (the launch's
  own rule, :mod:`yeto.launch.syncer_region`) and its price;
* :func:`budget_lines`: the fleet's hourly estimate (pools, each island's
  trainer, the syncer) over the wall clock, against ``budget.usd``.

Import-light: no torch, ray or miles; only local files are read.
"""

from __future__ import annotations

import io
import math
from collections.abc import Sequence
from typing import Any

SERVER_ENGINES = ("vllm", "sglang")  # external engines that load a release from the store
CPU_ENGINES = ("fake",)  # the CPU engine of the tests and dry rehearsals
KIMI_K31 = "kimi_k31"  # the one recipe whose partial-experts release is an expert overlay with a manifest
# the release cost estimate (release_every_warnings): a bf16 release of every parameter,
# fetched from the store at about 200 MB/s and loaded by the engine at about 1 GB/s
RELEASE_BYTES_PER_PARAM = 2
FETCH_BYTES_PER_S = 200e6
LOAD_BYTES_PER_S = 1e9
MIN_SUGGESTED_RELEASE_EVERY = 2
POLICY_AGE_MARGIN = 2  # the suggested --rl-max-policy-age is at least release_every + this


def island_label(island: Any) -> str:
    return f"island {island.index} ({island.name})"


def _argv_values(argv: Sequence[str], flag: str) -> list[str]:
    out = []
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            out.append(str(argv[i + 1]))
        elif a.startswith(flag + "="):
            out.append(a.split("=", 1)[1])
    return out


def _argv_value(argv: Sequence[str], flag: str) -> str | None:
    values = _argv_values(argv, flag)
    return values[-1] if values else None


def island_shards(trainer_argv: Sequence[str]) -> dict[int, list[str]]:
    """``--rl-island-buffer-shards k:a,b`` of the trainer argv: island -> shards, as
    the launcher reads them (names stripped, empty ones dropped)."""
    out: dict[int, list[str]] = {}
    for value in _argv_values([str(a) for a in trainer_argv], "--rl-island-buffer-shards"):
        index, sep, names = value.partition(":")
        if sep and index.strip().isdigit():
            out[int(index)] = [s.strip() for s in names.split(",") if s.strip()]
    return out


def island_release_every(island: Any, trainer_argv: Sequence[str]) -> int:
    """The island's release cadence: its own ``--rl-release-every`` override, else
    ``fleet.trainer_argv``'s, else 1 (every local version)."""
    if island.release_every is not None:
        return int(island.release_every)
    value = _argv_value([str(a) for a in trainer_argv], "--rl-release-every")
    return int(value) if value and value.isdigit() and int(value) > 0 else 1


def island_max_policy_age(island: Any, trainer_argv: Sequence[str]) -> int | None:
    if island.max_policy_age is not None:
        return int(island.max_policy_age)
    value = _argv_value([str(a) for a in trainer_argv], "--rl-max-policy-age")
    return int(value) if value and value.isdigit() else None


def _where(plan: Any, pool: str) -> tuple[Any, str]:
    island = plan.island_of_pool(pool)
    return island, (f"{island_label(island)}, pool {pool}" if island is not None else f"pool {pool}")


def shard_problem(plan: Any, trainer_argv: Sequence[str], pool: str, pool_cfg: dict[str, Any]) -> str | None:
    """The island's trainer reads only the shards of its ``--rl-island-buffer-shards``:
    a shard of the pool outside them is never trained on. (The caller names the
    island and the pool: :func:`pool_where`.)"""
    from yeto.launch.nc import cli as nc_cli  # noqa: PLC0415

    island = plan.island_of_pool(pool)
    if island is None:
        return "in no island of the fleet plan"
    named = island_shards(trainer_argv).get(island.index)
    if named is None:
        return (f"the trainer argv has no --rl-island-buffer-shards {island.index}:...: island {island.index}'s "
                "trainer would read no shard of its pools")
    limit = nc_cli.pool_shard_limit(pool_cfg)
    if limit is None:
        return "no shard_limit: a scale up or a replacement past it writes groups nobody trains on"
    base = str(pool_cfg.get("shard_base"))
    missing = [f"{base}-{n}" for n in range(limit) if f"{base}-{n}" not in set(named)]
    if missing:
        return (f"pool shard_limit {limit} but island {island.index}'s --rl-island-buffer-shards lacks "
                f"{missing[:5]}{' ...' if len(missing) > 5 else ''}: those nodes' groups would never be trained on")
    return None


def store_problem(plan: Any, pool: str, pool_cfg: dict[str, Any]) -> str | None:
    """A pool of island k writes ``<store>/trainer<k>``: the store its island's trainer
    reads. (The caller names the island and the pool: :func:`pool_where`.)"""
    from yeto.rl.up_consistency import _norm_store, pool_store  # noqa: PLC0415

    island = plan.island_of_pool(pool)
    if island is None:
        return "in no island of the fleet plan"
    written = pool_store(pool_cfg)
    if written is None or _norm_store(written) == _norm_store(island.store):
        return None
    return (f"its samplers write to {written} but island {island.index}'s trainer reads {island.store}: no group "
            "would reach it (fix the pool's run_prefix / store env bucket)")


def pool_where(plan: Any, pool: str) -> str:
    """``island k (name), pool <pool>``: where a per-pool preflight line of a fleet points."""
    return _where(plan, pool)[1]


def _fields(spec: dict[str, Any] | None) -> dict[str, Any] | None:
    from yeto.rl.adapters.sampler_engines.from_spec import rollout_fields  # noqa: PLC0415

    if spec is None:
        return None
    try:
        return rollout_fields(spec.get("rollout_fields") if isinstance(spec.get("rollout_fields"), dict) else None)
    except ValueError:
        return None  # the pool dry-run refuses the section


def trainer_rollout_wants(trainer_argv: Sequence[str]) -> dict[str, Any]:
    """What the trainers train on from the rollout: ``routed_experts`` (R3) and
    ``top_logprobs_num`` (score_centering's K; 0: none)."""
    from yeto.rl.algos.score_centering import DEFAULT_TOP_LOGPROBS  # noqa: PLC0415
    from yeto.rl.up_consistency import trainer_args, trainer_r3  # noqa: PLC0415

    argv = [str(a) for a in trainer_argv]
    args = trainer_args(argv)
    want_k = 0
    if _argv_value(argv, "--rl-loss") == "score_centering":
        want_k = int(_argv_value(argv, "--rl-rollout-top-logprobs-num") or DEFAULT_TOP_LOGPROBS)
    return {"routed_experts": bool(args is not None and trainer_r3(args)), "top_logprobs_num": want_k}


def _show(fields: dict[str, Any]) -> str:
    return ", ".join(f"{k} {str(v).lower() if isinstance(v, bool) else v}" for k, v in sorted(fields.items()))


def rollout_fields_problems(plan: Any, trainer_argv: Sequence[str],
                            pools: Sequence[tuple[str, dict[str, Any] | None]]) -> list[str]:
    """Per island: its pools' ``rollout_fields`` (``(compiled pool name, rendered
    spec)``; a spec that does not load is left to the pool dry-run) agree with each
    other and carry what the trainer trains on. Each problem names the island and pool."""
    want = trainer_rollout_wants(trainer_argv)
    specs = dict(pools)
    out = []
    for island in plan.islands:
        seen = [(p.name, _fields(specs.get(p.name))) for p in island.pools]
        seen = [(name, f) for name, f in seen if f is not None]
        for name, fields in seen:
            if want["routed_experts"] and not fields["routed_experts"]:
                out.append(f"{island_label(island)}, pool {name}: the trainer replays the rollout routing (R3) but "
                           "the pool's sampler spec has no rollout_fields.routed_experts: set it true")
            if want["top_logprobs_num"] and fields["top_logprobs_num"] != want["top_logprobs_num"]:
                out.append(f"{island_label(island)}, pool {name}: rollout_fields.top_logprobs_num "
                           f"{fields['top_logprobs_num'] or 'unset'} != the trainer's --rl-rollout-top-logprobs-num "
                           f"{want['top_logprobs_num']} (--rl-loss score_centering trains on them): set it to "
                           f"{want['top_logprobs_num']}")
        shapes = {_show(f) for _name, f in seen}
        if len(shapes) > 1:
            listed = "; ".join(f"{name}: {_show(f)}" for name, f in seen)
            out.append(f"{island_label(island)}: its pools record different rollout_fields ({listed}): one trainer "
                       "trains on every group of its island, so they must record the same fields (set the same "
                       "rollout_fields in each pool's sampler spec)")
    return out


def _engine(spec: dict[str, Any] | None) -> tuple[str | None, dict[str, Any]]:
    engine = (spec or {}).get("engine")
    if not isinstance(engine, dict):
        return None, {}
    adapter = engine.get("adapter")
    server = engine.get("server") if isinstance(engine.get("server"), dict) else {}
    return (str(adapter).removesuffix("-server") if adapter else None), server


def trainer_recipe(trainer_argv: Sequence[str]) -> str:
    """``--rl-model-recipe`` of the trainers as the ``yeto launch`` parser reads it
    (``generic`` when unset or when the argv does not parse)."""
    from yeto.rl.up_consistency import trainer_args  # noqa: PLC0415

    args = trainer_args([str(a) for a in trainer_argv])
    return str(getattr(args, "rl_model_recipe", None) or "generic") if args is not None else "generic"


def release_form_lines(plan: Any, pools: Sequence[tuple[str, dict[str, Any] | None]],
                       trainer_argv: Sequence[str] = ()) -> tuple[list[str], list[str]]:
    """``(problems, notes)``: whether each pool's engine loads releases of the fleet's
    training form. An external vLLM / SGLang server reloads each release (in place
    with ``server.in_place_reload``, else by a restart per release); another engine
    adapter loads no store release.

    What a release is: full parameters, and partial experts of the generic recipe,
    release a whole HF safetensors checkpoint each time
    (``release_plugin.save_hf_pretrained``) with no release manifest and no expert
    overlay, which the pool reloads whole; ``--rl-release-delta-full-every`` applies
    to them. Only the ``kimi_k31`` recipe releases the trained experts with a
    manifest, laid over the engine's ``base_dir`` (on vLLM its expert reload loads
    the selected experts alone)."""
    form = (plan.sync or {}).get("form")
    kind = {"full parameters": "full-parameter", "partial experts": "partial-experts"}.get(form or "", "policy")
    k31 = form == "partial experts" and trainer_recipe(trainer_argv) == KIMI_K31
    problems, notes = [], []
    for name, spec in pools:
        _island, where = _where(plan, name)
        engine, server = _engine(spec)
        if spec is None or engine is None or engine in CPU_ENGINES:
            continue
        if engine not in SERVER_ENGINES:
            problems.append(f"{where}: engine adapter {engine!r} loads no {kind} release from the store (a fleet's "
                            f"pools are external {' / '.join(e + '-server' for e in SERVER_ENGINES)} engines)")
            continue
        if not server.get("in_place_reload"):
            notes.append(f"WARNING: {where}: engine.server.in_place_reload is off: every {kind} release restarts "
                         f"the {engine} server (a cold start per release); turn it on once its on-card reload "
                         "test passed")
        if form != "partial experts":
            continue
        if k31:
            notes.append(f"{where}: partial experts (kimi_k31 recipe): each release carries the trained experts "
                         "and a release manifest, laid over engine.server.base_dir (the expert overlay)"
                         + (" and reloaded expert by expert with engine.options.expert_reload"
                            if engine == "vllm" else ""))
        else:
            notes.append(f"{where}: partial experts (generic recipe): each release is a whole HF safetensors "
                         "checkpoint with no release manifest and no expert overlay, and the "
                         f"{engine} server reloads it whole; --rl-release-delta-full-every applies to it")
    return problems, notes


def release_bytes(trainer_argv: Sequence[str]) -> tuple[float | None, str]:
    """``(bytes, source)``: the size of one release estimated as the parameter count
    of the trainers' ``--model`` config x 2 bytes (bf16), the config read locally (a
    model directory, the HF cache or the built-in table,
    :func:`yeto.memory_estimate.param_breakdown`); None when no config is found."""
    from yeto import memory_estimate  # noqa: PLC0415

    model = _argv_value([str(a) for a in trainer_argv], "--model")
    if not model:
        return None, "no --model"
    try:
        _arch, repo, source, cfg = memory_estimate._resolve(model)  # noqa: SLF001
        params = memory_estimate.param_breakdown(cfg)["total"] if cfg is not None else None
    except (KeyError, TypeError, ValueError):
        params = None
    if not params:
        return None, f"no local config.json of --model {model}"
    return float(params) * RELEASE_BYTES_PER_PARAM, f"{params / 1e9:.1f}B parameters of {repo} x 2 B"


def pool_drain_s(engine: str | None, spec: dict[str, Any] | None) -> tuple[float | None, str]:
    """``(seconds, why)``: the most sampling a release's drain may take from a pool.
    vLLM reloads drain-only (no request runs while the release waits for the running
    ones and loads): up to min(``agentic.drain_timeout_s``, ``reload_max_defer_s``),
    the sampler's drain cut. SGLang with ``engine.options.live_update`` loads during
    a generation: no drain; without it, it drains as vLLM does. None: no cut (both
    unset) or a setting the pool dry-run refuses."""
    from yeto.rl.engine import agentic_round as ar  # noqa: PLC0415

    if engine == "sglang" and ar.spec_live_engine(spec or {}):
        return 0.0, "live_update: no drain"
    try:
        cut, _grace = ar.drain_settings(spec or {}, drain=True)
    except Exception:  # noqa: BLE001 - the pool dry-run reports a bad drain setting
        return None, "drain cut unreadable"
    if cut is None:
        return None, "drain with no cut (drain_timeout_s and reload_max_defer_s unset)"
    return float(cut), f"drain up to {_dur(float(cut))} (min of drain_timeout_s, reload_max_defer_s)"


def _dur(seconds: float) -> str:
    return f"{seconds:.0f} s" if seconds < 120 else f"{seconds / 60:.1f} min"


def release_every_warnings(plan: Any, trainer_argv: Sequence[str],
                           pools: Sequence[tuple[str, dict[str, Any] | None]],
                           step_s: dict[int, float] | None = None) -> list[str]:
    """What a release costs each island in sampling, and the release cadence that keeps
    it within the island's steps. Per pool of an external engine: fetching the release
    (:func:`release_bytes` at about 200 MB/s), loading it (about 1 GB/s) and the drain
    (:func:`pool_drain_s`); the island's cost is its costliest pool's. With the island's
    step time (``step_s``: island -> seconds of one local version) the suggested
    ``--rl-release-every`` is N = max(2, ceil(cost / step)), and the island's
    ``--rl-max-policy-age`` should be at least N + 2; a warning when the island
    releases more often than N. Without a step time the line reports the cost alone
    (a warning when the island releases every local version)."""
    specs = dict(pools)
    argv = [str(a) for a in trainer_argv]
    size, size_why = release_bytes(argv)
    out = []
    for island in plan.islands:
        costs = []
        for p in island.pools:
            engine = _engine(specs.get(p.name))[0]
            if engine not in SERVER_ENGINES:
                continue
            drain, drain_why = pool_drain_s(engine, specs.get(p.name))
            costs.append((p.name, engine, drain, drain_why))
        if not costs:
            continue
        move = 0.0 if size is None else size / FETCH_BYTES_PER_S + size / LOAD_BYTES_PER_S
        name, engine, drain, drain_why = max(costs, key=lambda c: c[2] if c[2] is not None else 0.0)
        cost = move + (drain or 0.0)
        bound = "at least " if size is None or drain is None else ""
        moved = (f"fetch {_dur(size / FETCH_BYTES_PER_S)} + load {_dur(size / LOAD_BYTES_PER_S)} of a "
                 f"{size / 1e9:.0f} GB release ({size_why})" if size is not None
                 else f"fetch + load not estimated ({size_why})")
        head = (f"{island_label(island)}: a release costs {bound}{_dur(cost)} of sampling on pool {name} "
                f"({engine}): {moved} + {drain_why}")
        every = island_release_every(island, argv)
        age = island_max_policy_age(island, argv)
        step = (step_s or {}).get(island.index)
        if not step or step <= 0:
            if every <= 1:
                out.append(f"WARNING: {head}; the island releases every local version (--rl-release-every 1); "
                           f"set islands[{island.index}].trainer.step_s (or fleet.step_s), the seconds of one "
                           "local version, for a suggested --rl-release-every")
            else:
                out.append(f"{head}; release every {every} local versions (set step_s for a suggested cadence)")
            continue
        suggest = max(MIN_SUGGESTED_RELEASE_EVERY, math.ceil(cost / step))
        if every >= suggest:
            continue
        want_age = suggest + POLICY_AGE_MARGIN
        age_note = ("" if age is not None and age >= want_age else
                    f" and [\"--rl-max-policy-age\", \"{want_age}\"] (at least N + {POLICY_AGE_MARGIN}; it is "
                    f"{'the default' if age is None else age})")
        out.append(f"WARNING: {head}; at a step of about {_dur(step)} the island's --rl-release-every {every} "
                   f"releases faster than its pools take a release in: set islands[{island.index}].trainer.argv "
                   f"[\"--rl-release-every\", \"{suggest}\"]{age_note} (N = max(2, ceil({cost:.0f} s / "
                   f"{step:g} s)))")
    return out


def breaker_pool_problems(plan: Any, section: Any) -> list[str]:
    """``breakers.pools.<name>`` keys of a fleet must be compiled pool names
    (``i<k>-<name>``: the name each island's copy of a pool runs under, and the key its
    pool resolves its overrides by); a key that names no pool would be ignored."""
    pools = section.get("pools") if isinstance(section, dict) else None
    if not isinstance(pools, dict):
        return []
    rows = [p for island in plan.islands for p in island.pools]
    names = {p.name for p in rows}
    out = []
    for key in pools:
        if key in names:
            continue
        meant = [p.name for p in rows if p.source_name == key]
        out.append(f"breakers.pools.{key}: no pool of the fleet is named {key!r}; a fleet's pools run as i<k>-<name>"
                   + (f" (this one as {' / '.join(meant)}): name each island's copy" if meant
                      else f" ({', '.join(sorted(names))})"))
    return out


def syncer_lines(plan: Any, trainer_argv: Sequence[str], syncer: dict[str, Any] | None) -> list[str]:
    """The syncer's region as the launch resolves it (it follows islands pinned to one
    area unless ``--syncer-region`` is set) and the launch's warnings for islands placed
    apart from it, each island named; and its price (``syncer.usd_per_h``)."""
    from yeto.gpu_spec import parse_gpu_spec  # noqa: PLC0415
    from yeto.launch import syncer_region as sr  # noqa: PLC0415
    from yeto.rl.up_consistency import trainer_args  # noqa: PLC0415

    out = []
    args = trainer_args([str(a) for a in trainer_argv])
    try:
        specs = parse_gpu_spec(args.gpu) if args is not None and getattr(args, "gpu", None) else []
    except (ValueError, TypeError):
        specs = []
    if args is not None and specs:
        explicit = not sr.is_default_syncer_region(getattr(args, "syncer_region", None))
        followed = None if explicit else sr.followed_region(args, specs)
        sr.resolve_syncer_region(args, specs, out=io.StringIO())
        region = str(args.syncer_region)
        how = ("--syncer-region" if explicit else "follows the islands' @region" if followed
               else "the default: the islands are not all pinned to one area")
        out.append(f"fleet: syncer region {region} ({how})")
        names = ", ".join(f"{i.index} = {i.name}" for i in plan.islands)
        for finding in sr.region_findings(args, specs):
            out.append(f"WARNING: syncer: {finding} (launch islands: {names})")
    price = (syncer or {}).get("usd_per_h") if isinstance(syncer, dict) else None
    if price is None:
        out.append("fleet: syncer.usd_per_h is unset: the fleet spend counts the syncer at $0/h (set it when the "
                   "syncer runs on a billed VM)")
    return out


def budget_lines(plan: Any, *, pools: Sequence[tuple[str, int, float]], syncer_usd_per_h: float,
                 usd: float | None, wall_h: float | None, sandbox_usd_per_h: float = 0.0) -> list[str]:
    """The fleet's hourly cost estimate and what ``budget.usd`` buys of it. ``pools``:
    ``(compiled name, target samplers, usd per node hour)`` of each pool; ``sandbox_usd_per_h``:
    the agentic pools' sandboxes (``fleet.sandbox_usd_per_h``)."""
    pool_h = sum(max(0, n) * max(0.0, price) for _name, n, price in pools)
    islands_h = sum(max(0.0, i.usd_per_h or 0.0) for i in plan.islands)
    sandbox_h = max(0.0, sandbox_usd_per_h)
    hourly = pool_h + islands_h + max(0.0, syncer_usd_per_h) + sandbox_h
    listed = ", ".join(f"{name} {n} x ${price:g}" for name, n, price in pools)
    def island_price(i: Any) -> str:
        if i.usd_per_h is None:
            return f"{i.name} $0 (unset)"
        miles = len(getattr(i, "miles_samplers", ()) or ())
        return f"{i.name} ${i.usd_per_h:g}" + (f" with {miles} Miles sampler(s)" if miles else "")

    trainers = ", ".join(island_price(i) for i in plan.islands)
    line = (f"fleet: budget estimate ${hourly:.2f}/h = pools ${pool_h:.2f}/h ({listed}) + trainers "
            f"${islands_h:.2f}/h ({trainers}) + syncer ${max(0.0, syncer_usd_per_h):.2f}/h"
            + (f" + sandboxes ${sandbox_h:.2f}/h" if sandbox_h else ""))
    if wall_h is not None:
        line += f"; over the {wall_h:g} h wall clock ${hourly * wall_h:.2f}"
    if usd is not None:
        line += f" (budget.usd ${usd:g})"
    out = [line]
    for i in plan.islands:
        if i.usd_per_h is None and usd is not None:
            out.append(f"WARNING: {island_label(i)}: trainer.usd_per_h is unset: its trainer is counted at $0/h and "
                       "the budget guard does not see it")
    if usd is not None and hourly > 0:
        lasts = usd / hourly
        if wall_h is None or lasts < wall_h:
            out.append(("WARNING: " if wall_h is not None else "") +
                       f"fleet: budget.usd ${usd:g} lasts about {lasts:.1f} h at ${hourly:.2f}/h"
                       + (f", before the {wall_h:g} h wall clock: the budget breaker stops the whole fleet then"
                          if wall_h is not None else " (no wall clock): the budget breaker stops the whole fleet then"))
    return out
