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
  training form (full parameters / partial experts: a whole checkpoint, laid
  over the engine's base for partial experts, reloaded in place);
* :func:`release_every_warnings`: an island with a vLLM pool (its in-place
  reload drains the engine) whose trainer releases every local version;
* :func:`syncer_lines`: the syncer's region against the islands' (the launch's
  own rule, :mod:`yeto.launch.syncer_region`) and its price;
* :func:`budget_lines`: the fleet's hourly estimate (pools, each island's
  trainer, the syncer) over the wall clock, against ``budget.usd``.

Import-light: no torch, ray or miles; only local files are read.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from typing import Any

SERVER_ENGINES = ("vllm", "sglang")  # external engines that load a release from the store
CPU_ENGINES = ("fake",)  # the CPU engine of the tests and dry rehearsals
# MI355X (K3.1, vLLM ROCm): a drain-mode in-place reload costs 9-12 minutes of sampling
VLLM_DRAIN_MIN = "9-12 min"


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


def release_form_lines(plan: Any, pools: Sequence[tuple[str, dict[str, Any] | None]]
                       ) -> tuple[list[str], list[str]]:
    """``(problems, notes)``: whether each pool's engine loads releases of the fleet's
    training form. Full parameters and partial experts release a whole checkpoint
    (partial experts: the trained experts, laid over the engine's ``base_dir``), which
    an external vLLM / SGLang server reloads (in place with ``server.in_place_reload``,
    else by a restart per release). Another engine adapter loads no store release."""
    form = (plan.sync or {}).get("form")
    kind = {"full parameters": "full-parameter", "partial experts": "partial-experts"}.get(form or "", "policy")
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
        if form == "partial experts" and engine == "vllm":
            notes.append(f"{where}: partial experts on vLLM: each release is laid over engine.server.base_dir and "
                         "reloaded whole (only the Kimi-K3.1 expert overlay reloads the selected experts alone)")
    return problems, notes


def release_every_warnings(plan: Any, trainer_argv: Sequence[str],
                           pools: Sequence[tuple[str, dict[str, Any] | None]]) -> list[str]:
    """An island with a vLLM pool whose trainer releases every local version: vLLM's
    in-place reload drains the engine (no request runs while it reloads), so each
    release costs sampling time; the warning suggests an ``--rl-release-every`` for
    the island (at most its ``--rl-max-policy-age``, the launcher's rule)."""
    specs = dict(pools)
    argv = [str(a) for a in trainer_argv]
    rounds = _argv_value(argv, "--local-rl-rounds-per-sync")
    rounds = int(rounds) if rounds and rounds.isdigit() and int(rounds) > 0 else 2
    out = []
    for island in plan.islands:
        vllm = [p.name for p in island.pools if _engine(specs.get(p.name))[0] == "vllm"]
        if not vllm or island_release_every(island, argv) > 1:
            continue
        age = island_max_policy_age(island, argv)
        suggest = max(2, rounds) if age is None else max(2, min(rounds, age))
        raise_age = age is not None and age < suggest
        out.append(f"WARNING: {island_label(island)}: pool(s) {', '.join(vllm)} run vLLM, whose in-place reload drains "
                   f"the engine (about {VLLM_DRAIN_MIN} of sampling per release measured on MI355X), and the island's "
                   f"trainer releases every local version (--rl-release-every 1): set islands[{island.index}]."
                   f"trainer.argv [\"--rl-release-every\", \"{suggest}\"]"
                   + (f" and raise --rl-max-policy-age to at least {suggest} (it is {age}; release_every <= "
                      "max_policy_age)" if raise_age else ""))
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
                 usd: float | None, wall_h: float | None) -> list[str]:
    """The fleet's hourly cost estimate and what ``budget.usd`` buys of it. ``pools``:
    ``(compiled name, target samplers, usd per node hour)`` of each pool."""
    pool_h = sum(max(0, n) * max(0.0, price) for _name, n, price in pools)
    islands_h = sum(max(0.0, i.usd_per_h or 0.0) for i in plan.islands)
    hourly = pool_h + islands_h + max(0.0, syncer_usd_per_h)
    listed = ", ".join(f"{name} {n} x ${price:g}" for name, n, price in pools)
    def island_price(i: Any) -> str:
        if i.usd_per_h is None:
            return f"{i.name} $0 (unset)"
        miles = len(getattr(i, "miles_samplers", ()) or ())
        return f"{i.name} ${i.usd_per_h:g}" + (f" with {miles} Miles sampler(s)" if miles else "")

    trainers = ", ".join(island_price(i) for i in plan.islands)
    line = (f"fleet: budget estimate ${hourly:.2f}/h = pools ${pool_h:.2f}/h ({listed}) + trainers "
            f"${islands_h:.2f}/h ({trainers}) + syncer ${max(0.0, syncer_usd_per_h):.2f}/h")
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
