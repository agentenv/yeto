"""The per-island runtime of an island fleet's ``yeto run up`` (:mod:`yeto.rl.fleet`).

A fleet runs one synced-group trainer launch (every island's trainer and the syncer)
and every island's pools; island k's trainer reads its samples from and releases its
policy to ``<store>/trainer<k>``, which its pools write. The single-trainer monitor
reads the run store, where a fleet writes nothing but ``fleet/``, so the monitor's
store checks run per island here, each in its own sub-directory and with the
island's own resolved breakers (:func:`yeto.breakers.fleet.resolve_islands`):

* the first-group wait waits only for the islands with no group yet, by name;
* ``run.starvation`` and ``run.release_stall`` watch each island's buffer, sampler
  pools and ``LATEST`` release;
* the data end is per island: island k ended once every pool of it finished, no
  member is left and its task ledger has nothing left for it; its shards are then
  sealed in its sub-directory. The fleet's data ended once every island's did (or
  the syncer finished: the launch then ends as ``SUCCEEDED``);
* the spend is :func:`yeto.breakers.fleet.fleet_spend` (pools, each island's trainer
  for the launch's billed hours, the syncer); ``run.budget`` stops the whole fleet.

A trip of an island-scoped breaker (``run.first_group`` / ``run.starvation`` /
``run.release_stall`` / ``run.trainer_exit`` / ``run.trainer_silence`` with a stop
action) stops that island only (:class:`yeto.breakers.fleet.FleetStops`, kept in
``state["fleet_stops"]``): the trip is recorded and reported, the island's own STOP is
written (``fleet/trainers/<k>/STOP``, :mod:`yeto.launch.island_lifecycle`: the
launch's fleet controller releases that island's trainer and relaunches it no more)
and its pools are stopped (``nc pool-stop``). The other islands go on. Each island's
trainer is judged on its own: ``run.trainer_exit`` when its record in the run store
(``fleet/trainers/<k>/state.json``) shows it given up or failed, ``run.trainer_silence``
when its event tape stops growing. The run's STOP flag (the whole launch) is kept for
the fleet-wide stops: ``run.budget``, ``run.wall_clock`` and fewer islands running than
``fleet.min_islands`` (default 1: every island stopped). An island without an event tape
(a sky / Verda / fetchable SSH island, a launch without records) keeps the run's own
silence over the launcher log (:meth:`FleetRuntime.judges_every_trainer`). A fixed roster
(``plan.fixed_roster``) keeps the path before the island STOP: no island STOP is written
(its launch would end the whole fleet), its trainer starves inside the launch, and the
trainers are not judged here.

A restarted ``run up`` (same run) replays only the islands' own trips
(``state["fleet_island_stops"]``): those islands stay stopped and their pools are not
driven again. A fleet-scoped trip (``run.trainer_exit``, ``run.budget``, the launch)
ended the launch that the restart replaces, so it is not replayed; a new trainer
launch also drops every fleet-scoped stop of this process (:meth:`on_trainer_start`).

Import-light: no torch, ray or miles.
"""

from __future__ import annotations

import json
import os
from typing import Any


def _up() -> Any:
    from yeto.rl import up  # noqa: PLC0415 - up imports this module lazily

    return up


def _log(msg: str) -> None:
    _up()._log(msg)



def stopped_by_trainer_worker(record: dict[str, Any]) -> bool:
    """Island ``record`` (its run-store record) was stopped on an island STOP its fleet's
    trainer worker wrote: to move it (its next launch starts on the new machines), remove it
    or give it up (the worker releases its pools), or to forward the fleet's stop. None of
    them is an operator's stop: its pools are not stopped for it and it is not counted as a
    stopped island (a moved island's pools feed its next launch)."""
    stop = record.get("stop")
    if isinstance(stop, dict) and stop.get("by") is not None:
        from yeto.launch.island_lifecycle import STOP_BY_TRAINER_WORKER  # noqa: PLC0415

        return str(stop.get("by")) == STOP_BY_TRAINER_WORKER
    return False

class FleetRuntime:
    """The fleet half of one :class:`yeto.rl.up.Run` (its ``fleet_plan()`` is set)."""

    def __init__(self, run: Any):
        self.run = run
        self.plan = run.fleet_plan()
        self._objects: dict[int, Any] = {}
        self._resolved: list[Any] | None = None
        self._last: dict[int, dict[str, str] | None] = {}
        self._calls: dict[int, int] = {}
        self._starvation: dict[int, Any] = {}
        self._listed: dict[int, tuple[float, bool]] = {}
        self._release: dict[int, Any] = {}
        self._release_read: dict[int, tuple[float, Any, bool]] = {}
        self._stops: Any = None
        self._states: Any = None
        self._tape_seen: dict[int, tuple[Any, float]] = {}
        self._exit_seen: set[int] = set()
        self._records_read: tuple[float, dict[int, dict[str, Any]]] | None = None

    # -- islands
    def label(self, k: int) -> str:
        island = self.plan.islands[k]
        return f"island {island.index} ({island.name})"

    def pool_indices(self, k: int) -> list[int]:
        names = {p.name for p in self.plan.islands[k].pools}
        return [i for i, ref in enumerate(self.run.pools) if ref.name in names]

    def objects(self, k: int) -> Any:
        """The object API of island k's sub-directory ``<store>/trainer<k>``."""
        if k not in self._objects:
            self._objects[k] = self.run._objects_of(self.plan.islands[k].store)  # noqa: SLF001
        return self._objects[k]

    def resolved(self, k: int) -> Any:
        """Island k's resolved breakers (its run and trainer entries as its trainer applies them)."""
        if self._resolved is None:
            from yeto.breakers import fleet as breakers_fleet  # noqa: PLC0415

            self._resolved = [rb for _island, rb in breakers_fleet.resolve_islands(self.run.cfg)]
        return self._resolved[k]

    def stops(self) -> Any:
        """The fleet's :class:`~yeto.breakers.fleet.FleetStops`. A new runtime (a restarted
        ``run up``) replays the islands' own trips only (``state["fleet_island_stops"]``,
        written by :meth:`stop_island`), never a fleet-scoped one (module doc)."""
        if self._stops is None:
            from yeto.breakers import fleet as breakers_fleet  # noqa: PLC0415
            from yeto.breakers.events import stop_reason_for  # noqa: PLC0415

            self._stops = breakers_fleet.FleetStops(breakers_fleet.Island(i.index, i.name) for i in self.plan.islands)
            known = {i.index for i in self.plan.islands}
            for key, row in sorted((self.run.state.get("fleet_island_stops") or {}).items()):
                name = str((row or {}).get("breaker") or "")
                if not str(key).isdigit() or int(key) not in known:
                    continue
                if (row or {}).get("operator"):  # stopped by hand (yeto island stop), not a trip
                    self._stops.stop(int(key), str(row["operator"]))
                    continue
                if not name:
                    continue
                self._stops.record({"event": "rl_breaker", "state": "trip", "name": name,
                                    "stop_reason": stop_reason_for(name)}, island=int(key))
        return self._stops

    def island_stopped(self, k: int) -> str | None:
        """Why island k is stopped (None: it runs)."""
        return self.stops().stop_reason(k)

    def on_trainer_start(self) -> None:
        """A new trainer launch of this run: the stops come back to the islands' own trips
        (a fleet-scoped stop ended the launch this one replaces); ``state["fleet_stops"]``
        is written again (logged when that dropped any)."""
        before = (self.run.state.get("fleet_stops") or {}).get("islands") or ()
        self._stops = None
        stops = self.stops()
        dropped = [f"{self.label(int(r['index']))}: {r['stop_reason']}" for r in before
                   if r.get("stop_reason") and stops.stop_reason(int(r["index"])) != r["stop_reason"]]
        if dropped:
            _log(f"fleet: a new trainer launch: the earlier fleet-wide stop(s) no longer hold ({'; '.join(dropped)})")
        self.run.state["fleet_stops"] = stops.to_json()
        self.run.save()

    def pool_restart_refusal(self, pool: str) -> str | None:
        """Why ``run up`` must not drive pool ``pool`` again: its island is stopped by its
        own earlier trip (None: drive it)."""
        island = self.plan.island_of_pool(pool)
        reason = None if island is None else self.island_stopped(island.index)
        if reason is None:
            return None
        return (f"{self.label(island.index)} stays stopped ({reason}, an earlier trip of this run), so its pool "
                f"{pool} is not driven again; to run that island again, remove fleet_island_stops.{island.index} "
                f"from {self.run.state_path} (or start a new run name)")

    def ended(self, k: int) -> bool:
        return str(k) in (self.run.state.get("fleet_data_ended") or {})

    def active(self) -> list[int]:
        """Islands still sampling and training: not stopped by a trip, data not ended."""
        stops = self.stops()
        return [i.index for i in self.plan.islands if stops.stop_reason(i.index) is None and not self.ended(i.index)]

    # -- breakers
    def event(self, k: int | None, name: str, *, value: float | None, limit: float | None, detail: str = "",
              state: str = "trip", legacy_event: str | None = None) -> dict[str, Any]:
        """One ``rl_breaker`` event of island ``k`` (None: the fleet) on the run's tape, with
        the island's resolved action and its scope; a stop trip carries its
        ``stop_reason`` and is recorded in :meth:`stops` (it stops that island, or every
        island for a fleet-scoped entry), not as this process's stop reason: the fleet
        goes on with its other islands."""
        from yeto.breakers import fleet as breakers_fleet  # noqa: PLC0415
        from yeto.breakers.events import breaker_event, stop_reason_for  # noqa: PLC0415
        from yeto.breakers.spec import STOP_ACTIONS  # noqa: PLC0415

        action = source = None
        try:
            entry = (self.resolved(k) if k is not None else self.run.breakers).entry(name)
            action, source = entry.action, entry.source.location
        except Exception:  # noqa: BLE001 - a view only: the event still goes out
            pass
        reason = stop_reason_for(name) if state == "trip" and action in STOP_ACTIONS else None
        scope = breakers_fleet.Island(k, self.plan.islands[k].name).scope() if k is not None else None
        event = breaker_event(name, state=state, value=value, limit=limit, action=action, scope=scope, detail=detail,
                              legacy_event=legacy_event, config_source=source, stop_reason=reason, run=self.run.run)
        try:
            self.run._append_event(event)  # noqa: SLF001
        except Exception as error:  # noqa: BLE001 - reporting never changes what the breaker does
            _log(f"could not write rl_breaker {name}: {error}")
        self.record(event, island=k)
        return event

    def record(self, event: dict[str, Any], *, island: int | None = None) -> tuple[int, ...]:
        """Record an ``rl_breaker`` event in :meth:`stops` (and ``state["fleet_stops"]``)."""
        hit = self.stops().record(event, island=island)
        if hit:
            self.run.state["fleet_stops"] = self.stops().to_json()
            self.run.save()
        return hit

    def island_states(self) -> Any:
        """The islands' records and STOP channel in the run store
        (:class:`yeto.launch.island_lifecycle.IslandStateStore`)."""
        if self._states is None:
            from yeto.launch.island_lifecycle import IslandStateStore  # noqa: PLC0415

            self._states = IslandStateStore(self.run._objects(), log=_log)  # noqa: SLF001
        return self._states

    def stop_island(self, k: int, name: str, why: str, *, operator: str | None = None) -> None:
        """Island k's trip stopped it (module doc): its island STOP is written (the launch's
        fleet controller releases its trainer) and its pools are stopped. A fixed roster
        (``plan.fixed_roster``) writes no island STOP (its launch would end the whole fleet,
        exit 4): its pools are stopped and its trainer starves, as before the island STOP.
        An operator stop (``operator``: the island is ``done`` on a STOP its controller acted
        on already) writes none either: only its pools follow. A STOP written then would sit
        over a START that ``yeto island start`` put in its place, and the controller acts on
        no START while a STOP stands: the island would stay done."""
        pools = self.pool_indices(k)
        names = [self.run.pools[i].name for i in pools]
        miles = [m.name for m in self.miles_samplers(k)]
        from yeto.breakers.events import stop_reason_for  # noqa: PLC0415

        if self.plan.fixed_roster:
            _log(f"CRITICAL fleet_island_stop: {self.label(k)}: {name}: {why}; a fixed roster "
                 f"({self.plan.roster_reason or 'fleet.fixed_roster'}): the fleet's launch does not end one island, "
                 f"so its pools ({', '.join(names) or 'none'}) are stopped and the other islands go on; "
                 + (f"its Miles sampler island(s) {', '.join(miles)} are part of the launch and keep feeding its "
                    "trainer" if miles else
                    "its trainer starves until its trainer.sample_wait ends it and the launcher gives it up"))
        elif operator:
            _log(f"CRITICAL fleet_island_stop: {self.label(k)}: {why}; its trainer is done on its island STOP "
                 f"already (no island STOP written by run up); its pools ({', '.join(names) or 'none'}) are "
                 "stopped and the other islands go on"
                 + (f"; its Miles sampler island(s) {', '.join(miles)} follow its trainer" if miles else ""))
        else:
            states = self.island_states()
            # an island STOP already there (``yeto island stop``) is kept as written
            written = states.stop_requested(k) is not None or states.request_stop(k, stop_reason_for(name),
                                                                                  by="yeto run up")
            _log(f"CRITICAL fleet_island_stop: {self.label(k)}: {name}: {why}; "
                 + (f"its island STOP is written ({self.island_states().objects.__class__.__name__}: "
                    f"fleet/trainers/{k}/STOP): the launch releases its trainer"
                    if written else "its island STOP could not be written: its trainer starves until its "
                                    "trainer.sample_wait ends it and the launcher gives it up")
                 + f"; its pools ({', '.join(names) or 'none'}) are stopped and the other islands go on"
                 + (f"; its Miles sampler island(s) {', '.join(miles)} follow its trainer" if miles else ""))
        # Run.pool_stop: on the store control plane the stop names the run store and the pool, so
        # a pool whose controller is gone is still released from the store's state
        rc = max((self.run._cmd(self.run.pool_stop(self.run.pools[i].config),  # noqa: SLF001
                                log="down.log", timeout=1800) for i in pools), default=0)
        if rc:
            _log(f"CRITICAL: {self.label(k)}: pool-stop rc={rc}: its machines may still bill; release them with "
                 "`yeto run down --config <run.json>`")
        self.run.state.setdefault("fleet_island_stops", {})[str(k)] = {
            "at": _up()._now(), "breaker": None if operator else name, "why": why, "pools": names, "rc": rc,
            **({"operator": operator} if operator else {})}
        self.run.state["fleet_stops"] = self.stops().to_json()
        self.run.save()

    def trip_island(self, k: int, name: str, *, value: float | None, limit: float | None, why: str,
                    legacy_event: str | None = None) -> bool:
        """An island-scoped stop trip: the event, then :meth:`stop_island` when it stopped
        island k (True)."""
        event = self.event(k, name, value=value, limit=limit, detail=why, legacy_event=legacy_event)
        if not event.get("stop_reason"):
            return False
        self.stop_island(k, name, why)
        return True

    def all_stopped(self) -> str | None:
        """The fleet's stop reason once fewer islands run than ``fleet.min_islands`` (default 1:
        every island stopped by a trip), else None: the first stopped island's reason (a
        fleet-scoped trip's when there was one)."""
        stops = self.stops()
        running = stops.running()
        if len(running) >= max(1, int(getattr(self.plan, "min_islands", 1) or 1)):
            return None
        if stops.fleet_stop_reason is not None:
            return stops.fleet_stop_reason
        first = next((i.index for i in self.plan.islands if stops.stop_reason(i.index) is not None), None)
        return None if first is None else stops.stop_reason(first)

    OPERATOR_STOP_EVENT = "fleet_island_operator_stop"

    def operator_stop(self, k: int, record: dict[str, Any]) -> None:
        """Island k's trainer was stopped on an island STOP by hand (``yeto island stop``): its
        own event (no breaker trip), the island is stopped in :meth:`stops` and its pools are
        stopped (:meth:`stop_island`); the other islands go on."""
        stop = record.get("stop") or {}
        reason = str(stop.get("reason") or (record.get("decision") or {}).get("detail") or "STOP")[:200]
        by = str(stop.get("by") or "an operator")
        why = f"its trainer was stopped on its island STOP ({reason}, by {by}): its pools follow"[:500]
        island = self.plan.islands[k]
        try:
            self.run._append_event({"event": self.OPERATOR_STOP_EVENT, "island": island.index,  # noqa: SLF001
                                    "name": island.name, "reason": reason, "by": by, "run": self.run.run,
                                    "at": _up()._now()})
        except Exception as error:  # noqa: BLE001 - reporting never changes the stop
            _log(f"could not write {self.OPERATOR_STOP_EVENT}: {error}")
        self.stops().stop(k, f"operator:{reason}")
        self.stop_island(k, "operator", why, operator=f"operator:{reason}")

    # -- each island's trainer (run.trainer_exit, run.trainer_silence)
    def silence_limit_s(self, k: int) -> float:
        """Island k's ``run.trainer_silence`` limit (as :meth:`Run.silence_limit_s` of one trainer)."""
        entry = self.resolved(k).entry("run.trainer_silence")
        if entry.params["limit_s"] is not None:
            return float(entry.params["limit_s"])
        return max(entry.params["stall_factor_n"] * self.run.stall_timeout_s(), entry.params["floor_s"])

    RECORDS_TTL_S = 5.0  # one poll reads the islands' records once (the silence check and trainer_check)

    def trainer_records(self, now: float) -> dict[int, dict[str, Any]]:
        """The running islands' records in the run store (read at most every ``RECORDS_TTL_S``);
        a record that does not read is missing here (that island is not judged by it)."""
        cached = self._records_read
        if cached is not None and 0 <= now - cached[0] < self.RECORDS_TTL_S:
            return cached[1]
        active = self.active()
        records = self.island_states().read_all(active) if active else {}
        self._records_read = (now, records)
        return records

    def judges_every_trainer(self, now: float) -> bool:
        """True when :meth:`trainer_check` judges the silence of every running island (each has a
        record with an event tape): the run's own silence over the launcher log is then off.
        False for a fixed roster (the old path: the run's silence and the launcher's exit) and
        while any running island has no tape (a sky / Verda / fetchable SSH island, a launch
        without records, a record that did not read) or its recorded tape does not stat (not
        written yet, on another machine): the run's silence covers the fleet."""
        if self.plan.fixed_roster:
            return False
        active = self.active()
        if not active:
            return False
        records = self.trainer_records(now)
        for k in active:
            tape = (records.get(k) or {}).get("tape")
            if not tape:
                return False
            try:
                os.stat(str(tape))
            except OSError:
                return False
        return True

    def trainer_check(self, now: float) -> None:
        """Each running island's trainer from its record in the run store: given up or failed
        (``abandoned`` / ``failed``) trips its ``run.trainer_exit``; its event tape not
        growing for its ``run.trainer_silence`` limit while it runs trips that. Either stops
        that island only (:meth:`trip_island`). An island without a record (a launch that
        keeps none) or without a tape is not judged here (:meth:`judges_every_trainer`: the
        run's silence covers it); a tape that does not stat is unknown, not quiet: its quiet
        time starts again once it reads. A fixed roster is not judged here (the old path)."""
        if self.plan.fixed_roster:
            return
        active = self.active()
        if not active:
            return
        records = self.trainer_records(now)
        for k in active:
            record = records.get(k)
            if not record:
                continue
            state = record.get("state")
            if state in ("abandoned", "failed") and k not in self._exit_seen:
                self._exit_seen.add(k)
                decision = record.get("decision") or {}
                why = (f"its trainer is {state}"
                       + (f" ({decision.get('cause')}: {decision.get('detail')})" if decision.get("cause") else "")
                       + (f"; exit {record.get('exit')}" if record.get("exit") else ""))[:500]
                self.trip_island(k, "run.trainer_exit", value=None, limit=None, why=why)
                continue
            decision = record.get("decision") or {}
            if (state == "done" and decision.get("cause") == "stopped" and k not in self._exit_seen
                    and not stopped_by_trainer_worker(record)):
                # stopped on an island STOP this run up did not write (``yeto island stop``): its
                # trainer is released, so its pools follow; the other islands go on. Not a trip.
                self._exit_seen.add(k)
                self.operator_stop(k, record)
                continue
            if state != "running" or not record.get("tape"):
                self._tape_seen.pop(k, None)
                continue
            if not self.resolved(k).enabled("run.trainer_silence"):
                continue
            try:
                st = os.stat(str(record["tape"]))
            except OSError:  # unknown (not written yet, another machine): not counted as quiet
                self._tape_seen.pop(k, None)
                continue
            sig = (st.st_size, st.st_mtime)
            seen = self._tape_seen.get(k)
            if seen is None or seen[0] != sig:
                self._tape_seen[k] = (sig, now)
                continue
            quiet, limit = now - seen[1], self.silence_limit_s(k)
            if quiet > limit:
                self._tape_seen.pop(k, None)
                self.trip_island(k, "run.trainer_silence", value=quiet, limit=limit,
                                 why=f"its event tape is silent {quiet / 60:.0f} min (limit {limit / 60:.0f} min)")

    # -- the first group
    def buffer_keys(self, k: int) -> set[str] | None:
        try:
            return set(self.objects(k).list("sample-buffer/shards/"))
        except Exception as error:  # noqa: BLE001
            _log(f"{self.label(k)}: store list failed: {type(error).__name__}: {error}"[:300])
            return None

    def wait_first_group(self) -> str | None:
        """None once every island that runs holds a group in its sub-directory; else why
        the wait ended. Only the islands without a group are waited for (named in the
        log); an island past its ``run.first_group.timeout_s`` trips its own breaker
        (that island stops); the fleet's wait ends on a timeout only when no island
        has a group."""
        up = _up()
        run = self.run
        fg = run.cfg.get("first_group") or {}
        poll = float(fg.get("poll_s", 60))
        seen = run.state.setdefault("fleet_first_group", {})
        start = up.time.monotonic()
        shown = None
        while True:
            for k in self.active():
                if str(k) in seen:
                    continue
                n = len(self.buffer_keys(k) or ())
                if n:
                    seen[str(k)] = {"at": up._now(), "objects": n}
                    _log(f"first group: {self.label(k)}: {n} object(s) in {self.plan.islands[k].store}")
                    run.save()
            waiting = [k for k in self.active() if str(k) not in seen]
            if not waiting:
                if not seen:
                    return "first group timed out"
                run.mark("first_group", objects=sum(int(v.get("objects") or 0) for v in seen.values()),
                         islands=sorted(int(k) for k in seen))
                return None
            if waiting != shown:
                shown = waiting
                have = [self.label(int(k)) for k in sorted(seen, key=int)]
                _log(f"first group: waiting for {', '.join(self.label(k) for k in waiting)}"
                     + (f" ({', '.join(have)} has a group)" if have else ""))
            why = run.breach(trainer=False)
            if why:
                return why
            elapsed = up.time.monotonic() - start
            for k in waiting:
                entry = self.resolved(k).entry("run.first_group")
                timeout = entry.params.get("timeout_s") if entry.enabled else None
                if timeout is not None and elapsed >= float(timeout):
                    _log(f"first group: {self.label(k)}: timed out")
                    self.trip_island(k, "run.first_group", value=elapsed, limit=float(timeout),
                                     why=f"no sample-buffer object reached {self.plan.islands[k].store}")
            if self.all_stopped() is not None:
                return "first group timed out"
            if all(str(k) in seen for k in self.active()):
                continue  # the islands left all have a group: recorded at the top
            up.time.sleep(poll)

    # -- starvation
    def new_groups(self, k: int) -> int | None:
        """:meth:`Run.buffer_new_groups` of island k's sub-directory."""
        up = _up()
        calls = self._calls.get(k, 0)
        self._calls[k] = calls + 1
        found, self._last[k] = up.new_buffer_groups(lambda: self.objects(k), self._last.get(k), calls,
                                                    self.run.SHARD_SCAN_EVERY)
        return found

    def starvation_check(self, now: float) -> None:
        """``run.starvation`` per island that runs: its new groups, its pools' health; a
        stop verdict stops that island (:meth:`trip_island`)."""
        from yeto.rl.starvation import ALERT_EVENT, RECOVERED_EVENT, STOP, STOP_EVENT, StarvationConfig, StarvationWatch  # noqa: PLC0415

        run = self.run
        for k in self.active():
            try:
                cfg = StarvationConfig.from_resolved(self.resolved(k))
            except ValueError:
                continue  # the preflight refuses a bad combination
            if not cfg.enabled:
                continue
            watch = self._starvation.get(k)
            if watch is None:
                watch = self._starvation[k] = StarvationWatch(cfg, now)

            def listing(k: int = k) -> int | None:
                found = self.new_groups(k)
                self._listed[k] = (now, found is not None)
                return found

            listed_at = self._listed.get(k)
            listed_now = listed_at is None or now - listed_at[0] >= cfg.check_s
            new_groups = listing() if listed_now else None
            health = run.sampler_health(now, pools=self.pool_indices(k))
            cause = run.trainer_side_cause()
            verdict = watch.observe(now, new_groups=new_groups, health=health, cause=cause,
                                    listed=self._listed.get(k, (0.0, False))[1])
            if verdict.state == STOP and not listed_now:  # a stop is decided on a fresh listing only
                verdict = watch.observe(now, new_groups=listing(), health=health, cause=cause,
                                        listed=self._listed[k][1])
            rows = run.state.setdefault("fleet_starvation", {})
            if verdict.recovered:
                _log(f"{RECOVERED_EVENT}: {self.label(k)}: a new group reached its buffer")
                self.event(k, "run.starvation", value=verdict.starved_s, limit=cfg.alert_s or None, state="clear",
                           detail="a new group reached the buffer", legacy_event=RECOVERED_EVENT)
                rows.pop(str(k), None)
                run.save()
            if verdict.new_alert:
                _log(f"CRITICAL {ALERT_EVENT}: {self.label(k)}: {verdict.reason}")
                self.event(k, "run.starvation", value=verdict.starved_s, limit=cfg.alert_s or None, state="warn",
                           detail=verdict.reason, legacy_event=ALERT_EVENT)
                rows[str(k)] = {"at": _up()._now(), "event": ALERT_EVENT,
                                "starved_min": round(verdict.starved_s / 60), "why": verdict.reason}
                run.save()
            if verdict.state == STOP:
                _log(f"CRITICAL {STOP_EVENT}: {self.label(k)}: {verdict.reason}")
                self.trip_island(k, "run.starvation", value=verdict.starved_s, limit=cfg.stop_s,
                                 why=str(verdict.reason), legacy_event=STOP_EVENT)

    # -- release stall
    def release_stall_check(self, now: float) -> None:
        """``run.release_stall`` per island that runs, over its ``LATEST`` release; with
        ``action: stop`` a stall stops that island."""
        from yeto.rl.release_stall import ALERT_EVENT, RECOVERED_EVENT, STOP, STOP_EVENT, ReleaseStallConfig, ReleaseStallWatch  # noqa: PLC0415

        run = self.run
        tr = (run.state.get("procs") or {}).get("trainer") or {}
        for k in self.active():
            entry = self.resolved(k).entry("run.release_stall")
            if not entry.enabled:
                continue
            watch = self._release.get(k)
            if watch is None:
                watch = self._release[k] = ReleaseStallWatch(ReleaseStallConfig(
                    limit_s=entry.params["limit_s"], check_s=entry.params["check_s"], stop=entry.action == "stop"))
            read = self._release_read.get(k)
            if read is None or now - read[0] >= watch.config.check_s:
                read = self._release_read[k] = (now, *run.latest_release(self.objects(k)))
            _, release, ok = read
            verdict = watch.observe(now, release=release, incarnation=(tr.get("pid"), tr.get("started")), read=ok)
            rows = run.state.setdefault("fleet_release_stall", {})
            if verdict.recovered:
                _log(f"{RECOVERED_EVENT}: {self.label(k)}: a new policy release reached its store")
                self.event(k, "run.release_stall", value=verdict.stalled_s, limit=verdict.limit_s, state="clear",
                           detail="a new policy release reached the store", legacy_event=RECOVERED_EVENT)
                rows.pop(str(k), None)
                run.save()
            if verdict.new_alert:
                _log(f"CRITICAL {ALERT_EVENT}: {self.label(k)}: {verdict.reason}")
                if verdict.state != STOP:
                    self.event(k, "run.release_stall", value=verdict.stalled_s, limit=verdict.limit_s,
                               detail=verdict.reason, legacy_event=ALERT_EVENT)
                rows[str(k)] = {"at": _up()._now(), "event": ALERT_EVENT,
                                "stalled_min": round(verdict.stalled_s / 60), "why": verdict.reason}
                run.save()
            if verdict.state == STOP:
                self.trip_island(k, "run.release_stall", value=verdict.stalled_s, limit=verdict.limit_s,
                                 why=str(verdict.reason), legacy_event=STOP_EVENT)

    # -- the data end
    def pools_finished(self, k: int) -> bool:
        """Every pool of island k finished (a sampler exited 0 on the data end), holds no
        member that is not gone and writes island k's sub-directory."""
        from yeto.rl.up_consistency import _norm_store, pool_store  # noqa: PLC0415

        run = self.run
        indices = self.pool_indices(k)
        if not indices:
            return False
        for i in indices:
            if run.pools[i].data_path != "r2":
                return False
            try:
                written = pool_store(run.pools[i].load())
            except Exception:  # noqa: BLE001
                return False
            if written is not None and _norm_store(written) != _norm_store(self.plan.islands[k].store):
                return False  # its samplers write elsewhere: island k's trainer does not read it
            doc = run.pool_doc(i)
            if not doc or not doc.get("finished"):
                return False
            if any(m.get("state") != "gone" for m in doc.get("members") or ()):
                return False
        return True

    def settled(self, k: int) -> bool:
        """Island k's tasks are settled. Its own ledger (``dispense: island``): every slot
        trained / exhausted / dropped, as for one trainer. The fleet's shared ledger: no
        slot is left to give out (queued / unclaimed) and no claim still open (running /
        stale) is held by one of island k's samplers; the claims of other islands are
        theirs to finish. A status that does not read: not settled."""
        from yeto.rl import up_consistency  # noqa: PLC0415
        from yeto.rl.engine import store_dispense  # noqa: PLC0415

        run = self.run
        shards = set(self.plan.islands[k].buffer_shards())
        shared = self.plan.dispense_store
        left, rows = 0, []
        try:
            seen: set[tuple[str, str]] = set()
            for i in self.pool_indices(k):
                store, ns = run.dispense_ledger(i)
                key = (up_consistency._norm_store(store), ns)  # noqa: SLF001
                if key in seen:
                    continue
                seen.add(key)
                ledger = store_dispense.ledger_objects(run._objects_of(store))  # noqa: SLF001
                report = store_dispense.status(ledger, ns, details=False, settled=False)
                totals = report["totals"]
                if totals.get("ended"):
                    continue
                if shared and key[0] == up_consistency._norm_store(shared):  # noqa: SLF001
                    open_slots = int(totals.get("queued") or 0) + int(totals.get("unclaimed") or 0)
                    mine = [t for t in report["tasks"] if t.get("state") in ("running", "stale")
                            and t.get("holder") in shards]
                    if not totals.get("slots"):
                        open_slots = max(open_slots, 1)  # no plan: not known to be settled
                    n = open_slots + len(mine)
                else:
                    n = sum(int(totals.get(s) or 0) for s in ("queued", "running", "stale", "unclaimed")) or 1
                if n:
                    left += n
                    rows.append({"store": store, "namespace": ns, "totals": totals})
        except Exception as error:  # noqa: BLE001 - asked again next poll
            _log(f"data end: {self.label(k)}: the dispenser status did not read ({type(error).__name__}: "
                 f"{error}"[:300] + "): not sealed")
            return False
        incomplete = run.state.setdefault("fleet_data_end_incomplete", {})
        if not left:
            if incomplete.pop(str(k), None) is not None:
                run.save()
            return True
        if (incomplete.get(str(k)) or {}).get("left") != left:
            _log(f"CRITICAL data_end_incomplete: {self.label(k)}: every pool of it finished but {left} task(s) are "
                 f"not sampled ({json.dumps(rows, sort_keys=True)[:600]}): check why its pools finished; `yeto run "
                 "up` starts them again (the island's starvation watch stays on)")
            incomplete[str(k)] = {"at": _up()._now(), "left": left}
            run.save()
        return False

    def pool_shards(self, k: int) -> list[str]:
        """The shards of island k's external pools (the ones run up may seal; a Miles
        sampler island of the launch ends its own shard)."""
        island = self.plan.islands[k]
        return island.pool_shards() if hasattr(island, "pool_shards") else island.buffer_shards()

    def miles_samplers(self, k: int) -> tuple[Any, ...]:
        return tuple(getattr(self.plan.islands[k], "miles_samplers", ()) or ())

    def seal(self, k: int) -> bool:
        """End every pool shard island k's trainer reads that did not end (in its
        sub-directory). An island that also has Miles sampler islands is not ended by
        it: they keep feeding its trainer (their own shards), so only its pools' end is
        recorded (``state["fleet_pools_ended"]``)."""
        up = _up()
        from yeto.rl.engine.sample_buffer import FileSampleBuffer, shard_index  # noqa: PLC0415

        sealed = []
        try:
            objects = self.objects(k)
            for shard in self.pool_shards(k):
                writer = FileSampleBuffer(objects, batch_size=1, index=shard_index(shard))
                if not writer.data_ended:
                    if writer.owner is not None:
                        writer.claim_owner(up.SEAL_OWNER, by="operator")
                    writer.mark_data_end()
                    sealed.append(shard)
        except Exception as error:  # noqa: BLE001 - tried again next poll
            _log(f"data end: {self.label(k)}: could not end its buffer shards: {type(error).__name__}: {error}"[:300])
            return False
        listed = f"({', '.join(sealed[:8])}{' ...' if len(sealed) > 8 else ''})"
        miles = self.miles_samplers(k)
        if miles:
            _log(f"data end: {self.label(k)}: every pool of it finished; ended {len(sealed)} pool shard(s) no sampler "
                 f"ended {listed}; its Miles sampler(s) {', '.join(m.name for m in miles)} keep feeding its trainer "
                 "(the island ends with them or the syncer's final version)")
            self.run.state.setdefault("fleet_pools_ended", {})[str(k)] = {"at": up._now(), "sealed": sealed}
            self.run.save()
            return True
        _log(f"data end: {self.label(k)}: every pool of it finished; ended {len(sealed)} shard(s) no sampler ended "
             f"{listed}; its trainer trains the full batches left and ends its local training")
        self.run.state.setdefault("fleet_data_ended", {})[str(k)] = {"at": up._now(), "sealed": sealed}
        self.run.state.get("fleet_starvation", {}).pop(str(k), None)
        self.run.save()
        return True

    def seal_data_end(self) -> bool:
        """Seal each island whose data ended; True once every island's did (the fleet's data
        end: ``state["data_ended"]``)."""
        run = self.run
        if run.state.get("data_ended"):
            return True
        stops = self.stops()
        pools_ended = run.state.get("fleet_pools_ended") or {}
        for island in self.plan.islands:
            k = island.index
            if self.ended(k) or stops.stop_reason(k) is not None or str(k) in pools_ended:
                continue
            if self.pools_finished(k) and self.settled(k):
                self.seal(k)
        if not all(self.ended(i.index) for i in self.plan.islands):
            return False
        sealed = [s for v in (run.state.get("fleet_data_ended") or {}).values() for s in v.get("sealed") or ()]
        _log(f"data end: every island's data ended ({len(self.plan.islands)} island(s)); the fleet's trainers "
             "finish their local training")
        run.state["data_ended"] = {"at": _up()._now(), "sealed": sealed, "islands": len(self.plan.islands)}
        run.state.pop("starvation", None)
        run.save()
        return True

    def pools_data_ended(self) -> bool:
        return all(self.pools_finished(i.index) for i in self.plan.islands)

    def dispense_settled(self) -> bool:
        return all(self.settled(i.index) for i in self.plan.islands)

    def data_end_shards(self) -> list[str]:
        return [s for island in self.plan.islands for s in self.pool_shards(island.index)]

    # -- spend
    def spend(self, now: float | None = None) -> Any:
        """:func:`yeto.breakers.fleet.fleet_spend`: every pool's billed node hours (the pools
        the desired state removed included: :meth:`Run.retired_pool_spend`), each
        island's trainer price for the launch's billed hours (one launch runs every
        island's trainer), and the syncer (``syncer.usd_per_h``) for the same hours."""
        from yeto.breakers import fleet as breakers_fleet  # noqa: PLC0415

        run = self.run
        now = _up().time.time() if now is None else now
        hours = run.trainer_hours(now)
        island_hours = [hours] * len(self.plan.islands)
        island_prices = list(breakers_fleet.island_prices(run.cfg))
        if self.plan.per_island:  # its own launch per island: the trainer worker's launch records
            island_hours, island_prices = self.island_launch_spend(now, island_prices)
        return breakers_fleet.fleet_spend(
            pool_node_hours=[*(run.pool_spend_rate(i, now) for i in range(len(run.pools))),
                             *run.retired_pool_spend(now)],  # and the pools the desired state removed
            island_hours=island_hours, island_usd_per_h=island_prices,
            syncer_hours=hours, syncer_usd_per_h=breakers_fleet.syncer_price(run.cfg))

    def island_launch_spend(self, now: float, declared: list[float]) -> tuple[list[float], list[float]]:
        """A fleet launched island by island: each island launch's billed hours and dollars
        from its record (``yeto.rl.trainer_worker.billed``: every incarnation at the price
        it ran at): the islands added while it trains count, a removed island bills up to
        its end. The hours an island ran without a price bill its declared island's price (a
        warning; an added island without one bills 0); its priced hours keep their dollars. Without a readable store the
        declared islands bill the trainer's hours, as for one launch. Returned as hours and
        the hourly price that gives each island's dollars (``fleet_spend``'s form)."""
        from yeto.rl import trainer_worker  # noqa: PLC0415

        run = self.run
        worker = getattr(run, "trainer_worker", None)
        try:
            objects = worker.objects if worker is not None else run._objects()  # noqa: SLF001
            rows = trainer_worker.billed(objects, now)
        except Exception as error:  # noqa: BLE001 - the conservative estimate
            _log(f"fleet: spend: the island launch records were not read ({type(error).__name__}: {error}); "
                 "the declared islands bill the trainer's hours")
            hours = run.trainer_hours(now)
            return [hours] * len(declared), declared
        by_id = {i.learner: p for i, p in zip(self.plan.islands, declared)}
        hours, prices = [], []
        for k, (h, usd, fallback) in sorted(trainer_worker.island_spend(rows, by_id).items()):
            if fallback:
                warned = getattr(self, "_unpriced_warned", set())
                if k not in warned:
                    self._unpriced_warned = warned | {k}
                    _log(f"WARNING: fleet: spend: trainer island {k} ran without a price (--usd-per-h, nor a billing "
                         "row of the price table for its --gpu); billed "
                         + (f"at its declared ${by_id[k]:g}/h" if k in by_id else "nothing") + " for that time")
            hours.append(h)
            prices.append(usd / h if h > 0 else 0.0)
        return hours, prices

    def budget_event(self, spend_usd: float, usd: float | None, detail: str) -> dict[str, Any]:
        """The fleet's ``run.budget`` trip (:func:`yeto.breakers.fleet.budget_trip`, stop
        reason ``breaker:run.budget``); the caller records it (:meth:`record`: every island)."""
        from yeto.breakers import fleet as breakers_fleet  # noqa: PLC0415

        action = source = None
        try:
            entry = self.run.breakers.entry("run.budget")
            action, source = entry.action, entry.source.location
        except Exception:  # noqa: BLE001
            pass
        spend = self.spend()
        event = breakers_fleet.budget_trip(spend, usd, emit=self.run._append_event, action=action,  # noqa: SLF001
                                           config_source=source, run=self.run.run,
                                           detail=f"{detail}: {json.dumps(spend.to_json(), sort_keys=True)}")
        if event is None:  # the spend moved between the two reads: report the one breach() saw
            event = self.run.breaker_event("run.budget", value=spend_usd, limit=usd, detail=detail)
        return event

    # -- restart
    def releases(self) -> dict[int, bool]:
        """Which islands' sub-directories hold a trained policy release (raises on a
        failed listing, as :meth:`Run.store_has_release`)."""
        from yeto.rl.engine.policy_release import records_in_objects, trained_releases  # noqa: PLC0415

        out = {}
        for island in self.plan.islands:
            try:
                records = records_in_objects(self.objects(island.index))
            except Exception as error:  # noqa: BLE001
                raise RuntimeError(f"--rl-round-cuts off restart: cannot list {self.label(island.index)}'s policy "
                                   f"releases ({type(error).__name__}: {error})"[:400]) from error
            out[island.index] = bool(trained_releases(records))
        return out

    # -- status
    def _trainer_view(self, k: int) -> dict[str, Any] | None:
        """Island k's trainer record (state, incarnation, relaunches, machine), or None."""
        try:
            record = self.island_states().read(k)
        except Exception:  # noqa: BLE001 - a view only
            return None
        if not record:
            return None
        return {key: record.get(key) for key in ("state", "incarnation", "relaunches", "handle", "source", "exit")
                if record.get(key) is not None}

    def status(self) -> dict[str, Any]:
        """Per island: first group, data end, stop reason; the fleet's spend."""
        state = self.run.state
        stops = self.stops()
        islands = []
        for island in self.plan.islands:
            k = str(island.index)
            keys = self.buffer_keys(island.index)  # None: the listing failed (unknown, not 0)
            islands.append({"index": island.index, "name": island.name, "store": island.store,
                            "pools": [p.name for p in island.pools],
                            "buffer_objects": None if keys is None else len(keys),
                            "first_group": (state.get("fleet_first_group") or {}).get(k),
                            "data_ended": (state.get("fleet_data_ended") or {}).get(k),
                            "stop_reason": stops.stop_reason(island.index),
                            "trainer": self._trainer_view(island.index),
                            "starvation": (state.get("fleet_starvation") or {}).get(k),
                            "release_stall": (state.get("fleet_release_stall") or {}).get(k)})
        try:
            spend = self.spend().to_json()
        except Exception as error:  # noqa: BLE001 - a view only
            spend = {"error": f"{type(error).__name__}: {error}"[:300]}
        return {"islands": islands, "stop_reason": stops.fleet_stop_reason, "spend": spend}
