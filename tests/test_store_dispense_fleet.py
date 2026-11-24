"""A fleet's store-dispenser ledgers (``yeto.rl.engine.store_dispense``, shared
ledger): pools of several trainer islands write their own ``<store>/trainer<k>``
and share one ledger under ``dispense_store`` (``<store>/fleet/dispense``), or keep
one ledger per island without it. Samplers are built through the engine sampler's
spec path (``engine_sampler_entry.build_prompts``) on local directory stores."""

from __future__ import annotations

import json
import types
from collections import Counter
from pathlib import Path

import pytest

from yeto.rl import up
from yeto.rl.engine import engine_sampler_entry as entry
from yeto.rl.engine import store_dispense as sd
from yeto.rl.engine.task_dispense import SAMPLER_INCARNATION_ENV, DispenseError

ROWS = 24
NS = "task-dispense"


def _prompts(tmp_path: Path, n: int = ROWS, name: str = "prompts.jsonl") -> Path:
    path = tmp_path / name
    path.write_text("".join(json.dumps({"text": f"q{i}", "label": "", "metadata": {"task_id": f"t{i}"}}) + "\n"
                            for i in range(n)))
    return path


def _spec(run: Path, island: int, shard: str, prompts: Path, *, shared: bool, groups: int = 2,
          **dispense) -> dict:
    spec = {"sampler_id": shard, "store": str(run / f"trainer{island}"),
            "prompts": {"path": str(prompts), "input_key": "text", "label_key": "label", "groups_per_round": groups,
                        "dispense": {"backend": "store", "namespace": NS, "scan_s": 0.0, "max_scan_s": 0.0,
                                     **dispense}}}
    if shared:
        spec["dispense_store"] = sd.fleet_dispense_store(str(run))
    return spec


def _sampler(spec: dict, inc: str = "inc-1") -> sd.StoreDispensedPrompts:
    prompts = entry.build_prompts(spec, {SAMPLER_INCARNATION_ENV: inc})
    assert isinstance(prompts, sd.StoreDispensedPrompts)
    return prompts


def _drain(samplers: dict[str, sd.StoreDispensedPrompts]) -> dict[str, list[int]]:
    """Round robin until every sampler sees the end: each claim pass's slots are
    written (trained) at once. ``{sampler: [slots]}``."""
    got: dict[str, list[int]] = {name: [] for name in samplers}
    live = dict(samplers)
    for _ in range(10 * ROWS):
        if not live:
            break
        for name, s in list(live.items()):
            state, slots = s.poll(s.groups_per_round)
            if state == "items":
                got[name] += slots
                s.written(0, slots)
            elif state in ("end", "fenced"):
                del live[name]
    assert not live, f"samplers that never saw the end: {sorted(live)}"
    return got


FLEET_POOLS = (("poolA-i0-0", 0), ("poolB-i0-0", 0), ("poolC-i1-0", 1))


def test_a_fleet_ledger_gives_every_prompt_out_once_across_islands_and_pools(tmp_path):
    run, path = tmp_path / "run", _prompts(tmp_path)
    samplers = {shard: _sampler(_spec(run, island, shard, path, shared=True)) for shard, island in FLEET_POOLS}
    try:
        got = _drain(samplers)
    finally:
        for s in samplers.values():
            s.close()
    slots = [p for ps in got.values() for p in ps]
    assert sorted(slots) == list(range(ROWS)), "every slot once, none twice, across the fleet"
    rows = Counter(samplers["poolA-i0-0"].slots.row(p) for p in slots)
    assert sorted(rows) == list(range(ROWS)) and set(rows.values()) == {1}, "each prompt dispensed exactly once"
    assert all(got[shard] for shard, _ in FLEET_POOLS), "every pool of both islands took a share"
    ledger = run / "fleet" / "dispense" / NS
    assert (ledger / "plan.json").exists()
    assert not (run / "trainer0" / NS).exists() and not (run / "trainer1" / NS).exists()
    totals = sd.status(sd.DirObjects(run / "fleet" / "dispense"), NS, details=False)["totals"]
    assert totals["trained"] == ROWS and totals["ended"]
    assert {s.plan_id for s in samplers.values()} == {samplers["poolA-i0-0"].plan_id}, "one plan for the fleet"


def test_island_dispense_gives_every_island_the_whole_data(tmp_path):
    """No ``dispense_store`` (``dispense: island``): each island's pools share the
    ledger of their own ``trainer<k>`` store, so every island covers every prompt."""
    run, path = tmp_path / "run", _prompts(tmp_path)
    samplers = {shard: _sampler(_spec(run, island, shard, path, shared=False)) for shard, island in FLEET_POOLS}
    try:
        got = _drain(samplers)
    finally:
        for s in samplers.values():
            s.close()
    island0 = got["poolA-i0-0"] + got["poolB-i0-0"]
    assert sorted(island0) == list(range(ROWS)), "island 0's two pools split the data once"
    assert sorted(got["poolC-i1-0"]) == list(range(ROWS)), "island 1 covers the data on its own"
    for k in (0, 1):
        totals = sd.status(sd.DirObjects(run / f"trainer{k}"), NS, details=False)["totals"]
        assert totals["trained"] == ROWS and totals["ended"]
    assert not (run / "fleet").exists()


def test_a_replaced_pool_member_gives_back_its_unconfirmed_slots_and_keeps_its_trained_ones(tmp_path, monkeypatch):
    """The pool replaces a sampler of island 0 (lease end): it fences the shard
    through the pool's data path, which retires the sampler in the shared ledger.
    Its open claims go back at once to any sampler of the fleet; the groups it
    wrote stay trained and are never given out again."""
    from yeto.launch.nc import elastic
    from yeto.launch.pool.data_path import R2DataPath

    run, path = tmp_path / "run", _prompts(tmp_path)
    old = _sampler(_spec(run, 0, "poolA-i0-0", path, shared=True, groups=4), inc="a")
    other = _sampler(_spec(run, 1, "poolC-i1-0", path, shared=True, groups=2))
    state, held = old.poll(4)
    assert state == "items" and len(held) == 4
    old.written(0, held[:2], keep=held[2:])  # two groups confirmed, two still running
    trained, unconfirmed = held[:2], held[2:]

    monkeypatch.setattr(elastic, "fence_shard", lambda records, shard, by, **kw: {"incarnation": f"fence:{by}"})
    ledger = sd.DirObjects(run / "fleet" / "dispense")
    R2DataPath(records=object(), dispense=(ledger, NS)).fence_shard("poolA-i0-0", by="ticket-2")
    assert not any(old.still_holds(p) for p in unconfirmed), "the replaced sampler writes no group"
    assert old.written(0, unconfirmed)["trained"] == 0, "its late report settles nothing"

    new = _sampler(_spec(run, 0, "poolA-i0-1", path, shared=True))  # the replacement: a new shard of island 0
    try:
        got = _drain({"poolA-i0-1": new, "poolC-i1-0": other})
    finally:
        for s in (old, new, other):
            s.close()
    again = got["poolA-i0-1"] + got["poolC-i1-0"]
    assert not set(again) & set(trained), "a confirmed slot is never dispensed again"
    assert set(unconfirmed) <= set(again), "the unconfirmed slots go out again"
    assert sorted(trained + again) == list(range(ROWS)), "nothing lost, nothing twice"
    tasks = {t["slot"]: t for t in sd.status(ledger, NS)["tasks"]}
    assert all(tasks[p]["state"] == "trained" for p in range(ROWS))
    for p in unconfirmed:
        assert tasks[p]["fails"] == {"0": "release"} and tasks[p]["failure_reasons"] == ["0:release:retired"]
        assert tasks[p]["holder"] in ("poolA-i0-1", "poolC-i1-0")
    for p in trained:
        assert tasks[p]["holder"] == "poolA-i0-0" and tasks[p]["fails"] == {}


def test_a_restarted_fleet_sampler_recovers_from_its_own_shard_into_the_shared_ledger(tmp_path):
    run, path = tmp_path / "run", _prompts(tmp_path, 6)
    first = _sampler(_spec(run, 1, "poolC-i1-0", path, shared=True, groups=3), inc="a")
    _state, spots = first.poll(3)

    class Entry:  # slot spots[0] reached the island's buffer before the process died
        sampler_round = {"sampler": "poolC-i1-0", "incarnation": "a", "dispense_position": spots[0]}

    second = _sampler(_spec(run, 1, "poolC-i1-0", path, shared=True, groups=3), inc="b")
    peer = _sampler(_spec(run, 0, "poolA-i0-0", path, shared=True, groups=3))
    try:
        assert second.recover([Entry()]) == {"recovered": 1, "released": 2}
        got = _drain({"second": second, "peer": peer})
    finally:
        for s in (first, second, peer):
            s.close()
    assert sorted(got["second"] + got["peer"]) == sorted(set(range(6)) - {spots[0]})


def test_a_pool_with_another_prompt_file_is_refused_by_the_shared_ledger(tmp_path):
    run = tmp_path / "run"
    first = _sampler(_spec(run, 0, "poolA-i0-0", _prompts(tmp_path), shared=True))
    try:
        with pytest.raises(entry.SamplerSpecError, match="dispense mismatch"):
            _sampler(_spec(run, 1, "poolC-i1-0", _prompts(tmp_path, ROWS + 1, "other.jsonl"), shared=True))
    finally:
        first.close()


def test_spec_dispense_store_resolution(tmp_path):
    assert sd.fleet_dispense_store("s3://b/runs/r1/") == "s3://b/runs/r1/fleet/dispense"
    base = {"store": "s3://b/runs/r1/trainer0", "prompts": {"dispense": {"backend": "store"}}}
    assert sd.spec_dispense_store(base) is None
    assert sd.spec_dispense_store({**base, "dispense_store": "s3://b/runs/r1/fleet/dispense"}) == \
        "s3://b/runs/r1/fleet/dispense"
    named = {**base, "prompts": {"dispense": {"backend": "store", "store": "r2://b/runs/r1/fleet/dispense/"}}}
    assert sd.spec_dispense_store(named) == "r2://b/runs/r1/fleet/dispense/"
    assert sd.spec_dispense_store({**named, "dispense_store": "s3://b/runs/r1/fleet/dispense"}) == \
        "r2://b/runs/r1/fleet/dispense/", "the same store named twice"
    with pytest.raises(DispenseError, match="two ledgers"):
        sd.spec_dispense_store({**named, "dispense_store": "s3://b/runs/r1/trainer0"})
    for bad in ("", "  ", 3):
        with pytest.raises(DispenseError, match="dispense_store"):
            sd.spec_dispense_store({**base, "dispense_store": bad})
    spec = _spec(tmp_path / "run", 0, "poolA-i0-0", _prompts(tmp_path), shared=True)
    spec["prompts"]["dispense"]["store"] = str(tmp_path / "elsewhere")
    with pytest.raises(entry.SamplerSpecError, match="two ledgers"):
        _sampler(spec)


def test_without_dispense_store_the_ledger_stays_in_the_samplers_store(tmp_path):
    spec = _spec(tmp_path / "run", 0, "nc-0", _prompts(tmp_path), shared=False)
    spec["store"] = str(tmp_path / "store")
    s = _sampler(spec)
    try:
        assert (tmp_path / "store" / NS / "plan.json").exists()
    finally:
        s.close()


def test_pool_dispense_store_names_the_shared_ledger(tmp_path):
    from yeto.launch.nc.cli import pool_dispense_store

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({"prompts": {"dispense": {"backend": "store", "namespace": NS}}}))
    assert pool_dispense_store({"spec": str(spec_path)}) is None
    assert pool_dispense_store({}) is None
    spec_path.write_text(json.dumps({"dispense_store": "s3://b/runs/r1/fleet/dispense",
                                     "prompts": {"dispense": {"backend": "store", "namespace": NS}}}))
    assert pool_dispense_store({"spec": str(spec_path)}) == "s3://b/runs/r1/fleet/dispense"
    own = {"spec": str(spec_path), "sampler": {"dispense": {"backend": "store", "namespace": NS}}}
    assert pool_dispense_store(own) == "s3://b/runs/r1/fleet/dispense", "the pool's dispense knob keeps the spec's"
    own["sampler"]["dispense"]["store"] = "s3://b/runs/r1/other"
    assert pool_dispense_store(own) == "s3://b/runs/r1/other"
    spec_path.write_text(json.dumps({"dispense_store": "s3://b/runs/r1/fleet/dispense", "prompts": {}}))
    assert pool_dispense_store({"spec": str(spec_path)}) is None, "no store dispenser: no ledger"


# ------------------------------------------------------------------ run up


def _fleet_run(tmp_path, monkeypatch, *, shared: bool):
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    run_store = tmp_path / "run"
    entries = []
    for name, island in (("i0-a", 0), ("i0-b", 0), ("i1-c", 1)):
        spec = {"store": str(run_store / f"trainer{island}"),
                "prompts": {"dispense": {"backend": "store", "namespace": NS}}}
        if shared:
            spec["dispense_store"] = sd.fleet_dispense_store(str(run_store))
        spec_path = tmp_path / f"{name}.spec.json"
        spec_path.write_text(json.dumps(spec))
        entries.append({"name": name, "shard_base": f"{name}-base", "spec": str(spec_path)})
    cfg = {"run": "r1", "store": str(run_store), "samplers": entries,
           "trainer": {"argv": ["--gpu", "modal:2x8xb300"], "usd_per_h": 100},
           "first_group": {"timeout_s": 0, "poll_s": 0}, "budget": {}}
    return up.Run(cfg), run_store


def test_run_up_reads_one_shared_ledger_for_every_pool_of_the_fleet(tmp_path, monkeypatch):
    run, run_store = _fleet_run(tmp_path, monkeypatch, shared=True)
    fleet = sd.fleet_dispense_store(str(run_store))
    assert [run.dispense_ledger(i) for i in range(3)] == [(fleet, NS)] * 3
    assert run.dispense_ledgers() == [(fleet, NS, ["i0-a", "i0-b", "i1-c"])]
    assert run._namespace() == NS

    calls = []
    monkeypatch.setattr(up.subprocess, "run", lambda argv, **k: calls.append(argv) or types.SimpleNamespace(
        stdout="", stderr="", returncode=0))
    monkeypatch.setattr(run, "_objects", lambda: [])
    monkeypatch.setattr(run, "run_containers", lambda: None)
    out = run.status()
    dispense = [a for a in calls if "yeto.rl.engine.store_dispense" in a]
    assert len(dispense) == 1 and dispense[0][dispense[0].index("--store") + 1] == fleet
    assert "dispense_ledgers" not in out

    # the shared ledger settles once every slot is trained (dispense_settled reads it, not the run root)
    monkeypatch.setattr(run, "_objects", lambda: sd.DirObjects(run_store))
    ledger = sd.DirObjects(Path(fleet))
    ledger.write(f"{NS}/plan.json", json.dumps({"rows": 2, "passes": 1, "slots": 2}).encode())
    ledger.create(f"{NS}/done/{0:012d}.trained", b"{}")
    assert not run.dispense_settled()
    ledger.create(f"{NS}/done/{1:012d}.trained", b"{}")
    assert run.dispense_settled()


def test_run_up_reads_each_islands_own_ledger_without_dispense_store(tmp_path, monkeypatch):
    run, run_store = _fleet_run(tmp_path, monkeypatch, shared=False)
    # the pools write their trainer substores (pool_store): an island's own ledger each
    monkeypatch.setattr(up.up_consistency, "pool_store",
                        lambda cfg: json.loads(Path(cfg["spec"]).read_text())["store"])
    t0, t1 = str(run_store / "trainer0"), str(run_store / "trainer1")
    assert run.dispense_ledgers() == [(t0, NS, ["i0-a", "i0-b"]), (t1, NS, ["i1-c"])]
    calls = []
    monkeypatch.setattr(up.subprocess, "run", lambda argv, **k: calls.append(argv) or types.SimpleNamespace(
        stdout=argv[argv.index("--store") + 1] if "--store" in argv else "", stderr="", returncode=0))
    monkeypatch.setattr(run, "_objects", lambda: [])
    monkeypatch.setattr(run, "run_containers", lambda: None)
    out = run.status()
    assert out["dispense"].strip() == t0
    assert [(r["store"], r["pools"]) for r in out["dispense_ledgers"]] == [(t0, ["i0-a", "i0-b"]), (t1, ["i1-c"])]


def test_run_up_keeps_the_run_store_ledger_of_a_plain_run(tmp_path, monkeypatch):
    """No fleet: the pool writes the run store (or says nothing of it), so the ledger
    is read where it always was: the run config's store, verbatim."""
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps({"shard_base": "s", "sampler": {"dispense": {"backend": "store", "namespace": "k31/d"}}}))
    run = up.Run({"run": "r1", "store": "s3://b/runs/r1/", "pool": {"config": str(pool)},
                  "trainer": {"argv": ["--gpu", "modal:2x8xb300"], "usd_per_h": 100},
                  "first_group": {"timeout_s": 0, "poll_s": 0}, "budget": {}})
    assert run.dispense_ledgers() == [("s3://b/runs/r1/", "k31/d", ["pool"])]
    monkeypatch.setattr(up.up_consistency, "pool_store", lambda cfg: "s3://b/runs/r1")
    assert run.dispense_ledger(0) == ("s3://b/runs/r1/", "k31/d")
