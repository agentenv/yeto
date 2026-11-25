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


# ------------------------------------------------------------------ a ledger by store URI
# A fleet compiles dispense_store as a store URI (<run store>/fleet/dispense); on a pool
# island the sampler's own store is an s3 store (store_from_env: <run store>/trainer<k>,
# by URI_ENV, staged), and no island path names the fleet's ledger. The sampler opens it
# by URI with its own store implementation, credentials and stage.

BUCKET = "bkt"
RUN_URI = f"s3://{BUCKET}/runs/fleet-r"


def _s3_island(monkeypatch, tmp_path, s3, island: int) -> dict[str, str]:
    """The environment of a pool island whose sampler writes ``<run>/trainer<island>``
    (modal_provider.store_env / ssh_sampler.sampler_env), with ``s3`` as the store."""
    from yeto.cloud import s3_store
    from yeto.rl.engine.resume import STORE_IMPL_ENV

    monkeypatch.setattr(s3_store, "_client", lambda environ, require_credentials: s3)
    return {STORE_IMPL_ENV: s3_store.STORE_IMPL, s3_store.URI_ENV: f"{RUN_URI}/trainer{island}",
            s3_store.STAGE_ENV: str(tmp_path / f"stage{island}"), SAMPLER_INCARNATION_ENV: "inc-1"}


def _uri_spec(tmp_path, island: int, shard: str, prompts: Path, ledger: str | None, **dispense) -> dict:
    own = tmp_path / f"stage{island}" / BUCKET / "runs" / "fleet-r" / f"trainer{island}"
    spec = {"sampler_id": shard, "store": str(own),
            "prompts": {"path": str(prompts), "input_key": "text", "label_key": "label", "groups_per_round": 2,
                        "dispense": {"backend": "store", "namespace": NS, "scan_s": 0.0, "max_scan_s": 0.0,
                                     **dispense}}}
    if ledger is not None:
        spec["dispense_store"] = ledger
    return spec


def test_a_sampler_opens_a_fleet_ledger_named_by_store_uri_and_two_pools_share_it(tmp_path, monkeypatch):
    from tests.test_sampler_store_read_failures import OutageS3
    from yeto.cloud.s3_store import S3ObjectStore

    s3, path = OutageS3(), _prompts(tmp_path)
    fleet = sd.fleet_dispense_store(RUN_URI)
    samplers = {}
    for shard, island, ledger in (("vllm-i0-0", 0, fleet), ("sgl-i1-0", 1, fleet.replace("s3://", "r2://", 1))):
        env = _s3_island(monkeypatch, tmp_path, s3, island)
        prompts = entry.build_prompts(_uri_spec(tmp_path, island, shard, path, ledger), env)
        assert isinstance(prompts, sd.StoreDispensedPrompts)
        samplers[shard] = prompts
    try:
        got = _drain(samplers)
    finally:
        for s in samplers.values():
            s.close()
    slots = [p for ps in got.values() for p in ps]
    assert sorted(slots) == list(range(ROWS)), "every slot once across the two pools, none twice"
    assert all(got.values()), "both pools took a share"
    keys = {k for b, k in s3.objects if b == BUCKET}
    assert "runs/fleet-r/fleet/dispense/task-dispense/plan.json" in keys
    assert not [k for k in keys if k.startswith(("runs/fleet-r/trainer0/task-dispense",
                                                 "runs/fleet-r/trainer1/task-dispense"))]
    totals = sd.status(S3ObjectStore(fleet, client=s3), NS, details=False)["totals"]
    assert totals["trained"] == ROWS and totals["ended"]


def test_prompts_dispense_store_takes_a_store_uri_as_dispense_store_does(tmp_path, monkeypatch):
    from tests.test_sampler_store_read_failures import OutageS3

    s3 = OutageS3()
    env = _s3_island(monkeypatch, tmp_path, s3, 0)
    spec = _uri_spec(tmp_path, 0, "nc-0", _prompts(tmp_path), None, store=f"{RUN_URI}/shared")
    s = entry.build_prompts(spec, env)
    try:
        assert (BUCKET, "runs/fleet-r/shared/task-dispense/plan.json") in s3.objects
    finally:
        s.close()


def test_a_ledger_uri_without_an_object_store_implementation_is_a_spec_error(tmp_path):
    spec = _uri_spec(tmp_path, 0, "nc-0", _prompts(tmp_path), sd.fleet_dispense_store(RUN_URI))
    with pytest.raises(entry.SamplerSpecError, match="fleet/dispense.*is a URI"):
        entry.build_prompts(spec, {SAMPLER_INCARNATION_ENV: "inc-1"})


def test_store_at_opens_a_uri_like_the_islands_own_store(tmp_path):
    from tests.test_rl_object_store import FakeS3
    from yeto.cloud import s3_store
    from yeto.rl.engine.resume import STORE_IMPL_ENV, CheckpointStore, store_at

    s3 = FakeS3()
    env = {STORE_IMPL_ENV: s3_store.STORE_IMPL, s3_store.URI_ENV: f"{RUN_URI}/trainer0"}
    plain = store_at(tmp_path / "x", environ={})
    assert type(plain) is CheckpointStore and plain.root == tmp_path / "x"
    mounted = store_at(f"{RUN_URI}/fleet/dispense", environ=env, client=s3)
    assert isinstance(mounted, s3_store.S3Store) and mounted.uri == f"{RUN_URI}/fleet/dispense"
    assert not mounted.staged
    staged_env = {**env, s3_store.STAGE_ENV: str(tmp_path / "stage")}
    staged = store_at(f"r2://{BUCKET}/runs/fleet-r/fleet/dispense/", environ=staged_env, client=s3)
    assert staged.uri == f"{RUN_URI}/fleet/dispense" and staged.staged
    assert staged.root == tmp_path / "stage" / BUCKET / "runs" / "fleet-r" / "fleet" / "dispense"
    staged.objects.write("k", b"v")
    assert s3.objects[(BUCKET, "runs/fleet-r/fleet/dispense/k")] == b"v"
    # the store the island's own hook (store_from_env, as on a Modal r2 pool: staged) builds
    # when pointed at the ledger: the same URI, root and stage
    via_env = s3_store.store_from_env(tmp_path / "stage" / BUCKET / "runs" / "fleet-r" / "fleet" / "dispense",
                                      environ={**staged_env, s3_store.URI_ENV: f"{RUN_URI}/fleet/dispense"},
                                      client=s3)
    assert (via_env.uri, via_env.root, via_env.staged) == (staged.uri, staged.root, staged.staged)
    with pytest.raises(ValueError, match="opens none"):
        store_at(f"{RUN_URI}/fleet/dispense", environ={})


def test_the_pool_opens_a_shared_ledger_by_s3_or_r2_uri_or_path(tmp_path, monkeypatch):
    from tests.test_rl_object_store import FakeS3
    from yeto.cloud import s3_store
    from yeto.launch.nc import cli as nc_cli

    s3 = FakeS3()
    monkeypatch.setattr(s3_store, "_client", lambda environ, require_credentials: s3)
    env_file = tmp_path / "r2.env"
    env_file.write_text("R2_BUCKET=bkt\nAWS_ENDPOINT_URL=https://e\n")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({"prompts": {"dispense": {"backend": "store", "namespace": NS}}}))
    for shared in (f"{RUN_URI}/fleet/dispense", f"r2://{BUCKET}/runs/fleet-r/fleet/dispense"):
        ops = types.SimpleNamespace(cfg={"spec": str(spec_path), "r2_env_file": str(env_file),
                                         "sampler": {"dispense_store": shared}})
        objects, ns = nc_cli.NcSshOps._dispense_ledger(ops)
        assert ns == NS and (objects.bucket, objects.prefix) == (BUCKET, "runs/fleet-r/fleet/dispense")
    ops = types.SimpleNamespace(cfg={"spec": str(spec_path), "r2_env_file": str(env_file),
                                     "sampler": {"dispense_store": str(tmp_path / "ledger")}})
    objects, _ns = nc_cli.NcSshOps._dispense_ledger(ops)
    assert isinstance(objects, sd.DirObjects)


def test_a_local_ledger_create_is_never_seen_half_written(tmp_path, monkeypatch):
    """Two samplers opening a fresh local ledger at once: the loser reads the winner's
    plan.json, so the body is in place before the key exists (staged, then linked),
    never an empty file it would take for another plan."""
    import os

    objects = sd.DirObjects(tmp_path)
    seen = []
    real_link = os.link

    def link(src, dst):
        seen.append((Path(dst).exists(), Path(src).read_bytes()))  # what a reader could see before the key exists
        return real_link(src, dst)

    monkeypatch.setattr(sd.os, "link", link)
    assert objects.create("ns/plan.json", b'{"version": 3}') is True
    assert seen == [(False, b'{"version": 3}')]
    assert objects.create("ns/plan.json", b'{"version": 4}') is False
    assert objects.read("ns/plan.json") == b'{"version": 3}'
    assert sorted(p.name for p in (tmp_path / "ns").iterdir()) == ["plan.json"]  # no staged file left
