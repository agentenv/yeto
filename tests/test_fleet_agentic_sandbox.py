"""Agentic pools of an island fleet and the sandbox readiness manifest.

``yeto sandbox prepare`` (``yeto run up``'s sandbox step) writes the readiness
manifest once, under ``agentic.envs.ready_key`` of the run store. A fleet pool's
own store is ``<run store>/trainer<k>`` (the compile moves its ``run_prefix``), so
its dispenser gate would read ``trainer<k>/sandbox/ready.json``, which nobody
writes, and never claim a task. The compile names the run store in the pool's
``sandbox_store`` knob (spec key ``sandbox_store``); the gate reads the manifest
there. Without the knob (every single-trainer run) the gate reads the sampler's own
store, as before."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yeto.launch.nc import cli as nc_cli
from yeto.launch.nc import tasks
from yeto.rl import fleet, up, up_consistency
from yeto.rl import fleet_sandbox as fsb
from yeto.rl.engine import engine_sampler_entry as entry
from yeto.rl.engine import store_dispense as sd
from yeto.rl.engine.task_dispense import SAMPLER_INCARNATION_ENV
from yeto.sandbox import ready as ready_mod
from yeto.sandbox.gate import spec_sandbox_store

BUCKET = "bkt"
RUN_URI = f"s3://{BUCKET}/runs/fleet-a"
NS = "task-dispense"
TASKS = ("t0", "t1", "t2", "t3")
ENVS = {"run": "fleet-a", "hosts": [{"name": "h1", "docker_host": "ssh://sbx@h1"}]}


def _prompts(tmp_path: Path) -> Path:
    path = tmp_path / "prompts.jsonl"
    path.write_text("".join(json.dumps({"text": f"q{i}", "label": "", "metadata": {"task_id": t}}) + "\n"
                            for i, t in enumerate(TASKS)))
    return path


def _prepared(objects, ready: tuple[str, ...]) -> None:
    """What ``yeto sandbox prepare`` + ``validate`` leave: ``ready`` prepared and validated on h1."""
    ready_mod.merge_write(objects, "sandbox/ready.json", "fleet-a", tasks={
        t: {"portable": True, "hosts": {"h1": {"prepared": True, "validation": {"passed": True}}}} for t in ready})


def _island_env(monkeypatch, tmp_path, s3, island: int) -> dict[str, str]:
    """A fleet pool island whose sampler writes ``<run>/trainer<island>`` (the s3 store
    hook, staged, as on a Modal / NC pool)."""
    from yeto.cloud import s3_store
    from yeto.rl.engine.resume import STORE_IMPL_ENV

    monkeypatch.setattr(s3_store, "_client", lambda environ, require_credentials: s3)
    return {STORE_IMPL_ENV: s3_store.STORE_IMPL, s3_store.URI_ENV: f"{RUN_URI}/trainer{island}",
            s3_store.STAGE_ENV: str(tmp_path / f"stage{island}"), SAMPLER_INCARNATION_ENV: "inc-1"}


def _pool_spec(tmp_path: Path, island: int, shard: str, prompts: Path, *, sandbox_store: str | None) -> dict:
    own = tmp_path / f"stage{island}" / BUCKET / "runs" / "fleet-a" / f"trainer{island}"
    spec = {"sampler_id": shard, "store": str(own), "sampling": {"n": 1},
            "dispense_store": sd.fleet_dispense_store(RUN_URI),
            "prompts": {"path": str(prompts), "input_key": "text", "label_key": "label", "groups_per_round": 4,
                        "dispense": {"backend": "store", "namespace": NS, "scan_s": 0.0, "max_scan_s": 0.0}},
            "agentic": {"envs": dict(ENVS)}}  # no provider: no process sandbox cap in the gate
    if sandbox_store is not None:
        spec["sandbox_store"] = sandbox_store
    return spec


def _taken_tasks(prompts) -> list[str]:
    state, slots = prompts.poll(len(TASKS))  # one claim pass, never waiting
    return sorted(TASKS[prompts.slots.row(s)] for s in slots) if state == "items" else []


# ------------------------------------------------------------------ the gate passes


def test_an_agentic_fleet_pool_reads_the_manifest_prepare_wrote_in_the_run_store(tmp_path, monkeypatch):
    from tests.test_sampler_store_read_failures import OutageS3
    from yeto.cloud.s3_store import S3ObjectStore

    s3, path = OutageS3(), _prompts(tmp_path)
    _prepared(S3ObjectStore(RUN_URI, client=s3), ("t0", "t2"))  # the run store, as yeto sandbox prepare --store
    env = _island_env(monkeypatch, tmp_path, s3, 1)
    prompts = entry.store_dispensed_prompts(_pool_spec(tmp_path, 1, "pool-i1-0", path, sandbox_store=RUN_URI),
                                            environ=env)
    try:
        assert prompts.gate is not None
        view = prompts.gate.view()
        assert [view.classify(t)[0] for t in TASKS] == ["ready", "pending", "ready", "pending"]
        assert _taken_tasks(prompts) == ["t0", "t2"], "the gate passes the tasks prepare made ready"
    finally:
        prompts.close()
    keys = {k for b, k in s3.objects if b == BUCKET}
    assert "runs/fleet-a/sandbox/ready.json" in keys
    assert not [k for k in keys if k.startswith("runs/fleet-a/trainer1/sandbox")], "nothing read or written there"


def test_without_sandbox_store_a_fleet_pool_finds_no_manifest_and_claims_nothing(tmp_path, monkeypatch):
    """The failure the knob fixes: the gate reads ``trainer<k>/sandbox/ready.json``."""
    from tests.test_sampler_store_read_failures import OutageS3
    from yeto.cloud.s3_store import S3ObjectStore

    s3, path = OutageS3(), _prompts(tmp_path)
    _prepared(S3ObjectStore(RUN_URI, client=s3), TASKS)
    env = _island_env(monkeypatch, tmp_path, s3, 0)
    prompts = entry.store_dispensed_prompts(_pool_spec(tmp_path, 0, "pool-i0-0", path, sandbox_store=None),
                                            environ=env)
    try:
        assert {prompts.gate.view().classify(t)[0] for t in TASKS} == {"pending"}
        assert prompts.poll(len(TASKS)) == ("wait", [])
    finally:
        prompts.close()


def test_a_sandbox_store_path_on_a_directory_store(tmp_path):
    """Local directory stores (CPU rehearsals): ``sandbox_store`` may be a path."""
    run, path = tmp_path / "run", _prompts(tmp_path)
    _prepared(sd.DirObjects(run), ("t1",))
    spec = _pool_spec(tmp_path, 0, "pool-i0-0", path, sandbox_store=str(run))
    spec.update(store=str(run / "trainer0"), dispense_store=sd.fleet_dispense_store(str(run)))
    prompts = entry.store_dispensed_prompts(spec, environ={SAMPLER_INCARNATION_ENV: "inc-1"})
    try:
        assert _taken_tasks(prompts) == ["t1"]
    finally:
        prompts.close()


def test_a_bad_sandbox_store_is_a_spec_error(tmp_path):
    spec = _pool_spec(tmp_path, 0, "pool-i0-0", _prompts(tmp_path), sandbox_store=RUN_URI)
    del spec["dispense_store"]
    with pytest.raises(entry.SamplerSpecError, match="sandbox readiness manifest's store.*is a URI"):
        entry.store_dispensed_prompts(spec, environ={SAMPLER_INCARNATION_ENV: "inc-1"})  # no object store hook
    with pytest.raises(ValueError, match="sandbox_store must be a store path or URI"):
        spec_sandbox_store({"sandbox_store": ""})
    with pytest.raises(ValueError, match="pool sampler config: sandbox_store must be a store URI"):
        nc_cli._overrides({"sandbox_store": 3})


# ------------------------------------------------------------------ single trainer: unchanged


def test_a_single_trainer_agentic_sampler_still_reads_its_own_store(tmp_path):
    store, path = tmp_path / "store", _prompts(tmp_path)
    _prepared(sd.DirObjects(store), ("t3",))
    spec = _pool_spec(tmp_path, 0, "nc-0", path, sandbox_store=None)
    spec.update(store=str(store))
    del spec["dispense_store"]
    prompts = entry.store_dispensed_prompts(spec, environ={SAMPLER_INCARNATION_ENV: "inc-1"})
    try:
        assert _taken_tasks(prompts) == ["t3"]
    finally:
        prompts.close()


def test_the_override_without_sandbox_store_leaves_the_spec_as_it_was():
    spec = {"sampler_id": "x", "prompts": {}, "sampling": {}, "agentic": {"envs": dict(ENVS)}}
    knobs = nc_cli._overrides({"n": 4})
    assert knobs["sandbox_store"] is None
    out = tasks.apply_sampler_overrides(spec, **knobs)
    assert "sandbox_store" not in out
    assert out == tasks.apply_sampler_overrides(spec, n=4)
    assert tasks.apply_sampler_overrides(spec, sandbox_store=RUN_URI)["sandbox_store"] == RUN_URI


def test_the_k31_customer_pool_renders_no_sandbox_store():
    root = Path(__file__).resolve().parents[1] / "examples" / "k31"
    pool = json.loads((root / "pool.nc-mi355x-agentic-262k.json").read_text())
    pool["spec"] = str(root / "sampler-spec.nc-mi355x-vllm-agentic-262k.json")
    spec = up_consistency.effective_spec(pool)
    assert spec is not None and spec.get("agentic") and "sandbox_store" not in spec


# ------------------------------------------------------------------ the fleet compile


@pytest.fixture
def fleet_files(tmp_path, monkeypatch):
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    base = {"sampler_id": "x", "store": "/work/stage/b/runs/__RUN_PREFIX__", "engine": {"adapter": "vllm-server"},
            "event_tape": "/work/sampler/events-__RUN_PREFIX__.jsonl", "sampling": {"n": 8},
            "prompts": {"path": "p.jsonl", "input_key": "q", "groups_per_round": 1, "format": "agentic",
                        "dispense": {"backend": "store", "namespace": "task-dispense"}}}
    (tmp_path / "agentic.json").write_text(json.dumps(dict(base, agentic={
        "max_in_flight": 8, "task_defaults": {"provider": "remote-docker"}, "envs": dict(ENVS)})))
    (tmp_path / "plain.json").write_text(json.dumps(dict(base, prompts=dict(base["prompts"], format="text"))))
    (tmp_path / "store.env").write_text("R2_BUCKET=b\n")
    return tmp_path


def _pool(tmp_path: Path, name: str, spec: str, base: str) -> dict:
    return {"name": name, "provider": {"kind": "modal", "image": "i", "secret": "s", "bucket": "b", "gpu": "H100"},
            "spec": str(tmp_path / spec), "run_prefix": "r", "r2_env_file": str(tmp_path / "store.env"),
            "shard_base": base, "shard_limit": 2, "target": 1, "slots": [1, 2], "run": "r", "usd_per_h": 4.0}


def _fleet_cfg(tmp_path: Path) -> dict:
    from tests.test_run_up_fleet import STORE, TRAINER_ARGV

    return {"run": "r", "store": STORE, "budget": {"usd": 40, "wall_h": 4, "nc_usd_per_h": 0},
            "fleet": {"trainer_argv": list(TRAINER_ARGV)},
            "islands": [{"name": "east", "trainer": {"gpu": "modal:1x2xh100", "usd_per_h": 8.0},
                         "samplers": [_pool(tmp_path, "agent", "agentic.json", "ag"),
                                      _pool(tmp_path, "plain", "plain.json", "pl")]},
                        {"name": "west", "trainer": {"gpu": "modal:1x2xh100", "usd_per_h": 8.0},
                         "samplers": [_pool(tmp_path, "agent", "agentic.json", "ag")]}]}


def test_the_compile_names_the_run_store_for_agentic_pools_only(fleet_files):
    from tests.test_run_up_fleet import STORE

    cfg, plan = fleet.compile_fleet(_fleet_cfg(fleet_files))
    knobs = {p["name"]: p["sampler"] for p in cfg["samplers"]}
    assert knobs["i0-agent"]["sandbox_store"] == STORE and knobs["i1-agent"]["sandbox_store"] == STORE
    assert "sandbox_store" not in knobs["i0-plain"], "a pool without agentic.envs gets no knob"
    for pool in cfg["samplers"]:
        spec = up_consistency.effective_spec(pool)
        island = plan.island_of_pool(pool["name"])
        assert spec["store"] == f"/work/stage/b/runs/r/trainer{island.index}"  # its own store moved
        assert spec.get("sandbox_store") == (STORE if pool["name"].endswith("agent") else None)


def test_a_pool_may_not_set_sandbox_store_itself(fleet_files):
    cfg = _fleet_cfg(fleet_files)
    cfg["islands"][0]["samplers"][0]["sampler"] = {"sandbox_store": "s3://b/elsewhere"}
    with pytest.raises(ValueError, match="sampler.sandbox_store is written by the fleet compile"):
        fleet.compile_fleet(cfg)


# ------------------------------------------------------------------ the fleet's sandbox preflight


def _row(name: str, island: int, provider: str | None, *, mif: int = 16, nodes: int = 1, max_nodes: int | None = None,
         caps: dict | None = None, envs: dict | None = None, docker_host: str | None = None,
         gateway: str | None = None, base: str | None = None) -> fsb.SandboxPool:
    base = base or name.split("-", 1)[1]
    return fsb.SandboxPool(name=name, where=f"island {island} (n{island}), pool {name}", island=island,
                           provider=provider, max_in_flight=mif,
                           shards=tuple(f"{base}-i{island}-{n}" for n in range(nodes)),
                           max_nodes=max_nodes or nodes, caps=caps or {}, envs=envs, docker_host=docker_host,
                           gateway=gateway)


RD_CAP = {"max_sandboxes": {"remote-docker": 16}}


def test_provider_lines_name_each_providers_pools_and_warn_on_mixed_islands():
    pools = [_row("i0-a", 0, "remote-docker", caps=RD_CAP), _row("i1-b", 1, "boat")]
    problems, notes = fsb.provider_findings(pools, sandbox={})
    assert problems == []
    assert "fleet: sandbox provider boat: island 1 (n1), pool i1-b" in notes
    assert "fleet: sandbox provider remote-docker: island 0 (n0), pool i0-a" in notes
    assert any(n.startswith("WARNING: fleet: the islands' trials run on different sandbox providers (island 0: "
                            "remote-docker; island 1: boat)") for n in notes)
    same = [_row("i0-a", 0, "boat"), _row("i1-a", 1, "boat")]
    assert not any("different sandbox providers" in n for n in fsb.provider_findings(same, sandbox={})[1])


def test_the_refusals_a_sampler_would_raise_at_start_are_named_in_the_preflight():
    pools = [_row("i0-a", 0, "remote-docker"),                                    # no cap
             _row("i1-a", 1, "remote-docker", mif=32, caps=RD_CAP),              # above its cap
             _row("i1-b", 1, "boat", mif=120, caps={"boat_max_active_n": 100}),
             _row("i0-c", 0, "agentenv", mif=80, caps={"agentenv_max_active_n": 64})]
    problems, _notes = fsb.provider_findings(pools, sandbox={})
    assert any(p.startswith("island 0 (n0), pool i0-a: provider remote-docker needs harness.sandbox_caps."
                            "max_sandboxes") for p in problems)
    assert any(p.startswith("island 1 (n1), pool i1-a: agentic.max_in_flight 32 exceeds its remote-docker sandbox "
                            "cap 16") for p in problems)
    assert any("pool i1-b: agentic.max_in_flight 120 exceeds harness.sandbox_caps.boat_max_active_n 100" in p
               for p in problems)
    assert any("pool i0-c: agentic.max_in_flight 80 exceeds harness.sandbox_caps.agentenv_max_active_n 64" in p
               for p in problems)


def test_a_gated_pool_needs_the_run_to_prepare_the_manifest():
    pool = [_row("i1-a", 1, "remote-docker", caps=RD_CAP, envs=dict(ENVS))]
    assert fsb.provider_findings(pool, sandbox={"envs": "envs.json"})[0] == []
    for sandbox, why in (({}, "the run config has no sandbox.envs"),
                         ({"provider": "agentenv", "envs": "envs.json"}, "sandbox.provider is agentenv")):
        problems, _ = fsb.provider_findings(pool, sandbox=sandbox)
        assert len(problems) == 1 and problems[0].startswith("island 1 (n1), pool i1-a: its sampler spec has "
                                                             "agentic.envs") and why in problems[0]


def test_an_envs_label_built_from_the_islands_run_prefix_is_refused():
    pool = [_row("i0-a", 0, "remote-docker", caps=RD_CAP, envs=dict(ENVS, run="r/trainer0"))]
    problems, _ = fsb.provider_findings(pool, sandbox={"envs": "envs.json"})
    assert problems == ["island 0 (n0), pool i0-a: agentic.envs.run 'r/trainer0' is not a label of [A-Za-z0-9_.-] "
                        "(a fleet pool's run_prefix ends in /trainer<k>; do not build the label from "
                        "__RUN_PREFIX__): its samplers fail at start"]


def test_boat_pools_of_every_island_share_the_account_limit():
    ok = [_row("i0-a", 0, "boat", mif=30), _row("i1-a", 1, "boat", mif=30)]
    problems, notes = fsb.capacity_findings(ok)
    assert problems == [] and any("the boat account holds 100 active sandboxes and the fleet's boat pools hold 60" in n
                                  for n in notes)
    over = [_row("i0-a", 0, "boat", mif=30, nodes=2), _row("i1-a", 1, "boat", mif=30, nodes=2)]
    problems, notes = fsb.capacity_findings(over)
    assert len(problems) == 1 and "hold 120 at once (island 0 (n0), pool i0-a: 2 sampler(s) x 30; island 1 (n1), " \
                                  "pool i1-a: 2 sampler(s) x 30)" in problems[0] and "HTTP 429" in problems[0]
    assert any(n.startswith("WARNING: fleet: each boat sampler may start") and "together 48 > the account's 12" in n
               for n in notes)
    peak = [_row("i0-a", 0, "boat", mif=30, max_nodes=2), _row("i1-a", 1, "boat", mif=30, max_nodes=2)]
    problems, notes = fsb.capacity_findings(peak)
    assert problems == [] and any(n.startswith("WARNING: the boat account holds 100") and "120 at the pools' elastic "
                                  "maximum" in n for n in notes)


def test_agentenv_pools_on_one_gateway_share_its_limit():
    caps = {"agentenv_max_active_n": 32}
    pools = [_row("i0-a", 0, "agentenv", mif=20, caps=caps, gateway="https://gw1"),
             _row("i1-a", 1, "agentenv", mif=20, caps=caps, gateway="https://gw1"),
             _row("i1-b", 1, "agentenv", mif=20, caps=caps, gateway="https://gw2")]
    problems, notes = fsb.capacity_findings(pools)
    assert problems == ["AgentENV gateway https://gw1 holds 32 microVMs (harness.sandbox_caps.agentenv_max_active_n) "
                        "and the fleet's pools on it hold 40 at once (island 0 (n0), pool i0-a: 1 sampler(s) x 20; "
                        "island 1 (n1), pool i1-a: 1 sampler(s) x 20): lower agentic.max_in_flight or the pools' "
                        "targets"]
    assert "fleet: AgentENV gateway https://gw2: 20 microVMs at once (island 1 (n1), pool i1-b: 1 sampler(s) x 20) " \
           "of 32" in notes


def test_docker_host_room_counts_every_islands_samplers_on_it():
    envs = {"run": "fleet-a", "hosts": [{"name": "h1", "docker_host": "ssh://sbx@h1", "max_sandboxes": 20},
                                        {"name": "h2", "docker_host": "ssh://sbx@h2", "max_sandboxes": 20}]}
    # shard index 0 of each island lands on the same host (assign_host weighs by index)
    pools = [_row("i0-a", 0, "remote-docker", caps=RD_CAP, envs=envs, docker_host="envs"),
             _row("i1-a", 1, "remote-docker", caps=RD_CAP, envs=envs, docker_host="envs")]
    problems, notes = fsb.capacity_findings(pools)
    assert problems == []
    warned = [n for n in notes if n.startswith("WARNING: fleet: sandbox host h1 holds 20 trial containers")]
    assert warned and "run 32 at once (island 0 (n0), pool i0-a: 1 sampler(s) x 16; island 1 (n1), pool i1-a: " \
                      "1 sampler(s) x 16)" in warned[0]
    one = [_row("i0-a", 0, "remote-docker", caps=RD_CAP, envs=envs, docker_host="ssh://sbx@h2")]
    assert fsb.capacity_findings(one)[1] == ["fleet: sandbox host h2: 16 of 20 trial containers (island 0 (n0), "
                                             "pool i0-a: 1 sampler(s) x 16)"]


def test_the_sandboxes_count_in_the_fleet_spend():
    from yeto.breakers import fleet as bfleet

    spend = bfleet.fleet_spend(pool_node_hours=[(2.0, 4.0)], island_hours=[1.0], island_usd_per_h=[8.0],
                               sandbox_hours=2.0, sandbox_usd_per_h=3.0)
    assert spend.sandbox_usd == 6.0 and spend.total_usd == 22.0
    assert spend.to_json() == {"usd": 22.0, "pools_usd": 8.0, "islands_usd": [8.0], "syncer_usd": 0.0,
                               "sandbox_usd": 6.0}
    plain = bfleet.fleet_spend(pool_node_hours=[(2.0, 4.0)], island_hours=[1.0], island_usd_per_h=[8.0])
    assert "sandbox_usd" not in plain.to_json(), "an unpriced fleet's record is as it was"
    assert bfleet.budget_trip(spend, 20.0)["value"] == 22.0  # the sandboxes can trip the fleet budget
    assert bfleet.sandbox_price({"fleet": {"sandbox_usd_per_h": 3}}) == 3.0 and bfleet.sandbox_price({}) == 0.0
    boat = [_row("i0-a", 0, "boat")]
    assert fsb.spend_lines(boat, 0.0)[0].startswith("fleet: fleet.sandbox_usd_per_h is unset: the fleet spend counts "
                                                    "the sandboxes (boat) at $0/h")
    assert fsb.spend_lines(boat, 3.0) == ["fleet: sandboxes (boat) at $3/h in the fleet spend"]
    assert fsb.spend_lines([], 3.0) == []


def test_the_budget_estimate_adds_the_sandboxes(fleet_files):
    from yeto.rl import up_fleet

    _cfg, plan = fleet.compile_fleet(_fleet_cfg(fleet_files))
    rows = [("i0-agent", 1, 4.0)]
    line = up_fleet.budget_lines(plan, pools=rows, syncer_usd_per_h=0.0, usd=40, wall_h=2, sandbox_usd_per_h=3.0)[0]
    assert "= pools $4.00/h" in line and "+ sandboxes $3.00/h" in line and "$23.00/h" in line
    without = up_fleet.budget_lines(plan, pools=rows, syncer_usd_per_h=0.0, usd=40, wall_h=2)[0]
    assert "sandboxes" not in without


def test_fleet_sandbox_usd_per_h_is_checked_by_the_compile(fleet_files):
    cfg = _fleet_cfg(fleet_files)
    cfg["fleet"]["sandbox_usd_per_h"] = -1
    with pytest.raises(ValueError, match="fleet.sandbox_usd_per_h must be a number >= 0"):
        fleet.compile_fleet(cfg)
    cfg["fleet"]["sandbox_usd_per_h"] = 2.5
    fleet.compile_fleet(cfg)


def test_run_up_preflights_the_fleets_agentic_pools(fleet_files, monkeypatch):
    env = fleet_files / "agentic.env"
    env.write_text('HARBOR_MAX_SANDBOXES={"remote-docker": 8}\n')
    envs = fleet_files / "envs.json"
    envs.write_text(json.dumps({"run": "r", "hosts": [{"name": "h1", "docker_host": "ssh://sbx@h1",
                                                      "max_sandboxes": 12}]}))
    cfg = _fleet_cfg(fleet_files)
    for island in cfg["islands"]:
        for pool in island["samplers"]:
            pool["agentic_env_file"] = str(env)
    run = up.Run(up.load_config(_write_cfg(fleet_files, cfg)))
    pools = run.fleet_sandbox_pools()
    assert [p.name for p in pools] == ["i0-agent", "i1-agent"] and {p.process_cap() for p in pools} == {8}
    problems, _notes = run.fleet_sandbox_findings()
    assert len(problems) == 2 and all("the run config has no sandbox.envs" in p for p in problems)
    assert problems[0].startswith("island 0 (east), pool i0-agent:")
    assert "sandbox_store" in run.pools[0].load()["sampler"]
    cfg["sandbox"] = {"envs": str(envs), "docker_host": "ssh://sbx@h1"}
    cfg["fleet"]["sandbox_usd_per_h"] = 1.5
    run = up.Run(up.load_config(_write_cfg(fleet_files, cfg)))
    problems, notes = run.fleet_sandbox_findings()
    assert problems == []
    assert "fleet: sandbox provider remote-docker: island 0 (east), pool i0-agent; island 1 (west), pool i1-agent" \
           in notes
    assert any("sandboxes (remote-docker) at $1.5/h" in n for n in notes)
    assert run.fleet_preflight() is None


def _write_cfg(tmp_path: Path, cfg: dict) -> str:
    path = tmp_path / "run.json"
    path.write_text(json.dumps(cfg))
    return str(path)


# ------------------------------------------------------------------ an island's run_prefix with its slash


def test_an_islands_run_prefix_names_owners_that_never_cross_islands(tmp_path):
    """A fleet pool's ``__RUN_PREFIX__`` / ``YETO_NC_RUN_PREFIX`` is ``<run>/trainer<k>``:
    the slash is a legal docker label value, AgentENV metadata value and path segment,
    and the islands' sandboxes keep apart (their run and shard differ)."""
    from yeto.rl.harness.harbor import agentenv_labels, boat_labels
    from yeto.sandbox import trial_labels

    def env(k: int) -> dict[str, str]:
        return {"YETO_NC_RUN_PREFIX": f"r/trainer{k}", "YETO_SANDBOX_SHARD": f"ag-i{k}-0",
                "YETO_SAMPLER_INCARNATION": "inc-1"}

    own0, own1 = agentenv_labels.owner_from_env(env(0)), agentenv_labels.owner_from_env(env(1))
    assert own0.metadata() == {"yeto_run": "r/trainer0", "yeto_shard": "ag-i0-0", "yeto_inc": "inc-1"}
    later1 = agentenv_labels.Owner(run=own1.run, shard=own1.shard, inc="inc-2")
    assert later1.is_stale(own1.metadata()) and not later1.is_stale(own0.metadata())
    # remote-docker: one YETO_SANDBOX_RUN_LABEL for the fleet; the shard keeps the islands apart
    labels0 = trial_labels.trial_labels("t", {**env(0), "YETO_SANDBOX_RUN_LABEL": "r"})
    owner1 = trial_labels.Owner(run="r", shard="ag-i1-0", inc="inc-2")
    assert labels0["ai.yeto.shard"] == "ag-i0-0" and not owner1.is_stale(labels0)
    # boat names: a label value is cleaned to [A-Za-z0-9_-]
    name = boat_labels.labelled_name(boat_labels.Labels(run=boat_labels.clean("r/trainer1"), island="ag-i1-0"), "b")
    assert name == "ytl1.r-trainer1.ag-i1-0..b" and boat_labels.parse_labels(name).run == "r-trainer1"
    # the event tape of a spec "events-__RUN_PREFIX__.jsonl": one directory per island
    tapes = {entry.incarnation_tape(str(tmp_path / f"events-r/trainer{k}.jsonl"), "inc-1") for k in (0, 1)}
    assert tapes == {tmp_path / "events-r" / "trainer0.inc-1.jsonl", tmp_path / "events-r" / "trainer1.inc-1.jsonl"}
    write = entry.tape_writer(sorted(tapes)[0], "ag-i0-0")
    write("x")
    assert sorted(tapes)[0].is_file()


def test_the_preflight_reads_the_boat_shares_run_up_wrote_into_each_pool(fleet_files):
    """Run up splits the boat account over the fleet's boat pools (each pool's
    spec breakers section); the preflight summed the defaults instead and warned
    "together 12 > the account's 12" style for shares that add up to the account."""
    spec = json.loads((fleet_files / "agentic.json").read_text())
    spec["agentic"]["task_defaults"] = {"provider": "boat"}
    spec["agentic"].pop("envs")
    (fleet_files / "boat.json").write_text(json.dumps(spec))
    cfg = _fleet_cfg(fleet_files)
    for island in cfg["islands"]:
        for pool in island["samplers"]:
            pool["spec"] = str(fleet_files / ("boat.json" if pool["name"] == "agent" else "plain.json"))
    run = up.Run(up.load_config(_write_cfg(fleet_files, cfg)))
    pools = run.fleet_sandbox_pools()
    assert [p.name for p in pools] == ["i0-agent", "i1-agent"]
    rates = [p.caps["boat_starts_per_minute_n"] for p in pools]
    assert sum(rates) <= 12 and all(r < 12 for r in rates), rates
    _problems, notes = run.fleet_sandbox_findings()
    assert not any("each boat sampler may start" in n for n in notes), notes
