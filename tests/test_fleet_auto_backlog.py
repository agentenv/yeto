"""``max_backlog_groups: "auto"`` in an island fleet: each island is the buffer trainer of
its own pools, so its pools get the single trainer's formula with that island's release
cadence and target samplers (not the 256-group fallback). Fakes only: nothing is ordered."""

from __future__ import annotations

import json
import types

import pytest

from yeto.launch.nc import cli as nc_cli
from yeto.launch.rollout import pools as rollout_pools
from yeto.rl import fleet, up, up_consistency
from yeto.rl.engine import sample_roles

from tests.test_run_up_fleet import STORE, TRAINER_ARGV

ARGV = [*TRAINER_ARGV, "--rollout-batch-size", "8"]


def _spec(tmp_path, name, backlog):
    spec = {"sampler_id": "x", "store": "/work/stage/b/runs/__RUN_PREFIX__", "engine": {"adapter": "vllm-server"},
            "event_tape": "/work/sampler/events-__RUN_PREFIX__.jsonl", "sampling": {"n": 8},
            "prompts": {"path": "p.jsonl", "input_key": "q", "groups_per_round": 1},
            "max_backlog_groups": backlog}
    (tmp_path / f"{name}.json").write_text(json.dumps(spec))
    return str(tmp_path / f"{name}.json")


def _pool(tmp_path, name, base, *, target, backlog="auto"):
    return {"name": name, "provider": {"kind": "modal", "image": "i", "secret": "s", "bucket": "b", "gpu": "L4"},
            "spec": _spec(tmp_path, f"spec-{base}", backlog), "run_prefix": "r",
            "r2_env_file": str(tmp_path / "store.env"), "shard_base": base, "shard_limit": 4, "target": target,
            "slots": [1, 2, 3, 4], "run": "r", "usd_per_h": 1.0}


def _cfg(tmp_path):
    (tmp_path / "store.env").write_text("R2_BUCKET=b\n")
    return {"run": "r", "store": STORE, "budget": {"usd": 40, "wall_h": 4, "nc_usd_per_h": 0},
            "fleet": {"trainer_argv": list(ARGV)},
            "islands": [
                # east: releases every 2nd version, 2 + 1 target samplers ("auto" and a number)
                {"name": "east", "trainer": {"gpu": "modal:1x1xl40s", "argv": ["--rl-release-every", "2"]},
                 "samplers": [_pool(tmp_path, "a", "ea", target=2), _pool(tmp_path, "b", "eb", target=1, backlog=16)]},
                # west: releases every version, 1 target sampler
                {"name": "west", "trainer": {"gpu": "modal:1x1xl40s", "argv": ["--rl-release-every", "1"]},
                 "samplers": [_pool(tmp_path, "a", "wa", target=1)]}]}


def _expected(release_every, samplers):
    m = sample_roles.release_record_rounds(release_every=release_every, staged=True, skip_busy=False)
    return sample_roles.auto_backlog_groups(groups_per_round=8, release_every=release_every, record_rounds=m,
                                            samplers=samplers)


@pytest.mark.parametrize("backend", [[], ["--rl-backend", "verl"]], ids=["miles", "verl"])
def test_each_islands_auto_pools_get_their_islands_cap(tmp_path, backend):
    cfg = _cfg(tmp_path)
    cfg["fleet"]["trainer_argv"] += backend  # a Miles fleet and a verl fleet alike
    out, _plan = fleet.compile_fleet(cfg)
    caps = {p["name"]: p.get("auto_backlog") for p in out["samplers"]}
    # east counts all 3 target samplers of its pools (the numbered pool's too), west its 1
    east, west = _expected(2, 3), _expected(1, 1)
    assert caps["i0-a"]["groups"] == east == 16 and caps["i1-a"]["groups"] == west == 32
    assert caps["i0-a"]["why"] == ("island 0 (east): ceil(((release_every 2 + m 2 + 1) x 8 + 8) / samplers 3) = 16")
    assert caps["i0-b"] is None  # a number in the spec: left as it is
    assert east < sample_roles.AUTO_BACKLOG_FALLBACK_GROUPS


def test_the_pool_renders_its_islands_cap_and_not_the_runs_or_the_fallback(tmp_path, monkeypatch):
    out, _plan = fleet.compile_fleet(_cfg(tmp_path))
    [entry] = [p for p in out["samplers"] if p["name"] == "i1-a"]
    doc, _base, _path = rollout_pools.entry_doc(entry, "x")
    nc_cli.pool_config(doc, min_target=0)  # the key is a known pool key
    monkeypatch.setenv(sample_roles.AUTO_BACKLOG_ENV, "999")  # a run-level value never wins over the island's
    member = types.SimpleNamespace(shard="wa-i1-0", data_slot=None, replaces_shard=None, predecessor_lost=False)
    spec = nc_cli.render_spec(doc, member, ["wa-i1-0"])
    assert spec["max_backlog_groups"] == 32
    assert spec[sample_roles.AUTO_BACKLOG_NOTE_KEY].startswith("auto: island 1 (west): ")
    assert up_consistency.pool_backlog(doc)["cap"] == 32  # the breaker rules read the same cap


def test_the_breaker_view_of_a_fleet_pool_reads_its_islands_cap(tmp_path):
    from yeto.breakers import rules_facts

    cfg = _cfg(tmp_path)
    caps = {name: rules_facts.run_view_sources(cfg, pool=name).sources.spec["max_backlog_groups"]
            for name in ("i0-a", "i1-a", "i0-b")}
    assert caps == {"i0-a": 16, "i1-a": 32, "i0-b": 16}  # i0-b: the number its spec sets
    compiled, _plan = fleet.compile_fleet(cfg)  # what run up reads (breakers.resolved.json)
    assert rules_facts.run_view_sources(compiled, pool="i1-a").sources.spec["max_backlog_groups"] == 32


def test_a_bad_pool_cap_is_refused():
    for bad in ({"groups": 0}, {"groups": "8"}, {"groups": 8, "extra": 1}, [8]):
        with pytest.raises(ValueError, match="auto_backlog must be"):
            nc_cli.pool_auto_backlog({"auto_backlog": bad})
    assert nc_cli.pool_auto_backlog({}) is None


def test_a_pool_config_with_a_bad_cap_is_refused_at_load(tmp_path):
    out, _plan = fleet.compile_fleet(_cfg(tmp_path))
    [entry] = [p for p in out["samplers"] if p["name"] == "i1-a"]
    doc, _base, _path = rollout_pools.entry_doc(entry, "x")
    with pytest.raises(ValueError, match="auto_backlog must be"):
        nc_cli.pool_config({**doc, "auto_backlog": {"groups": -1}}, min_target=0)


def test_an_island_the_formula_does_not_cover_takes_the_fallback(tmp_path):
    cfg = _cfg(tmp_path)
    del cfg["islands"][1]["samplers"][0]["target"]  # west: no target sampler to share the need
    out, _plan = fleet.compile_fleet(cfg)
    caps = {p["name"]: p.get("auto_backlog") for p in out["samplers"]}
    assert caps["i1-a"] is None and caps["i0-a"]["groups"] == 16  # east keeps its own


def test_run_up_prints_each_fleet_pools_cap(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    path = tmp_path / "run.json"
    path.write_text(json.dumps(_cfg(tmp_path)))
    run = up.Run(up.load_config(str(path)))
    assert run.auto_backlog is None  # no run-level cap: the islands carry their own
    run.log_auto_backlog()
    out = capsys.readouterr().out
    assert "max_backlog_groups auto -> 16 per sampler of pool i0-a (island 0 (east): " in out
    assert "max_backlog_groups auto -> 32 per sampler of pool i1-a (island 1 (west): " in out
    assert "fallback" not in out
