"""The one-page view of a run config (``yeto breaker show --config``, run up's
``breakers.resolved.json``) reads the run's pool: its pool config, its samplers' pool-merged
spec and env, and its ``breakers.pools.<name>`` overrides. Without that the pool, sampler and
harness rows showed their defaults, not what the pool and its samplers run with."""

from __future__ import annotations

import argparse
import json

from yeto.breakers import cli as bcli

ELASTIC = {"engine_stall_s": 900, "start_timeout_s": 5400}
SPEC = {"reload_max_defer_s": 2400, "agentic": {"weight_switch": "drain", "drain_timeout_s": 2400}}


def _run(argv):
    p = argparse.ArgumentParser()
    bcli.add_parser(p.add_subparsers(dest="command"))
    return bcli.main(p.parse_args(bcli.normalize_argv(argv)))


def _pool(tmp_path, name="pool", **extra):
    spec = tmp_path / f"{name}-spec.json"
    spec.write_text(json.dumps(SPEC))
    cfg = {"spec": str(spec), "elastic": dict(ELASTIC),
           "sampler": {"dispense": {"backend": "store", "namespace": "n", "lease_s": 7200}}, **extra}
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(cfg))
    return str(path)


def _run_config(tmp_path, **extra):
    cfg = {"run": "r", "store": "s3://b/r",
           "trainer": {"argv": ["--training-mode", "rl", "--rl-stall-timeout", "3600"]}, **extra}
    path = tmp_path / "run.json"
    path.write_text(json.dumps(cfg))
    return path


def _show_json(capsys, *argv):
    rc = _run(["breaker", "show", "--json", *argv])
    return rc, json.loads(capsys.readouterr().out)


def _entry(data, name):
    return next(e for e in data["entries"] if e["name"] == name)


def test_show_reads_the_runs_only_pool(tmp_path, capsys):
    path = _run_config(tmp_path, pool={"config": _pool(tmp_path)})
    rc, data = _show_json(capsys, "--config", str(path))
    assert rc == 0 and data["pool"] == "pool"
    assert _entry(data, "pool.engine_stall")["params"]["limit_s"] == 900
    start = _entry(data, "pool.start_timeout")
    assert start["params"]["limit_s"] == 5400 and start["sources"]["limit_s"]["kind"] == "legacy"
    assert "start_timeout_s" in start["sources"]["limit_s"]["location"]
    lease = _entry(data, "sampler.dispense_lease")  # the pool's sampler.dispense replaces the spec's
    assert lease["params"]["lease_s"] == 7200 and lease["sources"]["lease_s"]["kind"] == "legacy"
    assert _entry(data, "sampler.drain")["params"]["reload_max_defer_s"] == 2400
    assert _entry(data, "launch.stall")["params"]["timeout_s"] == 3600  # the run's own rows are as before


def test_show_applies_the_pools_breakers_overrides(tmp_path, capsys):
    path = _run_config(tmp_path, pool={"config": _pool(tmp_path)},
                       breakers={"pools": {"pool": {"engine_stall": {"limit_s": 1200}}}})
    rc, data = _show_json(capsys, "--config", str(path))
    assert rc == 1  # the override and the pool file's engine_stall_s disagree: a conflict, shown
    assert any(c["name"] == "pool.engine_stall" for c in data["conflicts"])


def test_an_explicit_spec_wins_over_the_pools(tmp_path, capsys):
    path = _run_config(tmp_path, pool={"config": _pool(tmp_path)})
    spec = tmp_path / "other-spec.json"
    spec.write_text(json.dumps({"reload_max_defer_s": 1800}))
    rc, data = _show_json(capsys, "--config", str(path), "--spec", str(spec))
    assert rc == 0 and data["pool"] is None
    assert _entry(data, "sampler.drain")["params"]["reload_max_defer_s"] == 1800


def test_several_pools_need_a_name(tmp_path, capsys):
    path = _run_config(tmp_path, samplers=[{"name": "a", "config": _pool(tmp_path, "a")},
                                           {"name": "b", "config": _pool(tmp_path, "b", elastic={"engine_stall_s": 300})}])
    assert _run(["breaker", "show", "--config", str(path)]) == 0
    out = capsys.readouterr().out
    assert "pool / sampler / harness rows show defaults: 2 pools (a, b), pass --pool NAME" in out
    rc, data = _show_json(capsys, "--config", str(path))
    assert data["pool"] is None and _entry(data, "pool.engine_stall")["params"]["limit_s"] == 600
    rc, data = _show_json(capsys, "--config", str(path), "--pool", "b")
    assert rc == 0 and data["pool"] == "b" and _entry(data, "pool.engine_stall")["params"]["limit_s"] == 300
    assert _run(["breaker", "show", "--config", str(path), "--pool", "c"]) == 2
    assert "its pools: a, b" in capsys.readouterr().err


def test_an_unreadable_pool_config_is_bad_input(tmp_path, capsys):
    path = _run_config(tmp_path, pool={"config": str(tmp_path / "missing.json")})
    assert _run(["breaker", "show", "--config", str(path)]) == 2
    assert "missing.json" in capsys.readouterr().err


def test_check_keeps_reading_the_run_config(tmp_path, capsys):
    """``check --config``: the rules only; a missing pool file is left to the pool dry run."""
    path = _run_config(tmp_path, pool={"config": str(tmp_path / "missing.json")})
    assert _run(["breaker", "check", "--config", str(path)]) == 0


def _write_resolved(tmp_path, cfg):
    from yeto.rl.up import Run

    run = Run.__new__(Run)
    run.dir = tmp_path
    run.cfg = cfg
    run.write_breakers_resolved()
    return json.loads((tmp_path / "breakers.resolved.json").read_text())


def test_run_up_writes_the_pools_rows(tmp_path):
    cfg = json.loads(_run_config(tmp_path, pool={"config": _pool(tmp_path)}).read_text())
    doc = _write_resolved(tmp_path, cfg)
    assert doc["pool"] == "pool"
    assert _entry(doc, "pool.start_timeout")["params"]["limit_s"] == 5400
    assert _entry(doc, "sampler.dispense_lease")["params"]["lease_s"] == 7200


def test_run_up_falls_back_to_the_run_config_on_an_unreadable_pool(tmp_path, capsys):
    cfg = json.loads(_run_config(tmp_path, pool={"config": str(tmp_path / "missing.json")}).read_text())
    doc = _write_resolved(tmp_path, cfg)
    assert doc["pool"] is None and _entry(doc, "pool.start_timeout")["params"]["limit_s"] == 3600
    assert "breakers.resolved.json shows the run config alone" in capsys.readouterr().out
