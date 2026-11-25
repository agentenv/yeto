"""A sky pool on the node runtime (the SSH + docker route, :class:`cli.NcSshOps`): its
SGLang engine serves ``/metrics`` and the pool's probe reads SGLang's names there, and
its members are priced by their own offer (the ``member_ready`` event's ``usd_per_h``),
from the unified price table when the offer states no price. Nothing here orders."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from types import SimpleNamespace

from test_nc_elastic import Clock, _ssh_ops
from test_nc_sky_provider import FakeSky
from test_rl_hetero_sampler_engines import _config

from yeto.cost import prices
from yeto.launch.nc import cli, elastic
from yeto.launch.nc import client as ncc
from yeto.launch.nc import tunnel
from yeto.launch.nc.provider import NCProvider
from yeto.launch.nc.sky_provider import from_config
from yeto.launch.pool import money
from yeto.launch.pool import status as el
from yeto.launch.pool.island_pool import IslandPool
from yeto.rl.adapters.sampler_engines.sglang_server import SGLangServerEngine

NEBIUS_L40S_SPOT = {"infra": "nebius/eu-north1", "accelerators": "L40S:1", "use_spot": True}
SGLANG_METRICS = """# HELP sglang:generation_tokens_total Number of generation tokens processed.
# TYPE sglang:generation_tokens_total counter
sglang:generation_tokens_total{model_name="m",engine_type="unified"} 1234.0
sglang:num_running_reqs{model_name="m",engine_type="unified"} 6.0
sglang:num_queue_reqs{model_name="m",engine_type="unified"} 2.0
sglang:token_usage{model_name="m",engine_type="unified"} 0.4
"""
VLLM_METRICS = """vllm:generation_tokens_total{model_name="m",engine="0"} 500.0
vllm:generation_tokens_total{model_name="m",engine="1"} 250.0
vllm:num_requests_running{model_name="m",engine="0"} 3.0
vllm:num_requests_waiting{model_name="m",engine="0"} 1.0
"""


# -- /metrics ------------------------------------------------------------------

def test_the_sglang_engine_serves_metrics_once(tmp_path):
    argv = SGLangServerEngine(_config(tmp_path)).argv(tmp_path / "base")
    assert argv.count("--enable-metrics") == 1
    argv = SGLangServerEngine(_config(tmp_path, extra_args=("--enable-metrics",))).argv(tmp_path / "base")
    assert argv.count("--enable-metrics") == 1, "a spec that asks for it already is not doubled"


def _probe_line(tmp_path, metrics_text: str) -> str:
    """Run the pool's probe script with a fake ``curl`` (the engine's /metrics) and ``sudo``
    (docker: a running container) on PATH; returns its YETO_NC_PROBE line."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (tmp_path / "metrics.txt").write_text(metrics_text)
    for name, body in (("curl", f"cat {tmp_path / 'metrics.txt'}"),
                       ("sudo", 'case "$*" in (*inspect*) echo "running 0";; (*) :;; esac')):
        path = bin_dir / name
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    script = elastic.probe_script("c", str(tmp_path / "no-tape-*.jsonl"), metrics_port=30000)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=30)
    return next(line for line in out.stdout.splitlines() if line.startswith("YETO_NC_PROBE"))


def test_the_probe_reads_sglang_metrics(tmp_path):
    line = _probe_line(tmp_path, SGLANG_METRICS)
    assert "tok=1234 run=6 wq=2" in line
    p = el.Probe.parse(line, 0)
    assert (p.gen_tokens, p.running, p.queued) == (1234.0, 6, 2), "the pool sees the engine's progress"


def test_the_probe_reads_vllm_metrics_as_before(tmp_path):
    assert "tok=750 run=3 wq=1" in _probe_line(tmp_path, VLLM_METRICS)
    assert "tok=- run=- wq=-" in _probe_line(tmp_path, "# no engine counters\n")


# -- the price of a member -------------------------------------------------------

def _sky_ops(tmp_path, *, price=None, resources=NEBIUS_L40S_SPOT):
    offer = {"key": "L40S:1@nebius/eu-north1/spot", "resources": dict(resources)}
    if price is not None:
        offer["price_usd_h"] = price
    cfg = {"run": "run-s", "target": 1, "slots": [1], "shard_base": "sk", "spec": str(tmp_path / "spec.json"),
           "run_prefix": "run-s", "r2_env_file": str(tmp_path / "r2.env"),
           "provider": {"kind": "sky", "offers": [offer], "cluster_prefix": "sk"}}
    (tmp_path / "spec.json").write_text(json.dumps({"event_tape": "/work/events-__RUN_PREFIX__.jsonl"}))
    provider = from_config(cfg, "run-s", sky=FakeSky())
    return cfg, cli.NcSshOps(cfg, run=lambda *a, **k: None, provider=provider)


def test_a_sky_member_on_the_node_route_is_priced_by_its_offer(tmp_path):
    cfg, ops = _sky_ops(tmp_path, price=0.749)
    m = SimpleNamespace(ticket="sk-run-s-1", account=1)
    assert ops.hourly_cost(m) == 0.749 and money.currency_of(ops, m) == "usd"
    pool = IslandPool(cli.pool_config(cfg), ops, clock=Clock(), log=lambda _m: None)
    assert pool._usd_per_h(m) == 0.749  # noqa: SLF001 - what member_ready carries


def test_an_offer_without_a_price_takes_the_table_row(tmp_path):
    _cfg, ops = _sky_ops(tmp_path)
    assert ops.hourly_cost(SimpleNamespace(ticket="sk-run-s-1", account=1)) == 0.7492
    assert prices.offer_usd_h(dict(NEBIUS_L40S_SPOT, use_spot=False)) == 1.5484
    assert prices.offer_usd_h({"infra": "nebius/eu-west1", "accelerators": "L40S:1", "use_spot": True}) is None


def test_a_spot_offer_without_a_spot_row_is_counted_at_on_demand(tmp_path, capsys):
    table = tmp_path / "prices.json"
    table.write_text(json.dumps({"rows": [
        {"provider": "nebius", "gpu": "L40S", "gpus": 1, "region": "eu-north1", "spot": False, "grade": "billing",
         "usd_h": 1.5, "source": "test", "as_of": "2026-11-25"}]}))
    assert prices.offer_usd_h(NEBIUS_L40S_SPOT, path=str(table)) == 1.5, "never priced lower than on-demand"
    assert prices.spot_at_on_demand(NEBIUS_L40S_SPOT, path=str(table))
    assert not prices.spot_at_on_demand(NEBIUS_L40S_SPOT), "the shipped table has the spot row"
    assert not prices.spot_at_on_demand(dict(NEBIUS_L40S_SPOT, use_spot=False), path=str(table))


def test_the_dry_run_names_a_spot_offer_counted_at_on_demand(tmp_path, monkeypatch):
    cfg, _ops = _sky_ops(tmp_path)
    assert cli.spot_priced_on_demand(cfg) == []
    real = prices.offer_row

    def no_spot_rows(resources, *, path=None):
        row = real(resources, path=path)
        if resources.get("use_spot") and row is not None and row.spot:
            return real(dict(resources, use_spot=False), path=path)
        return row

    monkeypatch.setattr(prices, "offer_row", no_spot_rows)
    assert cli.spot_priced_on_demand(cfg) == ["L40S:1@nebius/eu-north1/spot"]
    assert cli.spot_on_demand_notes(cfg) == ["offer L40S:1@nebius/eu-north1/spot: the unified price table has no "
                                             "spot price for it; counted at its on-demand price"]
    stated = json.loads(json.dumps(cfg))
    stated["provider"]["offers"][0]["price_usd_h"] = 0.75
    assert cli.spot_priced_on_demand(stated) == [], "a stated price is the operator's"


def test_an_unpriced_offer_is_counted_at_the_pool_rate_never_none(tmp_path):
    cfg, ops = _sky_ops(tmp_path, resources={"infra": "nebius/xx-nowhere1", "accelerators": "L40S:1",
                                             "use_spot": True})
    assert cli.unpriced_offers(cfg) == ["L40S:1@nebius/eu-north1/spot"]
    value = ops.hourly_cost(SimpleNamespace(ticket="sk-run-s-1", account=1))
    assert value is not None and value > 0
    assert value == money.to_usd(ncc.hourly_cost_ucr(), "ucr"), "the MI355X rate, as the pool budget counts it"


def test_a_national_compute_member_keeps_the_pool_rate(tmp_path):
    book = ncc.Ledger(tunnel.state_dir() / "tickets.json")
    ops = _ssh_ops(tmp_path, [], provider=NCProvider([7], "run-x", ledger=book, allowed={7}))
    assert ops.hourly_cost(SimpleNamespace(ticket="t", account=7)) is None, "unchanged: the pool's MI355X rate"
