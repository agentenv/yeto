"""Several sandbox hosts: each sampler drives one host of ``agentic.envs.hosts``
(``agentic.setup.sandbox.docker_host: "envs"``), weighted by ``max_sandboxes``."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from yeto.launch import agentic_setup
from yeto.sandbox import cli as sandbox_cli
from yeto.sandbox.envs import (
    EnvsError,
    assign_host,
    assignment_report,
    host_sequence,
    parse_envs,
)
from yeto.sandbox.gate import gate_from_spec


def _hosts(*caps):
    return [{"name": f"h{i}", "docker_host": f"ssh://sbx@10.0.0.{i + 1}", "max_sandboxes": c,
             "tunnel_listen_host": f"172.17.{i}.1"} for i, c in enumerate(caps)]


def _spec(sampler_id, caps=(96, 96, 96, 96), **setup_extra):
    return {"sampler_id": sampler_id, "run_prefix": "k31",
            "agentic": {"proxy": {"port": 18080},
                        "envs": {"run": "k31", "hosts": _hosts(*caps), "tasksets": []},
                        "setup": {"sandbox": {"docker_host": "envs", "key": {"env": "SBX_KEY"}},
                                  "tunnel": {"listen": "172.17.0.1:19000-19063"},
                                  "records_secret": "generate", **setup_extra}}}


def test_sixteen_samplers_spread_over_four_hosts_by_capacity():
    cfg = parse_envs({"run": "r", "hosts": _hosts(96, 96, 96, 96)})
    count = Counter(h.name for h in host_sequence(cfg.hosts, 16))
    assert count == {"h0": 4, "h1": 4, "h2": 4, "h3": 4}
    # a host twice as large takes twice the samplers, interleaved
    cfg = parse_envs({"run": "r", "hosts": _hosts(160, 80, 80)})
    seq = [h.name for h in host_sequence(cfg.hosts, 16)]
    assert Counter(seq) == {"h0": 8, "h1": 4, "h2": 4} and seq[:4] == ["h0", "h1", "h2", "h0"]


def test_a_shard_keeps_its_host_and_scale_add_continues_the_sequence():
    cfg = parse_envs({"run": "r", "hosts": _hosts(96, 96, 96)})
    first = [assign_host(cfg, f"k31-{k}").name for k in range(8)]
    assert [assign_host(cfg, f"k31-{k}").name for k in range(8)] == first  # deterministic per shard
    more = [assign_host(cfg, f"k31-{k}").name for k in range(16)]
    assert more[:8] == first  # adding shards never moves the running ones
    pinned = parse_envs({"run": "r", "hosts": _hosts(96, 96, 96), "assign": {"k31-0": "h2"}})
    assert assign_host(pinned, "k31-0").name == "h2"
    with pytest.raises(EnvsError):
        parse_envs({"run": "r", "hosts": _hosts(96), "assign": {"k31-0": "nope"}})
    with pytest.raises(EnvsError):
        assign_host(cfg, "no-index")


def test_setup_points_docker_and_the_tunnel_at_the_assigned_host():
    setups = [agentic_setup.parse(_spec(f"k31-{k}")) for k in range(16)]
    assert Counter(s.sandbox_host for s in setups) == {"h0": 4, "h1": 4, "h2": 4, "h3": 4}
    s5 = setups[5]
    assert s5.sandbox.host == f"10.0.0.{int(s5.sandbox_host[1:]) + 1}" and s5.sandbox.user == "sbx"
    assert s5.tunnel.host == s5.sandbox.host  # the reverse tunnel ends on the same host
    assert s5.tunnel.listen_host == f"172.17.{s5.sandbox_host[1:]}.1" and s5.tunnel.listen_port == 19005
    env = s5.container_env()
    assert env["YETO_SANDBOX_HOST"] == s5.sandbox_host
    assert env["YETO_RECORDS_PROXY_PUBLIC_BASE"] == f"http://172.17.{s5.sandbox_host[1:]}.1:19005"
    # the requirements of the preflight dial the assigned host
    from yeto.launch import daytona_sampler

    assert daytona_sampler.requirements_of(_spec("k31-5")).ssh_hosts == (f"{s5.sandbox.host}:22",)


def test_envs_docker_host_refuses_a_fixed_tunnel_host_and_missing_hosts():
    with pytest.raises(ValueError, match="tunnel.host must be unset"):
        agentic_setup.parse(_spec("k31-1", tunnel={"listen": "172.17.0.1:19000-19063", "host": "10.9.9.9"}))
    spec = _spec("k31-1")
    spec["agentic"]["envs"]["hosts"] = []
    with pytest.raises(ValueError, match="needs agentic.envs.hosts"):
        agentic_setup.parse(spec)


def test_an_explicit_docker_host_is_still_named_for_the_gate():
    spec = _spec("k31-1")
    spec["agentic"]["setup"]["sandbox"]["docker_host"] = "ssh://sbx@10.0.0.3"
    setup = agentic_setup.parse(spec)
    assert setup.sandbox_host == "h2" and setup.container_env()["YETO_SANDBOX_HOST"] == "h2"


def test_the_gate_of_each_sampler_counts_its_own_host():
    for k in (0, 1, 2, 3):
        spec = _spec(f"k31-{k}")
        env = agentic_setup.parse(spec).container_env()
        made = []
        gate = gate_from_spec(agentic_setup.strip(spec), objects=None, environ=env,
                              docker_factory=lambda h: made.append(h.name) or object())
        assert gate.host_name == env["YETO_SANDBOX_HOST"] and made == [gate.host_name]


def test_assignment_report_warns_on_an_overfull_host_and_the_cli(tmp_path, capsys):
    cfg = parse_envs({"run": "r", "hosts": _hosts(320, 96)})
    report = assignment_report(cfg, [f"k31-{k}" for k in range(16)], per_sampler=20)
    assert sum(len(r["samplers"]) for r in report["hosts"].values()) == 16
    assert report["warnings"] == []  # 12 x 20 = 240 <= 320 and 4 x 20 = 80 <= 96
    assert [len(r["samplers"]) for r in report["hosts"].values()] == [12, 4]
    envs = tmp_path / "envs.json"
    envs.write_text(json.dumps({"run": "k31", "hosts": _hosts(96, 96)}))
    import argparse

    parser = argparse.ArgumentParser()
    sandbox_cli.add_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["sandbox", "assign", "--envs", str(envs), "--islands", "16", "--per-sampler", "20"])
    assert sandbox_cli.main(args) == 1  # 8 x 20 = 160 > 96 on each host
    out = capsys.readouterr().out
    assert "h0: 8 samplers, containers 160/96" in out and "WARNING h0" in out
    envs.write_text(json.dumps({"run": "k31", "hosts": _hosts(96, 96, 96, 96)}))
    args = parser.parse_args(["sandbox", "assign", "--envs", str(envs), "--islands", "16", "--per-sampler", "20"])
    assert sandbox_cli.main(args) == 0
    assert "h3: 4 samplers, containers 80/96" in capsys.readouterr().out


def test_host_room_is_the_smallest_of_cap_bridge_addresses_and_memory():
    from yeto.sandbox.envs import HostConfig, host_room

    h = HostConfig(name="h0", docker_host="ssh://u@h0", max_sandboxes=320, trial_memory_gib=4)
    room = host_room(h, bridge_subnets=["172.17.0.0/24"], mem_total_bytes=754 * 2**30)
    assert room["limits"] == {"max_sandboxes": 320, "bridge_addresses": 253, "memory": 188}
    assert room["effective"] == 188 and len(room["warnings"]) == 2
    wide = host_room(HostConfig(name="h0", docker_host="x", max_sandboxes=160, trial_memory_gib=4),
                     bridge_subnets=["172.17.0.0/20"], mem_total_bytes=754 * 2**30)
    assert wide["effective"] == 160 and wide["warnings"] == []
