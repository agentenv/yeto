"""NC pool: a ``docker inspect`` that fails (sudo, dockerd) says nothing about the sampler.
Only docker's own "No such object" reads as an absent container, and a ready sampler is
restarted for an absent container only when two probes in a row say so: a restart removes
the container first, so one wrong reading would kill a healthy sampler."""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from yeto.launch.nc import elastic
from yeto.launch.pool import status as el
from yeto.launch.pool.island_pool import GONE, READY

from test_nc_elastic import Clock, Cloud, _names, _pool, _ready_member, _run, _state  # noqa: F401

SUDO = """#!/bin/sh
# a stand-in for ``sudo docker ...`` on the node
[ "$2" = inspect ] || exit 0
case "$(cat "$PROBE_MODE_FILE")" in
  running) echo "running 0" ;;
  exited) echo "exited 137" ;;
  gone) echo "Error: No such object: yeto-sampler-nc-0" >&2; exit 1 ;;
  dockerd) echo "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?" >&2; exit 1 ;;
  sudo) echo "sudo: unable to resolve host node-7: Temporary failure in name resolution" >&2; exit 1 ;;
esac
"""


@pytest.mark.parametrize("shell", ["bash", "sh"])
def test_the_probe_script_tells_a_failed_inspect_from_a_container_that_is_not_there(tmp_path, shell):
    if shutil.which(shell) is None:
        pytest.skip(f"no {shell}")
    shim = tmp_path / "bin"
    shim.mkdir()
    (shim / "sudo").write_text(SUDO)
    (shim / "sudo").chmod(0o755)
    mode = tmp_path / "mode"
    env = {**os.environ, "PATH": f"{shim}:{os.environ.get('PATH', '')}", "PROBE_MODE_FILE": str(mode)}
    script = elastic.probe_script("yeto-sampler-nc-0", str(tmp_path / "events*.jsonl"))

    def probe(what):
        mode.write_text(what)
        out = subprocess.run([shell, "-c", script], capture_output=True, text=True, timeout=30, env=env)
        return el.Probe.parse(out.stdout, out.returncode)

    assert (probe("running").node, probe("running").sampler) == ("ok", "running")
    assert (probe("exited").sampler, probe("exited").exit_code) == ("exited", 137)
    assert (probe("gone").node, probe("gone").sampler) == ("ok", "absent")
    # the command failed: nothing is known about the sampler (as for a node that did not answer)
    assert probe("dockerd").node == el.PROBE_UNREACHABLE
    assert probe("sudo").node == el.PROBE_UNREACHABLE


class InspectCloud(Cloud):
    """``inspect_fails``: tickets whose ``docker inspect`` fails (the probe line says unknown)."""

    def __init__(self, clock, **kw):
        super().__init__(clock, **kw)
        self.inspect_fails = set()

    def probe(self, m):
        if m.ticket in self.inspect_fails:
            return el.Probe.parse("YETO_NC_PROBE status=unknown exit=- joined=1 age=3 tw=0 tok=- run=- wq=-", 0)
        return super().probe(m)


def _setup(**elastic_cfg):
    clock = Clock()
    cloud = InspectCloud(clock, ready_after=300)
    pool = _pool(clock, cloud, accounts=(1, 4), **elastic_cfg)
    return clock, cloud, pool, _ready_member(pool, clock, cloud)


def test_one_failed_docker_inspect_does_not_restart_a_healthy_sampler():
    clock, cloud, pool, m = _setup()
    cloud.inspect_fails.add(m.ticket)
    pool.tick()
    assert not cloud.kinds("restart") and m.state == READY and m.restarts == 0
    cloud.inspect_fails.clear()
    _run(pool, clock, clock.t + 180)
    assert not cloud.kinds("restart") and not cloud.kinds("drain") and m.state == READY
    assert "sampler_restart" not in _names(pool)


def test_a_docker_that_never_answers_is_a_lost_node_after_the_unreachable_grace():
    clock, cloud, pool, m = _setup(unreachable_grace_s=300)
    cloud.inspect_fails.add(m.ticket)
    _run(pool, clock, clock.t + 240)
    assert m.state == READY and not cloud.kinds("restart")
    _run(pool, clock, clock.t + 1800)
    assert m.state == GONE and "node_lost" in _names(pool)
    assert pool.members[-1].state == READY and pool.members[-1].replaces_shard == m.shard


def test_an_absent_container_is_restarted_only_when_two_probes_in_a_row_say_so():
    clock, cloud, pool, m = _setup()
    cloud.tickets[m.ticket].update(sampler="absent")  # one reading
    pool.tick()
    assert not cloud.kinds("restart") and m.state == READY
    cloud.tickets[m.ticket].update(sampler="running")
    clock.t += 60
    pool.tick()
    cloud.tickets[m.ticket].update(sampler="absent")  # not in a row: counted anew
    clock.t += 60
    pool.tick()
    assert not cloud.kinds("restart") and m.state == READY
    clock.t += 60
    pool.tick()  # the second reading in a row: the container is gone
    assert [e[1] for e in cloud.kinds("restart")] == [m.ticket] and m.restarts == 1
    _run(pool, clock, clock.t + 600)
    assert m.state == READY


def test_an_exited_container_is_still_restarted_at_once():
    clock, cloud, pool, m = _setup()
    cloud.tickets[m.ticket].update(sampler="exited", exit=137)
    pool.tick()
    assert [e[1] for e in cloud.kinds("restart")] == [m.ticket]
