"""Cross-node (multi-host) trainer island checks, measured on a 3-node Ascend RoCE island.

Every case here reproduces something that was observed on hardware on 2026-11-14/15:
an island that joined a Ray cluster spanning more nodes than it planned for, a round
whose samples did not divide over the island's cards, a Ray probe that timed out, and
a start script whose state directory landed under the wrong ``HOME``.
"""

from __future__ import annotations

import os
import subprocess
import threading

import pytest

from yeto.launch.node_inventory import Node
from yeto.launch.ssh_trainer_island import SshTrainerSpec, kill_script, remote_paths, start_script
from yeto.rl.adapters.verl.fully_async_round import (
    VerlTranslationError,
    balance_batch_disabled,
    check_trainer_batch_divisible,
)
from yeto.rl.adapters.verl.island_entry import (
    RAY_PROBE_TIMEOUT_ENV,
    RAY_PROBE_TIMEOUT_S,
    card_node_count,
    check_island_node_span,
    ray_probe_timeout_s,
)

PATHS = {"dir": "$HOME/d", "pid": "$HOME/d/pgid", "log": "$HOME/d/t.log",
         "rc": "$HOME/d/rc", "requant_log": "$HOME/d/rq.log"}


def _node(name: str, address: str) -> Node:
    return Node(name=name, host=address, device_family="npu", cpu_arch="aarch64", card="910B4",
                cards_per_node=8, hbm_gib=64, host_memory_gib=1024, fabric="roce",
                interface="eth0.100", address=address, runtime="rootfs")


def test_round_samples_must_divide_over_the_island_cards():
    check_trainer_batch_divisible(64, 8)
    check_trainer_batch_divisible(48, 12)
    # the measured failure: 8 groups x 8 samples on a 3 x 2 card island
    with pytest.raises(VerlTranslationError) as caught:
        check_trainer_batch_divisible(64, 6)
    assert "64 % 6" in str(caught.value)


def test_round_samples_check_needs_both_counts():
    with pytest.raises(VerlTranslationError):
        check_trainer_batch_divisible(0, 8)
    with pytest.raises(VerlTranslationError):
        check_trainer_batch_divisible(64, 0)


def test_card_node_count_ignores_nodes_without_cards():
    info = {"card_nodes": {"192.0.2.1": {"NPU": 8.0}, "192.0.2.2": {"NPU": 8.0}, "192.0.2.3": {}}}
    assert card_node_count(info) == 2
    assert card_node_count({}) == 0


def test_island_node_span_refuses_a_plan_that_ignores_the_other_nodes():
    one = {"card_nodes": {"192.0.2.1": {"NPU": 8.0}}}
    check_island_node_span(one, 1)  # single-node island: unchanged
    two = {"card_nodes": {"192.0.2.1": {"NPU": 2.0}, "192.0.2.2": {"NPU": 2.0}}}
    check_island_node_span(two, 2)  # the plan matches the cluster
    with pytest.raises(SystemExit) as caught:
        check_island_node_span(two, 1)
    assert "--actor-num-nodes 2" in str(caught.value)


def test_island_node_span_without_card_node_information_is_not_refused():
    check_island_node_span({"before": {"CPU": 32.0, "NPU": 8.0}}, 1)


def test_ray_probe_timeout_is_tunable(monkeypatch):
    monkeypatch.delenv(RAY_PROBE_TIMEOUT_ENV, raising=False)
    assert ray_probe_timeout_s() == float(RAY_PROBE_TIMEOUT_S)
    monkeypatch.setenv(RAY_PROBE_TIMEOUT_ENV, "900")
    assert ray_probe_timeout_s() == 900.0
    monkeypatch.setenv(RAY_PROBE_TIMEOUT_ENV, "nonsense")
    assert ray_probe_timeout_s() == float(RAY_PROBE_TIMEOUT_S)
    monkeypatch.setenv(RAY_PROBE_TIMEOUT_ENV, "0")
    assert ray_probe_timeout_s() == float(RAY_PROBE_TIMEOUT_S)


def test_start_script_exports_before_it_makes_the_state_directory():
    spec = SshTrainerSpec(nodes=(_node("a", "192.0.2.1"), _node("b", "192.0.2.2")), run_id="r",
                          workdir="/w", command=("x.py",), env={"HOME": "/srv/yeto/island-home"})
    script = start_script(spec, 1, {}, PATHS)
    # the state dir is pinned to the login HOME before the run's HOME is exported
    pin = script.index('export __yeto_state_home="$HOME"')
    assert pin < script.index("export HOME=/srv/yeto/island-home")
    assert 'mkdir -p "$__yeto_state_home/d"' in script and 'mkdir -p "$HOME/d"' not in script
    # the island's collective-communication variables still come from the inventory
    assert "export HCCL_SOCKET_IFNAME=eth0.100" in script
    assert "export GLOO_SOCKET_IFNAME=eth0.100" in script
    assert "--master-addr=192.0.2.1" in script and "--node-rank=1" in script


def test_island_node_span_leaves_a_multi_node_fixed_partition_island_alone():
    # 2 x 8 cards, --rl-placement fixed-partition --rollout-num-gpus 8: the trainer
    # rectangle is one node (actor_num_nodes=1), the rollout pool the other; runs on main
    two = {"card_nodes": {"192.0.2.1": {"GPU": 8.0}, "192.0.2.2": {"GPU": 8.0}}}
    check_island_node_span(two, 1, buffer_trainer=False)
    with pytest.raises(SystemExit):  # a buffer trainer on the same cluster is still refused
        check_island_node_span(two, 1, buffer_trainer=True)


def test_balance_batch_off_skips_the_divisibility_check():
    assert not balance_batch_disabled([])
    assert not balance_batch_disabled(["trainer.balance_batch=true"])
    assert balance_batch_disabled(["trainer.balance_batch=false"])
    assert balance_batch_disabled(["+trainer.balance_batch=False"])
    assert not balance_batch_disabled(["trainer.balance_batch=false", "trainer.balance_batch=true"])


def _run_kill(home, paths):
    env = {**os.environ, "HOME": str(home)}
    return subprocess.run(["bash", "-c", kill_script(paths, 1)], env=env, capture_output=True, text=True)


def test_kill_script_reads_the_pid_where_the_start_script_wrote_it(tmp_path):
    # start and kill both resolve the state dir under the login HOME, also when the
    # run exports another HOME
    spec = SshTrainerSpec(nodes=(_node("a", "192.0.2.1"),), run_id="r", workdir="/w", command=("x.py",),
                          env={"HOME": "/srv/yeto/island-home"})
    paths = remote_paths("r", "t", 1)
    start = start_script(spec, 0, {}, paths)
    kill = kill_script(paths, 1)
    pid_ref = '"$__yeto_state_home/.yeto' if '.yeto' in paths["pid"] else '"$__yeto_state_home/'
    assert pid_ref in start and pid_ref in kill
    assert 'export __yeto_state_home="$HOME"' in start and '__yeto_state_home="$HOME"' in kill
    proc = subprocess.Popen(["sleep", "300"], start_new_session=True)
    try:
        state = tmp_path / paths["pid"].replace("$HOME/", "")
        state.parent.mkdir(parents=True)
        state.write_text(str(proc.pid))
        # reap the child as soon as it dies: an unreaped zombie still answers kill -0
        reaper = threading.Thread(target=proc.wait, daemon=True)
        reaper.start()
        assert _run_kill(tmp_path, paths).returncode == 0
        reaper.join(timeout=10)
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill()


def test_kill_script_without_a_pid_fails_when_the_state_dir_exists(tmp_path):
    paths = remote_paths("r", "t", 1)
    done = _run_kill(tmp_path, paths)  # never started here: nothing to stop
    assert done.returncode == 0 and done.stderr == ""  # no launcher log line for it
    (tmp_path / paths["dir"].replace("$HOME/", "")).mkdir(parents=True)
    done = _run_kill(tmp_path, paths)  # a state dir with no pid: not confirmed stopped
    assert done.returncode != 0 and "may still run" in done.stderr


class _StopOps:
    """``SshOps.run`` answering every kill with one fixed result."""

    def __init__(self, rc: int, stderr: str):
        self.rc, self.stderr = rc, stderr

    def run(self, node, script, *, timeout=300.0):
        return subprocess.CompletedProcess([], self.rc, stdout="", stderr=self.stderr)


@pytest.mark.parametrize("rc,printed", [(0, False), (3, True)])
def test_release_logs_a_stop_only_when_it_failed(tmp_path, capsys, rc, printed):
    from yeto.launch.ssh_trainer_island import SshTrainerIslands

    spec = SshTrainerSpec(run_id="r", nodes=(_node("node-a", "198.51.100.7"),), workdir="/w",
                          command=("python", "-c", "pass"))
    islands = SshTrainerIslands(spec, name="t", jobs_path=tmp_path / "j.json",
                                ops=_StopOps(rc, "Warning: Permanently added 'node-a' to known hosts."))
    assert islands._kill(1) is (rc == 0)
    assert ("stop of trainer incarnation 1" in capsys.readouterr().out) is printed
