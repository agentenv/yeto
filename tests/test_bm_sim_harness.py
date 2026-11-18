"""bm-sim, the simulated bare-metal cluster of the ssh trainer island
(tests/integration/bm_sim).

The CPU tests here keep the harness in step with the launcher without any
cluster: the launcher still builds the island the harness patches, the
inventories it writes load and validate, and the node docker shim keeps the
container semantics the island relies on.

The cluster run itself is opt in: ``YETO_BM_SIM=1`` runs it on this host's
docker (Linux), ``YETO_BM_SIM=modal`` on a Modal CPU sandbox (``run.py --modal``);
``YETO_BM_SIM_NODES`` / ``YETO_BM_SIM_SCENARIOS`` (comma list) narrow it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.integration.bm_sim import cluster, scenario
from yeto.launch.node_inventory import load_inventory
from yeto.launch.ssh_trainer_island import SshTrainerSpec

ROOT = Path(__file__).resolve().parent.parent
BM_SIM = ROOT / "tests" / "integration" / "bm_sim"
SHIM = BM_SIM / "node" / "docker-shim"


@pytest.fixture
def store_env(monkeypatch, tmp_path):
    monkeypatch.setenv("YETO_RL_STORE_S3_ACCESS_KEY_ID", "bm-sim-key")
    monkeypatch.setenv("YETO_RL_STORE_S3_SECRET_ACCESS_KEY", "bm-sim-secret")
    monkeypatch.setenv("YETO_RL_STORE_S3_ENDPOINT_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("YETO_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr("yeto.runs.RUNS_DIR", tmp_path / "runs")


def _inventory(tmp_path, nodes=2, runtime="rootfs"):
    path = tmp_path / f"inventory-{runtime}.yaml"
    path.write_text(json.dumps(cluster.inventory(nodes, cluster.addresses(nodes), runtime=runtime)))
    return path


@pytest.mark.parametrize("runtime", ["rootfs", "docker"])
def test_the_launcher_builds_the_island_the_harness_patches(tmp_path, store_env, runtime):
    """The real launcher path (CLI parse, preflight checks, island task, inventory
    spec) for the sim's inventory; the harness swaps only the learner module and
    the setup, and the island env carries the staged s3 store for every node."""
    args = scenario.build_args(scenario.launcher_argv(inventory=_inventory(tmp_path, 3, runtime),
                                                      store_uri="s3://bm-sim/r", nodes=3, cards=cluster.CARDS,
                                                      run_id="bmsim-unit"))
    island, task = scenario.build_island(args, tmp_path / "events")
    spec = island.ssh
    assert island.cloud == "ssh" and spec.docker is (runtime == "docker")
    assert spec.addresses() == ["198.18.0.10", "198.18.0.11", "198.18.0.12"]
    assert f"python3 -m {scenario.FAKE_LEARNER_MODULE} " in spec.run_script
    assert scenario.LEARNER_MODULE not in spec.run_script
    assert "--rl-sample-role trainer" in spec.run_script and 'ray start --head' in spec.run_script
    assert task.setup == scenario.SETUP_STUB
    assert spec.env["YETO_RL_STORE_S3_STAGE"] and spec.env["YETO_RL_STORE_S3_URI"] == "s3://bm-sim/r"


def test_patch_task_swaps_exactly_one_learner_line():
    line = f"python3 -m {scenario.LEARNER_MODULE} --x 1\n"
    task = scenario.patch_task(SimpleNamespace(run="a\n" + line, setup="git clone ..."))
    assert task.run == f"a\npython3 -m {scenario.FAKE_LEARNER_MODULE} --x 1\n" and task.setup == scenario.SETUP_STUB
    for run in ("no learner\n", line + line):
        with pytest.raises(RuntimeError, match="bm-sim swaps exactly one"):
            scenario.patch_task(SimpleNamespace(run=run, setup=""))


def test_the_resident_form_drops_the_trainer_scheduler(tmp_path, monkeypatch):
    argv = scenario.launcher_argv(inventory=_inventory(tmp_path), store_uri="s3://bm-sim/r", nodes=2,
                                  cards=cluster.CARDS, run_id="bmsim-unit")
    monkeypatch.setattr(scenario, "git_checkout", lambda path: True)
    resident = scenario.resident_argv(inventory=_inventory(tmp_path), store_uri="s3://bm-sim/r", nodes=2,
                                      cards=cluster.CARDS, run_id="bmsim-unit")
    assert set(scenario.SCHEDULER_FLAGS) & set(argv) and not set(scenario.SCHEDULER_FLAGS) & set(resident)
    assert "--rl-trainer-start-seconds-fixed" not in resident and "--rl-code-pack" not in resident
    monkeypatch.setattr(scenario, "git_checkout", lambda path: False)
    assert scenario.resident_argv(inventory=_inventory(tmp_path), store_uri="s3://bm-sim/r", nodes=2,
                                  cards=cluster.CARDS, run_id="bmsim-unit")[-2:] == ["--rl-code-pack", "off"]


def test_the_sim_inventories_load_and_the_docker_one_passes_the_cuda_rdma_checks(tmp_path):
    rootfs = load_inventory(_inventory(tmp_path))
    docker = load_inventory(_inventory(tmp_path, runtime="docker"))
    assert [n.address for n in rootfs] == ["198.18.0.10", "198.18.0.11"]
    assert {n.fabric for n in docker} == {"ib"} and docker[1].env == {"BM_SIM_NODE": "n1",
                                                                      "NCCL_IB_HCA": "mlx5_0,mlx5_1"}
    SshTrainerSpec(nodes=tuple(docker), run_id="r", workdir="/opt/yeto", run_script="true",
                   image=scenario.SIM_IMAGE)  # validates the cuda RDMA docker flags
    assert cluster.addresses(3)["store"] == "198.18.0.5"
    with pytest.raises(ValueError):
        cluster.addresses(300)


def _shim(tmp_path, *argv, env=None):
    return subprocess.run([sys.executable, str(SHIM), *argv], capture_output=True, text=True, timeout=30,
                          env={"PATH": os.environ["PATH"], "HOME": str(tmp_path), **(env or {})})


@pytest.mark.skipif(sys.platform != "linux", reason="the shim keeps its state in /tmp of a Linux node")
def test_docker_shim_passes_only_the_named_env_and_needs_host_networking(tmp_path):
    name = f"bm-sim-unit-{os.getpid()}"
    done = _shim(tmp_path, "run", "--rm", "--name", name, "--network", "host", "--ipc", "host", "-e", "PASSED",
                 "-e", "SET=v", "-w", str(tmp_path), "img@sha256:" + "0" * 64, "bash", "-c",
                 'echo "$PASSED|${NOT_PASSED:-}|$SET|$PWD"', env={"PASSED": "p", "NOT_PASSED": "x"})
    assert done.returncode == 0 and done.stdout.strip() == f"p||v|{tmp_path}", done.stderr
    assert json.loads(Path(f"/tmp/bm-sim-docker/{name}.json").read_text())["env"] == ["HOME", "PASSED", "PATH",
                                                                                        "SET"]
    refused = _shim(tmp_path, "run", "--name", name, "img", "true")
    assert refused.returncode == 125 and "--network host" in refused.stderr
    assert _shim(tmp_path, "pull", "-q", "img:1").returncode == 0


@pytest.mark.skipif(not os.environ.get("YETO_BM_SIM"), reason="opt in: YETO_BM_SIM=1 (docker here) or =modal")
def test_bm_sim_cluster_end_to_end(tmp_path):
    mode = os.environ["YETO_BM_SIM"]
    if mode != "modal" and shutil.which("docker") is None:
        pytest.skip("YETO_BM_SIM=1 needs docker on this host")
    argv = [sys.executable, str(BM_SIM / "run.py"), "--nodes", os.environ.get("YETO_BM_SIM_NODES", "2"),
            "--out", str(tmp_path / "out"), "--work", str(tmp_path / "work")]
    for name in filter(None, os.environ.get("YETO_BM_SIM_SCENARIOS", "").split(",")):
        argv += ["--scenario", name]
    if mode == "modal":
        argv.append("--modal")
    done = subprocess.run(argv, cwd=ROOT, timeout=5400)
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert done.returncode == 0 and report["ok"], [c for c in report["checks"] if not c["ok"]]
