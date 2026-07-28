import importlib.util
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="local test interpreter does not include torch",
)


def test_two_rank_barrier_closes_exact_horizon_without_extra_steps(tmp_path):
    worker = Path(__file__).with_name("multirank_barrier_worker.py")
    env = dict(os.environ)
    env["YETO_BARRIER_TEST_OUTPUT"] = str(tmp_path)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        master_port = listener.getsockname()[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--master-addr=127.0.0.1",
            f"--master-port={master_port}",
            "--nproc-per-node=2",
            str(worker),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "MULTIRANK_BARRIER_PASS" in result.stdout
