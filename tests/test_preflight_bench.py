"""The benchmark script ``yeto preflight`` ships into the trainer container."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from yeto.launch.preflight import bench

SCRIPT = Path(bench.__file__)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize(("op", "world", "factor"), [
    ("all_reduce", 16, 2 * 15 / 16), ("all_to_all", 48, 47 / 48), ("sendrecv", 16, 1.0), ("all_reduce", 1, 1.0)])
def test_busbw_factor_matches_nccl_tests(op, world, factor):
    assert bench.busbw_factor(op, world) == pytest.approx(factor)


def test_parse_size():
    assert bench.parse_size("16M") == 16 << 20 and bench.parse_size("1G") == 1 << 30
    assert bench.parse_size("4096") == 4096 and bench.parse_size("64KB") == 64 << 10


def test_ports_listen_dial_busy_and_refused():
    open_port, taken = _free_port(), _free_port()
    holder = socket.socket()
    holder.bind(("0.0.0.0", taken))
    holder.listen(1)
    try:
        rec = bench.run_ports([open_port, taken], ["127.0.0.1"], wait_s=0.2, hold_s=0.4, dial_timeout_s=1.0,
                              me="n0")
    finally:
        holder.close()
    assert rec["listen"][str(open_port)] == "listening"
    assert rec["listen"][str(taken)].startswith("busy")
    assert rec["dial"]["127.0.0.1"] == {str(open_port): "ok", str(taken): "ok"}
    assert bench.dial("127.0.0.1", _free_port(), 1.0) == "refused"


def test_script_is_self_contained():
    """The script runs in an image of another yeto revision: no yeto import."""
    text = SCRIPT.read_text()
    assert "import yeto" not in text and "from yeto" not in text and "from ." not in text
    done = subprocess.run([sys.executable, "-I", str(SCRIPT), "ports", "--ports", str(_free_port()), "--wait", "0",
                           "--hold", "0"], capture_output=True, text=True, timeout=30, check=True)
    (line,) = [ln for ln in done.stdout.splitlines() if ln.startswith(bench.MARKER)]
    assert json.loads(line.split(" ", 1)[1])["mode"] == "ports"


def test_collectives_are_correct_over_gloo_on_two_fake_nodes(tmp_path):
    """all_reduce / all_to_all / sendrecv on 4 CPU ranks as 2 nodes x 2 (the NCCL path
    differs only in the device and the timer)."""
    pytest.importorskip("torch")
    port = _free_port()
    done = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc-per-node=4",
         "--master-addr=127.0.0.1", f"--master-port={port}", str(SCRIPT), "collectives", "--backend", "gloo",
         "--local-world", "2", "--sizes", "64K,256K", "--iters", "2", "--warmup", "1"],
        capture_output=True, text=True, timeout=300, check=False, cwd=tmp_path)
    recs = [json.loads(ln.split(" ", 1)[1]) for ln in done.stdout.splitlines() if ln.startswith(bench.MARKER)]
    assert done.returncode == 0, done.stderr[-2000:]
    init = [r for r in recs if r["event"] == "init"]
    results = [r for r in recs if r["event"] == "result"]
    assert init and init[0]["nodes"] == 2 and init[0]["world"] == 4
    assert {r["op"] for r in results} == {"all_reduce", "all_to_all", "sendrecv"} and len(results) == 6
    assert all(r["correct"] and r["busbw_gbps"] > 0 for r in results)
    assert recs[-1] == {"event": "done", "failed_ops": 0, "mode": "collectives"}
