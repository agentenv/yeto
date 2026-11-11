"""A Daytona sampler pool end to end on one Linux host, without Daytona or a GPU.

The real pool (``LeasePool`` + ``NcSshOps`` + ``DaytonaProvider`` + the process route)
runs against a fake Daytona client whose sandboxes are local directories, and a fake
``ssh`` that resolves the provider's alias through the user's ssh config (``Include``)
and runs the remote command with ``bash`` in that directory (``/work``, ``/models``,
the GPU device nodes and ``/proc/meminfo`` are the sandbox's own). Every command,
upload, start script and probe is the one the pool generates. The sampler's
interpreter is a stub that writes the JOIN event and waits; the preflight sees a
stub torch reporting 8 x gfx950. Needs ``/proc`` and ``setsid`` (skipped on macOS).
"""

from __future__ import annotations

import http.server
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto.launch import daytona_sampler as ds
from yeto.launch.nc import cli, elastic as el, node_pool
from yeto.launch.nc.daytona_provider import SSH_HOST, DaytonaProvider
from yeto.launch.nc.lease_pool import GONE, READY, LeasePool, PoolConfig

from test_nc_daytona_provider import FakeDaytona, FakeSdk

pytestmark = pytest.mark.skipif(not Path("/proc/self/environ").exists() or shutil.which("setsid") is None,
                                reason="needs Linux /proc and setsid")

TAG = "YETO_TEST_SANDBOX"


def _tagged(name: str) -> list[int]:
    """Processes started in sandbox ``name`` (their environment carries the tag; never by name)."""
    want = f"{TAG}={name}".encode()
    out = []
    for p in Path("/proc").iterdir():
        try:
            if p.name.isdigit() and want in (p / "environ").read_bytes().split(b"\0"):
                out.append(int(p.name))
        except OSError:
            pass
    return out


def _kill(name: str) -> None:
    for pid in _tagged(name):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


class Boxes(FakeDaytona):
    """Sandboxes as directories: started at create, processes killed at delete / eviction."""

    def __init__(self, root: Path):
        super().__init__()
        self.root = root

    def create(self, params, timeout=60):
        box = super().create(params, timeout)
        box.state = "started"
        r = self.root / box.name
        for d in ("work", "models", "dev/dri", "proc", "site/torch", "home"):
            (r / d).mkdir(parents=True, exist_ok=True)
        (r / "dev/kfd").write_text("")
        for i in range(8):
            (r / f"dev/dri/renderD{128 + i}").write_text("")
        (r / "proc/meminfo").write_text("MemTotal: 1610612736 kB\n")
        (r / "site/torch/__init__.py").write_text(
            "__version__ = '2.9.0+rocm'\nclass version:\n    hip = '7.0'\nclass _P:\n    gcnArchName = 'gfx950:x'\n"
            "class cuda:\n    is_available = staticmethod(lambda: True)\n    device_count = staticmethod(lambda: 8)\n"
            "    get_device_properties = staticmethod(lambda i: _P())\n"
            "    get_device_name = staticmethod(lambda i: 'MI355X')\n")
        for mod in ("vllm", "aiter", "boto3", "safetensors", "zstandard", "httpx", "numpy"):
            (r / "site" / f"{mod}.py").write_text(f"__version__ = {ds.VLLM_SAMPLER_ROCM_VERSION!r}\n")
        return box

    def delete(self, sandbox, timeout=60, wait=False):
        _kill(sandbox.name)
        super().delete(sandbox, timeout, wait)

    def evict(self, name):
        _kill(name)
        super().evict(name)


class Ssh:
    """``run`` for the pool: ``ssh`` goes into a sandbox directory, anything else runs as is."""

    def __init__(self, boxes: Boxes, ssh_config: Path, python: Path, bin_dir: Path):
        self.boxes, self.ssh_config, self.python, self.bin = boxes, ssh_config, python, bin_dir
        self.calls: list[str] = []

    def _resolve(self, alias: str) -> tuple[str | None, str | None]:
        lines: list[str] = []

        def read(path: Path) -> None:
            for line in path.read_text().splitlines() if path.exists() else []:
                if line.startswith("Include "):
                    read(Path(line.split()[1]))
                else:
                    lines.append(line.strip())

        read(self.ssh_config)
        user = host = None
        active = False
        for line in lines:
            key, _, value = line.partition(" ")
            if key == "Host":
                active = alias in value.split()
            elif active and key == "User":
                user = value
            elif active and key == "HostName":
                host = value
        return user, host

    def _subs(self, root: Path) -> list[tuple[str, str]]:
        return [(ds.PYTHON, str(self.python))] + [(p, str(root / p.lstrip("/"))) for p in
                                                   ("/work", "/models", "/dev/kfd", "/dev/dri", "/proc/meminfo")]

    def _map(self, text: str, root: Path) -> str:
        for path, target in self._subs(root):
            text = re.sub(r"(?<![\w.~/-])" + re.escape(path) + r"(?![\w.-])", target, text)
        return text

    def __call__(self, argv, input=None, capture_output=False, text=False, timeout=None, check=False, **kw):
        if not isinstance(argv, list) or argv[0] != "ssh":
            return subprocess.run(argv, input=input, capture_output=capture_output, text=text, timeout=timeout,
                                  check=check, **kw)
        args = argv[1:]
        while args[0].startswith("-"):
            args = args[2:] if args[0] == "-o" else args[1:]
        alias, rest = args[0], args[1:]
        user, host = self._resolve(alias)
        box = next((b for b in self.boxes.boxes.values()
                    if b.state == "started" and user in [t for t, _ in b.tokens]), None)
        if host != SSH_HOST or box is None:
            err = f"ssh: Could not resolve hostname {alias}" if host is None else "Permission denied"
            return subprocess.CompletedProcess(argv, 255, "" if text else b"", err if text else err.encode())
        root = self.boxes.root / box.name
        command = self._map(" ".join(rest), root)
        self.calls.append(command)
        data = input.encode() if isinstance(input, str) else input
        if data and b"\0" not in data:
            data = self._map(data.decode(), root).encode()
        env = {"PATH": f"{self.bin}:/usr/bin:/bin", "HOME": str(root / "home"), TAG: box.name,
               "PYTHONPATH": str(root / "site"), "LANG": "C.UTF-8"}
        out = subprocess.run(["bash", "-c", command], input=data, capture_output=True, timeout=timeout, env=env,
                             cwd=root / "home")
        back = []
        for raw in (out.stdout, out.stderr):
            s = raw.decode(errors="replace")
            for path, target in self._subs(root):
                s = s.replace(target, path)
            back.append(s if text else s.encode())
        return subprocess.CompletedProcess(argv, out.returncode, *back)


STUB_PYTHON = """#!/bin/bash
# the sampler interpreter: preflight probes go to the real python, the engine sampler is a stub
if [ "$1" = "-m" ] && [ "$2" = "yeto.rl.engine.engine_sampler" ]; then
  tape=$(env -u PYTHONPATH {py} -c 'import json,sys; print(json.load(open(sys.argv[1]))["event_tape"])' "$4")
  mkdir -p "$(dirname "$tape")"
  echo '{{"event": "rl_sampler_join"}}' > "${{tape%.jsonl}}.1.jsonl"
  echo "engine sampler stub up"
  exec sleep 600
fi
exec {py} "$@"
"""


@pytest.fixture()
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    ssh_config = tmp_path / "home" / ".ssh" / "config"
    monkeypatch.setattr(node_pool, "SSH_CONFIG", ssh_config)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), http.server.SimpleHTTPRequestHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    store_env = tmp_path / "store.env"  # the object store endpoint the preflight reaches
    store_env.write_text(f"R2_BUCKET=b\nAWS_ACCESS_KEY_ID=x\nAWS_SECRET_ACCESS_KEY=y\n"
                         f"AWS_ENDPOINT_URL=http://127.0.0.1:{server.server_address[1]}\n")
    store_env.chmod(0o600)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"event_tape": "/work/sampler/events.jsonl", "prompts": {"groups_per_round": 2},
                                "num_rollouts": 8, "max_backlog_groups": 8, "max_prompt_tokens": 64,
                                "sampling": {"max_tokens": 8},
                                "engine": {"factory": "yeto.rl.adapters.sampler_engines.from_spec:build",
                                           "adapter": "fake", "kwargs": {}},
                                "reward": "yeto.rl.numeric_answer_reward:score", "reward_kind": "miles"}))
    python = tmp_path / "bin" / "python3"
    python.parent.mkdir()
    python.write_text(STUB_PYTHON.format(py=shlex.quote(sys.executable)))
    python.chmod(0o755)
    (tmp_path / "bin" / "df").write_text("#!/bin/bash\necho 'F 1024-blocks Used Available C M'\n"
                                         "echo \"sim 8589934592 0 8589934592 0% ${@: -1}\"\n")
    (tmp_path / "bin" / "df").chmod(0o755)
    (tmp_path / "bin" / "aws").write_text("#!/bin/bash\nexit 0\n")
    (tmp_path / "bin" / "aws").chmod(0o755)
    boxes = Boxes(tmp_path / "boxes")
    provider = DaytonaProvider("run-e", slots=[1, 2], image="rocm/vllm:test", sdk=FakeSdk, client=boxes)
    cfg = {"target": 1, "accounts": [1, 2], "shard_base": "dt", "run": "run-e", "spec": str(spec),
           "run_prefix": "run-e", "r2_env_file": str(store_env),
           "node_hooks": [{"name": "marker", "command": "mkdir -p /work && echo {shard} > /work/hook.txt"}]}
    run = Ssh(boxes, ssh_config, python, tmp_path / "bin")
    ops = cli.NcSshOps(cfg, run=run, provider=provider)
    fenced = []
    ops.fence = lambda m: fenced.append(m.shard)  # the store fence is covered elsewhere (no object store here)
    ops.fence_incarnation = lambda m: fenced.append(m.shard)
    events = el.EventLog(log=lambda m: None, run="run-e")
    pool = LeasePool(PoolConfig(target=1, accounts=[1, 2], shard_base="dt", stop_below_ucr=0,
                                min_balance_to_order_ucr=0,
                                elastic=el.ElasticConfig(max_orders=4, unreachable_grace_s=2, engine_stall_s=None)),
                     ops, log=lambda m: None, events=events)
    yield SimpleNamespace(pool=pool, ops=ops, boxes=boxes, run=run, events=events, fenced=fenced,
                          ssh_config=ssh_config, tmp=tmp_path)
    for name in list(boxes.boxes):
        _kill(name)
    server.shutdown()


def _until(pool, cond, timeout=90):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        pool.tick()
        if cond():
            return
        time.sleep(0.5)
    raise AssertionError(f"not reached in {timeout}s; events {[e['event'] for e in pool.events.events]}")


def test_daytona_pool_starts_probes_replaces_and_stops_a_process_sampler(world):
    pool, boxes = world.pool, world.boxes
    _until(pool, lambda: pool.members and pool.members[0].state == READY and pool.members[0].joined)
    first = pool.members[0]
    root = boxes.root / first.ticket
    # the alias resolves through the user's ssh config (Include of the provider's file)
    assert f"Include {cli.tunnel.state_dir() / 'ssh_config'}" in world.ssh_config.read_text()
    assert (root / "work/hook.txt").read_text().strip() == first.shard
    pid = int((root / "work/sampler" / f"{first.shard}.pid").read_text())
    assert os.getsid(pid) == pid and pid in _tagged(first.ticket)  # the session leader of the sampler
    assert "engine sampler stub up" in (root / "work/logs" / f"{first.shard}.log").read_text()
    assert not any(re.search(r"docker (run|inspect|stop|logs)", c) for c in world.run.calls)

    boxes.evict(first.ticket)  # spot reclaim: no notice, processes gone
    _until(pool, lambda: first.state == GONE and len(pool.members) == 2 and pool.members[1].state == READY)
    second = pool.members[1]
    names = [e["event"].removeprefix("nc_elastic_") for e in world.events.events]
    assert "evicted" in names and first.shard in world.fenced and second.replaces_shard == first.shard
    spec = json.loads((boxes.root / second.ticket / "work/sampler" / f"{second.shard}.json").read_text())
    assert spec["continues_shard"] == first.shard and spec["start_on_latest"] is True

    pool.stop("test")
    _until(pool, lambda: all(m.state == GONE for m in pool.members))
    assert all(b.state == "destroyed" for b in boxes.boxes.values())
    assert not _tagged(second.ticket)
    # the drain stopped the sampler before the delete and kept its log tail
    assert "engine sampler stub up" in (boxes.root / second.ticket / "work/logs" / f"{second.shard}.last.log").read_text()

