"""The sampler host paths that are not a National Compute node with systemd: env files a
shell sources, the orderly SIGTERM leave, the prompts file upload, and the records relay
supervisor of a host without systemd (a Daytona / Modal sandbox). No machine: fake exec
handles, and a real local bash where the shell semantics are the point."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_agentic_records_relay import _spec as _relay_spec
from tests.test_agentic_records_relay import fake_keygen
from tests.test_daytona_sampler import SPEC, FakeBox
from yeto.launch import agentic_setup as su
from yeto.launch import daytona_sampler as ds
from yeto.launch.nc import cli as nc_cli
from yeto.launch.nc import ssh_sampler as ss

AWKWARD = {
    "HARBOR_ENV_KWARGS": '{"machine_type": "mi-microvm-2c1g"}',
    "HARBOR_MAX_SANDBOXES": '{"agentenv": 16}',
    "WITH_SPACE": "a b  c",
    "QUOTES": "it's \"quoted\"",
    "DOLLAR": "$HOME and `id` and $(id)",
    "EMPTY": "",
    "TRAILING": "x ",
}


def _bash_read(path: Path, names) -> dict[str, str]:
    """What ``set -a; . file`` gives a child process (the start script's reading)."""
    code = "import json, os, sys; print(json.dumps({k: os.environ.get(k) for k in sys.argv[1:]}))"
    out = subprocess.run(["bash", "-c", f'set -eu; set -a; . "$0"; set +a; exec "{sys.executable}" -c "$1" "${{@:2}}"',
                          str(path), code, *names], capture_output=True, text=True, check=True,
                         env={"PATH": os.environ["PATH"]})
    return json.loads(out.stdout)


def _systemd_read(text: str) -> dict[str, str]:
    """systemd's ``EnvironmentFile=`` grammar (src/basic/env-file.c, parse_env_file_internal):
    unquoted text, ``'...'`` literal, ``"..."`` with backslash escapes, segments concatenated;
    trailing whitespace of an unquoted value is dropped; ``#`` / ``;`` lines are comments."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.lstrip()
        if not line or line[0] in "#;":
            continue
        key, _, rest = line.partition("=")
        key = key.strip()
        value, i, quoted_end = "", 0, 0
        while i < len(rest):
            ch = rest[i]
            if ch == "'":
                j = rest.index("'", i + 1)
                value += rest[i + 1:j]
                i = j + 1
                quoted_end = len(value)
            elif ch == '"':
                i += 1
                while rest[i] != '"':
                    if rest[i] == "\\" and i + 1 < len(rest):
                        i += 1
                    value += rest[i]
                    i += 1
                i += 1
                quoted_end = len(value)
            elif ch == "\\" and i + 1 < len(rest):
                value += rest[i + 1]
                i += 2
            else:
                value += ch
                i += 1
        out[key] = value[:quoted_end] + value[quoted_end:].rstrip()
    return out


# -- env files -----------------------------------------------------------------------

def test_a_shell_env_file_reads_the_same_with_source_and_systemd(tmp_path):
    text = su.env_file_text(AWKWARD, style="shell")
    path = tmp_path / "setup.env"
    path.write_text(text)
    assert _bash_read(path, AWKWARD) == AWKWARD
    assert _systemd_read(text) == AWKWARD
    # the unquoted form of before: sourcing it fails (a JSON value turns into a command)
    bad = tmp_path / "bad.env"
    bad.write_text("".join(f"{k}={v}\n" for k, v in AWKWARD.items()))
    assert subprocess.run(["bash", "-c", f"set -eu; set -a; . {bad}"], capture_output=True).returncode != 0


def test_a_docker_env_file_is_literal_and_refuses_a_newline():
    text = su.env_file_text({"HARBOR_ENV_KWARGS": AWKWARD["HARBOR_ENV_KWARGS"]}, style="docker")
    assert text == 'HARBOR_ENV_KWARGS={"machine_type": "mi-microvm-2c1g"}\n'  # docker keeps quotes as written
    with pytest.raises(ValueError, match="newline"):
        su.env_file_text({"A": "x\ny"}, style="docker")
    with pytest.raises(ValueError, match="not a variable name"):
        su.env_file_text({"A B": "x"}, style="shell")


def test_an_operator_docker_env_file_is_made_safe_to_source(tmp_path):
    raw = ("# comment\nHARBOR_ENV_KWARGS={\"machine_type\": \"m x\"}\nPLAIN=600\nQUOTED='a b'\n"
           "export EXPORTED=1\nTRAILING=x \n")
    path = tmp_path / "op.env"
    path.write_text(su.shell_env_file(raw))
    got = _bash_read(path, ["HARBOR_ENV_KWARGS", "PLAIN", "QUOTED", "EXPORTED", "TRAILING"])
    # unquoted values read as docker --env-file reads them; a quoted one as the operator wrote it
    assert got == {"HARBOR_ENV_KWARGS": '{"machine_type": "m x"}', "PLAIN": "600", "QUOTED": "a b",
                   "EXPORTED": "1", "TRAILING": "x "}


# -- the Daytona / process route start -------------------------------------------------

class PromptBox(FakeBox):
    """FakeBox that keeps uploaded files and answers the prompts size / sha256 check."""

    def __init__(self, *, corrupt=False, have=(), **kw):
        super().__init__(**kw)
        self.files: dict[str, bytes] = {}
        self.corrupt, self.have = corrupt, set(have)

    def upload(self, data, remote):
        super().upload(data, remote)
        self.files[remote] = data

    def exec(self, command, *, timeout=600):
        for src in list(self.files):  # put(): upload to <path>.up, then mv
            if src.endswith(".up") and f"mv -f {src} " in command:
                self.files[src[:-3]] = self.files.pop(src)
        if "YETO_PROMPTS" in command:
            path = command.split("sha256sum ")[1].split(" ")[0]
            data = self.files.get(path, b"")
            if self.corrupt:
                data = data[:-1]
            self.commands.append(command)
            return ds.ExecResult(0, f"YETO_PROMPTS {len(data)} {hashlib.sha256(data).hexdigest()}\n")
        if command.split("; ")[-1].startswith("test -s "):
            self.commands.append(command)
            return ds.ExecResult(0 if command.split("test -s ")[1].strip() in self.have else 1)
        return super().exec(command, timeout=timeout)


def _start(tmp_path, box, **kw):
    env_file = tmp_path / "r2.env"
    env_file.write_text("R2_BUCKET=b\n")
    spec = {**SPEC, "prompts": {"path": "/work/sampler/prompts.jsonl", "input_key": "prompt",
                                "groups_per_round": 2}}
    return ds.start_sampler(box, spec=spec, shard="k31-0", run_prefix="runs-x", r2_env_file=env_file,
                            code=("abc123", b"TAR"), **kw)


def test_start_sets_the_reclaim_marker_and_the_shard_spread_key(tmp_path):
    box = PromptBox(have={"/work/sampler/prompts.jsonl"})
    out = _start(tmp_path, box)
    script = box.uploads[out.paths["script"] + ".up"].decode()
    assert "export YETO_SPOT_RECLAIM_MARKER=/work/sampler/k31-0.reclaim" in script
    assert "export YETO_RL_RELEASE_SPREAD_KEY=runs-x/k31-0" in script


def test_start_uploads_the_prompts_file_and_checks_size_and_sha(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt": "a b", "metadata": {"task": "t1"}}\n' * 3)
    events = []
    box = PromptBox()
    _start(tmp_path, box, prompts_file=prompts, emit=events.append)
    assert box.files["/work/sampler/prompts.jsonl"] == prompts.read_bytes()
    ev = next(e for e in events if e["event"] == "daytona_sampler_prompts")
    assert ev["bytes"] == len(prompts.read_bytes()) and ev["sha256"] == hashlib.sha256(prompts.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="does not match"):
        _start(tmp_path, PromptBox(corrupt=True), prompts_file=prompts)


def test_start_refuses_a_spec_whose_prompts_are_not_on_the_sandbox(tmp_path):
    with pytest.raises(RuntimeError, match="prompts_file"):
        _start(tmp_path, PromptBox())


def test_start_quotes_the_setup_env_and_the_operator_env_for_the_shell(tmp_path):
    spec = {**_relay_spec(tmp_path, sampler_id="k31-1"), **{k: v for k, v in SPEC.items() if k != "sampler_id"}}
    spec["agentic"]["setup"]["agentenv"]["machine_type"] = "mi-microvm-2c1g"
    op = tmp_path / "agentic.env"
    op.write_text('HARBOR_ENV_KWARGS={"machine_type": "mi-microvm-2c1g"}\nYETO_AGENTENV_DISK_SIZE_MB=65536\n')
    env_file = tmp_path / "r2.env"
    env_file.write_text("R2_BUCKET=b\n")
    box = FakeBox()
    ran = {}

    class Prov:
        def __init__(self, setup, node, **kw):
            ran["setup"] = setup

        def run(self):
            return {}

    import unittest.mock as mock
    with mock.patch.object(ds, "ProcessProvisioner", Prov):
        ds.start_sampler(box, spec=spec, shard="k31-1", run_prefix="p", r2_env_file=env_file, code=("abc", b"T"),
                         agentic_env_file=op, spec_dir=tmp_path)
    setup_env = tmp_path / "setup.env"
    setup_env.write_bytes(box.uploads[f"{ds.SAMPLER_DIR}/k31-1.agentic-setup.env.up"])
    want = ran["setup"].container_env(python_bin=str(Path(ds.PYTHON).parent))
    assert json.loads(want["HARBOR_ENV_KWARGS"]) == {"machine_type": "mi-microvm-2c1g"}
    assert _bash_read(setup_env, want) == want
    op_copy = tmp_path / "op.env"
    op_copy.write_bytes(box.uploads[f"{ds.SAMPLER_DIR}/k31-1.agentic.env.up"])
    assert _bash_read(op_copy, ["HARBOR_ENV_KWARGS"]) == {"HARBOR_ENV_KWARGS": '{"machine_type": "mi-microvm-2c1g"}'}


# -- the NC docker route ----------------------------------------------------------------

def _lease():
    from yeto.launch.nc.client import NodeLease

    return NodeLease(1, "t1", "192.0.2.7", "ubuntu", "ssh-ed25519 HOSTKEY", None, "2026-11-09T16:00:00Z", 8,
                     3_000_000)


def test_nc_route_sets_the_reclaim_marker_and_copies_the_prompts(tmp_path, monkeypatch):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt": "x"}\n')
    data = prompts.read_bytes()
    monkeypatch.setattr(ss, "upload_code", lambda alias, run: "/mnt/nvme/yeto/x")
    copied = []
    monkeypatch.setattr(ss, "upload_file", lambda alias, local, remote, **k: copied.append((str(local), remote)))
    calls = []

    def run(argv, **kw):
        calls.append(argv[-1])
        if "YETO_PROMPTS" in argv[-1]:
            return SimpleNamespace(returncode=0, stdout=f"YETO_PROMPTS {len(data)} {hashlib.sha256(data).hexdigest()}\n",
                                   stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    spec = {"sampler_id": "nc-0", "prompts": {"path": "/work/sampler/prompts.jsonl"}}
    ss.start_sampler(_lease(), spec=spec, shard="nc-0", run_prefix="p", r2_env_file="x", state_dir=tmp_path, run=run,
                     prompts_file=prompts)
    assert (str(prompts), "/mnt/nvme/work/sampler/prompts.jsonl") in copied
    docker_run = next(c for c in calls if "docker run" in c)
    assert "YETO_SPOT_RECLAIM_MARKER=/work/sampler/nc-0.reclaim" in docker_run
    with pytest.raises(ValueError, match="under /work/"):
        ss.start_sampler(_lease(), spec={"sampler_id": "nc-0", "prompts": {"path": "/data/p.jsonl"}}, shard="nc-0",
                         run_prefix="p", r2_env_file="x", state_dir=tmp_path, run=run, prompts_file=prompts)


def test_the_pool_dry_run_reports_the_prompts_file(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt": "x"}\n{"prompt": "y"}\n')
    spec = {"prompts": {"path": "/work/sampler/prompts.jsonl"}}
    lines, problems, notes = nc_cli.prompts_file_report({"prompts_file": str(prompts)}, spec)
    assert problems == [] and "2 rows" in lines[0] and "/work/sampler/prompts.jsonl" in lines[0]
    _, problems, _ = nc_cli.prompts_file_report({"prompts_file": str(tmp_path / "none")}, spec)
    assert problems and "none" in problems[0]
    _, problems, _ = nc_cli.prompts_file_report({"provider": {"kind": "daytona", "transport": "sdk"}}, spec)
    assert "set prompts_file" in problems[0]
    _, problems, notes = nc_cli.prompts_file_report({}, spec)
    assert problems == [] and "prompts_file" in notes[0]
    assert nc_cli.prompts_file_report({"node_hooks": [{"command": "scp"}]}, spec) == ([], [], [])
    assert "prompts_file" in nc_cli.POOL_KEYS


# -- the records relay on a process node without systemd ------------------------------------

class LocalBox:
    """A SandboxExec that runs commands in a local bash (paths moved under tmp_path)."""

    name = "local"

    def exec(self, command, *, timeout=600):
        out = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=timeout)
        return ds.ExecResult(out.returncode, out.stdout, out.stderr)

    def upload(self, data, remote):
        Path(remote).parent.mkdir(parents=True, exist_ok=True)
        Path(remote).write_bytes(data)


def test_the_process_route_supervisor_starts_restarts_and_checks_the_recorded_pid(tmp_path, monkeypatch):
    if not Path("/proc/self/cmdline").exists():
        pytest.skip("needs /proc (the supervisor's PID is checked by its command line there)")
    spec = _relay_spec(tmp_path, sampler_id="k31-1")
    setup = su.parse(spec)
    prov = ds.ProcessProvisioner(setup, ds.ProcessNode(LocalBox(), yeto_dir=str(Path(su.__file__).parents[2]),
                                                       python=sys.executable),
                                 local_run=fake_keygen)
    relay = tmp_path / "records-relay"
    monkeypatch.setattr(ds.ProcessProvisioner, "relay_dir", property(lambda self: str(relay)))
    relay.mkdir()
    # a stand-in for ssh: the supervisor module runs for real and restarts it; it never connects
    argv = [sys.executable, "-m", "yeto.launch.agentic_node_agent", "tunnel", "--events", str(relay / "events.jsonl"),
            "--settle-s", "0.2", "--", sys.executable, "-c", "import time; time.sleep(30)", su.RELAY_TAG]
    assert not prov._relay_supervisor_running()
    assert prov._relay_supervisor_start(argv).returncode == 0
    first = int((relay / "supervisor.pid").read_text())
    for _ in range(50):
        if os.getsid(first) == first and prov._relay_supervisor_state()[0] == "active":
            break
        time.sleep(0.1)
    assert os.getsid(first) == first  # its own session: the sampler's group stop does not end it
    assert prov._relay_supervisor_running() and prov._relay_supervisor_state() == ("active", "0")
    # a restart ends the recorded process group (supervisor and its ssh) before the new one
    assert prov._relay_supervisor_start(argv).returncode == 0
    second = int((relay / "supervisor.pid").read_text())
    assert second != first
    for _ in range(50):
        try:
            os.kill(first, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    with pytest.raises(ProcessLookupError):
        os.kill(first, 0)
    # a recorded PID that is alive but another program (a reused PID) is not the supervisor
    (relay / "supervisor.pid").write_text(str(os.getpid()))
    assert not prov._relay_supervisor_running()
    os.killpg(second, 9)
    for c in (prov._relay_alive(), ds._alive_cmdline("/x.pid", "a")):
        assert "pkill" not in c and "pgrep" not in c and "killall" not in c


def test_the_process_route_runs_the_relay_step_without_systemd(tmp_path):
    """ProcessProvisioner over a fake box: no systemctl call, the supervisor started with
    setsid and its PID recorded, the forward settled from the events file."""
    spec = _relay_spec(tmp_path, sampler_id="k31-1")

    class Box(FakeBox):
        def __init__(self):
            super().__init__()
            self.started = False

        def exec(self, command, *, timeout=600):
            if "YETO_INIT" in command:
                self.commands.append(command)
                return ds.ExecResult(0, "YETO_INIT none\n")
            if "systemctl" in command or "journalctl" in command:
                raise AssertionError(command)
            if "setsid bash -c" in command and "agentic_node_agent" in command:
                self.commands.append(command)
                self.started = True
                return ds.ExecResult(0)
            if "grep -c" in command and "events.jsonl" in command:
                self.commands.append(command)
                return ds.ExecResult(0, '0\n{"event": "rl_agentic_tunnel", "state": "up"}\n')
            if "/proc/$p/cmdline" in command and "supervisor.pid" in command:
                self.commands.append(command)
                return ds.ExecResult(0 if self.started else 1)
            return super().exec(command, timeout=timeout)

    box = Box()
    now = [0.0]
    prov = ds.ProcessProvisioner(su.parse(spec), ds.ProcessNode(box, yeto_dir="/work/yeto/abc"),
                                 local_run=fake_keygen, sleep=lambda s: now.__setitem__(0, now[0] + s),
                                 clock=lambda: now[0])
    prov.records_relay()
    assert prov.report["records_relay"] == "changed"
    start = next(c for c in box.commands if "setsid bash -c" in c)
    assert "/records-relay/supervisor.pid" in start and "-R 0.0.0.0:18282:127.0.0.1:18080 yeto-records-relay" in start
    assert "export PYTHONPATH=/work/yeto/abc" in start
