"""tests/syncer_reaper.py: a syncer a test leaves running is killed with the test."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import syncer_reaper

TESTS = Path(__file__).resolve().parent


def _fake_syncer(tmp_path: Path) -> Path:
    """``sleep`` under the name yeto-syncer (``ps`` shows the link's path)."""
    binary = tmp_path / "bin" / "yeto-syncer"
    binary.parent.mkdir()
    binary.symlink_to(shutil.which("sleep"))
    return binary


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # a zombie still answers kill(0); ps tells it apart
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout
    return bool(state.strip()) and not state.strip().startswith("Z")


def _wait_dead(pid: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


def test_names_syncer_reads_string_and_list_commands():
    assert syncer_reaper.names_syncer("bash -c '/x/target/debug/yeto-syncer --port 1'")
    assert syncer_reaper.names_syncer([Path("/x/yeto-syncer"), "--port", "1"])
    assert not syncer_reaper.names_syncer(["cargo", "build", "-q"])


INNER = '''
import subprocess, sys, pytest

BINARY = {binary!r}
PIDS = {pids!r}


def _record(process):
    with open(PIDS, "a") as out:
        out.write(f"{{process.pid}}\\n")


@pytest.fixture(scope="module")
def module_syncer():
    process = subprocess.Popen([BINARY, "300"])
    _record(process)
    yield process  # never stopped by the fixture


@pytest.fixture
def function_syncer():
    process = subprocess.Popen([BINARY, "300"])
    _record(process)
    return process


def test_function_fixture_leak(function_syncer):
    assert function_syncer.poll() is None


def test_module_fixture_still_running_in_its_module(module_syncer):
    assert module_syncer.poll() is None


def test_module_fixture_kept_for_the_next_test(module_syncer):
    assert module_syncer.poll() is None


def test_bash_wrapper_child_left_behind_by_a_failure():
    # the wrapper forks the syncer (two commands), so killing bash alone orphans it
    wrapper = subprocess.Popen(["bash", "-c", f"{{BINARY}} 300 & echo $! >> {{PIDS}}; wait"])
    for _ in range(200):
        if open(PIDS).read().count("\\n") >= 3:
            break
        __import__("time").sleep(0.05)
    wrapper.kill()
    raise AssertionError("fails with the syncer still running")
'''


def test_leaked_syncers_are_killed_when_their_test_or_module_ends(tmp_path):
    binary = _fake_syncer(tmp_path)
    pids = tmp_path / "pids.txt"
    pids.write_text("")
    inner = tmp_path / "inner"
    inner.mkdir()
    (inner / "test_inner_leak.py").write_text(textwrap.dedent(INNER.format(binary=str(binary), pids=str(pids))))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(TESTS), os.environ.get("PYTHONPATH", "")])}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "syncer_reaper", "-p", "no:cacheprovider",
         "--rootdir", str(inner), str(inner)],
        cwd=inner, env=env, capture_output=True, text=True, timeout=120)
    assert "1 failed, 3 passed" in result.stdout, result.stdout + result.stderr
    started = [int(line) for line in pids.read_text().split()]
    assert len(started) == 3, started
    survivors = [pid for pid in started if not _wait_dead(pid)]
    assert survivors == [], f"syncer processes outlived the session: {survivors}"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_no_orphaned_checkout_syncer_after_reap_orphans(tmp_path):
    binary = _fake_syncer(tmp_path)
    # orphan it on purpose: the intermediate shell exits at once
    subprocess.run(["sh", "-c", f"{binary} 300 >/dev/null 2>&1 &"], check=True)
    deadline = time.monotonic() + 10
    while not [row for row in syncer_reaper.checkout_syncers(tmp_path) if row[1] == 1]:
        assert time.monotonic() < deadline, "the orphaned fake syncer did not show up"
        time.sleep(0.05)
    killed = syncer_reaper.reap_orphans(tmp_path)
    assert killed and all(_wait_dead(pid) for pid in killed)
    assert syncer_reaper.checkout_syncers(tmp_path) == []
