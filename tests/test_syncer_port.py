"""A test syncer's port comes from the syncer itself (tests/syncer_port.py).

A port picked before the start can be taken by another test process before the syncer
binds it. The old start then saw something listen on the port and went on: its test
talked to the other process's syncer (a HELLO HMAC mismatch, a member another test
retired). Now a syncer that cannot bind is an error at its start, and a syncer on
``--port 0`` is reached on the port it bound.
"""

from __future__ import annotations

import shutil
import socket
from pathlib import Path

import pytest

from syncer_port import LISTENING, on_port_zero
from test_dispense_syncer import Syncer, syncer_binary  # noqa: F401 - the fixture

pytest_plugins = ["pytester"]


def test_a_syncer_on_port_zero_is_reached_on_the_port_it_bound(syncer_binary, tmp_path):  # noqa: F811
    syncer = Syncer(syncer_binary, tmp_path)
    try:
        assert syncer.port > 0
        assert LISTENING.search((tmp_path / "syncer.out").read_text()).group(1) == str(syncer.port)
        socket.create_connection(("127.0.0.1", syncer.port), timeout=5).close()
    finally:
        syncer.stop()


def test_a_syncer_whose_port_is_taken_fails_its_start_instead_of_reaching_the_other_listener(
        syncer_binary, tmp_path):  # noqa: F811
    with socket.socket() as other:  # another process's server on the port the test picked
        other.bind(("0.0.0.0", 0))
        other.listen()
        with pytest.raises(RuntimeError, match="bind port"):
            Syncer(syncer_binary, tmp_path, port=other.getsockname()[1])


def test_on_port_zero_rewrites_the_one_port_flag():
    assert on_port_zero("yeto-syncer --port 29400 --learners 2") == "yeto-syncer --port 0 --learners 2"
    with pytest.raises(ValueError, match="one --port"):
        on_port_zero("a --port 1 && b --port 2")


def test_a_test_syncer_runs_from_a_copy_no_cargo_build_rewrites(tmp_path, monkeypatch):
    """Every ``cargo build`` (a fresh one too) deletes and rewrites target/debug/yeto-syncer:
    another process starting a syncer from that path at that moment failed to exec or was
    killed before it printed anything (exit -9)."""
    import syncer_build

    monkeypatch.setattr(syncer_build, "COPIES", tmp_path / "copies")
    built = tmp_path / "target" / "yeto-syncer"
    built.parent.mkdir()
    built.write_bytes(b"v1")
    first = syncer_build.private_copy(built)
    built.unlink()  # what a rebuild in another process does to the build output
    built.write_bytes(b"v2")
    assert first.read_bytes() == b"v1" and first.name == "yeto-syncer"
    assert first.is_relative_to(tmp_path / "copies")
    second = syncer_build.private_copy(built)
    assert second != first and second.read_bytes() == b"v2" and first.read_bytes() == b"v1"


def test_a_nested_in_process_session_leaves_this_processs_binary_copy(syncer_binary, pytester):  # noqa: F811
    """A test running pytest in process with this conftest (tests/test_conftest_ray_guard.py)
    ends a session in the same process: removing "this process's" copies there removed the
    copy every later syncer test of the outer session executes (FileNotFoundError, exit 127)."""
    shutil.copy(Path(__file__).with_name("conftest.py"), pytester.path / "conftest.py")
    pytester.makepyfile(test_nested="def test_nothing():\n    pass\n")
    pytester.runpytest("-p", "no:cacheprovider").assert_outcomes(passed=1)
    assert Path(syncer_binary).is_file()
