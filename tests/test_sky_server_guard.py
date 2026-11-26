"""The test session never reaches or starts a sky API server (tests/sky_server_guard.py).

A full run once left a ``python -m sky.server.server`` of the test venv on this machine's
sky port: a launcher teardown in a test whose helper had cleared the test defaults ran
the real ``sky logs``, and that sky client started a local server, with the test's
temporary HOME, that outlived the session.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

import sky_server_guard as guard


def _drop_guard_env(monkeypatch):
    """What a hermetic helper does: no SKYPILOT_* variable in this process."""
    for name in list(os.environ):
        if name.startswith("SKYPILOT_"):
            monkeypatch.delenv(name)


def test_every_test_runs_against_the_refusing_endpoint():
    assert guard.ENDPOINT.endswith(":1")
    for name, value in guard.ENV.items():
        assert os.environ[name] == value


def test_an_auto_start_in_this_process_raises_a_clear_error(monkeypatch):
    pytest.importorskip("sky")
    from sky.server import common

    assert guard.is_patched(common), "the guard did not patch sky.server.common; nothing below may run"
    _drop_guard_env(monkeypatch)
    for call in (common.check_server_healthy_or_start_fn, common._start_api_server):
        with pytest.raises(guard.SkyServerStartRefused, match="unit tests never start one"):
            call()
    ran = []
    sdk_call = common.check_server_healthy_or_start(lambda: ran.append("sent"))  # how sky's SDK calls wrap
    with pytest.raises(guard.SkyServerStartRefused, match="check_server_healthy_or_start_fn"):
        sdk_call()
    assert ran == []
    # without the variable sky's endpoint is still the refusing one, never the local default
    assert common.get_server_url() == guard.ENDPOINT
    assert not common.is_api_server_local(common.get_server_url())


def test_a_child_without_the_variables_still_gets_them_and_starts_no_server(monkeypatch, tmp_path):
    pytest.importorskip("sky")
    _drop_guard_env(monkeypatch)
    env = {k: v for k, v in os.environ.items() if not k.startswith("SKYPILOT_")}
    env.update(HOME=str(tmp_path), SKY_RUNTIME_DIR=str(tmp_path))
    names = sorted(guard.ENV)
    code = (
        "import json, os, sys\n"
        f"names = {names!r}\n"
        "seen = {n: os.environ.get(n) for n in names}\n"
        "print(json.dumps({'seen': seen}), flush=True)\n"
        f"if seen != {dict(guard.ENV)!r}:\n"
        "    sys.exit(3)  # unguarded: never touch sky\n"
        "from sky.server import common\n"
        "dial = getattr(common, 'get_local_server_dial_url', lambda host: common.get_server_url(host))('127.0.0.1')\n"
        "out = {'endpoint': common.get_server_url(), 'dial': dial, 'local': common.is_api_server_local()}\n"
        f"if dial != {guard.ENDPOINT!r} or out['local']:\n"
        "    print(json.dumps(out), flush=True)\n"
        "    sys.exit(4)  # sky would dial another endpoint: never call it\n"
        "for name in ('check_local_api_server_enabled_or_raise', 'check_server_healthy_or_start_fn'):\n"
        "    if not hasattr(common, name):\n"
        "        out[name] = 'absent'\n"
        "        continue\n"
        "    try:\n"
        "        getattr(common, name)()\n"
        "        out[name] = 'returned'\n"
        "    except BaseException as error:\n"
        "        out[name] = type(error).__name__ + ': ' + str(error)[:300]\n"
        "print(json.dumps(out), flush=True)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], env=env, cwd=tmp_path, capture_output=True, text=True,
                          timeout=120)
    lines = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
    assert lines and lines[0]["seen"] == guard.ENV, proc.stdout + proc.stderr
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = lines[1]
    assert out["endpoint"] == out["dial"] == guard.ENDPOINT and out["local"] is False
    # sky's own switch refuses a local start (a sky version without it: the endpoint alone)
    switch = out["check_local_api_server_enabled_or_raise"]
    assert switch == "absent" or (switch.startswith("RuntimeError") and "SKYPILOT_DISABLE_LOCAL_API_SERVER" in switch)
    # and the refusing endpoint is no local server: a connection error, no server started
    assert out["check_server_healthy_or_start_fn"].startswith("ApiServerConnectionError"), out
    assert not (tmp_path / ".sky" / "api_server" / "server.log").exists()


def test_a_positional_env_of_popen_gets_the_variables_too(monkeypatch):
    _drop_guard_env(monkeypatch)
    code = f"import os; print(os.environ.get({'SKYPILOT_DISABLE_LOCAL_API_SERVER'!r}))"
    env = {k: v for k, v in os.environ.items() if not k.startswith("SKYPILOT_")}
    proc = subprocess.Popen([sys.executable, "-c", code], -1, None, None, subprocess.PIPE, None, None, True,
                            False, None, env, text=True)
    out, _ = proc.communicate(timeout=60)
    assert out.strip() == "1"
