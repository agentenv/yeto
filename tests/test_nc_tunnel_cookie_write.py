"""The relay cookie file is rewritten by every ssh ProxyCommand that finds it stale,
so several processes write it at once: each needs a temporary of its own."""

import errno
import json
import os
import subprocess

import pytest

from yeto.launch.nc import tunnel

REAL_DUMP = json.dump


@pytest.fixture(autouse=True)
def _tools(monkeypatch):
    monkeypatch.setattr(tunnel, "node_bin", lambda: "node")
    monkeypatch.setattr(tunnel, "tool_dir", lambda: "tools")


def _runner(base_url: str):
    def run(argv, **kw):
        doc = {"base_url": base_url, "cookie": f"c-{base_url}"}
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(doc) + "\n", stderr="")
    return run


def test_concurrent_fetches_do_not_steal_each_others_temporary(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path))
    real_replace = os.replace
    nested, started = [], []

    def replace(src, dst):
        if not started:  # a second ProxyCommand refreshes the cookie between our write and rename
            started.append(1)
            nested.append(tunnel.fetch_endpoint(1, 9000, run=_runner("https://b")))
        real_replace(src, dst)

    monkeypatch.setattr(tunnel.os, "replace", replace)
    first = tunnel.fetch_endpoint(1, 9000, run=_runner("https://a"))
    assert first.base_url == "https://a" and nested[0].base_url == "https://b"
    assert tunnel.cached_endpoint(1, 9000).base_url == "https://a"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["relay-1-9000.json"]
    assert (tmp_path / "relay-1-9000.json").stat().st_mode & 0o777 == 0o600


def test_failed_write_keeps_the_old_cookie_and_leaves_no_temporary(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path))
    tunnel.fetch_endpoint(1, 9000, run=_runner("https://old"))

    def dump(*a, **kw):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(tunnel.json, "dump", dump)
    with pytest.raises(OSError):
        tunnel.fetch_endpoint(1, 9000, run=_runner("https://new"))
    monkeypatch.setattr(tunnel.json, "dump", REAL_DUMP)
    assert tunnel.cached_endpoint(1, 9000).base_url == "https://old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["relay-1-9000.json"]


def test_cookie_file_format_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path))
    tunnel.fetch_endpoint(2, 9001, run=_runner("https://x"))
    assert (tmp_path / "relay-2-9001.json").read_text() == json.dumps(
        {"base_url": "https://x", "cookie": "c-https://x"})
