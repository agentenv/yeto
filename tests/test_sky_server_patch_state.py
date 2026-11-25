"""The patch state that counts is the sky API server's (sky provisions there, maybe from
another environment), not this Python's; anything unreadable is "not applied"."""

import json
import os
import sys
from types import SimpleNamespace

import pytest

import yeto.sky_patches as patches
from yeto.sky_patches import verda as vp

TARGET = vp.TARGET_MODULE


class _Common:
    def __init__(self, endpoint="http://127.0.0.1:46580", local=True):
        self.endpoint, self.local = endpoint, local

    def get_server_url(self):
        return self.endpoint

    def is_api_server_local(self, endpoint):
        return self.local


def _runner(site, status):
    calls = []

    def run(argv, **kw):
        calls.append((argv, kw))
        if "sysconfig" in argv[-1]:
            return SimpleNamespace(stdout=f"{site}\n")
        return SimpleNamespace(stdout=json.dumps({"status": status, "version": "0.14.3"}) + "\n")

    return run, calls


def _hook(site, mtime):
    path, _ = patches.ensure_local_pth("/repo/yeto", site_dir=str(site))
    os.utime(path, (mtime, mtime))
    return path


def test_a_remote_server_is_not_applied():
    ok, why = patches.server_patch_state(TARGET, server_common=_Common(local=False))
    assert not ok and "remote" in why


def test_the_local_servers_own_python_is_probed_without_pythonpath(tmp_path, monkeypatch):
    _hook(tmp_path, 1_000.0)
    monkeypatch.setattr(patches, "local_server_process", lambda port: ("/venvs/sky-api/bin/python", 2_000.0))
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    run, calls = _runner(tmp_path, {TARGET: "applied"})
    ok, why = patches.server_patch_state(TARGET, server_common=_Common(), run=run)
    assert ok and "/venvs/sky-api/bin/python" in why
    assert all(argv[0] == "/venvs/sky-api/bin/python" for argv, _ in calls)
    assert all("PYTHONPATH" not in kw["env"] and kw["cwd"] == "/" for _, kw in calls)
    assert f"import {TARGET}" in calls[-1][0][-1]


@pytest.mark.parametrize("case", ["no hook", "hook newer than the server", "probe skipped", "not absolute"])
def test_every_unproven_state_is_not_applied(tmp_path, monkeypatch, case):
    python = "python3" if case == "not absolute" else "/venvs/sky-api/bin/python"
    if case != "no hook":
        _hook(tmp_path, 3_000.0 if case == "hook newer than the server" else 1_000.0)
    monkeypatch.setattr(patches, "local_server_process", lambda port: (python, 2_000.0))
    status = {TARGET: "skipped: sky 0.15.0 not verified"} if case == "probe skipped" else {TARGET: "applied"}
    run, _ = _runner(tmp_path, status)
    ok, why = patches.server_patch_state(TARGET, server_common=_Common(), run=run)
    assert not ok
    assert {"no hook": "no yeto sky patch hook", "hook newer than the server": "after the sky API server started",
            "probe skipped": "not verified", "not absolute": "not an absolute"}[case] in why


def test_no_server_running_probes_the_one_sky_would_start_from_this_python(tmp_path, monkeypatch):
    _hook(tmp_path, 9_999_999_999.0)  # no start time to compare: the server is not running yet
    monkeypatch.setattr(patches, "local_server_process", lambda port: None)
    run, calls = _runner(tmp_path, {TARGET: "applied"})
    assert patches.server_patch_state(TARGET, server_common=_Common(), run=run)[0]
    assert calls[0][0][0] == sys.executable


def test_the_server_process_is_found_by_its_port():
    assert patches._server_port("http://127.0.0.1:8892") == 8892
    assert patches._server_port("http://127.0.0.1") == 46580
    assert patches._cmdline_port(["/p/python", "-m", "sky.server.server", "--port", "8892"]) == 8892
    assert patches._cmdline_port(["/p/python", "-m", "sky.server.server"]) == 46580


def test_an_unpatched_server_refuses_verda_unless_warn(monkeypatch, capsys):
    monkeypatch.setattr(patches, "server_patch_state", lambda target: (False, "no yeto sky patch hook"))
    with pytest.raises(ValueError) as refused:
        patches.require_patched_server(TARGET, "Verda island", environ={})
    msg = str(refused.value)
    assert "-m yeto.sky_patches install" in msg and "sky api stop" in msg and patches.SERVER_PATCH_ENV in msg
    assert patches.require_patched_server(TARGET, "Verda island", environ={patches.SERVER_PATCH_ENV: "warn"}) is False
    assert "auto-recovery off" in capsys.readouterr().err
    monkeypatch.setattr(patches, "server_patch_state", lambda target: (True, "applied"))
    assert patches.require_patched_server(TARGET, "Verda island", environ={}) is True


def test_the_sky_provider_checks_the_server_before_a_verda_order(monkeypatch):
    from tests.test_nc_sky_provider import FakeSky
    from yeto.launch.nc.sky_provider import SkyProvider

    monkeypatch.delenv(patches.SERVER_CHECK_ENV, raising=False)
    monkeypatch.delenv(patches.SERVER_PATCH_ENV, raising=False)
    from yeto.launch.nc.sky_provider import AWS_FORWARDED_ENVS, AWS_INHERITED_ENVS

    for name in AWS_FORWARDED_ENVS + AWS_INHERITED_ENVS:  # the run AWS env check is not under test here
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(patches, "require_matching_api_server", lambda out=None: None)
    monkeypatch.setattr(patches, "server_patch_state", lambda target: (False, "no yeto sky patch hook"))
    sky = FakeSky()
    verda = SkyProvider("run-v", slots=[1], resources={"infra": "verda", "accelerators": "L40S:1"}, sky=sky)
    verda._real_sky = True
    with pytest.raises(ValueError, match="not in effect in the sky API server"):
        verda.order(1)
    assert not [c for c in sky.calls if c[0] == "launch"]
    aws = SkyProvider("run-a", slots=[1], resources={"infra": "aws", "accelerators": "L4:1"}, sky=sky)
    aws._real_sky = True
    aws.order(1)  # no Verda offer: no server patch requirement
    assert [c for c in sky.calls if c[0] == "launch"]


def test_the_server_is_matched_by_port_even_with_a_one_string_command_line(monkeypatch):
    import psutil

    procs = [SimpleNamespace(info={"pid": 3, "cmdline": ["/bin/zsh", "-c", "python -m sky.server.server --port 46580"],
                                   "create_time": 3.0}),
             SimpleNamespace(info={"pid": 1, "cmdline": ["/int/venv/bin/python -m sky.server.server --host 127.0.0.1 "
                                                           "--port 46591"], "create_time": 1.0}),
             SimpleNamespace(info={"pid": 2, "cmdline": ["/venvs/sky-api/bin/python", "-m", "sky.server.server",
                                                           "--host=127.0.0.1", "--port=46580"], "create_time": 2.0})]
    monkeypatch.setattr(psutil, "process_iter", lambda attrs=None: iter(procs))
    assert patches.local_server_process(46580) == ("/venvs/sky-api/bin/python", 2.0)
    assert patches.local_server_process(46591) == ("/int/venv/bin/python", 1.0)
    assert patches.local_server_process(8892) is None


def test_a_verda_sky_pool_is_refused_at_its_start_and_dry_run_unless_warn(monkeypatch, capsys):
    from yeto.launch.nc import sky_provider as sp

    for name in sp.AWS_FORWARDED_ENVS + sp.AWS_INHERITED_ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(patches.SERVER_CHECK_ENV, raising=False)
    monkeypatch.delenv(patches.SERVER_PATCH_ENV, raising=False)
    monkeypatch.setattr(patches, "server_patch_state", lambda target: (False, "no yeto sky patch hook"))
    cfg = {"target": 1, "accounts": [1], "shard_base": "s", "run": "run-v",
           "provider": {"kind": "sky", "resources": {"infra": "verda", "accelerators": "L40S:1"}}}
    assert [p for p in sp.start_problems(cfg) if "Verda patch is not in effect" in p]
    assert [p for p in sp.offline_problems(cfg) if "Verda patch is not in effect" in p]
    monkeypatch.setenv(patches.SERVER_PATCH_ENV, "warn")
    assert not [p for p in sp.start_problems(cfg) if "Verda" in p] and "proceeding unpatched" in capsys.readouterr().out
    aws = {**cfg, "provider": {"kind": "sky", "resources": {"infra": "aws", "accelerators": "L4:1"}}}
    monkeypatch.delenv(patches.SERVER_PATCH_ENV)
    assert not [p for p in sp.start_problems(aws) if "Verda" in p]
