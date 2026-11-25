"""The patch state that counts is the sky API server's (sky provisions there, maybe from
another environment), not this Python's. It is read from that environment's files only
(nothing of it is run), from a server process of this user that listens on the port;
anything unreadable is "not applied"."""

import os
import subprocess
import sys
from types import SimpleNamespace

import psutil
import pytest

import yeto.sky_patches as patches
from yeto.sky_patches import verda as vp

TARGET = vp.TARGET_MODULE
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(patches.__file__))))
VERDA_SOURCE = open(os.path.join(REPO, "yeto", "sky_patches", "verda.py"), encoding="utf-8").read()


class _Common:
    def __init__(self, endpoint="http://127.0.0.1:46580", local=True):
        self.endpoint, self.local = endpoint, local

    def get_server_url(self):
        return self.endpoint

    def is_api_server_local(self, endpoint):
        return self.local


def _venv(tmp_path, *, sky="0.14.3", hook=True, attrs=vp.REQUIRED_ATTRS, mtime=1_000.0, repo=None):
    """A virtualenv on disk (never run): bin/python, its site-packages with sky's dist-info,
    the Verda module and (``hook``) the yeto hook pointing at a yeto checkout."""
    venv = tmp_path / "venv"
    site = venv / "lib" / "python3.12" / "site-packages"
    (site / "sky" / "provision" / "verda").mkdir(parents=True)
    (venv / "bin").mkdir()
    (venv / "bin" / "python").write_text("#!/bin/false\n")
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (site / f"skypilot-{sky}.dist-info").mkdir()
    (site / f"skypilot-{sky}.dist-info" / "METADATA").write_text(f"Metadata-Version: 2.1\nName: skypilot\nVersion: {sky}\n")
    (site / "sky" / "provision" / "verda" / "instance.py").write_text(
        "".join(f"def {a}():\n    pass\n" for a in attrs if a != "verda") + "from sky.adaptors import verda\n")
    if repo is None:
        repo = tmp_path / "yeto-checkout"
        (repo / "yeto" / "sky_patches").mkdir(parents=True)
        (repo / "yeto" / "sky_patches" / "__init__.py").write_text("")
        (repo / "yeto" / "sky_patches" / "verda.py").write_text(VERDA_SOURCE)
    files = [*(repo / "yeto" / "sky_patches").glob("*.py")]
    if hook:
        path, _ = patches.ensure_local_pth(str(repo), site_dir=str(site))
        files.append(path)
    for f in files:
        os.utime(f, (mtime, mtime))
    return str(venv / "bin" / "python"), site, repo


@pytest.fixture
def no_run(monkeypatch):
    """Nothing of the server's environment is ever run."""
    def refuse(*a, **k):
        raise AssertionError(f"ran a process: {a!r}")

    for name in ("run", "Popen", "check_output", "call"):
        monkeypatch.setattr(subprocess, name, refuse)
    monkeypatch.setattr(os, "system", refuse)


def _server(monkeypatch, python, started=2_000.0):
    monkeypatch.setattr(patches, "local_server_process", lambda port, host="127.0.0.1": (python, started))


def test_a_remote_server_is_not_applied():
    ok, why = patches.server_patch_state(TARGET, server_common=_Common(local=False))
    assert not ok and "remote" in why


def test_a_patched_server_is_read_from_its_files_without_running_anything(tmp_path, monkeypatch, no_run):
    python, _, _ = _venv(tmp_path)
    _server(monkeypatch, python)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake-secret-never-handed-on")
    ok, why = patches.server_patch_state(TARGET, server_common=_Common())
    assert ok and python in why


@pytest.mark.parametrize("case,expect", [
    ("no hook", "no yeto sky patch hook"),
    ("hook newer than the server", "changed after the sky API server started"),
    ("patch source newer than the server", "changed after the sky API server started"),
    ("unverified sky", "not in verified"),
    ("module lacks an attribute", "lacks ['_filter_instances']"),
    ("not absolute", "not an absolute"),
    ("not a virtualenv", "cannot be located"),
])
def test_every_unproven_state_is_not_applied(tmp_path, monkeypatch, no_run, case, expect):
    python, site, repo = _venv(
        tmp_path, hook=case != "no hook", sky="0.15.0" if case == "unverified sky" else "0.14.3",
        attrs=[a for a in vp.REQUIRED_ATTRS if a != "_filter_instances"] if case == "module lacks an attribute"
        else vp.REQUIRED_ATTRS)
    if case == "hook newer than the server":
        os.utime(site / patches.PTH_NAME, (3_000.0, 3_000.0))
    if case == "patch source newer than the server":  # the checkout was pulled after the server started
        os.utime(repo / "yeto" / "sky_patches" / "verda.py", (3_000.0, 3_000.0))
    if case == "not absolute":
        python = "python3"
    if case == "not a virtualenv":
        os.remove(os.path.join(os.path.dirname(os.path.dirname(python)), "pyvenv.cfg"))
    _server(monkeypatch, python)
    ok, why = patches.server_patch_state(TARGET, server_common=_Common())
    assert not ok and expect in why


def test_an_installed_yeto_in_the_server_env_is_the_patch_that_counts(tmp_path, monkeypatch, no_run):
    python, site, _ = _venv(tmp_path)
    (site / "yeto" / "sky_patches").mkdir(parents=True)
    (site / "yeto" / "sky_patches" / "__init__.py").write_text("")
    (site / "yeto" / "sky_patches" / "verda.py").write_text(  # an older yeto: verified for 0.13.0 only
        VERDA_SOURCE.replace('frozenset({"0.13.0", "0.14.3"})', 'frozenset({"0.13.0"})'))
    for f in (site / "yeto" / "sky_patches").glob("*.py"):
        os.utime(f, (1_000.0, 1_000.0))
    _server(monkeypatch, python)
    ok, why = patches.server_patch_state(TARGET, server_common=_Common())
    assert not ok and "not in verified ['0.13.0']" in why


def test_a_pth_that_imports_yeto_is_unreadable(tmp_path, monkeypatch, no_run):
    python, site, _ = _venv(tmp_path)
    (site / "__editable__.yeto-0.1.pth").write_text("import __editable___yeto_finder; __editable___yeto_finder.install()\n")
    _server(monkeypatch, python)
    ok, why = patches.server_patch_state(TARGET, server_common=_Common())
    assert not ok and "cannot be read without running it" in why


def test_no_server_running_reads_the_one_sky_would_start_from_this_python(tmp_path, monkeypatch, no_run):
    python, site, _ = _venv(tmp_path, mtime=9_999_999_999.0)  # no start time to compare: not running yet
    monkeypatch.setattr(patches, "local_server_process", lambda port, host="127.0.0.1": None)
    monkeypatch.setattr(sys, "executable", python)
    assert patches.server_patch_state(TARGET, server_common=_Common())[0]


def test_the_server_address_comes_from_the_endpoint():
    assert patches._server_addr("http://127.0.0.1:8892") == ("127.0.0.1", 8892)
    assert patches._server_addr("http://localhost") == ("localhost", 46580)
    assert patches._cmdline_port(["/p/python", "-m", "sky.server.server", "--port", "8892"]) == 8892
    assert patches._cmdline_port(["/p/python", "-m", "sky.server.server"]) == 46580


def _proc(pid, cmdline, *, uid=None, listens=(), create_time=1.0):
    uid = os.getuid() if uid is None else uid
    conns = [SimpleNamespace(status=psutil.CONN_LISTEN, laddr=SimpleNamespace(port=p)) for p in listens]
    return SimpleNamespace(info={"pid": pid, "cmdline": cmdline, "create_time": create_time,
                                 "uids": SimpleNamespace(real=uid)},
                           net_connections=lambda kind="inet": conns)


def _procs(monkeypatch, procs, *, in_use=()):
    monkeypatch.setattr(psutil, "process_iter", lambda attrs=None: iter(procs))
    monkeypatch.setattr(patches, "_port_in_use", lambda host, port: port in in_use)


def test_the_server_is_this_users_process_listening_on_the_port(monkeypatch):
    _procs(monkeypatch, [
        _proc(3, ["/bin/zsh", "-c", "python -m sky.server.server --port 46580"]),
        _proc(1, ["/int/venv/bin/python -m sky.server.server --host 127.0.0.1 --port 46591"], listens=[46591]),
        _proc(2, ["/venvs/sky-api/bin/python", "-u", "-X", "dev", "-m", "sky.server.server", "--host=127.0.0.1",
                  "--port=46580"], listens=[46580], create_time=2.0),
    ])
    assert patches.local_server_process(46580) == ("/venvs/sky-api/bin/python", 2.0)
    assert patches.local_server_process(46591) == ("/int/venv/bin/python", 1.0)
    assert patches.local_server_process(8892) is None


def test_another_users_server_is_ignored_and_its_port_is_unreadable(monkeypatch):
    _procs(monkeypatch, [_proc(5, ["/home/eve/venv/bin/python", "-m", "sky.server.server"], uid=os.getuid() + 1,
                               listens=[46580])], in_use={46580})
    with pytest.raises(patches.ServerUnreadable, match="not by a sky API server process of this user"):
        patches.local_server_process(46580)
    ok, why = patches.server_patch_state(TARGET, server_common=_Common())
    assert not ok and "could not be inspected" in why


def test_a_matching_process_not_listening_is_not_the_server(monkeypatch):
    _procs(monkeypatch, [_proc(6, ["/venvs/old/bin/python", "-m", "sky.server.server"])])
    assert patches.local_server_process(46580) is None


def test_two_listening_servers_are_ambiguous(monkeypatch):
    _procs(monkeypatch, [_proc(7, ["/a/bin/python", "-m", "sky.server.server"], listens=[46580]),
                         _proc(8, ["/b/bin/python", "-m", "sky.server.server"], listens=[46580])])
    with pytest.raises(patches.ServerUnreadable, match="2 sky API server processes"):
        patches.local_server_process(46580)


def test_an_unpatched_server_refuses_verda_unless_warn(monkeypatch, capsys):
    monkeypatch.setattr(patches, "server_patch_state", lambda target: (False, "no yeto sky patch hook"))
    monkeypatch.setattr(patches, "server_python", lambda server_common=None: "/venvs/sky-api/bin/python")
    with pytest.raises(ValueError) as refused:
        patches.require_patched_server(TARGET, "Verda island", environ={})
    msg = str(refused.value)
    assert "/venvs/sky-api/bin/python -m yeto.sky_patches install" in msg and "sky api stop" in msg
    assert patches.SERVER_PATCH_ENV in msg
    assert patches.require_patched_server(TARGET, "Verda island", environ={patches.SERVER_PATCH_ENV: "warn"}) is False
    assert "proceeding unpatched" in capsys.readouterr().err
    monkeypatch.setattr(patches, "server_patch_state", lambda target: (True, "applied"))
    assert patches.require_patched_server(TARGET, "Verda island", environ={}) is True


def test_the_sky_provider_checks_the_server_before_a_verda_order(monkeypatch):
    from tests.test_nc_sky_provider import FakeSky
    from yeto.launch.nc.sky_provider import AWS_FORWARDED_ENVS, AWS_INHERITED_ENVS, SkyProvider

    monkeypatch.delenv(patches.SERVER_CHECK_ENV, raising=False)
    monkeypatch.delenv(patches.SERVER_PATCH_ENV, raising=False)
    for name in AWS_FORWARDED_ENVS + AWS_INHERITED_ENVS:  # the run AWS env check is not under test here
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(patches, "require_matching_api_server", lambda out=None: None)
    monkeypatch.setattr(patches, "server_patch_state", lambda target: (False, "no yeto sky patch hook"))
    monkeypatch.setattr(patches, "server_python", lambda server_common=None: None)
    sky = FakeSky()
    verda = SkyProvider("run-v", slots=[1], resources={"infra": "verda", "accelerators": "L40S:1"}, sky=sky,
                        allow_no_autodown=True)
    verda._real_sky = True
    with pytest.raises(ValueError, match="not in effect in the sky API server"):
        verda.order(1)
    assert not [c for c in sky.calls if c[0] == "launch"]
    aws = SkyProvider("run-a", slots=[1], resources={"infra": "aws", "accelerators": "L4:1"}, sky=sky)
    aws._real_sky = True
    aws.order(1)  # no Verda offer: no server patch requirement
    assert [c for c in sky.calls if c[0] == "launch"]


def test_a_verda_sky_pool_is_refused_at_its_start_and_dry_run_unless_warn(monkeypatch, capsys):
    from yeto.launch.nc import sky_provider as sp

    for name in sp.AWS_FORWARDED_ENVS + sp.AWS_INHERITED_ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(patches.SERVER_CHECK_ENV, raising=False)
    monkeypatch.delenv(patches.SERVER_PATCH_ENV, raising=False)
    monkeypatch.setattr(patches, "server_patch_state", lambda target: (False, "no yeto sky patch hook"))
    monkeypatch.setattr(patches, "server_python", lambda server_common=None: None)
    cfg = {"target": 1, "accounts": [1], "shard_base": "s", "run": "run-v",
           "provider": {"kind": "sky", "allow_no_autodown": True,
                        "resources": {"infra": "verda", "accelerators": "L40S:1"}}}
    assert [p for p in sp.start_problems(cfg) if "Verda patch is not in effect" in p]
    assert [p for p in sp.offline_problems(cfg) if "Verda patch is not in effect" in p]
    monkeypatch.setenv(patches.SERVER_PATCH_ENV, "warn")
    assert not [p for p in sp.start_problems(cfg) if "Verda" in p] and "proceeding unpatched" in capsys.readouterr().out
    aws = {**cfg, "provider": {"kind": "sky", "resources": {"infra": "aws", "accelerators": "L4:1"}}}
    monkeypatch.delenv(patches.SERVER_PATCH_ENV)
    assert not [p for p in sp.start_problems(aws) if "Verda" in p]
