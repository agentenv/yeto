"""yeto refuses a local sky API server of another sky version than its own (sky
accepts it silently), and checks only when the launch sends sky requests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from yeto import sky_patches


class _ServerCommon:
    """sky.server.common's stand-in: one API server at ``endpoint``."""

    def __init__(self, version, *, status="healthy", local=True, endpoint="http://127.0.0.1:46580"):
        self.version, self.status, self.local, self.endpoint = version, status, local, endpoint

    def get_server_url(self):
        return self.endpoint

    def get_api_server_status(self, endpoint):
        assert endpoint == self.endpoint
        return SimpleNamespace(status=SimpleNamespace(value=self.status), version=self.version)

    def is_api_server_local(self, endpoint):
        return self.local


def test_a_local_sky_api_server_of_another_version_is_refused_with_the_way_out():
    import sky

    mine = sky.__version__
    major, minor, *_ = mine.split(".")
    other = f"{major}.{int(minor) + 1}.0"  # any version but this Python's own
    assert sky_patches.api_server_version_problem(_ServerCommon(mine), environ={}) is None
    assert sky_patches.api_server_version_problem(_ServerCommon(None, status="unhealthy"), environ={}) is None
    msg, refuse = sky_patches.api_server_version_problem(_ServerCommon(other), environ={})
    assert refuse and f"runs {other}" in msg and f"runs SkyPilot {mine}" in msg
    assert "sky api stop; sky api start" in msg and "sky api start --port" in msg and "SKY_RUNTIME_DIR" in msg
    msg, refuse = sky_patches.api_server_version_problem(_ServerCommon(other),
                                                         environ={"YETO_SKY_SERVER_MISMATCH": "allow"})
    assert not refuse and "proceeding" in msg
    msg, refuse = sky_patches.api_server_version_problem(
        _ServerCommon(other, local=False, endpoint="https://sky.example"), environ={})
    assert not refuse and "https://sky.example" in msg  # a remote server: sky's supported setup, a warning


def test_the_launch_preflight_checks_the_sky_server_only_when_the_launch_uses_sky(monkeypatch):
    from yeto import launch_preflight as lp

    modal, ssh, aws = (SimpleNamespace(cloud=c) for c in ("modal", "ssh", "aws"))
    no_sync = SimpleNamespace(rl_single_island_no_sync=True, controller="local")
    assert not lp.launch_uses_sky(no_sync, [modal]) and not lp.launch_uses_sky(no_sync, [ssh])
    assert lp.launch_uses_sky(no_sync, [modal, aws])
    assert lp.launch_uses_sky(SimpleNamespace(rl_single_island_no_sync=False), [modal])  # the syncer
    assert lp.launch_uses_sky(SimpleNamespace(controller="head", rl_single_island_no_sync=True), [modal])
    monkeypatch.delenv("YETO_SKY_SERVER_CHECK")
    monkeypatch.setattr(sky_patches, "api_server_version_problem", lambda: ("server 0.13.0", True))
    with pytest.raises(ValueError, match="server 0.13.0"):
        sky_patches.require_matching_api_server()
    monkeypatch.setenv("YETO_SKY_SERVER_CHECK", "off")
    sky_patches.require_matching_api_server()  # off: no check
