"""The launch preflight refuses a Verda island unless yeto's Verda patch is in effect in
the sky API server (applied / not applied / unreadable), before any cloud work."""

import pytest

import yeto.sky_patches as patches

NO_SYNC = ("--rl-single-island-no-sync", "--controller", "local")


def _args(gpu, monkeypatch, tmp_path):
    import yeto.launcher as launcher
    from test_rl_algorithm_provenance import _launcher_args
    from yeto import runs

    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("YETO_PREFLIGHT_MODEL", "off")
    args = _launcher_args("ports", NO_SYNC, gpu=gpu)
    launcher._prepare_rl_args(args)
    return args


@pytest.fixture
def server(monkeypatch):
    """The sky API server as the preflight sees it: its version matches; its Verda patch
    state is what the test sets (never a real server)."""
    monkeypatch.delenv(patches.SERVER_CHECK_ENV, raising=False)
    monkeypatch.delenv(patches.SERVER_PATCH_ENV, raising=False)
    monkeypatch.setattr(patches, "require_matching_api_server", lambda out=None: None)
    monkeypatch.setattr(patches, "server_python", lambda server_common=None: "/venvs/sky-api/bin/python")
    asked = []

    def set_state(ok, why):
        monkeypatch.setattr(patches, "server_patch_state", lambda target: asked.append(target) or (ok, why))

    set_state.asked = asked
    return set_state


@pytest.mark.parametrize("state", [
    (False, "no yeto sky patch hook in the sky API server's environment"),
    (False, "the sky API server could not be inspected (port 46580 is in use, but not by a sky API server "
            "process of this user whose command line can be read)"),
])
def test_a_verda_island_is_refused_on_an_unpatched_or_unreadable_server(monkeypatch, tmp_path, server, state):
    import yeto.launch_preflight as lp

    args = _args("verda:1xrtx-pro-6000@FIN-01", monkeypatch, tmp_path)
    server(*state)
    with pytest.raises(ValueError) as refused:
        lp.pre_cloud_checks(args, threads=False)
    msg = str(refused.value)
    assert "Verda island" in msg and state[1] in msg
    assert "/venvs/sky-api/bin/python -m yeto.sky_patches install" in msg and patches.SERVER_PATCH_ENV in msg
    assert server.asked == [patches.verda.TARGET_MODULE]


def test_a_verda_island_passes_on_a_patched_server_and_warns_with_the_switch(monkeypatch, tmp_path, server, capsys):
    import yeto.launch_preflight as lp

    args = _args("verda:1xrtx-pro-6000@FIN-01", monkeypatch, tmp_path)
    server(True, "applied in the sky API server's environment")
    assert lp.pre_cloud_checks(args, threads=False)["preflight"]["verda_patched"] is True
    args = _args("verda:1xrtx-pro-6000@FIN-01", monkeypatch, tmp_path)
    server(False, "no yeto sky patch hook")
    monkeypatch.setenv(patches.SERVER_PATCH_ENV, "warn")
    assert lp.pre_cloud_checks(args, threads=False)["preflight"]["verda_patched"] is False
    assert "proceeding unpatched" in capsys.readouterr().err


def test_a_launch_without_verda_does_not_read_the_server_patch(monkeypatch, tmp_path, server):
    import yeto.launch_preflight as lp

    args = _args("aws:1xh100", monkeypatch, tmp_path)
    server(False, "no yeto sky patch hook")
    assert "verda_patched" not in lp.pre_cloud_checks(args, threads=False)["preflight"]
    assert server.asked == []
