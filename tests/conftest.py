def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: end-to-end tests that require a CUDA accelerator (run on the "
        "self-hosted GPU runner; skipped automatically without CUDA)",
    )


import pytest as _pytest


@_pytest.fixture(autouse=True)
def _no_teardown_diagnostics(monkeypatch):
    """launcher.teardown_island pulls logs over ssh/sky before a teardown;
    unit tests exercise it explicitly (test_verda_provider.py) instead."""
    monkeypatch.setenv("YETO_TEARDOWN_DIAG", "0")


@_pytest.fixture(autouse=True)
def _fixed_thread_reading(monkeypatch):
    """launch-preflight-guards: launch() reads this user's thread count from
    /proc; unit tests must not depend on (or wait for) the host's threads.
    Tests of the thread preflight pass their own counter."""
    from yeto import launch_preflight

    monkeypatch.setattr(launch_preflight, "count_threads",
                        lambda *a, **k: launch_preflight.ThreadCount(100, 0, False, 1))
