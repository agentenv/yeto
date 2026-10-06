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
