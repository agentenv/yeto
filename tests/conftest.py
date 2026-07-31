def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: tests that use a locally running external service",
    )
    config.addinivalue_line(
        "markers",
        "gpu: end-to-end tests that require a CUDA accelerator (run on the "
        "self-hosted GPU runner; skipped automatically without CUDA)",
    )
