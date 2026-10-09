RAY_LOCAL_HINT = (
    "this test starts Ray on this machine (ray.init). "
    "Add the ray_local marker to the test, and run it only with --run-ray-local."
)


def pytest_addoption(parser):
    parser.addoption(
        "--run-ray-local",
        action="store_true",
        default=False,
        help="collect and run tests marked ray_local (they start Ray on this machine)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "ray_local: test starts Ray on this machine (gcs_server/raylet/ray.init); "
        "not collected unless --run-ray-local is given",
    )
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


def pytest_collection_modifyitems(config, items):
    """Deselect ray_local tests unless --run-ray-local is given (deselect, not skip,
    so that no fixture of these tests runs)."""
    if config.getoption("--run-ray-local"):
        return
    keep, drop = [], []
    for item in items:
        (drop if item.get_closest_marker("ray_local") else keep).append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep


def _blocked_ray_init(*_args, **_kwargs):
    raise RuntimeError(RAY_LOCAL_HINT)


def _patch_ray_init(ray_module):
    """Replace ray.init (and the worker init it aliases); return an undo function."""
    import importlib

    worker = importlib.import_module("ray._private.worker")
    saved = [(ray_module, ray_module.init), (worker, worker.init)]
    ray_module.init = _blocked_ray_init
    worker.init = _blocked_ray_init

    def undo():
        for owner, original in saved:
            owner.init = original

    return undo


class _PatchRayOnImport:
    """Meta-path finder: patch ray.init right after `import ray` runs.
    The guard does not import ray itself, so tests that assert
    "ray" not in sys.modules still hold."""

    def __init__(self):
        self.undo = None

    def find_spec(self, name, path=None, target=None):
        if name != "ray":
            return None
        import importlib.util
        import sys

        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec("ray")
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        real_exec = spec.loader.exec_module
        finder = self

        class _Loader:
            def create_module(self, spec):
                return None

            def exec_module(self, module):
                real_exec(module)
                finder.undo = _patch_ray_init(module)

        spec.loader = _Loader()
        return spec


@_pytest.fixture(autouse=True)
def _block_unmarked_ray_init(request):
    """Default run: make ray.init fail in tests without the ray_local marker.
    This does not stop a subprocess that runs `ray start`; the marker audit covers that."""
    if request.config.getoption("--run-ray-local") or request.node.get_closest_marker("ray_local"):
        yield
        return
    import sys

    if "ray" in sys.modules:
        undo = _patch_ray_init(sys.modules["ray"])
        try:
            yield
        finally:
            undo()
        return
    finder = _PatchRayOnImport()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
        if finder.undo is not None:
            finder.undo()
