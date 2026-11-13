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
    parser.addoption(
        "--shard",
        default=None,
        metavar="I/N",
        help="run only shard I (0-based) of N: the selected tests sorted by node id, "
        "dealt round-robin, so every test runs in exactly one shard",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_nebius_sg_cleanup: keep cli._nebius_sg_cleanup unpatched (the test fakes the cloud itself)",
    )
    config.addinivalue_line(
        "markers",
        "ray_local: test starts Ray on this machine (gcs_server/raylet/ray.init); "
        "not collected unless --run-ray-local is given",
    )


import os as _os

import pytest as _pytest

# YETO_REQUIRE_MILES=1 (the Miles image venv): a test that would skip because a
# ``miles.*`` module is not importable fails instead. Without it the loss_hub
# -> loss/hub move silently skipped every upstream-parity test.
if _os.environ.get("YETO_REQUIRE_MILES") == "1":
    _importorskip = _pytest.importorskip

    def _importorskip_requiring_miles(modname, *args, **kwargs):
        if modname == "miles" or modname.startswith(("miles.", "miles_plugins")):
            import importlib

            importlib.import_module(modname)  # ImportError: the path moved or Miles is missing
        return _importorskip(modname, *args, **kwargs)

    _pytest.importorskip = _importorskip_requiring_miles

@_pytest.fixture
def ports_v1_pin(monkeypatch):
    """Pin the run to the frozen ports-v1 lineage head (the DiLoCo fork/image).

    Feature gates (DP replicas, sharded masters, distopt reader, loss variants)
    and the pin-bound GPU declarations hold on that lineage; on the k31 lineage
    they are off until the batch that brings the feature back (B2/B8) or the
    GPU re-verification lands (yeto-adapter.md 4.2/4.5). Tests of the gate and
    declaration machinery use this fixture; test_rl_engine_pins checks the k31
    pin's own (empty) state.
    """
    from yeto.rl.adapters.miles import config as miles_config
    from yeto.rl.adapters.miles import pins

    monkeypatch.setattr(pins, "MILES_NEXT_COMMIT", pins.MILES_PORTS_V1_COMMIT)
    monkeypatch.setattr(miles_config, "MILES_NEXT_COMMIT", pins.MILES_PORTS_V1_COMMIT)
    return pins.MILES_PORTS_V1_COMMIT


# decoupling 7.7a: islands must declare a compat_group (no default in code).
# Unit tests that build identities at import time run as one H100 island;
# tests of the missing-value error delete it with monkeypatch.
_os.environ.setdefault("YETO_RL_COMPAT_GROUP", "nvidia-h100")


import os as _os

# secret-handling-hardening: the legacy syncer refuses to start without an
# island HMAC key. Unit tests spawn it without one on 127.0.0.1; the explicit
# opt-out is inherited by every syncer subprocess.
_os.environ.setdefault("YETO_SYNCER_ALLOW_UNAUTHENTICATED", "1")


_ORIGINAL_ISLAND_HMAC_KEY = _os.environ.get("YETO_ISLAND_HMAC_KEY")


def _reset_island_hmac_key():
    if _ORIGINAL_ISLAND_HMAC_KEY is None:
        _os.environ.pop("YETO_ISLAND_HMAC_KEY", None)
    else:
        _os.environ["YETO_ISLAND_HMAC_KEY"] = _ORIGINAL_ISLAND_HMAC_KEY


@_pytest.fixture(autouse=True)
def _island_hmac_key_does_not_leak():
    """launcher.island_hmac_secret may generate a legacy key into os.environ
    (also from module-scoped fixtures); reset it around every test so no
    syncer subprocess of another test inherits it."""
    _reset_island_hmac_key()
    yield
    _reset_island_hmac_key()


@_pytest.fixture(autouse=True)
def _reap_syncer_groups():
    """Kill every process group a test started with process_groups.group_popen,
    also when the test failed or timed out before its own cleanup ran."""
    yield
    from process_groups import reap_groups

    reap_groups()


@_pytest.fixture(autouse=True)
def _no_teardown_diagnostics(monkeypatch):
    """launcher.teardown_island pulls logs over ssh/sky before a teardown;
    unit tests exercise it explicitly (test_verda_provider.py) instead."""
    monkeypatch.setenv("YETO_TEARDOWN_DIAG", "0")


@_pytest.fixture(autouse=True)
def _no_real_nebius_sg_cleanup(request, monkeypatch):
    """`yeto down` of a Nebius run deletes the run's leaked security groups through
    the real Nebius API; unit tests must never reach the cloud. Tests of the cleanup
    call yeto.cloud.nebius_sg_cleanup with a fake sky module instead."""
    if request.node.get_closest_marker("real_nebius_sg_cleanup") is None:
        from yeto import cli

        monkeypatch.setattr(cli, "_nebius_sg_cleanup", lambda name, meta: [])


@_pytest.fixture(autouse=True)
def _no_real_model_preflight(monkeypatch):
    """The launch's model preflight asks the Hub / Modal / the bucket; unit tests
    never reach them. Tests of the check set --preflight-model and fake the probes."""
    monkeypatch.setenv("YETO_PREFLIGHT_MODEL", "off")


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
    so that no fixture of these tests runs). Then keep only this --shard."""
    if not config.getoption("--run-ray-local"):
        keep, drop = [], []
        for item in items:
            (drop if item.get_closest_marker("ray_local") else keep).append(item)
        if drop:
            config.hook.pytest_deselected(items=drop)
            items[:] = keep
    _select_shard(config, items)


def _select_shard(config, items):
    spec = config.getoption("--shard")
    if not spec:
        return
    index, count = (int(x) for x in spec.split("/"))
    if not 0 <= index < count:
        raise _pytest.UsageError(f"--shard {spec}: need 0 <= I < N")
    # Round-robin over sorted node ids: deterministic across runners, and the
    # slow parametrizations of one file spread over all shards.
    mine = {id(item) for n, item in enumerate(sorted(items, key=lambda i: i.nodeid))
            if n % count == index}
    keep = [item for item in items if id(item) in mine]
    drop = [item for item in items if id(item) not in mine]
    print(f"\nshard {index}/{count}: {len(keep)} of {len(items)} tests")
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


PPU_TEST_IPS = {f"<ppu{i}-eth0-ip>": f"192.0.2.{73 + i}" for i in range(1, 5)}  # TEST-NET-1


@_pytest.fixture(scope="session")
def ppu_plan(tmp_path_factory):
    """examples/stage1/glm53-flash-ppu.yaml + ppu-inventory.yaml with the
    <ppuN-eth0-ip> placeholders filled in with test IPs 192.0.2.74..77."""
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "examples" / "stage1"
    dst = tmp_path_factory.mktemp("stage1-ppu")
    for name in ("glm53-flash-ppu.yaml", "ppu-inventory.yaml"):
        text = (src / name).read_text()
        for placeholder, ip in PPU_TEST_IPS.items():
            text = text.replace(placeholder, ip)
        (dst / name).write_text(text)
    return dst / "glm53-flash-ppu.yaml"


# No yeto-syncer subprocess outlives the test that started it (tests/syncer_reaper.py).
from syncer_reaper import (  # noqa: E402,F401
    pytest_fixture_setup,
    pytest_runtest_teardown,
    pytest_sessionfinish,
    pytest_sessionstart,
)


# Most launch tests build RL argv without --inner-lr; yeto refuses that
# (yeto.launch.rl_args.refuse_unset_inner_lr). Before any test module builds a
# parser (some at import), the CLI default reads as an explicit value here;
# tests/test_rl_inner_lr_guard.py puts the marked default back.
import yeto.cli as _yeto_cli  # noqa: E402

_yeto_cli.DEFAULT_INNER_LR = 3e-4


@_pytest.fixture(autouse=True)
def _no_release_download_spread(monkeypatch):
    """Background release downloads start at once in tests (production spreads them 0..5 s)."""
    from yeto.rl.engine import policy_release as _pr

    monkeypatch.setattr(_pr, "DOWNLOAD_SPREAD_S", 0.0)
