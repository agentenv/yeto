"""Keep the test session away from every sky API server of this machine.

A sky client that finds no healthy API server at a local endpoint starts one itself
(``sky.server.common.check_server_healthy_or_start_fn`` -> ``_start_api_server``): a
detached ``python -m sky.server.server`` on the default port that outlives the test
session, with the test's cwd and ``HOME``. A test that reaches the real ``sky`` CLI or
SDK (a teardown that ran ``sky logs`` because a helper cleared the test defaults) then
leaves a server of the test's Python and temporary home on the port the developer's own
server uses, and any later sky command on this machine talks to it.

Three layers, all installed by conftest.py before any ``yeto`` or ``sky`` module is
imported:

* :func:`isolate`: ``SKYPILOT_API_SERVER_ENDPOINT`` points at a port nothing listens on
  (a connection is refused at once, and it is not a local server URL, so sky never
  starts a server for it) and ``SKYPILOT_DISABLE_LOCAL_API_SERVER=1`` (sky's own switch
  that refuses every local start);
* :func:`install`: in this process, sky's auto-start functions raise
  :class:`SkyServerStartRefused`, and sky's endpoint resolves to the refusing one when a
  test removed the variable (instead of the local server's default port);
* every child process (``subprocess.Popen``, so also ``subprocess.run`` and asyncio's
  subprocesses) gets the two variables, also when the test or the code under test
  removed them from ``os.environ`` or passed an ``env`` without them.

Tests of code that talks to sky use a fake ``sky`` module or a fake ``server_common``.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

# Port 1: no server listens there, so a request is refused at once. Not the local server
# port (46580, SKYPILOT_API_SERVER_LOCAL_PORT), so sky treats it as a remote endpoint and
# reports a connection error instead of starting a server.
ENDPOINT = "http://127.0.0.1:1"
ENV = {
    "SKYPILOT_API_SERVER_ENDPOINT": ENDPOINT,
    "SKYPILOT_DISABLE_LOCAL_API_SERVER": "1",
}
COMMON = "sky.server.common"
GUARDED = ("check_server_healthy_or_start_fn", "_start_api_server")
HINT = ("a test tried to reach or start a sky API server (sky.server.common.{name}); unit tests "
        "never start one: it would outlive the session on this machine's sky port. Use a fake sky "
        "module or a fake server_common (tests/sky_server_guard.py)")


class SkyServerStartRefused(RuntimeError):
    """A test reached sky's API server auto-start."""


def isolate() -> None:
    """Point this process (and every child that inherits its environment) away from
    any sky API server of this machine."""
    os.environ.update(ENV)


def child_env(env):
    """``env`` for a child process with the guard variables in it (None: this process's
    environment)."""
    base = os.environ if env is None else env
    if all(base.get(name) == value for name, value in ENV.items()):
        return env
    return {**base, **ENV}


def _refusing(name: str):
    def refuse(*_args, **_kwargs):
        raise SkyServerStartRefused(HINT.format(name=name))

    refuse.yeto_sky_server_guard = True
    return refuse


def _server_url(host=None):  # noqa: ARG001 - sky's signature; a test never addresses a local server
    """sky's endpoint in a test: the variable a test set, else the refusing endpoint (never
    the local server's default port, nor a ~/.sky config)."""
    return (os.environ.get("SKYPILOT_API_SERVER_ENDPOINT") or ENDPOINT).rstrip("/")


_server_url.yeto_sky_server_guard = True
_server_url.cache_clear = lambda: None  # sky clears the cache of the function it replaces


def patch_common(module) -> None:
    """Make sky's auto-start functions in ``module`` (sky.server.common) raise and its
    endpoint resolve as :func:`_server_url`."""
    for name in GUARDED:
        current = getattr(module, name, None)
        if current is not None and not getattr(current, "yeto_sky_server_guard", False):
            setattr(module, name, _refusing(name))
    if getattr(module, "get_server_url", None) is not None:
        module.get_server_url = _server_url


def is_patched(module) -> bool:
    return all(getattr(getattr(module, name, None), "yeto_sky_server_guard", True)
               for name in (*GUARDED, "get_server_url"))


class _PatchOnImport:
    """Meta-path finder: patch sky.server.common right after it is imported. The guard
    never imports sky itself, so tests that assert "sky" is not imported still hold."""

    def find_spec(self, name, path=None, target=None):
        if name != COMMON:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        real_exec = spec.loader.exec_module

        class _Loader:
            def create_module(self, spec):
                return None

            def exec_module(self, module):
                real_exec(module)
                patch_common(module)

        spec.loader = _Loader()
        return spec


def _guard_popen() -> None:
    real_init = subprocess.Popen.__init__
    if getattr(real_init, "yeto_sky_server_guard", False):
        return

    def __init__(self, args, *a, **kw):
        # env is Popen's 11th positional parameter (the 10th after args); by keyword otherwise
        if len(a) >= 10:
            a = (*a[:9], child_env(a[9]), *a[10:])
        else:
            kw["env"] = child_env(kw.get("env"))
        real_init(self, args, *a, **kw)

    __init__.yeto_sky_server_guard = True
    subprocess.Popen.__init__ = __init__


def install() -> None:
    """In-process and child-process layers (idempotent)."""
    if COMMON in sys.modules:
        patch_common(sys.modules[COMMON])
    elif not any(isinstance(f, _PatchOnImport) for f in sys.meta_path):
        sys.meta_path.insert(0, _PatchOnImport())
    _guard_popen()
