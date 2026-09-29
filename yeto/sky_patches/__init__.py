"""Runtime patches yeto applies to SkyPilot (never a fork, never upstream).

`install()` registers an import hook so each patch is applied the moment
its target sky module is imported — in yeto's own process and, through
the `.pth` entry `pth_line()` writes, in every Python process of the
environment (notably sky's API server, which is where provisioning runs).
Each patch carries its own version guard; `status()` reports what applied.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
import threading

from . import verda

# target module -> patch function(module) -> bool (applied?)
_PATCHES = {verda.TARGET_MODULE: verda.apply}
_STATUS: dict[str, str] = {}
_LOCK = threading.Lock()
PTH_NAME = "yeto_sky_patches.pth"


def status() -> dict[str, str]:
    """{target module: 'applied' | 'skipped: <why>' | 'pending'}."""
    out = {name: "pending" for name in _PATCHES}
    out.update(_STATUS)
    return out


def _apply(name: str, module) -> None:
    with _LOCK:
        if name in _STATUS:
            return
        try:
            ok, why = _PATCHES[name](module)
        except Exception as exc:  # noqa: BLE001 - a broken patch must not break sky
            ok, why = False, f"error: {exc}"
        _STATUS[name] = "applied" if ok else f"skipped: {why}"


class _Loader(importlib.abc.Loader):
    def __init__(self, inner, name):
        self._inner, self._name = inner, name

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        _apply(self._name, module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in _PATCHES or fullname in _STATUS:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                if spec.loader is not None and hasattr(spec.loader, "exec_module"):
                    spec.loader = _Loader(spec.loader, fullname)
                return spec
        return None


_FINDER = _Finder()


def install() -> dict[str, str]:
    """Idempotent. Patches already-imported targets now, the rest on import."""
    if _FINDER not in sys.meta_path:
        sys.meta_path.insert(0, _FINDER)
    for name in _PATCHES:
        mod = sys.modules.get(name)
        if mod is not None:
            _apply(name, mod)
    return status()


def pth_line(repo_dir: str) -> str:
    """One `.pth` line: put `repo_dir` on sys.path and install the hook.
    Wrapped so a missing/broken yeto never stops the interpreter."""
    code = (
        "import sys\n"
        "try:\n"
        f"    sys.path.append({repo_dir!r})\n"
        "    import yeto.sky_patches as _y\n"
        "    _y.install()\n"
        "except Exception as _e:\n"
        "    sys.stderr.write('[yeto] sky patches not installed: %s\\n' % _e)\n"
    )
    return f"import sys; exec({code!r})\n"


def head_pth_command(repo_dir: str = "~/sky_workdir") -> str:
    """Shell step: write the `.pth` into the head Python's site-packages."""
    return (
        "python3 - <<'YETO_PTH'\n"
        "import os, site, sysconfig\n"
        "from yeto.sky_patches import PTH_NAME, pth_line\n"
        "d = sysconfig.get_paths()['purelib']\n"
        f"p = os.path.join(d, PTH_NAME); open(p, 'w').write(pth_line(os.path.expanduser({repo_dir!r})))\n"
        "print('[yeto] sky patch hook:', p)\n"
        "YETO_PTH"
    )


def ensure_local_pth(repo_dir: str, site_dir: str | None = None) -> tuple[str, bool]:
    """Make this machine's sky API server load the patches too.

    sky 0.13 provisions inside its API server process (started from this
    same Python), not in the process that called sky.launch — so the hook
    must be in this environment's site-packages as a `.pth`. Returns
    (path, newly_written); a server already running from before the
    write keeps the old code until restarted (`sky api stop`)."""
    import os
    import sysconfig

    site_dir = site_dir or sysconfig.get_paths()["purelib"]
    path = os.path.join(site_dir, PTH_NAME)
    line = pth_line(repo_dir)
    try:
        with open(path, encoding="utf-8") as f:
            if f.read() == line:
                return path, False
    except OSError:
        pass
    with open(path, "w", encoding="utf-8") as f:
        f.write(line)
    return path, True
