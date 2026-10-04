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


_PTH_TAG = "# yeto-sky-patches repo="


def pth_line(repo_dir: str) -> str:
    """`.pth` content: a lazy hook that does nothing until some process
    imports sky's Verda provisioner; only then is yeto imported (an
    installed yeto first, else `repo_dir`) and the patch applied. Other
    Python processes of the environment never import yeto, and a missing
    or foreign yeto is skipped silently."""
    code = (
        "import sys, os\n"
        f"_R = {repo_dir!r}\n"
        "class _YetoLazy:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name != 'sky.provision.verda.instance':\n"
        "            return None\n"
        "        try:\n"
        "            sys.meta_path.remove(self)\n"
        "        except ValueError:\n"
        "            pass\n"
        "        try:\n"
        "            try:\n"
        "                import yeto.sky_patches as y\n"
        "            except ImportError:\n"
        "                if not os.path.isfile(os.path.join(_R, 'yeto', 'sky_patches', '__init__.py')):\n"
        "                    return None\n"
        "                sys.path.append(_R)\n"
        "                import yeto.sky_patches as y\n"
        "            y.install()\n"
        "            return y._FINDER.find_spec(name, path, target)\n"
        "        except Exception as e:\n"
        "            sys.stderr.write('[yeto] sky patches not applied: %s\\n' % e)\n"
        "            return None\n"
        "sys.meta_path.insert(0, _YetoLazy())\n"
    )
    return f"{_PTH_TAG}{repo_dir}\nimport sys; exec({code!r})\n"


def pth_repo(path: str) -> str | None:
    """The repo a yeto `.pth` points at (None if not ours / unreadable)."""
    try:
        with open(path, encoding="utf-8") as f:
            first = f.readline().rstrip("\n")
    except OSError:
        return None
    return first[len(_PTH_TAG):] if first.startswith(_PTH_TAG) else None


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


def _site_dir(site_dir: str | None) -> str:
    import sysconfig

    return site_dir or sysconfig.get_paths()["purelib"]


def ensure_local_pth(repo_dir: str, site_dir: str | None = None) -> tuple[str, bool]:
    """Make this machine's sky API server load the patches too.

    sky 0.13 provisions inside its API server process (started from this
    same Python), not in the process that called sky.launch — so the hook
    must be in this environment's site-packages as a `.pth`. Returns
    (path, newly_written). A newly written hook is not in effect for a
    server that was already running; the launcher removes the file it
    wrote at the end of the run (`remove_local_pth`). A hook of another
    worktree is replaced with a warning (only one repo can be hooked)."""
    import os

    path = os.path.join(_site_dir(site_dir), PTH_NAME)
    line = pth_line(repo_dir)
    try:
        with open(path, encoding="utf-8") as f:
            if f.read() == line:
                return path, False
    except OSError:
        pass
    other = pth_repo(path)
    if other and other != repo_dir:
        sys.stderr.write(
            f"[yeto] WARNING: {path} pointed at another worktree ({other}); now {repo_dir}. "
            "Runs from that worktree load this worktree's sky patches.\n"
        )
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(line)
    os.replace(tmp, path)
    return path, True


def remove_local_pth(path: str, repo_dir: str | None = None) -> bool:
    """Remove a hook this process wrote — only if it still is ours."""
    import os

    from yeto import launcher  # noqa: F401 - REPO_ROOT default below

    want = pth_line(repo_dir or str(launcher.REPO_ROOT))
    try:
        with open(path, encoding="utf-8") as f:
            if f.read() != want:
                return False
        os.remove(path)
        return True
    except OSError:
        return False


def uninstall(site_dir: str | None = None) -> str | None:
    """Remove the yeto hook from this environment, whoever wrote it."""
    import os

    path = os.path.join(_site_dir(site_dir), PTH_NAME)
    if pth_repo(path) is None:
        return None
    os.remove(path)
    return path
