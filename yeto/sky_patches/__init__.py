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

from . import nebius, verda

# target module -> patch function(module) -> bool (applied?)
_PATCHES = {verda.TARGET_MODULE: verda.apply, nebius.TARGET_MODULE: nebius.apply}
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
    installed yeto first, else `repo_dir`; an older installed yeto without
    sky_patches gets `repo_dir/yeto` appended to its package path) and the patch applied. Other
    Python processes of the environment never import yeto, and a missing
    or foreign yeto is skipped silently."""
    code = (
        "import sys, os\n"
        f"_R = {repo_dir!r}\n"
        "class _YetoLazy:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name not in ('sky.provision.verda.instance', 'sky.provision.nebius.utils'):\n"
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
        "                _y = sys.modules.get('yeto')\n"
        "                if _y is not None and hasattr(_y, '__path__'):\n"
        "                    _y.__path__.append(os.path.join(_R, 'yeto'))\n"
        "                import yeto.sky_patches as y\n"
        "            y.install()\n"
        "            return y._FINDER.find_spec(name, path, target)\n"
        "        except Exception as e:\n"
        "            sys.stderr.write('[yeto] sky patches not applied: %s\\n' % e)\n"
        "            return None\n"
        "sys.meta_path.insert(0, _YetoLazy())\n"
    )
    # site.addpackage exec()s .pth lines inside a function: without an
    # explicit globals dict, _R lands in its locals and find_spec (looking it
    # up as a global) fails with NameError, leaving sky unpatched.
    return f"{_PTH_TAG}{repo_dir}\nimport sys; exec({code!r}, {{'__name__': '_yeto_pth'}})\n"


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

    sky (0.13 and later) provisions inside its API server process (started from this
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


# -- the sky API server this process talks to ------------------------------------

SERVER_MISMATCH_ENV = "YETO_SKY_SERVER_MISMATCH"  # "allow": proceed on a mismatched local server
SERVER_CHECK_ENV = "YETO_SKY_SERVER_CHECK"  # "off": no check (the unit tests)


def api_server_version_problem(server_common=None, environ=None) -> tuple[str, bool] | None:
    """(message, refuse) when the running sky API server is another sky version than
    this Python's, else None. sky itself accepts any server from 0.11 on (its API
    version negotiation), so a mismatch is silent there: requests would run on the
    other version's server, where yeto's patches (and sky's own behaviour) are not
    the verified ones, and this process's direct sky state reads would migrate the
    shared state database to its own schema under the older server. ``refuse``: the
    server is local (a remote one is sky's supported setup: a warning only). No
    running server, or one not answering: None (sky starts one from this Python)."""
    import os

    environ = os.environ if environ is None else environ
    try:
        import sky

        if server_common is None:
            from sky.server import common as server_common
        endpoint = server_common.get_server_url()
        info = server_common.get_api_server_status(endpoint)
    except Exception:  # noqa: BLE001 - sky absent or API drift: sky reports its own errors
        return None
    if getattr(info.status, "value", info.status) != "healthy" or not info.version:
        return None
    mine = getattr(sky, "__version__", None)
    if info.version == mine:
        return None
    local = bool(server_common.is_api_server_local(endpoint))
    msg = (f"this Python runs SkyPilot {mine}, but the sky API server at {endpoint} runs {info.version}. "
           "sky does not refuse the mismatch, so every launch, status and down would run on that server "
           "(yeto's sky patches are verified per version)")
    if not local:
        return msg + "; use a server of the same version for verified behaviour", False
    msg += (", and this process would migrate the local sky state database to its own schema under it. "
            f"Either restart the local server from this environment (sky api stop; sky api start), or keep it "
            f"and start a separate {mine} server with its own state: SKY_RUNTIME_DIR=<dir> sky api start "
            f"--port <port>, then run yeto with SKY_RUNTIME_DIR=<dir> "
            f"SKYPILOT_API_SERVER_ENDPOINT=http://127.0.0.1:<port>")
    if environ.get(SERVER_MISMATCH_ENV) == "allow":
        return msg + f" ({SERVER_MISMATCH_ENV}=allow: proceeding)", False
    return msg + f" (or set {SERVER_MISMATCH_ENV}=allow to proceed anyway)", True


def require_matching_api_server(out=None) -> None:
    """Refuse (ValueError) on a mismatched local sky API server; warn on a remote one.
    Call before the first sky request of a launch that uses sky."""
    import os

    if os.environ.get(SERVER_CHECK_ENV) == "off":
        return
    problem = api_server_version_problem()
    if problem is None:
        return
    msg, refuse = problem
    if refuse:
        raise ValueError(msg)
    print(f"[yeto] WARNING: {msg}", file=out or sys.stderr, flush=True)


# -- the patch state of the sky API server (where sky provisions), not of this Python
SERVER_PATCH_ENV = "YETO_SKY_SERVER_PATCH"  # "warn": proceed (a warning) on an unpatched server
DEFAULT_SERVER_PORT = 46580
_PY_OPTIONS_WITH_VALUE = ("-X", "-W", "-Q")


class ServerUnreadable(RuntimeError):
    """The local sky API server cannot be told apart, or read, safely."""


def _server_addr(endpoint: str) -> tuple[str, int] | None:
    from urllib.parse import urlparse

    try:
        url = urlparse(endpoint)
        return url.hostname or "127.0.0.1", url.port or DEFAULT_SERVER_PORT
    except ValueError:
        return None


def _server_port(endpoint: str) -> int | None:
    addr = _server_addr(endpoint)
    return addr[1] if addr else None


def _cmdline_port(cmdline: list[str]) -> int:
    for i, arg in enumerate(cmdline):
        if arg == "--port" and i + 1 < len(cmdline) and cmdline[i + 1].isdigit():
            return int(cmdline[i + 1])
        if arg.startswith("--port=") and arg[7:].isdigit():
            return int(arg[7:])
    return DEFAULT_SERVER_PORT


def _server_argv(cmdline: list[str] | None) -> list[str] | None:
    """The argv of ``<python> [interpreter options] -m sky.server.server ...``, else None
    (a shell or anything else that only mentions it). Some processes report their whole
    command line as one argument: it is split again."""
    import shlex

    joined = " ".join(cmdline or [])
    if "sky.server.server" not in joined:
        return None
    try:
        argv = shlex.split(joined)
    except ValueError:
        return None
    i = 1
    while i < len(argv) and argv[i].startswith("-") and argv[i] != "-m":
        i += 2 if argv[i] in _PY_OPTIONS_WITH_VALUE else 1
    return argv if argv[i:i + 2] == ["-m", "sky.server.server"] else None


def _listens(proc, port: int) -> bool:
    import psutil

    try:
        conns = getattr(proc, "net_connections", None) or proc.connections
        return any(c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port for c in conns(kind="inet"))
    except Exception:  # noqa: BLE001 - unreadable: not shown to listen
        return False


def _port_in_use(host: str, port: int) -> bool:
    import socket

    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def local_server_process(port: int, host: str = "127.0.0.1"):
    """(python, start time, cwd, PYTHONPATH) of this user's sky API server process
    listening on ``port``, or None (nothing listens there). Only a process of this user
    (real uid) that is seen listening on the port counts. Two such processes, a port in
    use by none of them (another user's server, a command line that cannot be read), or a
    cwd / environment that cannot be read: ServerUnreadable. Of its environment only
    ``PYTHONPATH`` is read (it decides which yeto its patch hook imports)."""
    import os

    import psutil

    uid = os.getuid()
    found = []
    for proc in psutil.process_iter(attrs=["pid", "cmdline", "create_time", "uids"]):
        argv = _server_argv(proc.info.get("cmdline"))
        if argv is None or _cmdline_port(argv) != port:
            continue
        uids = proc.info.get("uids")
        if uids is None or uids.real != uid:  # another user's process: never read, never run
            continue
        if _listens(proc, port):
            found.append((proc, argv[0], float(proc.info["create_time"])))
    if len(found) > 1:
        raise ServerUnreadable(f"{len(found)} sky API server processes of this user listen on port {port}")
    if found:
        proc, python, started = found[0]
        try:  # python -m puts its cwd first on sys.path, then PYTHONPATH: both decide which yeto it imports
            cwd = proc.cwd()
            pythonpath = proc.environ().get("PYTHONPATH", "")
        except Exception as exc:  # noqa: BLE001
            raise ServerUnreadable(f"the cwd / PYTHONPATH of the sky API server process cannot be read "
                                   f"({type(exc).__name__})") from None
        return python, started, cwd, pythonpath
    if _port_in_use(host, port):
        raise ServerUnreadable(f"port {port} is in use, but not by a sky API server process of this user "
                               "whose command line can be read")
    return None


def python_site_packages(python: str) -> str | None:
    """The purelib site-packages of interpreter ``python``, found from its files (it is
    never run): this Python's own, or a virtualenv's (``<venv>/bin/python``, one
    ``lib/python3.*/site-packages``). None: not found that way."""
    import glob
    import os
    import sysconfig

    prefix = os.path.dirname(os.path.dirname(python))
    if os.path.isfile(os.path.join(prefix, "pyvenv.cfg")):
        sites = glob.glob(os.path.join(prefix, "lib", "python3.*", "site-packages"))
        return sites[0] if len(sites) == 1 else None
    if os.path.abspath(python) == os.path.abspath(sys.executable):
        return sysconfig.get_paths()["purelib"]
    return None


def _patch_source(site: str, repo: str, front: tuple[str, ...] = ()) -> str:
    """The ``yeto/sky_patches`` directory the hook in ``site`` imports. The hook imports
    ``yeto.sky_patches`` as the server's ``sys.path`` finds it first (``front``: its cwd,
    then PYTHONPATH; then site-packages and the paths its ``.pth`` files add); the first
    regular ``yeto`` package found decides: with ``sky_patches`` it is the source, without,
    the hook falls back to its repo. A ``yeto`` directory without ``__init__.py`` (a
    checkout's root, seen from its parent) does not decide: Python prefers a regular
    package anywhere on the path. Only such directories: the first of them with a
    ``sky_patches`` package, else (the hook's ImportError fallback) its repo. No yeto
    anywhere: the hook's repo."""
    import os

    portions: list[str] = []

    def yeto_in(root: str) -> str | None:
        """The patch source of a regular yeto package at ``root``, "" when it has none,
        None when there is no regular yeto package there."""
        pkg = os.path.join(root, "yeto")
        if not os.path.isdir(pkg):
            return None
        if not os.path.isfile(os.path.join(pkg, "__init__.py")):
            portions.append(pkg)  # a namespace portion: a regular package later on the path wins
            return None
        patches = os.path.join(pkg, "sky_patches")
        return patches if os.path.isfile(os.path.join(patches, "__init__.py")) else ""

    roots = [*front, site]
    for name in sorted(os.listdir(site)):
        if not name.endswith(".pth") or name == PTH_NAME:
            continue
        with open(os.path.join(site, name), encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line.startswith("import") and "yeto" in line:
                    raise ServerUnreadable(f"{name} in the server's site-packages imports yeto: which yeto "
                                           "applies the patch cannot be read without running it")
                if line and not line.startswith(("#", "import")):
                    roots.append(line if os.path.isabs(line) else os.path.join(site, line))
    for root in roots:
        found = yeto_in(root)
        if found:
            return found
        if found == "":  # a yeto without sky_patches comes first: the hook falls back to its repo
            break
    else:
        # only namespace portions: ``import yeto.sky_patches`` finds it in the first portion
        # that has it; else the ImportError sends the hook to its repo (appended to yeto's path)
        for pkg in portions:
            patches = os.path.join(pkg, "sky_patches")
            if os.path.isfile(os.path.join(patches, "__init__.py")):
                return patches
    found = os.path.join(repo, "yeto", "sky_patches")
    if not os.path.isfile(os.path.join(found, "__init__.py")):
        raise ServerUnreadable(f"the patch hook points at {repo}, which has no yeto/sky_patches")
    return found


def _literal(node):
    import ast

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("frozenset", "set") \
            and len(node.args) == 1:
        return set(ast.literal_eval(node.args[0]))
    return ast.literal_eval(node)


def _patch_spec(src: str, target: str) -> dict | None:
    """``VERIFIED_SKY_VERSIONS`` and ``REQUIRED_ATTRS`` of the patch of ``target`` in
    ``src``, read from its source (never imported)."""
    import ast
    import glob
    import os

    for path in sorted(glob.glob(os.path.join(src, "*.py"))):
        if os.path.basename(path) in ("__init__.py", "__main__.py"):
            continue
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read(), path)
        values = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                    and node.targets[0].id in ("TARGET_MODULE", "VERIFIED_SKY_VERSIONS", "REQUIRED_ATTRS"):
                try:
                    values[node.targets[0].id] = _literal(node.value)
                except ValueError:
                    pass
        if values.get("TARGET_MODULE") == target:
            return {"verified": set(values.get("VERIFIED_SKY_VERSIONS") or ()),
                    "required": tuple(values.get("REQUIRED_ATTRS") or ())}
    return None


def _dist_version(site: str) -> str | None:
    import glob
    import os

    found = []
    for name in ("skypilot", "skypilot_nightly"):
        for d in glob.glob(os.path.join(site, f"{name}-*.dist-info")):
            with open(os.path.join(d, "METADATA"), encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("Version:"):
                        found.append((os.path.basename(d), line.split(":", 1)[1].strip()))
                        break
    if len(found) > 1:
        raise ServerUnreadable(f"several sky installs in the server's site-packages: {[d for d, _ in found]}")
    return found[0][1] if found else None


def _top_level_names(path: str) -> set[str]:
    import ast

    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), path)
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _front_paths(cwd: str, pythonpath: str) -> tuple[str, ...]:
    """What ``python -m`` puts before site-packages on ``sys.path``: its cwd, then PYTHONPATH."""
    import os

    return (cwd, *(os.path.join(cwd, p) for p in pythonpath.split(os.pathsep) if p))


def _patch_state_in(python: str, target: str, started: float | None,
                    front: tuple[str, ...] = ()) -> tuple[bool, str]:
    """Would the patch of ``target`` be in effect in a process of ``python`` started at
    ``started`` (``front``: its sys.path before site-packages)? Read from that
    environment's files only: nothing of it is run."""
    import glob
    import os

    where = f"the sky API server's environment ({python})"
    if not os.path.isabs(python):
        return False, f"the sky API server runs {python!r}, not an absolute Python path"
    site = python_site_packages(python)
    if site is None:
        return False, f"the site-packages of {python} cannot be located from its files"
    hook = os.path.join(site, PTH_NAME)
    repo = pth_repo(hook)
    if not repo:
        return False, f"no yeto sky patch hook in {where} ({hook})"
    with open(hook, encoding="utf-8") as f:
        if repr(target) not in f.read():
            return False, f"the patch hook {hook} predates the patch of {target} (install it again)"
    src = _patch_source(site, repo, front)
    spec = _patch_spec(src, target)
    if spec is None:
        return False, f"the yeto at {src} has no patch of {target}"
    version = _dist_version(site)
    if version not in spec["verified"]:
        return False, f"not applied in {where}: sky {version} not in verified {sorted(spec['verified'])}"
    module_file = os.path.join(site, *target.split(".")) + ".py"
    missing = [a for a in spec["required"] if a not in _top_level_names(module_file)]
    if missing:
        return False, f"not applied in {where}: {target} lacks {missing}"
    if started is not None:
        for path in [hook, *sorted(glob.glob(os.path.join(src, "*.py")))]:
            if os.path.getmtime(path) > started:
                return False, f"{path} changed after the sky API server started (restart it)"
    return True, f"applied in {where}"


def server_patch_state(target: str, *, server_common=None) -> tuple[bool, str]:
    """(applied, why) of yeto's patch of sky module ``target`` in the process where sky
    provisions: the sky API server, not this Python (it may run from another
    environment). Only a local server of this user can be inspected, and only from its
    files: the hook in its site-packages, the yeto that hook imports, its sky version and
    module (nothing of it is run). A hook or patch source changed after the server
    started is not in effect. No server running: the one sky would start, from this
    Python, cwd and PYTHONPATH. Anything unreadable: not applied. It is an approximation
    of the import-time checks (the target module's top-level names, not its objects),
    not a proof."""
    try:
        if server_common is None:
            from sky.server import common as server_common
        endpoint = server_common.get_server_url()
        if not server_common.is_api_server_local(endpoint):
            return False, f"the sky API server at {endpoint} is remote (its patches cannot be read from here)"
        addr = _server_addr(endpoint)
        if addr is None:
            return False, f"the sky API server endpoint {endpoint!r} cannot be read"
        found = local_server_process(addr[1], addr[0])
    except Exception as exc:  # noqa: BLE001
        return False, f"the sky API server could not be inspected ({exc})"[:300]
    import os

    # no server: sky starts one from this Python, inheriting this cwd and environment
    python, started, cwd, pythonpath = found or (sys.executable, None, os.getcwd(), os.environ.get("PYTHONPATH", ""))
    try:
        return _patch_state_in(python, target, started, _front_paths(cwd, pythonpath))
    except Exception as exc:  # noqa: BLE001
        return False, f"the sky API server's environment could not be read: {exc} ({python})"[:600]


def server_python(server_common=None) -> str | None:
    """The Python of the local sky API server (this one when none runs); None: unknown."""
    try:
        if server_common is None:
            from sky.server import common as server_common
        addr = _server_addr(server_common.get_server_url())
        found = local_server_process(addr[1], addr[0]) if addr else None
        return found[0] if found else sys.executable
    except Exception:  # noqa: BLE001 - the hint names a placeholder then
        return None


def server_patch_install_hint(python: str | None = None) -> str:
    return (f"install the hook into the sky API server's Python, from a yeto checkout that stays on main: "
            f"cd <yeto checkout> && {python or '<sky API server python>'} -m yeto.sky_patches install, "
            f"then restart that server (sky api stop; sky api start)")


def require_patched_server(target: str, what: str, *, out=None, environ=None) -> bool:
    """Refuse (ValueError) to order ``what`` (Verda) when the sky API server's patch of
    ``target`` is not in effect; ``YETO_SKY_SERVER_PATCH=warn`` warns instead. Returns
    whether the patch is in effect."""
    import os

    environ = os.environ if environ is None else environ
    ok, why = server_patch_state(target)
    if ok:
        return True
    msg = f"{what}: yeto's patch of {target} is not in effect in the sky API server: {why}. " + \
        server_patch_install_hint(server_python())
    if environ.get(SERVER_PATCH_ENV) == "warn":
        print(f"[yeto] WARNING: {msg} ({SERVER_PATCH_ENV}=warn: proceeding unpatched; a failed Verda launch can "
              "delete running instances of the same name)", file=out or sys.stderr, flush=True)
        return False
    raise ValueError(msg + f" (or set {SERVER_PATCH_ENV}=warn to proceed unpatched)")
