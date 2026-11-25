"""The cargo toolchain to build the syncer with, when HOME is not the caller's own.

rustup's ``cargo`` proxy finds its toolchains under ``RUSTUP_HOME`` (default
``$HOME/.rustup``) and cargo its registry under ``CARGO_HOME`` (default
``$HOME/.cargo``). With HOME pointed at an empty directory (an isolated HOME,
as goldens and dry-run baselines are recorded under) the proxy on PATH finds no
default toolchain and every build fails.

:func:`resolve` tries, in order: the environment as it is (``CARGO_HOME`` /
``RUSTUP_HOME`` when set, else HOME's own, or any cargo on PATH); then the
caller's own home from the password database. The second sets only the
variables the environment leaves unset. With neither running cargo it says why.

Taking the owner's toolchain means a run under an isolated HOME builds with the
owner's ``CARGO_HOME``: cargo may add to its registry and git caches there (a
cache, nothing a run reads back as state). A cargo found off PATH is appended to
PATH, so no other binary of the owner's cargo bin shadows one already on it.
Each resolve runs ``cargo --version`` once.
"""

from __future__ import annotations

import os
import pwd
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Toolchain:
    env: dict[str, str] = field(default_factory=dict)  # variables to set; empty: the environment runs cargo as is
    cargo: str | None = None  # the cargo that runs; None: none does
    reason: str = ""  # why none does

    @property
    def ok(self) -> bool:
        return self.cargo is not None


def _runs(cargo: str | None, env: Mapping[str, str], run: Callable[..., subprocess.CompletedProcess]) -> bool:
    if cargo is None:
        return False
    try:
        return run([cargo, "--version"], env=dict(env), capture_output=True, text=True, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _owner_home() -> str | None:
    try:
        return pwd.getpwuid(os.getuid()).pw_dir
    except (KeyError, OSError):
        return None


def resolve(environ: Mapping[str, str] | None = None, *, owner_home: str | None = None,
            run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> Toolchain:
    """The toolchain cargo runs with (see the module doc)."""
    env = dict(os.environ if environ is None else environ)
    path = env.get("PATH", "")
    first = shutil.which("cargo", path=path)
    if _runs(first, env, run):
        return Toolchain(cargo=first)
    home = owner_home if owner_home is not None else _owner_home()
    tried = f"cargo on PATH: {first or 'none'}"
    if home and Path(home) != Path(env.get("HOME", "")):
        extra: dict[str, str] = {}
        if "RUSTUP_HOME" not in env and (Path(home) / ".rustup").is_dir():
            extra["RUSTUP_HOME"] = str(Path(home) / ".rustup")
        if "CARGO_HOME" not in env and (Path(home) / ".cargo").is_dir():
            extra["CARGO_HOME"] = str(Path(home) / ".cargo")
        cargo_home = extra.get("CARGO_HOME") or env.get("CARGO_HOME")
        if first is None and cargo_home and (Path(cargo_home) / "bin" / "cargo").exists():
            extra["PATH"] = f"{path}{os.pathsep}{Path(cargo_home) / 'bin'}" if path else str(Path(cargo_home) / "bin")
        cargo = shutil.which("cargo", path=extra.get("PATH", path))
        if extra and _runs(cargo, {**env, **extra}, run):
            return Toolchain(env=extra, cargo=cargo)
        tried += f"; the owner's home {home}: {'no toolchain' if not extra else 'cargo does not run'}"
    return Toolchain(reason=f"no Rust toolchain runs cargo ({tried}; HOME={env.get('HOME', '')!r}, "
                            f"CARGO_HOME={env.get('CARGO_HOME')!r}, RUSTUP_HOME={env.get('RUSTUP_HOME')!r})")
