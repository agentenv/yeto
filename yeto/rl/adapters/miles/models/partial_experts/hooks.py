"""Patch a Miles module now, or as soon as it is imported.

Model runtimes run from ``sitecustomize`` before Miles is imported, so a
patch cannot simply ``import miles...`` (that would import torch / megatron in
the launcher). :func:`install_or_defer` patches an already imported module in
place and otherwise arms a one-shot :class:`importlib.abc.MetaPathFinder`
that applies the patch right after the module executes.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from collections.abc import Callable
from types import ModuleType

Installer = Callable[[ModuleType], None]


class _Loader(importlib.abc.Loader):
    def __init__(self, wrapped, installer: Installer) -> None:
        self.wrapped = wrapped
        self.installer = installer

    def create_module(self, spec):
        create = getattr(self.wrapped, "create_module", None)
        return None if create is None else create(spec)

    def exec_module(self, module) -> None:
        self.wrapped.exec_module(module)
        self.installer(module)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, fullname: str, installer: Installer) -> None:
        self.fullname = fullname
        self.installer = installer

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.fullname:
            return None
        try:
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _Loader(spec.loader, self.installer)
        return spec


_FINDERS: list[_Finder] = []


def install_or_defer(fullname: str, installer: Installer) -> None:
    """Run ``installer(module)`` on ``fullname`` now if imported, else on import."""

    loaded = sys.modules.get(fullname)
    if loaded is not None:
        installer(loaded)
        return
    finder = _Finder(fullname, installer)
    _FINDERS.append(finder)
    sys.meta_path.insert(0, finder)


def armed_module_names() -> tuple[str, ...]:
    """Modules with a pending import hook (test / diagnostics helper)."""

    return tuple(finder.fullname for finder in _FINDERS if finder in sys.meta_path)


def install_once(module: ModuleType, patch: Callable[[ModuleType], None], *, namespace: str) -> None:
    """Apply ``patch(module)`` once per module; the marker is a module attribute.

    The marker name is ``_<namespace>__<patch name>__installed`` so two model
    runtimes patching the same module never collide.
    """

    marker = f"_{namespace}__{patch.__name__.lstrip('_')}__installed"
    if getattr(module, marker, False):
        return
    patch(module)
    setattr(module, marker, True)
