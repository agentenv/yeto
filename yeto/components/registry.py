"""Registry for diffusion components."""

from __future__ import annotations

from typing import Iterable

from .base import DiffusionComponent

_COMPONENTS: dict[str, DiffusionComponent] = {}
_DEFAULT: list[str] = []


def register_component(component: DiffusionComponent, *, default: bool = False) -> DiffusionComponent:
    if not component.name:
        raise ValueError("component must set a non-empty name")
    _COMPONENTS[component.name] = component
    if default or not _DEFAULT:
        _DEFAULT[:] = [component.name]
    return component


def get_component(name: str | None) -> DiffusionComponent:
    key = name or default_component()
    try:
        return _COMPONENTS[key]
    except KeyError:
        raise ValueError(
            f"unknown diffusion component {key!r}; known: {', '.join(component_names())}"
        ) from None


def component_names() -> list[str]:
    return list(_COMPONENTS)


def all_components() -> Iterable[DiffusionComponent]:
    return list(_COMPONENTS.values())


def default_component() -> str:
    return _DEFAULT[0] if _DEFAULT else ""
