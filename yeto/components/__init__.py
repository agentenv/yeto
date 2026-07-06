"""Trainable component registry for generic task backends."""

from .base import DiffusionComponent
from .registry import (
    all_components,
    component_names,
    default_component,
    get_component,
    register_component,
)

# Built-in lightweight component registrations.
from . import nava as _nava  # noqa: E402,F401

__all__ = [
    "DiffusionComponent",
    "all_components",
    "component_names",
    "default_component",
    "get_component",
    "register_component",
]
