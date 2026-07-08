"""NAVA as a diffusion component, not a top-level Yeto task."""

from .adapter import NavaComponent
from ..registry import register_component

register_component(NavaComponent(), default=True)

__all__ = ["NavaComponent"]
