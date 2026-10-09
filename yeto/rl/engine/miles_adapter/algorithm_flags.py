"""Forwarding module (yeto-framework-decoupling 5.1): moved to ``yeto.rl.adapters.miles.algorithm_flags``.

The old import path resolves to the very same module object; delete once no
caller uses the old path (design D2).
"""

import sys as _sys

from yeto.rl.adapters.miles import algorithm_flags as _module

_sys.modules[__name__] = _module
