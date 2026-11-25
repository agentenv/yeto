"""Moved to :mod:`yeto.launch.pool.sky_provider` (unified-sampler-pool 7.3).

This old name forwards: it is the same module object (``sys.modules``), so imports,
``provider.factory`` strings such as ``"yeto.launch.nc.sky_provider:..."`` and monkeypatch
targets under the old name keep working. New code imports ``yeto.launch.pool.sky_provider``.
"""

import sys
import warnings

from yeto.launch.pool import sky_provider as _moved

warnings.warn("yeto.launch.nc.sky_provider moved to yeto.launch.pool.sky_provider; the old name forwards",
              DeprecationWarning, stacklevel=2)
sys.modules[__name__] = _moved
