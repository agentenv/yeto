"""Moved to :mod:`yeto.launch.pool.daytona_provider` (unified-sampler-pool 7.3).

This old name forwards: it is the same module object (``sys.modules``), so imports,
``provider.factory`` strings such as ``"yeto.launch.nc.daytona_provider:..."`` and monkeypatch
targets under the old name keep working. New code imports ``yeto.launch.pool.daytona_provider``.
"""

import sys
import warnings

from yeto.launch.pool import daytona_provider as _moved

warnings.warn("yeto.launch.nc.daytona_provider moved to yeto.launch.pool.daytona_provider; the old name forwards",
              DeprecationWarning, stacklevel=2)
sys.modules[__name__] = _moved
