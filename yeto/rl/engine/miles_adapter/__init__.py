"""Forwarding package (yeto-framework-decoupling 5.1): moved to ``yeto.rl.adapters.miles``.

Every submodule here replaces itself in ``sys.modules`` with the module of the
same name under ``yeto.rl.adapters.miles``, so ``yeto.rl.engine.miles_adapter.X``
and ``yeto.rl.adapters.miles.X`` are the same object.  Kept until no caller uses
the old path (design D2).
"""

from yeto.rl.adapters.miles import *  # noqa: F401,F403
from yeto.rl.adapters.miles import LoopRunner  # noqa: F401
