"""Adapter from yeto's role-based engine ports to upstream Miles (design D1).

Every call into upstream Miles is concentrated in this package so that an
upstream pin bump only touches it. Importing the package is cheap: miles, ray
and torch are imported lazily inside the functions that need them.

Modules:

* ``config``            RLRunConfig + AlgorithmSpec -> upstream Miles argv (3.1)
* ``rollout``           RolloutPool over InferenceController/RolloutExecutor (3.2)
* ``rollout_meta_hook`` in-rollout-process metadata extraction (3.2, D3)
* ``trainer``           TrainerGroup over the single-cell actor TrainGroup (3.3)
* ``state_plugin``      code run *inside* each Megatron rank via run_plugin (3.4)
* ``state``             PolicyState over ``run_plugin`` (3.4, D4)
* ``publish``           Publisher over upstream ``update_weights`` (3.5)
* ``placement``         read-only placement description + rewrite detection (3.6)
"""

from __future__ import annotations

# The runner is engine-neutral (yeto/rl/engine/loop_runner.py); this name stays
# importable for the Miles ports.
from yeto.rl.engine.loop_runner import LoopRunner

__all__ = ["LoopRunner"]
