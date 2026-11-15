"""Model adaptation registry: which trainer backend, sampler engine, image pins,
parallel layout and quantization fit a model on given hardware (advisory only).

Off by default. It runs only for ``yeto adapt <model> --hardware ...`` (a dry run that
reads ``config.json`` and prints a recommendation and a run-config draft) or when a
launch passes ``--rl-model-auto`` (prints the recommendation; it never changes an
argument). The ``--rl-model-recipe`` if/elif paths (``run_config.py``) stay the source
of truth; images are referenced by pin constant name (``module:CONST``), never copied.
Design: the hetero-1114 A21 report, section 3. stdlib only (plus the stdlib-only yeto.launch.engine_images, the device-family image defaults).
"""

from .match import facts_from_config, load_config, match_family
from .recommend import recommend
from .schema import Constraints, HardwareFacts, ModelFacts, Recommendation

__all__ = ["Constraints", "HardwareFacts", "ModelFacts", "Recommendation", "facts_from_config",
           "load_config", "match_family", "recommend"]
