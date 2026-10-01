"""Algorithm extensions for the ports path (change ``rl-algorithm-capabilities``).

Each follow-up algorithm change (``rl-algo-mismatch-correction``,
``rl-algo-grpo-knobs``, ``rl-algo-seq-and-adv``, ``rl-algo-loss-variants``)
adds its own module under this package that calls the registration API of
:mod:`yeto.rl.engine.algorithm` (``register_field`` / ``register_mechanism`` /
``register_rejection``) and :mod:`yeto.rl.engine.miles_adapter.algorithm_flags`
(``register_flag``), and then adds exactly one line to
:data:`EXTENSION_MODULES`. The modules are imported lazily, once, the first
time a spec is parsed or its mechanisms are evaluated.

This file is a shared registration file owned by WP-CAP (alignment.md §7):
follow-up changes submit the one-line addition as a patch.
"""

EXTENSION_MODULES: tuple[str, ...] = (
    # "yeto.rl.algos.mismatch_observe",  # rl-algo-mismatch-correction (example)
    "yeto.rl.algos.grpo_knobs",  # rl-algo-grpo-knobs
)
