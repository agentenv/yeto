"""Algorithm extensions for the ports path (change ``rl-algorithm-capabilities``).

Each follow-up algorithm change (``rl-algo-mismatch-correction``,
``rl-algo-grpo-knobs``, ``rl-algo-seq-and-adv``, ``rl-algo-loss-variants``)
adds its own module under this package that calls the registration API of
:mod:`yeto.rl.engine.algorithm` (``register_field`` / ``register_mechanism`` /
``register_rejection``) -- neutral fields only; its Miles flag rows and argv
translation go in :mod:`yeto.rl.engine.miles_adapter.algo_flag_rows`
(``register_flag``, decoupling task 4.3) -- and then adds exactly one line to
:data:`EXTENSION_MODULES`. The modules are imported lazily, once, the first
time a spec is parsed or its mechanisms are evaluated.

This file is a shared registration file owned by WP-CAP (alignment.md §7):
follow-up changes submit the one-line addition as a patch.
"""

EXTENSION_MODULES: tuple[str, ...] = (
    # "yeto.rl.algos.mismatch_observe",  # rl-algo-mismatch-correction (example)
    "yeto.rl.algos.grpo_knobs",  # rl-algo-grpo-knobs
    "yeto.rl.algos.seq_adv",  # rl-algo-seq-and-adv
    "yeto.rl.algos.mismatch_correction",  # rl-algo-mismatch-correction
    "yeto.rl.algos.loss_variants",  # rl-algo-loss-variants
    "yeto.rl.algos.critic",  # rl-algo-critic-family
    "yeto.rl.algos.sao",  # rl-algo-critic-family 8.3
)
