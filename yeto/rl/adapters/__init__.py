"""Backend adapters (yeto-framework-decoupling design D2).

Each sub-package translates yeto's neutral core (``yeto.rl.engine``,
``yeto.rl.rewards``, ``yeto.rl.algos``, ``yeto.rl.harness``) into one training
framework.  Adapters never import each other (boundary check 5.6).
"""
