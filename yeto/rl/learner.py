"""Command-line entry only: ``python -m yeto.rl.learner <argv>`` runs
:mod:`yeto.rl.adapters.miles.island_entry` as ``__main__``.

This is not an import path (rl-code-quality-cleanup 0.6): import
``yeto.rl.adapters.miles.island_entry`` instead.
"""

if __name__ != "__main__":
    raise ImportError("yeto.rl.learner is a command-line entry only; "
                      "import yeto.rl.adapters.miles.island_entry")

import runpy

runpy.run_module("yeto.rl.adapters.miles.island_entry", run_name="__main__", alter_sys=True)
