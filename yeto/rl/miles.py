"""Forwarding module (yeto-framework-decoupling phase 4): moved to ``yeto.rl.adapters.miles.legacy.engine``.

Importing the old path yields the very same module object; running it with
``python -m`` runs the new module.  Delete once no caller uses the old path
(design D2).
"""

if __name__ == "__main__":
    import runpy

    runpy.run_module("yeto.rl.adapters.miles.legacy.engine", run_name="__main__", alter_sys=True)
else:
    import sys as _sys

    from yeto.rl.adapters.miles.legacy import engine as _module

    _sys.modules[__name__] = _module
