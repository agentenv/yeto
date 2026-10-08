"""Stage-W entry: run plain Miles ``train.py`` under the ports driver's Ray pinning.

``python3 -m yeto.rl.stage_w_entry <miles>/train.py <stage-W argv>``

Plain ``train.py`` never calls ``ray.init``; Ray auto-connects through
``RAY_ADDRESS`` with no job ``runtime_env``, so its actors inherit the
raylet's environment.  The island starts Ray without ``/root/Megatron-LM`` on
``PYTHONPATH`` (launcher.py island_megatron_path: only the learner gets it,
and the ports driver forwards it to every actor through
``connect_island_ray``).  The image's editable Megatron finder maps only
``megatron.core`` / ``megatron.training``, so the stage-W critic's
``get_model`` -> ``from megatron.post_training.checkpointing import ...``
(taken whenever nvidia-modelopt is importable) failed with
``ModuleNotFoundError`` (s13-g1-modal-20261007b).  This entry connects to
the island's Ray exactly like the ports driver, then runs ``train.py``.
"""

from __future__ import annotations

import os
import runpy
import sys


def main(argv: list[str] | None = None, *, connect=None, run_path=runpy.run_path) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit("usage: python3 -m yeto.rl.stage_w_entry <miles>/train.py [argv...]")
    script = argv[0]
    if connect is None:
        from yeto.rl.adapters.miles.entry import connect_island_ray as connect
    connect(environ=os.environ)
    sys.argv = [script, *argv[1:]]
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))  # as `python3 train.py` would
    run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
