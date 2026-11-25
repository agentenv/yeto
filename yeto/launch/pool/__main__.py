"""``python -m yeto.launch.pool``: the provider-neutral name of the pool commands
(``pool``, ``pool-stop``, ``pool-status``, ``pool-scale``, ...).

The same command line as ``python -m yeto.launch.nc``, which keeps working unchanged.
"""

from yeto.launch.nc.cli import main

raise SystemExit(main(prog="python -m yeto.launch.pool"))
