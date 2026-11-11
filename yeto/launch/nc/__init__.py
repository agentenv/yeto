"""National Compute (MI355X bare metal) as SkyPilot SSH node pools for yeto.

:mod:`.client` (guarded order tool), :mod:`.tunnel` (ssh through the workspace
gateway), :mod:`.node_pool` (alias, host bootstrap, ``sky ssh up``, AMD GPUs),
:mod:`.tasks` (sky tasks on the pool), :mod:`.lease_pool` (lease-aware sampler
pool), :mod:`.cli` (``python -m yeto.launch.nc``).
"""
