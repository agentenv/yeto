"""National Compute (MI355X bare metal) nodes as yeto sampler pools.

:mod:`.client` (guarded order tool), :mod:`.tunnel` (ssh through the workspace
gateway), :mod:`.node_pool` (the node's ssh alias and lease key),
:mod:`.ssh_sampler` (docker on NVMe, model pull, sampler container),
:mod:`.tasks` (a member's sampler spec), :mod:`.lease_pool` (lease-aware sampler
pool), :mod:`.cli` (``python -m yeto.launch.nc``).
"""
