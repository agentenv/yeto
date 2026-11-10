"""Cost protection for unattended runs: resource ledger, budget breaker,
idle shutdown, exact cleanup and the cost report.

:mod:`.ledger`   crash-safe per-run list of every billed resource this run created
:mod:`.billing`  what those resources cost (Modal billing API + live estimate, NC lease x rate)
:mod:`.idle`     idle detection with a loading-phase exemption
:mod:`.guard`    the breaker thread the launcher runs (warn, graceful stop, forced cleanup)
:mod:`.cleanup`  ``yeto cleanup --run``: release exactly what the ledger lists
:mod:`.report`   ``yeto cost --run``: per resource / per hour cost and GPU effective utilization
"""
