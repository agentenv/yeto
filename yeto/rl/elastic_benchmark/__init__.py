"""RL elastic resource benchmark suite.

Study contracts, capability gating, evidence indexing, paired workloads and
layered reporting for fixed-versus-elastic GPU partition experiments. Nothing
in this package imports a GPU runtime; runners plug in through the runtime
capability attestation described in ``capabilities.py``.
"""
