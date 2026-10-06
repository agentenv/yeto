"""Protocol gateway core (design D1/D5, tasks 4.1/4.2): three entry points -> chat -> TITO session.

In-process library; the HTTP shell (task 10.x) wraps ``core.Gateway`` later.
"""
from .chains import ChainBreakReason, ChainRegistry, Locate, LocateKind, SessionMismatch
from .context import ContextProvider, IdentityContextProvider
from .core import AdmissionClosed, Gateway, GatewayConfig, GatewayError, SamplingOverrideError, TrajectoryInvalid, UnsupportedShape

__all__ = [
    "ChainBreakReason", "ChainRegistry", "Locate", "LocateKind", "SessionMismatch",
    "ContextProvider", "IdentityContextProvider",
    "AdmissionClosed", "Gateway", "GatewayConfig", "GatewayError", "SamplingOverrideError", "TrajectoryInvalid", "UnsupportedShape",
]
