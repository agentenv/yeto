"""Backend-neutral agentic RL layer (S25 N10.2).

The harness (Harbor) owns tasks, sandboxes, the agent and the verifier.  This
package owns only what yeto adds around it: neutral types, admission,
outcome classification, the trust boundary and metrics.  Backend adapters:
``yeto.rl.adapters.miles.harbor_agent`` and
``yeto.rl.adapters.verl.harbor_agent_loop``.
"""

from .admission import AdmissionClosed, AdmissionGate, HarnessMetrics
from .outcome import classify
from .recorder import TokenRecorder
from .runner import AgentHarness, HarnessConfigError, run_agent_task
from .tokens import join_turns
from .trust import TrustViolation
from .types import (
    OUTCOME_KEY,
    AgentBudget,
    AgentTask,
    AgentTrajectory,
    Outcome,
    TokenAlignmentError,
    TokenSegments,
    TokenTurn,
    TrialVerdict,
)

__all__ = [
    "OUTCOME_KEY", "AdmissionClosed", "AdmissionGate", "AgentBudget", "AgentHarness", "AgentTask",
    "AgentTrajectory", "HarnessConfigError", "HarnessMetrics", "Outcome", "TokenAlignmentError",
    "TokenRecorder", "TokenSegments", "TokenTurn", "TrialVerdict", "TrustViolation", "classify", "join_turns",
    "run_agent_task",
]
