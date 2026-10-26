"""Token recorder port: the OpenAI endpoint an agent calls, with token bookkeeping (S25 N10.3).

A Harbor installed agent (codex) calls the model over OpenAI text protocol
from inside the sandbox and records no tokens.  A token-in/token-out trainer
needs the exact ids the sampler rendered and generated, so a recorder sits in
front of the sampler: it renders the prompt, generates with the session's
sampling parameters, and keeps (prompt ids, completion ids, logprobs) per call.

Miles already has one (its session server).  For verl the implementation is
N10.5 (in front of SGLang / vllm-ascend); this module only fixes the port.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from .types import TokenTurn


class TokenRecorder(Protocol):
    async def open(self, trajectory_id: str, sampling_params: Mapping[str, Any]) -> str:
        """Start a session; return the OpenAI base URL the agent must call.

        The recorder applies ``sampling_params`` to every call of the session,
        whatever the agent sends, so the trainer's sampling settings hold.
        """

    async def close(self, trajectory_id: str) -> Sequence[TokenTurn]:
        """End the session; return its model calls in call order."""
