"""Miles pieces of the codex OpenEnv generate wrapper (yeto-framework-decoupling 5.7).

``yeto.rl.harness.codex.codex_openenv_generate`` keeps the trajectory
bookkeeping (neutral) and reaches these through the backend registry
(role ``harness_glue``): upstream ``agentic_tool_call.generate``, the
``Sample.Status.ABORTED`` value and the per-session sample collection.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable


def load_agentic_upstream() -> Callable[[Any], Awaitable[Any]]:
    from miles.rollout.generate_hub.agentic_tool_call import generate as upstream

    return upstream


def aborted_status() -> Any:
    from miles.utils.types import Sample

    return Sample.Status.ABORTED


async def collect_segment_session(input: Any, router: str, session_id: str) -> tuple[list[Any], dict[str, Any]]:
    """Collect (and delete) one extra session exactly like upstream's tracer does."""
    from miles.rollout.generate_utils.openai_endpoint_utils import (
        COMPUTED_FIELDS,
        ROLLOUT_SAMPLING_MASK_FIELDS,
        OpenAIEndpointTracer,
        should_return_sampling_mask,
    )

    fields = COMPUTED_FIELDS
    if should_return_sampling_mask(input.args, input.sampling_params, evaluation=input.evaluation):
        fields += ROLLOUT_SAMPLING_MASK_FIELDS
    tracer = OpenAIEndpointTracer(router_url=router, session_id=session_id, samples_wire_fields=fields)
    reply = await tracer.collect_samples(input.sample, max_seq_len=getattr(input.args, "max_seq_len", None))
    return list(reply.samples), dict(reply.session_metadata or {})
