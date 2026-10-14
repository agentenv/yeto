"""Miles rollout functions for the durable sample buffer (L4.5).

Loaded by Miles in the rollout executor process (``--rollout-function-path``, set
by :func:`.buffer_rollout.rollout_argv`); the contract and the logic live in
:mod:`.buffer_rollout`.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any

from miles.rollout.base_types import (
    BaseRolloutFn,
    RolloutFnConstructorInput,
    RolloutFnInput,
    RolloutFnTrainOutput,
)
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn
from miles.utils.types import Sample

from .buffer_rollout import (
    DEFAULT_POLL_S,
    DEFAULT_WAIT_S,
    MAX_POLICY_AGE_ENV,
    POLL_ENV,
    SAMPLER_ID_ENV,
    WAIT_ENV,
    BatchReader,
    GroupWriter,
    SampleBufferRolloutError,
    open_buffer,
    register_reader,
    renumber_group,
)


def _flat(group: Any) -> list[Any]:
    out: list[Any] = []
    for item in group if isinstance(group, list) else [group]:
        out.extend(_flat(item) if isinstance(item, list) else [item])
    return out


class BufferWriterRolloutFn(InferenceRolloutFn):
    """Sampler: Miles' rollout; every finished group is also written to the buffer."""

    def __init__(self, input: RolloutFnConstructorInput) -> None:
        super().__init__(input)
        sampler_id = os.environ.get(SAMPLER_ID_ENV)
        if not sampler_id:
            raise SampleBufferRolloutError(f"{SAMPLER_ID_ENV} is not set")
        # a per-process suffix: a restarted sampler repeats Miles group indexes, so the
        # sample ids stay unique; the trainer renumbers groups by buffer position
        self._writer = GroupWriter(open_buffer(), f"{sampler_id}-{uuid.uuid4().hex[:8]}")

    async def _call_train(self, input: Any) -> Any:
        output = await super()._call_train(input)
        await asyncio.to_thread(self._write, input.rollout_id, output.samples)
        return output

    def _write(self, rollout_id: int, groups: list[Any]) -> None:
        for group in groups:
            samples = _flat(group)
            if any(s.status == Sample.Status.ABORTED for s in samples):
                continue  # an aborted group is regenerated, never trained
            key = f"r{int(rollout_id)}-g{samples[0].group_index}"
            self._writer.write([s.to_dict() for s in samples], group_key=key)


class BufferReaderRolloutFn(BaseRolloutFn):
    """Trainer: the round is selected from the buffer, nothing is generated."""

    def __init__(self, input: RolloutFnConstructorInput) -> None:
        super().__init__(input)
        args = input.args
        age = os.environ.get(MAX_POLICY_AGE_ENV)
        if age is None:
            raise SampleBufferRolloutError(f"{MAX_POLICY_AGE_ENV} is not set")
        self.groups = int(args.rollout_batch_size)
        self.samples_per_group = int(args.n_samples_per_prompt)
        self.reader = BatchReader(
            open_buffer(batch_size=self.groups), max_policy_age=int(age),
            wait_s=float(os.environ.get(WAIT_ENV, DEFAULT_WAIT_S)),
            poll_s=float(os.environ.get(POLL_ENV, DEFAULT_POLL_S)))
        register_reader(self.reader)

    def __call__(self, input: RolloutFnInput) -> RolloutFnTrainOutput:
        if input.evaluation:
            raise SampleBufferRolloutError("the sample buffer reader does not evaluate; "
                                           "set --eval-function-path or disable eval")
        groups, report = self.reader.take(input.rollout_id, self.groups,
                                          current_version=input.weight_version)
        samples = [[Sample.from_dict(d) for d in renumber_group(
            group, position=position, samples_per_group=self.samples_per_group)]
            for position, group in groups]
        return RolloutFnTrainOutput(samples=samples, metrics=report.metrics())
