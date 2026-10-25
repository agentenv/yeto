"""Miles rollout functions for the durable sample buffer (L4.5).

Loaded by Miles in the rollout executor process (``--rollout-function-path``, set
by :func:`.buffer_rollout.rollout_argv`); the contract and the logic live in
:mod:`.buffer_rollout`.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from miles.rollout.base_types import (
    BaseRolloutFn,
    RolloutFnConstructorInput,
    RolloutFnInput,
    RolloutFnTrainOutput,
)
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn
from miles.utils.function_registry import load_function
from miles.utils.types import Sample

from yeto.rl.engine.buffer_payload import complete_prompt_tokens
from yeto.rl.engine.sampler_resume import new_incarnation, reclaim_requested
from yeto.rl.engine.offpolicy_source import flat_metrics, offpolicy_source_event, payload_source_sample

from .buffer_rollout import (
    DATA_SLOT_ENV,
    DEFAULT_POLL_S,
    DEFAULT_WAIT_S,
    INCARNATION_ENV,
    MAX_POLICY_AGE_ENV,
    POLL_ENV,
    SAMPLER_ID_ENV,
    SHARD_ENV,
    TRAINER,
    WAIT_ENV,
    BatchReader,
    DataSlot,
    SampleBufferRolloutError,
    SamplerRoundWriter,
    apply_data_slot,
    convertible_mode,
    open_buffer,
    open_lane_log,
    register_reader,
    renumber_group,
)
from .rollout_meta_hook import data_cursor, record_round_metadata


def _flat(group: Any) -> list[Any]:
    out: list[Any] = []
    for item in group if isinstance(group, list) else [group]:
        out.extend(_flat(item) if isinstance(item, list) else [item])
    return out


class BufferWriterRolloutFn(InferenceRolloutFn):
    """Sampler: Miles' rollout; every finished group is also written to the buffer.

    A replacement incarnation of the island continues where the buffer says the
    earlier ones stopped (:class:`.buffer_rollout.SamplerRoundWriter`): the data
    source is put at the saved cursor before the first round, and a round
    already in the buffer is refused."""

    resumable = True  # a convertible node keeps its own rollout numbering (no resume)

    def __init__(self, input: RolloutFnConstructorInput) -> None:
        super().__init__(input)
        sampler_id = os.environ.get(SAMPLER_ID_ENV)
        if not sampler_id:
            raise SampleBufferRolloutError(f"{SAMPLER_ID_ENV} is not set")
        # the incarnation suffix keeps the sample ids unique: a restarted sampler repeats
        # Miles group indexes; the trainer renumbers groups by buffer position
        incarnation = os.environ.get(INCARNATION_ENV) or new_incarnation()
        self._rounds = SamplerRoundWriter(self._open_writer_buffer(), sampler_id,
                                          incarnation=incarnation, resume=self.resumable,
                                          reclaimed=lambda: reclaim_requested(os.environ))
        self._shuffle = bool(getattr(input.args, "rollout_shuffle", False))  # the seek re-shuffles
        # several sampler islands draw disjoint slots of the data (S26 E9.3); before any seek
        apply_data_slot(self.data_source, DataSlot.parse(os.environ.get(DATA_SLOT_ENV)),
                        shuffle=self._shuffle)

    def _open_writer_buffer(self) -> Any:
        return open_buffer(sampler=True)

    async def _call_train(self, input: Any) -> Any:
        self._rounds.seek(self.data_source, shuffle=self._shuffle)
        output = await super()._call_train(input)
        cursor, _ = data_cursor(self.data_source)
        await asyncio.to_thread(self._write, input.rollout_id, output.samples, cursor)
        return output

    def _write(self, rollout_id: int, groups: list[Any], cursor: dict[str, int] | None) -> None:
        finished = []
        for group in groups:
            samples = _flat(group)
            if any(s.status == Sample.Status.ABORTED for s in samples):
                continue  # an aborted group is regenerated, never trained
            finished.append((f"r{int(rollout_id)}-g{samples[0].group_index}",
                             [s.to_dict() for s in samples]))
        self._rounds.write_round(rollout_id, finished, cursor)


class _BufferRound:
    """Selects one training round from the buffer (the reader side of both the
    trainer island and a convertible node in trainer mode)."""

    def __init__(self, args: Any, *, create_index: bool = True, merge_shards: bool = True) -> None:
        """``merge_shards``: the trainer merges the sampler shards of an object
        store (``open_buffer(reader=True)``); a convertible node does not (the
        head merges its pool's shards)."""
        age = os.environ.get(MAX_POLICY_AGE_ENV)
        if age is None:
            raise SampleBufferRolloutError(f"{MAX_POLICY_AGE_ENV} is not set")
        self.args = args
        self.groups = int(args.rollout_batch_size)
        self.samples_per_group = int(args.n_samples_per_prompt)
        buffer = open_buffer(batch_size=self.groups, create_index=create_index, reader=merge_shards)
        # S27 G-EA: with YETO_SAMPLE_RELAY_ADDR the spot samplers' groups arrive through
        # the head's relay; the pump appends them to this (local) buffer. Unset: no pump.
        from yeto.rl.spot_rollout.relay_buffer import maybe_start_relay_pump

        self.relay_pump = maybe_start_relay_pump(buffer)
        self.reader = BatchReader(
            buffer, cursor_log=open_lane_log(buffer),
            max_policy_age=int(age),
            wait_s=float(os.environ.get(WAIT_ENV, DEFAULT_WAIT_S)),
            poll_s=float(os.environ.get(POLL_ENV, DEFAULT_POLL_S)))
        register_reader(self.reader)
        self._tokenizer: Any = None

    def __call__(self, input: RolloutFnInput) -> RolloutFnTrainOutput:
        if input.evaluation:
            raise SampleBufferRolloutError("the sample buffer reader does not evaluate; "
                                           "set --eval-function-path or disable eval")
        groups, report = self.reader.take(input.rollout_id, self.groups,
                                          current_version=input.weight_version)
        groups = [(position, [complete_prompt_tokens(d, self._tokenize) for d in group])
                  for position, group in groups]
        source = offpolicy_source_event(
            (payload_source_sample(d) for _, group in groups for d in group),
            rollout_id=input.rollout_id, backend="miles",
            trainer_version=-1 if input.weight_version is None else int(input.weight_version))
        samples = [[Sample.from_dict(d) for d in renumber_group(
            group, position=position, samples_per_group=self.samples_per_group)]
            for position, group in groups]
        record_round_metadata(self.args, int(input.rollout_id),
                              buffer_stale_groups=int(report.stale_groups))
        self._run_round_hooks(samples)
        return RolloutFnTrainOutput(samples=samples, metrics={**report.metrics(), **flat_metrics(source)})

    def _tokenize(self, text: str) -> list[int]:
        """Prompt ids for a sample whose engine gave none (loads the tokenizer on first use)."""
        if self._tokenizer is None:  # the one Miles tokenizer load yeto allows (test_provenance)
            from .rollout_meta_hook import _load_miles_tokenizer

            self._tokenizer = _load_miles_tokenizer(self.args)
        return list(self._tokenizer(text, add_special_tokens=True)["input_ids"])

    def _run_round_hooks(self, samples: list[list[Any]]) -> None:
        """Call the end-of-round hooks Miles' generating rollout calls, so the yeto
        rollout metadata reaches the driver. Every selected group is trained, so the
        kept groups and all groups are the same list; there is no data source."""
        if f := load_function(self.args.rollout_sample_filter_path):
            f(self.args, samples)
        if f := load_function(self.args.rollout_all_samples_process_path):
            f(self.args, samples, None)


class BufferReaderRolloutFn(BaseRolloutFn):
    """Trainer: the round is selected from the buffer, nothing is generated."""

    def __init__(self, input: RolloutFnConstructorInput) -> None:
        super().__init__(input)
        self.args = input.args
        self._round = _BufferRound(input.args)
        self.reader = self._round.reader

    def __call__(self, input: RolloutFnInput) -> RolloutFnTrainOutput:
        return self._round(input)


class ConvertibleRolloutFn(BufferWriterRolloutFn):
    """Convertible node (rl-trainer-start-stop L7.5): in ``sampler`` mode Miles
    generates the round and every group goes to this node's shard of the index;
    in ``trainer`` mode the round is read from the buffer like
    :class:`BufferReaderRolloutFn`. The node switches the mode in this process
    (``buffer_rollout.executor_set_mode``) when it converts."""

    resumable = False

    def __init__(self, input: RolloutFnConstructorInput) -> None:
        super().__init__(input)
        # the head merges the shards into the index: this node never writes it
        self._round = _BufferRound(input.args, create_index=False, merge_shards=False)

    def _open_writer_buffer(self) -> Any:
        shard = os.environ.get(SHARD_ENV)
        if not shard:
            raise SampleBufferRolloutError(f"{SHARD_ENV} is not set")
        return open_buffer(shard=shard)

    def __call__(self, input: RolloutFnInput) -> Any:
        if convertible_mode() == TRAINER:
            return self._round(input)
        return super().__call__(input)
