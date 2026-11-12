"""Miles rollout functions for the durable sample buffer.

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

from yeto.rl.engine.buffer_payload import complete_prompt_tokens, miles_sample_dict
from yeto.rl.engine.sampler_resume import new_incarnation, reclaim_requested
from yeto.rl.engine.offpolicy_source import flat_metrics, offpolicy_source_event, payload_source_sample

from .buffer_rollout import (
    DATA_SLOT_ENV,
    is_padding,
    EXTERNAL_CUT_ROUND_ATTR,
    DEFAULT_POLL_S,
    DEFAULT_WAIT_S,
    INCARNATION_ENV,
    MAX_POLICY_AGE_ENV,
    POLL_ENV,
    SAMPLER_ID_ENV,
    SHARD_ENV,
    TRAINER,
    WAIT_ENV,
    unidentified_metadata,
    BatchReader,
    DataSlot,
    SampleBufferRolloutError,
    SamplerRoundWriter,
    apply_data_slot,
    check_xhw_downweight,
    convertible_mode,
    merged_release_mode,
    miles_groups,
    open_buffer,
    open_lane_log,
    padding_samples,
    r3_layer_layout,
    r3_num_experts,
    register_reader,
    reader_basis_version,
    stamp_merged_release_cuts,
    xhw_round,
)
from yeto.rl.engine.data_split import row_runs
from yeto.rl.engine.xhw_guard import ENV as XHW_ENV
from yeto.rl.engine.task_dispense import DISPENSE_POSITION_KEY

from .proc_state import set_proc_state
from .rollout_meta_hook import current_policy_token, data_cursor, record_round_metadata
from .task_dispense import written as dispense_written

# takes in a row whose every group is malformed before the round fails (a systematic cause)
MAX_UNCONVERTIBLE_TAKES = 8

# the boat VM pools of the Harbor agentic path (``close_pools_now``, also used by .harbor_rollout)
BOAT_POOL_MODULE = "yeto.rl.harness.harbor.boat_pool"

def _flat(group: Any) -> list[Any]:
    out: list[Any] = []
    for item in group if isinstance(group, list) else [group]:
        out.extend(_flat(item) if isinstance(item, list) else [item])
    return out


class _DefaultRolloutLog:
    """Miles' args with no custom rollout log (its default logging, called once more)."""

    def __init__(self, args: Any) -> None:
        self._args = args

    def __getattr__(self, name: str) -> Any:
        if name == "custom_rollout_log_function_path":
            return None
        return getattr(self._args, name)

    def __iter__(self):
        for key, value in dict(self._args).items():
            yield key, (None if key == "custom_rollout_log_function_path" else value)

    def keys(self):
        return dict(self).keys()

    def __getitem__(self, key: str) -> Any:
        return dict(self)[key]


def log_rollout_data(rollout_id: Any, args: Any, samples: Any, rollout_extra_metrics: Any,
                     rollout_time: Any) -> bool:
    """``--custom-rollout-log-function-path`` of a buffer reader: Miles' default rollout
    metrics (rewards, response lengths, truncation, ...) over the real samples only; the
    padding samples of a short round (``metadata.yeto_padding``) would pull them toward
    one copied sample. Adds ``rollout/padded_samples`` and ``rollout/padded_fraction``.
    True: Miles skips its own logging (done here)."""
    flat = _flat(list(samples))
    real = [s for s in flat if not is_padding(s)]
    if len(real) == len(flat):
        return False  # nothing padded: Miles logs as usual
    from miles.ray.rollout import metrics as miles_metrics  # noqa: PLC0415

    extra = {**(rollout_extra_metrics or {}), "rollout/padded_samples": len(flat) - len(real),
             "rollout/padded_fraction": (len(flat) - len(real)) / max(1, len(flat))}
    miles_metrics.log_rollout_data(rollout_id, _DefaultRolloutLog(args), real, extra, rollout_time)
    return True


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
        # several sampler islands draw disjoint slots of the data; before any seek
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

    async def dispose(self) -> None:
        """Miles calls this at ``RolloutExecutor.dispose()`` (the island's end). The
        Harbor agentic path leases boat VMs in this process; trials that are still
        suspended between rounds (a convertible node keeps them through its trainer
        stints) hold theirs. Delete them here, as :mod:`.harbor_rollout` does for
        the stock rollout function; otherwise they live until their boat TTL."""
        import sys

        try:
            await super().dispose()
        finally:
            pools = sys.modules.get(BOAT_POOL_MODULE)  # imported only by a Harbor trial
            if pools is not None:
                deleted = await asyncio.to_thread(pools.close_pools_now)
                if deleted:
                    print(f"[sample-buffer] boat pool: deleted {deleted} VM(s) at rollout dispose",
                          flush=True)

    def _write(self, rollout_id: int, groups: list[Any], cursor: dict[str, int] | None) -> None:
        finished = []
        positions: list[int | None] = []
        for group in groups:
            samples = _flat(group)
            if any(s.status == Sample.Status.ABORTED for s in samples):
                continue  # an aborted group is regenerated, never trained
            finished.append((f"r{int(rollout_id)}-g{samples[0].group_index}",
                             [s.to_dict() for s in samples]))
            position = (getattr(samples[0], "metadata", None) or {}).get(DISPENSE_POSITION_KEY)
            positions.append(position if isinstance(position, int) and not isinstance(position, bool)
                             else None)
        written = self._rounds.write_round(rollout_id, finished, cursor, positions)
        # the rows of the groups now in the buffer are done for the dispenser
        dispense_written(rollout_id, [p for p in positions[:written] if p is not None])


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
        check_xhw_downweight(std_normalized=bool(getattr(args, "grpo_std_normalization", True)))
        self.groups = int(args.rollout_batch_size)
        self.samples_per_group = int(args.n_samples_per_prompt)
        buffer = open_buffer(batch_size=self.groups, create_index=create_index, reader=merge_shards)
        # with YETO_SAMPLE_RELAY_ADDR the spot samplers' groups arrive through
        # the head's relay; the pump appends them to this (local) buffer. Unset: no pump.
        from yeto.rl.spot_rollout.relay_buffer import maybe_start_relay_pump

        self.relay_pump = maybe_start_relay_pump(buffer)
        self.reader = BatchReader(
            buffer, cursor_log=open_lane_log(buffer),
            max_policy_age=int(age),
            wait_s=float(os.environ.get(WAIT_ENV, DEFAULT_WAIT_S)),
            poll_s=float(os.environ.get(POLL_ENV, DEFAULT_POLL_S)),
            group_size=self.samples_per_group)
        register_reader(self.reader)
        self._tokenizer: Any = None

    def __call__(self, input: RolloutFnInput) -> RolloutFnTrainOutput:
        if input.evaluation:
            raise SampleBufferRolloutError("the sample buffer reader does not evaluate; "
                                           "set --eval-function-path or disable eval")
        # the driver's basis (its published policy token), not Miles' weight_version
        r3_layers, num_experts = self._r3_layers()
        unconvertible: list[str] = []
        for _attempt in range(MAX_UNCONVERTIBLE_TAKES):
            groups, report = self.reader.take(input.rollout_id, self.groups,
                                              current_version=self._basis_version(input))
            groups = [(position, [complete_prompt_tokens(d, self._tokenize) for d in group])
                      for position, group in groups]
            converted, problems = miles_groups(groups, samples_per_group=self.samples_per_group,
                                               r3_layers=r3_layers, r3_num_experts=num_experts)
            unconvertible += problems
            if converted:
                break
            # every group of this take was malformed: its positions are consumed, take again
        else:
            raise SampleBufferRolloutError(
                f"rollout {input.rollout_id}: {MAX_UNCONVERTIBLE_TAKES} takes in a row with no group that "
                f"converts to Miles samples: {unconvertible[-1]}")
        if problems:  # left out of the round (padded instead): the padding copies kept groups only
            kept_positions = {position for position, _ in converted}
            groups = [(position, group) for position, group in groups if position in kept_positions]
        groups, xhw_meta = xhw_round(groups, report, getattr(self.reader, "xhw_actions", {}),
                                     std_normalized=bool(getattr(self.args, "grpo_std_normalization", True)),
                                     enabled=bool((os.environ.get(XHW_ENV) or "").strip()
                                                  not in ("", "0", "off", "false")))
        merged_cuts = getattr(self, "relay_pump", None) is None and merged_release_mode()
        pruned = 0
        if merged_cuts:  # a merged-lane synced trainer: name each group's merged release
            kept = [(position, group) for position, group in groups if stamp_merged_release_cuts(group)]
            pruned = len(groups) - len(kept)
            if not kept:
                raise SampleBufferRolloutError(
                    f"rollout {input.rollout_id}: all {len(groups)} groups came from merged "
                    "releases the store already pruned")
            groups = kept
        source = offpolicy_source_event(
            (payload_source_sample(d) for _, group in groups for d in group),
            rollout_id=input.rollout_id, backend="miles",
            trainer_version=-1 if input.weight_version is None else int(input.weight_version))
        samples = [[Sample.from_dict(d) for d in group] for _, group in converted]
        # Miles trains exactly rollout_batch_size * n samples: a round short of them
        # (short groups, pruned groups) is filled with loss-free padding groups
        missing = max(0, self.groups * self.samples_per_group - sum(len(group) for group in samples))
        report.padded_samples = missing
        report.padded_fraction = missing / max(1, self.groups * self.samples_per_group)
        record_round_metadata(self.args, int(input.rollout_id),
                              buffer_stale_groups=int(report.stale_groups) + pruned,
                              buffer_incomplete_groups=int(getattr(report, "incomplete_groups", 0)),
                              buffer_dropped_small_groups=int(getattr(report, "dropped_small_groups", 0)),
                              buffer_carried_over_groups=int(getattr(report, "carried_over_groups", 0)),
                              buffer_padded_samples=int(missing),
                              **xhw_meta,
                              **({"buffer_stale_positions": row_runs(stale)}
                                 if (stale := getattr(report, "stale_positions", None)) else {}),
                              **unidentified_metadata(report))
        if getattr(self, "relay_pump", None) is not None or merged_cuts:
            # the round's evidence names external merged cuts and bounds
            # their age by the merged version this take gated on
            if report.gate_version is None:
                raise SampleBufferRolloutError(f"rollout {input.rollout_id}: a relay or merged-lane "
                                               "trainer took groups without a merged gate version")
            set_proc_state(self.args, EXTERNAL_CUT_ROUND_ATTR,
                           {"rollout_id": int(input.rollout_id), "merged_version": int(report.gate_version)})
        self._run_round_hooks(samples)
        if missing > 0:  # after the hooks: the round's yeto metadata counts real samples only
            pads = padding_samples([group for _, group in groups], missing,
                                   samples_per_group=self.samples_per_group)
            samples.extend([Sample.from_dict(miles_sample_dict(d, r3_layers=r3_layers,
                                                               r3_num_experts=num_experts))] for d in pads)
        return RolloutFnTrainOutput(samples=samples, metrics={
            **report.metrics(), "sample_buffer/unconvertible_groups": float(len(unconvertible)),
            **flat_metrics(source)})

    @staticmethod
    def _basis_version(input: RolloutFnInput) -> int | None:
        try:
            token = current_policy_token()
        except ImportError:  # no Ray in this process (unit fixtures): no published token
            token = None
        return reader_basis_version(input.weight_version, token)

    def _r3_layers(self) -> tuple[tuple[int, int] | None, int | None]:
        """``((num_layers, first_dense_layers), num_routed_experts)`` of the trained model,
        for the R3 layer axis and id range."""
        if not hasattr(self, "_r3_layers_cached"):
            checkpoint = getattr(self.args, "hf_checkpoint", None)
            self._r3_layers_cached = (r3_layer_layout(checkpoint), r3_num_experts(checkpoint))
        return self._r3_layers_cached

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
        from .rollout_meta_hook import BUFFER_READER_ATTR  # noqa: PLC0415

        set_proc_state(self.args, BUFFER_READER_ATTR, True)  # metadata gets the batch summary
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
    """Convertible node: in ``sampler`` mode Miles
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
