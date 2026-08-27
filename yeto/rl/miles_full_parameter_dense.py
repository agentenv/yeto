"""Production Miles GRPO hook for H=1 dense full-parameter DiLoCo."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..protocol import DTYPE_F32, SyncerClient
from .dense_sweep_wire import (
    DenseSweepConfig,
    DenseSweepWire,
    PendingDenseWirePolicy,
)
from .local_learner import (
    ComponentIdentity,
    dense_sweep_session_contract_hash,
)
from .miles_chunked_full_parameter import (
    MilesChunkedFullParameterAdapter,
    ReferencedPolicyCut,
    StoredAuthoritativeFragment,
)
from .trajectory_evidence import read_trajectory_batch_evidence


@dataclass(frozen=True)
class MilesDenseFullParameterConfig:
    """Closed configuration supplied by Yeto's learner launcher."""

    component: ComponentIdentity
    wire: DenseSweepConfig
    learner_generations: tuple[int, ...]
    minimum_fragments: int
    training_contract_hash: str
    expected_layout_hash: str
    max_fragment_bytes: int = 2 << 30
    max_chunk_bytes: int = 256 << 20

    def __post_init__(self) -> None:
        if self.component.role != "actor":
            raise ValueError("GRPO dense sync requires one actor component")
        if (
            not self.learner_generations
            or self.wire.learner_id >= len(self.learner_generations)
            or self.learner_generations[self.wire.learner_id]
            != self.wire.learner_generation
            or any(
                type(value) is not int or value < 0
                for value in self.learner_generations
            )
        ):
            raise ValueError("dense learner generation is outside the fixed roster")
        if (
            type(self.minimum_fragments) is not int
            or self.minimum_fragments < 1
            or type(self.max_fragment_bytes) is not int
            or not 4 <= self.max_fragment_bytes <= 2 << 30
            or self.max_fragment_bytes % 4
            or type(self.max_chunk_bytes) is not int
            or not 4 <= self.max_chunk_bytes < 1 << 30
            or self.max_chunk_bytes % 4
            or type(self.expected_layout_hash) is not str
            or len(self.expected_layout_hash) != 64
            or any(
                value not in "0123456789abcdef"
                for value in self.expected_layout_hash
            )
        ):
            raise ValueError("dense full-parameter transport bounds are invalid")
        if (
            type(self.training_contract_hash) is not str
            or len(self.training_contract_hash) != 64
            or any(
                value not in "0123456789abcdef"
                for value in self.training_contract_hash
            )
        ):
            raise ValueError("dense training contract hash is malformed")

    @property
    def roster(self) -> dict[int, int]:
        return dict(enumerate(self.learner_generations))


@dataclass
class _PendingPublication:
    base: ReferencedPolicyCut
    local: ReferencedPolicyCut
    target: ReferencedPolicyCut
    wire: PendingDenseWirePolicy
    applied: bool = False


class MilesFullParameterDenseSync:
    """Bridge one real Miles optimizer step through an atomic dense sweep."""

    def __init__(self, args: Any, config: MilesDenseFullParameterConfig) -> None:
        self.args = args
        self.config = config
        self.actor_model = None
        self.rollout_manager = None
        self.adapter: MilesChunkedFullParameterAdapter | None = None
        self.wire: DenseSweepWire | None = None
        self.current: ReferencedPolicyCut | None = None
        self.pending_publication: _PendingPublication | None = None
        self.initial_publication_pending = False
        self.finished = False
        self.evaluation_policy_versions = tuple(
            getattr(args, "yeto_rl_eval_policy_versions", ())
        )
        self.evaluation_results: dict[int, dict[str, Any]] = {}
        if self.evaluation_policy_versions:
            if self.evaluation_policy_versions != (0, config.wire.policy_rounds):
                raise ValueError(
                    "dense heldout evaluation must bind initial and terminal policies"
                )
            dataset_name = getattr(args, "yeto_rl_eval_dataset_name", None)
            prompt_count = getattr(args, "yeto_rl_eval_prompt_count", None)
            samples = getattr(args, "yeto_rl_eval_samples_per_prompt", None)
            summary = Path(
                str(getattr(args, "yeto_rl_eval_summary_path", "") or "")
            )
            if (
                not isinstance(dataset_name, str)
                or not dataset_name
                or type(prompt_count) is not int
                or prompt_count < 1
                or type(samples) is not int
                or samples < 1
                or not summary.is_absolute()
            ):
                raise ValueError("dense heldout evaluation contract is malformed")

    async def initialize(self, *, actor_model, rollout_manager) -> None:
        if os.environ.get("MILES_EXPERIMENTAL_FT_TRAINER", "0") != "0":
            raise RuntimeError("dense full-parameter sync requires Miles v1 actors")
        if self.actor_model is not None:
            raise RuntimeError("dense full-parameter sync was already initialized")
        if int(getattr(self.args, "start_rollout_id", 0)) != 0:
            raise RuntimeError("Milestone-1 dense sync requires a version-zero start")
        if (
            int(getattr(self.args, "num_rollout", self.config.wire.policy_rounds))
            != self.config.wire.policy_rounds
        ):
            raise RuntimeError(
                "Miles rollout count differs from the dense policy budget"
            )
        if getattr(self.args, "yeto_rl_trajectory_evidence_dir", None) is None:
            raise RuntimeError("dense GRPO requires immutable trajectory evidence")

        adapter, anchor = await MilesChunkedFullParameterAdapter.initialize(
            actor_model,
            policy_version=0,
            algorithm="grpo",
            components=(self.config.component,),
            # Derive P from the topology-owner lower bound, exactly as the
            # frozen probe does. The probe's P is an expectation, not a new
            # lower bound to feed back into the packing algorithm.
            minimum_fragments=1,
            expected_fragments=self.config.minimum_fragments,
            expected_layout_hash=self.config.expected_layout_hash,
            max_fragment_bytes=self.config.max_fragment_bytes,
            max_chunk_bytes=self.config.max_chunk_bytes,
        )
        session_hash = dense_sweep_session_contract_hash(
            adapter.layout,
            policy_rounds=self.config.wire.policy_rounds,
            learner_generations=self.config.roster,
            training_contract_hash=self.config.training_contract_hash,
        )
        client = SyncerClient(
            self.config.wire.syncer_addr,
            self.config.wire.learner_id,
            adapter.layout.fragments,
            dtype=DTYPE_F32,
            num_streams=self.config.wire.wan_streams,
            max_reconnects=None,
            session_contract_hash=session_hash,
        )
        wire = DenseSweepWire(adapter.layout.fragments, self.config.wire, client=client)
        sink = adapter.fragment_sink(anchor)
        try:
            initial = wire.start(
                {
                    fragment_id: adapter.fragment_parts(anchor, fragment_id)
                    for fragment_id in range(adapter.layout.fragments.num_fragments)
                },
                policy_version=0,
                payload_sink=sink,
            )
            if not all(
                isinstance(value, StoredAuthoritativeFragment) for value in initial
            ):
                raise RuntimeError("dense syncer returned an untyped initial cut")
            adapter.verify_initial_fragments(anchor, initial)
        except BaseException:
            sink.release_all()
            wire.close()
            adapter.release(anchor)
            raise
        sink.release_all()
        self.actor_model = actor_model
        self.rollout_manager = rollout_manager
        self.adapter = adapter
        self.wire = wire
        self.current = anchor
        self.initial_publication_pending = True
        self.args.start_rollout_id = 0

    async def after_local_train(
        self,
        *,
        rollout_id: int,
        actor_model,
        rollout_data,
    ) -> bool:
        del rollout_data
        if (
            actor_model is not self.actor_model
            or self.adapter is None
            or self.wire is None
            or self.current is None
            or self.initial_publication_pending
            or self.pending_publication is not None
            or rollout_id != self.current.policy_version
        ):
            raise RuntimeError("Miles called dense sync outside one active H=1 round")
        from .miles import get_current_published_policy_identity

        if get_current_published_policy_identity(
            self.args,
            expected_policy_version=rollout_id,
        ) != (rollout_id, self.current.policy_hash):
            raise RuntimeError("GRPO rollout does not bind the current dense anchor")
        trajectories = self._trajectory_evidence(rollout_id)
        receipt = await self.adapter.record_grpo_local_step(
            actor_model,
            anchor=self.current,
            rollout_id=rollout_id,
            learner_id=self.config.wire.learner_id,
            learner_generation=self.config.wire.learner_generation,
            trajectories=trajectories,
        )
        local = await self.adapter.capture(
            actor_model,
            policy_version=rollout_id,
            local_step_generation=1,
        )
        update_id = _sweep_update_id(receipt, self.current, local)
        sink = self.adapter.fragment_sink(self.current)
        wire_pending = self.wire.exchange(
            base_policy_version=rollout_id,
            trained_tokens=receipt.trained_tokens,
            sweep_update_id=update_id,
            delta_parts={
                fragment_id: self.adapter.delta_parts(
                    self.current,
                    local,
                    fragment_id,
                )
                for fragment_id in range(self.adapter.layout.fragments.num_fragments)
            },
            payload_sink=sink,
        )
        if not all(
            isinstance(value, StoredAuthoritativeFragment)
            for value in wire_pending.payloads
        ):
            raise RuntimeError("dense syncer returned an untyped target cut")
        target = self.adapter.assemble_target(
            self.current,
            target_policy_version=rollout_id + 1,
            fragments=wire_pending.payloads,
        )
        self._append_event(
            {
                "event": "rl_dense_local_step",
                "base_policy_version": rollout_id,
                "base_policy_hash": self.current.policy_hash,
                "target_policy_version": rollout_id + 1,
                "target_policy_hash": target.policy_hash,
                "input_batch_hash": receipt.input_batch_hash,
                "trained_tokens": receipt.trained_tokens,
                "trajectory_count": len(receipt.trajectory_ids),
                "optimizer_steps": receipt.optimizer_steps,
                "sweep_update_id": update_id,
            }
        )
        pending_publication = _PendingPublication(
            self.current,
            local,
            target,
            wire_pending,
        )
        self.pending_publication = pending_publication
        optimizer_before = tuple(await actor_model.full_parameter_optimizer_states())
        await self.adapter.apply(actor_model, target, commit_token=update_id)
        optimizer_after = tuple(await actor_model.full_parameter_optimizer_states())
        _validate_optimizer_apply_transition(
            optimizer_before,
            optimizer_after,
            base_policy_version=rollout_id,
            target_policy_version=rollout_id + 1,
        )
        pending_publication.applied = True
        sink.release_all()
        return wire_pending.terminal

    async def after_inference_publication(
        self,
        *,
        rollout_id: int | None,
        actor_model,
    ) -> None:
        if actor_model is not self.actor_model or self.current is None:
            raise RuntimeError("Miles published through the wrong dense actor group")
        from .miles import (
            _policy_token,
        )

        if rollout_id is None:
            if (
                not self.initial_publication_pending
                or self.pending_publication is not None
            ):
                raise RuntimeError("unexpected initial dense inference publication")
            await self._set_rollout_token(_policy_token(self.current.policy_version))
            await self._set_published_policy_identity(
                policy_version=self.current.policy_version,
                policy_hash=self.current.policy_hash,
            )
            self._record_publication(
                self.current.policy_version,
                self.current.policy_hash,
            )
            self.initial_publication_pending = False
            await self._evaluate_published_policy(
                self.current.policy_version,
                self.current.policy_hash,
            )
            return

        pending = self.pending_publication
        if (
            pending is None
            or not pending.applied
            or rollout_id != pending.base.policy_version
        ):
            raise RuntimeError(
                "dense inference publication has no matching applied cut"
            )
        await self._set_rollout_token(_policy_token(pending.target.policy_version))
        await self._set_published_policy_identity(
            policy_version=pending.target.policy_version,
            policy_hash=pending.target.policy_hash,
        )
        if self.wire is None or self.adapter is None:
            raise RuntimeError("dense wire disappeared before publication commit")
        self.wire.commit_applied(pending.wire)
        self.current = pending.target
        self.pending_publication = None
        self.adapter.release(pending.base)
        self.adapter.release(pending.local)
        self.finished = pending.wire.terminal
        self._record_publication(
            self.current.policy_version,
            self.current.policy_hash,
        )
        await self._evaluate_published_policy(
            self.current.policy_version,
            self.current.policy_hash,
        )

    async def finalize(self) -> None:
        if (
            not self.finished
            or self.initial_publication_pending
            or self.pending_publication is not None
            or self.wire is None
            or self.current is None
            or self.current.policy_version != self.config.wire.policy_rounds
            or self.wire.policy_version != self.current.policy_version
            or set(self.evaluation_results) != set(self.evaluation_policy_versions)
        ):
            raise RuntimeError("Miles stopped before dense full-parameter finalization")
        self.wire.close()
        if self.adapter is not None:
            self.adapter.release(self.current)
        self.current = None

    def _append_event(self, event: dict[str, Any]) -> None:
        from .miles import _append_rl_event

        _append_rl_event(self.args, event)

    def _record_publication(self, policy_version: int, policy_hash: str) -> None:
        self._append_event(
            {
                "event": "rl_dense_policy_publication",
                "policy_version": policy_version,
                "sync/global_policy_hash": policy_hash,
                "terminal": policy_version == self.config.wire.policy_rounds,
            }
        )

    async def _evaluate_published_policy(
        self,
        policy_version: int,
        policy_hash: str,
    ) -> None:
        if policy_version not in self.evaluation_policy_versions:
            return
        if policy_version in self.evaluation_results:
            raise RuntimeError("dense heldout policy was evaluated more than once")
        await self.rollout_manager.eval.remote(rollout_id=policy_version)
        event = self._read_evaluation_event(policy_version, policy_hash)
        self.evaluation_results[policy_version] = event
        self._write_evaluation_summary()

    def _read_evaluation_event(
        self,
        policy_version: int,
        policy_hash: str,
    ) -> dict[str, Any]:
        event_path = Path(self.args.yeto_rl_event_tape)
        matches = []
        try:
            with event_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.endswith("\n"):
                        raise RuntimeError("dense evaluation event tape has a torn row")
                    value = json.loads(line)
                    if (
                        isinstance(value, dict)
                        and value.get("event") == "rl_eval_result"
                        and value.get("policy_version") == policy_version
                    ):
                        matches.append(value)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError("cannot read dense heldout evaluation evidence") from error
        if len(matches) != 1:
            raise RuntimeError(
                "dense heldout evaluation did not emit exactly one result"
            )
        value = matches[0]
        result = value.get("rl/eval/result")
        pass_at_1 = value.get("rl/eval/pass_at_1")
        expected_samples = (
            self.args.yeto_rl_eval_prompt_count
            * self.args.yeto_rl_eval_samples_per_prompt
        )
        if (
            value.get("island_id") != self.config.wire.learner_id
            or value.get("rollout_id") != policy_version
            or value.get("dataset_name") != self.args.yeto_rl_eval_dataset_name
            or value.get("sample_count") != expected_samples
            or value.get("sync/global_policy_hash") != policy_hash
            or type(result) not in (int, float)
            or isinstance(result, bool)
            or not math.isfinite(result)
            or not 0.0 <= result <= 1.0
            or type(pass_at_1) not in (int, float)
            or isinstance(pass_at_1, bool)
            or not math.isfinite(pass_at_1)
            or not 0.0 <= pass_at_1 <= 1.0
        ):
            raise RuntimeError("dense heldout evaluation result identity differs")
        return {
            "policy_version": policy_version,
            "policy_hash": policy_hash,
            "sample_count": expected_samples,
            "result": float(result),
            "pass_at_1": float(pass_at_1),
        }

    def _write_evaluation_summary(self) -> None:
        path = Path(self.args.yeto_rl_eval_summary_path)
        if path.is_symlink() or not path.parent.is_dir() or path.parent.is_symlink():
            raise RuntimeError("dense heldout summary path is unsafe")
        payload = {
            "schema": "yeto-m1-dense-heldout-summary-v1",
            "island_id": self.config.wire.learner_id,
            "dataset_name": self.args.yeto_rl_eval_dataset_name,
            "prompt_count": self.args.yeto_rl_eval_prompt_count,
            "samples_per_prompt": self.args.yeto_rl_eval_samples_per_prompt,
            "expected_policy_versions": list(self.evaluation_policy_versions),
            "complete": set(self.evaluation_results)
            == set(self.evaluation_policy_versions),
            "results": [
                self.evaluation_results[version]
                for version in sorted(self.evaluation_results)
            ],
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            raw = (
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                if handle.write(raw) != len(raw):
                    raise OSError("short dense heldout summary write")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _trajectory_evidence(self, rollout_id: int):
        directory = getattr(self.args, "yeto_rl_trajectory_evidence_dir", None)
        if type(directory) is not str:
            raise RuntimeError(
                "Miles trajectory evidence directory is unavailable"
            )
        from .trajectory_evidence import trajectory_batch_evidence_path

        path = trajectory_batch_evidence_path(directory, rollout_id)
        if not path.exists():
            raise RuntimeError(
                "Miles did not persist this rollout's trajectory evidence"
            )
        evidence = read_trajectory_batch_evidence(path)
        if evidence.rollout_id != rollout_id:
            raise RuntimeError("trajectory evidence rollout identity changed")
        return evidence

    async def _set_rollout_token(self, token: str) -> None:
        info = await self.rollout_manager.get_updatable_engines_and_lock.remote()
        engines: Sequence[Any] = tuple(info.rollout_engines)
        if not engines:
            raise RuntimeError("Miles created no updatable inference worker")
        await asyncio.gather(
            *(engine.update_weight_version.remote(token) for engine in engines)
        )

    async def _set_published_policy_identity(
        self,
        *,
        policy_version: int,
        policy_hash: str,
    ) -> None:
        identity = await self.rollout_manager.set_external_policy_identity.remote(
            policy_version,
            policy_hash,
        )
        if identity != (policy_version, policy_hash):
            raise RuntimeError("rollout process installed the wrong policy identity")
        from .miles import set_current_published_policy_identity

        set_current_published_policy_identity(
            self.args,
            policy_version=policy_version,
            policy_hash=policy_hash,
        )


def _sweep_update_id(
    receipt, anchor: ReferencedPolicyCut, local: ReferencedPolicyCut
) -> str:
    payload = json.dumps(
        {
            "receipt": asdict(receipt),
            "anchor_policy_hash": anchor.policy_hash,
            "local_policy_hash": local.policy_hash,
            "layout_hash": anchor.layout_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(b"yeto-dense-sweep-update-v1\0" + payload).hexdigest()


def _validate_optimizer_apply_transition(
    before: tuple[Any, ...],
    after: tuple[Any, ...],
    *,
    base_policy_version: int,
    target_policy_version: int,
) -> None:
    if not before or len(before) != len(after):
        raise RuntimeError("dense global apply returned incomplete Adam evidence")
    volatile = {"installed_policy_version", "local_step_generation"}
    for old, new in zip(before, after, strict=True):
        old_values = vars(old)
        new_values = vars(new)
        if (
            set(old_values) != set(new_values)
            or old.installed_policy_version != base_policy_version
            or old.local_step_generation != 1
            or new.installed_policy_version != target_policy_version
            or new.local_step_generation != 0
            or any(
                old_values[name] != new_values[name]
                for name in old_values.keys() - volatile
            )
        ):
            raise RuntimeError("dense global apply changed local Adam state")


def create_miles_full_parameter_dense_sync(args) -> MilesFullParameterDenseSync:
    config = getattr(args, "yeto_rl_dense_full_parameter_config", None)
    if not isinstance(config, MilesDenseFullParameterConfig):
        raise TypeError("Miles dense full-parameter configuration is missing")
    return MilesFullParameterDenseSync(args, config)
