"""In-memory CPU fake engine implementing every port, plus fake syncers.

For tests only. ``FakeEngine`` keeps one LoRA state shared by trainer, policy
state and publisher, and logs every port call to ``engine.calls`` so tests can
assert event order and residency (trainer onloaded/offloaded).

``FakeStrictSyncer`` / ``FakeDecoupledSyncer`` speak the ``SyncerClient``
surface the bridges use (drain_updates/drain_pulls/push_fragment/...),
in-process and thread-safe, standing in for the Rust syncer where cargo is
unavailable. Semantics are the simplest configuration of each preset
(outer-lr 1, momentum 0, full quorum), which is what the legacy CPU
integration tests exercise against the real binary.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from yeto.protocol import (
    DTYPE_F32,
    BcastFragment,
    FinalFragment,
    FinalManifest,
    PullRequest,
)
from yeto.rl.contracts import InferencePublicationManifest, LocalStepReceipt
from yeto.rl.core import CanonicalLoraState, canonical_state
from yeto.tensor_io import apply_fragment, pack_fragment, pack_tensor, unpack_fragment

from .algorithm import BOUNDED_NONZERO_STD_FILTER
from .capabilities import EngineCapabilities, ExecutionCapabilities
from .driver import TrainStepMetrics, policy_token
from .ports import (
    GroupMetadata,
    PlacementDescription,
    PublicationResult,
    RolloutBatchHandle,
)
from .trainable_state import TrainableState

MODEL_REVISION = "a" * 40
LORA_CONFIG_HASH = "b" * 64


def fake_capabilities(**overrides) -> EngineCapabilities:
    values = dict(
        engine="fake",
        runtime_fingerprint="sha256:" + "0" * 64,
        parameter_layouts={"lora"},
        placements={"colocated", "fixed-partition"},
        advantage_estimators={"grpo"},
        dynamic_sampling_filters={BOUNDED_NONZERO_STD_FILTER},
        execution_modes={"colocated-serial"},
        # rl-algo-mismatch-correction 6.1: the fake declares every correction
        # mechanism (OPSM per logprob source) for CPU tests.
        corrections={"none", "tis", "opsm", "custom", "mismatch_observe", "icepop",
                     "opsm_trainer", "opsm_rollout", "mis", "mis_mask"},
        features={"mismatch_metrics", "rollout_logprobs_as_old"},
        # rl-algo-loss-variants 5.1: the fake declares the policy-loss variants
        # for CPU tests (the Miles adapter does not, GPU validation pending).
        losses={"policy_loss", "cispo", "sapo", "gmpo"},
        # Mechanism dimensions: the R0 defaults (EngineCapabilities); execution
        # as the Miles adapter declares it (rl-algorithm-capabilities 3.4).
        execution=ExecutionCapabilities(
            critic=False, max_policy_staleness=0, rollout_logprobs=True
        ),
    )
    values.update(overrides)
    # rl-algo-grpo-knobs 8.3: the fake declares what the Miles adapter declares for
    # 1b (grpo_knobs.G1_DECLARED == the 1b part of MILES_DECLARED).
    from yeto.rl.algos.grpo_knobs import merge_declared

    return merge_declared(EngineCapabilities(**values))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class FakeEngine:
    """Shared state behind all fake ports."""

    tensors: dict[str, torch.Tensor]
    step_delta: torch.Tensor | Mapping[str, torch.Tensor] | float = 1.0
    members_ids: tuple[str, ...] = ("rollout-0", "rollout-1")
    groups: int = 2
    samples_per_group: int = 2
    placement_kind: str = "colocated"
    base_model_revision: str = MODEL_REVISION
    lora_config_hash: str = LORA_CONFIG_HASH
    calls: list[tuple] = field(default_factory=list)
    # IR-3: the expected_policy_version each generate() call received, and the
    # rounds whose samples drifted from it (rollout side reports ABORTED).
    expected_policy_versions: list = field(default_factory=list)
    policy_drift_rounds: set[int] = field(default_factory=set)
    zero_grad_rounds: set[int] = field(default_factory=set)
    failed_step_rounds: set[int] = field(default_factory=set)
    unacked_member_rounds: dict[int, str] = field(default_factory=dict)
    stale_token_rounds: set[int] = field(default_factory=set)
    constant_reward_rounds: set[int] = field(default_factory=set)
    # Rounds whose trainer reports a masked-token fraction (design D6);
    # others report none (TrainStepMetrics.masked_fraction stays unset).
    masked_fraction_rounds: dict[int, float] = field(default_factory=dict)
    grad_norm_reported: bool = True
    # Rounds whose optimizer step applies LR 0 (gradients flow, parameters
    # do not move) -- the decayed-to-zero schedule of fix-decoupled-lr-schedule.
    zero_lr_rounds: set[int] = field(default_factory=set)
    lr: float = 1e-5
    # rl-algo-critic-family 3.2: a shared actor/critic trainer (receipt family
    # "ppo"); the critic trains first each round and reports value metrics.
    critic: bool = False

    def __post_init__(self) -> None:
        self.tensors = {k: v.detach().clone().float() for k, v in self.tensors.items()}
        self.moments = {k: torch.ones_like(v) for k, v in self.tensors.items()}
        self.scheduler_step = 0
        self.trainer_resident = True
        self.published: tuple[int, str] | None = None  # (version, tensor hash)
        self.published_tensors: dict[str, torch.Tensor] | None = None
        self._last_metrics: TrainStepMetrics | None = None
        self.rollout = FakeRolloutPool(self)
        self.trainer = FakeTrainerGroup(self)
        self.policy_state = FakePolicyState(self)
        self.publisher = FakePublisher(self)
        self.placement = FakePlacement(self)

    def canonical(self, version: int, tensors=None) -> CanonicalLoraState:
        return canonical_state(
            version,
            {k: v.clone() for k, v in (tensors or self.tensors).items()},
            base_model_revision=self.base_model_revision,
            lora_config_hash=self.lora_config_hash,
        )

    def engine_placement_colocated(self) -> bool:
        return self.placement_kind == "colocated"

    def delta_for(self, name: str) -> torch.Tensor:
        delta = self.step_delta
        if isinstance(delta, Mapping):
            return delta[name]
        if isinstance(delta, torch.Tensor):
            return delta.reshape(self.tensors[name].shape)
        return torch.full_like(self.tensors[name], float(delta))


class FakeRolloutPool:
    def __init__(self, engine: FakeEngine) -> None:
        self.engine = engine

    def generate(
        self, rollout_id: int, *, expected_policy_version: str | None = None
    ) -> RolloutBatchHandle:
        e = self.engine
        e.calls.append(("generate", rollout_id))
        e.expected_policy_versions.append(expected_policy_version)  # IR-3
        if e.engine_placement_colocated() and e.trainer_resident:
            raise RuntimeError("generation while the colocated trainer is resident")
        if e.published is None:
            raise RuntimeError("generation before any publication")
        version, digest = e.published
        token = policy_token(version, digest)
        groups = []
        for g in range(e.groups):
            gtoken = (
                policy_token(max(0, version - 1), digest)
                if rollout_id in e.stale_token_rounds and g == 0
                else token
            )
            std = 0.0 if rollout_id in e.constant_reward_rounds else 0.5
            groups.append(
                GroupMetadata(
                    f"r{rollout_id}-g{g}",
                    tuple(f"r{rollout_id}-g{g}-s{s}" for s in range(e.samples_per_group)),
                    gtoken,
                    0.5,
                    std,
                    3 * e.samples_per_group,
                )
            )
        drifted = (
            rollout_id in e.policy_drift_rounds
            or (expected_policy_version is not None and expected_policy_version != token)
        )
        return RolloutBatchHandle(
            rollout_id, version, digest, tuple(groups), len(groups), 0, payload=object(),
            policy_age_violation=(1 if drifted else None),
        )

    def abort(self) -> None:
        self.engine.calls.append(("abort",))

    def members(self) -> frozenset[str]:
        return frozenset(self.engine.members_ids)


class FakeTrainerGroup:
    def __init__(self, engine: FakeEngine) -> None:
        self.engine = engine
        if not engine.grad_norm_reported:
            self.step_metrics = None  # type: ignore[assignment]

    def onload(self) -> None:
        self.engine.calls.append(("onload",))
        self.engine.trainer_resident = True

    def offload(self) -> None:
        self.engine.calls.append(("offload",))
        self.engine.trainer_resident = False

    def train_step(self, batch: RolloutBatchHandle) -> LocalStepReceipt:
        e = self.engine
        self.critic_metrics = {}
        if e.critic:
            e.calls.append(("critic_train", batch.rollout_id))
            # deterministic, finite stand-ins for Miles' value_loss and the
            # state plugin's explained variance
            self.critic_metrics = {
                "critic/value_loss": 1.0 / (batch.rollout_id + 2),
                "critic/explained_variance": 1.0 - 1.0 / (batch.rollout_id + 2),
            }
        e.calls.append(("train", batch.rollout_id))
        if not e.trainer_resident:
            raise RuntimeError("train step on an offloaded trainer")
        zero = batch.rollout_id in e.zero_grad_rounds
        failed = batch.rollout_id in e.failed_step_rounds
        applied_lr = 0.0 if batch.rollout_id in e.zero_lr_rounds else e.lr
        norm_sq = 0.0
        if not zero and not failed:
            for name in sorted(e.tensors):
                delta = e.delta_for(name)
                norm_sq += float(delta.pow(2).sum())
                if applied_lr == 0.0:
                    continue
                e.tensors[name] = e.tensors[name] + delta
                e.moments[name] = e.moments[name] + delta.abs()
            e.scheduler_step += 1
        e._last_metrics = TrainStepMetrics(
            grad_norm=norm_sq**0.5, loss=0.1, pg_loss=0.1, lr=1e-5,
            train_step=batch.rollout_id,
            applied_lrs=None if failed else (applied_lr,),
            **(
                {"masked_fraction": e.masked_fraction_rounds[batch.rollout_id]}
                if batch.rollout_id in e.masked_fraction_rounds
                else {}
            ),
        )
        tokens = sum(g.token_count for g in batch.groups)
        ids = tuple(s for g in batch.groups for s in g.sample_ids)
        return LocalStepReceipt(
            algorithm="ppo" if e.critic else "grpo",
            learner_id=0,
            learner_generation=0,
            base_policy_version=batch.policy_version,
            base_policy_hash=batch.policy_hash,
            input_batch_hash=_sha("|".join(ids).encode()),
            trajectory_ids=ids,
            trained_tokens=tokens,
            optimizer_steps=0 if failed else 1,
            optimizer_step_succeeded=not failed,
            parameter_layout_hash=e.canonical(0).layout_hash,
        )

    def step_metrics(self) -> TrainStepMetrics:
        return self.engine._last_metrics

    def round_metrics(self) -> dict[str, float]:
        return dict(getattr(self, "critic_metrics", None) or {})


class FakePolicyState:
    def __init__(self, engine: FakeEngine) -> None:
        self.engine = engine
        self.applied: list[tuple[int, str, int]] = []

    def export(self) -> TrainableState:
        self.engine.calls.append(("export",))
        return TrainableState.from_lora(self.engine.canonical(self.engine.scheduler_step))

    def apply(self, state: TrainableState, *, optimizer: str, local_step: int) -> None:
        e = self.engine
        e.calls.append(("apply", state.policy_version, optimizer))
        if not e.trainer_resident:
            raise RuntimeError("apply on an offloaded trainer")
        lora = state.to_lora()
        if set(lora.tensors) != set(e.tensors):
            raise ValueError("layout mismatch")
        for name, value in lora.tensors.items():
            # Copy into the existing storage: never rebind parameters.
            e.tensors[name].copy_(value)
        if optimizer == "reset":
            for moment in e.moments.values():
                moment.zero_()
        elif optimizer != "preserve":
            raise ValueError(optimizer)
        e.scheduler_step = local_step
        self.applied.append((state.policy_version, optimizer, local_step))


class FakePublisher:
    def __init__(self, engine: FakeEngine) -> None:
        self.engine = engine

    def publish(self, state: TrainableState) -> PublicationResult:
        e = self.engine
        e.calls.append(("publish", state.policy_version))
        if not e.trainer_resident:
            raise RuntimeError("publish reads weights from an offloaded trainer")
        digest = state.policy_tensor_hash()
        payload = b"".join(
            state.tensors[n].float().contiguous().numpy().tobytes() for n in state.tensor_names
        )
        members = set(e.members_ids)
        missing = e.unacked_member_rounds.get(state.policy_version)
        if missing is not None:
            members.discard(missing)
        else:
            e.published = (state.policy_version, digest)
            e.published_tensors = {k: v.clone() for k, v in state.tensors.items()}
        manifest = InferencePublicationManifest(
            publication_mode="full",
            base_policy_version=None,
            target_policy_version=state.policy_version,
            target_policy_hash=digest,
            target_manifest_hash=_sha(f"{state.policy_version}:{digest}".encode()),
            payload_hash=_sha(payload),
            payload_bytes=len(payload),
            complete=True,
        )
        return PublicationResult(manifest, frozenset(members))


class FakePlacement:
    def __init__(self, engine: FakeEngine) -> None:
        self.engine = engine

    def describe(self) -> PlacementDescription:
        if self.engine.placement_kind == "colocated":
            return PlacementDescription("colocated", ("gpu0",), ("gpu0",))
        return PlacementDescription("fixed-partition", ("gpu0",), ("gpu1",))


# ---------------------------------------------------------------------------
# Fake syncers
# ---------------------------------------------------------------------------
class _FakeClientBase:
    def __init__(self, syncer, learner_id: int) -> None:
        self.syncer = syncer
        self.learner_id = learner_id
        self.updates: list[BcastFragment] = []
        self.pulls: list[PullRequest] = []
        self.finalizing = threading.Event()
        self.finalization_timeout = 5.0
        self.closed = False
        self.acked: FinalManifest | None = None

    def start(self) -> None:
        self.syncer.connect(self)

    def check_health(self) -> None:
        if self.syncer.error is not None:
            raise RuntimeError(self.syncer.error)

    def drain_updates(self):
        with self.syncer.lock:
            values, self.updates = self.updates, []
        return values

    def drain_pulls(self):
        with self.syncer.lock:
            values, self.pulls = self.pulls, []
        return values

    def wait_for_final_fragments(self, timeout=None):
        if not self.finalizing.wait(timeout or self.finalization_timeout):
            raise TimeoutError("fake syncer never finalized")
        return self.syncer.final_manifest, list(self.syncer.final_fragments)

    def acknowledge_finalization(self, manifest, timeout=None):
        if manifest != self.syncer.final_manifest:
            raise RuntimeError("cannot acknowledge a stale final manifest")
        self.acked = manifest

    def close(self) -> None:
        self.closed = True


class FakeStrictSyncer:
    """Strict-avg over a single fragment with full quorum and outer-lr 1."""

    def __init__(self, layout, *, learners: int, total_steps: int) -> None:
        self.layout = layout
        self.fragment = layout.fragments[0]
        self.learners = learners
        self.total_steps = total_steps
        self.lock = threading.Lock()
        self.clients: dict[int, _FakeClientBase] = {}
        self.params: torch.Tensor | None = None
        self.version: int | None = None
        self.pending: dict[int, torch.Tensor] = {}
        self.final_manifest: FinalManifest | None = None
        self.final_fragments: list[FinalFragment] = []
        self.error: str | None = None
        self.history: list[int] = []

    def client(self, learner_id: int) -> "FakeStrictClient":
        return FakeStrictClient(self, learner_id)

    def connect(self, client) -> None:
        with self.lock:
            self.clients[client.learner_id] = client
            if self.version is not None:
                self._send_state(client)

    def _send_state(self, client) -> None:
        data = pack_tensor(self.params, DTYPE_F32)
        client.updates.append(BcastFragment(0, self.version, data))
        if self.final_manifest is not None:
            client.finalizing.set()
        elif self.version < self.total_steps:
            client.pulls.append(PullRequest(0, self.version + 1, 1))

    def init(self, client, flat: torch.Tensor) -> None:
        with self.lock:
            if self.version is not None:
                return
            self.params, self.version = flat.clone(), 0
            self.history.append(0)
            for c in self.clients.values():
                self._send_state(c)

    def push(self, client, fragment_id, step, attempt, base, data) -> None:
        with self.lock:
            if fragment_id != 0 or base != self.version or step != self.version + 1:
                self.error = f"stale push step={step} base={base} v={self.version}"
                return
            self.pending[client.learner_id] = unpack_fragment(self.fragment, data, DTYPE_F32)
            if len(self.pending) < self.learners:
                return
            delta = torch.stack(list(self.pending.values())).mean(0)
            self.pending.clear()
            self.params = self.params + delta
            self.version = step
            self.history.append(step)
            if step >= self.total_steps:
                self.final_manifest = FinalManifest(step, (step,))
                self.final_fragments = [
                    FinalFragment(0, step, pack_tensor(self.params, DTYPE_F32))
                ]
            for c in self.clients.values():
                self._send_state(c)


class FakeStrictClient(_FakeClientBase):
    def send_init_parts(self, fragment_id, parts) -> bool:
        data = b"".join(bytes(p) for p in parts)
        self.syncer.init(self, unpack_fragment(self.syncer.fragment, data, DTYPE_F32))
        return True

    def push_fragment(self, fragment_id, step, attempt, base, local_step, c_steps, c_tokens, payload):
        self.syncer.push(self, fragment_id, step, attempt, base, payload)


class FakeDecoupledSyncer:
    """Fragment syncer with outer-lr 1, momentum 0, full quorum.

    PULL for global step ``s`` targets fragment ``(s-1) % F`` with base
    version ``max(0, s-F)``; ``pipeline`` steps are outstanding at once.
    """

    def __init__(
        self, layout, template: Mapping[str, torch.Tensor], *, learners: int,
        total_steps: int, pipeline: int,
    ) -> None:
        self.layout = layout
        self._init_parts: dict[int, bytes] = {}
        self.learners = learners
        self.total_steps = total_steps
        self.pipeline = pipeline
        self.lock = threading.Lock()
        self.clients: dict[int, _FakeClientBase] = {}
        self.params = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in template.items()}
        self.versions: list[int] = [0] * layout.num_fragments
        self.pending: dict[int, dict[int, torch.Tensor]] = {}
        self.next_pull = 1
        self.final_manifest: FinalManifest | None = None
        self.final_fragments: list[FinalFragment] = []
        self.error: str | None = None

    def client(self, learner_id: int) -> "FakeDecoupledClient":
        return FakeDecoupledClient(self, learner_id)

    def connect(self, client) -> None:
        with self.lock:
            self.clients[client.learner_id] = client

    def _flat(self, fragment_id: int) -> torch.Tensor:
        from yeto.tensor_io import fragment_flat

        return fragment_flat(self.layout.fragments[fragment_id], self.params).clone()

    def init(self, fragment_id: int, data: bytes) -> None:
        with self.lock:
            self._init_parts[fragment_id] = data
            if len(self._init_parts) < self.layout.num_fragments:
                return
            for fid, payload in sorted(self._init_parts.items()):
                frag = self.layout.fragments[fid]
                apply_fragment(frag, unpack_fragment(frag, payload, DTYPE_F32), self.params)
            for fid in range(self.layout.num_fragments):
                self._bcast(fid)
            while self.next_pull <= min(self.pipeline, self.total_steps):
                self._pull()

    def _bcast(self, fid: int) -> None:
        data = pack_fragment(self.layout.fragments[fid], self.params, DTYPE_F32)
        for c in self.clients.values():
            c.updates.append(BcastFragment(fid, self.versions[fid], data))

    def _pull(self) -> None:
        step = self.next_pull
        self.next_pull += 1
        fid = (step - 1) % self.layout.num_fragments
        for c in self.clients.values():
            c.pulls.append(PullRequest(fid, step, 1))

    def push(self, client, fid, step, base, data) -> None:
        with self.lock:
            if base != self.versions[fid] or (step - 1) % self.layout.num_fragments != fid:
                self.error = f"bad fragment push fid={fid} step={step} base={base}"
                return
            frag = self.layout.fragments[fid]
            self.pending.setdefault(step, {})[client.learner_id] = unpack_fragment(
                frag, data, DTYPE_F32
            )
            if len(self.pending[step]) < self.learners:
                return
            delta = torch.stack(list(self.pending.pop(step).values())).mean(0)
            apply_fragment(frag, self._flat(fid) + delta, self.params)
            self.versions[fid] = step
            if step >= self.total_steps:
                self.final_manifest = FinalManifest(step, tuple(self.versions))
                self.final_fragments = [
                    FinalFragment(f, self.versions[f], pack_fragment(
                        self.layout.fragments[f], self.params, DTYPE_F32))
                    for f in range(self.layout.num_fragments)
                ]
                for c in self.clients.values():
                    c.finalizing.set()
                return
            self._bcast(fid)
            if self.next_pull <= self.total_steps:
                self._pull()


class FakeDecoupledClient(_FakeClientBase):
    def send_init(self, fragment_id, payload) -> None:
        self.syncer.init(fragment_id, payload)

    def push_fragment(self, fid, step, attempt, base, local_step, c_steps, c_tokens, payload):
        self.syncer.push(self, fid, step, base, payload)
