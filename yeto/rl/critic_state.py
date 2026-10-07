"""Critic state contract (change ``rl-algo-critic-family``, design D4).

The critic is a second trainable role next to the actor:

* :func:`critic_layout_hash` -- critic layout identity (backbone parameter
  specs + value head shape + ``critic.param_mode`` + the reserved LoRA shape),
  hashed apart from the actor layout;
* :class:`CriticRoundReceipt` -- per round: both layout hashes, param mode,
  initialization source and its content hash, critic weight hash, value
  metrics; :func:`check_critic_layouts` refuses islands whose critic (or actor)
  layouts differ;
* :class:`TwoRoleStrictAvg` -- strict-avg of actor and critic as ONE round:
  actor first, then critic; the round commits only if both averages succeed,
  otherwise both roles keep the previous round (decision 1, 2026-10-06);
* :class:`CriticCheckpointStore` -- critic weights + optimizer state per
  round; restore refuses an actor/critic round mismatch.

Pure torch/stdlib; no Miles imports. The engine-side capture/apply of critic
tensors is the adapter's job.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

CRITIC_LAYOUT_DOMAIN = b"yeto-rl-critic-layout-v1\0"
CRITIC_WEIGHTS_DOMAIN = b"yeto-rl-critic-weights-v1\0"
_HEX = frozenset("0123456789abcdef")


class CriticStateError(RuntimeError):
    pass


class CriticLayoutMismatch(CriticStateError):
    pass


class CriticRoundMismatch(CriticStateError):
    pass


class RoleAverageFailed(CriticStateError):
    """One role's average failed; the round was not committed."""


def _sha(name: str, value: str | None, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or len(value) != 64 or not set(value) <= _HEX:
        raise ValueError(f"{name} must be a lowercase SHA256")


# --------------------------------------------------------------------------
# 4.1 layout + receipt
# --------------------------------------------------------------------------


def critic_layout_hash(
    specs: Sequence[tuple[str, Sequence[int], str]],
    *,
    value_head: str,
    param_mode: str = "full",
    lora: Mapping[str, Any] | None = None,
) -> str:
    """Identity of the critic layout.

    ``specs``: ``(name, shape, dtype)`` of every trainable critic parameter
    (the backbone copied from the actor plus the value head); ``value_head``
    names the value-head parameter, whose shape must end in 1 output.
    ``lora`` is the reserved D9 shape (rank / alpha / target modules) and is
    part of the identity once ``param_mode='lora'`` exists.
    """

    if param_mode not in ("full", "lora"):
        raise ValueError(f"unsupported critic param_mode {param_mode!r}")
    if (param_mode == "lora") != (lora is not None):
        raise ValueError("critic lora shape is given exactly when param_mode='lora'")
    rows = sorted((str(n), [int(d) for d in shape], str(dtype)) for n, shape, dtype in specs)
    names = [r[0] for r in rows]
    if not rows or len(set(names)) != len(names):
        raise ValueError("critic layout needs unique parameter names")
    head = {r[0]: r[1] for r in rows}.get(value_head)
    if head is None or not head or head[0] != 1:
        raise ValueError(f"value head {value_head!r} must be a [1, hidden] parameter of the layout")
    payload = {
        "schema": 1,
        "param_mode": param_mode,
        "lora": None if lora is None else {k: lora[k] for k in sorted(lora)},
        "value_head": value_head,
        "specs": rows,
    }
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(CRITIC_LAYOUT_DOMAIN + data).hexdigest()


def critic_weights_sha256(tensors: Mapping[str, torch.Tensor]) -> str:
    """Content hash of critic weights (name, shape, dtype, fp32 bytes)."""

    digest = hashlib.sha256(CRITIC_WEIGHTS_DOMAIN)
    for name in sorted(tensors):
        value = tensors[name].detach().to("cpu").contiguous()
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        digest.update(value.float().numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CriticRoundReceipt:
    """Critic evidence of one trained round (next to ``LocalStepReceipt``)."""

    rollout_id: int
    actor_layout_hash: str
    critic_layout_hash: str
    critic_param_mode: str
    critic_init: str  # copy_actor_backbone | load
    critic_init_sha256: str | None = None  # warm-up product / loaded checkpoint
    critic_weights_sha256: str | None = None
    value_loss: float | None = None
    explained_variance: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.rollout_id, bool) or not isinstance(self.rollout_id, int) \
                or self.rollout_id < 0:
            raise ValueError("rollout_id must be a non-negative int")
        _sha("actor_layout_hash", self.actor_layout_hash)
        _sha("critic_layout_hash", self.critic_layout_hash)
        _sha("critic_init_sha256", self.critic_init_sha256, optional=True)
        _sha("critic_weights_sha256", self.critic_weights_sha256, optional=True)
        if self.critic_param_mode not in ("full", "lora"):
            raise ValueError(f"unsupported critic param_mode {self.critic_param_mode!r}")
        if self.critic_init not in ("copy_actor_backbone", "load"):
            raise ValueError(f"unsupported critic init {self.critic_init!r}")

    def to_event(self) -> dict[str, Any]:
        return {f"rl/critic/{k}" if k != "rollout_id" else k: v for k, v in asdict(self).items()}


def check_critic_layouts(receipts: Mapping[int, CriticRoundReceipt]) -> None:
    """Outer sync precondition: every island has the same actor and critic layout."""

    for role, attr in (("actor", "actor_layout_hash"), ("critic", "critic_layout_hash"),
                       ("critic param mode", "critic_param_mode")):
        values = {island: getattr(r, attr) for island, r in sorted(receipts.items())}
        if len(set(values.values())) > 1:
            raise CriticLayoutMismatch(
                f"{role} layout differs across islands: {values}; the round is refused"
            )


# --------------------------------------------------------------------------
# 4.2 strict-avg of both roles as one round
# --------------------------------------------------------------------------

Tensors = Mapping[str, torch.Tensor]


def mean_average(states: Sequence[Tensors]) -> dict[str, torch.Tensor]:
    names = set(states[0])
    if any(set(s) != names for s in states):
        raise ValueError("islands disagree on the parameter names")
    return {n: torch.stack([s[n].detach().float() for s in states]).mean(0) for n in names}


@dataclass(frozen=True)
class CommittedRound:
    version: int
    actor: dict[str, torch.Tensor]
    critic: dict[str, torch.Tensor]

    @property
    def actor_sha256(self) -> str:
        return critic_weights_sha256(self.actor)

    @property
    def critic_sha256(self) -> str:
        return critic_weights_sha256(self.critic)


class TwoRoleStrictAvg:
    """Strict-avg of actor + critic: actor first, then critic, commit both or none."""

    def __init__(self, actor: Tensors, critic: Tensors, *,
                 average_actor: Callable[[Sequence[Tensors]], dict] = mean_average,
                 average_critic: Callable[[Sequence[Tensors]], dict] = mean_average) -> None:
        self.committed = CommittedRound(0, dict(actor), dict(critic))
        self.average_actor = average_actor
        self.average_critic = average_critic

    def round(self, local: Mapping[int, tuple[Tensors, Tensors]], *,
              receipts: Mapping[int, CriticRoundReceipt] | None = None) -> CommittedRound:
        """``local``: island -> (actor, critic) trained from the committed round."""

        if receipts is not None:
            check_critic_layouts(receipts)
        islands = sorted(local)
        try:
            actor = self.average_actor([local[i][0] for i in islands])
        except Exception as error:
            raise RoleAverageFailed(f"actor average failed: {error}") from error
        try:
            critic = self.average_critic([local[i][1] for i in islands])
        except Exception as error:
            raise RoleAverageFailed(
                f"critic average failed after the actor average: {error}; round "
                f"{self.committed.version + 1} is not committed, both roles stay at "
                f"round {self.committed.version}"
            ) from error
        self.committed = CommittedRound(self.committed.version + 1, actor, critic)
        return self.committed


# --------------------------------------------------------------------------
# 4.3 checkpoint store
# --------------------------------------------------------------------------


class CriticCheckpointStore:
    """``<root>/critic/round-<n>/``: weights + optimizer state + manifest (last)."""

    MANIFEST = "critic-manifest.json"

    def __init__(self, root: str | os.PathLike) -> None:
        self.root = Path(root) / "critic"

    def save(self, *, round_id: int, weights: Tensors, optimizer: Mapping[str, Any]) -> dict:
        target = self.root / f"round-{round_id}"
        target.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in weights.items()}, target / "weights.pt")
        torch.save(dict(optimizer), target / "optimizer.pt")
        manifest = {
            "schema": 1,
            "round": int(round_id),
            "weights_sha256": critic_weights_sha256(weights),
            "optimizer_sha256": _file_sha256(target / "optimizer.pt"),
        }
        tmp = target / (self.MANIFEST + ".tmp")
        tmp.write_text(json.dumps(manifest, sort_keys=True))
        os.replace(tmp, target / self.MANIFEST)  # manifest committed last
        return manifest

    def restore(self, *, actor_round: int, critic_round: int | None = None) -> tuple[dict, dict]:
        """Weights + optimizer of ``critic_round`` (default: newest); refuses a
        critic round different from the actor's."""

        if critic_round is None:
            rounds = sorted(
                int(p.name.split("-", 1)[1]) for p in self.root.glob("round-*")
                if (p / self.MANIFEST).exists()
            )
            if not rounds:
                raise CriticStateError(f"no committed critic checkpoint under {self.root}")
            critic_round = rounds[-1]
        if critic_round != actor_round:
            raise CriticRoundMismatch(
                f"checkpoint actor round {actor_round} != critic round {critic_round}; "
                "actor and critic must come from the same committed round"
            )
        target = self.root / f"round-{critic_round}"
        manifest = json.loads((target / self.MANIFEST).read_text())
        weights = torch.load(target / "weights.pt", weights_only=True)
        optimizer = torch.load(target / "optimizer.pt", weights_only=True)
        if manifest["round"] != critic_round:
            raise CriticRoundMismatch(f"manifest round {manifest['round']} != {critic_round}")
        if critic_weights_sha256(weights) != manifest["weights_sha256"] or \
                _file_sha256(target / "optimizer.pt") != manifest["optimizer_sha256"]:
            raise CriticStateError(f"critic checkpoint round {critic_round} fails its manifest")
        return weights, optimizer


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
