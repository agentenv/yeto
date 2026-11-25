"""Backend identity of the verl adapter (decoupling 6.1; rl-verl-backend).

Engine ``verl``, the pinned fork commit, the device family of the card this
island runs on (``nvidia`` on CUDA, ``ascend`` on an NPU) and the hash of the
parameter-name map (``param_names.PARAM_MAP``).  By default (strict) all
of them are in the session contract, so mixed islands are refused.  Under
``--rl-identity relaxed`` (hetero-1114) they are provenance: the contract binds
only ``BackendIdentity.semantic_sha256()``, so verl FSDP2, verl Megatron and
Miles islands on any device family can share a session.
"""

from __future__ import annotations

from yeto.rl.engine.backend_identity import (FLEET_FULL_PARAMETER, METRIC_PPO_KL, METRIC_TRAIN_ROLLOUT_KL,
                                             PER_LEARNER_RELEASE, ROUND_METRICS, BackendIdentity, param_map_sha256)

from .param_names import PARAM_MAP
from .pins import VERL_COMMIT

ENGINE = "verl"
# Default family when the caller names none and no accelerator is visible
# (unit tests, dry runs). A real island resolves it from its own device:
# ``yeto.hw.catalog.device_family()``.
DEVICE_FAMILY = "nvidia"
PARAM_MAP_SHA256 = param_map_sha256(PARAM_MAP)
# The Megatron engine (verl ``model_engine=megatron``) names parameters
# by Megatron-Core module path; Megatron-Bridge maps them to the HF names the
# rollout loads. In strict mode (default) a Megatron island and an FSDP2
# island do not share a session contract; --rl-identity relaxed: provenance.
MEGATRON_PARAM_MAP = {"names": "megatron-core module paths", "export": "megatron-bridge export_hf_weights"}
PARAM_MAP_SHA256_BY_BACKEND = {"fsdp2": PARAM_MAP_SHA256, "megatron": param_map_sha256(MEGATRON_PARAM_MAP)}
# What the verl adapter implements (``yeto.rl.engine.backends.capabilities``): a
# full-parameter fleet trainer (its layout carries no shard identity, so a fleet is all
# verl: rl-fleet-verl-islands) whose buffer trainer releases its own local versions into
# its sub-store trainer<k> (``--rl-release-mode per_learner``, ``buffer_ports`` module doc).
# Not declared: partial experts, per-segment training of compacted trials (one sample is
# one token stream), a release cadence outside per-learner releases.
CAPABILITIES: frozenset[str] = frozenset({FLEET_FULL_PARAMETER, PER_LEARNER_RELEASE})
# The round metrics a verl trainer reports (``backend_identity.reported_metrics``): its
# rounds carry no train-rollout KL and no PPO KL; the dashboard says "not reported by verl".
REPORTED_METRICS: frozenset[str] = ROUND_METRICS - {METRIC_TRAIN_ROLLOUT_KL, METRIC_PPO_KL}


def backend_identity(*, device_family: str | None = None,
                     train_backend: str = "fsdp2", compat_group: str | None = None) -> BackendIdentity:
    """Identity of a verl island.

    ``device_family`` None means "read it from this node's accelerator", so an
    Ascend island declares ``ascend`` without any call site change. The family
    is part of the strict (default) contract: a GPU island and an NPU island
    are refused (task 3.10). Under ``--rl-identity relaxed`` it is provenance
    and they merge (with a warning).
    """
    if train_backend not in PARAM_MAP_SHA256_BY_BACKEND:
        raise ValueError(f"verl train backend {train_backend!r} has no parameter-name map yet")
    from yeto.hw.catalog import device_family as detect_device_family
    from yeto.hw.catalog import island_compat_group

    family = device_family if device_family is not None else detect_device_family()
    return BackendIdentity(ENGINE, VERL_COMMIT, family, PARAM_MAP_SHA256_BY_BACKEND[train_backend],
                           island_compat_group(compat_group))
