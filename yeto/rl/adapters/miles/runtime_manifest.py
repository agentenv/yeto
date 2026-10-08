"""Miles runtime-manifest declarations (yeto-framework-decoupling 2.5). Import-light.

The core :mod:`yeto.rl.engine.runtime_manifest` collects and checks a
manifest from a :class:`~yeto.rl.engine.runtime_manifest.RuntimeDescription`;
this module is the Miles one: the fork interfaces it probes, which capability
needs which interface, the version modules, the Miles/SGLang checkouts, the
image build record and commit pins, and the overlay record (kept under the
manifest field ``miles_overlay``). Values are exactly those the core module
hard-coded before the move.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from yeto.rl.engine.runtime_manifest import RuntimeDescription

# interface name -> (module, attribute path). Probed with getattr only.
INTERFACES: Mapping[str, tuple[str, str]] = {
    "run_plugin": ("miles.ray.train.group", "TrainerController.run_plugin"),
    "fork_m1_placement_map": ("miles.ray.placement_group", "parse_placement_map"),
    "fork_m2_start_stop_cells": ("miles.ray.rollout.inference_controller", "InferenceController.start_cells"),
    "fork_m3_router_cordon": ("miles.router.router", "MilesRouter.cordon_worker"),
    "fork_m4_member_publish": (
        "miles.ray.rollout.inference_controller",
        "InferenceController._start_member_update_weights",
    ),
    "fork_m6_rebuild_trainer": ("miles.ray.placement_group", "rebuild_training_models"),
}

# Declared capability -> interfaces it needs. Missing any => refused.
CAPABILITY_REQUIREMENTS: Mapping[str, tuple[str, ...]] = {
    "ports-colocated-serial": ("run_plugin",),
    "ports-partitioned-serial": ("run_plugin",),
    "fixed-partition-standby": ("run_plugin", "fork_m1_placement_map"),
    "RolloutPool.add_engines": ("fork_m2_start_stop_cells", "fork_m3_router_cordon"),
    "RolloutPool.remove_engines": ("fork_m2_start_stop_cells", "fork_m3_router_cordon"),
    "RolloutPool.drain": ("fork_m3_router_cordon",),
    "Publisher.publish(members)": ("fork_m4_member_publish",),
    "Placement.reconfigure": ("fork_m1_placement_map", "fork_m6_rebuild_trainer"),
}

VERSION_MODULES = ("torch", "megatron.core", "torch_memory_saver", "peft", "sglang", "ray")
REQUIRED_VERSIONS = ("torch", "megatron.core", "cuda", "nccl", "torch_memory_saver", "peft")


def image_manifest() -> dict[str, Any]:
    try:
        from yeto.rl import MILES_NEXT_IMAGE_MANIFEST

        return json.loads(Path(MILES_NEXT_IMAGE_MANIFEST).read_text())
    except Exception:
        return {}


def overlay_record() -> dict[str, Any] | None:
    # S13: a critic-family overlay patches ~/miles after the image was built, so
    # neither git HEAD nor the image build manifest describes the running code.
    from yeto.rl import miles_overlay

    return miles_overlay.read_applied_record()


def expected_pins() -> dict[str, str]:
    from yeto.rl import MILES_NEXT_COMMIT, MILES_NEXT_IMAGE, SGLANG_NEXT_COMMIT

    return {"miles": MILES_NEXT_COMMIT, "sglang": SGLANG_NEXT_COMMIT, "image": MILES_NEXT_IMAGE}


MILES_RUNTIME = RuntimeDescription(
    interfaces=INTERFACES,
    capability_requirements=CAPABILITY_REQUIREMENTS,
    version_modules=VERSION_MODULES,
    required_versions=REQUIRED_VERSIONS,
    checkouts=("miles", "sglang"),
    nested_checkouts=("sglang",),  # sglang/python/sglang layout
    overlay_checkout="miles",
    overlay_field="miles_overlay",
    image_manifest=image_manifest,
    overlay_record=overlay_record,
    expected_pins=expected_pins,
)
