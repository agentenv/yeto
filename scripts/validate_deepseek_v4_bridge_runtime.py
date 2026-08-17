"""GPU runtime validation for Yeto's pinned DeepSeek-V4 bridge.

This intentionally builds a small decoder with the production V4 attention
geometry.  It validates the complete 43-layer PEFT contract on meta tensors,
then validates real Megatron LoRA injection and collective conversion on the
small decoder.  The optional PP2 mode has two deliberately narrow components:
two GPUs prove the stage-local attention mapping, while 16 or 32 ranks reproduce
the production TP8 x PP2 x EP8 expert-full task ownership at DP1 or DP2 on only
four decoder layers.  The latter fails before rollout if Bridge conversion
tasks do not cover every locally flagged expert parameter.  Optional DDP and
real-optimizer flags reproduce the production replicated-Adam FP32-master
lifecycle without loading a checkpoint, starting TMS, or launching rollout.
The sleep-process-group gate additionally isolates Miles' pre-TMS process-group
teardown.  Run it inside the pinned Miles image under ``torchrun``.
``--task-bridge-model`` keeps construction on ``--model`` while resolving task
mappings through the same dual-source actor path as production.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from argparse import Namespace
from collections.abc import Mapping

import torch

_CANONICAL_PREFIX = "base_model.model."
_LAYER = re.compile(r"\.layers\.(\d+)\.")
_MEGATRON_LAYER = re.compile(r"^decoder\.layers\.(\d+)\.")
_MEGATRON_WILDCARD_LAYER = re.compile(
    r"^(?:.*\.)?decoder\.layers\.\*\..+$"
)


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--pipeline-parallel", type=int, choices=(1, 2), default=1)
    parser.add_argument("--expert-full-count", type=int, choices=(0, 16), default=0)
    parser.add_argument("--expert-full-lr", type=float, default=1e-6)
    parser.add_argument("--wrap-with-ddp", action="store_true")
    parser.add_argument("--build-real-optimizer", action="store_true")
    parser.add_argument("--validate-sleep-process-groups", action="store_true")
    parser.add_argument("--task-bridge-model")
    parser.add_argument("--load-base-weights", action="store_true")
    parser.add_argument("--forward-backward", action="store_true")
    parser.add_argument("--expect-clone-split", action="store_true")
    return parser.parse_args(argv)


def _canonical_name(raw_name: str) -> str:
    return (
        raw_name
        if raw_name.startswith(_CANONICAL_PREFIX)
        else _CANONICAL_PREFIX + raw_name
    )


def _layer_number(name: str, pattern: re.Pattern[str]) -> int | None:
    match = pattern.search(name)
    return int(match.group(1)) if match is not None else None


def _parallel_sizes(
    world_size: int,
    pipeline_parallel: int,
    *,
    expert_full_count: int = 0,
) -> tuple[int, int, int]:
    """Return the deliberately narrow TP/PP/EP runtime topology."""

    if world_size <= 0:
        raise ValueError("runtime world size must be positive")
    if pipeline_parallel == 1:
        if expert_full_count:
            raise ValueError("expert-full component validation requires PP2")
        return world_size, 1, world_size
    if pipeline_parallel == 2 and world_size == 2 and not expert_full_count:
        return 1, 2, 1
    if (
        pipeline_parallel == 2
        and world_size in (16, 32)
        and expert_full_count == 16
    ):
        return 8, 2, 8
    if expert_full_count:
        raise ValueError(
            "PP2 expert-full validation requires exactly 16 or 32 ranks "
            "(TP8 x PP2 x EP8 at DP1 or DP2)"
        )
    raise ValueError(
        "PP2 bridge validation requires exactly two ranks (TP1 x PP2 x EP1)"
    )


def _expected_router_modes(
    *,
    layers: int,
    pipeline_parallel: int,
    tensor_parallel: int,
    data_parallel: int = 1,
) -> list[tuple[bool, bool]]:
    """Return router modes at the scope used by the runtime assertion.

    PP1 validates the current rank's local layers.  PP2 gathers router
    records across the full world, so each global layer occurs once for every
    TP/EP coordinate and DP replica.
    """

    if layers < 4:
        raise ValueError("router validation requires at least four layers")
    modes = [(True, False)] * 3 + [(False, True)] * (layers - 3)
    if pipeline_parallel == 1:
        return modes
    if pipeline_parallel != 2 or tensor_parallel <= 0:
        raise ValueError("invalid PP/TP router validation topology")
    if data_parallel <= 0:
        raise ValueError("data parallel size must be positive")
    return modes * tensor_parallel * data_parallel


def _pipeline_layer_counts(
    layers: int,
    pipeline_parallel: int,
) -> tuple[int, ...]:
    if layers <= 0 or pipeline_parallel <= 0:
        raise ValueError("layer and pipeline counts must be positive")
    if pipeline_parallel == 1:
        return (layers,)
    if pipeline_parallel != 2:
        raise ValueError("runtime validator supports only PP1 or PP2")
    first = (layers + 1) // 2
    return first, layers - first


def _runtime_targets(
    full_targets: list[str],
    *,
    layers: int,
    pipeline_parallel: int,
) -> list[str]:
    if pipeline_parallel == 1:
        return [
            target
            for target in full_targets
            if (_layer_number(target, _MEGATRON_LAYER) or 0) < layers
        ]
    invalid = [
        target
        for target in full_targets
        if _MEGATRON_WILDCARD_LAYER.fullmatch(target) is None
    ]
    if invalid:
        raise ValueError(
            f"PP2 adapter targets are not pipeline-local wildcards: {invalid[:2]}"
        )
    return list(full_targets)


def _validate_expert_full_component_environment(args: argparse.Namespace) -> None:
    if not args.expert_full_count:
        return
    if args.pipeline_parallel != 2 or args.experts != 288:
        raise ValueError(
            "expert-full component requires --pipeline-parallel 2 --experts 288"
        )
    required = {
        "YETO_DSV4_EXPERT_CLONE": "1",
        "YETO_DSV4_EXPERT_FULL": "1",
        "YETO_DSV4_EXPERT_FULL_COUNT": str(args.expert_full_count),
        "NVTE_GROUPED_LINEAR_SINGLE_PARAM": "0",
    }
    mismatched = {
        name: (os.environ.get(name), expected)
        for name, expected in required.items()
        if os.environ.get(name) != expected
    }
    try:
        actual_expert_lr = float(os.environ.get("YETO_DSV4_EXPERT_FULL_LR", ""))
    except ValueError:
        actual_expert_lr = None
    if actual_expert_lr != args.expert_full_lr:
        mismatched["YETO_DSV4_EXPERT_FULL_LR"] = (
            os.environ.get("YETO_DSV4_EXPERT_FULL_LR"),
            str(args.expert_full_lr),
        )
    if mismatched:
        raise ValueError(
            "expert-full component environment mismatch: "
            f"{mismatched}"
        )


def _adapter_sides(model_bridge, models):
    tasks_by_base = model_bridge.build_adapter_conversion_tasks(models)
    sides = []
    for base_name in sorted(tasks_by_base):
        tasks = sorted(
            tasks_by_base[base_name],
            key=lambda task: task.adapter_key or "",
        )
        for task in tasks:
            for side in (task.linear_in_task, task.linear_out_task):
                hf_param = side.mapping.hf_param
                if isinstance(hf_param, str):
                    raw_name = hf_param
                elif isinstance(hf_param, dict) and len(set(hf_param.values())) == 1:
                    raw_name = next(iter(hf_param.values()))
                else:
                    raise AssertionError(
                        f"ambiguous adapter mapping for {side.param_name}: {hf_param}"
                    )
                sides.append((_canonical_name(raw_name), side))
    sides.sort(key=lambda item: item[0])
    names = [name for name, _side in sides]
    assert names and len(names) == len(set(names)), "empty or duplicate adapter sides"
    return sides


def _export(bridge, models) -> dict[str, torch.Tensor]:
    exported = {}
    for raw_name, weight, _megatron_name in bridge.export_adapter_weights(
        models,
        cpu=True,
        show_progress=False,
    ):
        name = _canonical_name(raw_name)
        value = weight.detach().to(dtype=torch.float32).contiguous()
        previous = exported.get(name)
        assert previous is None or torch.equal(previous, value), name
        exported[name] = value
    return exported


def _deterministic_value(shape: tuple[int, ...], ordinal: int) -> torch.Tensor:
    count = 1
    for dimension in shape:
        count *= dimension
    # Fractional f32 values deliberately do not round-trip through bf16.  The
    # exact assertion therefore proves that policy conversion uses optimizer
    # masters rather than the lower-precision model copies.
    values = torch.arange(count, device="cuda", dtype=torch.float32)
    return ((values + ordinal) / 1009.0 - 0.25).reshape(shape)


def _pp2_attention_round_trip(
    bridge,
    models,
    expected_specs,
    *,
    pipeline_layer_counts: tuple[int, int],
    require_existing_masters: bool = False,
) -> dict[str, object]:
    """Exercise the production attention mapping/export path on real PP ranks."""

    import torch.distributed as dist
    from megatron.core import parallel_state

    from yeto.rl import deepseek_v4_expert_full_runtime as expert_runtime

    specs = tuple(sorted(expected_specs.values(), key=lambda spec: spec.name))
    actor = Namespace(
        args=Namespace(yeto_rl_expected_specs=specs),
        model=models,
    )
    actor._yeto_expert_full_bridge = bridge
    sides = expert_runtime._attention_sides(actor)
    assert set(sides) == set(expected_specs)

    trainable_parameters = [
        parameter
        for model in models
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    for parameter in trainable_parameters:
        main_param = getattr(parameter, "main_param", None)
        if main_param is None:
            assert not require_existing_masters, (
                "real optimizer did not attach an FP32 main parameter"
            )
            parameter.main_param = torch.nn.Parameter(
                parameter.detach().to(dtype=torch.float32).clone(),
                requires_grad=True,
            )
        else:
            assert main_param.dtype == torch.float32
            assert main_param.numel() == parameter.numel()

    local_sides = {
        name: side for name, side in sides.items() if side.param_weight is not None
    }
    assert 0 < len(local_sides) < len(sides), (
        len(local_sides),
        len(sides),
    )
    mapped = {id(side.param_weight) for side in local_sides.values()}
    trainable = {
        id(parameter)
        for parameter in trainable_parameters
        if not getattr(parameter, "_yeto_expert_full", False)
    }
    assert mapped == trainable, (len(mapped), len(trainable))

    parsed_layers = {_layer_number(name, _LAYER) for name in local_sides}
    assert None not in parsed_layers
    local_layers = sorted(int(layer) for layer in parsed_layers)
    pipeline_rank = int(parallel_state.get_pipeline_model_parallel_rank())
    layer_records = [None] * dist.get_world_size()
    dist.all_gather_object(
        layer_records,
        {"pipeline_rank": pipeline_rank, "layers": local_layers},
    )
    first_layers, last_layers = pipeline_layer_counts
    expected_owner_layers = {
        0: list(range(first_layers)),
        1: list(range(first_layers, first_layers + last_layers)),
    }
    assert all(
        record["layers"] == expected_owner_layers[record["pipeline_rank"]]
        for record in layer_records
    ), layer_records

    parameter_names = {
        id(parameter): name
        for model in models
        for name, parameter in model.named_parameters()
    }
    stage_one_local_zero = any(
        _layer_number(name, _LAYER) == first_layers
        and re.search(
            r"(?:^|\.)decoder\.layers\.0\.",
            parameter_names[id(side.param_weight)],
        )
        is not None
        for name, side in local_sides.items()
    )
    stage_one_checks = [None] * dist.get_world_size()
    dist.all_gather_object(
        stage_one_checks,
        {"pipeline_rank": pipeline_rank, "global_layer_two": stage_one_local_zero},
    )
    assert all(
        record["global_layer_two"] == (record["pipeline_rank"] == 1)
        for record in stage_one_checks
    ), stage_one_checks

    expected_values = {}
    with torch.no_grad(), expert_runtime._attention_masters_as_model_parameters(
        sides
    ):
        for ordinal, spec in enumerate(specs):
            value = _deterministic_value(spec.shape, ordinal)
            side = sides[spec.name]
            if side.param_weight is not None:
                converted = side.mapping.hf_to_megatron(
                    value,
                    side.megatron_module,
                )
                assert converted is not None
                assert converted.numel() == side.param_weight.numel(), (
                    spec.name,
                    tuple(converted.shape),
                    tuple(side.param_weight.shape),
                )
                master = expert_runtime._optimizer_master_view(side.param_weight)
                master.copy_(
                    converted.reshape(side.param_weight.shape).to(
                        dtype=torch.float32
                    )
                )
            expected_values[spec.name] = value.cpu()

    with torch.no_grad():
        for side in local_sides.values():
            master = expert_runtime._optimizer_master_view(side.param_weight)
            side.param_weight.copy_(master.to(dtype=side.param_weight.dtype))

    rank = dist.get_rank()
    round_trip = expert_runtime._export_attention(actor, retain=rank == 0)
    if rank == 0:
        assert set(round_trip) == set(expected_values)
        for name, expected in expected_values.items():
            assert torch.equal(round_trip[name], expected), name
    else:
        assert round_trip == {}

    return {
        "owner_layers": expected_owner_layers,
        "stage_one_local_zero_global_layer": first_layers,
        "local_sides": len(local_sides),
        "global_sides": len(sides),
    }


def _pp2_expert_task_coverage(
    bridge,
    models,
    expected_specs,
    *,
    pipeline_layer_counts: tuple[int, int],
    pre_wrap_parameter_ids: set[int] | None = None,
    model_path: str | None = None,
    task_bridge_model: str | None = None,
    actor_and_record=None,
) -> dict[str, object]:
    """Prove real Bridge tasks cover every flagged expert on every PP stage."""

    import torch.distributed as dist

    from yeto.rl import deepseek_v4_expert_full_runtime as expert_runtime

    if actor_and_record is None:
        actor, local_record = _expert_task_coverage_snapshot(
            bridge,
            models,
            expected_specs,
            pre_wrap_parameter_ids=pre_wrap_parameter_ids,
            model_path=model_path,
            task_bridge_model=task_bridge_model,
        )
    else:
        actor, local_record = actor_and_record
    coverage = [None] * dist.get_world_size()
    dist.all_gather_object(coverage, local_record)
    failures = _expert_task_coverage_failures(coverage)
    assert not failures, (
        "expert-full conversion-task coverage mismatch by PP stage: "
        f"{[_compact_coverage_failure(record) for record in failures]}"
    )

    views = expert_runtime._expert_views(actor)
    local_view_names = sorted(views)
    gathered_names = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_names, local_view_names)
    expected_names = {
        spec.name
        for spec in expected_specs.values()
        if expert_runtime.selected_expert_hf_name(
            spec.name,
            expert_count=16,
        )
    }
    owners = {
        name: [rank for rank, names in enumerate(gathered_names) if name in names]
        for name in expected_names
    }
    replicas = len(coverage) // (2 * 8)
    assert replicas in (1, 2) and len(coverage) == 2 * 8 * replicas
    invalid_owners = {
        name: ranks for name, ranks in owners.items() if len(ranks) != replicas
    }
    assert not invalid_owners, (
        "expert-full canonical views do not have one owner per DP replica: "
        f"{dict(sorted(invalid_owners.items())[:4])}"
    )

    for record, names in zip(coverage, gathered_names, strict=True):
        record["canonical_views"] = len(names)
    expected_ep_balance = [4, 4, 4, 4, 0, 0, 0, 0]
    for pipeline_rank, stage_layers in enumerate(pipeline_layer_counts):
        stage = [
            record
            for record in coverage
            if record["pipeline_rank"] == pipeline_rank
        ]
        assert len(stage) == 8 * replicas
        parameters_per_expert = stage_layers * 2
        views_per_expert = stage_layers * 3
        for expert_rank, expected_balance in enumerate(expected_ep_balance):
            rank_records = [
                record for record in stage if record["expert_rank"] == expert_rank
            ]
            assert len(rank_records) == replicas
            assert all(
                record["flagged_parameters"] // parameters_per_expert
                == expected_balance
                for record in rank_records
            )
            assert all(
                record["canonical_views"] // views_per_expert
                == expected_balance
                for record in rank_records
            )

    return {
        "expected_ep_expert_balance": expected_ep_balance,
        "data_parallel_replicas": replicas,
        "canonical_expert_tensors": len(expected_names),
        "rank_coverage": coverage,
    }


def _pp2_expert_transport_shards(
    actor,
    expected_specs,
    *,
    model_config,
    clone_contract,
) -> dict[str, object]:
    """Prove canonical-owner sharding without returning tensor payloads."""

    import torch.distributed as dist

    from yeto.rl import deepseek_v4_expert_full_runtime as expert_runtime

    rank = dist.get_rank()
    fragment = expert_runtime._export_experts(
        actor,
        retain=True,
        canonical_sources_only=True,
    )
    local_meta = {
        "rank": rank,
        "names": tuple(sorted(fragment)),
        "tensor_bytes": sum(
            expert_runtime._tensor_bytes(name, tensor)
            for name, tensor in fragment.items()
        ),
    }
    shard_meta = [None] * dist.get_world_size()
    dist.all_gather_object(shard_meta, local_meta)
    expected_names = {
        spec.name
        for spec in expected_specs.values()
        if expert_runtime.selected_expert_hf_name(
            spec.name,
            expert_count=16,
        )
    }
    counts = {name: 0 for name in expected_names}
    for record in shard_meta:
        for name in record["names"]:
            assert name in counts, f"canonical export returned unexpected {name!r}"
            counts[name] += 1
    invalid = {name: count for name, count in counts.items() if count != 1}
    assert not invalid, (
        "canonical transport shards do not form an exact one-copy union: "
        f"{dict(sorted(invalid.items())[:4])}"
    )

    # The tiny real topology proves owner selection and DP replica comparison.
    # Full16 production geometry is derived independently from the canonical
    # 43-layer specs so this gate never allocates or returns a 64.5 GiB object.
    production_bytes, expected_shard_bytes = _full16_transport_geometry(
        model_config,
        clone_contract,
    )
    expected_total_bytes = 69_256_347_648
    assert production_bytes == expected_total_bytes, production_bytes
    assert expected_shard_bytes == (8_858_370_048, 8_455_716_864)

    return {
        "canonical_shards": [
            {
                "rank": record["rank"],
                "tensors": len(record["names"]),
                "tensor_bytes": record["tensor_bytes"],
            }
            for record in shard_meta
        ],
        "one_copy_union": len(counts),
        "full16_total_bytes": production_bytes,
        "full16_shard_bytes_by_pp_stage": expected_shard_bytes,
    }


def _full16_transport_geometry(
    model_config,
    clone_contract,
) -> tuple[int, tuple[int, int]]:
    """Return the exact full16 bytes and maximum owner shard by PP stage."""

    import torch

    from yeto.rl.deepseek_v4_expert_full import expert_full_specs

    specs = expert_full_specs(
        model_config,
        expert_count=16,
        expected_selection_sha256=clone_contract.selection_sha256,
        expected_selection_contract_sha256=(
            clone_contract.selection_contract_sha256
        ),
    )
    total = sum(int(torch.tensor(spec.shape).prod().item()) * 4 for spec in specs)
    bytes_per_layer_expert = total // (43 * 16)
    if bytes_per_layer_expert * 43 * 16 != total:
        raise AssertionError("full16 expert tensor geometry is not uniform")
    return total, tuple(
        layers * 4 * bytes_per_layer_expert for layers in (22, 21)
    )


def _parallel_coordinates(parallel_state) -> dict[str, int]:
    """Return only topology coordinates; never tensor or model data."""

    return {
        "pipeline_rank": int(
            parallel_state.get_pipeline_model_parallel_rank()
        ),
        "pipeline_world_size": int(
            parallel_state.get_pipeline_model_parallel_world_size()
        ),
        "tensor_rank": int(parallel_state.get_tensor_model_parallel_rank()),
        "tensor_world_size": int(
            parallel_state.get_tensor_model_parallel_world_size()
        ),
        "expert_rank": int(parallel_state.get_expert_model_parallel_rank()),
        "expert_world_size": int(
            parallel_state.get_expert_model_parallel_world_size()
        ),
    }


def _expert_task_coverage_snapshot(
    bridge,
    models,
    expected_specs,
    *,
    pre_wrap_parameter_ids: set[int] | None = None,
    model_path: str | None = None,
    task_bridge_model: str | None = None,
):
    """Build one rank's name-only expert task coverage snapshot."""

    import torch.distributed as dist
    from megatron.core import parallel_state

    from yeto.rl import deepseek_v4_expert_full_runtime as expert_runtime

    specs = tuple(sorted(expected_specs.values(), key=lambda spec: spec.name))
    actor = _expert_coverage_actor(
        bridge,
        models,
        specs,
        model_path=model_path,
        task_bridge_model=task_bridge_model,
    )

    task_bridge = expert_runtime._actor_bridge(actor)
    all_tasks = task_bridge.get_conversion_tasks(models)
    tasks = expert_runtime.filter_selected_expert_tasks(
        all_tasks,
        expert_count=16,
    )
    local_record = _local_expert_task_coverage(
        models,
        tasks,
        all_tasks=all_tasks,
    )
    flagged_parameters = [
        parameter
        for model in models
        for parameter in model.parameters()
        if getattr(parameter, "_yeto_expert_full", False)
    ]
    local_record["optimizer_master_parameters"] = sum(
        getattr(parameter, "main_param", None) is not None
        and parameter.main_param.dtype == torch.float32
        and parameter.main_param.numel() == parameter.numel()
        for parameter in flagged_parameters
    )
    local_record["sharded_optimizer_master_parameters"] = sum(
        bool(getattr(parameter, "main_param_sharded", False))
        for parameter in flagged_parameters
    )
    current_parameter_ids = set(_flagged_expert_parameters(models))
    if pre_wrap_parameter_ids is not None:
        local_record["pre_wrap_flagged_parameters"] = len(
            pre_wrap_parameter_ids
        )
        local_record["ddp_preserved_parameter_ids"] = (
            current_parameter_ids == pre_wrap_parameter_ids
        )
    local_record.update(
        {
            "world_rank": dist.get_rank(),
            **_parallel_coordinates(parallel_state),
        }
    )
    return actor, local_record


def _expert_coverage_actor(
    provider_bridge,
    models,
    specs,
    *,
    model_path: str | None,
    task_bridge_model: str | None,
):
    """Build the minimal actor inputs used by production ``_actor_bridge``."""

    if task_bridge_model is not None and model_path is None:
        raise ValueError("task bridge validation requires the provider model path")
    actor = Namespace(
        args=Namespace(
            yeto_rl_expected_specs=specs,
            hf_checkpoint=model_path,
            ref_load=task_bridge_model,
            yeto_rl_trust_remote_code=True,
        ),
        model=models,
    )
    if task_bridge_model is None:
        actor._yeto_expert_full_bridge = provider_bridge
    return actor


def _expert_task_coverage_failures(coverage) -> list[dict[str, object]]:
    return [
        record
        for record in coverage
        if record.get("snapshot_error") is not None
        or record.get("expert_views_error") is not None
        or record.get("missing_parameters")
        or record.get("unexpected_task_parameters", 0)
        or record.get("flagged_parameters") != record.get("task_parameters")
        or record.get("optimizer_master_parameters")
        != record.get("flagged_parameters")
        or record.get("sharded_optimizer_master_parameters", 0)
        or record.get("ddp_preserved_parameter_ids") is False
    ]


def _compact_coverage_failure(record) -> dict[str, object]:
    """Bound failure output while retaining names needed for diagnosis."""

    missing = list(record.get("missing_parameters", ()))[:4]
    mappings = record.get("missing_task_mappings", {})
    return {
        key: record.get(key)
        for key in (
            "world_rank",
            "pipeline_rank",
            "pipeline_world_size",
            "tensor_rank",
            "tensor_world_size",
            "expert_rank",
            "expert_world_size",
            "flagged_parameters",
            "selected_tasks",
            "task_parameters",
            "optimizer_master_parameters",
            "sharded_optimizer_master_parameters",
            "unexpected_task_parameters",
            "snapshot_error",
            "expert_views_error",
        )
    } | {
        "missing_parameters": missing,
        "missing_task_mappings": {
            name: list(mappings.get(name, ()))[:2] for name in missing
        },
    }


def _sleep_process_group_task_coverage(
    bridge,
    models,
    expected_specs,
    *,
    pipeline_layer_counts: tuple[int, int],
    pre_wrap_parameter_ids: set[int] | None = None,
    model_path: str | None = None,
    task_bridge_model: str | None = None,
) -> dict[str, object]:
    """Validate expert coverage during Miles' destroyed-group sleep state.

    This deliberately excludes checkpoint reload and TMS.  It isolates the
    first structural transition in ``MegatronTrainRayActor.sleep``: destroying
    the reloadable model-parallel groups while leaving the default world group
    alive.  Groups are always reloaded before this helper returns or raises.
    """

    import torch.distributed as dist
    from megatron.core import parallel_state
    from miles.utils.reloadable_process_group import (
        destroy_process_groups,
        reload_process_groups,
    )

    from yeto.rl import deepseek_v4_expert_full_runtime as expert_runtime

    before = _parallel_coordinates(parallel_state)
    local_record = None
    try:
        destroy_process_groups()
        try:
            actor, local_record = _expert_task_coverage_snapshot(
                bridge,
                models,
                expected_specs,
                pre_wrap_parameter_ids=pre_wrap_parameter_ids,
                model_path=model_path,
                task_bridge_model=task_bridge_model,
            )
            try:
                expert_runtime._expert_views(actor)
            except RuntimeError as error:
                expected = (
                    "expert-full conversion tasks do not cover local parameters"
                )
                local_record["expert_views_error"] = (
                    "coverage_guard" if expected in str(error) else type(error).__name__
                )
        # Every rank must reach the default-world diagnostic gather and group
        # reload even if task construction itself fails on just one rank.
        except Exception as error:  # noqa: BLE001
            local_record = {
                "world_rank": dist.get_rank(),
                **_parallel_coordinates(parallel_state),
                "snapshot_error": type(error).__name__,
            }
        destroyed_coverage = [None] * dist.get_world_size()
        dist.all_gather_object(destroyed_coverage, local_record)
    finally:
        reload_process_groups()

    after = _parallel_coordinates(parallel_state)
    recovery = _pp2_expert_task_coverage(
        bridge,
        models,
        expected_specs,
        pipeline_layer_counts=pipeline_layer_counts,
        pre_wrap_parameter_ids=pre_wrap_parameter_ids,
        model_path=model_path,
        task_bridge_model=task_bridge_model,
    )
    failures = _expert_task_coverage_failures(destroyed_coverage)
    assert not failures, (
        "expert-full conversion-task coverage changed while Miles model-parallel "
        "groups were destroyed: "
        f"{[_compact_coverage_failure(record) for record in failures]}"
    )
    return {
        "coordinates_before_destroy": before,
        "coordinates_after_reload": after,
        "destroyed_rank_coverage": destroyed_coverage,
        "reloaded_task_coverage": recovery,
    }


def _task_hf_names(task) -> tuple[str, ...]:
    value = getattr(getattr(task, "mapping", None), "hf_param", None)
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping) and all(
        isinstance(item, str) for item in value.values()
    ):
        return tuple(sorted(value.values()))
    return ()


def _local_expert_task_coverage(
    models,
    tasks,
    *,
    all_tasks=None,
) -> dict[str, object]:
    """Compare flagged local parameters with non-remote Bridge task owners."""

    local_parameters = _flagged_expert_parameters(models)

    task_parameters = {
        id(parameter): parameter
        for task in tasks
        if (parameter := getattr(task, "param_weight", None)) is not None
    }
    missing = sorted(
        local_parameters[parameter_id]
        for parameter_id in set(local_parameters) - set(task_parameters)
    )
    all_tasks = tasks if all_tasks is None else all_tasks
    missing_ids = set(local_parameters) - set(task_parameters)
    missing_task_mappings = {
        local_parameters[parameter_id]: sorted(
            {
                name
                for task in all_tasks
                if (parameter := getattr(task, "param_weight", None)) is not None
                and id(parameter) == parameter_id
                for name in _task_hf_names(task)
            }
        )
        for parameter_id in missing_ids
    }
    unexpected = len(set(task_parameters) - set(local_parameters))
    return {
        "flagged_parameters": len(local_parameters),
        "selected_tasks": len(tasks),
        "task_parameters": len(task_parameters),
        "missing_parameters": missing,
        "missing_task_mappings": missing_task_mappings,
        "unexpected_task_parameters": unexpected,
    }


def _flagged_expert_parameters(models) -> dict[int, str]:
    local_parameters = {}
    for model in models:
        for name, parameter in model.named_parameters():
            if getattr(parameter, "_yeto_expert_full", False):
                local_parameters[id(parameter)] = name
    return local_parameters


def _forward_backward_step(models, seq_length: int, vocab_size: int) -> float:
    for model in models:
        model.train()
    trainable = [
        parameter
        for model in models
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=1.0e-4)
    tokens = (
        torch.arange(seq_length, device="cuda", dtype=torch.long)
        .remainder(vocab_size)
        .unsqueeze(0)
    )
    positions = torch.arange(seq_length, device="cuda", dtype=torch.long).unsqueeze(0)
    output = models[0](
        input_ids=tokens,
        position_ids=positions,
        attention_mask=None,
    )
    if isinstance(output, tuple):
        output = output[0]
    loss = output.float().square().mean()
    assert torch.isfinite(loss).item(), f"non-finite validation loss: {loss.item()}"
    loss.backward()
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all().item()
        and parameter.grad.abs().max().item() > 0
        for parameter in trainable
    ), "no finite non-zero LoRA gradient"
    before = [parameter.detach().clone() for parameter in trainable]
    optimizer.step()
    assert any(
        not torch.equal(old, parameter)
        for old, parameter in zip(before, trainable, strict=True)
    ), "optimizer did not update LoRA"
    return float(loss.detach())


def _validate_clone_routers(models, contract, hidden_size: int, vocab_size: int):
    from yeto.rl.deepseek_v4_expert_clone import (
        ORIGINAL_EXPERTS,
        TOTAL_EXPERTS,
        logical_to_training_expert_id,
    )

    records = []
    for model in models:
        for name, module in model.named_modules():
            if not name.endswith("mlp.router"):
                continue
            layer_id = int(module.layer_number) - 1
            assert type(module).__name__ == "DeepSeekV4CloneSplitRouter"
            assert module.weight.shape == (ORIGINAL_EXPERTS, hidden_size)
            assert module.config.num_moe_experts == ORIGINAL_EXPERTS
            assert module._yeto_total_experts == TOTAL_EXPERTS
            sources = contract.source_experts_by_layer[layer_id]
            if module.tid2eid is not None:
                source = sources[0]
                alternatives = [
                    expert
                    for expert in range(ORIGINAL_EXPERTS)
                    if expert not in sources and expert != source
                ][:5]
                table = torch.tensor(
                    [source, *alternatives],
                    dtype=module.tid2eid.dtype,
                    device=module.tid2eid.device,
                )
                with torch.no_grad():
                    module.tid2eid.copy_(table.expand(vocab_size, -1))
            tokens = torch.arange(64, device="cuda", dtype=torch.long).remainder(
                vocab_size
            ).reshape(64, 1)
            hidden = torch.randn(
                64,
                1,
                hidden_size,
                device="cuda",
                dtype=torch.bfloat16,
            )
            with torch.no_grad():
                probs, routing_map = module(hidden, input_ids=tokens)
            assert probs.shape == routing_map.shape == (64, TOTAL_EXPERTS)
            assert torch.equal(
                routing_map.sum(dim=1),
                torch.full((64,), 6, device="cuda"),
            )
            assert torch.all(torch.isfinite(probs)).item()
            for rank, source in enumerate(sources):
                clone = ORIGINAL_EXPERTS + rank
                training_source = logical_to_training_expert_id(source)
                training_clone = logical_to_training_expert_id(clone)
                assert not torch.any(
                    routing_map[:, training_source]
                    & routing_map[:, training_clone]
                ).item()
            if module.tid2eid is not None:
                assert routing_map[
                    :, logical_to_training_expert_id(ORIGINAL_EXPERTS)
                ].any().item()
                assert routing_map[
                    :, logical_to_training_expert_id(sources[0])
                ].any().item()
            records.append(
                {
                    "name": name,
                    "layer": layer_id,
                    "gate_experts": int(module.weight.shape[0]),
                    "dispatch_experts": int(routing_map.shape[1]),
                    "hash_router": module.tid2eid is not None,
                }
            )
    return sorted(records, key=lambda row: row["layer"])


def _build_real_optimizer(models, args):
    """Build the patched production replicated Adam without training."""

    from megatron.core.optimizer import OptimizerConfig
    from miles.backends.megatron_utils import model as miles_model

    config = OptimizerConfig(
        optimizer="adam",
        lr=args.expert_full_lr,
        min_lr=args.expert_full_lr,
        weight_decay=0.0,
        use_distributed_optimizer=False,
        bf16=True,
        clip_grad=1.0,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_eps=1e-8,
    )
    config.timers = None
    return miles_model.get_megatron_optimizer(
        config=config,
        model_chunks=models,
        use_gloo_process_groups=False,
    )


def main() -> None:
    args = _args()
    if args.layers not in (4, 5, 43):
        raise ValueError("runtime validation supports 4, 5, or 43 layers")
    if args.layers != 4 and not (
        args.pipeline_parallel == 2 and args.expert_full_count == 16
    ):
        raise ValueError("5/43-layer validation is reserved for PP2 expert-full")
    if args.wrap_with_ddp and not args.expert_full_count:
        raise ValueError("DDP lifecycle validation requires expert-full mode")
    if args.build_real_optimizer and not args.wrap_with_ddp:
        raise ValueError("real optimizer validation requires --wrap-with-ddp")
    if args.validate_sleep_process_groups and not args.build_real_optimizer:
        raise ValueError(
            "sleep process-group validation requires --build-real-optimizer"
        )
    if args.task_bridge_model and not args.expert_full_count:
        raise ValueError("task bridge validation requires expert-full mode")
    assert args.experts > 0 and args.rank > 0 and args.seq_length >= 128
    _validate_expert_full_component_environment(args)
    if args.pipeline_parallel > 1 and (
        args.load_base_weights
        or args.forward_backward
        or (args.expect_clone_split and not args.expert_full_count)
    ):
        raise ValueError(
            "PP2 mode does not support base loading or forward/backward"
        )
    if args.forward_backward and not args.load_base_weights:
        raise ValueError(
            "forward/backward validation requires --load-base-weights; "
            "the randomly initialized V4 fixture is not numerically stable"
        )

    if args.validate_sleep_process_groups:
        from miles.utils.reloadable_process_group import monkey_patch_torch_dist

        # Production calls this before Megatron creates any topology groups.
        monkey_patch_torch_dist()

    from yeto.rl.deepseek_v4_bridge import ensure_deepseek_v4_bridge

    ensure_deepseek_v4_bridge()

    from megatron.bridge import AutoBridge
    from miles.backends.megatron_utils.lora_utils import create_lora_instance

    from yeto.rl.export import derive_peft_lora_specs
    from yeto.rl.learner import megatron_adapter_targets

    clone_contract = None
    model_config = None
    if args.expect_clone_split or args.expert_full_count:
        from transformers import AutoConfig

        from yeto.rl.deepseek_v4_expert_clone import contract_from_config

        model_config = AutoConfig.from_pretrained(
            args.model,
            trust_remote_code=True,
            local_files_only=True,
        )
        clone_contract = contract_from_config(model_config)
        assert clone_contract is not None
        assert args.experts == 288

    full_specs = derive_peft_lora_specs(
        args.model,
        None,
        rank=args.rank,
        targets="attention",
        trust_remote_code=True,
    )
    assert len(full_specs) == 214, len(full_specs)

    bridge = AutoBridge.from_hf_pretrained(args.model, trust_remote_code=True)
    full_targets = megatron_adapter_targets(
        full_specs,
        bridge,
        pipeline_parallel=args.pipeline_parallel,
    )
    if args.pipeline_parallel == 1:
        assert len(full_targets) == 107, len(full_targets)

    expected_attention_specs = {
        spec.name: spec
        for spec in full_specs
        if (_layer_number(spec.name, _LAYER) or 0) < args.layers
    }
    expected_specs = dict(expected_attention_specs)
    expected_expert_specs = {}
    if args.expert_full_count:
        from yeto.rl.deepseek_v4_expert_full import expert_full_specs

        expected_expert_specs = {
            spec.name: spec
            for spec in expert_full_specs(
                model_config,
                expert_count=args.expert_full_count,
                expected_selection_sha256=clone_contract.selection_sha256,
                expected_selection_contract_sha256=(
                    clone_contract.selection_contract_sha256
                ),
            )
            if (_layer_number(spec.name, _LAYER) or 0) < args.layers
        }
        expected_specs.update(expected_expert_specs)
    tiny_targets = _runtime_targets(
        full_targets,
        layers=args.layers,
        pipeline_parallel=args.pipeline_parallel,
    )
    if args.layers == 4:
        assert len(expected_attention_specs) == 18, len(expected_attention_specs)
    elif args.layers == 43:
        assert len(expected_attention_specs) == len(full_specs)
    else:
        assert len(expected_attention_specs) > 18
    if args.expert_full_count:
        assert len(expected_expert_specs) == args.layers * 16 * 3
    if args.pipeline_parallel == 1:
        assert len(tiny_targets) == 9, len(tiny_targets)
    else:
        assert tiny_targets and all(".layers.*." in target for target in tiny_targets)

    # Megatron-Bridge's automatic HF load hook runs before get_model() moves
    # modules to CUDA.  With TP>1 that makes its NCCL scatter attempt to carry
    # CPU tensors.  Register the same load explicitly after a per-rank CUDA
    # materialization hook so this validation exercises the real collective
    # conversion path without requiring a second Gloo process-group topology.
    provider = bridge.to_megatron_provider(load_weights=False)
    if args.load_base_weights:
        provider.perform_initialization = False
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    tensor_parallel, pipeline_parallel, expert_parallel = _parallel_sizes(
        world_size,
        args.pipeline_parallel,
        expert_full_count=args.expert_full_count,
    )
    data_parallel = world_size // (tensor_parallel * pipeline_parallel)
    assert data_parallel * tensor_parallel * pipeline_parallel == world_size
    provider.tensor_model_parallel_size = tensor_parallel
    provider.pipeline_model_parallel_size = pipeline_parallel
    provider.expert_model_parallel_size = expert_parallel
    provider.expert_tensor_parallel_size = 1
    provider.context_parallel_size = 1
    provider.sequence_parallel = tensor_parallel > 1
    pipeline_layer_counts = _pipeline_layer_counts(
        args.layers,
        pipeline_parallel,
    )
    provider.num_layers = args.layers
    provider.moe_layer_freq = [1] * args.layers
    provider.dsv4_compress_ratios = list(provider.dsv4_compress_ratios)[
        : args.layers
    ]
    provider.dsv4_n_hash_layers = min(
        int(provider.dsv4_n_hash_layers),
        args.layers,
    )
    if pipeline_parallel == 2 and args.layers % 2:
        provider.num_layers_in_first_pipeline_stage = pipeline_layer_counts[0]
        provider.num_layers_in_last_pipeline_stage = pipeline_layer_counts[1]
    provider.num_moe_experts = args.experts
    if not args.load_base_weights:
        provider.vocab_size = 1024
    provider.seq_length = args.seq_length
    provider.max_position_embeddings = args.seq_length
    provider.mtp_num_layers = None
    provider.mtp_enabled = False
    provider.finalize()

    lora = create_lora_instance(
        Namespace(
            lora_type="canonical_lora",
            target_modules=tiny_targets,
            exclude_modules=None,
            lora_rank=args.rank,
            lora_alpha=args.rank,
            lora_dropout=0.0,
            lora_A_init_method="xavier",
            lora_B_init_method="zero",
            experts_shared_outer_loras=False,
        )
    )

    def apply_lora(model_chunks):
        transformed = lora(model_chunks, training=True)
        lora.set_params_to_save(transformed)
        return transformed

    pre_wrap_parameter_ids: set[int] = set()

    def snapshot_expert_parameters(model_chunks):
        pre_wrap_parameter_ids.update(
            _flagged_expert_parameters(model_chunks)
        )
        return model_chunks

    if args.load_base_weights:

        def materialize_and_load(model_chunks):
            device = torch.cuda.current_device()
            for model in model_chunks:
                model.cuda(device)
            bridge.load_hf_weights(model_chunks)
            return model_chunks

        provider.register_pre_wrap_hook(materialize_and_load)
    provider.register_pre_wrap_hook(apply_lora)
    if args.expert_full_count:
        provider.register_pre_wrap_hook(snapshot_expert_parameters)
    if args.wrap_with_ddp:
        from megatron.bridge.training.config import DistributedDataParallelConfig

        ddp_config = DistributedDataParallelConfig(
            use_distributed_optimizer=False,
            grad_reduce_in_fp32=True,
        )
        ddp_config.finalize()
        models = provider.provide_distributed_model(
            wrap_with_ddp=True,
            ddp_config=ddp_config,
        )
    else:
        models = provider.provide_distributed_model(
            wrap_with_ddp=False,
            mixed_precision_wrapper=None,
        )

    optimizer = None
    if args.build_real_optimizer:
        optimizer = _build_real_optimizer(models, args)

    local_routers = [
        (
            name,
            int(module.layer_number),
            module.tid2eid is not None,
            module.expert_bias is not None,
        )
        for name, module in models[0].named_modules()
        if name.endswith("mlp.router")
    ]
    if pipeline_parallel == 1:
        routers = local_routers
    else:
        router_partitions = [None] * world_size
        torch.distributed.all_gather_object(router_partitions, local_routers)
        routers = [router for partition in router_partitions for router in partition]
    assert sorted(router[2:] for router in routers) == sorted(
        _expected_router_modes(
            layers=args.layers,
            pipeline_parallel=pipeline_parallel,
            tensor_parallel=tensor_parallel,
            data_parallel=data_parallel,
        )
    ), routers

    clone_routers = None
    if args.expect_clone_split:
        clone_routers = _validate_clone_routers(
            models,
            clone_contract,
            provider.hidden_size,
            provider.vocab_size,
        )
        assert len(clone_routers) == args.layers

    pp2 = None
    if pipeline_parallel == 2:
        pp2 = _pp2_attention_round_trip(
            bridge,
            models,
            expected_attention_specs,
            pipeline_layer_counts=pipeline_layer_counts,
            require_existing_masters=args.build_real_optimizer,
        )
        if args.expert_full_count:
            coverage_actor, _coverage_snapshot = _expert_task_coverage_snapshot(
                bridge,
                models,
                expected_specs,
                pre_wrap_parameter_ids=pre_wrap_parameter_ids,
                model_path=args.model,
                task_bridge_model=args.task_bridge_model,
            )
            pp2["expert_task_coverage"] = _pp2_expert_task_coverage(
                bridge,
                models,
                expected_specs,
                pipeline_layer_counts=pipeline_layer_counts,
                pre_wrap_parameter_ids=pre_wrap_parameter_ids,
                model_path=args.model,
                task_bridge_model=args.task_bridge_model,
                actor_and_record=(coverage_actor, _coverage_snapshot),
            )
            pp2["expert_transport_shards"] = _pp2_expert_transport_shards(
                coverage_actor,
                expected_specs,
                model_config=model_config,
                clone_contract=clone_contract,
            )
            if args.validate_sleep_process_groups:
                pp2["sleep_process_groups"] = (
                    _sleep_process_group_task_coverage(
                        bridge,
                        models,
                        expected_specs,
                        pipeline_layer_counts=pipeline_layer_counts,
                        pre_wrap_parameter_ids=pre_wrap_parameter_ids,
                        model_path=args.model,
                        task_bridge_model=args.task_bridge_model,
                    )
                )
        side_count = int(pp2["global_sides"])
    else:
        model_bridge = bridge._model_bridge
        sides = _adapter_sides(model_bridge, models)
        assert {name for name, _side in sides} == set(expected_specs)
        mapped = {
            id(side.param_weight)
            for _name, side in sides
            if side.param_weight is not None
        }
        trainable = {
            id(parameter)
            for model in models
            for parameter in model.parameters()
            if parameter.requires_grad
        }
        assert mapped == trainable, (len(mapped), len(trainable))

        initial = _export(bridge, models)
        assert set(initial) == set(expected_specs)
        assert {
            name: tuple(value.shape) for name, value in initial.items()
        } == {
            name: spec.shape for name, spec in expected_specs.items()
        }

        expected_values = {}
        with torch.no_grad():
            for ordinal, (name, side) in enumerate(sides):
                value = _deterministic_value(expected_specs[name].shape, ordinal)
                converted = side.mapping.hf_to_megatron(
                    value,
                    side.megatron_module,
                )
                assert converted.numel() == side.param_weight.numel(), (
                    name,
                    tuple(converted.shape),
                    tuple(side.param_weight.shape),
                )
                side.param_weight.copy_(converted.reshape(side.param_weight.shape))
                expected_values[name] = value.cpu()

        round_trip = _export(bridge, models)
        assert set(round_trip) == set(expected_values)
        for name, expected in expected_values.items():
            assert torch.equal(round_trip[name], expected), name

        # The exact round-trip fixture deliberately writes moderately sized
        # integer values into both LoRA sides. Leaving those values installed
        # makes the product of four production-width decoder layers overflow
        # and turns this into a numerical-stress test instead of a gradient-flow
        # test. Restore the original initialization before forward/backward.
        with torch.no_grad():
            for name, side in sides:
                value = initial[name].to(device=side.param_weight.device)
                converted = side.mapping.hf_to_megatron(
                    value,
                    side.megatron_module,
                )
                assert converted.numel() == side.param_weight.numel(), name
                side.param_weight.copy_(converted.reshape(side.param_weight.shape))
        side_count = len(sides)

    loss = None
    if args.forward_backward:
        loss = _forward_backward_step(models, args.seq_length, provider.vocab_size)

    rank = torch.distributed.get_rank()
    if rank == 0:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "world_size": world_size,
                    "tensor_parallel_size": tensor_parallel,
                    "pipeline_parallel_size": pipeline_parallel,
                    "expert_parallel_size": expert_parallel,
                    "data_parallel_size": data_parallel,
                    "pipeline_layer_counts": pipeline_layer_counts,
                    "wrapped_with_ddp": args.wrap_with_ddp,
                    "real_optimizer": optimizer is not None,
                    "sleep_process_groups": (
                        args.validate_sleep_process_groups
                    ),
                    "separate_task_bridge": args.task_bridge_model is not None,
                    "full_modules": len(full_targets),
                    "full_sides": len(full_specs),
                    "tiny_modules": len(tiny_targets),
                    "tiny_sides": side_count,
                    "router_modes": routers,
                    "clone_routers": clone_routers,
                    "pp2": pp2,
                    "round_trip": "exact",
                    "forward_backward_loss": loss,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
