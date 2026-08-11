#!/usr/bin/env python3
"""GPU runtime validation for Yeto's pinned DeepSeek-V4 bridge.

This intentionally builds a small decoder with the production V4 attention
geometry.  It validates the complete 43-layer PEFT contract on meta tensors,
then validates real Megatron LoRA injection and collective TP conversion on the
small decoder.  Run it inside the pinned Miles image under ``torchrun``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from argparse import Namespace

import torch


_CANONICAL_PREFIX = "base_model.model."
_LAYER = re.compile(r"\.layers\.(\d+)\.")
_MEGATRON_LAYER = re.compile(r"^decoder\.layers\.(\d+)\.")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--load-base-weights", action="store_true")
    parser.add_argument("--forward-backward", action="store_true")
    parser.add_argument("--expect-clone-split", action="store_true")
    return parser.parse_args()


def _canonical_name(raw_name: str) -> str:
    return (
        raw_name
        if raw_name.startswith(_CANONICAL_PREFIX)
        else _CANONICAL_PREFIX + raw_name
    )


def _layer_number(name: str, pattern: re.Pattern[str]) -> int | None:
    match = pattern.search(name)
    return int(match.group(1)) if match is not None else None


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
    # Small integers are represented exactly by bf16, so a collective
    # import/export round trip can use an exact equality assertion.
    values = (torch.arange(count, device="cuda", dtype=torch.int64) + ordinal) % 17
    return (values - 8).to(dtype=torch.float32).reshape(shape)


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


def main() -> None:
    args = _args()
    assert args.layers == 4, "the validation compression fixture is four layers"
    assert args.experts > 0 and args.rank > 0 and args.seq_length >= 128
    if args.forward_backward and not args.load_base_weights:
        raise ValueError(
            "forward/backward validation requires --load-base-weights; "
            "the randomly initialized V4 fixture is not numerically stable"
        )

    from yeto.rl.deepseek_v4_bridge import ensure_deepseek_v4_bridge

    ensure_deepseek_v4_bridge()

    from megatron.bridge import AutoBridge
    from miles.backends.megatron_utils.lora_utils import create_lora_instance
    from yeto.rl.export import derive_peft_lora_specs
    from yeto.rl.learner import megatron_adapter_targets

    clone_contract = None
    if args.expect_clone_split:
        from transformers import AutoConfig
        from yeto.rl.deepseek_v4_expert_clone import contract_from_config

        clone_contract = contract_from_config(
            AutoConfig.from_pretrained(
                args.model,
                trust_remote_code=True,
                local_files_only=True,
            )
        )
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
    full_targets = megatron_adapter_targets(full_specs, bridge)
    assert len(full_targets) == 107, len(full_targets)

    expected_specs = {
        spec.name: spec
        for spec in full_specs
        if (_layer_number(spec.name, _LAYER) or 0) < args.layers
    }
    tiny_targets = [
        target
        for target in full_targets
        if (_layer_number(target, _MEGATRON_LAYER) or 0) < args.layers
    ]
    assert len(expected_specs) == 18, len(expected_specs)
    assert len(tiny_targets) == 9, len(tiny_targets)

    # Megatron-Bridge's automatic HF load hook runs before get_model() moves
    # modules to CUDA.  With TP>1 that makes its NCCL scatter attempt to carry
    # CPU tensors.  Register the same load explicitly after a per-rank CUDA
    # materialization hook so this validation exercises the real collective
    # conversion path without requiring a second Gloo process-group topology.
    provider = bridge.to_megatron_provider(load_weights=False)
    if args.load_base_weights:
        provider.perform_initialization = False
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    provider.tensor_model_parallel_size = world_size
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = world_size
    provider.expert_tensor_parallel_size = 1
    provider.context_parallel_size = 1
    provider.sequence_parallel = world_size > 1
    provider.num_layers = args.layers
    provider.moe_layer_freq = [1] * args.layers
    provider.dsv4_compress_ratios = [0, 0, 4, 128]
    provider.dsv4_n_hash_layers = 3
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

    if args.load_base_weights:

        def materialize_and_load(model_chunks):
            device = torch.cuda.current_device()
            for model in model_chunks:
                model.cuda(device)
            bridge.load_hf_weights(model_chunks)
            return model_chunks

        provider.register_pre_wrap_hook(materialize_and_load)
    provider.register_pre_wrap_hook(apply_lora)
    models = provider.provide_distributed_model(
        wrap_with_ddp=False,
        mixed_precision_wrapper=None,
    )

    routers = [
        (
            name,
            int(module.layer_number),
            module.tid2eid is not None,
            module.expert_bias is not None,
        )
        for name, module in models[0].named_modules()
        if name.endswith("mlp.router")
    ]
    assert [router[2:] for router in routers] == [
        (True, False),
        (True, False),
        (True, False),
        (False, True),
    ], routers

    clone_routers = None
    if clone_contract is not None:
        clone_routers = _validate_clone_routers(
            models,
            clone_contract,
            provider.hidden_size,
            provider.vocab_size,
        )
        assert len(clone_routers) == args.layers

    model_bridge = bridge._model_bridge
    sides = _adapter_sides(model_bridge, models)
    assert {name for name, _side in sides} == set(expected_specs)
    mapped = {id(side.param_weight) for _name, side in sides if side.param_weight is not None}
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
            converted = side.mapping.hf_to_megatron(value, side.megatron_module)
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

    # The exact round-trip fixture deliberately writes moderately sized integer
    # values into both LoRA sides.  Leaving those values installed makes the
    # product of four production-width decoder layers overflow and turns this
    # into a numerical-stress test instead of a gradient-flow test.  Restore
    # the original Xavier-A/zero-B initialization before forward/backward.
    with torch.no_grad():
        for name, side in sides:
            value = initial[name].to(device=side.param_weight.device)
            converted = side.mapping.hf_to_megatron(value, side.megatron_module)
            assert converted.numel() == side.param_weight.numel(), name
            side.param_weight.copy_(converted.reshape(side.param_weight.shape))

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
                    "full_modules": len(full_targets),
                    "full_sides": len(full_specs),
                    "tiny_modules": len(tiny_targets),
                    "tiny_sides": len(sides),
                    "router_modes": routers,
                    "clone_routers": clone_routers,
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
