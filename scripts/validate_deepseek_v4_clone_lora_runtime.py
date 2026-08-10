#!/usr/bin/env python3
"""Validate E288 clone-only expert LoRA on real Megatron TP/EP ranks."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
from argparse import Namespace
from types import SimpleNamespace

import torch


_LAYER = re.compile(r"\.layers\.(\d+)\.")
_EXPERT = re.compile(r"\.mlp\.experts\.(\d+)\.")


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--tensor-parallel", type=int)
    parser.add_argument("--expert-parallel", type=int)
    return parser.parse_args()


def _layer(name: str) -> int | None:
    match = _LAYER.search(name)
    return None if match is None else int(match.group(1))


def _copy_masters_to_model(parameters) -> None:
    with torch.no_grad():
        for parameter in parameters:
            parameter.copy_(parameter.main_param)


def _exercise_optimizer(
    models,
    *,
    seq_length: int,
    vocab_size: int,
    dp_group,
    dp_size: int,
    dp_rank: int,
) -> dict[str, object]:
    """Run real forward/backward and an AdamW update through clone slices."""

    from yeto.rl.deepseek_v4_clone_lora import assert_original_expert_lora_zero

    trainable = [
        parameter
        for model in models
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    for parameter in trainable:
        parameter.main_param = torch.nn.Parameter(
            parameter.detach().float().clone(),
            requires_grad=True,
        )
    optimizer = torch.optim.AdamW(
        [parameter.main_param for parameter in trainable],
        lr=1.0e-4,
        weight_decay=0.01,
    )

    tokens = (
        torch.arange(seq_length, device="cuda", dtype=torch.long)
        .remainder(vocab_size)
        .unsqueeze(0)
    )
    positions = torch.arange(
        seq_length,
        device="cuda",
        dtype=torch.long,
    ).unsqueeze(0)
    output = models[0](
        input_ids=tokens,
        position_ids=positions,
        attention_mask=None,
    )
    if isinstance(output, tuple):
        output = output[0]
    # Different DP replicas intentionally see different loss scales.  The
    # explicit Megatron DP all-reduce below must make every corresponding LoRA
    # shard identical before the optimizer step.
    loss = output.float().square().mean() * float(dp_rank + 1)
    if not torch.isfinite(loss).item():
        raise RuntimeError(f"non-finite clone LoRA validation loss: {loss.item()}")
    loss.backward()

    dp_synced_parameters = 0
    if dp_size > 1:
        for parameter in trainable:
            if parameter.grad is None:
                continue
            torch.distributed.all_reduce(parameter.grad, group=dp_group)
            parameter.grad.div_(dp_size)
            dp_synced_parameters += 1
        if not dp_synced_parameters:
            raise RuntimeError("Megatron DP group synchronized no LoRA gradients")

        # The three moments are calculated independently on every member after
        # the collective.  Exact agreement is a compact fail-closed check that
        # corresponding TP/EP shards received the same reduced gradients.
        signature = torch.zeros(3, dtype=torch.float64, device="cuda")
        for parameter in trainable:
            if parameter.grad is None:
                continue
            gradient = parameter.grad.double()
            signature[0] += gradient.sum()
            signature[1] += gradient.abs().sum()
            signature[2] += gradient.square().sum()
        signatures = [torch.empty_like(signature) for _ in range(dp_size)]
        torch.distributed.all_gather(signatures, signature, group=dp_group)
        if any(not torch.equal(signatures[0], value) for value in signatures[1:]):
            raise RuntimeError("LoRA gradient signatures differ after DP all-reduce")

    clone_grad_nonzero = 0
    original_grad_nonzero = 0
    clone_before = []
    found_masks = 0
    for model in models:
        for module in model.modules():
            mask = getattr(module, "_yeto_clone_active_mask", None)
            if mask is None:
                continue
            found_masks += 1
            active = mask.bool()
            for parameter in (module.linear_in.weight, module.linear_out.weight):
                gradient = parameter.grad
                if gradient is None:
                    continue
                if not torch.isfinite(gradient).all().item():
                    raise RuntimeError("clone-only expert LoRA gradient is non-finite")
                if (~active).any():
                    original_grad_nonzero += int(
                        torch.count_nonzero(gradient[~active]).item()
                    )
                if active.any():
                    clone_grad_nonzero += int(
                        torch.count_nonzero(gradient[active]).item()
                    )
                    clone_before.append(
                        (parameter, active, parameter.main_param[active].clone())
                    )
    if not found_masks or original_grad_nonzero:
        raise RuntimeError(
            "clone-only gradient mask is missing or an original slice received gradient"
        )
    reduced_grad = torch.tensor(clone_grad_nonzero, device="cuda", dtype=torch.long)
    torch.distributed.all_reduce(reduced_grad)
    if reduced_grad.item() <= 0:
        raise RuntimeError("no clone expert slice received a non-zero gradient")

    for parameter in trainable:
        parameter.main_param.grad = (
            None
            if parameter.grad is None
            else parameter.grad.detach().float().clone()
        )
    optimizer.step()
    _copy_masters_to_model(trainable)
    assert_original_expert_lora_zero(models)

    clone_updated = sum(
        not torch.equal(before, parameter.main_param[active])
        for parameter, active, before in clone_before
    )
    reduced_updates = torch.tensor(clone_updated, device="cuda", dtype=torch.long)
    torch.distributed.all_reduce(reduced_updates)
    if reduced_updates.item() <= 0:
        raise RuntimeError("AdamW did not update any clone expert slice")
    return {
        "loss": float(loss.detach()),
        "clone_grad_nonzero": int(reduced_grad.item()),
        "clone_parameters_updated": int(reduced_updates.item()),
        "optimizer_state_entries": len(optimizer.state),
        "dp_synced_parameters": dp_synced_parameters,
        "dp_gradient_signature": (
            None if dp_size == 1 else [float(value) for value in signatures[0]]
        ),
        "optimizer": optimizer,
        "trainable": trainable,
    }


def main() -> None:
    args = _args()
    if not 1 <= args.layers <= 43 or args.rank <= 0 or args.seq_length < 128:
        raise ValueError(
            "clone LoRA runtime fixture requires 1..43 layers, positive rank, "
            "and at least 128 tokens"
        )

    from yeto.rl.deepseek_v4_bridge import ensure_deepseek_v4_bridge

    ensure_deepseek_v4_bridge()

    from megatron.bridge import AutoBridge
    from miles.backends.megatron_utils import trainable_state
    from miles.backends.megatron_utils.lora_utils import create_lora_instance
    from yeto.rl.deepseek_v4_clone_lora import assert_original_expert_lora_zero
    from yeto.rl.deepseek_v4_expert_clone import contract_from_config
    from yeto.rl.export import derive_peft_lora_specs
    from yeto.rl.learner import megatron_adapter_targets
    from transformers import AutoConfig

    bridge = AutoBridge.from_hf_pretrained(args.model, trust_remote_code=True)
    hf_config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    )
    contract = contract_from_config(hf_config)
    if contract is None:
        raise RuntimeError("clone LoRA runtime requires an expanded E288 checkpoint")
    full_specs = derive_peft_lora_specs(
        args.model,
        None,
        rank=args.rank,
        targets="attention-routed-experts",
        trust_remote_code=True,
    )
    tiny_specs = tuple(
        spec for spec in full_specs if (_layer(spec.name) or 0) < args.layers
    )
    full_targets = megatron_adapter_targets(
        full_specs,
        bridge,
        standard_grouped_experts=True,
    )
    tiny_targets = [
        target for target in full_targets if (_layer(target) or 0) < args.layers
    ]
    tiny_attention_targets = [
        target for target in tiny_targets if ".mlp.experts." not in target
    ]
    expert_targets = [target for target in tiny_targets if ".mlp.experts." in target]
    expert_spec_count = sum(
        ".mlp.experts." in spec.name for spec in tiny_specs
    )
    attention_spec_count = len(tiny_specs) - expert_spec_count
    if (
        len(full_specs) != 74_518
        or expert_spec_count != args.layers * 288 * 3 * 2
        or attention_spec_count % 2
    ):
        raise RuntimeError(
            "unexpected E288 policy sizes "
            f"full={len(full_specs)}, selected={len(tiny_specs)}, "
            f"expert={expert_spec_count}, attention={attention_spec_count}"
        )
    if (
        len(tiny_attention_targets) != attention_spec_count // 2
        or len(expert_targets) != args.layers * 2
    ):
        raise RuntimeError(
            "standard grouped LoRA target resolution produced an unexpected "
            f"attention/expert split: {len(tiny_attention_targets)}/"
            f"{len(expert_targets)}"
        )

    provider = bridge.to_megatron_provider(load_weights=False)
    provider.perform_initialization = False
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    tensor_parallel = args.tensor_parallel or world_size
    expert_parallel = args.expert_parallel or tensor_parallel
    if (
        tensor_parallel <= 0
        or expert_parallel <= 0
        or world_size % tensor_parallel
        or tensor_parallel % expert_parallel
    ):
        raise ValueError(
            "invalid runtime topology: "
            f"world={world_size}, tp={tensor_parallel}, ep={expert_parallel}"
        )
    provider.tensor_model_parallel_size = tensor_parallel
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = expert_parallel
    provider.expert_tensor_parallel_size = 1
    provider.context_parallel_size = 1
    provider.sequence_parallel = tensor_parallel > 1
    provider.num_layers = args.layers
    provider.moe_layer_freq = [1] * args.layers
    config_path = os.path.join(args.model, "config.json")
    with open(config_path, encoding="utf-8") as config_handle:
        raw_config = json.load(config_handle)
    compress_ratios = list(raw_config.get("compress_ratios") or ())
    if len(compress_ratios) < 43:
        raise RuntimeError(
            f"expanded V4 config has only {len(compress_ratios)} compress ratios"
        )
    provider.dsv4_compress_ratios = compress_ratios[: args.layers]
    provider.dsv4_n_hash_layers = min(
        int(raw_config.get("n_hash_layers", 3)),
        args.layers,
    )
    provider.num_moe_experts = 288
    provider.seq_length = args.seq_length
    provider.max_position_embeddings = args.seq_length
    provider.mtp_num_layers = None
    provider.mtp_enabled = False
    provider.finalize()

    lora = create_lora_instance(
        Namespace(
            lora_type="lora",
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
    assert_original_expert_lora_zero(models)

    from megatron.core import parallel_state

    def group_hosts(group) -> list[str]:
        hosts: list[str | None] = [None] * torch.distributed.get_world_size(group)
        torch.distributed.all_gather_object(
            hosts,
            socket.gethostname(),
            group=group,
        )
        if any(host is None for host in hosts):
            raise RuntimeError("distributed topology host gather was incomplete")
        return [str(host) for host in hosts]

    tp_group = parallel_state.get_tensor_model_parallel_group()
    ep_group = parallel_state.get_expert_model_parallel_group()
    dp_group = parallel_state.get_data_parallel_group()
    tp_hosts = group_hosts(tp_group)
    ep_hosts = group_hosts(ep_group)
    dp_hosts = group_hosts(dp_group)
    dp_size = parallel_state.get_data_parallel_world_size()
    dp_rank = parallel_state.get_data_parallel_rank()
    if (
        len(tp_hosts) != tensor_parallel
        or len(ep_hosts) != expert_parallel
        or len(dp_hosts) != dp_size
        or len(set(tp_hosts)) != 1
        or len(set(ep_hosts)) != 1
    ):
        raise RuntimeError(
            "TP/EP escaped its node or parallel groups have unexpected sizes: "
            f"tp={tp_hosts}, ep={ep_hosts}, dp={dp_hosts}"
        )
    if dp_size > 1 and len(set(dp_hosts)) != dp_size:
        raise RuntimeError(f"Megatron DP group did not cross nodes: {dp_hosts}")

    # Make clone activation deterministic in the three hash-routed layers.
    # The learned fourth layer remains untouched and uses the real checkpoint.
    for model in models:
        for module in model.modules():
            if not hasattr(module, "tid2eid") or module.tid2eid is None:
                continue
            layer = int(module.layer_number) - 1
            sources = contract.source_experts_by_layer[layer]
            source = sources[0]
            alternatives = [
                expert
                for expert in range(256)
                if expert != source and expert not in sources
            ][:5]
            table = torch.tensor(
                [source, *alternatives],
                dtype=module.tid2eid.dtype,
                device=module.tid2eid.device,
            )
            with torch.no_grad():
                module.tid2eid.copy_(table.expand(provider.vocab_size, -1))

    tasks_by_base = bridge._model_bridge.build_adapter_conversion_tasks(models)
    side_records = []
    mapped = set()
    for base_name in sorted(tasks_by_base):
        for task in sorted(
            tasks_by_base[base_name],
            key=lambda value: value.adapter_key or "",
        ):
            for side_name, side in (
                ("in", task.linear_in_task),
                ("out", task.linear_out_task),
            ):
                if side.param_weight is not None:
                    mapped.add(id(side.param_weight))
                if ".mlp.experts." in str(side.mapping.hf_param):
                    side_records.append(
                        {
                            "base": base_name,
                            "side": side_name,
                            "param_name": side.param_name,
                            "hf_param": side.mapping.hf_param,
                            "shape": (
                                None
                                if side.param_weight is None
                                else tuple(int(value) for value in side.param_weight.shape)
                            ),
                        }
                    )
    trainable_ids = {
        id(parameter)
        for model in models
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    if mapped != trainable_ids:
        raise RuntimeError(
            f"adapter conversion coverage mismatch: mapped={len(mapped)} "
            f"trainable={len(trainable_ids)}"
        )

    runtime = _exercise_optimizer(
        models,
        seq_length=args.seq_length,
        vocab_size=provider.vocab_size,
        dp_group=dp_group,
        dp_size=dp_size,
        dp_rank=dp_rank,
    )
    optimizer = runtime["optimizer"]
    trainable = runtime["trainable"]

    class Scheduler:
        def __init__(self):
            self.num_steps = dp_size

        def step(self, increment):
            self.num_steps += increment

    backups = []
    child_optimizer = SimpleNamespace(
        optimizer=optimizer,
        _copy_main_params_to_model_params=lambda: _copy_masters_to_model(trainable),
    )
    actor = SimpleNamespace(
        args=SimpleNamespace(
            hf_checkpoint=args.model,
            tensor_model_parallel_size=tensor_parallel,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=expert_parallel,
            global_batch_size=dp_size,
            num_steps_per_rollout=1,
            yeto_rl_clone_only_lora=True,
            yeto_rl_canonical_lora_names=tuple(spec.name for spec in tiny_specs),
        ),
        model=models,
        optimizer=SimpleNamespace(chained_optimizers=[child_optimizer]),
        opt_param_scheduler=Scheduler(),
        weights_backuper=SimpleNamespace(backup=backups.append),
    )
    # Scale the production contract to the selected runtime layer count.
    trainable_state._CLONE_LAYERS = args.layers
    exported = trainable_state._collective_adapter_tensors(actor)
    state = trainable_state.make_trainable_state(1, exported)
    actor.args.yeto_rl_layout_hash = state.layout_hash

    expert_tensors = {
        name: value for name, value in state.tensors.items() if ".mlp.experts." in name
    }
    shared_tensors = [name for name in exported if ".shared_experts." in name]
    original_nonzero = []
    clone_names = []
    expert_ids = set()
    for name, value in expert_tensors.items():
        match = _EXPERT.search(name)
        if match is None:
            raise RuntimeError(f"cannot parse exported expert adapter {name}")
        expert_id = int(match.group(1))
        expert_ids.add(expert_id)
        if expert_id < 256 and torch.count_nonzero(value).item():
            original_nonzero.append(name)
        if expert_id >= 256:
            clone_names.append(name)
    if original_nonzero:
        raise RuntimeError(
            f"original expert adapters exported nonzero values: {original_nonzero[:4]}"
        )
    if expert_ids != set(range(288)):
        raise RuntimeError(
            f"collective adapter export covered expert IDs {sorted(expert_ids)[:4]}.."
            f"{sorted(expert_ids)[-4:] if expert_ids else []}, not 0..287"
        )
    if shared_tensors:
        raise RuntimeError("clone-only target unexpectedly adapted shared experts")

    # Destroy the local policy, then prove canonical -> packed apply restores
    # it exactly and clears every AdamW state entry for local adapter masters.
    with torch.no_grad():
        for parameter in trainable:
            parameter.main_param.zero_()
    _copy_masters_to_model(trainable)
    reset_count = trainable_state.apply_trainable_state(
        actor,
        state,
        reset_optimizer=True,
    )
    if reset_count != len(state.tensors):
        raise RuntimeError(
            f"policy apply reset count {reset_count}, expected {len(state.tensors)}"
        )
    if optimizer.state:
        raise RuntimeError("policy apply did not clear local AdamW adapter state")
    if backups != ["actor"]:
        raise RuntimeError(f"policy apply backup contract failed: {backups}")
    assert_original_expert_lora_zero(models)
    round_trip = trainable_state._collective_adapter_tensors(actor)
    if set(round_trip) != set(state.tensors):
        raise RuntimeError("policy re-export changed the canonical tensor set")
    mismatches = [
        name
        for name, value in round_trip.items()
        if not torch.equal(value, state.tensors[name])
    ]
    if mismatches:
        raise RuntimeError(f"policy round-trip mismatches: {mismatches[:4]}")

    rank = torch.distributed.get_rank()
    if rank == 0:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "world_size": world_size,
                    "tensor_parallel_size": tensor_parallel,
                    "expert_parallel_size": expert_parallel,
                    "data_parallel_size": dp_size,
                    "tp_hosts": tp_hosts,
                    "ep_hosts": ep_hosts,
                    "dp_hosts": dp_hosts,
                    "attention_targets": len(tiny_attention_targets),
                    "expert_targets": len(expert_targets),
                    "trainable_parameters": len(trainable_ids),
                    "exported_tensors": len(exported),
                    "expert_tensors": len(expert_tensors),
                    "clone_tensors": len(clone_names),
                    "exported_f32_bytes": sum(
                        tensor.numel() * tensor.element_size()
                        for tensor in exported.values()
                    ),
                    "forward_backward_loss": runtime["loss"],
                    "clone_grad_nonzero": runtime["clone_grad_nonzero"],
                    "clone_parameters_updated": runtime[
                        "clone_parameters_updated"
                    ],
                    "dp_synced_parameters": runtime["dp_synced_parameters"],
                    "dp_gradient_signature": runtime[
                        "dp_gradient_signature"
                    ],
                    "optimizer_state_entries_before_reset": runtime[
                        "optimizer_state_entries"
                    ],
                    "optimizer_state_entries_after_reset": len(optimizer.state),
                    "policy_layout_hash": state.layout_hash,
                    "policy_round_trip": "exact",
                    "policy_apply_reset_tensors": reset_count,
                    "expert_side_records": side_records[:8],
                    "expert_name_sample": sorted(expert_tensors)[:8],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
