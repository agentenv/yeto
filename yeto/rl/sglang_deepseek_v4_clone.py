"""Install SGLang's DeepSeek-V4 256-gate/288-expert clone adapter.

This module is imported through ``sitecustomize`` in every Miles/Ray/SGLang
Python process.  It patches only ``DeepseekV2MoE`` instances whose checkpoint
config carries a valid ``yeto_routed_expert_clone`` contract; ordinary models
retain the pinned SGLang implementation byte-for-byte at runtime.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from types import MethodType, ModuleType
from typing import Any


_TARGET = "sglang.srt.models.deepseek_v2"
_LORA_TARGET = "sglang.srt.lora.lora_manager"
_LORA_ADAPTER_TARGET = "sglang.srt.lora.lora"
_LORA_POOL_TARGET = "sglang.srt.lora.mem_pool"
_FINDER = None
_LORA_FINDER = None
_LORA_ADAPTER_FINDER = None
_LORA_POOL_FINDER = None


_RUNTIME_Q_B_TARGET = "self_attn.wq_b"
_RUNTIME_INDEXER_Q_B_TARGET = "indexer.wq_b"


def _runtime_lora_targets(targets):
    mapping = {
        "q_a_proj": "wq_a",
        # Engine integrations may normalize PEFT targets before LoRAManager
        # sees them.  V4 has an independent q-a projection, so translate the
        # DeepSeek-V2 fused spelling back to that runtime leaf as well.
        "fused_qkv_a_proj_with_mqa": "wq_a",
        # ``wq_b`` is ambiguous in SGLang V4: the ordinary attention
        # projection and the DSA indexer both use that leaf name.  The pinned
        # target normalizer deliberately maps a *bare* ``wq_b`` to
        # ``indexer.wq_b``.  Keep the parent qualifier so only the policy's
        # self-attention projection is wrapped.
    }
    if isinstance(targets, str):
        if targets in {"q_b_proj", "wq_b"}:
            return [_RUNTIME_Q_B_TARGET, _RUNTIME_INDEXER_Q_B_TARGET]
        return mapping.get(targets, targets)
    converted = []
    for name in targets:
        # Miles maps the canonical PEFT q_b projection to the runtime leaf
        # ``wq_b`` before constructing ServerArgs.  In the attested V4 clone
        # recipe that leaf still denotes the complete policy target: both the
        # main attention q-b and the DSA indexer q-b.  Expand either spelling
        # before SGLang's generic normalizer collapses bare wq_b to the indexer.
        if name in {"q_b_proj", "wq_b"}:
            converted.extend(
                (_RUNTIME_Q_B_TARGET, _RUNTIME_INDEXER_Q_B_TARGET)
            )
        else:
            converted.append(mapping.get(name, name))
    # Preserve deterministic order for sequence inputs while retaining set
    # semantics for SGLang's server-side target set.
    if isinstance(targets, set):
        return set(converted)
    return type(targets)(dict.fromkeys(converted))


def _runtime_target_normalizer(original, targets):
    """Preserve the qualified V4 attention q_b target through SGLang.

    SGLang's generic normalizer reduces every input to its final leaf before
    applying aliases, so ``self_attn.wq_b`` would otherwise become the DSA
    indexer target ``indexer.wq_b``.  Only the explicit qualified marker gets
    this exception; ordinary models and an explicitly requested indexer target
    retain the pinned behavior.
    """

    normalized = set(original(targets))
    if isinstance(targets, str):
        requested = {targets}
    else:
        requested = set(targets)
    if _RUNTIME_Q_B_TARGET in requested:
        if "indexer.wq_b" not in requested:
            normalized.discard("indexer.wq_b")
        normalized.add(_RUNTIME_Q_B_TARGET)
    return normalized


def _replace_first_dimension(parameter, rows: int):
    import torch

    replacement = torch.nn.Parameter(
        parameter.detach().new_empty((rows, *parameter.shape[1:])),
        requires_grad=parameter.requires_grad,
    )
    # SGLang/model loaders attach metadata directly to Parameters.
    replacement.__dict__.update(parameter.__dict__)
    return replacement


def _patch_topk_forward(
    topk,
    *,
    layer_id: int,
    sources,
    hash_topk: bool,
    device,
) -> None:
    if getattr(topk, "_yeto_clone_split_installed", False):
        raise RuntimeError("SGLang clone split was installed twice on one TopK")

    import torch

    from .deepseek_v4_expert_clone import remap_topk_ids_torch

    # This stable per-layer buffer must exist before SGLang records its decode
    # CUDA graph.  Constructing it from Python integers inside capture would
    # attempt a forbidden CPU-to-GPU copy.  Registering it also lets a later
    # module .to(...) move preserve the usual PyTorch device semantics.
    topk.register_buffer(
        "_yeto_clone_source_expert_ids",
        torch.tensor(tuple(sources), dtype=torch.int32, device=device),
        persistent=False,
    )

    original_forward = topk.forward

    def clone_split_forward(
        topk_self,
        hidden_states,
        router_logits,
        *args,
        input_ids=None,
        **kwargs,
    ):
        if input_ids is None:
            raise RuntimeError("SGLang clone-split routing requires input token IDs")
        if hash_topk:
            output = original_forward(
                hidden_states,
                router_logits,
                *args,
                input_ids=input_ids,
                **kwargs,
            )
        else:
            output = original_forward(
                hidden_states,
                router_logits,
                *args,
                **kwargs,
            )
        topk_ids = getattr(output, "topk_ids", None)
        if topk_ids is None or not hasattr(output, "_replace"):
            raise RuntimeError(
                "SGLang clone split requires the standard explicit TopK output"
            )
        remapped = remap_topk_ids_torch(
            topk_ids,
            input_ids,
            layer_id=layer_id,
            source_experts=sources,
            source_expert_ids=topk_self._yeto_clone_source_expert_ids,
        )
        return output._replace(topk_ids=remapped)

    topk.forward = MethodType(clone_split_forward, topk)
    topk._yeto_clone_split_installed = True
    topk._yeto_clone_source_experts = tuple(sources)
    topk._yeto_clone_layer_id = layer_id


def _configure_moe(module, config, *, require_runtime_flags: bool = True) -> None:
    from .deepseek_v4_expert_clone import (
        ORIGINAL_EXPERTS,
        TOTAL_EXPERTS,
        contract_from_config,
    )

    contract = contract_from_config(config)
    if contract is None:
        return
    if getattr(module, "is_nextn", False):
        raise RuntimeError("expanded DeepSeek V4 checkpoint must not construct MTP")
    layer_id = int(getattr(module, "layer_id", -1))
    if not 0 <= layer_id < len(contract.source_experts_by_layer):
        raise RuntimeError("SGLang clone-split MoE has an invalid layer ID")
    if int(config.n_routed_experts) != TOTAL_EXPERTS:
        raise RuntimeError("SGLang clone-split experts were not constructed as 288-way")
    if int(getattr(module, "num_fused_shared_experts", -1)) != 0:
        raise RuntimeError(
            "logical expert cloning requires --sglang-disable-shared-experts-fusion"
        )

    if require_runtime_flags:
        from sglang.srt.runtime_context import get_server_args

        server_args = get_server_args()
        if bool(getattr(server_args, "enable_eplb", False)):
            raise RuntimeError("logical expert cloning is incompatible with SGLang EPLB")

    gate = module.gate
    if tuple(gate.weight.shape) != (TOTAL_EXPERTS, int(config.hidden_size)):
        raise RuntimeError("SGLang constructed an unexpected expanded gate shape")
    gate.weight = _replace_first_dimension(gate.weight, ORIGINAL_EXPERTS)

    correction_bias = getattr(gate, "e_score_correction_bias", None)
    if correction_bias is not None:
        if tuple(correction_bias.shape) != (TOTAL_EXPERTS,):
            raise RuntimeError("SGLang constructed an unexpected correction-bias shape")
        gate.e_score_correction_bias = _replace_first_dimension(
            correction_bias,
            ORIGINAL_EXPERTS,
        )
        if hasattr(module.topk, "topk_config"):
            module.topk.topk_config.correction_bias = gate.e_score_correction_bias
        if hasattr(module, "correction_bias"):
            module.correction_bias = gate.e_score_correction_bias.data

    hash_topk = hasattr(module.topk, "tid2eid")
    if hash_topk:
        module.topk.num_experts = ORIGINAL_EXPERTS
        initializer = getattr(module.topk, "_init_default_tid2eid", None)
        if initializer is not None:
            initializer()
    else:
        topk_config = getattr(module.topk, "topk_config", None)
        if topk_config is None:
            raise RuntimeError("learned clone router has no explicit TopK config")
        # Bypassed/fused carriers cannot be rewritten after selection.  The
        # pinned Triton MoE runner accepts StandardTopKOutput directly.
        from sglang.srt.layers.moe.topk import TopKOutputFormat

        topk_config.output_format = TopKOutputFormat.STANDARD

    _patch_topk_forward(
        module.topk,
        layer_id=layer_id,
        sources=contract.source_experts_by_layer[layer_id],
        hash_topk=hash_topk,
        device=gate.weight.device,
    )
    # Existing DSV4 call sites pass input IDs to TopK for hash layers.  Marking
    # all clone-split layers this way supplies stable token IDs to learned
    # routers too and disables incompatible bypass/dual-stream shortcuts.
    module.is_hash = True
    module._yeto_clone_selection_sha256 = contract.selection_sha256


def install_on_module(module: ModuleType) -> None:
    cls = getattr(module, "DeepseekV2MoE", None)
    if cls is None:
        raise RuntimeError("pinned SGLang module has no DeepseekV2MoE")
    if getattr(cls, "_yeto_clone_patch_installed", False):
        return
    original_init = cls.__init__

    def patched_init(self, config, *args, **kwargs):
        original_init(self, config, *args, **kwargs)
        _configure_moe(self, config)

    cls.__init__ = patched_init
    cls._yeto_clone_patch_installed = True
    cls._yeto_clone_original_init = original_init


def install_on_lora_manager(module: ModuleType) -> None:
    """Keep routed-expert targets from wrapping dense/shared MLP linears.

    A PEFT adapter names logical routed leaves as ``gate_proj``, ``up_proj``
    and ``down_proj``.  SGLang normalizes those names to the fused routed-MoE
    buffers, but its generic module scan would also wrap every ordinary
    ``gate_up_proj``/``down_proj`` in dense and shared-expert MLPs.  Those
    modules are outside the canonical E288 policy and can have a different
    width, so associating the routed buffer with them is both incorrect and a
    shape error.  Filter only that generic scan for the attested clone recipe;
    the routed ``FusedMoE`` modules remain visible and take the dedicated path.
    """

    cls = getattr(module, "LoRAManager", None)
    if cls is None:
        raise RuntimeError("pinned SGLang module has no LoRAManager")
    if getattr(cls, "_yeto_clone_patch_installed", False):
        return
    original_init_lora_modules = cls.init_lora_modules
    original_init_lora_shapes = cls.init_lora_shapes
    original_validate_new_adapter = cls.validate_new_adapter
    original_normalize_targets = getattr(
        module,
        "get_normalized_target_modules",
        None,
    )
    if original_normalize_targets is None:
        raise RuntimeError(
            "pinned SGLang LoRA manager has no target-module normalizer"
        )

    def patched_normalize_targets(targets):
        return _runtime_target_normalizer(original_normalize_targets, targets)

    # ``init_lora_shapes`` resolves this imported module global at call time.
    module.get_normalized_target_modules = patched_normalize_targets

    def patched_init_lora_shapes(
        self,
        max_lora_rank=None,
        target_modules=None,
    ):
        from .deepseek_v4_expert_clone import contract_from_config

        clone_contract = contract_from_config(self.base_hf_config)
        if clone_contract is not None:
            # LoRAManager imports mem_pool while it is itself being imported.
            # Re-assert the pool hook at the last point before buffer creation
            # so a reload/import-order change cannot silently restore the
            # generic DeepSeek-V2 geometry.
            import importlib

            pool_module = importlib.import_module(_LORA_POOL_TARGET)
            install_on_lora_pool(pool_module)
            manager_pool_cls = getattr(module, "LoRAMemoryPool", None)
            if manager_pool_cls is None:
                raise RuntimeError(
                    "pinned SGLang LoRA manager has no memory-pool class"
                )
            pool_module._yeto_clone_bind_pool_geometry(manager_pool_cls)
            if target_modules is not None:
                target_modules = _runtime_lora_targets(target_modules)
            for config in self.configs.values():
                config.target_modules = list(
                    _runtime_lora_targets(config.target_modules)
                )
        if clone_contract is None:
            return original_init_lora_shapes(
                self,
                max_lora_rank=max_lora_rank,
                target_modules=target_modules,
            )

        # This pinned manager treats every DSA-indexer target as incompatible
        # with WK/weights_proj fusion, and imports a stale module-level flag to
        # enforce that.  V4's indexer keeps wq_b as an independent linear even
        # when WK/weights_proj are fused, so only that qualified leaf is safe.
        # Temporarily remove exactly wq_b from the compatibility check; other
        # indexer projections retain the pinned fail-closed behavior.
        indexer_names = getattr(module, "DSA_INDEXER_LORA_NAMES", None)
        if indexer_names is None or _RUNTIME_INDEXER_Q_B_TARGET not in indexer_names:
            raise RuntimeError(
                "pinned SGLang manager has an unexpected DSA indexer contract"
            )
        module.DSA_INDEXER_LORA_NAMES = frozenset(
            set(indexer_names) - {_RUNTIME_INDEXER_Q_B_TARGET}
        )
        try:
            result = original_init_lora_shapes(
                self,
                max_lora_rank=max_lora_rank,
                target_modules=target_modules,
            )
        finally:
            module.DSA_INDEXER_LORA_NAMES = indexer_names

        legacy_attention_targets = {
            "q_a_proj",
            "q_b_proj",
            "fused_qkv_a_proj_with_mqa",
        }.intersection(self.target_modules)
        if legacy_attention_targets:
            raise RuntimeError(
                "expanded DeepSeek V4 retained legacy attention LoRA targets "
                f"{sorted(legacy_attention_targets)} after runtime normalization; "
                f"resolved targets are {sorted(self.target_modules)}"
            )
        runtime_attention_targets = {
            "wq_a",
            _RUNTIME_Q_B_TARGET,
            _RUNTIME_INDEXER_Q_B_TARGET,
        }
        present_attention_targets = runtime_attention_targets.intersection(
            self.target_modules
        )
        if present_attention_targets and (
            present_attention_targets != runtime_attention_targets
        ):
            raise RuntimeError(
                "expanded DeepSeek V4 attention LoRA requires the complete "
                "runtime q-a/q-b/indexer target set; resolved targets are "
                f"{sorted(self.target_modules)}"
            )
        self._yeto_clone_runtime_attention_targets = tuple(
            sorted(present_attention_targets)
        )
        return result

    def patched_validate_new_adapter(self, lora_config, lora_ref):
        from .deepseek_v4_expert_clone import contract_from_config

        if contract_from_config(self.base_hf_config) is not None:
            lora_config.target_modules = list(
                _runtime_lora_targets(lora_config.target_modules)
            )
        return original_validate_new_adapter(self, lora_config, lora_ref)

    def patched_init_lora_modules(self, *args, **kwargs):
        from .deepseek_v4_expert_clone import contract_from_config

        if contract_from_config(self.base_hf_config) is None:
            return original_init_lora_modules(self, *args, **kwargs)
        routed_targets = {"gate_up_proj", "down_proj"}
        requested_routed = routed_targets.intersection(self.target_modules)
        if requested_routed and requested_routed != routed_targets:
            raise RuntimeError(
                "expanded DeepSeek V4 LoRA requires both routed expert targets"
            )

        if not requested_routed:
            result = original_init_lora_modules(self, *args, **kwargs)
            skipped = []
            self._yeto_clone_excluded_lora_modules = ()
            wrapped_names = {
                name.rsplit(".", 1)[-1]
                for layer_modules in self.lora_modules
                for name in layer_modules
            }
            requested_attention = {"wq_a", "wq_b"}.intersection(
                self.target_modules
            )
            missing_attention = requested_attention.difference(wrapped_names)
            if missing_attention:
                raise RuntimeError(
                    "expanded DeepSeek V4 attention LoRA could not find runtime "
                    f"modules {sorted(missing_attention)}; "
                    "SGLANG_OPT_FUSE_WQA_WKV must be 0"
                )
            return result

        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        base_model = self.base_model
        original_named_modules = base_model.named_modules
        had_instance_override = "named_modules" in base_model.__dict__
        previous_override = base_model.__dict__.get("named_modules")
        skipped = []

        def routed_policy_named_modules(*named_args, **named_kwargs):
            for name, child in original_named_modules(*named_args, **named_kwargs):
                leaf = name.rsplit(".", 1)[-1]
                generic_mlp_linear = leaf in routed_targets and not isinstance(
                    child, FusedMoE
                )
                shared_fused_moe = isinstance(child, FusedMoE) and bool(
                    getattr(child, "is_shared_fused_moe", False)
                )
                if generic_mlp_linear or shared_fused_moe:
                    skipped.append(name)
                    continue
                yield name, child

        object.__setattr__(base_model, "named_modules", routed_policy_named_modules)
        try:
            result = original_init_lora_modules(self, *args, **kwargs)
        finally:
            if had_instance_override:
                object.__setattr__(base_model, "named_modules", previous_override)
            else:
                object.__delattr__(base_model, "named_modules")
        if not skipped:
            raise RuntimeError(
                "expanded DeepSeek V4 LoRA found no non-routed MLP modules to exclude"
            )
        self._yeto_clone_excluded_lora_modules = tuple(skipped)
        wrapped_names = {
            name.rsplit(".", 1)[-1]
            for layer_modules in self.lora_modules
            for name in layer_modules
        }
        requested_attention = {"wq_a", "wq_b"}.intersection(self.target_modules)
        missing_attention = requested_attention.difference(wrapped_names)
        if missing_attention:
            raise RuntimeError(
                "expanded DeepSeek V4 attention LoRA could not find runtime "
                f"modules {sorted(missing_attention)}; "
                "SGLANG_OPT_FUSE_WQA_WKV must be 0"
            )
        return result

    cls.init_lora_shapes = patched_init_lora_shapes
    cls.validate_new_adapter = patched_validate_new_adapter
    cls.init_lora_modules = patched_init_lora_modules
    cls._yeto_clone_patch_installed = True
    cls._yeto_clone_original_init_lora_modules = original_init_lora_modules
    cls._yeto_clone_original_init_lora_shapes = original_init_lora_shapes
    cls._yeto_clone_original_validate_new_adapter = original_validate_new_adapter
    cls._yeto_clone_original_normalize_targets = original_normalize_targets


def install_on_lora_adapter(module: ModuleType) -> None:
    """Map canonical V4 q_a/q_b adapter leaves to SGLang runtime leaves."""

    cls = getattr(module, "LoRAAdapter", None)
    if cls is None:
        raise RuntimeError("pinned SGLang module has no LoRAAdapter")
    if getattr(cls, "_yeto_clone_patch_installed", False):
        return
    original_normalize = cls.normalize_fused_qkv_a_proj

    def patched_normalize(self, weight_names, weights):
        from .deepseek_v4_expert_clone import contract_from_config

        if contract_from_config(self.base_hf_config) is None:
            return original_normalize(self, weight_names, weights)
        for weight_name in list(weight_names):
            runtime_name = weight_name.replace("q_a_proj", "wq_a").replace(
                "q_b_proj", "wq_b"
            )
            if runtime_name == weight_name:
                continue
            if runtime_name in weights:
                raise RuntimeError(
                    f"duplicate canonical/runtime V4 LoRA tensor {runtime_name!r}"
                )
            weights[runtime_name] = weights.pop(weight_name)

    cls.normalize_fused_qkv_a_proj = patched_normalize
    cls._yeto_clone_patch_installed = True
    cls._yeto_clone_original_normalize_fused_qkv_a_proj = original_normalize


def install_on_lora_pool(module: ModuleType) -> None:
    """Teach SGLang's buffer allocator the unfused V4 q_a/q_b geometry."""

    if getattr(module, "_yeto_clone_patch_installed", False):
        patched = getattr(module, "_yeto_clone_patched_get_hidden_dim", None)
        binder = getattr(module, "_yeto_clone_bind_pool_geometry", None)
        if patched is None:
            raise RuntimeError(
                "SGLang LoRA memory-pool patch marker lost its patched geometry"
            )
        if binder is None:
            raise RuntimeError(
                "SGLang LoRA memory-pool patch marker lost its class binder"
            )
        # importlib.reload retains arbitrary module attributes while replacing
        # imported globals.  Restore the attested function in that case.
        module.get_hidden_dim = patched
        pool_cls = getattr(module, "LoRAMemoryPool", None)
        if pool_cls is not None:
            binder(pool_cls)
        return
    original_get_hidden_dim = module.get_hidden_dim
    original_normalize_targets = getattr(
        module,
        "get_normalized_target_modules",
        None,
    )
    if original_normalize_targets is None:
        raise RuntimeError(
            "pinned SGLang LoRA memory pool has no target-module normalizer"
        )

    def patched_get_hidden_dim(
        module_name,
        config,
        base_model=None,
        layer_idx=0,
        lora_added_tokens_size=None,
    ):
        if module_name == "wq_a":
            return int(config.hidden_size), int(config.q_lora_rank)
        if module_name in {"wq_b", _RUNTIME_Q_B_TARGET}:
            # DeepSeek-V4's checkpoint-native q projection is
            # ``num_attention_heads * head_dim``.  SGLang may attach a
            # compatibility ``qk_nope_head_dim`` (commonly the V2 value 128)
            # to the runtime config; that legacy field must not override V4's
            # authoritative 512-wide head geometry.
            q_head_dim = getattr(config, "head_dim", None)
            if q_head_dim is None:
                q_head_dim = int(config.qk_nope_head_dim) + int(
                    config.qk_rope_head_dim
                )
            return (
                int(config.q_lora_rank),
                int(config.num_attention_heads) * int(q_head_dim),
            )
        if module_name == _RUNTIME_INDEXER_Q_B_TARGET:
            # The pinned generic helper gates this geometry on
            # is_deepseek_dsa(config) and therefore rejects DeepSeek V4,
            # even though V4's C4 compressor uses the same independent
            # replicated indexer wq_b projection.
            return (
                int(config.q_lora_rank),
                int(config.index_n_heads) * int(config.index_head_dim),
            )
        return original_get_hidden_dim(
            module_name,
            config,
            base_model,
            layer_idx,
            lora_added_tokens_size,
        )

    patched_get_hidden_dim._yeto_clone_geometry = True

    def bind_pool_geometry(pool_cls) -> None:
        if pool_cls is None:
            raise RuntimeError("pinned SGLang module has no LoRAMemoryPool")
        shape_method = getattr(pool_cls, "get_lora_B_shape", None)
        if shape_method is None:
            raise RuntimeError(
                "pinned SGLang LoRAMemoryPool has no B-shape allocator"
            )
        # The manager can retain a LoRAMemoryPool class whose methods refer to
        # an older module dictionary after SGLang's worker-side import/reload
        # sequence.  Rebinding only sys.modules[...].get_hidden_dim is then
        # insufficient.  Bind the exact globals used by the class that the
        # manager will instantiate.
        pool_globals = shape_method.__globals__
        pool_globals["get_hidden_dim"] = patched_get_hidden_dim
        replicated = pool_globals.get("REPLICATED_LINEAR_LORA_NAMES")
        if replicated is None:
            replicated = getattr(module, "REPLICATED_LINEAR_LORA_NAMES", None)
            if replicated is not None:
                pool_globals["REPLICATED_LINEAR_LORA_NAMES"] = replicated
        if replicated is None:
            raise RuntimeError(
                "pinned SGLang LoRAMemoryPool lost replicated-linear metadata"
            )
        if "wq_a" not in replicated:
            replicated.append("wq_a")

    module.get_hidden_dim = patched_get_hidden_dim
    module.get_normalized_target_modules = lambda targets: (
        _runtime_target_normalizer(original_normalize_targets, targets)
    )
    if "wq_a" not in module.REPLICATED_LINEAR_LORA_NAMES:
        module.REPLICATED_LINEAR_LORA_NAMES.append("wq_a")

    pool_cls = getattr(module, "LoRAMemoryPool", None)
    if pool_cls is not None:
        bind_pool_geometry(pool_cls)
    if pool_cls is not None and not getattr(
        pool_cls, "_yeto_clone_layout_check_installed", False
    ):
        original_pool_init = pool_cls.__init__

        def patched_pool_init(self, *args, **kwargs):
            original_pool_init(self, *args, **kwargs)

            from .deepseek_v4_expert_clone import contract_from_config

            if contract_from_config(self.base_hf_config) is None:
                return
            config = self.base_hf_config
            slots = int(self.max_loras_per_batch)
            rank = int(self.max_lora_rank)
            q_rank = int(config.q_lora_rank)
            hidden = int(config.hidden_size)
            head_dim = getattr(config, "head_dim", None)
            if head_dim is None:
                qk_nope = int(config.qk_nope_head_dim)
                head_dim = qk_nope + int(config.qk_rope_head_dim)
            q_output = int(config.num_attention_heads) * int(head_dim)
            if q_output % int(self.tp_size):
                raise RuntimeError(
                    "expanded DeepSeek V4 q-b output is not divisible by TP"
                )
            expected = {
                "wq_a": (
                    (slots, rank, hidden),
                    (slots, q_rank, rank),
                ),
                _RUNTIME_Q_B_TARGET: (
                    (slots, rank, q_rank),
                    (slots, q_output // int(self.tp_size), rank),
                ),
                _RUNTIME_INDEXER_Q_B_TARGET: (
                    (slots, rank, q_rank),
                    (
                        slots,
                        int(config.index_n_heads) * int(config.index_head_dim),
                        rank,
                    ),
                ),
            }
            for name in set(self.target_modules).intersection(expected):
                if name not in self.A_buffer or name not in self.B_buffer:
                    raise RuntimeError(
                        f"expanded DeepSeek V4 LoRA pool omitted target {name!r}"
                    )
                actual_a = {tuple(tensor.shape) for tensor in self.A_buffer[name]}
                actual_b = {tuple(tensor.shape) for tensor in self.B_buffer[name]}
                expected_a, expected_b = expected[name]
                if actual_a != {expected_a} or actual_b != {expected_b}:
                    raise RuntimeError(
                        "expanded DeepSeek V4 LoRA pool geometry mismatch for "
                        f"{name!r}: A={sorted(actual_a)} expected={expected_a}; "
                        f"B={sorted(actual_b)} expected={expected_b}; "
                        f"get_hidden_dim={module.get_hidden_dim.__module__}."
                        f"{module.get_hidden_dim.__name__}"
                    )

        pool_cls.__init__ = patched_pool_init
        pool_cls._yeto_clone_layout_check_installed = True
        pool_cls._yeto_clone_original_init = original_pool_init

    module._yeto_clone_patch_installed = True
    module._yeto_clone_patched_get_hidden_dim = patched_get_hidden_dim
    module._yeto_clone_bind_pool_geometry = bind_pool_geometry
    module._yeto_clone_original_get_hidden_dim = original_get_hidden_dim
    module._yeto_clone_original_normalize_targets = original_normalize_targets


class _Loader(importlib.abc.Loader):
    def __init__(self, wrapped, installer) -> None:
        self.wrapped = wrapped
        self.installer = installer

    def create_module(self, spec):
        create = getattr(self.wrapped, "create_module", None)
        return None if create is None else create(spec)

    def exec_module(self, module) -> None:
        self.wrapped.exec_module(module)
        self.installer(module)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, fullname: str, installer) -> None:
        self.fullname = fullname
        self.installer = installer

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.fullname:
            return None
        try:
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot locate required module {self.fullname}")
        spec.loader = _Loader(spec.loader, self.installer)
        return spec


def install() -> None:
    """Patch imported SGLang modules or arm process-wide import hooks."""

    global _FINDER, _LORA_FINDER, _LORA_ADAPTER_FINDER, _LORA_POOL_FINDER
    loaded = sys.modules.get(_TARGET)
    if loaded is not None:
        install_on_module(loaded)
    elif _FINDER is None:
        _FINDER = _Finder(_TARGET, install_on_module)
        sys.meta_path.insert(0, _FINDER)

    loaded_lora = sys.modules.get(_LORA_TARGET)
    if loaded_lora is not None:
        install_on_lora_manager(loaded_lora)
    elif _LORA_FINDER is None:
        _LORA_FINDER = _Finder(_LORA_TARGET, install_on_lora_manager)
        sys.meta_path.insert(0, _LORA_FINDER)

    loaded_adapter = sys.modules.get(_LORA_ADAPTER_TARGET)
    if loaded_adapter is not None:
        install_on_lora_adapter(loaded_adapter)
    elif _LORA_ADAPTER_FINDER is None:
        _LORA_ADAPTER_FINDER = _Finder(
            _LORA_ADAPTER_TARGET,
            install_on_lora_adapter,
        )
        sys.meta_path.insert(0, _LORA_ADAPTER_FINDER)

    loaded_pool = sys.modules.get(_LORA_POOL_TARGET)
    if loaded_pool is not None:
        install_on_lora_pool(loaded_pool)
    elif _LORA_POOL_FINDER is None:
        _LORA_POOL_FINDER = _Finder(_LORA_POOL_TARGET, install_on_lora_pool)
        sys.meta_path.insert(0, _LORA_POOL_FINDER)

    # Ray trainer workers are independent Python processes, so the bridge
    # registration performed by the main learner is not inherited.  Arm a
    # lazy hook here (sitecustomize already calls this function) and register
    # V4 only when a worker imports Miles' Megatron bridge helper.
    from .deepseek_v4_bridge import install_deepseek_v4_actor_bridge_hook

    install_deepseek_v4_actor_bridge_hook()
