"""Select NeMo's existing whole-layer checkpoint helper for pinned native Qwen.

The installed predicate ordinarily selects this helper only for HF-native
layers. Extend that selection narrowly to the exact native Qwen27B architecture,
before FSDP wrapping, rather than nesting whole-layer and submodule wrappers.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def qualified_native_qwen(model, scopes, *, enable_compile):
    config = getattr(model.config, 'text_config', model.config)
    return (any(cls.__module__ == 'nemo_automodel.components.models.qwen3_5.model'
                and cls.__name__ == 'Qwen3_5ForConditionalGeneration' for cls in type(model).__mro__)
            and config.num_hidden_layers == 64 and config.hidden_size == 5120
            and not getattr(config, 'num_kv_shared_layers', 0)
            and tuple(scopes) == ('all',) and not enable_compile)


def install():
    from nemo_automodel.components.distributed import activation_checkpointing as ac
    from nemo_automodel.components.distributed import parallelizer as p
    expected = [(p, 'dc786d522e5703f345ebc08f349910807e73c5b150bc365fb0273be29e3242bc'),
                (ac, '9f222b5ec2dd3da585e7414dc7578b642e1dcff256a8bbed81547b70bac5eae1')]
    if any(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != digest for module,digest in expected):
        raise RuntimeError('Whole-block checkpoint selection does not match the pinned NeMo runtime')
    if getattr(p, '_yeta_qwen_whole_block_v1', False):
        return
    original = p._should_use_hf_native_gradient_checkpointing
    def choose_full_layer(model, layer_groups, scopes, *, enable_compile):
        selected = qualified_native_qwen(model, scopes, enable_compile=enable_compile)
        print({'event': 'whole_block_checkpoint_selection', 'model_class': str(type(model)),
               'selected': selected, 'scopes': scopes, 'enable_compile': enable_compile}, flush=True)
        if selected:
            language = layer_groups.get('language', [])
            if len(language) != 64 or any(not any(p.requires_grad for p in layer.parameters()) for layer in language):
                raise RuntimeError('Whole-block selection requires all64trainable native decoder blocks')
            return True
        return original(model, layer_groups, scopes, enable_compile=enable_compile)
    p._should_use_hf_native_gradient_checkpointing = choose_full_layer
    p._yeta_qwen_whole_block_v1 = True


def verify_model(model):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
    container = model.model.language_model.layers
    layers = list(container.values()) if hasattr(container, "values") else list(container)
    if len(layers) != 64 or not all(isinstance(layer, CheckpointWrapper) for layer in layers):
        raise RuntimeError('All64 native decoder blocks must use whole-block checkpoint wrappers')
    wrappers = [module for module in model.modules() if isinstance(module, CheckpointWrapper)]
    if len(wrappers) != 64:
        raise RuntimeError('Unexpected nested or missing activation checkpoint wrappers')
    return {'whole_block_wrappers': len(wrappers), 'nested_wrappers': False}
