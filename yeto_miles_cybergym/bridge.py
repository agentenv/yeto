"""Megatron-Bridge hooks used by the standalone Miles baseline."""

from __future__ import annotations


def configure_miles_bridge(args) -> None:
    """Forward the requested attention backend into AutoBridge providers.

    Miles' LoRA path constructs its model through Megatron-Bridge.  The
    provider created by ``AutoBridge`` does not otherwise inherit
    ``args.attention_backend``, which lets Transformer Engine select its
    cuDNN fused-attention backend even when the command requested a different
    backend.  Install a narrow, temporary wrapper before the model is built.
    """

    from megatron.bridge import AutoBridge
    from miles.backends.megatron_utils import model as miles_model

    backend = getattr(args, "attention_backend", None)
    if backend is None:
        raise RuntimeError("Miles Bridge setup received no attention backend")

    setup = miles_model._setup_lora_model_via_bridge

    def configured_setup(runtime_args):
        to_provider = AutoBridge.to_megatron_provider

        def configured_provider(bridge, *provider_args, **provider_kwargs):
            provider = to_provider(bridge, *provider_args, **provider_kwargs)
            provider.attention_backend = runtime_args.attention_backend
            return provider

        AutoBridge.to_megatron_provider = configured_provider
        try:
            return setup(runtime_args)
        finally:
            AutoBridge.to_megatron_provider = to_provider

    miles_model._setup_lora_model_via_bridge = configured_setup
