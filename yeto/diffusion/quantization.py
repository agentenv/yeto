"""Compatibility helpers for quantized Diffusers models."""

from __future__ import annotations

from contextlib import contextmanager

import torch


@contextmanager
def npu_bitsandbytes_load(device):
    """Allow recent bitsandbytes NPU backends through Diffusers' CUDA gate."""
    if getattr(device, "type", None) != "npu":
        yield
        return
    try:
        import bitsandbytes as bnb
        from diffusers.quantizers.bitsandbytes import BnB4BitDiffusersQuantizer
    except ImportError:
        yield
        return
    if "npu" not in getattr(bnb, "supported_torch_devices", ()):
        yield
        return

    original_validate = BnB4BitDiffusersQuantizer.validate_environment
    original_device_map = BnB4BitDiffusersQuantizer.update_device_map
    index = device.index
    if index is None:
        index = torch.npu.current_device()

    def validate_environment(quantizer, *args, **kwargs):
        cuda_is_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: True
        try:
            return original_validate(quantizer, *args, **kwargs)
        finally:
            torch.cuda.is_available = cuda_is_available

    def update_device_map(quantizer, device_map):
        if device_map is None:
            return {"": f"npu:{index}"}
        return original_device_map(quantizer, device_map)

    BnB4BitDiffusersQuantizer.validate_environment = validate_environment
    BnB4BitDiffusersQuantizer.update_device_map = update_device_map
    try:
        yield
    finally:
        BnB4BitDiffusersQuantizer.validate_environment = original_validate
        BnB4BitDiffusersQuantizer.update_device_map = original_device_map
