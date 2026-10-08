"""Backend traits the Miles adapter declares to the core (decoupling 2.6/2.7). Import-light."""

from __future__ import annotations

from ..capabilities import BackendTraits

# Neutral transport -> Miles/upstream name (miles protocol.py:73-89): colocate
# uses CUDA IPC; a LoRA fixed partition must use NCCL broadcast. These names are
# what tapes recorded before decoupling and stay the recorded labels.
MILES_WEIGHT_TRANSPORTS = {"same-device-ipc": "cuda-ipc", "collective-broadcast": "nccl-broadcast"}

MILES_TRAITS = BackendTraits(weight_transport_names=MILES_WEIGHT_TRANSPORTS)
