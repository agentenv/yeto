"""Backend traits the Miles adapter declares to the core (decoupling 2.6/2.7). Import-light."""

from __future__ import annotations

from ..capabilities import BackendTraits

# Neutral transport -> Miles/upstream name (miles protocol.py:73-89): colocate
# uses CUDA IPC; a LoRA fixed partition must use NCCL broadcast. These names are
# what tapes recorded before decoupling and stay the recorded labels.
MILES_WEIGHT_TRANSPORTS = {"same-device-ipc": "cuda-ipc", "collective-broadcast": "nccl-broadcast"}

# --offload-train: upstream train.py sleeps the actor *before* update_weights and
# publishes from host backups (S13 FN OOM fix), so Miles can publish while asleep.
MILES_TRAITS = BackendTraits(weight_transport_names=MILES_WEIGHT_TRANSPORTS, publish_while_offloaded=True)

# Ledger ``engine_discarded.mechanism`` for groups aborted in flight when the
# batch filled (partial_rollout off; cut-audit §3, sglang_rollout.py:420-437).
MILES_ABORT_MECHANISM = "miles generate_rollout abort"
