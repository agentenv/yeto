# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Megatron Bridge - A component of the Megatron ecosystem."""

# yeto miles-ports image: the eager ``import modelopt.torch`` that Megatron-Bridge
# 8cd3466 runs here is removed (rl-trainer-cold-start-cost L2.2).  It cost
# 21-30 s per trainer process (coldstart-gpu-review §6.2).  Every bridge module
# that uses modelopt (training.checkpointing, distillation, conversion.modelopt_utils)
# imports it itself, so its plugins still load when a modelopt feature is used.
# Written over the base image's file by docker/miles-ports/make_layer.py.

import megatron.bridge.diffusion.models  # noqa: F401 — registers diffusion bridges
import megatron.bridge.models  # noqa: F401 — triggers all bridge and HF Auto class registrations
from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.package_info import (
    __contact_emails__,
    __contact_names__,
    __download_url__,
    __homepage__,
    __package_name__,
    __repository_url__,
    __version__,
)


__all__ = [
    "__version__",
    "__package_name__",
    "__contact_names__",
    "__contact_emails__",
    "__homepage__",
    "__repository_url__",
    "__download_url__",
    "AutoBridge",
]
