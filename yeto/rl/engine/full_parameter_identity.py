"""Identity rules of a full-parameter island, shared by every engine.

The Miles island (``adapters/miles/island_entry.py``) and the verl island
(``adapters/verl/full_parameter_stream.py``) had their own copies of these
rules; both now call this module (S27 G-DV, s27-next), so the two engines
cannot drift apart.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

_SHA256 = re.compile(r"[0-9a-f]{64}")


def identity_sha256(value: Any) -> str:
    """SHA256 of the compact, key-sorted JSON of ``value`` (the island identity
    hashes; NaN is not refused, unlike the reward-pipeline hash)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def full_parameter_identity(model_config_sha256: str) -> tuple[str, str]:
    """``(identity layout hash, config hash)`` of a full-parameter island: no
    LoRA layout, the model config is the identity."""
    return model_config_sha256, identity_sha256(
        {"schema": "yeto-full-parameter-mode-v1", "model_config_sha256": model_config_sha256}
    )


def is_sha256(value: Any) -> bool:
    """A lowercase hex SHA256 string."""
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def local_init(init_mode: str, async_rounds: bool) -> bool:
    """Local init (S27): every island launched by one run loads the same base
    checkpoint, so async rounds keep it as version 0 by default (``auto``);
    the syncer refuses the run if the islands' fragment digests disagree."""
    return init_mode == "local" or (init_mode == "auto" and async_rounds)
