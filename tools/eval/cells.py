"""The capability x backend matrix of the evaluation suite.

Model Qwen3-0.6B (pinned revision), GSM8K, LoRA, Modal A10G, the fewest cards
the capability needs. Each cell names its harness, its configuration (what the
judge reads too) and whether the backend supports the capability: an
unsupported cell still runs, and its failure is the expected baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CAPABILITIES = {
    "E1": "island rollout + trainer",
    "E2": "pure sampler island reclaimed and replaced",
    "E3": "synced trainer islands -> sampler island",
    "E4": "async rollout + off-policy correction + over-age drop",
    "E5": "round cut: SCHEDULED stop + resume assertions",
}
BACKENDS = ("miles", "verl")
# all-in estimate per A10G container hour: GPU $1.10 + the CPU (8-16 cores) and memory
# (32-64 GiB) the harness requests, which Modal bills too (list prices; not a bill)
A10G_USD_PER_HOUR = 2.0


@dataclass(frozen=True)
class Cell:
    capability: str
    backend: str
    harness: str  # "modal_verl" | "miles_local" | "miles_head"
    cards: int
    config: dict[str, Any] = field(default_factory=dict)
    supported: bool = True
    note: str = ""

    @property
    def key(self) -> str:
        return f"{self.capability}-{self.backend}"


# ------------------------------------------------------------------ verl (direct Modal harness)
_VERL = {"gpu": "A10G", "rounds": 4, "groups_per_round": 4, "samples_per_group": 4, "max_tokens": 256,
         "lr": 1e-4, "lora_r": 32, "seq_len": 768, "seed": 1, "sglang_mem": 0.6, "max_model_len": 1024,
         "timeout_s": 2700, "learners": 1, "max_policy_age": 1, "sampler_rounds": 24, "max_backlog_groups": 8,
         "reclaim_after_round": None}
# the plain island cells reuse the S25 C4 configuration (8 groups x 4, lr 1e-5)
_VERL_ISLAND = {**_VERL, "kind": "island", "groups_per_round": 8, "lr": 1e-5, "seq_len": 1024, "seed": 17,
                "stop_after_rounds": 2}

# ------------------------------------------------------------------ Miles (yeto launcher on Modal)
_MILES = {"gpu": "A10G", "hard_s": 3000, "rollout_batch_size": 16, "n_samples": 4, "max_response_len": 512,
          "seq_len": 1024, "lora_r": 16, "lr": 1e-5, "seed": 17}

CELLS: dict[str, Cell] = {c.key: c for c in (
    Cell("E1", "verl", "modal_verl", 1, {**_VERL_ISLAND, "phases": ["c"]}),
    Cell("E1", "miles", "miles_local", 1, {**_MILES, "rounds": 4, "islands": 1}),
    # E2: the S25 C1 shape (two samplers, one reclaimed after it wrote round 1) on Miles; verl's
    # sampler island is the neutral engine sampler, which resumes after a reclaim since S25 G1C
    Cell("E2", "verl", "modal_verl", 2,
         {**_VERL, "kind": "buffer", "max_policy_age": 2, "max_backlog_groups": 12, "reclaim_after_round": 1}),
    Cell("E2", "miles", "miles_local", 3,
         {**_MILES, "rounds": 8, "islands": 3, "samplers": 2, "max_policy_age": 4, "reclaim_label": "l0",
          "reclaim_round": 1, "reclaim_component": "l0"}),
    Cell("E3", "verl", "modal_verl", 3,
         {**_VERL, "kind": "buffer", "learners": 2, "max_policy_age": 2, "max_backlog_groups": 16,
          "sampler_rounds": 20}),
    Cell("E3", "miles", "miles_head", 3, {**_MILES, "rounds": 4, "learners": 2, "hard_s": 2700}),
    # E4: a sampler island far ahead of the trainer (backlog 3 rounds, short responses) and
    # max_policy_age 1: round 2 meets v0 groups, which must be dropped
    Cell("E4", "verl", "modal_verl", 2,
         {**_VERL, "kind": "buffer", "max_policy_age": 1, "max_backlog_groups": 12, "max_tokens": 128}),
    Cell("E4", "miles", "miles_local", 2,
         {**_MILES, "rounds": 4, "islands": 2, "max_policy_age": 1, "sampler_rounds": 10, "backlog_rounds": 3,
          "max_response_len": 256}),
    Cell("E5", "verl", "modal_verl", 1,
         {**_VERL_ISLAND, "phases": ["a", "b", "c"], "compare_prompt_sets": True}),
    Cell("E5", "miles", "miles_local", 2,
         {**_MILES, "rounds": 6, "islands": 2, "max_policy_age": 2, "min_round_s": 120}),
)}


def get(capability: str, backend: str) -> Cell:
    key = f"{capability.upper()}-{backend}"
    if key not in CELLS:
        raise SystemExit(f"no cell {key}; known: {', '.join(sorted(CELLS))}")
    return CELLS[key]
