"""The sampler island of a verl launch (S26 E9.4).

``yeto launch --rl-backend verl --rl-island-sample-role 0:sampler,1:trainer``
starts every island with the verl island entry. A sampler island runs no verl
trainer: :func:`run_sampler_island` builds the engine sampler spec
(:mod:`yeto.rl.engine.engine_sampler_entry`) from the island flags and runs the
engine sampler in-process, on the vLLM of the verl image (adapter
``vllm-server``). It writes the same buffer and follows the same releases as
the S25 C2 sampler (which ran SGLang in the Miles image, from an eval script).

* sampler id = ``--rl-sampler-node`` (the launcher's node name in a pair, the
  buffer shard with several samplers: ``--rl-buffer-shards``), else
  ``island<learner id>`` (a separate sampler launch beside synced trainers, as
  the Miles sampler); a replacement incarnation resumes under the same id;
* store = ``--rl-resume-store`` (the run store the launcher mounts; the trainer
  reads ``sample-buffer/`` and writes ``policy-releases/`` there);
* rounds = ``--global-rounds`` (the launcher's share of the run per sampler),
  ``--groups-per-round`` x ``--samples-per-group`` responses per round;
* prompts = the train split of ``--data`` rendered with the model's chat
  template and ``--apply-chat-template-kwargs``; ``--rl-sampler-data-slot``
  keeps this sampler's slot (several samplers draw disjoint prompts);
* backlog = ``--rl-sampler-max-backlog-rounds`` rounds (default 4).

:func:`sampler_spec` and :func:`prompt_rows` are pure (unit tested);
:func:`run_sampler_island` needs the verl image. Not yet run on a GPU.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from yeto.rl.engine.sample_roles import DEFAULT_MAX_BACKLOG_ROUNDS

# the spec names the engine factory by its import path; an adapter does not import
# another adapter (import boundaries, yeto/rl/engine/README.md)
ENGINE_FACTORY = "yeto.rl.adapters.sampler_engines.from_spec:build"
ENGINE_ADAPTER = "vllm-server"
MAX_PROMPT_TOKENS = 512  # the verl island's max_prompt (island_entry.main)
# launcher --reward-function spelling -> the Miles-style reward the engine sampler calls
SAMPLER_REWARDS = {
    "gsm8k_reward:score": "yeto.rl.gsm8k_reward:score",
    "yeto.rl.gsm8k_reward:score": "yeto.rl.gsm8k_reward:score",
    "yeto.rl.gsm8k_reward.score": "yeto.rl.gsm8k_reward:score",
}


class SamplerIslandError(ValueError):
    """The flags of a verl sampler island are incomplete."""


def sampler_reward(reward_function: str) -> str:
    try:
        return SAMPLER_REWARDS[reward_function]
    except KeyError:
        raise SamplerIslandError(f"no engine sampler reward for {reward_function!r} "
                                 f"(mapped: {sorted(SAMPLER_REWARDS)})") from None


def prompt_rows(rows: Iterable[Mapping[str, Any]], render: Callable[[list], str]) -> list[dict[str, str]]:
    """``{"text", "label"}`` records of the dataset rows (``messages``, ``label``)."""
    out = []
    for row in rows:
        messages = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
        out.append({"text": render(messages), "label": str(row["label"])})
    if not out:
        raise SamplerIslandError("the prompt data has no rows")
    return out


def sampler_spec(args: Any, *, model_path: str, prompts_path: str, card_model: str, engine_version: str,
                 device_family: str = "cuda", event_tape: str, python: str = sys.executable,
                 gpu_memory_utilization: float = 0.5) -> dict[str, Any]:
    """The engine sampler spec of one verl sampler island (module doc)."""
    sampler_id = getattr(args, "rl_sampler_node", None) or f"island{int(args.learner_id)}"
    store = getattr(args, "rl_resume_store", None)
    if not store:
        raise SamplerIslandError("a verl sampler island needs --rl-resume-store (the run store)")
    groups = int(args.groups_per_round)
    backlog = int(getattr(args, "rl_sampler_max_backlog_rounds", None) or DEFAULT_MAX_BACKLOG_ROUNDS)
    max_len = MAX_PROMPT_TOKENS + int(args.rollout_max_response_len)
    spec: dict[str, Any] = {
        "sampler_id": str(sampler_id),
        "engine": {"factory": ENGINE_FACTORY, "adapter": ENGINE_ADAPTER,
                   "server": {"base_dir": str(model_path), "card_model": card_model,
                              "device_family": device_family, "engine_version": engine_version,
                              "precision": "bf16", "quant_scheme": "bf16", "topology": "tp1",
                              "max_model_len": max_len, "port": 30000, "python": python,
                              "start_timeout_s": 1200.0,
                              "extra_args": ["--gpu-memory-utilization", f"{float(gpu_memory_utilization):g}"]},
                   "options": {"tp": 1, "served_model_name": "yeto-policy"}},
        "sampling": {"max_tokens": int(args.rollout_max_response_len), "temperature": 1.0, "top_p": 1.0,
                     "top_k": -1, "n": int(args.samples_per_group), "seed": int(args.seed)},
        "num_rollouts": int(args.global_rounds),
        "max_backlog_groups": groups * backlog,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "base_version": 0, "base_dir": str(model_path),
        "store": str(store),
        "prompts": {"path": str(prompts_path), "input_key": "text", "label_key": "label",
                    "groups_per_round": groups, "format": "text"},
        "reward": sampler_reward(args.reward_function), "reward_kind": "miles",
        "checks": {"expected_expert_matrices": None, "fixed_prompts": []},
        "event_tape": str(event_tape),
    }
    if getattr(args, "rl_sampler_data_slot", None):
        spec["prompts"]["data_slot"] = str(args.rl_sampler_data_slot)
    if getattr(args, "rl_buffer_shards", None):
        spec["buffer_shards"] = list(args.rl_buffer_shards)
    if getattr(args, "rl_sampler_min_round_seconds", None) is not None:
        spec["min_round_s"] = float(args.rl_sampler_min_round_seconds)
    return spec


def run_sampler_island(args: Any, *, manifest: Mapping[str, Any], card: str, output: Path) -> int:
    """The island process of a verl sampler island (verl image)."""
    import os

    import pandas as pd
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoTokenizer

    from yeto.rl.engine import engine_sampler_entry as entry

    model_path = snapshot_download(args.model, revision=args.model_revision)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    chat = json.loads(args.apply_chat_template_kwargs) if args.apply_chat_template_kwargs else {}
    frame = pd.read_parquet(hf_hub_download(args.data, "train.parquet", repo_type="dataset",
                                            revision=args.data_revision))
    work = Path(os.path.expanduser("~/yeto-verl/sampler"))
    work.mkdir(parents=True, exist_ok=True)
    prompts = work / "prompts.jsonl"
    rows = prompt_rows((row for _, row in frame.iterrows()), lambda messages: tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **chat))
    prompts.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    versions = manifest.get("versions") or {}
    spec = sampler_spec(args, model_path=model_path, prompts_path=str(prompts),
                        card_model=card,
                        engine_version=str(versions.get("vllm")),
                        device_family=str(manifest.get("device_family") or "cuda"),
                        event_tape=os.path.expanduser(args.event_tape),
                        gpu_memory_utilization=float(os.environ.get("YETO_VERL_GPU_MEM", "0.5")))
    (output / f"verl-sampler-spec-{args.learner_id}.json").write_text(json.dumps(spec, indent=1))
    try:
        rounds = entry.run(spec)
    except entry.SamplerSpecError as error:
        print(f"[yeto-verl] sampler spec rejected: {error}", file=sys.stderr, flush=True)
        return 2
    print(f"[yeto-verl] sampler island {spec['sampler_id']} ran {rounds} rounds", flush=True)
    entry.hold_after_reclaim()  # reclaimed: let the cloud run the island again
    return 0
