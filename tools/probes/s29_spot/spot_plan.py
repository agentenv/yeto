"""S29 spot: the spec of one Nebius spot rollout island (Qwen3-0.6B, TB2.1 x Codex in boat).

``python -m yeto.rl.spot_rollout.sampler --spec SPEC.json`` on one preemptible GPU VM:
SGLang TP1 + the records proxy, Harbor trials in boat sandboxes, weights from the
npu1 syncer as a receive-only subscriber, groups to relay channel ``channel`` (the
trainer island ``channel`` on npu1). The VM reaches the syncer at
``127.0.0.1:29400`` through a reverse SSH tunnel opened from npu1 (npu1 takes no
inbound connection but its own SSH port); the boat sandboxes reach the records
proxy on the VM's public IP.

Boat budget (S29: 22 creations left for the UTC day): VM reuse is on
(``YETO_BOAT_POOL``), so a sandbox VM serves the next trial instead of a new
create; each island keeps at most ``max_in_flight`` (4) active sandboxes and
starts at most ``starts_per_hour`` per hour.

    python tools/probes/s29_spot/spot_plan.py --island 0 --public-ip 1.2.3.4 \\
        --engine-version 0.5.x --python /usr/bin/python3 --out /root/spot > spec+env.json
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

PROXY_PORT = 31600
MAX_MODEL_LEN = 40960  # Qwen3-0.6B max_position_embeddings
TURN_MAX_TOKENS = 4096
MAX_PROMPT_LEN = 12288
MAX_RESPONSE_LEN = 20480  # trainer --seq-len 32768 = MAX_PROMPT_LEN + MAX_RESPONSE_LEN
PROBE_PAIRS = [["The capital of France is", " Paris. It is known for the Eiffel Tower and the Louvre museum."]]


def spec(*, island: int, islands: int, base: str, data: str, out: str, engine_version: str, python: str,
         card_model: str, samples_per_group: int = 4, max_in_flight: int = 4) -> dict[str, Any]:
    server = {"base_dir": base, "card_model": card_model, "device_family": "cuda", "engine_version": engine_version,
              "precision": "bf16", "quant_scheme": "bf16", "topology": "tp1", "max_model_len": MAX_MODEL_LEN,
              "python": python, "in_place_reload": True, "start_timeout_s": 1200.0, "request_timeout_s": 1800.0,
              "extra_args": ["--mem-fraction-static", "0.7"]}
    return {
        "sampler_id": f"s29-spot-{island}",
        "engine": {"factory": "yeto.rl.adapters.sampler_engines.from_spec:build", "adapter": "sglang-server",
                   "server": server, "options": {"tp": 1}},
        "sampling": {"max_tokens": TURN_MAX_TOKENS, "temperature": 1.0, "top_p": 1.0, "n": samples_per_group},
        "max_backlog_groups": 1,
        "max_prompt_tokens": MAX_MODEL_LEN - TURN_MAX_TOKENS,
        "base_version": 0, "base_dir": base, "poll_s": 10.0,
        "prompts": {"path": f"{data}/prompts.jsonl", "input_key": "prompt", "groups_per_round": 1,
                    "format": "agentic", "task_key": "metadata", "data_slot": f"{island}/{islands}"},
        "agentic": {"template": "qwen3", "tokenizer": base, "model": "qwen3-0.6b",
                    "turn_max_tokens": TURN_MAX_TOKENS, "max_response_len": MAX_RESPONSE_LEN,
                    "max_prompt_len": MAX_PROMPT_LEN, "max_in_flight": max_in_flight, "min_samples": 2,
                    "proxy": {"host": "0.0.0.0", "port": PROXY_PORT},
                    "task_defaults": {"agent_timeout_s": 900.0, "wall_clock_s": 1500.0},
                    "compaction": {"split": True}, "require_agent": "codex"},
        "checks": {"expected_expert_matrices": 0, "require_fixed_answers": False, "fixed_max_tokens": 256,
                   "probe_pairs": PROBE_PAIRS, "max_probe_jump": 0.5},
        "event_tape": f"{out}/events.jsonl",
        # spot channels (yeto.rl.spot_rollout.sampler module doc)
        "syncer": "127.0.0.1:29400", "channel": island, "base_model": base, "work_dir": f"{out}/work",
        "model_name": "qwen3", "layout_timeout_s": 3600.0, "syncer_lost_s": 1800.0, "relay_gone_s": 900.0,
    }


def gsm8k_spec(agentic: dict[str, Any], *, data: str, samples_per_group: int = 4,
               groups_per_round: int = 1) -> dict[str, Any]:
    """The same island on GSM8K chat prompts with the numeric reward: no boat sandbox,
    seconds per group, rewards that are not all 0 (so merges move the weights). Used
    for the long run and the fault drills while the day's boat creations are spent."""
    out = {k: v for k, v in agentic.items() if k != "agentic"}
    out["prompts"] = {"path": f"{data}/gsm8k-tasks512.jsonl", "input_key": "messages", "label_key": "label",
                      "groups_per_round": groups_per_round, "format": "chat",
                      "data_slot": agentic["prompts"]["data_slot"]}
    out["sampling"] = {**agentic["sampling"], "max_tokens": 1024, "n": samples_per_group}
    out["reward"], out["reward_kind"] = "yeto.rl.gsm8k_reward:score", "miles"
    out["max_backlog_groups"] = 2
    return out


def fake_spec(gsm: dict[str, Any], *, reload_every: int = 1, version_board: bool = True) -> dict[str, Any]:
    """The GSM8K island on the CPU fake engine (``sampler_engines.fake``): every link of
    the spot island (subscriber, materialize, relay, version board) without a GPU, for
    local drills and long runs next to the trainers."""
    out = dict(gsm)
    out["engine"] = {"factory": "yeto.rl.adapters.sampler_engines.from_spec:build", "adapter": "fake",
                     "kwargs": {"max_model_len": 4096}}
    out["checks"] = {"expected_expert_matrices": 0, "require_fixed_answers": False, "probe_pairs": []}
    out["max_prompt_tokens"] = 2048  # the trainer rows: --seq-len 4096 - response 2048
    out["reload_every"], out["version_board"], out["telemetry"] = int(reload_every), bool(version_board), False
    return out


def env(*, data: str, public_ip: str, max_in_flight: int = 4, starts_per_hour: int = 8) -> dict[str, str]:
    return {
        "HARBOR_ENV_TYPE": "boat", "HARBOR_TASKS_DIR": f"{data}/harbor-tasks",
        "HARBOR_TRIALS_DIR": "/tmp/harbor-trials", "HARBOR_SETUP_RETRIES": "1",
        "HARBOR_ENV_KWARGS": json.dumps({"ttl_s": 3600}, separators=(",", ":")),
        "AGENT_MODEL_NAME": "qwen3-0.6b", "AGENT_TIMEOUT": "900", "AGENT_TRIAL_TIMEOUT": "900",
        "YETO_BOAT_MAX_ACTIVE": str(max_in_flight), "YETO_BOAT_STARTS_PER_MINUTE": "4",
        "YETO_BOAT_STARTS_PER_HOUR": str(starts_per_hour),
        "YETO_BOAT_POOL": "1", "YETO_BOAT_POOL_IDLE_TTL_S": "1500", "YETO_BOAT_POOL_IDLE_VM_TTL_S": "1800",
        "YETO_BOAT_POOL_MAX_IDLE": str(max_in_flight), "YETO_BOAT_CODEX_CACHE": "1",
        "YETO_CODEX_CONTEXT_WINDOW": str(MAX_MODEL_LEN), "YETO_CODEX_AUTO_COMPACT_RATIO": "0.85",
        "YETO_RECORDS_PROXY_PUBLIC_BASE": f"http://{public_ip}:{PROXY_PORT}",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--island", type=int, required=True)
    p.add_argument("--islands", type=int, default=2)
    p.add_argument("--public-ip", required=True)
    p.add_argument("--engine-version", required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--card-model", required=True)
    p.add_argument("--base", default="/root/models/Qwen3-0.6B")
    p.add_argument("--data", default="/root/s29data")
    p.add_argument("--out", default="/root/spot")
    p.add_argument("--mode", choices=("tb21", "gsm8k", "fake"), default="tb21")
    p.add_argument("--syncer", default="127.0.0.1:29400")
    p.add_argument("--reload-every", type=int, default=1)
    p.add_argument("--no-version-board", action="store_true")
    p.add_argument("--groups-per-round", type=int, default=1)
    a = p.parse_args(argv)
    sp = spec(island=a.island, islands=a.islands, base=a.base, data=a.data, out=a.out,
              engine_version=a.engine_version, python=a.python, card_model=a.card_model)
    if a.mode in ("gsm8k", "fake"):
        sp = gsm8k_spec(sp, data=a.data, groups_per_round=a.groups_per_round)
    if a.mode == "fake":
        sp = fake_spec(sp, reload_every=a.reload_every, version_board=not a.no_version_board)
    sp["syncer"] = a.syncer
    print(json.dumps({"spec": sp, "env": env(data=a.data, public_ip=a.public_ip)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
