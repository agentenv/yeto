"""Overlong presets of the S27 efficiency algorithm spec (``algo-eff-miles.json``).

Each preset is ``algo-eff-miles.json`` plus one of the existing yeto overlong
mechanisms (no new mechanism):

* ``filter``  -- ``sampling.overlong_filter``: a response cut at
  ``--rollout-max-response-len`` keeps its reward in the group statistics but
  gets loss mask 0 (``yeto.rl.algos.sample_filters``, DAPO overlong filtering).
* ``pen2048`` / ``pen3072`` / ``pen4096`` -- ``advantage.reward_shapers``
  ``overlong_penalty`` (``yeto.rl.algos.reward_pipeline``, DAPO soft overlong
  punishment) through the yeto reward dispatcher. ``max_length`` must equal the
  run's ``--rollout-max-response-len`` (the launcher refuses a larger value).
  ``pen2048`` penalises 1536 < len <= 2048 at today's cap; ``pen3072`` and
  ``pen4096`` start the penalty at 2048, so a response of at most 2048 tokens
  keeps today's reward.

Nothing selects a preset by default: ``algo-eff-miles.json`` is unchanged and a
run uses a preset only through ``--rl-algorithm-spec``.

usage (cwd = repository root): python -m tools.eval.efficiency.overlong_specs [--check]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path("tools/eval/efficiency/algo-eff-miles.json")
PRESETS: dict[str, dict] = {
    "filter": {"sampling": {"overlong_filter": True}},
    "pen2048": {"shaper": {"max_length": 2048, "cache_length": 512}},
    "pen3072": {"shaper": {"max_length": 3072, "cache_length": 1024}},
    "pen4096": {"shaper": {"max_length": 4096, "cache_length": 2048}},
}


def preset_path(name: str) -> Path:
    return BASE.with_name(f"algo-eff-miles-overlong-{name}.json")


def build(name: str, base: dict | None = None):
    """The AlgorithmSpec of one preset (plugins listed by ``with_pipeline_plugins``)."""

    from yeto.rl.algos import grpo_knobs as gk
    from yeto.rl.algos import reward_pipeline as rp
    from yeto.rl.engine.algorithm import AlgorithmSpec

    payload = json.loads(json.dumps(base if base is not None else json.loads(BASE.read_text())))
    patch = PRESETS[name]
    if "sampling" in patch:
        payload["sampling"].update(patch["sampling"])
    if "shaper" in patch:
        payload["advantage"]["reward_postprocess"] = rp.dispatcher_ref().to_dict()
        payload["advantage"]["reward_shapers"] = [{"name": "overlong_penalty", **patch["shaper"]}]
    return gk.with_pipeline_plugins(AlgorithmSpec.from_dict(payload))


def render(name: str) -> str:
    return json.dumps(build(name).to_dict(), indent=2, sort_keys=True) + "\n"


def main(argv: list[str]) -> int:
    stale = []
    for name in PRESETS:
        path, text = preset_path(name), render(name)
        if "--check" in argv:
            if not path.exists() or path.read_text() != text:
                stale.append(str(path))
        else:
            path.write_text(text)
            print(path)
    if stale:
        print("stale presets (rerun without --check): " + " ".join(stale), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
