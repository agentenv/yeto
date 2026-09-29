"""Regenerate the example specs (PluginRef SHA256s follow the current sources).

    PYTHONPATH=. /tmp/yeto-venv/bin/python openspec/changes/rl-algo-seq-and-adv/examples/make_examples.py
"""

import json
import pathlib

from yeto.rl.engine.algorithm import load_extensions

load_extensions()
from yeto.rl.algos import grpo_knobs, reward_pipeline as rp  # noqa: E402
from yeto.rl.engine.algorithm import AlgorithmSpec  # noqa: E402

HERE = pathlib.Path(__file__).parent
V2 = "yeto-rl-algorithm-spec-v2"
D = rp.dispatcher_ref().to_dict()
KL = {"placement": "reward", "coef": 0.01}
EXAMPLES = {
    "gspo": {"advantage": {"estimator": "gspo"}, "loss": {"eps_clip": 3e-4, "eps_clip_high": 4e-4}},
    "gspo_noclip": {"advantage": {"estimator": "gspo"}},
    "rpp": {"advantage": {"estimator": "reinforce_plus_plus", "whiten": True}, "kl": KL},
    "rpp_gamma": {"advantage": {"estimator": "reinforce_plus_plus", "whiten": True, "gamma": 0.99}},
    "rpp_baseline": {"advantage": {"estimator": "reinforce_plus_plus_baseline", "whiten": True},
                     "kl": KL},
    "maxrl": {"advantage": {"estimator": "grpo", "transform": "maxrl", "reward_binary": True,
                            "reward_postprocess": D} },
    "mapo": {"advantage": {"estimator": "grpo", "transform": "mapo", "reward_binary": True,
                           "reward_postprocess": D} },
    "gdpo": {"advantage": {"estimator": "grpo", "transform": "gdpo", "reward_postprocess": D,
                           "gdpo": {"components": [{"name": "correctness", "weight": 1.0},
                                                   {"name": "format", "weight": 1.0}],
                                    "whiten": True}},
             },
}
for name, body in EXAMPLES.items():
    spec = AlgorithmSpec.from_dict({"schema": V2, **body})
    if "transform" in body["advantage"]:
        spec = grpo_knobs.with_pipeline_plugins(spec)  # pipeline module PluginRefs
    (HERE / f"{name}.json").write_text(json.dumps(spec.structured_dict(), indent=2) + "\n")
