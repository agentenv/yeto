"""Print sha256 + translated algorithm argv of pre-existing specs (run under two trees and diff)."""
import glob, json, os, sys
from yeto.rl.engine.algorithm import AlgorithmSpec, BOUNDED_NONZERO_STD_FILTER, load_extensions
load_extensions()
from yeto.rl.engine.miles_adapter.algorithm_flags import algorithm_argv
root = sys.argv[1]
specs = {
    "default": AlgorithmSpec(),
    "v1_bounded": AlgorithmSpec(advantage_estimator="grpo", kl_coef=0.0,
                                dynamic_sampling_filter=BOUNDED_NONZERO_STD_FILTER,
                                dynamic_sampling_max_replacements=2),
    "kl_loss": AlgorithmSpec(kl={"placement": "loss", "coef": 0.01, "estimator": "k3"}),
    "clip_higher": AlgorithmSpec(loss={"eps_clip": 0.2, "eps_clip_high": 0.28}),
    "whiten_rpp": AlgorithmSpec(advantage={"estimator": "reinforce_plus_plus", "whiten": True}),
    "cispo": AlgorithmSpec(loss={"policy_loss_variant": "cispo", "eps_clip": 0.2, "eps_clip_high": 0.28}),
    "tis": AlgorithmSpec(correction={"method": "tis", "tis_clip": 2.0, "tis_clip_low": 0.0}),
}
for path in sorted(glob.glob(f"{root}/examples/rl_algorithms/*.json")
                   + glob.glob(f"{root}/openspec/changes/*/examples/*.json")):
    specs[os.path.relpath(path, root)] = AlgorithmSpec.from_json_file(path)
for name, spec in specs.items():
    print(name, spec.sha256(), json.dumps(algorithm_argv(spec)))
