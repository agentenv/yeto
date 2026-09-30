"""Step 1 (yeto venv): translate AlgorithmSpecs to Miles argv with yeto's table."""
import json
from yeto.rl.engine.algorithm import AlgorithmSpec, load_extensions
from yeto.rl.engine.miles_adapter.algorithm_flags import algorithm_argv
load_extensions()
V2 = "yeto-rl-algorithm-spec-v2"
specs = {
    "cispo": {"policy_loss_variant": "cispo", "eps_clip": 0.2, "eps_clip_high": 0.28, "aggregation": "token"},
    "sapo": {"policy_loss_variant": "sapo", "sapo_tau_pos": 0.9, "sapo_tau_neg": 1.2},
    "gmpo": {"policy_loss_variant": "gmpo", "gmpo_log_clip_low": 0.3, "gmpo_log_clip_high": 0.5},
    "gmpo_token_REJECTED_BY_YETO": {"policy_loss_variant": "gmpo", "aggregation": "token"},
}
out = {}
for name, loss in specs.items():
    s = AlgorithmSpec.from_dict({"schema": V2, "loss": loss})
    out[name] = {"argv": ["--advantage-estimator", "grpo"] + algorithm_argv(s),
                 "yeto_rejections": s.rejections(), "sha256": s.sha256()}
print(json.dumps(out, indent=1))
