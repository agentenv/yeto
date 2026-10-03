"""4-card L40S (T1R1S2<->T1R3S0) variant. Local reconstruction of the Miles argv + runtime fingerprint for the A4 args (n2run-style). usage: python fp_local.py <repo> [extra cli args...]"""
import sys, json, os
repo = sys.argv[1]; extra = sys.argv[2:]
sys.path.insert(0, repo); sys.path.insert(0, repo + "/tests")
from pathlib import Path
from types import SimpleNamespace
import tempfile
class MP:  # minimal monkeypatch
    def __getattr__(self, n): 
        import _pytest.monkeypatch as m
        return getattr(self._m, n)
import _pytest.monkeypatch as _m
mp = _m.MonkeyPatch()
from rl_e2e_launch import island_run, learner_from_run
RES = "/home/michael/work/gpu-b1-runs/cfg/resources-4.json"
cli = ["--gpu", "nebius:4xl40s@eu-north1", "--on-demand", "--model", "Qwen/Qwen3-0.6B", "--model-revision", "c1899de289a04d12100db370d81485cdf75e47ca",
 "--data", "zhuzilin/gsm8k", "--data-revision", "0cbd9f31d91ac21a7613dcbc7fef992adac459ae", "--reward-function", "gsm8k_reward:score", "--tuning", "lora", "--lora-r", "16",
 "--lora-targets", "all-linear", "--fragments", "1", "--pipeline", "1", "--rollout-batch-size", "4", "--n-samples-per-prompt", "8", "--rollout-max-response-len", "384",
 "--seq-len", "1024", "--inner-lr", "1e-5", "--seed", "17", "--apply-chat-template-kwargs", '{"enable_thinking": false}', "--trust-remote-code",
 "--total-steps", "12", "--rl-placement", "fixed-partition", "--rl-rollout-gpus", "1", "--rl-standby-gpus", "2", "--rl-elastic", "--rl-elastic-declare-cells",
 "--rl-elastic-cells", "c0,c1,c2", "--rl-elastic-resources", RES, "--rl-elastic-initial-config", "T1R1S2", "--rl-observe-timeline",
 "--rl-single-island-no-sync", "--controller", "local"] + extra
run = island_run(tuple(cli), mp)
tmp = Path(tempfile.mkdtemp())
args, env = learner_from_run(run, tmp / "home")
from yeto.rl import learner
from yeto.rl.engine.run_config import resolve_rl_run_config
class GPTModelProvider(SimpleNamespace): pass
prov = GPTModelProvider(hidden_size=1024, num_attention_heads=16, num_layers=28, ffn_hidden_size=3072, num_query_groups=8, kv_channels=128,
    multi_latent_attention=False, num_moe_experts=None, seq_length=40960, layernorm_epsilon=1e-6, rotary_base=1000000, vocab_size=151936,
    max_position_embeddings=40960, qk_layernorm=True, gated_linear_unit=True, share_embeddings_and_output_weights=True, add_bias_linear=False,
    add_qkv_bias=False, activation_func=None)
snap = "/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
rc = resolve_rl_run_config(args, model_path=snap, rollout_model_path=None, prompt_path="/root/yeto-rl/prompts.jsonl", eval_prompt_path=None,
    provider=prov, target_modules=["down_proj","gate_proj","k_proj","o_proj","q_proj","up_proj","v_proj"], yeto_policy_sync=False)
pl = learner.build_ports_launch(args, rc, ())
from yeto.rl.engine.miles_adapter.entry import ports_runtime_fingerprint
print(json.dumps({"fp": ports_runtime_fingerprint(pl), "argv": list(pl.argv)}))
