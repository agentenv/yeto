"""S26 E6 probe (Modal CPU, verl image): what vLLM receives from the tier 1 verl overrides.

The overrides come from ``yeto.rl.quant.tiers.vllm_online_overrides`` for the real
Qwen3-4B-Instruct-2507 checkpoint (computed on the client, passed as JSON). In the
pinned verl image (fork acad9875, vLLM 0.29) the container:

1. composes verl's ``ppo_trainer`` config with Hydra and these overrides;
2. turns the rollout section into verl's ``RolloutConfig`` (``omega_conf_to_dataclass``)
   and builds the vLLM argument dict as ``vLLMHttpServer`` does (verl's own FP8
   ``hf_overrides`` first, then ``**engine_kwargs``), then the CLI list with verl's
   ``build_cli_args_from_config``;
3. parses that list with vLLM's ``AsyncEngineArgs`` parser and reads
   ``hf_overrides`` / ``kv_cache_dtype``;
4. builds vLLM's ``Fp8Config`` from the received ``quantization_config`` and asks
   vLLM's ``is_layer_skipped`` for the fused Qwen3 modules.

No GPU, no model weights. Run: ``modal run tools/probes/s26_e6/verl_hf_overrides_probe.py
--overrides-json FILE --out FILE`` (client venv with modal).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

app = modal.App("yeto-s26-e6-verl-hf-overrides-probe")
if modal.is_local():  # the image is defined on the client; the container only runs probe()
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from yeto.rl.adapters.verl.image import modal_image
    from yeto.rl.adapters.verl.pins import VERL_ROOT

    image = modal_image(modal)
else:
    VERL_ROOT = "/workspace/verl"  # yeto.rl.adapters.verl.pins.VERL_ROOT
    image = modal.Image.debian_slim()

CONTAINER = r'''
import json, sys
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
overrides = ["actor_rollout_ref.rollout.name=vllm", *json.loads(sys.argv[1])]; layers = int(sys.argv[2])
with initialize_config_dir(config_dir=VERL_ROOT + "/verl/trainer/config", version_base=None):
    cfg = compose(config_name="ppo_trainer", overrides=overrides)
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config.rollout import RolloutConfig
OmegaConf.resolve(cfg)
section = cfg.actor_rollout_ref.rollout
try:
    rollout = omega_conf_to_dataclass(section, dataclass_type=RolloutConfig)
    source = "RolloutConfig"
except Exception as error:  # the composed-but-unwired config may miss fields the trainer fills
    print("RolloutConfig failed: " + repr(error)[:300], file=sys.stderr)
    rollout, source = section, "DictConfig"
engine_kwargs = rollout.get("engine_kwargs", {}).get("vllm", {}) or {}
engine_kwargs = {k: v for k, v in engine_kwargs.items() if v is not None}
verl_fp8 = {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8", "weight_block_size": [128, 128],
            "ignored_layers": [f"model.layers.{i}.mlp.gate" for i in range(layers)]}
args = {"quantization": rollout.quantization, "hf_overrides": {"quantization_config": verl_fp8}, **engine_kwargs}
from verl.workers.rollout.vllm_rollout.utils import build_cli_args_from_config
cli = build_cli_args_from_config(args)
# CPU container: vLLM cannot infer a device type; DeviceConfig's check only matters for
# an engine, not for parsing arguments, so it is skipped here (parsing is what is probed).
import vllm.config.device as _device
_device.DeviceConfig.__post_init__ = lambda self: None
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser
parser = AsyncEngineArgs.add_cli_args(FlexibleArgumentParser())
ns = parser.parse_args(["--model", "/nonexistent", *cli])
ea = AsyncEngineArgs.from_cli_args(ns)
qc = ea.hf_overrides["quantization_config"]
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
fp8 = Fp8Config.from_config(qc)
fused = {"qkv_proj": ["q_proj", "k_proj", "v_proj"], "gate_up_proj": ["gate_proj", "up_proj"]}
probe = {p: bool(is_layer_skipped(p, fp8.ignored_layers, fused)) for p in
         ["model.layers.0.self_attn.qkv_proj", "model.layers.0.self_attn.o_proj",
          "model.layers.35.self_attn.qkv_proj", "model.layers.0.mlp.gate_up_proj", "model.layers.35.mlp.down_proj"]}
import vllm
print("RESULT " + json.dumps({
    "vllm": vllm.__version__, "rollout_config_source": source, "rollout_quantization": rollout.quantization,
    "engine_kwargs_types": {k: type(v).__name__ for k, v in engine_kwargs.items()},
    "cli_flags": [c for c in cli if c.startswith("--")],
    "kv_cache_dtype": ea.kv_cache_dtype, "quantization": ea.quantization,
    "received_ignored_layers": sorted(fp8.ignored_layers), "weight_block_size": fp8.weight_block_size,
    "is_layer_skipped": probe}))
'''.replace("VERL_ROOT", repr(VERL_ROOT))


@app.function(image=image, cpu=2, memory=8192, timeout=900)
def probe(overrides: list[str], layers: int) -> str:
    import subprocess

    done = subprocess.run([f"{VERL_ROOT}/.venv/bin/python", "-c", CONTAINER, json.dumps(overrides), str(layers)],
                          capture_output=True, text=True, cwd=VERL_ROOT)
    return json.dumps({"rc": done.returncode, "stdout": done.stdout[-20000:], "stderr": done.stderr[-8000:]})


@app.local_entrypoint()
def main(overrides_json: str, out: str) -> None:
    spec = json.loads(Path(overrides_json).read_text())
    raw = json.loads(probe.remote(spec["overrides"], int(spec["num_layers"])))
    Path(out).write_text(json.dumps(raw, indent=1))  # raw first
    line = next((l for l in raw["stdout"].splitlines() if l.startswith("RESULT ")), None)
    if raw["rc"] != 0 or line is None:
        print(raw["stderr"][-3000:])
        raise SystemExit(f"probe failed rc={raw['rc']}")
    result = json.loads(line[len("RESULT "):])
    expected = sorted(json.loads(spec["overrides"][2].split("=", 1)[1][1:-1])["quantization_config"]["ignored_layers"])
    result["ignored_layers_equal_sent"] = result["received_ignored_layers"] == expected
    Path(out).with_suffix(".result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "received_ignored_layers"}, indent=1))
