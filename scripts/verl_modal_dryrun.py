"""CPU dry run of the verl engine image on Modal (rl-verl-backend 1.1 acceptance).

Builds ``yeto.rl.adapters.verl.image`` (Modal caches it for the GPU runs) and,
on CPU only: checks the verl commit and library versions, that the read-back
patch is in place, imports the verl-side yeto modules, composes the hydra
config from ``config.build_overrides`` and reads back the asserted keys, runs
the three-way binding check with verl importable, and derives the canonical
LoRA specs of Qwen3-0.6B (must be the 392 expected names).

Usage (from the repo root):
  modal run scripts/verl_modal_dryrun.py > dryrun.json
"""

import json
import os
import subprocess
import sys

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, "/root/sky_workdir")  # inside the container (mounted below)

from yeto.rl.adapters.verl import image as verl_image  # noqa: E402

APP = "yeto-verl-dryrun-s17"
img = verl_image.modal_image(modal).add_local_dir(os.path.join(ROOT, "yeto"), "/root/sky_workdir/yeto",
                                                   copy=False)
app = modal.App(APP, image=img)

MODEL = "Qwen/Qwen3-0.6B"
MODEL_REV = "c1899de289a04d12100db370d81485cdf75e47ca"


@app.function(cpu=4, memory=16384, timeout=30 * 60)
def dryrun():
    os.environ["PYTHONPATH"] = "/root/sky_workdir"
    sys.path.insert(0, "/root/sky_workdir")
    out = {}

    def sh(cmd):
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd="/workspace/verl")
        return r.returncode, r.stdout[-3000:], r.stderr[-3000:]

    from yeto.rl.adapters.verl import config as vconf
    from yeto.rl.adapters.verl import island_entry, patch_verl
    from yeto.rl.adapters.verl.param_names import expected_qwen3_names

    out["runtime"] = island_entry.runtime_manifest(0)
    target = open(f"/workspace/verl/{patch_verl.TARGET}").read()
    out["patch_in_place"] = target.count("_yeto_after_add_lora(self, VLLM_LORA_INT_ID)") == 1
    out["imports"] = sh("/workspace/verl/.venv/bin/python -c 'import yeto.rl.adapters.verl.trainer, "
                        "yeto.rl.adapters.verl.ports_impl, yeto.rl.adapters.verl.verl_main, "
                        "yeto.rl.adapters.verl.vllm_readback; "
                        "from verl.trainer.ppo.v1 import get_trainer_cls; print(get_trainer_cls(\"yeto_sync\"))'")
    cfg = vconf.VerlRunConfig(model_path="/tmp/model", train_file="/tmp/t.parquet", val_file="/tmp/v.parquet",
                              out_dir="/tmp/out", chat_template_kwargs={"enable_thinking": False},
                              reward_path="/root/sky_workdir/yeto/rl/adapters/verl/reward_fn.py",
                              reward_name="compute_score_gsm8k")
    overrides = vconf.build_overrides(cfg) + ["actor_rollout_ref.rollout.agent.num_workers=2",
                                              "reward.num_workers=2"]
    code, stdout, stderr = 0, "", ""
    r = subprocess.run(["/workspace/verl/.venv/bin/python", "-m", "verl.trainer.main_ppo", "--cfg", "job",
                        *overrides], capture_output=True, text=True, cwd="/workspace/verl")
    out["hydra_rc"] = r.returncode
    out["hydra_err_tail"] = r.stderr[-2500:]
    import yaml

    composed = yaml.safe_load(r.stdout) if r.returncode == 0 else {}

    def lookup(path):
        node = composed
        for part in path.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        return node

    out["asserted"] = {k: {"want": island_entry._override_value(overrides, k), "got": lookup(k)}
                       for k in vconf.ASSERTED_KEYS}
    out["composed_extra"] = {k: lookup(k) for k in (
        "data.apply_chat_template_kwargs", "reward.custom_reward_function.path",
        "reward.custom_reward_function.name", "algorithm.rollout_correction.rollout_is",
        "algorithm.rollout_correction.rollout_is_threshold", "actor_rollout_ref.actor.clip_ratio",
        "actor_rollout_ref.actor.clip_ratio_low", "actor_rollout_ref.actor.clip_ratio_high",
        "actor_rollout_ref.actor.loss_agg_mode", "actor_rollout_ref.actor.fsdp_config.model_dtype",
        "actor_rollout_ref.model.lora.merge", "trainer.v1.trainer_mode", "actor_rollout_ref.rollout.mode",
        "actor_rollout_ref.rollout.checkpoint_engine.backend")}
    from yeto.rl.adapters.verl.binding import check_bindings

    try:
        out["binding"] = check_bindings(verl_available=True)
    except Exception as exc:  # noqa: BLE001
        out["binding"] = {"error": repr(exc)}
    try:
        from huggingface_hub import snapshot_download

        from yeto.rl.export import derive_peft_lora_specs

        path = snapshot_download(MODEL, revision=MODEL_REV, allow_patterns=["*.json"])
        specs = derive_peft_lora_specs(path, None, rank=32, targets="all-linear")
        names = sorted(s.name for s in specs)
        out["specs"] = {"count": len(names), "equal_expected": names == expected_qwen3_names(28),
                        "first": names[:2], "shapes": {s.name: list(s.shape) for s in specs[:4]}}
    except Exception as exc:  # noqa: BLE001
        import traceback

        out["specs"] = {"error": repr(exc), "trace": traceback.format_exc()[-1500:]}
    print(json.dumps(out, indent=1, default=str))
    return out


@app.local_entrypoint()
def main():
    print(json.dumps(dryrun.remote(), indent=1, default=str))
