"""CPU dry run of the verl fully_async path on Modal (agentic-rollout-utilization 6.4b,
pre-launch review S19-VERL-64B section 1 item 3).  No GPU.

Rebuilds the verl image (patch_verl.py has the second, llm_server.py hook, so
the image id changes) and, on CPU only: both patches in place, the fully_async
yeto modules import (incl. the Ray-actor subclass), hydra composes
``fully_async_ppo_trainer`` from ``fully_async_round.fully_async_run_overrides``
and every asserted key reads back; the async_training values equal 6.4a's
``fully_async_overrides``.  Prints the image id.

Usage (from the repo root):
  modal run scripts/verl_fa_modal_dryrun.py > dryrun.json

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

APP = "yeto-verl-fa-dryrun-s19"
img = verl_image.modal_image(modal).add_local_dir(os.path.join(ROOT, "yeto"), "/root/sky_workdir/yeto",
                                                   copy=False)
app = modal.App(APP, image=img)

MODEL = "Qwen/Qwen3-0.6B"
MODEL_REV = "c1899de289a04d12100db370d81485cdf75e47ca"


@app.function(cpu=2, memory=8192, timeout=20 * 60)
def dryrun():
    os.environ["PYTHONPATH"] = "/root/sky_workdir"
    sys.path.insert(0, "/root/sky_workdir")
    out = {"image_id": os.environ.get("MODAL_IMAGE_ID"), "task_id": os.environ.get("MODAL_TASK_ID")}
    py = "/workspace/verl/.venv/bin/python"

    from yeto.rl.adapters.verl import config as vconf
    from yeto.rl.adapters.verl import island_entry, patch_verl
    from yeto.rl.adapters.verl.fully_async_round import (FULLY_ASYNC_ASSERTED_KEYS,
                                                         fully_async_run_overrides)
    from yeto.rl.adapters.verl.fully_async_translate import fully_async_overrides

    out["runtime"] = island_entry.runtime_manifest(0)
    out["patches_in_place"] = {t: open(f"/workspace/verl/{t}").read().count(h) == 1
                               for t, _a, h in patch_verl.PATCHES}
    r = subprocess.run([py, "-c", "import yeto.rl.adapters.verl.fully_async_runner as m, "
                        "yeto.rl.adapters.verl.verl_main as v; print(m.YetoFullyAsyncTrainer, "
                        "m.YetoFullyAsyncTaskRunner, v.main_fully_async)"],
                       capture_output=True, text=True, cwd="/workspace/verl")
    out["imports"] = [r.returncode, r.stdout[-1500:], r.stderr[-2500:]]
    groups, rounds, limit = 8, 5, 1
    cfg = vconf.VerlRunConfig(model_path="/tmp/model", train_file="/tmp/t.parquet", val_file="/tmp/v.parquet",
                              out_dir="/tmp/out", chat_template_kwargs={"enable_thinking": False},
                              reward_path="/root/sky_workdir/yeto/rl/adapters/verl/reward_fn.py",
                              reward_name="compute_score_gsm8k", groups_per_round=groups,
                              samples_per_group=4, total_rounds=rounds, correction="tis")
    sync = vconf.build_overrides(cfg) + ["actor_rollout_ref.rollout.agent.num_workers=2",
                                         "reward.num_workers=2"]
    overrides = fully_async_run_overrides(sync, limit, groups_per_round=groups, rounds=rounds)
    out["overrides"] = overrides
    # Same entry as the island (verl_main.main_fully_async): the module is imported, not
    # run with -m (the first GPU run failed here; the -m form hid it).
    entry = ("from verl.experimental.fully_async_policy import fully_async_main as f; "
             "from yeto.rl.adapters.verl.verl_main import fully_async_hydra_entry as e; e(f)()")
    r = subprocess.run([py, "-c", entry, "--cfg", "job", *overrides],
                       capture_output=True, text=True, cwd="/workspace/verl")
    out["hydra_rc"] = r.returncode
    out["hydra_err_tail"] = r.stderr[-2500:]
    import yaml

    composed = yaml.safe_load(r.stdout) if r.returncode == 0 else {}

    def lookup(path):
        node = composed
        for part in path.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        return node

    keys = tuple(vconf.ASSERTED_KEYS) + FULLY_ASYNC_ASSERTED_KEYS
    out["asserted"] = {k: {"want": island_entry._override_value(overrides, k), "got": lookup(k)} for k in keys}

    def norm(v):
        t = str(v).strip().lower()
        try:
            return repr(float(t))
        except ValueError:
            return t

    out["asserted_mismatch"] = [k for k, v in out["asserted"].items()
                                if v["want"] is not None and norm(v["got"]) != norm(v["want"])]
    want = fully_async_overrides(limit, samples_per_round=groups, ppo_mini_batch_size=groups)
    out["translate_6_4a"] = {k: {"want": v, "got": lookup(k)} for k, v in want.items()}
    out["translate_equal"] = all(norm(lookup(k)) == norm(v) for k, v in want.items())
    out["composed_extra"] = {k: lookup(k) for k in (
        "rollout.total_rollout_steps", "rollout.nnodes", "actor_rollout_ref.rollout.checkpoint_engine.backend",
        "actor_rollout_ref.actor.ppo_mini_batch_size", "algorithm.rollout_correction.bypass_mode",
        "algorithm.rollout_correction.rollout_is", "trainer.use_v1")}
    out["pass"] = (not out["runtime"]["problems"] and all(out["patches_in_place"].values())
                   and out["imports"][0] == 0 and out["hydra_rc"] == 0 and not out["asserted_mismatch"]
                   and out["translate_equal"])
    text = json.dumps(out, indent=1, default=str)
    print(text)
    return text  # a string: the local side has no torch to unpickle verl objects


@app.local_entrypoint()
def main():
    print("YETO_DRYRUN_RESULT " + json.dumps(json.loads(dryrun.remote())))
