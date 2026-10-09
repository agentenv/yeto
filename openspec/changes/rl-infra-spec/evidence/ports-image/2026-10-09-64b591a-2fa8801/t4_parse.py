"""S19 #1: parse_args check of image 64b591a-2fa8801 on Modal T4 (one container, <=15 min)."""
import modal
IMG = "ghcr.io/michaellchung/yeto-miles-ports@sha256:fa2413be4c4fcf066437f946365f01392d6f884002f8fa19c770c12c08f2acf0"
img = (modal.Image.from_registry(IMG)
       .add_local_dir("/home/michael/work/s19-critic", "/yeto",
                      ignore=["**/.git", "**/__pycache__", "openspec/**", "docs/**"]))
app = modal.App("yeto-s19-img-parse-20261009a", image=img)

@app.function(gpu="T4", timeout=900, cpu=4, memory=16384)
def check():
    import subprocess, json, os
    out = {}
    out["nvidia"] = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                   capture_output=True, text=True).stdout.strip()
    out["manifest"] = json.load(open("/opt/yeto/image-manifest.json"))["miles"]["commit"]
    out["miles_head"] = subprocess.run(["git", "-C", "/root/miles", "rev-parse", "HEAD"],
                                       capture_output=True, text=True).stdout.strip()
    env = dict(os.environ, PYTHONPATH="/root/miles:/yeto")
    # try6: all three groups (try5: only the two failing families, full tracebacks.  The miles
    # arguments dump floods "Captured stdout call"; keep each captured block
    # but cut it at 1500 chars so the tracebacks stay visible.
    r = subprocess.run(["python", "-m", "pytest", "-p", "no:cacheprovider", "-rfE", "--run-ray-local",
                        "--tb=long", "--show-capture=no",
                        "tests/test_rl_miles_adapter_config.py::test_upstream_parse_args_accepts_translation", "tests/test_rl_miles_adapter_config.py::test_upstream_parse_args_accepts_critic_family_on_pinned_image", "tests/test_rl_algorithm_flags_upstream.py"],
                       cwd="/yeto", env=env, capture_output=True, text=True)
    out["rc"] = r.returncode
    text = r.stdout + r.stderr
    import re
    blocks = re.split(r"-+ Captured (stdout|stderr|log) (call|setup) -+\n", text)
    kept = [blocks[0]]
    for i in range(1, len(blocks), 3):
        kept.append(blocks[i + 2][:1500] if i + 2 < len(blocks) else "")
    out["traceback"] = "".join(kept)[-60000:]
    return out

@app.local_entrypoint()
def main():
    import json
    res = check.remote()
    print(json.dumps(res, indent=1))
    open("/home/michael/work/s1-runs/s19-img-parse-20261009a/result.json", "w").write(json.dumps(res, indent=1))
