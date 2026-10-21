"""S26 E6: check the pinned Miles ports image on a Modal CPU container (no GPU).

Asserts in the image: the SGLang and Miles checkouts are at the pins, the
image manifest names the same commits, the SGLang FP8 activation-cast fix is
in Qwen2 MLP / Qwen3 attention, and Miles has the FP8 ignore list.

    python tools/probes/s26_e6/image_check_modal.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def check(pins: dict) -> dict:
    import subprocess

    def head(path: str) -> str:
        return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()

    manifest = json.load(open("/opt/yeto/image-manifest.json"))
    models = Path("/sgl-workspace/sglang/python/sglang/srt/models")
    result = {
        "sglang_head": head("/sgl-workspace/sglang"),
        "miles_head": head("/root/miles"),
        "manifest_sglang": manifest["sglang"]["commit"],
        "manifest_miles": manifest["miles"]["commit"],
        "cast_fix": {name: "dense_activation_cast_dtype(" in (models / name).read_text()
                     for name in ("qwen2.py", "qwen3.py")},
        "miles_fp8_ignore": Path("/root/miles/miles/utils/fp8_ignore.py").is_file()
        and "is_ignored(key, ignored)" in Path("/root/miles/tools/convert_hf_to_fp8.py").read_text(),
    }
    helper = subprocess.run(
        ["/opt/sglang/bin/python", "-c",
         "import torch; from sglang.srt.true_on_policy.config import dense_activation_cast_dtype as f; "
         "print(f.__module__)"], capture_output=True, text=True)
    result["helper_import_rc"] = helper.returncode
    result["ok"] = (result["sglang_head"] == result["manifest_sglang"] == pins["sglang"]
                    and result["miles_head"] == result["manifest_miles"] == pins["miles"]
                    and all(result["cast_fix"].values()) and result["miles_fp8_ignore"]
                    and helper.returncode == 0)
    return result


def main() -> int:
    import modal

    sys.path.insert(0, str(REPO))
    from modal._vendor import cloudpickle

    from yeto.rl.adapters.miles.pins import MILES_NEXT_COMMIT, MILES_NEXT_IMAGE, SGLANG_NEXT_COMMIT

    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    app = modal.App("yeto-e6-image-check")
    image = modal.Image.from_registry(MILES_NEXT_IMAGE.removeprefix("docker:"))
    fn = app.function(image=image, cpu=2.0, memory=8192, timeout=900, serialized=True, name="check")(check)
    with app.run():
        result = fn.remote({"sglang": SGLANG_NEXT_COMMIT, "miles": MILES_NEXT_COMMIT})
    result["image"] = MILES_NEXT_IMAGE
    print(json.dumps(result, indent=1))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
