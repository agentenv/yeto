#!/bin/bash
# Runs inside the ports image (see plan.md).  Prints PASS/FAIL per check.
set -u
cd /work/yeto
fails=0
check() { if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fails=$((fails+1)); fi; }
echo "== gpu"; nvidia-smi -L
check gpu_is_l40s 'nvidia-smi --query-gpu=name --format=csv,noheader | grep -q L40S'
echo "== manifest"; cat /opt/yeto/image-manifest.json
check manifest_matches_pins 'python3 - <<PY
import json, sys
sys.path.insert(0, "/work/yeto")
import yeto.rl as rl
m = json.load(open(rl.MILES_NEXT_IMAGE_MANIFEST))
assert m["miles"]["commit"] == rl.MILES_NEXT_COMMIT and m["miles"]["repository"] == rl.MILES_NEXT_REPOSITORY
assert m["sglang"]["commit"] == rl.SGLANG_NEXT_COMMIT and m["sglang"]["repository"] == rl.SGLANG_NEXT_REPOSITORY
assert rl.MILES_NEXT_BASE_IMAGE.endswith(m["base"]["index_digest"])
assert m["base"]["manifest_digest"] == "sha256:4d69750720eb1a4b99976a3fed45ed018e1fc4fff3d240667fa17d258dfc6e0c"
PY'
echo "== imports / metadata"
check imports_point_at_forks 'python3 - <<PY
import importlib.metadata as md, inspect, os
import miles, sglang
from miles.ray.train.group import RayTrainGroup
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
print("miles", miles.__file__, md.version("miles"))
print("sglang", sglang.__file__, md.version("sglang"), sglang.__version__)
assert os.path.realpath(miles.__file__).startswith("/root/miles/miles/")
assert os.path.realpath(sglang.__file__).startswith("/sgl-workspace/sglang/python/sglang/")
assert md.version("sglang") == sglang.__version__ == "0.5.21.dev67+g9f29303"
assert hasattr(RayTrainGroup, "run_plugin")
assert "enable_disk_backup" in inspect.signature(TorchMemorySaverAdapter.region).parameters
print("sglang dists:", sorted(d.metadata["Version"] for d in md.distributions() if d.metadata["Name"] == "sglang"))
assert len([d for d in md.distributions() if d.metadata["Name"] == "sglang"]) == 1
PY'
for r in /root/miles:0394715083c91182b5eb0c526eeee4196ac694b9:https://github.com/michaellchung/miles /sgl-workspace/sglang:9f29303bef1eea38eb613e5f454a52db1326422d:https://github.com/michaellchung/sglang; do
  d=${r%%:*}; rest=${r#*:}; c=${rest%%:*}; o=${rest#*:}
  echo "$d HEAD=$(git -C $d rev-parse HEAD) origin=$(git -C $d config --get remote.origin.url)"
  git -C $d status --porcelain --untracked-files=all | head -5
  check "git_$d" "[ \"\$(git -C $d rev-parse HEAD)\" = $c ] && [ \"\$(git -C $d config --get remote.origin.url)\" = $o ] && [ -z \"\$(git -C $d status --porcelain --untracked-files=all)\" ]"
done
echo "== launcher ports setup in image"
python3 -c 'import sys; sys.path.insert(0, "/work/yeto"); from yeto.launcher import _miles_source_setup as f; m, s = f("ports"); open("/tmp/setup.sh", "w").write(m + "\n" + s + "\n")'
check launcher_ports_setup 'HOME=/root GIT_TRACE=0 bash -x /tmp/setup.sh > /tmp/setup.log 2>&1'
tail -20 /tmp/setup.log
check setup_used_image_sglang 'grep -q "image provides sglang 9f29303bef1eea38eb613e5f454a52db1326422d" /tmp/setup.log && [ "$(readlink /root/sglang)" = /sgl-workspace/sglang ]'
check setup_skipped_miles_fetch '! grep -q "git -C /root/miles fetch" /tmp/setup.log'
check after_setup_still_fork 'cd /tmp && PYTHONPATH=/root/miles:/root/sglang/python python3 -c "import miles, sglang, os; assert os.path.realpath(miles.__file__).startswith(\"/root/miles/\"); assert os.path.realpath(sglang.__file__).startswith(\"/sgl-workspace/sglang/python/\"); print(miles.__file__, sglang.__file__)"'
echo "== upstream parse_args tests (lr-fix set)"
(python3 -c 'import pytest' 2>/dev/null || pip install -q pytest)
check parse_args_tests 'PYTHONPATH=/root/miles:/work/yeto python3 -c "import miles.utils.arguments, megatron.training; print(\"imports ok\")" && PYTHONPATH=/root/miles:/work/yeto python3 -m pytest -q -rs -p no:cacheprovider tests/test_rl_miles_adapter_config.py tests/test_rl_argv_snapshot.py 2>&1 | tail -15 | tee /tmp/pytest.txt; grep -qE "passed" /tmp/pytest.txt && ! grep -qE "[0-9]+ (failed|error)" /tmp/pytest.txt'
echo "fails=$fails"
exit $fails
