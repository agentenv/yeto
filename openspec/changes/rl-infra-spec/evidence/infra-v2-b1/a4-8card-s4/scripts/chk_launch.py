"""Chain-head self-check (case `chk`, a8go_strict.sh): bring up the SAME island cluster the training launcher would request
(launcher.make_miles_island_task -> identical sky Resources/image/setup, so the later items reuse the cluster), but run a
check script instead of the learner: fork pin + `workers_lost` present, Megatron `hc_head_contraction`, the LoRA module import.
No training, no attestation.  usage: chk_launch.py <args.txt> <cluster name> [check script file]   exit 0 = all checks passed.
DRY_CHK=1 prints the check script and exits (no sky, no cloud prerequisites); DRY_CHK=2 also builds the sky Task in memory (needs ~/.sky) and prints its resources."""
import json, os, shlex, sys, time

ARGS, CL = sys.argv[1], sys.argv[2]; SCRIPT = sys.argv[3] if len(sys.argv) > 3 else None
sys.argv = ["yeto"] + shlex.split(open(ARGS).read())
from yeto import cli, launcher  # noqa: E402
from yeto.gpu_spec import parse_gpu_spec  # noqa: E402
from yeto import rl as _rl  # noqa: E402

args = cli.build_parser().parse_args(sys.argv[1:])
spec = parse_gpu_spec(args.gpu)[0]
pin = _rl.MILES_NEXT_COMMIT
DRY = os.environ.get("DRY_CHK", "0")
if DRY != "1":
    launcher.prepare_launch_args(args)
CHK_PY = os.environ.get("CHK_PY", "import miles_plugins.models.qwen3_8_next.lora; from sglang.srt.models.qwen4_exp import "
                        "Qwen4ExpForConditionalGeneration as M; print(M.supported_lora_modules)")
MEG = os.environ.get("CHK_MEGATRON_FILE", "/root/Megatron-LM/megatron/core/transformer/transformer_block.py")
check = (SCRIPT and open(SCRIPT).read()) or f"""set -u
export PYTHONPATH=$HOME/miles:${{PYTHONPATH:-}}
fail=0
h=$(git -C ~/miles rev-parse HEAD 2>/dev/null); echo "[chk] fork HEAD=$h pin={pin}"; [ "$h" = {pin} ] || {{ echo "[chk] FAIL fork pin"; fail=1; }}
ls ~/miles/*SHA* ~/miles/.yeto* /root/*SHA* 2>/dev/null | sed 's/^/[chk] sha file: /'
f=$(grep -rl workers_lost ~/miles --include=*.py 2>/dev/null | head -1); [ -n "$f" ] && echo "[chk] workers_lost in $f" || {{ echo "[chk] FAIL workers_lost not in the fork"; fail=1; }}
grep -q hc_head_contraction {MEG} 2>/dev/null && echo "[chk] hc_head_contraction in {MEG}" || {{ echo "[chk] FAIL hc_head_contraction missing in {MEG}"; fail=1; }}
python -c {shlex.quote(CHK_PY)} && echo "[chk] lora module import ok" || {{ echo "[chk] FAIL lora module import"; fail=1; }}
nvidia-smi --query-gpu=index,name --format=csv,noheader | sed 's/^/[chk] gpu /'
[ $fail = 0 ] && echo CHK_OK || echo CHK_FAILED
exit $fail
"""
if DRY == "1":
    print(f"cluster={CL} spec={spec} image={getattr(args, 'rl_image', None)} pin={pin}\n--- run script ---\n{check}")
    sys.exit(0)

import sky  # noqa: E402

task = launcher.make_miles_island_task(args, spec, 0, 1, "none")   # same setup (fork checkout verified against the pin) and resources
task.run = check
if DRY == "2":
    print(f"cluster={CL} resources={task.resources} setup_lines={len((task.setup or '').splitlines())}\n--- run script ---\n{check}")
    sys.exit(0)
print(f"[chk] launching {CL} ({spec.accelerators} {spec.cloud}/{spec.region}) image={getattr(args, 'rl_image', None)}", flush=True)
rid = sky.launch(task, cluster_name=CL, retry_until_up=getattr(args, "retry_until_up", False))
job_id, _handle = sky.stream_and_get(rid)
print(f"[chk] job {job_id} submitted at {time.strftime('%FT%TZ', time.gmtime())}", flush=True)
try:
    sky.tail_logs(CL, job_id, follow=True)
except Exception as e:  # the status below is authoritative
    print(f"[chk] tail_logs: {e}", flush=True)
status = None
for _ in range(60):
    status = sky.get(sky.job_status(CL, [job_id])).get(job_id)
    if status is not None and status.is_terminal():
        break
    time.sleep(5)
print(f"[chk] job {job_id} status={status}", flush=True)
ok = status is not None and str(status).endswith("SUCCEEDED")
json.dump({"cluster": CL, "job_id": job_id, "status": str(status), "ok": ok, "pin": pin}, open(os.environ.get("CHK_OUT", "chk_result.json"), "w"))
sys.exit(0 if ok else 1)
