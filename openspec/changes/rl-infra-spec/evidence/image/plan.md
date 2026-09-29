# IMG smoke plan (written before launch, 2026-09-29)

- Task: rl-infra-spec 1.1 prerequisite -- private ports image (MILES_NEXT_IMAGE).
- Image under test: ghcr.io/michaellchung/yeto-miles-ports@sha256:5da40a07dabb3ea3fcf921efb4b2a21ca1178220bde40dc178c79b734fdaa540 (tag 0394715-9f29303).
- Cloud: Modal, app `img-smoke`, one Sandbox, GPU `L40S` (1x), cpu 8, mem 64 GiB.
  Sandbox `timeout=2400` s (Modal-side hard stop) + local `timeout 3000` wrapper + trap that terminates
  the sandbox and `modal app stop -y img-smoke`.  Expected ~15-25 min incl. image pull; cost estimate
  < $1.5 (L40S ~$1.95/h + CPU/mem); budget $5.
- Registry login: Modal `from_registry(secret=Secret.from_dict(REGISTRY_USERNAME/PASSWORD))` built in
  memory from the ghcr entry of ~/.docker/config.json (read-only, never printed, no named Modal secret).
- Pass conditions (all must hold; any failure = smoke fails, no rerun without a diagnosed fix):
  1. private pull succeeds via the secret; GPU name contains "L40S";
  2. /opt/yeto/image-manifest.json miles/sglang commits == yeto/rl pins; base digests == build record;
  3. `import miles`, `import sglang` resolve under /root/miles and /sgl-workspace/sglang/python;
     /root/miles and /sgl-workspace/sglang git HEAD == pins, origin == fork repos, trees clean;
     fork-only code present (miles TrainerController.run_plugin; sglang TorchMemorySaverAdapter.region
     has enable_disk_backup); importlib.metadata sglang version == 0.5.21.dev67+g9f29303 == sglang.__version__;
  4. the launcher's own ports setup (`_miles_source_setup("ports")`) runs with exit 0 in the image,
     takes the image-sglang branch and does not fetch Miles;
  5. lr-fix's upstream parse_args tests (tests/test_rl_miles_adapter_config.py, tests/test_rl_argv_snapshot.py; test_rl_applied_lr.py is not on this branch)
     pass with Miles from /root/miles (PYTHONPATH=/root/miles).

## Attempt 1 (sb-IZC1WVhHlNN5pT145otSVK, 528 s, evidence attempt1/) -- smoke-script bugs, not image faults
- imports_point_at_forks FAIL: the script imported `RayTrainGroup`; the fork's class is
  `miles.ray.train.group.TrainerController` (run_plugin at line 467).  Fixed the name.
- parse_args_tests FAIL: the previous check ran `cd /tmp` in the script's own shell (eval), so
  pytest could not find tests/.  Fixed by running that check in a subshell.
All other checks passed (private pull, L40S, manifest==pins, git identity, launcher setup reuse).
Attempt 2 reruns the same plan unchanged otherwise.

## Attempt 2 (sb-39ljVyIC1pSw4NlV18Qs5a, L40S, 190 s, evidence attempt2/)
**Note: in attempt 2 the import check FAILED; afterwards the check method was corrected
(PYTHONPATH=/root/miles) and only that one check was rerun, on a CPU-only sandbox -- not a
full GPU rerun.**  Scripts actually used: attempt1/*.used (commit 9638262),
attempt2/*.used (commit 3efd272); the CPU rerun used attempt2/imports_cpu.py with the
corrected smoke_in_image.sh at 47c110e.
PASS: gpu_is_l40s, manifest_matches_pins, git identity (/root/miles, /sgl-workspace/sglang),
launcher_ports_setup (exit 0), setup_used_image_sglang, setup_skipped_miles_fetch,
after_setup_still_fork, parse_args_tests (31 passed, 1 skipped).
FAIL imports_point_at_forks -- check-script assumption: without PYTHONPATH, `/root` is on the base
image's sys.path so `import miles` becomes a namespace package over the /root/miles repo dir
(`miles.__file__` None).  ns_probe.log shows the *public base image* behaves identically; yeto
always runs with PYTHONPATH=$HOME/miles (regular package; after_setup_still_fork PASS).
The check was changed to use PYTHONPATH=/root/miles (runtime condition) and rerun alone on a
CPU-only sandbox (sb-88pkNZTN7G3r9UF63pup52, imports_cpu.log): PASS (miles /root/miles/miles,
sglang /sgl-workspace/sglang/python, version 0.5.21.dev67+g9f29303 x1, run_plugin and
enable_disk_backup present).  No GPU rerun: remaining checks are GPU-independent.

## Cost / cleanup
L40S sandboxes 528 s + 190 s; CPU sandboxes (2 probes, 2 cpu/8 GiB) < 15 min total.
Estimate (Modal list prices): ~ $0.40 GPU + ~ $0.20 CPU/mem = ~ $0.6 (not billing-confirmed).
All sandboxes terminated in `finally`; `modal app stop -y img-smoke` after each run; app list
shows every img-smoke app `stopped`, 0 tasks (modal_app_list_after_stop.txt); watchdog killed.
No volumes / secrets created (pull secret was an in-memory Secret.from_dict).

## Post-review notes (no rebuild)
- The pushed image's /opt/yeto/image-manifest.json still carries the old miles "install"
  wording; the build script now writes the accurate one (namespace package without
  PYTHONPATH; yeto runs with PYTHONPATH=$HOME/miles).  Takes effect on the next build.
- Rebuilds are file-content equivalent, not byte-identical (.git/index, pack).
