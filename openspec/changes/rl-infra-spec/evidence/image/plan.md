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
