# ports image e3a11ab-a1240c5 (sglang yeto/lora-checksum, 2026-10-01)
- Only the sglang commit changed: 9f29303 -> a1240c530d40 (yeto/lora-checksum, WeightChecker checksum covers LoRA adapter A/B). Miles stays e3a11ab3. Base unchanged.
- Built daemonless (no docker permission on host): `SGLANG_COMMIT=a1240c530... CRANE=/tmp/img-tools/crane scripts/build_miles_ports_image.sh --push` (1m02s). Private ghcr.
- Tag ghcr.io/michaellchung/yeto-miles-ports:e3a11ab-a1240c5 @ sha256:12fcd9e583d63287d6814dfc87158364a0e370a22795462d19962a2857e53069 (the script's first push of the tag was sha256:8fc41532..., then re-pointed by the label step; crane digest after == build-record == pin). Old image e3a11ab-9f29303 untouched.
- digest-check.txt: registry == build-record == pin. container-identity.txt: image export (crane export of the pinned digest, 16 min) -> /root/miles HEAD e3a11ab3, /sgl-workspace/sglang HEAD a1240c530, /opt/yeto/image-manifest.json agrees, version 0.5.21.dev68+ga1240c5.
- CPU checks use the extracted trees on the host (miles-next-venv, torch 2.13 CPU), not a running container (no docker): import sglang+miles OK; sglang test_weight_checker_lora.py 3 passed (sglang-unit.log); Miles get_miles_extra_args_provider parse smoke on e3a11ab3 (miles-parse-smoke.log; same method as 2026-09-30-parse-5c1b49e). Not covered: full Megatron validate_args, GPU.
- Not changed: tools/probes/e2_cut_harness.py pre-registered plan-v6 (IMAGE_DIGEST 2cc5cc52 / IMAGE_TAG e3a11ab-9f29303): its check_pins will now report mismatch; INFRA/E2 owner decides.
- New test tests/test_rl_ports_image.py::test_commits_and_digest_are_pinned_together_with_the_build_record.
- Full pytest: full-pytest-summary.txt.
