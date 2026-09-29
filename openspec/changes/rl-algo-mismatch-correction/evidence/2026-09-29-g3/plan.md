# G3 plan, TIS two islands strict-avg (task 7.5; final, committed before launch, 2026-09-29)

- Mechanism: TIS only (declared on the adapter: corrections tis; no --rl-allow-unverified-mechanism).
  IcePop is not declared -> not in this round. Spec tis.json (tis_clip 2.0, tis_clip_low 0.0),
  identical to G1.
- Code: algo-1a at YETO_SHA; upload tree = `git archive YETO_SHA` without openspec/tests/docs,
  plus gsm8k_reward.py. Image MILES_NEXT_IMAGE (registry login exported in-process, not printed).
- Topology (R0 7.1 way, cf. rl-engine-ports evidence 2026-09-29-default-params-modal): two Modal
  islands `--gpu modal:1xh100,modal:1xh100 --modal-gpu-exact`, strict-avg, syncer = LocalSyncer on
  this machine via harness/run_local_head.py (cli.cmd_head, head controller on this host,
  SYNCER_PUBLIC_IP=185.189.44.160). Syncer binary: R0's yeto-syncer (sha256 in syncer_sha256.txt),
  HOME isolated to /tmp/algo1a/g3home (symlinks to ~/.modal.toml and ~/.sky only).
- Port: the launcher hardcodes SYNCER_PORT=29400, which INFRA's syncer uses. The head script sets
  `launcher.SYNCER_PORT = 29410` before cmd_head (29410 checked free with `ss -ltn` right before
  launch; recorded in port_check.txt). Launcher code is not changed. If the islands cannot reach
  :29410 the run fails as an environment problem and is reported; no retry on another port without a
  new committed note.
- Command args: as R0 default-params-modal (Qwen3-0.6B c1899de, zhuzilin/gsm8k 0cbd9f3, LoRA r16
  all-linear, --total-steps 3 --fragments 1 --pipeline 1, 4x8 samples, response 384, seq 1024,
  lr 1e-5, seed 17) plus `--rl-engine ports --rl-algorithm-spec tis.json --rl-sync-preset strict-avg
  --cluster-prefix algo1a-g3`.
- Pass criteria (all; exact equality, no tolerance, as spec "两岛 strict-avg"):
  1. head exit 0; syncer tape has 3 outer steps with sync/responders == 2 and sync/rejected_stale_updates == 0;
  2. both island tapes: rl_engine_selected present, same rl/algorithm_spec_sha256 == sha256(tis.json),
     no rl/unverified_mechanisms;
  3. for each policy_version 0..3 both islands' rl_publication sync/publication_payload_hash are equal
     (post-outer-sync weights identical);
  4. each island: 3 rl_local_round events with finite grad_norm, no invariant/failure events
     (rl_invariant_failed, rl_round_failed, rl_algorithm_mismatch, rl_algorithm_island_rejected);
  5. Miles log carries train/tis, train/tis_abs, train/tis_clipfrac for every step of both islands.
- Reclaim: `timeout 3000` around the head; trap stops Modal app yeto-algo1a-g3 and kills the local
  syncer; independent setsid watchdog stops the app after 3300 s; after the run `modal app list`
  must show it stopped with 0 tasks; `ss -ltn` shows 29410 closed.
- Cost: 2x H100 ~20 min = ~0.7 H100 h, ~ $3; cap $8. Single attempt; a harness failure before the
  app exists may be fixed and rerun once after a committed diagnosis.

## Attempt 1 (harness, before any cloud resource)
- `RL reward source must be inside the synced Yeto workdir`: gsm8k_reward resolved from harness/
  (the head script's directory is sys.path[0]). Fix: copy run_local_head.py into the upload tree
  and run it from there. Rerun once.
