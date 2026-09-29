# 7.6 G3 plan: MaxRL two islands strict-avg (committed before launch, 2026-09-29, Agent ALGO-2a)

- Task 7.6: "MaxRL 两岛 strict-avg（每岛 1 卡），用正式声明（不带放行参数），2–3 轮。验证：两岛算法哈希一致、
  每轮外层应用后状态 hash 一致、不变量无误报；证据存入 evidence/。"
- Code: YETO_SHA = 4652f73 (integ-decl, all declarations merged; checked out in a private worktree
  /tmp/a2-g3, not the integ-decl worktree). Example specs verified current at that SHA
  (`test_example_specs_are_current` passes; make_examples.py produces no diff). Spec
  maxrl.json (copy in this directory, from the 4652f73 tree). Upload = `git archive YETO_SHA`
  without tests/docs. Image MILES_NEXT_IMAGE (registry login exported in-process, not printed).
- Mechanism: formally declared (dry-run at 4652f73: 2 islands, outer_sync true,
  unverified_mechanisms []); no --rl-allow-unverified-mechanism.
- Topology as R0 7.1 / algo-1a G3: `--gpu modal:1xh100,modal:1xh100 --modal-gpu-exact`, strict-avg,
  LocalSyncer on this machine through harness/run_local_head.py (cmd_head, head controller
  local, SYNCER_PUBLIC_IP=185.189.44.160). Syncer binary = R0 yeto-syncer (sha256 in
  syncer_sha256.txt; syncer sources unchanged between algo-1a's G3 SHA 8a491f5 and 4652f73).
- Port: 29400 and 29410 are in use (ss -ltn); the head script sets launcher.SYNCER_PORT = 29420,
  checked free right before launch (port_check.txt). If the islands cannot reach :29420 the run
  fails as an environment problem and is reported; no retry on another port without a new
  committed note.
- Args: Qwen3-0.6B c1899de, zhuzilin/gsm8k 0cbd9f3, reward yeto.rl.algos.gdpo_reward:correctness_reward
  (binary), LoRA r16 all-linear, --total-steps 3 --fragments 1 --pipeline 1, 4x8, response 384,
  seq 1024, lr 1e-5, seed 17, cluster prefix algo2a-g3.
- Evidence: island tapes from the P0 4373cd9 event echo (`<run dir>/events/*.jsonl`), syncer tape
  yeto-tape.jsonl, launch.log. Criteria evaluated by harness/check_g3.py (written before launch).
- Exit code reading (declared now): 0 = normal completion; 2 = launcher notice that island
  artifacts could not be fetched -> the tapes decide (criteria 1-4 must still hold);
  3 = a tape lacks rl_learner_finalized -> FAIL; any other non-zero -> FAIL.
- Pass criteria (all; exact equality, no tolerance):
  1. syncer tape: 3 outer steps, each sync/responders == 2 and sync/rejected_stale_updates == 0;
  2. both island tapes: rl_engine_selected with the same rl/algorithm_spec_sha256 == sha256 of
     maxrl.json at YETO_SHA, and no unverified mechanisms;
  3. for policy_version 0..3 both islands' rl_publication sync/publication_payload_hash are
     equal (state after each outer apply identical);
  4. each island: 3 trained rounds with finite grad_norm, rl_learner_finalized present, and no
     rl_invariant_failed / rl_round_failed / rl_algorithm_mismatch / rl_algorithm_island_rejected.
  Observations only: per-island rl_advantage_transform counts, nonzero_advantages per round,
  grad norms.
- Reclaim: `timeout 3000` around the head; trap stops Modal app yeto-algo2a-g3 and kills the
  local syncer; independent setsid watchdog (3300 s) stops the app and syncer; afterwards
  app list must show stopped / 0 tasks and port 29420 closed.
- Cost: 2x H100 ~20 min ~ 0.7 H100 h ~ $3.5; change cap $20 (spent ~$8). Single attempt; a harness
  failure before any cloud resource may be fixed and rerun once after a committed diagnosis.
