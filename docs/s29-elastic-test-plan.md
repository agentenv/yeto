# S29 elastic RL / pure-spot rollout island health test design

Goal: on a real cross-cloud topology, run a thorough health test of yeto's "pure-spot rollout islands + synchronous trainer group" and expose as many defects as possible.

## 1. Topology under test

| Role | Location | Notes |
|---|---|---|
| syncer | npu1 CPU, `/data/work/s29-spot/yeto-syncer` (aarch64 musl) | `--learners 2 --max-subscribers 2 --sample-relay`, full-parameter async, 8 fragments |
| trainer island 0 / 1 | n4 card 0 / card 1 (avoiding the flapping-link cards 4 and 5), verl FSDP2 full-parameter relay trainer | Qwen3-0.6B, 1 group × 4 samples per round, `--rl-max-policy-age 4`, TIS |
| spot island a (channel 0) | Nebius eu-north1, 1×H100 preemptible, `s29-spot-a` | SGLang TP1 + records proxy + Harbor/Codex/boat |
| spot island b (channel 1) | Nebius us-central1, 1×RTX6000 preemptible, `s29-spot-b` | Same as above |
| Link | npu1 -> VM reverse SSH tunnel `-R 127.0.0.1:29400` | VMs see the syncer only via the tunnel; boat sandboxes reach the proxy via the VM public IP:31600 |

Data: TB2.1 89 tasks (`agentic-tb21-89.jsonl`), island i takes `data_slot i/2`. VM reuse enabled for boat sandboxes.

Control scripts: npu1 `/data/work/s29-spot/ctl.sh`, n4 `/data/work/s29-spot/n4ctl.sh`.

## 2. Evidence sources

- syncer tape: `run/yeto-tape.jsonl`, syncer log `run/syncer.log`
- trainer tape: n4 `run/i{0,1}/tape.jsonl`, logs `run/island{0,1}.log`
- spot island tape: VM `/root/spot/events.<incarnation>.jsonl`, `sampler.log`, `vm_run.log`
- boat: `boat limits` (activeSandboxes, starts.day), `boat` list
- Nebius: `nebius compute instance list` (only the `s29-spot-` prefix)

## 3. Test items and pass criteria

### T1 Baseline path (5 steps)
- Both spot islands emit `rl_spot_sampler_channels` and `rl_sampler_load_checked`, and produce groups (`rl_sampler_engine_round` groups ≥ 1).
- syncer relay stats `stored` / `acked` are > 0 on both channels.
- Both trainer islands each complete 5 `rl_round_trained`, `trained_groups` ≥ 1.
- Criterion: all satisfied; no abnormal trainer exit (ISLAND-RC=0 or ended by design).

### T2 Tape integrity and consistency
- Every tape line is valid JSON, `time_unix` non-decreasing (within one process).
- Each spot island incarnation: exactly one `rl_sampler_join`, ending with `rl_sampler_leave` (except those killed by kill -9, recorded as missing).
- Set of sample ids trained by the trainer ⊆ set of sample ids written by spot islands; no sample_id trained twice.
- For each trained group, the difference between its behavior version and the trainer's version at the time ≤ `--rl-max-policy-age`.
- Criterion: all satisfied; any violation is a defect.

### T3 Dashboard / summary data correctness
- `yeto status` / event summaries (reward_mean, groups, tokens, step_seconds) match recomputation from raw tape lines (zero error).
- Three-way reconciliation of relay stats (syncer log), group counts in spot island tapes, and group counts in trainer tapes.
- Criterion: all three agree, or the difference is fully explained by in-flight/dropped items.

### T4 syncer sync / version advancement / staleness
- After each merge both islands' `policy_version` advances; the spot island `PolicyFollower` loads the new version (`rl_sampler_load_checked` version increasing).
- The post-load probe on the spot island passes (`max_probe_jump` within 0.5).
- Groups over the staleness limit are skipped and recorded (`rl_buffer_stale_skipped`) and never enter training.
- Criterion: versions advance monotonically; spot island loaded version ≤ syncer latest version and lag ≤ 2; over-age groups never enter training.

### T5 Relay sample flow
- Each group has a complete chain from spot island write to trainer consumption: producer seq -> syncer stored -> consumer acked -> trainer buffer.
- Channel isolation: channel 0 groups only go to island 0, channel 1 only to island 1.
- Criterion: no loss (except recorded un-acked items of a fenced producer), no cross-channel leakage, no duplicates.

### T6 Preemption and fault injection (record recovery time and post-recovery state for each)
| ID | Injection | Expected |
|---|---|---|
| T6.1 | kill -9 the spot island sampler process (VM untouched), then restart | New incarnation supersedes the old one; un-acked relay items handled as designed; trainer does not crash and keeps training |
| T6.2 | SIGTERM (simulated cloud preemption notice) | Writes the reclaim marker and leaves as PREEMPTED (`rl_sampler_leave` reason PREEMPTED) |
| T6.3 | sky down one VM then reopen | Trainer waits (no timeout exit); new VM subscribes and continues producing samples from the current version |
| T6.4 | Cut the tunnel for 60 s then restore | subscriber/relay reconnect; no exit within `syncer_lost_s`; no duplicate samples |
| T6.5 | syncer restart (with `--resume`) | Trainers and spot islands reconnect; version and cursor do not roll back |
| T6.6 | kill -9 a trainer island then restart | Island restores cursor/version from checkpoint; backlog groups on the relay channel are consumed correctly |
| T6.7 | Both spot islands disconnect at once | Trainer waits; behavior when `--rl-sample-wait-s` expires matches the design |

Criterion: all expectations met for each item; after recovery versions do not roll back, samples are not duplicated, cursors are correct.

### T7 Version and cursor correctness after recovery
- After each injection check: trainer `train_step` is contiguous, `policy_version` does not roll back, spot island `data_cursor` does not take the same prompt row twice (except design-allowed recycling).

### T8 Multi-island quorum / grace
- quorum=1: when one trainer island stalls the other still advances; the stalled island catches up when it returns (within `max-base-lag`).
- Criterion: responders and grace records in the syncer tape match the config; no deadlock.

### T9 Long-run resource leaks (30+ steps)
- trainer: NPU HBM and host RSS do not grow monotonically with steps (first-to-last difference < 10%).
- spot island: GPU memory, process RSS, `/tmp/harbor-trials` directory, weights directory (old versions should be cleaned) do not grow unboundedly.
- syncer: RSS and relay depth do not grow unboundedly.
- Criterion: first-to-last difference < 10% or bounded.

### T10 boat sandbox leaks
- While running, activeSandboxes ≤ sum of each island's `max_in_flight` (8).
- After each injection (kill/down) the active count drops within ≤ 20 minutes (idle VM TTL); 0 after everything ends.
- Criterion: after the end, `boat limits` activeSandboxes = 0.

### T11 Error injection
- Corrupted relay items / connections with a wrong HMAC are rejected (syncer records refused).
- A trainer with a wrong layout is rejected (layout hash check already observed in the baseline).
- Spot island load failures (e.g. probe over threshold) go through `max_consecutive_load_failures` handling.
- Criterion: errors are explicitly rejected and recorded, never silent.

## 4. boat quota strategy

About 22 creations left today (resets to 200 at 00:00 UTC). Real TB2.1 trials are prioritized for T1/T6; when the quota runs out, path-level tests (T4/T5/T6/T9) continue on a fixed-reward path that does not need new sandboxes, and real TB2.1 is rerun after the reset.

## 5. Cost guardrails

Total VM spend ≤ $60, ≤ 3 concurrent VMs, shut down while idle; manage only resources with the `s29-spot-` prefix.

## 6. Results (10-31 03:45)

| Item | Result | Evidence / defects |
|---|---|---|
| T1 baseline TB2.1 5 steps | Passed (each island trained 5 local steps, 4.4–9.6 s per step, reward all 0) | D1 D3 D5 D6 fixed; docs/s29-evidence/t1-tb21 |
| T2 tape consistency | Issues found and fixed: duplicate training (D6 across runs, D12 on restart) | D6 D12 |
| T3 three-way reconciliation | relay stored/acked, local buffer and trainer trained counts agree (after fixes) | final/*.tgz |
| T4 version advancement/staleness | Merges advance normally; spot islands cannot keep up with versions (WAN 0.18 MB/s) | D9 D10 D13 |
| T5 relay flow | No loss, no cross-channel leakage; groups acked before a crash were not lost (D2 closed) | D2 |
| T6.1 kill -9 sampler | Passed after fix (orphaned SGLang process group reaped) | D8 |
| T6.2 SIGTERM | Passed (left as PREEMPTED, boat active count back to 0) | — |
| T6.3 sky down VM | Down done; reopen deferred to the TB2.1 regression after 08:00 (boat quota) | — |
| T6.4 tunnel cut 60 s | Passed (recovered automatically after 7 redials, no exit) | drill.log |
| T6.5 syncer --resume restart | Passed (restored to step 42, all members reconnected, no fence) | drill.log |
| T6.6 trainer island kill -9 | Failed under the default config (D11); passed with island pool enabled and D12 fixed | D11 D12 |
| T6.7 both spot islands offline | Trainer waits without exiting (after the D10 fix); not timed separately | D10 |
| T8 quorum/grace | With quorum=1 a single island can advance merges (only 1 responder in the syncer tape) | syncer.log |
| T9 long-run leaks | About 150 rounds in total; trainer shared memory +35% to be confirmed | D14 |
| T10 boat leaks | Passed (activeSandboxes=0 after everything ended) | — |
| T11 error injection | Layout hash mismatch and policy age above the verl cap are both explicitly rejected at startup | baseline logs |
