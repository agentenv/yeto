# SFT and CoT handoff — September 9, 2026 UTC

This handoff records verified launches and remaining work. Read the latest status and `PROJECT_CONTEXT.md` before taking action; older project-context bullets are historical snapshots.

## User decisions

- Run full no-CoT Qwen3.8-27B SFT on n7's eight H200s: LR 1e-5, full text-parameter updates, full 262,144-token context, discard excess tokens. Keep the prepared native rendering and loss masks unchanged.
- User explicitly canceled the separate two-update GPU qualification. Apply both memory repairs, run the code checks, and start production directly. Sustained memory stability must be established by actual production updates.
- Save complete checkpoints after completed updates 1, 5, 10, … and at completion; keep the latest three. The failed run's first update had no checkpoint and cannot be resumed.
- Use the other six nodes for CoT. Preserve full original prefix plus next-five generation. Latest direction: **stop reviewer work, generate first, review later**. The reviewer is stopped and generation-only is running. Generated candidates remain `review_pending`; no generated CoT enters the no-CoT baseline.
- The later review's key criterion is no future-information leakage. Factual claims must be grounded before the gap; reasonable prospective plans are allowed. Do not reject harmless stylistic differences. Since generation sees the next five events, unreviewed candidates do not establish absence of leakage. Failed/uncertain reviews never authorize acceptance.
- Labeling stays paused. Do not restart it automatically. Use Walden's credentials where available. Do not install SSH keys; preserve the existing encrypted Mac tunnel approach.

## SFT

The prior production container `qwen38-no-cot-baseline-20260909t000455z` completed one update, then OOMed in the next forward when optimizer state was resident. It has stopped. Its data, model and Triton cache remain on n7.

Both repairs are implemented: contiguous context-parallel gather/local reordering and whole-block activation checkpointing. CPU value/gradient equivalence tests pass in the exact NeMo image. The new checkpoint helper also passes an actual CPU shuffled-loader/Adam resume comparison with identical sample suffix and final weights. This does not establish a new GPU checkpoint round-trip or sustained 262k memory result.

Dataset: `/data/sft_baseline_20260908/datasets/full-mix/manifest.audited.json`, SHA256 `12208706d0228d155fdbb2fe87fd844d0ede85e87348134b351a2e9714efe7a3`. There are 16,538 training examples and 145 validation examples; training has 2,053,079,654 input tokens. Preserve the validated byte-offset index and sampler seed.

Runtime: pinned `nvcr.io/nvidia/nemo-automodel:26.08.00` image, FSDP2/CP8/TP1/DP1, BF16 compute and FP32 optimizer state. Model revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

**Corrected full run is active and has completed two updates:** container `0e822f284b4727eca64f6e9278cbb33adf3fc2d5ca5164928db6db55b8e26a85`, named `qwen38-no-cot-baseline-20260909t012008z`, run `/data/sft_baseline_20260908/runs/baseline-20260909T012008Z`. Code stage `/data/sft_baseline_20260908/code/memory-repair-01c56ecfe6cc88a0`; recipe SHA256 `b6ff0861ffb30319714066e0b42eed6aa1eb898a370fa87d956cfc8f83aa447b`. It entered the real training loop around01:22:52UTC. Startup confirmed64whole-block wrappers with no nesting and8CUDA forward/backward parity cases across8NCCL ranks.

- Update0 completed01:25:32: loss0.750721872, gradient norm3.216692701,97,495 supervised tokens; peak allocated51.1318GiB / reserved65.7988GiB across ranks.
- Update1 completed01:31:41 with optimizer state resident: loss0.383037001, gradient norm1.748026183,137,339 supervised tokens; peak allocated85.2737GiB / reserved95.1641GiB. No OOM reported. Exact full262k sustained behavior still needs observation during later production updates; no separate GPU qualification was run.
- First full checkpoint is published: `checkpoints/LATEST` points to `epoch_0_step_0`. Eight model shards total54,730,161,681bytes; eight optimizer shards269,005,601,273bytes. Read-only checks confirm next_step1/epoch0,8consumed microbatches per rank, all8RNG states and matching runtime/dataset resume_guard. No actual GPU reload was performed after these repairs.

The immediately preceding `baseline-20260909T011247Z` container exited1 before training because its guard iterated ModuleDict keys instead of values. That verification bug was fixed, with3targeted runtime CPU tests passed; it was not a new measured OOM. The earlier `baseline-20260909T000455Z` is the run that OOMed after one update.

W&B: https://wandb.ai/yeta/yeto-h200/runs/e6113681, published with the verified local `walden-lee` login through `sft_baseline/mirror_wandb.py`. Local mirror PID69046; state `sft_baseline/wandb-mirror-baseline-20260909T012008Z-state.json`. Both completed updates and exact memory peaks were verified in remote W&B history. The helper handles the trainer's progress-bar prefix and structured numeric update lines, binds the exact source container/metric directory, and prevents duplicate history steps. It depends on this Mac remaining connected. Native trainer W&B is offline with no manager credential mount. Old mirror `b55d4362` is finished, source exit137.

## CoT

Frozen source on manager host: `/mnt/lvm_data/sft_analysis/sft_baseline_20260908/cot-source-v4-projection-v2.jsonl`, SHA256 `ed833db681fd8fd1e3c2315d96503016e8753bf7b4036ed6fbda4f88981c8bed`. It contains 14,146 usable traces and 643,513 original reasoning gaps from the authorized 20k selection. The original source and prior journals are preserved.

Generation uses the v5 prompt. The new v4 reviewer adds literal referent, presupposition and expected-versus-observed checks, exact prefix evidence IDs and a strict final JSON schema. Accuracy-first settings use 32,768 thinking tokens plus final-answer headroom (49,152 total). Never truncate the prefix to force a request to fit; overflows remain excluded.

Why the reviewer needs a final check: the latest 128-gap pilot yielded 77 provisional approvals, 20 rejects, 3 uncertain, 27 exhausted-output reviews and 1 context overflow. Agent inspection identified unsupported claims in some approved samples. Its 77 approvals are not a production acceptance guarantee. Known-case controls include older ordinals 22/37 and newer 13/64. All 255 actual pilot HTTP prompt counts matched local rendering.

The minimal generation-only mode in `cot_filler/corpus_worker.py` stores candidates as `review_pending`, never calls the reviewer or approves them, and can resume into ordinary review without regenerating those candidates. Twenty worker tests pass. Bulk and bounded-review coordinator threads must share one actual manager inference-lock owner while keeping separate SQLite connections/journals.

n5's same existing `dsv4-eng` model child was restarted with `--enable-strict-thinking`; no container/cache removal or access change occurred. It became healthy at about 00:57:56 UTC. New model PID 4129424; original private rollback snapshot is on n5 only. The existing Mac local33120 tunnel is used through a fresh manager reverse port33122. Other nodes need verified strict-thinking enforcement before routing budgeted reviewer requests to them; plain generation can use the existing healthy fleet.

Joint coordinator launched at 01:09:25 UTC: PID 1483495 / startticks 37483037 on manager host. Run directory `/mnt/lvm_data/sft_analysis/sft_baseline_20260908/cot-joint-accuracy-v4-bulk`, with `launch.json` and `lock-handoff.json`. Immutable code stage is `cot-code-v5-accuracy-bulk`, manifest SHA256 `da05d85a7873f2f15fda36f9f33444732bc1d344be07a150552c56a31046d7d4`.

The coordinator holds the real shared inference lock. It started generation-only for all 643,513 gaps with 48 workers and a 6,291,456-token admission budget across the six-node router. At01:15:38UTC,201 actual generation responses/candidates were persisted, all201 `review_pending`, zero bulk reviews/approvals;48 were generating,117 context overflows were excluded and643147 remained queued. All six inference backends were healthy and loaded; n7 was unavailable/load0. A later audit reached235 generation receipts, all matching local prompt counts.

**Generation-only transition completed.** Active manager supervisor PID1488373/startticks37556563, run `cot-generation-only-20260909/`, resumed01:22:13UTC using the same journal and pinned code. At01:24:15 there were516 durable candidates (56 new after the transition),38 generating,292 context overflows and642667 queued; zero bulk review records. All six nodes remained active. The old joint process stopped new work and committed all460 then-active generations before termination; no committed candidate was lost or regenerated.

The paused strict32k review experiment preserved19 actual responses (9 provisional passes,9 rejections,1 uncertain),2 context overflows and2 possibly unfinished HTTP requests; no outcome was invented. The later regression subset and the older8k proposal issued no requests. **The reviewer is stopped; do not restart it automatically.** See `cot_filler/BULK_HANDOFF_20260909.md` for exact status and continuation commands.

Do not give an hours-long full-completion promise from the fast generation stage. The prior six-node 128-gap generation/review pilot took 709.5 seconds; a naive extrapolation to 643,513 gaps is about 41 days. That pilot deliberately sampled longer contexts and used a different reviewer, so this is a warning about review cost, not a reliable corpus ETA. Measure separate generation and accepted-review rates on the actual bulk distribution.

## Reviewer-requested regeneration — implemented, inactive

The local code now connects an actual grounding rejection to a separate bounded retry ledger: `cot_filler/regeneration.py` and `cot_filler/regeneration_worker.py`. It only retries explicit unsupported-prefix/future-observation rejections, at most twice per original gap. Replacement prompts contain the original prefix and generic machine feedback codes, with no next-five events, target action, rejected text or freeform reviewer notes. Every replacement needs a fresh review before becoming a separate accepted selection.

Original candidates/reviews remain read only. Requests, responses, decisions and failures are append only. Completed generation resumes at review; unknown interrupted request outcomes are excluded rather than silently repeated. Actual returned model, token counts, config, code and source identities are checked. Finite cohorts support interruption-safe pagination. A future fleet activation requires real backend readiness checks; no deployment or model request was made for this feature.

Thirty-one focused tests and independent code review passed. The final broader check passed138unique tests;137 passed in the sandbox and the loopback review-server test passed separately with local networking permitted. See [REGENERATION.md](cot_filler/REGENERATION.md) for the CLI/API, recovery rules and later activation. **Current live generation continues with the reviewer stopped.** The branch update includes CoT code/tests, mirror utilities and these handoffs; SFT runtime code remains available in the pinned n7 stage and local workspace.

## Access and continuation

Use existing SSH with strict known hosts at `/private/tmp/yeta-labeling-known-hosts`; never print credentials. Manager is `c@65.19.161.135`, n7 is `ubuntu@100.66.155.50`, n5 is `ubuntu@100.79.82.11`. The shared inference lock is `/mnt/lvm_data/sft_analysis/extracted/trace_labeling/runs/fleet_medium/inference.lock`.

Next checks: monitor ongoing generation and SFT memory on later full-length examples, plus scheduled checkpoint retention. Keep review paused until the later review phase is requested. Before training on generated CoTs, apply prefix-only leakage checks and accept only passed originals or freshly reviewed replacement selections. Record real throughput before giving a full-corpus ETA. Preserve original artifacts and use explicit identities when restarting any process.
