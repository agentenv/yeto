# S29 MFU-B: Real-Length (262k + Compressed Segmentation) RL Measurement Report

Data: Modal base samples, 68 groups / 262 trajectories (`npu1:/data/work/s29-mfu/sample-buffer-modal68`), untruncated.
Real length distribution: p50 28.4k, p90 116.5k, max 208.6k, 12.22M tokens total; prompt max 32.4k (compressed-segmentation samples), response max 200.6k.
Cluster: npu1+n2+n3, 24× 910B3 in total (64 GiB/card), Nemotron 120B, TP4/PP3/EP8 as the baseline layout.

## 1. Conclusions (results first)

- **Plan B runs on this cluster, but cannot cover the upper end of real lengths.** The stable training configuration is a per-card micro-batch budget of 28672 tokens; samples longer than 29k cannot enter training.
- **The bottleneck is memory, not compute.** Every step up in the per-card micro-batch budget OOMs; the measured limit lies between 28672 and 36864.
- **All three CP options failed in practice.** CP8/TP1, CP4/TP2 and CP2/TP4 all OOM at per-card budgets of 28672~40960.
- **Measured MFU 8.5%** (real lengths, same accounting as SFT), below Plan A's 11.2% and SFT's 18.6%.

## 2. Memory limit measurements (descending: budget from large to small)

All are real 24-card runs. "Shortfall" is the size of the last failed allocation at OOM; already-allocated at OOM is consistently about 57 GiB/61 GiB.

| Run | Layout | max_token_len | Per-card tokens | Result | Shortfall |
|---|---|---|---|---|---|
| s29-b1 | TP4/PP3/CP1 | 262144 | 262144 | OOM | 8.37 GiB |
| s29-b4 | TP4/PP3/CP1 | 131072 | 131072 | OOM | 3.17 GiB |
| s29-b5 | TP4/PP3/CP1 | 98304 | 98304 | OOM | 960 MiB |
| s29-bg | TP4/PP3/CP2 | 81920 | 40960 | OOM | 954 MiB |
| s29-b6 | TP4/PP3/CP1 | 81920 | 81920 | OOM | 2.01 GiB |
| s29-b7 | TP4/PP3/CP1 | 73728 | 73728 | OOM | 1.77 GiB |
| s29-b8 | Same as above + E=1 expert recompute on all stages | 73728 | 73728 | OOM | 698 MiB |
| s29-bh | TP4/PP3/CP2 | 65536 | 32768 | OOM | 822 MiB |
| s29-bn | TP2/PP3/CP4 | 114688 | 28672 | OOM | 584 MiB |
| s29-bf | TP1/PP3/CP8 (full coverage of 208.6k) | 262144 | 32768 | OOM | 2.05 GiB |
| s29-b9 | TP4/PP3/CP1 | 57344 | 57344 | OOM | 1.37 GiB |
| s29-bi | TP4/PP3/CP1 | 49152 | 49152 | OOM | 856 MiB |
| s29-bj | TP4/PP3/CP1 | 40960 | 40960 | OOM | 338 MiB |
| s29-bk/bl | TP4/PP3/CP1 | 36864 | 36864 | OOM | 926 MiB |
| **s29-bm** | **TP4/PP3/CP1** | **28672** | **28672** | **Runs** | — |
| **s29-bo** | **TP4/PP3/CP1 (larger step)** | **28672** | **28672** | **Runs** | — |

Control experiment (ruling out environment factors): s29-bl uses an environment **identical** to A6 (no E=1, comm overlap on), changing only the data to B's real lengths; 36864 still OOMs. The limit is therefore set by data length and token budget, not by these switches.

## 3. Main memory consumer at OOM

The OOM stack consistently lands on the MoE expert FC1 activation:
`experts.py:777 bias_act_func → ascend_moe_fused_relu2.py:84 torch.empty_like`.
The reason is that the MoE permute activation scales as `tokens × top_k(22) / EP`; for long sequences it is far larger than attention.
Enabling expert recompute on all stages (adding `E=1` of `ASCEND_RECOMP_POLICY` to pp0/pp1) does help: at the same 73728 budget, the failed allocation drops from 1.77 GiB to 698 MiB, but this is not enough to clear the limit.

## 4. Why CP does not help

A refuted assumption: it was previously believed that "EP8 and CP are mutually exclusive". This was discarded after checking `megatron/core/parallel_state.py:780-800` — Megatron 0.18.2 uses MoE Parallel Folding; expert groups are built independently from `ETP × EP × PP × EDP = world_size` (`RankGenerator(..., cp=1)`), unrelated to the dense CP/DP. EP8 is fully valid under TP4×PP3×CP2.

The real problem with CP is that **it does not reduce the per-card token count on the expert side**: the measured OOM shortfall varies with `max_token_len` (= MAXTOK×CP), not with the per-card MAXTOK. CP2 (per-card 40960/32768) and CP4 (per-card 28672) both OOM, and CP itself needs extra communication buffers.
All three options were tried:

| Layout | Longest sample covered | Result |
|---|---|---|
| TP1×PP3×CP8 | 262144 (all) | OOM 2.05 GiB (under TP1, dense weights and logits for vocab 131072 are not sharded) |
| TP2×PP3×CP4 | 114688 | OOM 584 MiB (TP2 doubles dense weights compared with TP4) |
| TP4×PP3×CP2 | 65536 | OOM 822 MiB |

PP6 is infeasible: `world_size % (ETP × EP × PP) == 0` means EP8+PP6 requires 48 cards.

## 5. MFU (same accounting: tokens × 7.34e10 / (s × 24 × 313e12))

| Run | Data | tokens/step | Step time | MFU |
|---|---|---|---|---|
| SFT 18.60 recipe | 28k binpack | 2.642M | 138.8s | 18.59% |
| A6 | Pre-filtered short samples, 28k | 2.646M | 230.3s | 11.23% |
| **s29-bo** | **Real lengths ≤28672** | **3.012M** | **345.1s** | **8.53%** |
| s29-bm | Real lengths ≤28672, small step | 1.506M | 259.6s | 5.68% |

Accounting notes:
- Step times for A6 and B are both train steps (including the extra forward for old_log_prob) and are not fully isomorphic to SFT's 138.8s; A-series profiling measured A6's update_actor alone at 187.7s (13.77%). No per-phase timing was obtained for B.
- **The 7.34e10 constant excludes the quadratic attention term.** B's sequences are longer and its true FLOPs higher, so the table underestimates B. Estimated from this step's actual Σ L²=5.49e10, 8 full-attention layers, hidden 4096 and full recompute, attention contributes about 6.5% extra, i.e. s29-bo's attention-inclusive MFU is about **9.1%**. This is an estimate, not a measurement.

## 6. Why B is slower than A

At the same per-card token budget (28672), B is about 24% slower than A (token-normalized). Confirmed factors:
1. **Longer sequences**: B's mean length is 17.1k vs. shorter for pre-filtered A, so each micro-batch holds fewer sequences and length-dependent attention and Mamba costs rise.
2. **Fewer micro-batches**: B has 3.01M tokens per step / 2 DP / 28672 ≈ 52, comparable to A; but B's long sequences make binpack packing less even.
3. **The two items confirmed by A-series profiling also apply**: old_log_prob costs about 42s per iteration; the host side adds about 1.3s per micro-batch.

**Not done**: the msprof + py-spy descending bottleneck table for B was not produced. The time went into bisecting the 15 OOMs above, and the runnable configuration was obtained near the end. This is the largest gap of this round.

## 7. What it takes to reach real 262k

As measured, the per-card micro-batch budget limit is ≈ 28672 tokens, while covering the 208.6k sample requires `MAXTOK × CP ≥ 208589`, i.e. CP ≥ 8. The only feasible 24-card layout for CP8 is TP1×PP3×CP8, which OOMs because TP1 does not shard dense weights and logits. Therefore:

- **It does not fit on 24 cards.** More cards are needed (e.g. 48 cards for TP2×PP3×CP8, or TP4×PP3×CP4), or
- squeeze dense-side memory one more notch: more aggressive gradient/optimizer offload, chunked logits (vocab 131072 is the main consumer), activation CPU offload,
- or keep using compressed segmentation and cut overlong trajectories to under 29k at sampling time (currently 88% of samples are already under 131k, but only about 1/3 are under 29k).

## 8. Artifacts

- Run directories: `npu1:/data/work/s29-mfu/runs/s29-b{1,4,5,6,7,8,9,f,g,h,i,j,k,l,m,n,o}`
- Store build scripts: `npu1:/data/work/s29-mfu/build_b.py` (filters by length cap, takes the longest k per group), `filter_store.py`
- Parameterized launch scripts: `npu1:/data/work/s29-mfu/island_b.sh` (TP/PP/EP made environment variables) and the various `b*_chain.sh`
- No yeto source was modified, so no local branch commit was created.

---

## 9. Descending bottleneck table for B (measured, added after profiling)

Profiling method: reuse the s29-prof tap (forward_step/backward_step hooks in `megatron.core`); the B arm and SFT arm use the same window (2 micro-batches), same card positions (one per PP stage), same START=70 (second step). B arm traces: `npu1:/data/work/s29-prof/traces/b/pp{0,1,2}`, comparison result `cmp_b_vs_sft.json`.

### Window wall time (interval union, mutually exclusive)

| stage | Arm | Window wall time s | Pure compute % | Pure comm % | Overlap % | Uncovered idle % |
|---|---|---|---|---|---|---|
| pp0 | B | 29.406 | 7.8 | 87.6 | 4.1 | 0.5 |
| pp0 | SFT | 33.245 | 9.1 | 85.2 | 5.4 | 0.3 |
| pp1 | B | 16.009 | 14.3 | 77.5 | 7.4 | 0.8 |
| pp1 | SFT | 18.531 | 16.2 | 73.2 | 9.7 | 0.9 |
| pp2 | B | 3.823 | 63.6 | 0.1 | 31.0 | 5.3 |
| pp2 | SFT | 5.154 | 58.3 | 5.7 | 34.2 | 1.8 |

pp0 device time breakdown (summed kernel time, may overlap): every compute op category in B is only 0.59~0.74× of SFT (matmul 0.67, moe_gmm 0.72, mamba 0.72, attention 0.74), except elementwise 1.52 and reduce 2.26.

### Descending bottlenecks (by impact on MFU, largest first)

| # | Bottleneck | Magnitude | Evidence | Optimizable? |
|---|---|---|---|---|
| 1 | **binpack fill rate only 71.4%** | Directly wastes about 29% of compute | Measured packing: 176 samples packed into 176 micro-batches, i.e. one per sample. Real samples average 17.1k, so any two exceed the 28672 budget. This is why every op category in B's trace is 0.7× of SFT. | Raising fill requires raising MAXTOK, and MAXTOK>28672 always OOMs (see Section 2). **No solution on 24 cards** |
| 2 | **About 1.3s host-side per micro-batch** | B about 114s / 338s step (34%) | A-series profiling measured an extra 1.3s per micro-batch not present in the device trace. B has 88 micro-batches per DP (A6 has 47), so this item nearly doubles in B | Optimizable, but the specific function was not located this round (py-spy not done) |
| 3 | **old_log_prob extra full forward** | A-series measured about 42s/step | Included in step time; B's sequences are longer, so this only grows | **Cannot be avoided via bypass**: yeto explicitly rejects it (it would set old_log_probs=rollout_log_probs and lose pi_prox) |
| 4 | GRPO loss elementwise / reduce | 1.52x / 2.26x on pp0, about 0.15s/window absolute | See table above | Small gain, low priority |

### Optimizations tried that failed

| Optimization | Result |
|---|---|
| Raise MAXTOK (131072→28672, 10 steps) | All OOM, limit 28672 |
| CP2 / CP4 / CP8 (three TP combinations) | All OOM |
| MoE recompute E=1 on all stages | Helps but insufficient (failed allocation 1.77 GiB → 698 MiB) |
| PP layer splits 42/42/5, 24/x/30 | Still OOM, with larger shortfall (pressure shifts to another stage) |
| Toggle comm overlap / expandable_segments | Already in effect, no additional gain |
| old_log_prob bypass | Rejected by yeto by design |

## 10. Final numbers

| Run | Data | tokens/step | Step time | MFU |
|---|---|---|---|---|
| SFT 18.60 recipe | 28k binpack, fill about 100% | 2.642M | 138.8s | 18.59% |
| A6 | Pre-filtered short samples | 2.646M | 230.3s | 11.23% |
| **s29-br (B final)** | **Real lengths, fill 71.4%** | **3.012M** | **338.2s** | **8.70%** |

The 18.6% target was not reached. The root cause is item 1: real-length samples cannot fill the 28672 per-card budget, and the budget cannot be raised due to the hard memory limit. On 24 cards this is a physical limit, not a tuning problem.

## 11. Direct evidence that fill rate is the main cause

Using the same packing approximation (first-fit decreasing, budget 28672), compute the actual micro-batch composition of both arms:

| Arm | Samples | Mean length | Micro-batches | Fill rate | Step-time MFU |
|---|---|---|---|---|---|
| A6 | 168 | 24.3k | 168 | 84.6% | 11.23% |
| B (s29-br) | 176 | 20.5k | 176 | 71.4% | 8.70% |

In both arms **each sample occupies one micro-batch** (168 samples in 168 bins, 176 samples in 176 bins): agentic trajectories are 17~28k long, any two exceed the 28672 budget, and verl's packing cannot merge them. Thus the effective tokens per micro-batch per card equal the sample's own length, and **MFU is approximately proportional to mean length / budget**:
- A6: 24.3/28.672 = 84.6% → 11.23%
- B: 20.5/28.672 = 71.4% → 8.70%
- Ratio 0.84 vs 0.77; fill rate explains most of the gap between the arms, with the remainder from length-dependent attention/Mamba costs of long sequences and the difference in micro-batch count.

A control group with "all samples ≤14336 so each bin holds two" was intended for direct validation, but only 3 groups / 12 samples in the real data meet that condition, too few to form a step, so it was not done.

**Implication**: on this cluster there are only two ways to raise B's MFU — (a) raise the per-card token budget (hard memory limit, 28672 is already the cap); (b) cut overlong trajectories shorter at sampling time so several fit in one budget (i.e. push compressed segmentation harder). Scheduling and operator-level optimizations (comm overlap, plugin stack, recompute policy) were shown in the A series to have only a few tenths of a percentage point of headroom.

## 12. Where the 114s host-side time goes (measured with py-spy)

Method: `py-spy dump` on the `ray::WorkerDict` process, 92 consecutive samples (3s interval), active threads only (Ray actor methods run on "AsyncIO Thread: default", not MainThread — the first two sampling attempts failed because only MainThread was inspected). `--native` is not supported on aarch64, so plain dump was used. Sample locations: 83 in `actor_update_actor`, 8 in `actor_compute_log_prob`.

| Share | Leaf frame | Call chain | Nature |
|---|---|---|---|
| 30% (28/92) | `synchronize (torch_npu/npu/streams.py:258)` | `_maybe_dtoh_and_synchronize (moe/token_dispatcher.py:930)` ← `token_dispatch` ← `moe_layer.dispatch` | Before MoE all-to-all, per-expert token counts must be D2H'd to host: one full-device sync per MoE layer per micro-batch |
| 24% (22/92) | `synchronize (torch_npu/npu/utils.py:75)` | `_communicate_shapes (p2p_communication.py:263)` ← `_communicate` ← `send_forward` | Shape handshake before pipeline P2P send, with a full-device sync as race protection |
| 13% (12/92) | `_fused_adam (torch/optim/adam.py:879)` | `adam` ← `optimizer.step` | With the optimizer offloaded to CPU, Adam runs serially on host |
| 4% (4/92) | `_engine_run_backward` | Recompute backward | Normal compute |
| Rest | MoE permute / mamba / TE gemm | — | Normal compute |

The three items together account for 67% of the host thread.

**Relation to SFT**: SFT pays these three items too (same recipe, same optimizer offload). What makes B more expensive is **how many times it pays** — the first two are fixed "per micro-batch × per layer" costs, and because B's fill rate is only 71.4%, the same token count is split into 88 micro-batches/DP (SFT: 48), so these host syncs are paid about 1.8× more. This collapses bottleneck item 2 back into item 1: **fill rate is the overall root cause**.

**Possible optimizations (not implemented, for the next round)**:
1. Whether the full-device sync in `_maybe_dtoh_and_synchronize` can be replaced with a stream-level wait, or per-expert counts kept on device (MindSpeed already has fused permute, but counts still go back to host). This is the largest single item.
2. The `_communicate_shapes` sync: the existing plugin `ascend_sync_p2p_shape.py` only caches construction of the shape tensor, and its comment explicitly says it "does not change the `torch.cuda.synchronize()` race protection", so this 24% remains.
3. CPU Adam 13%: consider lowering the optimizer offload fraction from 1.0, trading memory for time — but memory is exactly the current hard constraint, so the two conflict.

## 13. Before/after optimization comparison (measured item by item)

Baseline repeatability: the same configuration was run three times, step times 338.2 / 345.1 / 345.5s, run-to-run variation about 2%. An optimization must exceed this band to count as effective.

| # | Optimization | Approach | Step time | vs. baseline | Conclusion |
|---|---|---|---|---|---|
| 0 | Baseline (s29-br/bo/bu) | — | 338.2 / 345.1 / 345.5s | — | MFU 8.53~8.70% |
| 1 | Eliminate MoE D2H sync (host item 1, 30%) | `moe_pad_expert_input_to_capacity=True` + `moe_expert_capacity_factor=1.25` (drop-and-pad; the dispatcher uses static counts and does no D2H sync at all) | **351.3s** | **+1.7% (within noise, no gain)** | **Ineffective**. Capacity padding also adds expert-side compute and introduces token drops affecting correctness; not recommended |

**This negative result matters**: the 54% of `synchronize` in the host thread is not host overhead itself but **the host waiting on the device**. Removing the MoE sync point entirely did not improve step time, showing the device side is the bottleneck; the A-series "extra 1.3s host overhead per micro-batch", derived by back-calculation from step time, should not be read directly as recoverable host time.

This corrects the Section 9 descending table: of the three host-side items (MoE sync 30%, P2P shape sync 24%, CPU Adam 13%), at least item 1 is confirmed unrecoverable. Items 2 and 3 are likely the same (both are waits overlapping device compute), but were not separately verified.

**The conclusion stands and is stronger**: the real bottleneck is item 1 — binpack fill rate 71.4%, i.e. the device does only 71% useful work per micro-batch. The only ways up are raising the per-card token budget (hard memory limit 28672) or cutting trajectories shorter at sampling time.

## 14. Important correction: the fill-rate hypothesis is refuted by experiment

Sections 9, 11 and 13 treated "binpack fill rate 71.4%" as the top bottleneck. That judgment **is wrong** and has been overturned by a control experiment.

Approach: use `build_b_split.py` to cut each real sample by token position into a segment ≤14336 (prompt kept intact, response truncated, `loss_mask` and `rollout_log_probs` truncated in sync at the same index so the three arrays stay aligned), reproducing the length distribution after "finer compressed segmentation". Fill rate rises from 71.4% to 98.6%, and micro-batch count drops from 176 to 114.

| Run | Mean length | Fill rate | Micro-batches | tokens/step | Step time | MFU |
|---|---|---|---|---|---|---|
| Baseline s29-bo/br/bu | 20.5k | 71.4% | 176 | 3.01M | 338.2 / 345.1 / 345.5s | 8.53~8.70% |
| **s29-s2 (simulated finer segmentation)** | **14.1k** | **98.6%** | **114** | **3.17M** | **364.3s** | **8.50%** |

**Fill rate up 38%, micro-batch count down 35%, and no MFU improvement at all.**

Reason: verl has `use_remove_padding=True` (THD packing), so a partially filled micro-batch computes only on real tokens and **there is no padding waste**. The trace observation that "B's ops are only 0.7× of SFT" merely reflects that B's micro-batches carry fewer tokens, not waste. Reading it as waste was an inference error on my part.

Both main hypotheses have now been refuted by experiment:

| Hypothesis | Verification | Conclusion |
|---|---|---|
| Host-side synchronize is recoverable overhead (Section 12) | Drop-and-pad eliminates MoE D2H sync | Refuted: 351.3s vs 338~346s, no gain. Those synchronize calls are the host waiting on the device |
| binpack fill rate is the top bottleneck (Sections 9/11/13) | Simulated finer segmentation raises fill rate to 98.6% | Refuted: 364.3s, no gain. Under THD packing there was no padding waste to begin with |

**Unexplained gap** (in effective tokens per second):

| Arm | tokens/step | Step time | tok/s | vs. SFT |
|---|---|---|---|---|
| SFT | 2.642M | 138.8s | 19035 | 1.00 |
| A6 | 2.646M | 230.3s | 11490 | 0.60 |
| B (s29-s2) | 3.17M | 364.3s | 8706 | 0.46 |

Of A6's 1.66x vs. SFT, A-series profiling has attributed: update_actor 187.7s (1.35x) + old_log_prob about 42s. B is a further 1.32x slower than A6, still slower with fewer, fuller micro-batches and shorter samples; **the cause is not located**. The next round should directly do same-accounting per-phase timing on both B and A6 arms (old_log_prob / forward-backward / optimizer / rest), rather than continue inferring from trace ratios — both inferences this round were falsified by experiment.

## 15. Three supplementary checks

### 15.1 Phase breakdown (py-spy sampling)

The 92 active-thread samples from Section 12 already give a phase distribution (which Ray actor method each sample landed in):

| Phase | Sample share | Scaled to 345s step time |
|---|---|---|
| `actor_update_actor` (training forward/backward + optimizer) | 90% (83/92) | about 311s |
| `actor_compute_log_prob` (extra forward for old_log_prob) | 9% (8/92) | about 34s |

Same order of magnitude as A-series profiling's ~42s old_log_prob for A6. **Conclusion: the gap is not in old_log_prob but in update_actor itself.** B's update_actor is about 311s / 3.01M tokens, SFT's whole step 138.8s / 2.64M tokens; per token, update_actor is about 2.0× slower.

### 15.2 FLOPs accounting: attention O(L²) is not the explanation

Only 8 of Nemotron 120B's 89 layers are full attention (measured from `layers_block_type`), hidden 4096; with full recompute each attention layer costs about `8·L²·d`, i.e. about `2.62e5 · ΣL²` for the whole model.

| Arm | ΣL² (per step) | attention FLOPs | Linear-term FLOPs | attention share | MFU incl. attention |
|---|---|---|---|---|---|
| B (s29-s2, 224 samples × 14.1k) | 4.45e10 | 1.17e16 | 2.33e17 | +5.0% | 8.50% → **8.93%** |
| SFT (28k binpack, estimated with mean doc length 8k) | 2.11e10 | 5.5e15 | 1.94e17 | +2.8% | 18.59% → **19.11%** |

The gap is almost unchanged under both accountings (2.19x → 2.14x). **The quadratic attention term of long sequences does not explain the gap.** The SFT row's ΣL² uses an estimated mean doc length of 8k, so it is an estimate, but even extreme values would not change the order of magnitude.

### 15.3 Config diff: recompute and other key items match on both sides

The `override_transformer_config` actually in effect for RL (extracted from s29-s2's hydra overrides, 13 items):
`recompute_granularity=full`, `recompute_method=uniform`, `recompute_num_layers=1`, `moe_token_dispatcher_type=alltoall`, `moe_permute_fusion=True`, `moe_grouped_gemm=True`, `moe_router_dtype=fp32`, `moe_aux_loss_coeff=0.0`, `moe_router_bias_update_rate=0.0`, `gradient_accumulation_fusion=False`, `tp_comm_overlap=True`, `num_layers_in_first_pipeline_stage=29`, `num_layers_in_last_pipeline_stage=30`.

The SFT recipe (`/data/work/sft-run/go_pp3.sh`) defaults to `RECOMPUTE=full`, `RECOMPUTE_METHOD=uniform`, `RECOMPUTE_NUM=1`, and likewise layers `ASCEND_RECOMP_POLICY` on top. **Recompute configuration is identical on both sides** and is not a source of difference. The RL-side plugin stack is `/data/work/sft-run/plugins`, the same one.

The only known misaligned item: `stage_timer.py` of `ASCEND_STAGE_TIMER` is excluded from the plugin stack on the RL side (it imports ms-swift at module level, which the RL path lacks), so RL cannot get the recipe's built-in per-phase timing — which is also why this round could only do the phase breakdown via py-spy sampling.

### 15.4 Conclusion and next steps

Inside update_actor it is 2.0× slower per token, while recompute/plugins/parallel shapes/attention accounting have all been ruled out. The most likely remaining directions (unverified):
1. verl's Megatron training loop and ms-swift's SFT loop differ in **fixed per-micro-batch overhead** (data movement, TensorDict slicing, loss construction). Note that Section 14 proved this is not a matter of micro-batch *count* — B is still slower with fewer, fuller micro-batches, so the difference is **inside each micro-batch**.
2. The GRPO loss path (TIS correction, advantage, per-token ratios) is heavier than SFT's CE. The trace's elementwise 1.52x and reduce 2.26x already point here, but the absolute values looked small at the time — after ruling out other items, this is worth re-quantifying.
The next round should add timers directly in the verl actor's micro-batch loop, splitting one micro-batch into data prep / forward / backward / loss / rest, and compare with SFT under the same accounting.

## 16. Wide-window profiling (NWIN=20) and PP schedule check

### 16.1 PP schedule suspicion ruled out

At `verl/workers/engine/megatron/transformer_impl.py:919`, `forward_backward_func` receives all micro-batches **at once** (`num_microbatches=n_micro_batch`, fed uniformly by `make_batch_generator`); the 1F1B pipeline is not invoked per micro-batch. `ppo_mini_batch_size = NGROUPS × rollout.n` gives only 1 mini-batch per step, hence only one warmup/cooldown. **The suspicion that "verl runs a separate 1F1B per micro-batch, so PP3 idles two-thirds of the time" does not hold.**

### 16.2 In-window vs. end-to-end: the gap is outside the micro-batches

Normalizing the 2-micro-batch window (Section 9) by tokens:

| Arm | Window | tokens | pp0 window wall time | tok/s (pp0 view) |
|---|---|---|---|---|
| SFT | 2 micro-batches | 56k | 33.245s | 1687 |
| B | 2 micro-batches | 41k | 29.406s | 1394 |

**In-window, B is only 1.21× slower than SFT**, while end-to-end it is 2.0× slower. The difference lies outside the window.

### 16.3 NWIN=20 wide-window results (partial)

`s29-pb21`, START=70, NWIN=20, traces in `/data/work/s29-prof/traces/b21/`, comparison result `cmp_b21.json`.

| stage | Arm | Window wall time | Per micro-batch | Pure compute % | Overlap % | Uncovered idle % |
|---|---|---|---|---|---|---|
| pp2 | B (20 micro-batches) | 47.889s | 2.39s | 65.7 | 30.9 | 3.1 |
| pp2 | SFT (2 micro-batches) | 5.154s | 2.58s | 58.3 | 34.2 | 1.8 |

pp0 and pp1 were marked REFUSED by the analyzer (with a wide window the scope count violates the "exactly one" constraint of `analyze_fullstep.py`). These two stages are exactly the ones that previously showed 85~88% in `pp_sendrecv`; **the two most important stages were not captured**, which is the gap in this item.

The available pp2 data show: **uncovered device idle is only 3.1%**, and 2.39s per micro-batch is even slightly faster than SFT's 2.58s (while B's micro-batches carry only 0.73× the tokens). So at pp2, B is 1.26× slower per token, consistent with the in-window conclusion of 16.2 and far from explaining the end-to-end 2.0×.

Compared end-to-end: B has 88 micro-batches/DP per step and update_actor about 328s → 3.73s wall time per micro-batch; in-window measured 2.39s (and the window still includes profiler overhead, measured at about 2.64× amplification on the recipe). The two differ several-fold, **further pointing to "time spent outside micro-batch forward/backward"**, consistent with the original A-series profiling conclusion (attributed to the host side at the time, but Sections 12~14 showed it is not recoverable host overhead).

### 16.4 The single entry point for the next round

Bypass the analyzer's scope constraint and instrument the verl micro-batch loop directly (data prep / forward / backward / loss / rest), comparing with SFT under the same accounting. This round has narrowed the scope to: **inside update_actor, outside micro-batch forward/backward, and not recoverable host overhead, not PP scheduling, not fill rate, not recompute config, not attention accounting**.

## 17. update_actor phase timing, diagnosis and fix (B2, measured)

### 17.1 Method

Added phase timing to the `update_actor` / `compute_log_prob` paths in verl-fork (`verl/utils/s29_phase_timer.py`; at each phase boundary call `torch.npu.synchronize()` then read wall clock); each call writes one JSON line per rank to `$HOME/phase/`. Phases: entry → build iterator → train_mode enter/exit (onload/offload) → token count all_gather → `zero_grad` → `forward_backward_batch` (subdivided into token count all_reduce, `prepare_micro_batches`, `forward_backward_func`, post-processing) → `optimizer.step` → lr scheduler → metrics aggregation → output `.cpu()`. Level 2 additionally wraps Megatron's `forward_step` / `backward_step` and each `P2PCommunicator` call (per micro-batch, synchronized), separating each stage's compute from pipeline waits. The patch and report scripts are in `tools/s29-phase/` on local branch `s29-mfu-b2` (`apply_patch.py`, `s29_phase_timer.py`, `phase_report.py`, `verl-fork-s29-phase.diff`); the original files on all three machines were backed up to `/data/work/s29-mfu/verl-fork.bak-20261031-0646.tgz`.

Configuration is identical to s29-bu (44 groups×4, SEQ 28672, 3.0117M tokens/step, same store data); each run does 3 iterations.

### 17.2 First finding: all previous baselines measured only iteration 0 (including warmup)

| Run | Iteration 0 | Iteration 1 | Iteration 2 |
|---|---|---|---|
| s29-c1 (baseline, with timers) | 349.3s | 249.9s | 252.1s |
| s29-c2 (baseline for iterations 0/1) | 346.1s | 254.0s | — |

Iteration 0 is about 100s slower than steady state (forward/backward 227s vs 149s, optimizer 49~60s vs 43~48s, old_log_prob 53s vs 44s), due to first-iteration compilation/allocator/communication setup overhead. Sections 12~16 and the 338~346s baseline are all iteration-0 numbers. **The steady-state baseline is 250~254s, MFU 11.7%, not 8.5%.** Most of the earlier "3.73s wall time per micro-batch vs 2.39s on device" gap is this warmup; also, there are actually only 63 micro-batches per DP, not 88.

### 17.3 Steady-state descending time table (s29-c1 iteration 1, level 2; pp0/pp1/pp2 medians)

| # | Phase | pp0 | pp1 | pp2 | Notes |
|---|---|---|---|---|---|
| 1 | update_actor backward_step (63 calls) | 99.2s | 97.5s | 100.5s | Normal compute (including full recompute) |
| 2 | compute_log_prob total | 43.8s | 44.1s | 44.6s | old_log_prob forward, 99% inside forward_backward_func |
| 3 | **optimizer.step** | **41.8s** | **43.0s** | **48.3s** | **Single-threaded CPU Adam (see 17.4)** |
| 4 | update_actor forward_step (63 calls) | 38.9s | 39.3s | 44.1s | Normal compute |
| 5 | P2P wait (pipeline bubble + inter-stage imbalance) | 11.8s | 11.4s | 2.4s | pp0 mostly in send_forward_recv_backward |
| 6 | prepare_micro_batches | 0.47s | 0.45s | 0.46s | |
| 7 | token count all_reduce + `.item()`, all_gather, zero_grad, metrics, onload/offload, output `.cpu()` | all <0.15s | | | Negligible |

forward_backward_func itself is 147~150s, of which device compute (fwd+bwd) is 138~145s and P2P wait ≤12s (≤8%). **Neither pipeline bubbles nor inter-stage imbalance are major items**: fwd+bwd differs by only 3% across the three stages (pp2 carries logits and is heaviest), the 29/x/30 split is already near balanced, and sorting micro-batches by length could recover at most part of this ≤12s. onload/offload, metrics and `.item()` syncs are all negligible.

**Outside micro-batch forward/backward, the only major item is optimizer.step (42~48s per step, 22~25% of update_actor).**

### 17.4 Root cause: Ray sets OMP_NUM_THREADS to 1 for WorkerDict

The timer also records the worker process's `torch.get_num_threads()` and `OMP_NUM_THREADS`: **both are 1** (Ray defaults to 1 for actors that do not set it explicitly). The recipe uses `optimizer_offload_fraction=1`, so Adam runs entirely on host, hence single-threaded `_fused_adam`. A standalone test on npu1 with 200M parameters: 1 thread 1.75s, 4 threads 0.44s, 16 threads 0.12s (8.8 → 0.6 ns/param), matching the 43~48s magnitude.

### 17.5 Fix and measurements

| Run | Approach | Iteration 0 | Steady-state step time (iterations 1/2) | optimizer.step | Forward/backward FBF | MFU (steady state) |
|---|---|---|---|---|---|---|
| s29-c1 / c2 iteration 1 | Baseline (threads=1) | 349.3 / 346.1s | 249.9 / 252.1 / 254.0s | 42~50s | 146~148s | **11.7%** |
| s29-c2 iteration 2 | Global `torch.set_num_threads(16)` | — | 230.7s | 11~19s | 160~163s (+15s) | 12.7% |
| **s29-c3** | **16 threads only during optimizer.step, restored afterwards** | **298.9s** | **217.3 / 213.7s** | **10.7~15.0s** | 146~150s (unchanged) | **13.7%** |

- Enabling threads globally slows forward/backward by 15s (idle OpenMP threads compete for CPU with host-side kernel launches), so the fix raises the thread count only inside `optimizer.step` (`verl/utils/s29_phase_timer.optimizer_threads`, controlled by `OPT_THREADS_ON`/`YETO_S29_OPT_THREADS`, off by default).
- Steady-state step time 252 → 215.5s (−14.5%), MFU 11.7% → **13.7%**; iteration 0 348 → 299s (−14%), MFU 8.5% → 9.85% under the old accounting.
- Numerics: only the CPU thread count changes, Adam results are unchanged; clip_fraction matches on both sides (c1 0.00494/0.00482/0.00484, c3 0.00495/0.00493/0.00498, differences on the order of run-to-run noise).
- MFU accounting: 3.0117e6 × 7.34e10 / (seconds × 24 × 313e12).

### 17.6 Remaining gap vs. SFT

SFT: 138.8s / 2.642M tokens (19035 tok/s); after the fix, B steady state: 215.5s / 3.012M (13976 tok/s), 1.36x slower per token (2.0x under the original accounting). About 44s of this is the RL-only old_log_prob forward (SFT has none); excluding it, update_actor itself is 161s / 3.01M = 18700 tok/s, **essentially on par with SFT (0.98x)**. Remaining options: further tune optimizer thread count (nothing beyond 16 was swept); old_log_prob could be eliminated entirely if the algorithm allows reusing rollout log probs (an algorithm decision, not touched this round); the ≤12s P2P wait is the pipeline bubble ceiling.
