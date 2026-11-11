"""Miles / SGLang pins, bundle and image constants.

Moved from ``yeto/rl/__init__.py``; ``yeto.rl`` re-exports every name so
``from yeto.rl import MILES_NEXT_IMAGE`` keeps working.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

from yeto.rl.image_registry import pinned_image


class MilesRevisionPins(NamedTuple):
    """Expected Miles checkout identity (``revision.verify_miles_revision``)."""

    repository: str
    commit: str
    # Origins that also identify ``repository`` (the fork's earlier homes).
    # Images already built keep the origin they were built with.
    former_repositories: tuple[str, ...] = ()

    def accepts_origin(self, origin: str) -> bool:
        """True when ``origin`` (``.git`` suffix and trailing ``/`` ignored) names this fork."""

        normalized = origin.rstrip("/").removesuffix(".git").rstrip("/")
        return normalized in (self.repository, *self.former_repositories)

MILES_PEFT_VERSION = "0.20.0"

# Ports engine (the only Miles engine): upstream Miles plus yeto compatibility
# commits on michaellchung/miles ``yeto/ports``; fetched directly, no bundle.
MILES_NEXT_REPOSITORY = "https://github.com/michaellchung/miles"
# Earlier homes of the fork that pinned images may still carry as their
# /root/miles origin. Empty while the newer commits (d1616ba12 and its lineage)
# live only on michaellchung/miles; the move to agentenv/miles lists the
# michaellchung URL here.
MILES_NEXT_FORMER_REPOSITORIES: tuple[str, ...] = ()
# Upstream radixark/miles base of the frozen ports-v1 lineage (yeto/ports ..
# yeto/next2; the DiLoCo image 28bc6cc-2fa8801 runs it).
MILES_PORTS_V1_UPSTREAM_COMMIT = "9e4260de047a704208535c0e90c531929879ab40"
# Upstream radixark/miles base of the k31 lineage (upstream/main at the fork
# point of k31/f1, the `k31-base` tag of the fork clone).
MILES_K31_UPSTREAM_COMMIT = "778794a807866a4f51f34d5e87dda3e55d9e85ad"
MILES_NEXT_UPSTREAM_COMMIT = MILES_K31_UPSTREAM_COMMIT
# yeto/ports (= yeto-elastic-m1-m6 after four review rounds): run_plugin,
# --worker-dynamic-port-start, the elastic M1-M6 changes and the LoRA bridge
# calculate_per_token_loss fix on top of the base, plus (5c1b49eb) the 2b
# loss variants: --policy-loss-variant {policy_loss,cispo,sapo,gmpo},
# --sapo-tau-{pos,neg}, --gmpo-log-clip-{low,high}; plus (2f23a0fc) cell declaration:
# placement map rollout_cells, deferred/unbound cells, describe_cells,
# public slice_pg_info, bundle-free check before any cell starts; plus
# (fb04d6ff) M5: LoRA DP-invariant restore no longer drops DistOpt exp_avg/exp_avg_sq,
# and (e3a11ab3) its cross-optimizer error text/docstring and tests.
# image-m3a27b (c35702e = merge of m3-qwen4exp-lora ab904f43a [Qwen3.8-Next LoRA layout]
# + fork-a27-worker-loss 857fc9592 [workers_lost cells, bounded weight-update group
# rendezvous --update-weight-group-timeout-s; ExternalFailureError /
# RolloutEngineJoinError: an engine-side failure no longer kills the trainer]
# on yeto/ports e3a11ab38).
# raw-lora-disagg (8bc52237a = c35702e + broadcast LoRA gathers the adapter across PP onto one sender,
# actor allows --megatron-to-hf-mode raw for non-colocated LoRA).
# abort-discard-stats (efbbc63ea = 8bc52237a + rollout abort tallies discarded
# groups/samples/response tokens without partial rollout).
# 2f7871fb2 = efbbc63ea + the same tally on the train path actually in use
# (inference_rollout_train.abort; efbbc63ea only patched sglang_rollout.abort) and a
# failed surplus group no longer fails the abort.
# critic-algosup (64b591a4b = ddce20992 + critic family + algo-supplement metrics,
# local fork branch, not pushed; built into the image from the local clone).
# full-parameter-rpc (56068bd56 = 64b591a4b + full-parameter shard state,
# transport and TrainerController methods ported from agentenv/miles e25048edd;
# needed by full-parameter DiLoCo on the ports engine).
# sao-streaming-roles (386ae3b07 = 56068bd56 + LoRA-hybrid full-parameter
# transport (95533a0cd, 5f7ac4e32, 587bd258a) + SAO streaming
# primitives ported from agentenv/miles d5eb5e82a: content identity, per-role
# chunks, critic group RPC).
# miles-import2 (44d41efd0 = 17aeb5cb5 + the megatron_utils package
# import stays light, so RayWorkerManager no longer loads megatron.bridge).
# 17aeb5cb5 = 386ae3b07 + light Ray actors
# stay off megatron_utils + the master port is held until use.
# distopt-fp (56190a5a8 = ed5f64fd7 + full-parameter sync reads DP=1
# distributed-optimizer masters, incl. CPU offload); cc693c693 = + the
# Adam proof leaves the hybrid optimizer's master_param mirror out;
# a477bcbe0 = + one parameter may span several fragments via flat ranges;
# 0b64b40c1 = + rebase_full_parameter_scheduler_anchor for peer-snapshot recovery;
# b13c170ce = + a chunked apply skips fragments the masters already hold;
# aa6b887d8 = + deferred export with on-demand materialize, faster finite/hash
# checks, hybrid R3 layer mapping;
# d1616ba12 = + THD packs carry a per-sample seq_idx, so Mamba conv/SSM state
# restarts per sample;
# 8edea35a7 = + full-parameter chunks stored in Ray as numpy buffers (zero-copy
# get, one-copy put);
# yeto/closing (ebc6519af = merge of 8edea35a7 + 921cde8c7 full-parameter
# data-parallel replicas + d78bf39e4 HybridDeviceOptimizer CPU FP32
# masters in pinned memory, and b7868074c = d1616ba12 + the
# language-only bridge of the Nemotron-H VL checkpoint);
# 59aea07f9 = merge of ebc6519af + 63cb372b3 per-fragment full-parameter round
# boundary (lazy export, partial commit) (yeto/closing);
# 7402a53d5 = 59aea07f9 + acb798c7b sharded FP32 masters (distributed optimizer
# at DP>1) merged with 76a84963c set MoE runtime flags copied onto the
# bridge provider (yeto/eff);
# 697fb2000 = 7402a53d5 + training step without host syncs in R3 replay, loss
# logging, THD packing and Megatron's DeepEP multihot (runtime patch; same
# values bit for bit, MILES_HOST_SYNC_FREE=0 restores the old paths)
# (yeto/sync-free);
# b6662508e = 7402a53d5 + 102672760 stored broadcast fragments verified ahead of
# the apply (transport feature verified_apply) merged with 697fb2000
# (yeto/e3);
# 8d03f386a = 7402a53d5 + shape-free layout support: partition reports,
# installed owner maps, free-order owner plans (transport feature
# shape_free_layout; yeto/shape-free-layout); 3f6760f2b = + a
# gated linear_fc1 reports its gate and up blocks;
# 8faa7fccc = b6662508e merged with 3f6760f2b (yeto/next);
# ddd416167 = b6662508e + the commit reads its chunks ahead of the copy and
# casts the model on the master's device (same bits), e9410717a = + the
# read-ahead runs on Miles' hashing pool (yeto/sync-commit);
# 970807006 = 8faa7fccc merged with e9410717a (yeto/next);
# e99e2a168 = 7402a53d5 + the session server records OpenAI Responses calls
# (Codex) and logs failed model calls (cherry-picks of
# f01bc24cd, 1773afaa1) + a Nemotron-3 Nano Codex trajectory test
# (yeto/agentic-nemotron);
# 28bc6cc24 = 970807006 merged with e99e2a168 (yeto/next2).
MILES_PORTS_V1_COMMIT = "28bc6cc245e6a45c8ca2d36b632efa479f4f7a14"
# k31 lineage head: the F1 fork branch k31/f1 (upstream main 778794a8 + the
# yeto_ext trait, run_plugin, --worker-dynamic-port-start, light megatron_utils
# imports, rollout_runtime_stats and the Kimi-K3.1 model).
# k31/integ 4a839d8a = k31/f1 0eeb816b + F2 (128 trainable experts per layer);
# the commit the k31 image ships at /root/miles (docker/miles-k31/build.sh).
MILES_K31_F1_COMMIT = "4a839d8a663eead89ac4815b03304098321157f4"
# k31/b2: the B2 batch on top of k31/f1: the full-parameter shard state and
# transport ported from ports-v1 (miles.ray.full_parameter_transport,
# miles.backends.megatron_utils.full_parameter_{state,plugin,replicas,
# sharded_masters,chunk_reader}, the TrainerController proxies) plus the
# per-rank state helpers yeto reads (full_parameter_rank_state,
# set_model_weight_version) and apply_full_parameter_chunked_cut(weight_version=).
# k31/b2 f9454517 = feature head 3fb93880 + a019dede (the in-image test fix) +
# f9454517 (CPU test fakes that hold on every image host; tests only). The four
# k31 batch heads below are one linear history: f9454517 < a07553f0 < 96437f37 <
# ca31b08f < efc83590 (the commits the branches were rebased onto; earlier SHAs of
# the same branches are gone).
MILES_K31_B2_COMMIT = "f9454517e81f6b41087bcc85a605258d80fb6b9c"
# k31/b3-b7 a07553f0 = k31/b2 f9454517 + the B3/B5/B6/B7 batches ported from
# ports-v1: B3 an engine that dies or fails to join the weight-update group is
# an ExternalFailureError (RolloutEngineJoinError), not a trainer loss; B5 abort
# tallies / over-sampling metrics / --agentic-suspend-between-turns
# --agentic-suspend-max-rounds; B6 the session server records OpenAI Responses
# calls (Codex) and the LoRA bridge passes --calculate-per-token-loss; B7 the set
# MoE runtime flags copied onto the bridge provider (defaults read from the
# installed Megatron parser), the host-sync-free training step behind the typed
# DebugConfig.host_sync_free (--host-sync-free / --no-host-sync-free, default
# on; the MILES_HOST_SYNC_FREE environment variable is gone), the language-only
# bridge of the Nemotron-H VL checkpoint (NemotronH_Omni_Reasoning_V3) and the
# R3 replay stream mapping by router layer numbers for hybrid models (bridge
# copies --mamba-training-ssm-states-dtype).
MILES_K31_B3B7_COMMIT = "a07553f0a4f361b1618a707b416b690c5666e03d"
# k31/b4 96437f37 = k31/b3-b7 a07553f0 + the B4 elastic batch ported from
# ports-v1 M1-M6 onto the upstream structure: the explicit role -> bundle
# placement map (--yeto-placement-map, ClusterConfig; miles.ray.placement_map,
# re-exported by placement_group with the public slice_pg_info), deferred and
# rebindable cells on the startup placement group, the Miles router's cordon
# (/cordon_worker, /uncordon_worker, /worker_inflight, cordoned registration),
# epoch-guarded cell membership (InferenceMembershipMixin: start_cells /
# stop_cells(expected_epoch), wait_cells_tracked, drain_cells, uncordon_cells,
# admit_cells, start/end_commit_weight_version), the member publish
# (placement_group.update_member_weights / commit_weight_version, module
# miles.ray.rollout.member_publish) and the trainer rebuild
# (miles.ray.trainer_rebuild.rebuild_training_models). Dropped verbs, with what
# replaces them: cordon_cells (drain_cells), set_cells_weight_version /
# get_cells_weight_versions (the commit checks every Serving cell's version),
# ServerCell.wait_drained (get_inflight + drain_cells),
# start_cells(wait_tracked_timeout_seconds=) (start_cells, then
# wait_cells_tracked), placement_group.update_weights(members=)
# (update_member_weights), placement_group._slice_pg_info (slice_pg_info).
MILES_K31_B4_COMMIT = "96437f3732a1232bb8a99c8a6d18f6fd2a37dc24"
# k31/b8 = k31/b4 96437f37 + the B8 loss / critic batch: 4ff093a8
# --policy-loss-variant {policy_loss,cispo,sapo,gmpo} with --sapo-tau-{pos,neg}
# and --gmpo-log-clip-{low,high}; cfafdaec --gae-variant / --gae-lambd-mode /
# --gae-length-alpha / --gae-critic-lambd and the VAPO positive-example LM loss
# (--positive-example-*); 6442e19e --policy-objective sao_dis, the categorical
# value head (--value-loss-type classification, --value-*, --hl-gauss-sigma-ratio),
# --num-critic-epochs (alias --critic-updates-per-step), --critic-freeze-attention
# and the value head on language_model of multimodal bridge wrappers. The options
# live in the YetoAlgoConfig trait; the CLI names are the ports-v1 fork's, so
# yeto's argv does not change. The fork's dual_clipfrac metric (2be012421) is not
# ported: yeto reports yeto/dual_clipfrac itself (state_plugin).
# k31/b8 ca31b08f = 6442e19e + a test-only fix (tests/fast/ray/train/test_group.py
# matches the engine-side failure across the Ray traceback lines).
# k31/b8 efc83590 = ca31b08f + the workers_lost cell state, from Miles k31/b4-lost
# efc83590 (fast-forwarded onto k31/b8): when the liveness scan finds workers of a
# cell dead and tears the cell down, RayWorkerManager.describe_cells reports it
# "workers_lost" with the dead workers in "lost_workers" (None in every other
# state) instead of "stopped"; stop_cells acknowledges it, start_cells clears it,
# InferenceController.describe_cells passes both fields through. The ports-v1
# fork's 857fc9592 has the same state. The B8 features and workers_lost are
# registered on the branch head, which carries every commit above.
# MILES_K31_IMAGE ships this commit (the lineage head, MILES_NEXT_COMMIT), so
# every k31 batch above is in the pinned image.
MILES_K31_B8_COMMIT = "efc83590dcb8daf9b230c71b45ffb4f1e6b54cb3"
# k31/gpu-a-b8 fc15023c = efc83590 + the four fixes the first B300 runs needed:
# split_moe without moe_use_legacy_grouped_gemm, the mbridge linear_attn mixin
# forwarding keyword options, the optional q_lora atomic group for Kimi K3, and
# the DSA layer check against the model's own layer count. No new features.
MILES_K31_GPUA_COMMIT = "fc15023cb346d5205984c9e68957b673b00e4b9a"
# k31/full = MILES_K31_GPUA_COMMIT merged with k31/qat f3be3dc0 (--routed-expert-qat
# mxfp4: straight-through MXFP4 fake-quant of the trainable routed experts).
MILES_K31_FULL_COMMIT = "63834eaa53b94f449a0e75ed86d6b597528817a9"
# k31/full-stub ecb33cba = 63834eaa + the full-parameter traversals skip Megatron's
# is_stub_optimizer ChainedOptimizer children (SC2 EP2 distributed optimizer) and
# log the chained child layout once. No new features.
MILES_K31_FULL_STUB_COMMIT = "ecb33cbaa2b8eac25de24f443e3861113cbbd2ff"
# k31/full-b 83d5db6b = k31/full-offload 8b75ab19 (63834eaa + --optimizer-cpu-offload on
# the reset-optimizer reload path: the reset clears the HybridDeviceOptimizer's
# sub-optimizers too) merged with k31/full-stub ecb33cba. No new features.
MILES_K31_FULL_B_COMMIT = "83d5db6b60677e9c8372ee505bdd6bb5713047e7"
# k31/full-c 690ebe49 = k31/full-b + the trainable experts' masters dequantize on the
# experts' device at load (d9ab0957) + the CPU-offload reset reload test fix (d976b958).
# No new features.
MILES_K31_FULL_C_COMMIT = "690ebe4910337cf0a96542e9d057c1baaee5ae16"
# k31/full-d-clipfrac c351a33c = k31/full-c + the score-centering loss logs
# sc_is_clipped (tokens whose TIS/MIS weight truncates pi_train / pi_rollout).
MILES_K31_FULL_D_COMMIT = "c351a33c643326c478a9edaf9bae6c43299508fb"
# k31/integ-trainer 1f29e430 = c351a33c + the DSA indexer scored in query blocks
# (c5393bbe), the attention-residual rows mixed in checkpointed blocks (a0cef1d4), the
# first-stage expert grads under full recompute (2fc2b14a) and the finite score-centering
# loss at head mass >= 1 (0cb40f2f). It is the Miles of MILES_K31_FULL_IMAGE (2de5d796,
# read back from the image: git -C ~/miles rev-parse HEAD). No new features.
MILES_K31_FULL_E_COMMIT = "1f29e43043e4c9aace8287a445eca3851e8b52c7"
# sgl-project/sglang ``sglang-miles`` head when radixark/miles@9e4260d was
# committed (upstream's Dockerfile follows that branch unpinned).
SGLANG_PORTS_V1_UPSTREAM_COMMIT = "571212b636baca45e10fa3b4da11a289123f3235"
# yeto/lora-checksum (a1240c530 = 9f29303 + WeightChecker checksum covers LoRA adapter A/B);
# yeto/ports 9f29303: the ported agentenv/sglang patches (see sglang-patch-port.md).
# m3-qwen4exp-lora (4e4148f1b = a1240c530 + Qwen4ExpForConditionalGeneration LoRA hooks).
# n17-qwen3coder-single-call (2fa880182 = 4e4148f1b + qwen3_coder honors
# parallel_tool_calls=false under tool_choice=auto: at most one tool call).
SGLANG_PORTS_V1_REPOSITORY = "https://github.com/michaellchung/sglang"
SGLANG_PORTS_V1_COMMIT = "2fa880182eefdb31e64d8ea70317be27f9f32121"
# sgl-project/sglang branch ``sglang-miles-kimi-k31`` (the K3.1 lineage's SGLang;
# the radixark/miles:kimi-k31 image builds it in).
SGLANG_K31_REPOSITORY = "https://github.com/sgl-project/sglang"
SGLANG_K31_BRANCH = "sglang-miles-kimi-k31"
SGLANG_K31_COMMIT = "35708f76dfb720f831f7028ddb68250b6caa8f12"

# The sampler-side SGLang / vLLM pins (engine sampler, apart from the Miles image) live in
# ``yeto/rl/sampler_pins.py``: the sampler adapters read them without importing this adapter.

# Fork lineage on michaellchung/miles yeto/ports, oldest first (the ports-v1
# lineage, FROZEN: the DiLoCo runs keep this fork and image). Each entry is the
# previous one plus the change named here; the newest is MILES_PORTS_V1_COMMIT.
# Feature and GPU-evidence sets are derived from it through
# :data:`MILES_LINEAGES` / :func:`pin_has`; a commit that is not in any lineage
# carries nothing.
MILES_FORK_LINEAGE: tuple[str, ...] = (
    # 0af62f4d: LoRA bridge honours calculate_per_token_loss (g1c).
    "0af62f4d48ed6a5b185c257578d8f7e22312aa87",
    # 5c1b49eb: fast-forward of 0af62f4d with the 2b loss variants
    # (--policy-loss-variant, --sapo-tau-*, --gmpo-log-clip-*).
    "5c1b49ebccbc7508c1d9ef89eacc2db3e448b6ba",
    "2f23a0fca9b80f6a7300da401703c343014b03c0",  # + cell declaration (placement/worker files only)
    "fb04d6ffa30edc28c7ba0a2e88802a84bbbd28f9",  # + M5 (LoRA dp_invariant_state.py only)
    "e3a11ab38cbb7fd911b23fdd62a4eb6dfbb1c841",  # + M5 error text/docstring/tests
    # 1023269 = e3a11ab3 + Qwen3.8-Next LoRA layout + worker loss / bounded
    # weight-update group; no loss_hub file.
    "1023269412bf4e54a1d95c8d2deaee795871aa72",
    # c35702e = + 857fc9592 engine-failure propagation; no loss path.
    "c35702eefcf2862cee155e46870e6ad30568d2c6",
    # 8bc52237a = + raw-lora-disagg; no loss path.
    "8bc52237a1102306abd8f89a2ea2090aa2df6850",
    # efbbc63ea = + abort-discard-stats; rollout bookkeeping only.
    "efbbc63ea0e8c6dcbc200ad5df5bc56badfce7bf",
    # 2f7871fb2 = + inference_rollout_train.abort tally; rollout bookkeeping only.
    "2f7871fb2e80d37420b16ef81f001c7b0ba8e32e",
    # ddce20992 = + agentic-suspend; rollout scheduling only.
    "ddce20992c9ee82270a22e259e9e2bf55f3053d1",
    # 64b591a4b = + critic family + rl-algo-supplement metrics; new flags
    # default off, loss bitwise unchanged with defaults.
    "64b591a4bec1ffa37fb089d3e3b99c84773b1ffc",
    # 56068bd56 = + full-parameter shard state and transport; no loss path.
    "56068bd561df6658e929fe1b2ca051b5385d3d9b",
    # 95533a0cd / 5f7ac4e32 / 587bd258a = + LoRA actor opt-in to the full-parameter
    # transport, base-weight publish for a LoRA hybrid; no loss path.
    "95533a0cd2097175980afa4a9c43a817512ebca6",
    "5f7ac4e32ba3131ebf92a77bd6571c467bfa6220",
    "587bd258a350cc4757e3e774436eb48085d808cc",
    # 386ae3b07 = + SAO streaming primitives; no loss path.
    "386ae3b07138ccd45ce4a07fc0383587f7968a3a",
    # 17aeb5cb5 = + lazy imports for light actors, master port held; no loss path.
    "17aeb5cb51635b9ba6ef4ca10db95762695c7e03",
    # 44d41efd0 = + megatron_utils runtime patches moved out of the package init; no loss path.
    "44d41efd0a0428cb12b4ee34422de6f5e1ccb9d9",
    # 9462de532 = + FP8 rollout honors the FP8 ignore list, --rollout-hf-checkpoint; no loss path.
    "9462de532f747cf688e090bbaf6b24f80e378fe8",
    # ed5f64fd7 = + DP-invariant state export/restore carries the CPU-offload optimizer state; no loss path.
    "ed5f64fd7a183cccc8ab7ca3da90fba570108c2b",
    # 56190a5a8 = + full-parameter sync reads DP=1 distributed-optimizer masters,
    # incl. the HybridDeviceOptimizer's offloaded FP32 copies; no loss path.
    "56190a5a86d14f1eb96d592038d76a9ad8651b0c",
    # cc693c693 = + the optimizer-state proof skips the HybridDeviceOptimizer's
    # master_param mirror, which a global apply overwrites; no loss path.
    "cc693c6930be178f411f711f55c3d334a087c097",
    # a477bcbe0 = + FullParameterFragmentPlan.flat_ranges: a parameter above the
    # fragment bound spans several fragments; plans without ranges keep their hash.
    "a477bcbe04bc6278bfed017cf8931ae90bb8e3e3",
    # 0b64b40c1 = + rebase_full_parameter_scheduler_anchor after a peer snapshot
    # restore; no loss path.
    "0b64b40c15fa972d20bd1393aca6067dfd2492a4",
    # b13c170ce = + per-fragment full-parameter apply: a chunked apply skips
    # fragments the masters already hold.
    "b13c170ce7611104aee28e38ed01c188e95417a2",
    # 23d4de975 = + deferred full-parameter export with on-demand materialize,
    # faster finite/hash checks, hybrid R3 layer mapping.
    "23d4de975b18f76e35f2f3d1093ad774a3a68086",
    # aa6b887d8 = + push checks small chunks inline; no loss path.
    "aa6b887d8387d5c575f7485ad6008c0f503423a8",
    # d1616ba12 = + THD packs carry a per-sample seq_idx (total_tokens in
    # PackedSeqParams), so Mamba state restarts per sample.
    "d1616ba1203f7b70b372d60803c0f740561e5b38",
    # 8edea35a7 = + full-parameter chunks travel through Ray as numpy buffers;
    # same values, no loss path.
    "8edea35a76026d83e0a819038baf09d88bd448d9",
    # b7868074c = d1616ba12 + MilesNemotronHLanguageOnlyBridge: trains the language
    # model of the Nemotron-H VL checkpoint, vision frozen (not built);
    # no loss path. A side branch (not after 8edea35a7), listed before the DP
    # replica commit because it has none; merged at ebc6519af.
    "b7868074ca1e081581aba3290b48738734ab5600",
    # 63cb372b3 = 8edea35a7 + per-fragment full-parameter round boundary: lazy
    # export (no master read or hash at the cut), materialize returns fragment
    # descriptors, a commit writes only the parameters it received;
    # no loss path. A side branch, listed before the DP replica commit
    # because it has none; merged at 59aea07f9.
    "63cb372b3900e66f9ce1e90b3b5ddff5f29c7224",
    # ccd85ec73 = + unused test imports dropped; tests only.
    "ccd85ec73eec71720d28aa3a5e621a10798c79b1",
    # 921cde8c7 = 8edea35a7 + full-parameter data-parallel replicas (DP>1, EP
    # spanning DP): one owner rank per replica group exports, the replicas apply
    # the owner's bytes; no loss path.
    "921cde8c7fe7657471bb2be255a4af9bf3653c3d",
    # d78bf39e4 = + HybridDeviceOptimizer CPU FP32 masters in pinned memory
    # (cherry-pick of 2d3aaf949); same values, no loss path.
    "d78bf39e4d51db5ea7b241e996bcc37a0b854c55",
    # ebc6519af = merge of d78bf39e4 and b7868074c (yeto/closing).
    "ebc6519af7c19c0f4b3ba82f109fd19072647a8c",
    # 59aea07f9 = merge of ebc6519af and ccd85ec73 (yeto/closing).
    "59aea07f9a1e48b6fcb9fed03c70964a00275a40",
    # 76a84963c = 59aea07f9 + set MoE runtime flags copied onto the bridge
    # provider, router fusion refused with routing replay, --cuda-graph-impl
    # refused in bridge mode; no loss path. A side branch without
    # sharded masters, listed before them; merged at 7402a53d5.
    "76a84963cf48916c4768523408dcacfa7dae671e",
    # d50fb04dd = 59aea07f9 + sharded FP32 masters: full-parameter sync at DP>1
    # on Megatron's distributed optimizer; same synced values.
    "d50fb04dd2111f4a7c6022e82ef256a0dee0d696",
    # acb798c7b = + several distributed-optimizer instances refused.
    "acb798c7b74f32004ad7ba4c5c07a4e69a82911a",
    # 7402a53d5 = merge of acb798c7b and 76a84963c (yeto/eff).
    "7402a53d58dfcccd57ad6b59c0408127158f5704",
    # 102672760 = 7402a53d5 + verify_stored_fragment_chunks; prepare skips the
    # fragments the caller already verified (transport feature verified_apply);
    # same applied bytes, no loss path. A side branch without the
    # host-sync-free commits, listed before them; merged at b6662508e.
    "102672760a31c86697972a06b060c05d4efaeb60",
    # daeb98903 = 7402a53d5 merged with 102672760 (same tree as 102672760;
    # yeto/e3).
    "daeb98903779df2a470249d4142d97b610ad96c1",
    # 17841205b = + no host syncs in R3 replay copies, loss_function logging
    # tensors, THD cu_seqlens/max_seqlen and Mamba seq_idx; loss
    # values unchanged bit for bit.
    "17841205b19fde01279fd9585711cb610bad67a8",
    # 697fb2000 = + host-sync-free DeepEP _indices_to_multihot, a runtime patch
    # of Megatron a84b10547; no loss path.
    "697fb2000182955cbaa265345cd507924a1178a7",
    # b6662508e = merge of daeb98903 and 697fb2000 (yeto/e3).
    "b6662508e77e2e351cac668f392d1c747921a1c2",
    # 8d03f386a = 7402a53d5 + shape-free layout support;
    # canonical-order plans keep their hashes and bytes. A side branch without
    # the verified-apply commits above, listed after them; merged at 8faa7fccc.
    "8d03f386a72c82a07af60dab9dd13d44c0a0fc5b",
    # 3f6760f2b = + a gated linear_fc1 reports two TP blocks.
    "3f6760f2bd7885a71b1bdd30a70956da88f775c0",
    # 8faa7fccc = merge of b6662508e and 3f6760f2b (yeto/next).
    "8faa7fccc31625bfc4cc62ff3f34aed741ca62fe",
    # ddd416167 = b6662508e + commit chunk read-ahead (page mapping off the
    # copying thread) and the device-side model cast of sharded masters, a
    # sample of every write checked bit for bit against the host cast;
    # same committed bits, no loss path. A side branch without the
    # shape-free commits, listed after them; merged at 970807006.
    "ddd416167698741085b2544faf28004481be3991",
    # e9410717a = + the read-ahead reuses the two-thread hashing pool.
    "e9410717aa7a44bf830872c4403497c1e4daf522",
    # 970807006 = merge of 8faa7fccc and e9410717a (yeto/next).
    "970807006ed0b9a958a2167039e046cae52ffdd0",
    # 212da6791 = 7402a53d5 + session server route /v1/responses (Codex), recorded
    # through the chat-completions path (cherry-pick of f01bc24cd); no
    # training path. A side branch without the verified-apply / shape-free / sync-commit
    # commits, listed after them; merged at 28bc6cc24.
    "212da67914cf02f2fc2e0b7ee8f3455b79359ed8",
    # babe2adca = + one log line per failed session model call (cherry-pick of 1773afaa1).
    "babe2adca3aee101ae9960af693cf0fcda689d28",
    # e99e2a168 = + test only (Nemotron-3 Nano Codex trajectory; yeto/agentic-nemotron).
    "e99e2a168e3c53f6d930b6c08138702bda349618",
    # 28bc6cc24 = merge of 970807006 and e99e2a168 (yeto/next2).
    "28bc6cc245e6a45c8ca2d36b632efa479f4f7a14",
)

# First Miles commit whose full-parameter sync reads distributed-optimizer
# masters (and the HybridDeviceOptimizer's offloaded FP32 copies) at DP=1
# (fork branch yeto/distopt-fp on ed5f64fd7, in the image since
# 56190a5-2fa8801; the image pin is cc693c6-2fa8801).
MILES_DISTOPT_FULL_PARAMETER_COMMIT = "56190a5a86d14f1eb96d592038d76a9ad8651b0c"


# First Miles commit whose full-parameter sync supports data-parallel replicas
# (DP>1, EP spanning DP): one rank of each replica group owns and exports a
# parameter, the other ranks apply the owner's bytes (fork branch
# yeto/dp-sync on 8edea35a7, whose head d78bf39e4 also pins the
# HybridDeviceOptimizer CPU masters).  In the image since ebc6519-2fa8801
# (yeto/closing); a pin before it refuses a full-parameter sync at DP>1.
MILES_DP_REPLICA_FULL_PARAMETER_COMMIT = "921cde8c7fe7657471bb2be255a4af9bf3653c3d"


# First Miles commit whose full-parameter sync at DP>1 runs on Megatron's
# distributed optimizer (each rank keeps one FP32 range of each parameter):
# the replicas contribute their ranges to the owner's export, an apply writes
# the whole model and the local range; same layout and bytes as the replica
# path (fork branch yeto/dp-distopt on 59aea07f9).  In the image
# since 7402a53-2fa8801 (yeto/eff); with a pin before it
# ``--rl-dp-optimizer auto`` keeps the non-distributed optimizer at DP>1.
MILES_DP_SHARDED_MASTER_COMMIT = "d50fb04dd2111f4a7c6022e82ef256a0dee0d696"


class MilesLineage(NamedTuple):
    """One linear history of Miles commits yeto has driven.

    ``commits`` is oldest first (hand-linearized; a side branch is listed
    where the comments in :data:`MILES_FORK_LINEAGE` say). ``features`` maps a
    feature name to the first commit of this lineage carrying it; a commit
    has the feature from that entry on. Two lineages never share a commit.
    """

    name: str
    upstream_base: str
    commits: tuple[str, ...]
    features: Mapping[str, str]

    def index(self, commit: str) -> int | None:
        try:
            return self.commits.index(commit)
        except ValueError:
            return None

    def has(self, feature: str, commit: str) -> bool:
        first = self.features.get(feature)
        if first is None:
            return False
        position = self.index(commit)
        return position is not None and position >= self.commits.index(first)

    def from_commit(self, commit: str) -> frozenset[str]:
        position = self.index(commit)
        return frozenset() if position is None else frozenset(self.commits[position:])


# ports-v1 features: the first fork commit carrying each (the comments of
# MILES_FORK_LINEAGE name the change). Frozen with the lineage.
PORTS_V1_FEATURES: Mapping[str, str] = {
    "run_plugin": "0af62f4d48ed6a5b185c257578d8f7e22312aa87",
    "worker_dynamic_port_start": "0af62f4d48ed6a5b185c257578d8f7e22312aa87",
    "elastic_m1_m6": "0af62f4d48ed6a5b185c257578d8f7e22312aa87",
    "lora_per_token_loss": "0af62f4d48ed6a5b185c257578d8f7e22312aa87",
    "loss_variants": "5c1b49ebccbc7508c1d9ef89eacc2db3e448b6ba",
    "cell_declaration": "2f23a0fca9b80f6a7300da401703c343014b03c0",
    "lora_dp_invariant_state": "fb04d6ffa30edc28c7ba0a2e88802a84bbbd28f9",
    "qwen3_8_next_lora": "1023269412bf4e54a1d95c8d2deaee795871aa72",
    "bounded_weight_update_group": "1023269412bf4e54a1d95c8d2deaee795871aa72",
    "engine_failure_propagation": "c35702eefcf2862cee155e46870e6ad30568d2c6",
    "raw_lora_disagg": "8bc52237a1102306abd8f89a2ea2090aa2df6850",
    "abort_discard_stats": "efbbc63ea0e8c6dcbc200ad5df5bc56badfce7bf",
    "agentic_suspend": "ddce20992c9ee82270a22e259e9e2bf55f3053d1",
    "critic_algosup": "64b591a4bec1ffa37fb089d3e3b99c84773b1ffc",
    "full_parameter_transport": "56068bd561df6658e929fe1b2ca051b5385d3d9b",
    "lora_hybrid_full_parameter": "95533a0cd2097175980afa4a9c43a817512ebca6",
    "sao_streaming": "386ae3b07138ccd45ce4a07fc0383587f7968a3a",
    "light_actor_imports": "17aeb5cb51635b9ba6ef4ca10db95762695c7e03",
    "rollout_hf_checkpoint": "9462de532f747cf688e090bbaf6b24f80e378fe8",
    "distopt_full_parameter": "56190a5a86d14f1eb96d592038d76a9ad8651b0c",
    "dp_replica_full_parameter": "921cde8c7fe7657471bb2be255a4af9bf3653c3d",
    "dp_sharded_master": "d50fb04dd2111f4a7c6022e82ef256a0dee0d696",
    "verified_apply": "102672760a31c86697972a06b060c05d4efaeb60",
    "host_sync_free": "17841205b19fde01279fd9585711cb610bad67a8",
    "shape_free_layout": "8d03f386a72c82a07af60dab9dd13d44c0a0fc5b",
    "session_responses": "212da67914cf02f2fc2e0b7ee8f3455b79359ed8",
}

# k31 lineage: upstream main 778794a8 + the F1 fork. Every batch of the sync
# plan that lands appends its head here and registers what it brought back
# (B1 run_plugin is in F1; B2 full-parameter; B3/B5/B6/B7 in k31/b3-b7; B4
# elastic in k31/b4; B8 loss variants / GAE / critic family in k31/b8).
# TODO(F1): confirm the feature list against the F1 fork commits.
K31_FEATURES: Mapping[str, str] = {
    "run_plugin": MILES_K31_F1_COMMIT,
    "worker_dynamic_port_start": MILES_K31_F1_COMMIT,
    "light_actor_imports": MILES_K31_F1_COMMIT,
    "rollout_runtime_stats": MILES_K31_F1_COMMIT,
    "yeto_ext": MILES_K31_F1_COMMIT,
    "kimi_k31": MILES_K31_F1_COMMIT,
    # B2 (k31/b2): the full-parameter transport the synced K3.1 trainer islands
    # need (chunked_full_parameter imports miles.ray.full_parameter_transport).
    "full_parameter_transport": MILES_K31_B2_COMMIT,
    "full_parameter_rank_state": MILES_K31_B2_COMMIT,
    # The same batch ports the ports-v1 full-parameter commits after 56068bd56
    # (B2 bb31ae18 / cf5d6eac / 0be83bfa / a05d0f01; full_parameter_state.py,
    # full_parameter_replicas.py and full_parameter_sharded_masters.py differ
    # from the ports-v1 head 28bc6cc24 only by FullParameterRankState and the
    # typed config reads, and the fork's tests pass unchanged but for the fakes'
    # typed args): DP=1 distributed-optimizer masters incl. the
    # HybridDeviceOptimizer's offloaded copies (fork 56190a5a8, cc693c693; B2
    # bb31ae18, test_full_parameter_state.py::test_distributed_optimizer_*);
    # data-parallel replicas, EP spanning DP (fork 921cde8c7; B2 cf5d6eac,
    # test_full_parameter_dp_replicas.py); sharded FP32 masters on the
    # distributed optimizer at DP>1, several instances refused (fork d50fb04dd,
    # acb798c7b; B2 0be83bfa, test_full_parameter_dp_sharded_masters.py); an apply
    # skipping the prepare checks of fragments verified ahead (fork 102672760;
    # B2 0be83bfa, test_full_parameter_verified_apply.py); the shape-free layout
    # (fork 8d03f386a, 3f6760f2b; B2 a05d0f01, test_full_parameter_free_order.py).
    # Every k31 commit from B2 on carries them; B2 is the first lineage entry.
    "distopt_full_parameter": MILES_K31_B2_COMMIT,
    "dp_replica_full_parameter": MILES_K31_B2_COMMIT,
    "dp_sharded_master": MILES_K31_B2_COMMIT,
    "verified_apply": MILES_K31_B2_COMMIT,
    "shape_free_layout": MILES_K31_B2_COMMIT,
    # B3 (k31/b3-b7): RolloutEngineJoinError / ExternalFailureError; a cell keeps
    # its actors, the broadcast update fails as NonRetryableError.
    "engine_failure_isolation": MILES_K31_B3B7_COMMIT,
    # B5: abort() publishes the abort_discard tally, MetricGatherer the
    # rollout/over_sampling/* metrics, --agentic-suspend-between-turns /
    # --agentic-suspend-max-rounds suspend agent groups across the weight update.
    "abort_tally": MILES_K31_B3B7_COMMIT,
    "agentic_suspend": MILES_K31_B3B7_COMMIT,
    # B6: POST /sessions/{id}/v1/responses recorded through the chat path;
    # LoRA bridge provider.calculate_per_token_loss.
    "session_responses_api": MILES_K31_B3B7_COMMIT,
    "lora_per_token_loss": MILES_K31_B3B7_COMMIT,
    # B7: MoE runtime flags copied onto the bridge provider when set
    # (megatron_cli_defaults), DebugConfig.host_sync_free (--host-sync-free /
    # --no-host-sync-free), MilesNemotronHLanguageOnlyBridge, R3 replay streams
    # mapped by router layer numbers + --mamba-training-ssm-states-dtype on the bridge.
    "bridge_moe_runtime_flags": MILES_K31_B3B7_COMMIT,
    "host_sync_free_flag": MILES_K31_B3B7_COMMIT,
    "nemotron_h_vl_bridge": MILES_K31_B3B7_COMMIT,
    "replay_hybrid_layers": MILES_K31_B3B7_COMMIT,
    # B4 (k31/b4): --yeto-placement-map (placement_group.parse_placement_map,
    # slice_pg_info, standby view, declared rollout cells).
    "placement_map": MILES_K31_B4_COMMIT,
    # epoch-guarded start_cells / stop_cells, wait_cells_tracked,
    # get_membership_status / restore_membership_state, describe_cells.
    "membership_epoch": MILES_K31_B4_COMMIT,
    # --router miles cordons: /cordon_worker, /worker_inflight, drain_cells /
    # uncordon_cells, cordoned registration (admit_cells).
    "router_cordon": MILES_K31_B4_COMMIT,
    # placement_group.update_member_weights / commit_weight_version (the
    # full-fleet update_weights takes no members= any more).
    "member_publish": MILES_K31_B4_COMMIT,
    # miles.ray.trainer_rebuild.rebuild_training_models (not re-exported by
    # placement_group: that would be an import cycle).
    "trainer_rebuild": MILES_K31_B4_COMMIT,
    # B8 (k31/b8): --policy-loss-variant cispo / sapo / gmpo and their parameters.
    "loss_variants": MILES_K31_B8_COMMIT,
    # --gae-variant / --gae-lambd-mode / --positive-example-* (VAPO, CompactionRL).
    "gae_variants": MILES_K31_B8_COMMIT,
    # --policy-objective sao_dis, the categorical value head, --num-critic-epochs,
    # --critic-freeze-attention (algos/critic_fork.py).
    "critic_family": MILES_K31_B8_COMMIT,
    # k31/b4-lost efc83590 (on k31/b8): describe_cells state "workers_lost" with
    # "lost_workers" (the watchdog journals such a cell workers_lost, not no_workers).
    "workers_lost": MILES_K31_B8_COMMIT,
}

MILES_LINEAGES: Mapping[str, MilesLineage] = {
    "ports-v1": MilesLineage("ports-v1", MILES_PORTS_V1_UPSTREAM_COMMIT, MILES_FORK_LINEAGE, PORTS_V1_FEATURES),
    "k31": MilesLineage("k31", MILES_K31_UPSTREAM_COMMIT,
                        (MILES_K31_F1_COMMIT, MILES_K31_B2_COMMIT, MILES_K31_B3B7_COMMIT,
                         MILES_K31_B4_COMMIT, MILES_K31_B8_COMMIT, MILES_K31_GPUA_COMMIT,
                         MILES_K31_FULL_COMMIT, MILES_K31_FULL_STUB_COMMIT,
                         MILES_K31_FULL_B_COMMIT, MILES_K31_FULL_C_COMMIT,
                         MILES_K31_FULL_D_COMMIT, MILES_K31_FULL_E_COMMIT),
                        K31_FEATURES),
}
CURRENT_LINEAGE = "k31"
# MILES_K31_IMAGE (63ad0d2a) carries FULL_D; the full-form image names its own commit
# (IMAGE_MILES_COMMITS, forwarded as --miles-commit)
MILES_NEXT_COMMIT = MILES_K31_FULL_D_COMMIT
SGLANG_NEXT_REPOSITORY = SGLANG_K31_REPOSITORY
SGLANG_NEXT_UPSTREAM_COMMIT = SGLANG_K31_COMMIT  # the branch head is the pin: no fork on top
SGLANG_NEXT_COMMIT = SGLANG_K31_COMMIT


def lineage_of(commit: str | None) -> MilesLineage | None:
    """The lineage holding ``commit``; None for a commit yeto never drove."""
    if not commit:
        return None
    for lineage in MILES_LINEAGES.values():
        if lineage.index(commit) is not None:
            return lineage
    return None


def current_miles_commit(args: object = None) -> str:
    """The Miles commit a run is pinned to: ``--miles-commit`` when given, else
    :data:`MILES_NEXT_COMMIT`. The one source every feature/declaration check uses."""
    override = getattr(args, "miles_commit", None) if args is not None else None
    return str(override) if override else MILES_NEXT_COMMIT


def pin_has(feature: str, commit: str | None = None) -> bool:
    """Whether ``commit`` (default :func:`current_miles_commit`) carries
    ``feature``: its lineage registers the feature and the commit is at or
    after the first one carrying it. A commit in no lineage carries nothing;
    this never raises."""
    lineage = lineage_of(current_miles_commit() if commit is None else commit)
    return lineage is not None and lineage.has(feature, commit or current_miles_commit())


def miles_fork_lineage_from(commit: str) -> frozenset[str]:
    """``commit`` and every later commit of its lineage; empty for a commit in
    no lineage (it used to raise ``ValueError`` on the single ports-v1 tuple)."""
    lineage = lineage_of(commit)
    return frozenset() if lineage is None else lineage.from_commit(commit)


def commits_with(feature: str) -> frozenset[str]:
    """Every commit of every lineage that carries ``feature`` (empty when no
    lineage registers it)."""
    return frozenset().union(*(
        lineage.from_commit(lineage.features[feature])
        for lineage in MILES_LINEAGES.values() if feature in lineage.features))


# Critic family (--policy-objective sao_dis, the categorical value head,
# --num-critic-epochs, --critic-freeze-attention, the GAE variants), oldest
# first: the k31 commits from k31/b8 on, which carry it in the main lineage.
# The former chain (yeto-critic-c357 6e7365b6, yeto-critic-c357-tied 6574a9c8,
# local commits on c35702e that reached a run only through the island-setup
# critic overlay) went with that overlay; no lineage carries it.
CRITIC_FORK_LINEAGE: tuple[str, ...] = tuple(
    c for c in MILES_LINEAGES["k31"].commits if MILES_LINEAGES["k31"].has("critic_family", c))

# torch_memory_saver preload patch used by the SSH harness (was a separate pin
# in yeto/rl/ssh_harness.py).
TMS_PRELOAD_PATCH_REPOSITORY = "https://github.com/fzyzcjy/torch_memory_saver"
TMS_PRELOAD_PATCH_BASE_COMMIT = "fdb09ad342f8d150e11afe719f798453cef742ad"
TMS_PRELOAD_PATCH_COMMIT = "d7a3a51bce723a80736dabf233017b927de03df8"
TMS_PRELOAD_BASE_BINARY_SHA256 = (
    "c26551247616cabb0bb79dbd26abb61a688a4bb47c9b6a6f9873cfc98093640d"
)
MILES_NEXT_PINS = MilesRevisionPins(
    MILES_NEXT_REPOSITORY, MILES_NEXT_COMMIT, MILES_NEXT_FORMER_REPOSITORIES
)
MILES_PORTS_V1_PINS = MilesRevisionPins(
    MILES_NEXT_REPOSITORY, MILES_PORTS_V1_COMMIT, MILES_NEXT_FORMER_REPOSITORIES
)
# radixark/miles:dev multi-arch index (upstream docker/Dockerfile at
# radixark 9e4260d, sglang v0.5.20 base): the base MILES_NEXT_IMAGE extends
# (its linux/amd64 manifest).  Public.
MILES_NEXT_BASE_IMAGE = (
    "docker:docker.io/radixark/miles@sha256:"
    "90940828dcd4d54fd907ff668b43537cbd94778047580e4160d6560af548b74d"
)
# The base plus the pinned forks: MILES_NEXT_COMMIT at /root/miles and
# SGLANG_NEXT_COMMIT at /sgl-workspace/sglang, both the base's editable
# installs (pure-Python overlay, scripts/build_miles_ports_image.sh;
# docker/miles-ports/Dockerfile).  /opt/yeto/image-manifest.json records
# every SHA.  A private image needs SKYPILOT_DOCKER_{USERNAME,PASSWORD,SERVER}
# or --rl-image-private (yeto.launcher.registry_login_for; read:packages token).
# Tag 28bc6cc-2fa8801; Miles 28bc6cc24 = 970807006 merged with e99e2a168 (the
# session server records Codex Responses calls); build record
# tests/fixtures/openspec_evidence/ports-image-28bc6cc-2fa8801/. linux/amd64 only.
MILES_PORTS_V1_IMAGE = pinned_image(
    "yeto-miles-ports", "7a52de2e783c462961daa3e92af76da31154dee81668d5fd1e4bc8555cf8eab2")
# The k31 image: radixark/miles:kimi-k31 (CUDA 13.0.3, SGLang 35708f76) plus the
# yeto overlay layer with Miles k31/b8 efc83590 = MILES_NEXT_COMMIT
# (docker/miles-k31/build.sh). Registry: yeto.rl.image_registry (YETO_IMAGE_REGISTRY); tag
# k31-efc8359-35708f7-ybfa703b, built 2026-11-09 from yeto k31/integ bfa703bc
# (docker/ and this file; the commit that pins the digest is not in the image,
# as before). A last layer records the yeto commit in
# /opt/yeto/image/yeto-source.json and the label ai.yeto.yeto.commit.
# Previous: k31-4a839d8-35708f7 sha256:2447eaf6... (Miles F1 4a839d8a).
# Current: k31-83d5db6-35708f7-full-b-83d5db6, built 2026-11-10 from Miles k31/full-b
# 83d5db6b (MILES_NEXT_COMMIT; read back from the pushed image).
# Previous: k31/full-stub ecb33cba sha256:8863a8a7...; FULL k31/full 63834eaa
# sha256:5c91e1f5...; total efc83590 sha256:6a9512b8...
MILES_K31_IMAGE = pinned_image(
    "yeto-miles-k31", "63ad0d2a9e70a139221050f51214f073a5bc53f66c0c57090ae926012e439c04")
# The Kimi-K3.1 full-parameter (partial-experts) trainer image: the default
# --rl-image of --rl-model-recipe kimi_k31 --tuning full (the LoRA form and every
# other recipe keep MILES_K31_IMAGE). Same registry rule; the digest is fixed.
MILES_K31_FULL_IMAGE = pinned_image(
    "yeto-miles-k31", "2de5d796de19d38f4e3165efce2a9528a7f7d3e60e3cb32fe29fd00c6abf9b5b")
MILES_NEXT_IMAGE = MILES_K31_IMAGE
# image digest -> the Miles commit inside it, when it is not MILES_NEXT_COMMIT: the island
# checks its checkout against this commit (--miles-commit) instead of the default pin
IMAGE_MILES_COMMITS: dict[str, str] = {
    "2de5d796de19d38f4e3165efce2a9528a7f7d3e60e3cb32fe29fd00c6abf9b5b": MILES_K31_FULL_E_COMMIT,
}


def image_miles_commit(image_ref: str | None) -> str | None:
    """The Miles commit of a pinned ``--rl-image`` (``...@sha256:<digest>``) known to carry
    another commit than MILES_NEXT_COMMIT; None otherwise."""
    digest = str(image_ref or "").rpartition("@sha256:")[2]
    return IMAGE_MILES_COMMITS.get(digest)
MILES_NEXT_IMAGE_MANIFEST = "/opt/yeto/image-manifest.json"
# Nebius VM images whose /var/lib/docker already holds a docker image's
# layers (scripts/bake_nebius_image.sh; COLDSTART-PLAN.md #3), keyed by that
# image's digest, then region.  sky still runs `docker pull <digest>`, which
# is then a no-op.  Only an exact digest match is used: a stale entry for an
# older pin would just cost a full pull on top of a non-default base disk, so
# the launcher warns and keeps the stock image instead.  Base disk: Nebius
# public family ubuntu24.04-cuda13.0 (sky's own GPU default), so the NVIDIA
# driver and container toolkit are unchanged.
NEBIUS_BAKED_IMAGES: dict[str, dict[str, str]] = {
    # MILES_NEXT_IMAGE c35702e-4e4148f; baked 2026-10-05 (cs2-bake.log).
    "sha256:37ac689e29caeecf9faf8587a3ad58c154ecffc7798711d5bd59d792d002b9f9": {
        "eu-north1": "computeimage-e00xts577c333r08gv",
    },
}
# Where MILES_NEXT_IMAGE installed the SGLang fork (editable).
MILES_NEXT_IMAGE_SGLANG_ROOT = "/sgl-workspace/sglang"


def default_rl_image(*, recipe: str | None = None, tuning: str | None = None) -> str:
    """The digest-pinned ``--rl-image`` default.

    MILES_NEXT_IMAGE: upstream Miles' image with the pinned Miles/SGLang
    forks preinstalled, on the registry of :mod:`yeto.rl.image_registry`
    (``YETO_IMAGE_REGISTRY``, default ``ghcr.io/agentenv``; the digest is fixed).
    A registry login is injected only via SKYPILOT_DOCKER_* or
    --rl-image-private; see yeto.launcher.registry_login_for. The packages
    are private until they are opened: pass credentials as docs/CLOUDS.md
    ("Default container images") describes. ``recipe="kimi_k31"`` with
    ``tuning="full"`` selects :data:`MILES_K31_FULL_IMAGE`.
    """

    if recipe == "kimi_k31" and tuning == "full":
        return MILES_K31_FULL_IMAGE
    return MILES_NEXT_IMAGE
