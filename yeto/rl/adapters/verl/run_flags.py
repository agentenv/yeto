"""The verl island entry's flags (``python3 -m yeto.rl.adapters.verl.island_entry``).

The island parses the one island flag schema (``yeto.rl.engine.run_flags``, layer
``island``: the neutral flags and the default backend's, which is what the launcher
emits for every island) plus the verl-only flags below, defined here once. A flag in
neither is refused at start (no ``parse_known_args``: a misspelled flag never runs as
its default).

Where the verl island differs from the schema it says so here:

* :data:`ISLAND_DEFAULTS`: defaults the verl island keeps (it had them before it read
  the schema; a different default would change what an island runs when the launcher
  leaves the flag out);
* :data:`ISLAND_OPTIONAL`: flags the schema requires on an island and the verl island
  does not (it runs without them); no flag it does not read is required either;
* :data:`READ`: the island flags the verl island reads. A schema flag outside it (a
  Miles-only option the launcher sends anyway) is accepted, unused and recorded
  (``unread_flags``); the Miles agentic and unmapped options among them are refused
  (``run_config_rules.refuse_agentic_argv``, ``train_options.refuse_unmapped_argv``).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from typing import Any

from yeto.exact_argparse import ExactArgumentParser
from yeto.rl.engine.run_flags import ISLAND, add_rl_flags, island_flag

# --rl-release-mode (buffer_ports module doc): a synced group's publisher releases the
# merged version (the default); an island fleet's trainer releases its own local versions
MERGED_RELEASE = "merged"
PER_LEARNER_RELEASE = "per_learner"
RELEASE_MODES = (MERGED_RELEASE, PER_LEARNER_RELEASE)

ISLAND_FLAGS = (
    # FSDP2 host-memory offload (e.g. 64 GiB Ascend cards with 30B full parameters)
    island_flag("--rl-fsdp-offload", choices=("none", "param", "optimizer", "both"), default="none"),
    # a memory / MFU measurement run on a buffer trainer writes no policy
    # release (one 120B release is about 225 GB and a dozen minutes per round)
    island_flag("--rl-measure-no-release", action="store_true"),
    # encode, digest and upload each staged (delta) release on a worker thread
    # while the next rounds train; needs --rl-release-delta-full-every
    island_flag("--rl-release-background", action="store_true"),
    # a measurement run also skips the full policy hash of the first and the last
    # round (about 480 GB of FP32 masters, ~30 min each on the 120B island); needs
    # --rl-measure-no-release and --rl-verify-every (train_options.verify_plan)
    island_flag("--rl-measure-no-verify", action="store_true"),
    # release only these policy versions (comma-separated), e.g. "2": a
    # multi-step run trains on the samples it has and releases once for the samplers
    island_flag("--rl-release-only-versions", default=None,
                type=lambda text: [int(v) for v in text.split(",") if v.strip()]),
    # a buffer trainer of an island fleet releases its own local versions into its
    # sub-store trainer<k> (buffer_ports module doc, per-learner releases); merged:
    # a synced group's learner 0 releases the merged version (the default)
    island_flag("--rl-release-mode", choices=RELEASE_MODES, default=MERGED_RELEASE),
    # a non-relay buffer trainer's prompt row width: the widest first prompt its
    # sampler pools write (agentic.max_prompt_len); unset: 512, or the sequence
    # budget on a per-learner trainer (island_entry.island_prompt_width)
    island_flag("--rl-buffer-max-prompt-len", type=int, default=None, metavar="TOKENS"),
    # precision-aware optimizer on the card, bf16 Adam moments, FP32 master
    island_flag("--rl-megatron-bf16-moments", action="store_true"),
    # uneven pipeline stages and full-layer recompute (megatron_engine.MegatronShape)
    island_flag("--rl-megatron-first-stage-layers", type=int, default=None),
    island_flag("--rl-megatron-last-stage-layers", type=int, default=None),
    island_flag("--rl-megatron-recompute-full-layers", type=int, default=None),
    # verl's fused log-prob kernels. Off unless asked for: on the 120B model they
    # are 47.8% of the activation per token, and they do not yet run on Ascend.
    island_flag("--rl-megatron-fused-logprob-kernels", action="store_true"),
    # the SFT recipe's communication overlap (args.json: overlap_grad_reduce,
    # overlap_param_gather, tp_comm_overlap all true); verl's DDP default is off
    island_flag("--rl-megatron-comm-overlap", action="store_true"),
    # verl's old-log-prob pass computes the token entropy only for the
    # "actor/entropy" metric; it is that pass's memory peak (megatron_engine.MegatronShape)
    island_flag("--rl-no-old-logprob-entropy", action="store_true"),
    # skip the old-log-prob recompute when old == current (onpolicy_old_logprob;
    # refused unless one optimizer step per round). measure = keep it, log the difference
    island_flag("--rl-onpolicy-old-logprob", choices=("off", "on", "measure"), default=None),
)

ISLAND_DEFAULTS: dict[str, Any] = {
    "lora_r": 32, "lora_targets": "all-linear", "actor_num_nodes": 1,
    # unset means "not given" on the verl island (run_config_rules, sample_roles, round cuts)
    "sync_preset": None, "rl_model_recipe": None, "rl_sample_role": None, "rl_resume_state_dir": None,
    "rl_placement": None, "sglang_mem_fraction_static": None, "rl_lr_schedule": None,
}
ISLAND_OPTIONAL = frozenset({"completed_groups_path", "source_sha256", "reward_sha256", "actor_num_nodes",
                             "actor_num_gpus_per_node"})

# Generated once from the hand-written verl island parser (the dests it defined).
READ = frozenset({
    "rl_expected_algorithm_sha256",  # checked at start (island_entry.verify_algorithm_hash)
    "actor_num_gpus_per_node", "actor_num_nodes", "apply_chat_template_kwargs", "bcast_delta",
    "completed_groups_path", "context_parallel", "data", "data_revision", "eval_data", "eval_data_sha256",
    "eval_dataset_name", "eval_interval", "eval_max_context_len", "eval_max_prompt_len",
    "eval_max_response_len", "eval_samples_per_prompt", "eval_temperature", "eval_top_p", "event_tape",
    "expert_full_count", "expert_parallel", "fragments", "full_parameter_rounds", "global_rounds",
    "groups_per_round", "init_mode", "initial_adapter", "initial_adapter_sha256", "inner_lr",
    "learner_generation", "learner_id", "local_horizon", "local_init_fallback", "lora_r", "lora_targets",
    "model", "model_revision", "num_learners", "parameter_layout_sha256", "parameter_mode", "pipeline",
    "pipeline_parallel", "reward_function", "reward_sha256", "rl_activation_recompute", "rl_agentic_harbor",
    "sync_max_learners",
    "rl_algorithm_spec", "rl_allow_pipeline_parallel_sync", "rl_ascend_plugin_recipe",
    "rl_ascend_plugin_stack", "rl_buffer_shards", "rl_cut_auto_mtbf_s", "rl_cut_every", "rl_cut_keep",
    "rl_cut_max_interval_s", "rl_data_dispense", "rl_data_position_store", "rl_data_slot", "rl_debug_dump",
    "rl_dispense_addr", "rl_dispense_by_label", "rl_dispense_labels", "rl_dispense_mix",
    "rl_dispense_prefetch_blocks", "rl_echo_events", "rl_expert_selection", "rl_export_hf_rounds",
    "rl_export_hf_store", "rl_fsdp_offload", "rl_heartbeat_interval", "rl_island_cut_store",
    "rl_island_gpus_per_node", "rl_island_pool", "rl_label_column", "rl_lr_schedule", "rl_max_policy_age",
    "rl_max_tokens_per_gpu", "rl_measure_no_release", "rl_measure_no_verify", "rl_megatron_bf16_moments",
    "rl_megatron_comm_overlap", "rl_megatron_first_stage_layers", "rl_megatron_fused_logprob_kernels",
    "rl_megatron_last_stage_layers", "rl_megatron_recompute_full_layers", "rl_model_recipe",
    "rl_no_grad_nan_check", "rl_no_old_logprob_entropy", "rl_observe_timeline", "rl_onpolicy_old_logprob",
    "rl_optimizer_offload_fraction", "rl_peer_recover", "rl_peer_recovery_donor_write",
    "rl_peer_recovery_manifest_wait_s", "rl_peer_recovery_staging", "rl_peer_recovery_store",
    "rl_peer_recovery_ttl_s", "rl_placement", "rl_prompt_column", "rl_release_background",
    "rl_release_delta_full_every", "rl_release_every", "rl_release_mode", "rl_release_only_versions",
    "rl_buffer_max_prompt_len", "rl_resource_sample_interval",
    "rl_resume_allow_config_change", "rl_resume_state_dir", "rl_resume_store", "rl_rollout_ahead",
    "rl_rollout_quant_tier", "rl_round_cuts", "rl_sample_lanes", "rl_sample_relay", "rl_sample_relay_prefetch",
    "rl_sample_role", "rl_sample_store", "rl_sample_wait_s", "rl_sampler_data_slot",
    "rl_sampler_max_backlog_rounds", "rl_sampler_min_round_seconds", "rl_sampler_node",
    "rl_single_island_no_sync", "rl_spot_subscribers", "rl_stop_after_rounds", "rl_sync_island_lease_s",
    "rl_sync_matrix_merge", "rl_syncer_epoch", "rl_verify_every", "rl_verify_every_round",
    "rollout_max_response_len", "rollout_num_gpus", "rollout_num_gpus_per_engine", "samples_per_group", "seed",
    "seq_len", "sglang_mem_fraction_static", "source_sha256", "sweep_expert_token_weights", "sweep_final_cut",
    "sweep_fragment_max_bytes", "sweep_max_local_rounds_ahead", "sweep_overlap", "sweep_push_dtype",
    "sweep_push_int4_block", "sweep_push_int4_error_feedback", "sweep_push_int4_stochastic",
    "sweep_push_int8_block", "sweep_push_no_tensor_blocks", "sweep_wait_timeout_s", "sync_preset", "syncer",
    "syncer_profile_sha256", "tensor_parallel", "total_fragment_steps", "trust_remote_code",
    "use_rollout_routing_replay", "wan_stall_timeout", "wan_stream_schedule", "wan_streams",
})


def build_parser() -> argparse.ArgumentParser:
    """The verl island's parser (module doc). Abbreviations are off: an option is spelled
    out, or it is refused."""
    from yeto.wandb_logger import add_arguments as add_wandb_arguments

    parser = ExactArgumentParser(prog="yeto.rl.adapters.verl.island_entry")
    add_rl_flags(parser, ISLAND)  # yeto/rl/engine/run_flags.py
    for flag in ISLAND_FLAGS:
        options, kwargs = flag.argparse_call(ISLAND)
        parser.add_argument(*options, **kwargs)
    add_wandb_arguments(parser)  # the launcher forwards --wandb* to every island
    parser.set_defaults(**ISLAND_DEFAULTS)
    for action in parser._actions:
        # a flag this island does not read is never required here (the Miles island's are)
        if action.dest in ISLAND_OPTIONAL or action.dest not in READ:
            action.required = False
    return parser


def unread_flags(parser: argparse.ArgumentParser, argv: Sequence[str]) -> list[str]:
    """The tokens of ``argv``'s options this island does not read (each option with its
    value), in argv order: what ``parse_known_args`` used to leave over."""
    out: list[str] = []
    tokens = list(argv)
    i = 0
    while i < len(tokens):
        token = tokens[i]
        i += 1
        if not token.startswith("--"):
            continue
        option, has_value, _ = token.partition("=")
        action = parser._option_string_actions.get(option)
        if action is None or action.dest in READ:
            continue
        out.append(token)
        if not has_value and action.nargs != 0:
            out += tokens[i:i + 1]
            i += 1
    return out


def chat_template_kwargs(value: Any) -> dict:
    """``--apply-chat-template-kwargs``: the schema parses it as JSON (a dict); a string
    (a plan, or a caller that did not parse it) is read the same way; unset: {}."""
    if not value:
        return {}
    return dict(value) if isinstance(value, Mapping) else json.loads(value)
