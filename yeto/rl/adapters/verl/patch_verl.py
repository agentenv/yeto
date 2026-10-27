"""Local patch applied to the verl source at image build time (rl-verl-backend D5).

Not pushed to the verl fork: one hook line after verl's ``add_lora`` in the
vLLM worker extension calls yeto's read-back (``vllm_readback.after_add_lora``)
so the inference side reports a checksum of the adapter it registered.  The
hook is inert unless ``YETO_VERL_READBACK_DIR`` is set and never raises into
vLLM.  The anchor must occur exactly once, otherwise the build fails.

Second hook (agentic-rollout-utilization 6.4b, also local, not pushed): inside
``FullyAsyncLLMServerClient.generate``'s resume loop, every generate call
appends ``[global_steps, new token count]`` to the trajectory's
``extra_fields["yeto_resume_calls"]`` (verl itself keeps only min/max), so yeto
rebuilds per-token version segments (``fully_async_round``).  The agent loop
copies every extra field into ``non_tensor_batch``; nothing reads it but yeto.

Third and fourth patches (S19 async5 fix, local, not pushed): LoRA over the
disaggregated checkpoint engine.  The sender line in ``engine_workers.py`` and
the receiver call in ``checkpoint_engine/base.py`` are *replaced* so the PEFT
config and phase markers travel in the tensor stream (``lora_wire``); the
rollout then reaches ``add_lora`` (and the read-back hook) as in the colocated
path.

Fifth patch (S27 G-DC, local, not pushed; ``--profile megatron`` only): verl's THD
packing (``verl/models/mcore/util.py``) builds Megatron's ``PackedSeqParams``
without ``total_tokens``, so Megatron never derives ``seq_idx`` and a Mamba
layer runs the whole pack as one sequence: the conv and SSM state of one
sample flow into the next (S27 G-AL found the same in Miles, GAL-REPORT.md:
train/rollout log-prob gap 0.0885 -> 0.0179 once ``seq_idx`` is set). The
patch passes the global pack length at every context-parallel size (S28 CP2;
it used to pass it only at CP=1).

Last patch (S28 FCE, local, not pushed; ``--profile megatron`` only): with
``use_fused_kernels`` on, verl's Megatron engine calls yeto's chunked vocab-parallel
log-probs (``chunked_logprob``) instead of its Triton kernel, and Megatron's HybridModel
gets the ``output_processor`` hook verl's fused forward needs.

Profiles: ``fsdp`` (default; the image ``verl-build``) applies patches 1-4 and,
since S27 G-EK, the sixth (zmq SUB, below: the FSDP2 separate mode and the LoRA
fully_async path use the same NCCL checkpoint engine); ``megatron`` (the image ``verl-megatron-build``) applies 1, 2, 5 and the last (S28 FCE) -- the
Megatron island trains full parameters, so the LoRA wire patches 3 and 4
(which replace the checkpoint-engine sender) are left out.

Usage: python patch_verl.py <verl source root> [--profile fsdp|megatron]
"""

import sys
from pathlib import Path

ANCHOR = "            self.add_lora(lora_request)\n"
HOOK = (
    "            try:  # yeto rl-verl-backend D5: registered-adapter read-back\n"
    "                from yeto.rl.adapters.verl.vllm_readback import after_add_lora as _yeto_after_add_lora\n"
    "            except ImportError:\n"
    "                _yeto_after_add_lora = None\n"
    "            if _yeto_after_add_lora is not None:\n"
    "                _yeto_after_add_lora(self, VLLM_LORA_INT_ID)\n"
)
TARGET = "verl/workers/rollout/vllm_rollout/utils.py"

CALLS_ANCHOR = '            global_steps = output.extra_fields.get("global_steps", None)\n'
CALLS_HOOK = (
    "            # yeto agentic-rollout-utilization 6.4b: per-call version + new token count\n"
    "            final_output.extra_fields.setdefault(\"yeto_resume_calls\", []).append(\n"
    "                [global_steps, len(output.token_ids)])\n"
)
CALLS_TARGET = "verl/workers/rollout/llm_server.py"

SEND_TARGET = "verl/workers/engine_workers.py"
SEND_ANCHOR = "            per_tensor_param, _ = self.actor.engine.get_per_tensor_param()\n"
SEND_HOOK = (
    "            # yeto S19 6.4b: LoRA config + phases in-band (lora_wire)\n"
    "            from yeto.rl.adapters.verl.lora_wire import sender_stream as _yeto_sender_stream\n"
    "            per_tensor_param = _yeto_sender_stream(self)\n"
)
RECV_TARGET = "verl/checkpoint_engine/base.py"
RECV_ANCHOR = (
    "        weights = self.checkpoint_engine.receive_weights(global_steps=global_steps)\n"
    "        await self.server_adapter.update_weights(\n"
    "            weights,\n"
    "            global_steps=global_steps,\n"
    "            wire_format=getattr(self.checkpoint_engine, \"wire_format\", \"named_tensors\"),\n"
    "        )\n"
)
RECV_HOOK = (
    "        # yeto S19 6.4b: split the stream into LoRA phases (lora_wire)\n"
    "        from yeto.rl.adapters.verl.lora_wire import receive_phases as _yeto_receive_phases\n"
    "        weights = self.checkpoint_engine.receive_weights(global_steps=global_steps)\n"
    "        async for _yeto_sub, _yeto_kw in _yeto_receive_phases(weights):\n"
    "            await self.server_adapter.update_weights(\n"
    "                _yeto_sub,\n"
    "                global_steps=global_steps,\n"
    "                wire_format=getattr(self.checkpoint_engine, \"wire_format\", \"named_tensors\"),\n"
    "                **_yeto_kw,\n"
    "            )\n"
)

SEQ_IDX_TARGET = "verl/models/mcore/util.py"
SEQ_IDX_HELPER_ANCHOR = "def _compute_fp8_thd_align_size("
SEQ_IDX_HELPER = (
    "def _yeto_seq_idx_args(cp_size, seqlens_in_batch_padded_cpu):\n"
    "    \"\"\"yeto S27 G-DC, S28 CP2: ``total_tokens`` of the *global* THD pack, so\n"
    "    ``PackedSeqParams`` derives ``seq_idx`` and a Mamba layer restarts its conv and\n"
    "    SSM state at every packed sample.\n"
    "\n"
    "    The pack length stays the global one at every ``cp_size``. Megatron-core's Mamba\n"
    "    context parallel is all-to-all, not a sequence split: ``pre_conv_ssm``\n"
    "    (``megatron/core/ssm/mamba_context_parallel.py``) gathers the whole global\n"
    "    sequence onto every CP rank and shards the heads instead, so the convolution and\n"
    "    the SSM scan see every token of the pack and ``seq_idx`` has to span it. verl\n"
    "    already passes the global ``cu_seqlens_q_padded``, and\n"
    "    ``sum(seqlens_in_batch_padded_cpu) == cu_seqlens_padded[-1]`` on both preprocess\n"
    "    paths (every pad bump raises the last entry of both).\n"
    "\n"
    "    S28 CP2 measured this on CPU (``s1-runs/s28-g120/cp_seqidx_check.py``, 2 and 4\n"
    "    gloo ranks, megatron's own ``MambaContextParallel``): with the global pack length\n"
    "    the CP>1 output is bit-exact against the CP=1 reference; without it (the earlier\n"
    "    ``cp_size != 1`` bail-out) the relative error is 0.377, because one sample's state\n"
    "    flows into the next.\n"
    "    \"\"\"\n"
    "    del cp_size  # the pack length is global at every CP size; see the docstring\n"
    "    if not _packed_seq_params_supports(\"total_tokens\"):\n"
    "        return {}\n"
    "    return {\"total_tokens\": int(sum(seqlens_in_batch_padded_cpu))}\n"
    "\n"
    "\n"
)
SEQ_IDX_PACKED_ANCHOR = (
    "        cu_seqlens_kv_padded=cu_seqlens_padded,\n"
    "    )\n"
    "    if pre_process:\n"
    "        return input_ids_rmpad.unsqueeze(0), packed_seq_params\n"
)
SEQ_IDX_PACKED_HOOK = (
    "        cu_seqlens_kv_padded=cu_seqlens_padded,\n"
    "        **_yeto_seq_idx_args(cp_size, seqlens_in_batch_padded_cpu),  # yeto S27 G-DC\n"
    "    )\n"
    "    if pre_process:\n"
    "        return input_ids_rmpad.unsqueeze(0), packed_seq_params\n"
)
SEQ_IDX_ENGINE_ANCHOR = "        **extra_packed_args,\n    )\n"
SEQ_IDX_ENGINE_HOOK = (
    "        **extra_packed_args,\n"
    "        **_yeto_seq_idx_args(cp_size, seqlens_in_batch_padded_cpu),  # yeto S27 G-DC\n"
    "    )\n"
)

# Sixth patch (S27 G-DC, megatron profile): the NCCL checkpoint engine sends each bucket's
# metadata over zmq PUB/SUB, and every update made each consumer open a NEW SUB socket and
# connect it just before the first bucket. A SUB socket that is not subscribed yet when the
# publisher sends drops the message ("slow joiner"): s27-dc-smoke3's second update hung with
# one consumer waiting in recv_string and the rest in the NCCL broadcast. The consumer now
# keeps one SUB socket per publisher address, and waits once after the first connect.
ZMQ_TARGET = "verl/checkpoint_engine/nccl_checkpoint_engine.py"
ZMQ_REUSE_ANCHOR = (
    "        assert not self.is_master, \"Master process should not connect to other processes.\"\n"
    "        context = zmq.Context()\n"
)
ZMQ_REUSE_HOOK = (
    "        assert not self.is_master, \"Master process should not connect to other processes.\"\n"
    "        # yeto S27 G-DC: one SUB socket per publisher (zmq slow joiner, see patch_verl)\n"
    "        _yeto_key = (metadata.zmq_ip, metadata.zmq_port)\n"
    "        if getattr(self, \"_yeto_sub_key\", None) == _yeto_key and getattr(self, \"socket\", None) is not None:\n"
    "            return\n"
    "        self._yeto_sub_key = _yeto_key\n"
    "        context = zmq.Context()\n"
)
ZMQ_WAIT_ANCHOR = "        self.socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)\n"
ZMQ_WAIT_HOOK = "        time.sleep(1.0)  # yeto S27 G-DC: let the subscription reach the publisher\n"
ZMQ_SUB = ((ZMQ_TARGET, ZMQ_REUSE_ANCHOR, ZMQ_REUSE_HOOK, "replace"),
           (ZMQ_TARGET, ZMQ_WAIT_ANCHOR, ZMQ_WAIT_HOOK, "after"))

# Seventh patch (S27 G-EQ2, local, not pushed; every profile): verl picks the HF auto
# class by walking the checkpoint's ``auto_map`` and taking the FIRST key whose value
# mentions ``architectures[0]``. Nemotron-3 Nano's auto_map is
#   {"AutoConfig": ..., "AutoModel": "modeling_nemotron_h.NemotronHForCausalLM",
#    "AutoModelForCausalLM": "modeling_nemotron_h.NemotronHForCausalLM"}
# so the first match is "AutoModel" and verl builds NemotronHModel -- the base model.
# The island then dies twice over: NemotronHModel refuses gradient checkpointing, and
# its forward returns BaseModelOutputWithPast, so verl's
# ``prepare_model_outputs`` raises "'BaseModelOutputWithPast' object has no attribute
# 'logits'" on the first training step (npu1, 19:26Z). The patch prefers the auto_map
# entry whose key matches the architecture suffix (``...ForCausalLM`` ->
# "AutoModelForCausalLM") and only then falls back to the first match.
AUTO_CLASS_TARGET = "verl/utils/model.py"
AUTO_CLASS_ANCHOR = (
    '        auto_class = next(k for k, v in hf_config.auto_map.items() '
    'if hf_config.architectures[0] in v)\n'
)
AUTO_CLASS_HOOK = (
    "        # yeto S27 G-EQ2: prefer the auto_map key that matches the architecture's\n"
    "        # task suffix; a first-match walk picks AutoModel for Nemotron-H (base model).\n"
    "        _yeto_matches = [k for k, v in hf_config.auto_map.items()\n"
    "                         if hf_config.architectures[0] in v]\n"
    "        _yeto_task = \"AutoModel\" + hf_config.architectures[0].split(\"For\", 1)[1] \\\n"
    "            if \"For\" in hf_config.architectures[0] else None\n"
    "        auto_class = (\"AutoModelFor\" + hf_config.architectures[0].split(\"For\", 1)[1]\n"
    "                      if _yeto_task and (\"AutoModelFor\" + hf_config.architectures[0].split(\"For\", 1)[1])\n"
    "                      in _yeto_matches else _yeto_matches[0])\n"
)
AUTO_CLASS = (AUTO_CLASS_TARGET, AUTO_CLASS_ANCHOR, AUTO_CLASS_HOOK, "replace")

# Eighth patch (S27 G-EZ2, local, not pushed; fsdp profile): one host copy of the
# weights, not one per rank. verl's FSDP engine builds the model on every rank with
# ``from_pretrained`` inside ``accelerate.init_empty_weights`` on every rank but the
# mesh's coordinate 0 (``fsdp_utils.get_init_weight_context_manager``), then
# ``fsdp2_load_full_state_dict`` broadcasts rank 0's full state dict. With
# transformers 5.x ``from_pretrained`` still reads the weights into host memory under
# that context, so every rank holds a whole FP32 model: npu1 n2/n3, Nemotron 120B on
# 16 ranks, the host's available memory fell from 1432 GiB to 125 GiB at 12 % of
# loading (G-EQ3, Nano 31.6B on 8 ranks: 993 GiB peak on n1). The hook builds the
# meta-context ranks from the config on the meta device instead: same class, same
# dtype (``from_config`` defaults to ``config.dtype`` -- bf16 here -- so the dtype is
# passed), no weights read. Rank 0 still loads, and the broadcast fills parameters
# and persistent buffers as before. ``tie_word_embeddings`` models keep verl's path
# (verl itself turns the meta init off for them).
META_INIT_TARGET = "verl/workers/engine/fsdp/transformer_impl.py"
META_INIT_ANCHOR = (
    "                module = auto_class.from_pretrained(\n"
    "                    pretrained_model_name_or_path=self.model_config.local_path,\n"
    "                    torch_dtype=torch_dtype,\n"
    "                    config=self.model_config.hf_config,\n"
    "                    trust_remote_code=self.model_config.trust_remote_code,\n"
    "                )\n"
)
META_INIT_HOOK = (
    "                # yeto S27 G-EZ2: a meta-context rank builds from the config (no weights read);\n"
    "                # under init_empty_weights transformers 5.x from_pretrained still loads them.\n"
    "                if getattr(init_context, \"__name__\", \"\") == \"init_empty_weights\":\n"
    "                    with torch.device(\"meta\"):\n"
    "                        module = auto_class.from_config(\n"
    "                            self.model_config.hf_config,\n"
    "                            torch_dtype=torch_dtype,\n"
    "                            trust_remote_code=self.model_config.trust_remote_code,\n"
    "                        )\n"
    "                else:\n"
    "                    module = auto_class.from_pretrained(\n"
    "                        pretrained_model_name_or_path=self.model_config.local_path,\n"
    "                        torch_dtype=torch_dtype,\n"
    "                        config=self.model_config.hf_config,\n"
    "                        trust_remote_code=self.model_config.trust_remote_code,\n"
    "                    )\n"
)
META_INIT = (META_INIT_TARGET, META_INIT_ANCHOR, META_INIT_HOOK, "replace")

# Ninth patch (S27 G-EZ2, local, not pushed; fsdp profile; opt-in by environment): rank 0
# reads the weights in the checkpoint's own dtype, and the FSDP2 CPU offload can keep the
# offloaded state pageable. After the eighth patch only rank 0 reads weights, but it reads
# them as a whole FP32 model (``model_dtype=fp32``): on npu1 n2 its process held 864 GiB
# after loading the 120B text model, and with the eight ranks' shards the node fell to
# 13 % available memory. With ``YETO_VERL_RANK0_CHECKPOINT_DTYPE=1`` rank 0 loads with
# ``torch_dtype="auto"`` (the checkpoint's bf16; ``_keep_in_fp32_modules_strict`` tensors
# such as the router bias stay f32) only as the broadcast source, builds its training
# module on the meta device like every other rank, and ``fsdp2_load_full_state_dict``
# materializes rank 0 with ``to_empty`` when its module is on meta. torch's
# ``_broadcast_state_dict`` sends the source dtype and ``load_state_dict`` copies it into
# the FP32 shards, so the masters are the bf16 values exactly. Exact only when every
# checkpoint tensor is stored in the config dtype or kept f32 by the model (the 120B
# text checkpoint: 41603 BF16 + 40 F32 router biases, preflight); a model with
# non-persistent buffers is refused (rank 0's would be uninitialized).
# ``YETO_VERL_OFFLOAD_PIN_MEMORY=0`` builds ``CPUOffloadPolicy(pin_memory=False)``.
RANK0_SOURCE_TARGET = META_INIT_TARGET
RANK0_SOURCE_ANCHOR = (
    "                else:\n"
    "                    module = auto_class.from_pretrained(\n"
    "                        pretrained_model_name_or_path=self.model_config.local_path,\n"
    "                        torch_dtype=torch_dtype,\n"
    "                        config=self.model_config.hf_config,\n"
    "                        trust_remote_code=self.model_config.trust_remote_code,\n"
    "                    )\n"
)
RANK0_SOURCE_HOOK = (
    "                elif (os.environ.get(\"YETO_VERL_RANK0_CHECKPOINT_DTYPE\") == \"1\"\n"
    "                      and not self.model_config.hf_config.tie_word_embeddings):\n"
    "                    # yeto S27 G-EZ2: rank 0 reads the checkpoint dtype as the broadcast source only\n"
    "                    _yeto_source = auto_class.from_pretrained(\n"
    "                        pretrained_model_name_or_path=self.model_config.local_path,\n"
    "                        torch_dtype=\"auto\",\n"
    "                        config=self.model_config.hf_config,\n"
    "                        trust_remote_code=self.model_config.trust_remote_code,\n"
    "                    )\n"
    "                    self._yeto_full_state = _yeto_source.state_dict()\n"
    "                    _yeto_persistent = set(self._yeto_full_state)\n"
    "                    _yeto_transient = [n for n, _b in _yeto_source.named_buffers() if n not in _yeto_persistent]\n"
    "                    if _yeto_transient:\n"
    "                        raise RuntimeError(\"YETO_VERL_RANK0_CHECKPOINT_DTYPE: non-persistent buffers \"\n"
    "                                           f\"{_yeto_transient[:3]} would be uninitialized on rank 0\")\n"
    "                    del _yeto_source\n"
    "                    with torch.device(\"meta\"):\n"
    "                        module = auto_class.from_config(\n"
    "                            self.model_config.hf_config,\n"
    "                            torch_dtype=torch_dtype,\n"
    "                            trust_remote_code=self.model_config.trust_remote_code,\n"
    "                        )\n"
    + RANK0_SOURCE_ANCHOR
)
RANK0_STATE_ANCHOR = "            full_state = module.state_dict()\n"
RANK0_STATE_HOOK = (
    "            # yeto S27 G-EZ2: rank 0's broadcast source in the checkpoint dtype (ninth patch)\n"
    "            full_state = getattr(self, \"_yeto_full_state\", None)\n"
    "            self._yeto_full_state = None\n"
    "            if full_state is None:\n"
    "                full_state = module.state_dict()\n"
)
RANK0_LOAD_TARGET = "verl/utils/fsdp_utils.py"
RANK0_LOAD_ANCHOR = (
    "    if dist.get_rank() == 0:\n"
    "        model = model.to(device=get_device_id(), non_blocking=True)\n"
)
RANK0_LOAD_HOOK = (
    "    # yeto S27 G-EZ2: a rank 0 built on meta (ninth patch) is materialized like the others\n"
    "    if dist.get_rank() == 0 and not any(t.is_meta for t in [*model.parameters(), *model.buffers()]):\n"
    "        model = model.to(device=get_device_id(), non_blocking=True)\n"
)
PIN_ANCHOR = "                offload_policy = CPUOffloadPolicy(pin_memory=True)\n"
PIN_HOOK = (
    "                # yeto S27 G-EZ2: YETO_VERL_OFFLOAD_PIN_MEMORY=0 keeps the offloaded state pageable\n"
    "                offload_policy = CPUOffloadPolicy(\n"
    "                    pin_memory=os.environ.get(\"YETO_VERL_OFFLOAD_PIN_MEMORY\", \"1\") != \"0\")\n"
)
RANK0_SOURCE = ((RANK0_SOURCE_TARGET, RANK0_SOURCE_ANCHOR, RANK0_SOURCE_HOOK, "replace"),
                (RANK0_SOURCE_TARGET, RANK0_STATE_ANCHOR, RANK0_STATE_HOOK, "replace"),
                (RANK0_LOAD_TARGET, RANK0_LOAD_ANCHOR, RANK0_LOAD_HOOK, "replace"),
                (RANK0_SOURCE_TARGET, PIN_ANCHOR, PIN_HOOK, "replace"))

# (target, anchor, hook, mode): "after" inserts the hook after the anchor,
# "before" in front of it, "replace" substitutes it.
READBACK = (TARGET, ANCHOR, HOOK, "after")
RESUME_CALLS = (CALLS_TARGET, CALLS_ANCHOR, CALLS_HOOK, "after")
LORA_SEND = (SEND_TARGET, SEND_ANCHOR, SEND_HOOK, "replace")
LORA_RECV = (RECV_TARGET, RECV_ANCHOR, RECV_HOOK, "replace")
SEQ_IDX = ((SEQ_IDX_TARGET, SEQ_IDX_HELPER_ANCHOR, SEQ_IDX_HELPER, "before"),
           (SEQ_IDX_TARGET, SEQ_IDX_PACKED_ANCHOR, SEQ_IDX_PACKED_HOOK, "replace"),
           (SEQ_IDX_TARGET, SEQ_IDX_ENGINE_ANCHOR, SEQ_IDX_ENGINE_HOOK, "replace"))
# S27 G-EQ3: with no mamba kernels (Ascend NPU, CPU) transformers' NemotronH torch path
# sums the inter-chunk SSM states over the wrong axis, which puts the response log-probs
# 0.44 nats per token away from any kernel-based sampler (G-ES measured both sides; the
# fix is yeto.rl.nemotron_h_torch_fix, verified on 910B3: 0.4413 -> 0.0178). The engine
# builds the model in every worker process, so the fix is applied there, right before.
NEMOTRON_FIX_TARGET = "verl/workers/engine/fsdp/transformer_impl.py"
NEMOTRON_FIX_ANCHOR = (
    "    def _build_module(self):\n"
    "        from verl.utils.model import get_hf_auto_model_class\n"
)
NEMOTRON_FIX_HOOK = (
    "    def _build_module(self):\n"
    "        # yeto S27 G-EQ3: fix transformers' NemotronH torch path before the model is built\n"
    "        try:\n"
    "            from yeto.rl import nemotron_h_torch_fix as _yeto_nh_fix\n"
    "\n"
    "            _yeto_nh_result = _yeto_nh_fix.apply()\n"
    "            print(f'[yeto] nemotron_h_torch_fix: {_yeto_nh_result}', flush=True)\n"
    "        except Exception as _yeto_nh_error:  # a build without yeto, or another model family\n"
    "            print(f'[yeto] nemotron_h_torch_fix not applied: {_yeto_nh_error!r}', flush=True)\n"
    "        # yeto S27 G-EQ4: this implementation has no recompute hook, so an Ascend node\n"
    "        # installs one (64 GiB card; yeto.rl.nemotron_h_recompute decides by device family)\n"
    "        try:\n"
    "            from yeto.rl import nemotron_h_recompute as _yeto_nh_rc\n"
    "\n"
    "            _yeto_nh_rc_result = _yeto_nh_rc.apply()\n"
    "            print(f'[yeto] nemotron_h_recompute: {_yeto_nh_rc_result}', flush=True)\n"
    "        except Exception as _yeto_nh_rc_error:  # a build without yeto, or another family\n"
    "            print(f'[yeto] nemotron_h_recompute not applied: {_yeto_nh_rc_error!r}', flush=True)\n"
    "        from verl.utils.model import get_hf_auto_model_class\n"
)
NEMOTRON_FIX = (NEMOTRON_FIX_TARGET, NEMOTRON_FIX_ANCHOR, NEMOTRON_FIX_HOOK, "replace")

# S28 FCE (megatron profile): verl's fused log-prob path (use_fused_kernels=True) only
# knows GPTModel and a Triton/CUDA kernel. Inside _maybe_enable_fused_kernels -- reached
# only with use_fused_kernels on -- yeto.rl.adapters.verl.chunked_logprob swaps in its
# plain-torch chunked kernel and gives Megatron's HybridModel (NemotronH) the
# output_processor hook before verl patches the model, then checks every model chunk.
# Errors propagate: with the switch on there is no working fallback on Ascend.
CHUNKED_LOGPROB_TARGET = "verl/workers/engine/megatron/transformer_impl.py"
CHUNKED_LOGPROB_ANCHOR = (
    "        from verl.models.mcore.model_forward_fused import patch_fused_forward\n"
    "\n"
    "        for model in self.module:\n"
    "            patch_fused_forward(model)\n"
)
CHUNKED_LOGPROB_HOOK = (
    "        # yeto S28 FCE: chunked vocab-parallel log-probs (no [T, V] logits, no Triton)\n"
    "        from yeto.rl.adapters.verl import chunked_logprob as _yeto_chunked_logprob\n"
    "\n"
    "        print(f'[yeto] chunked_logprob install: {_yeto_chunked_logprob.install()}', flush=True)\n"
    + CHUNKED_LOGPROB_ANCHOR +
    "        print(f'[yeto] chunked_logprob check: {_yeto_chunked_logprob.check_installed(self.module)}',\n"
    "              flush=True)\n"
)
CHUNKED_LOGPROB = (CHUNKED_LOGPROB_TARGET, CHUNKED_LOGPROB_ANCHOR, CHUNKED_LOGPROB_HOOK, "replace")

PATCHES = (READBACK, RESUME_CALLS, LORA_SEND, LORA_RECV)
# S27 G-EK: the FSDP2 full-parameter island in verl's separate mode publishes over the
# same NCCL checkpoint engine as the Megatron island (and the LoRA fully_async path
# did already), so the fsdp profile takes the zmq SUB fix as well.
PROFILES = {"fsdp": (*PATCHES, *ZMQ_SUB, AUTO_CLASS, META_INIT, NEMOTRON_FIX, *RANK0_SOURCE),
            "megatron": (READBACK, RESUME_CALLS, *SEQ_IDX, *ZMQ_SUB, AUTO_CLASS, CHUNKED_LOGPROB)}


def _patched(text: str, anchor: str, hook: str, mode: str) -> str:
    if mode == "after":
        return text.replace(anchor, anchor + hook)
    if mode == "before":
        return text.replace(anchor, hook + anchor)
    return text.replace(anchor, hook)


def _applied(text: str, hook: str) -> bool:
    """The hook is already in ``text``: verbatim, or -- for a hook that opens with its
    own ``# yeto`` comment line -- that line is (a later patch may edit the rest of the
    hook: the ninth patch extends the eighth's ``else`` branch, S27 G-EZ2)."""
    if hook in text:
        return True
    first = hook.splitlines(keepends=True)[0] if hook else ""
    return first.strip().startswith("# yeto") and first in text


def apply(root: Path, profile: str = "fsdp") -> None:
    try:
        patches = PROFILES[profile]
    except KeyError:
        raise SystemExit(f"patch_verl: unknown profile {profile!r} (known: {sorted(PROFILES)})") from None
    for target, anchor, hook, mode in patches:
        path = root / target
        text = path.read_text()
        if _applied(text, hook):
            continue
        if text.count(anchor) != 1:
            raise SystemExit(f"patch_verl: anchor occurs {text.count(anchor)} times in {target}")
        path.write_text(_patched(text, anchor, hook, mode))


if __name__ == "__main__":
    argv = sys.argv[1:]
    chosen = "fsdp"
    if "--profile" in argv:
        at = argv.index("--profile")
        chosen = argv[at + 1]
        argv = argv[:at] + argv[at + 2:]
    apply(Path(argv[0]), chosen)
    print(f"patch_verl: ok ({chosen})")
