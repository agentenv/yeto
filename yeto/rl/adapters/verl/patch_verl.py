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
patch passes the pack length when context parallelism is off.

Profiles: ``fsdp`` (default; the image ``verl-build``) applies patches 1-4 and,
since S27 G-EK, the sixth (zmq SUB, below: the FSDP2 separate mode and the LoRA
fully_async path use the same NCCL checkpoint engine); ``megatron`` (the image ``verl-megatron-build``) applies 1, 2 and 5 -- the
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
    "    \"\"\"yeto S27 G-DC: ``total_tokens`` of the THD pack, so ``PackedSeqParams`` derives\n"
    "    ``seq_idx`` and a Mamba layer restarts its state at every packed sample (CP off only).\"\"\"\n"
    "    if cp_size != 1 or not _packed_seq_params_supports(\"total_tokens\"):\n"
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

# (target, anchor, hook, mode): "after" inserts the hook after the anchor,
# "before" in front of it, "replace" substitutes it.
READBACK = (TARGET, ANCHOR, HOOK, "after")
RESUME_CALLS = (CALLS_TARGET, CALLS_ANCHOR, CALLS_HOOK, "after")
LORA_SEND = (SEND_TARGET, SEND_ANCHOR, SEND_HOOK, "replace")
LORA_RECV = (RECV_TARGET, RECV_ANCHOR, RECV_HOOK, "replace")
SEQ_IDX = ((SEQ_IDX_TARGET, SEQ_IDX_HELPER_ANCHOR, SEQ_IDX_HELPER, "before"),
           (SEQ_IDX_TARGET, SEQ_IDX_PACKED_ANCHOR, SEQ_IDX_PACKED_HOOK, "replace"),
           (SEQ_IDX_TARGET, SEQ_IDX_ENGINE_ANCHOR, SEQ_IDX_ENGINE_HOOK, "replace"))
PATCHES = (READBACK, RESUME_CALLS, LORA_SEND, LORA_RECV)
# S27 G-EK: the FSDP2 full-parameter island in verl's separate mode publishes over the
# same NCCL checkpoint engine as the Megatron island (and the LoRA fully_async path
# did already), so the fsdp profile takes the zmq SUB fix as well.
PROFILES = {"fsdp": (*PATCHES, *ZMQ_SUB), "megatron": (READBACK, RESUME_CALLS, *SEQ_IDX, *ZMQ_SUB)}


def _patched(text: str, anchor: str, hook: str, mode: str) -> str:
    if mode == "after":
        return text.replace(anchor, anchor + hook)
    if mode == "before":
        return text.replace(anchor, hook + anchor)
    return text.replace(anchor, hook)


def apply(root: Path, profile: str = "fsdp") -> None:
    try:
        patches = PROFILES[profile]
    except KeyError:
        raise SystemExit(f"patch_verl: unknown profile {profile!r} (known: {sorted(PROFILES)})") from None
    for target, anchor, hook, mode in patches:
        path = root / target
        text = path.read_text()
        if hook in text:
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
