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

Usage: python patch_verl.py <verl source root>
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

# (target, anchor, hook, mode): "after" inserts the hook after the anchor,
# "replace" substitutes it.
PATCHES = ((TARGET, ANCHOR, HOOK, "after"), (CALLS_TARGET, CALLS_ANCHOR, CALLS_HOOK, "after"),
           (SEND_TARGET, SEND_ANCHOR, SEND_HOOK, "replace"), (RECV_TARGET, RECV_ANCHOR, RECV_HOOK, "replace"))


def apply(root: Path) -> None:
    for target, anchor, hook, mode in PATCHES:
        path = root / target
        text = path.read_text()
        if hook in text:
            continue
        if text.count(anchor) != 1:
            raise SystemExit(f"patch_verl: anchor occurs {text.count(anchor)} times in {target}")
        path.write_text(text.replace(anchor, anchor + hook if mode == "after" else hook))


if __name__ == "__main__":
    apply(Path(sys.argv[1]))
    print("patch_verl: ok")
