"""Apply the S29 phase-timer hooks to a verl-fork tree. Usage: python3 apply_patch.py <verl-fork-root>"""
import shutil, sys, os
root = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
shutil.copy(os.path.join(here, "s29_phase_timer.py"), os.path.join(root, "verl/utils/s29_phase_timer.py"))
IMP = "from verl.utils.s29_phase_timer import optimizer_threads as _s29_opt_threads, phase as _s29_phase, flush as _s29_flush, patch_megatron as _s29_patch, lap as _s29_lap, lap_start as _s29_lap_start\n"

def sub(s, old, new, count=1, after=None):
    base = 0 if after is None else s.index(after)
    i = s.index(old, base)
    assert count == 1
    return s[:i] + new + s[i + len(old):]

def edit(rel, fn):
    p = os.path.join(root, rel); s = open(p).read()
    if "_s29_" in s:
        print("already patched", rel); return
    s = fn(s); open(p, "w").write(s); print("patched", rel)

def ew(s):
    s = sub(s, "from codetiming import Timer\n", "from codetiming import Timer\n" + IMP)
    # update_actor / compute_log_prob
    s = sub(s, """    def compute_log_prob(self, data: TensorDict) -> TensorDict:
        output = self.actor.infer_batch(data)

        return output.cpu() if output is not None else None
""", """    def compute_log_prob(self, data: TensorDict) -> TensorDict:
        _s29_patch("compute_log_prob")
        with _s29_phase("lp.TOTAL"):
            output = self.actor.infer_batch(data)
            with _s29_phase("lp.output_cpu"):
                output = output.cpu() if output is not None else None
        _s29_flush("compute_log_prob")
        return output
""")
    s = sub(s, """    def update_actor(self, data: TensorDict) -> TensorDict:
        output = self.actor.train_mini_batch(data=data)
        return output.cpu() if output is not None else None
""", """    def update_actor(self, data: TensorDict) -> TensorDict:
        _s29_patch()
        with _s29_phase("ua.TOTAL"):
            output = self.actor.train_mini_batch(data=data)
            with _s29_phase("ua.output_cpu"):
                output = output.cpu() if output is not None else None
        _s29_flush("update_actor")
        return output
""")
    # train_mini_batch
    A = "    def train_mini_batch(self, data: TensorDict) -> TensorDict:"
    s = sub(s, "        maybe_fix_3d_position_ids(data)\n        batch_size_per_dp = data.shape[0]\n",
            "        _s29_lap_start(\"ua\")\n        maybe_fix_3d_position_ids(data)\n        batch_size_per_dp = data.shape[0]\n", after=A)
    s = sub(s, """        with (
            self.engine.train_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="train_batch", logger=None),
        ):""", """        _s29_lap("ua", "ua.prep_iterator")
        with (
            self.engine.train_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="train_batch", logger=None),
        ):""", after=A)
    s = sub(s, "            for batch_idx, mini_batch_td in enumerate(dataloader):\n",
            "            for batch_idx, mini_batch_td in enumerate(dataloader):\n                _s29_lap(\"ua\", \"ua.ctx_enter+dataloader_next\")\n", after=A)
    s = sub(s, "                with torch.profiler.record_function(f\"mini_batch{batch_idx}\"):\n                    actor_output = self.train_batch(mini_batch_td)\n",
            "                _s29_lap(\"ua\", \"ua.token_num_allgather\")\n                with torch.profiler.record_function(f\"mini_batch{batch_idx}\"):\n                    actor_output = self.train_batch(mini_batch_td)\n                _s29_lap(\"ua\", \"ua.train_batch\")\n", after=A)
    s = sub(s, "            if self.engine.is_mp_src_rank_with_outputs():\n                actor_output = [tu.get(output, \"metrics\")",
            "            _s29_lap(\"ua\", \"ua.profiler_step\")\n            if self.engine.is_mp_src_rank_with_outputs():\n                actor_output = [tu.get(output, \"metrics\")", after=A)
    s = sub(s, "                output = None\n        return output\n",
            "                output = None\n            _s29_lap(\"ua\", \"ua.metrics_agg\")\n        _s29_lap(\"ua\", \"ua.ctx_exit\")\n        return output\n", after=A)
    # worker train_batch
    B = "    def train_batch(self, data: TensorDict) -> TensorDict:"
    s = sub(s, """        with (
            self.engine.train_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="train_batch", logger=None) as timer,
        ):""", """        _s29_lap_start("tb")
        with (
            self.engine.train_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="train_batch", logger=None) as timer,
        ):""", after=B)
    s = sub(s, "        delta_time = timer.last\n", "        delta_time = timer.last\n        _s29_lap(\"tb\", \"tb.engine_train_batch\")\n", after=B)
    s = sub(s, "        if self.engine.is_mp_src_rank_with_outputs():\n            # we don't need model_output",
            "        _s29_lap(\"tb\", \"tb.lr_scheduler\")\n        if self.engine.is_mp_src_rank_with_outputs():\n            # we don't need model_output", after=B)
    s = sub(s, "        return final_output\n", "        _s29_lap(\"tb\", \"tb.postprocess_cpu\")\n        return final_output\n", after=B)
    C = "    def infer_batch(self, data: TensorDict) -> TensorDict:"
    s = sub(s, """        with (
            self.engine.eval_mode(disable_auto_offload=disable_auto_offload),""", """        _s29_lap_start("ib")
        with (
            self.engine.eval_mode(disable_auto_offload=disable_auto_offload),""", after=C)
    s = sub(s, "        delta_time = timer.last\n", "        delta_time = timer.last\n        _s29_lap(\"ib\", \"ib.engine_infer_batch\")\n", after=C)
    s = sub(s, "            final_output = None\n", "            final_output = None\n        _s29_lap(\"ib\", \"ib.postprocess_cpu\")\n", after=C)
    return s
edit("verl/workers/engine_workers.py", ew)

def eb(s):
    s = sub(s, "class BaseEngineCtx:", IMP + "\n\nclass BaseEngineCtx:")
    s = sub(s, """    def __enter__(self):
        self.engine.mode = self.mode
        self._context_switch(get_device_name())

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._context_switch("cpu")
""", """    def __enter__(self):
        self.engine.mode = self.mode
        with _s29_phase(f"ctx.{self.mode}.enter_onload"):
            self._context_switch(get_device_name())

    def __exit__(self, exc_type, exc_val, exc_tb):
        with _s29_phase(f"ctx.{self.mode}.exit_offload"):
            self._context_switch("cpu")
""")
    s = sub(s, """        self.optimizer_zero_grad()
        outputs = self.forward_backward_batch(data, loss_function, forward_only=False)
        grad_norm = self.optimizer_step()
""", """        with _s29_phase("tb.zero_grad"):
            self.optimizer_zero_grad()
        with _s29_phase("tb.forward_backward_batch"):
            outputs = self.forward_backward_batch(data, loss_function, forward_only=False)
        with _s29_phase("tb.optimizer_step"), _s29_opt_threads():
            grad_norm = self.optimizer_step()
""")
    return s
edit("verl/workers/engine/base.py", eb)

def et(s):
    s = sub(s, "from megatron.core.pipeline_parallel import get_forward_backward_func\n",
            "from megatron.core.pipeline_parallel import get_forward_backward_func\n" + IMP)
    A = "    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> Any:\n"
    s = sub(s, A, A + "        _s29_lap_start(\"fbb\")\n")
    s = sub(s, "        vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()\n        if vpp_size is not None and vpp_size > 1:\n            num_batches_divided_by",
            "        _s29_lap(\"fbb\", \"fbb.tokcount_allreduce_item\")\n        vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()\n        if vpp_size is not None and vpp_size > 1:\n            num_batches_divided_by", after=A)
    s = sub(s, "        if num_batches_divided_by is not None:\n            assert len(micro_batches)",
            "        _s29_lap(\"fbb\", \"fbb.prepare_micro_batches\")\n        if num_batches_divided_by is not None:\n            assert len(micro_batches)", after=A)
    s = sub(s, "        losses_reduced = forward_backward_func(\n",
            "        _s29_lap(\"fbb\", \"fbb.setup\")\n        losses_reduced = forward_backward_func(\n", after=A)
    s = sub(s, "            forward_only=forward_only,\n        )\n",
            "            forward_only=forward_only,\n        )\n        _s29_lap(\"fbb\", \"fbb.FBF.fwd_only\" if forward_only else \"fbb.FBF.train\")\n", after=A)
    s = sub(s, "            RouterReplay.clear_global_router_replay_action()\n        return output\n",
            "            RouterReplay.clear_global_router_replay_action()\n        _s29_lap(\"fbb\", \"fbb.post\")\n        return output\n", after=A)
    s = sub(s, "        batch: TensorDict = next(batch_iter)\n\n        batch = batch.to(get_device_id())\n",
            "        batch: TensorDict = next(batch_iter)\n\n        with _s29_phase(\"mb.batch_h2d\", level=2):\n            batch = batch.to(get_device_id())\n")
    s = sub(s, "        output = {\"loss\": loss.detach().item(), \"metrics\": metrics}\n",
            "        with _s29_phase(\"mb.loss_item\", level=2):\n            _loss_item = loss.detach().item()\n        output = {\"loss\": _loss_item, \"metrics\": metrics}\n")
    return s
edit("verl/workers/engine/megatron/transformer_impl.py", et)
