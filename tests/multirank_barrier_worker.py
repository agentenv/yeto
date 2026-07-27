"""Two-rank CPU worker exercising the distributed adapter barrier."""

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from yeto.fragments import build_layout
from yeto.learner import run_inner_loop
from yeto.protocol import DTYPE_BF16
from yeto.tensor_io import pack_fragment


class TinyLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(11, 5)
        self.proj = torch.nn.Linear(5, 11, bias=False)

    def forward(self, input_ids):
        return SimpleNamespace(logits=self.proj(self.embed(input_ids)))


class Flag:
    def __init__(self) -> None:
        self.value = False

    def is_set(self) -> bool:
        return self.value


class FakeBarrierClient:
    def __init__(self, layout, params, total_steps: int) -> None:
        self.dtype = DTYPE_BF16
        self.connect_timeout = 10.0
        self.shutdown = Flag()
        self.layout = layout
        self.total_steps = total_steps
        self.updates = [
            SimpleNamespace(
                fragment_id=fid,
                version=0,
                data=pack_fragment(fragment, params, DTYPE_BF16),
            )
            for fid, fragment in enumerate(layout.fragments)
        ]
        self.pulls = []
        self.pushes = []
        self._queue_round(1)

    def _queue_round(self, first_step: int) -> None:
        for offset, _fragment in enumerate(self.layout.fragments):
            step = first_step + offset
            self.pulls.append(SimpleNamespace(fragment_id=offset, global_step=step))

    def check_health(self) -> None:
        return None

    def drain_updates(self):
        updates, self.updates = self.updates, []
        return updates

    def drain_pulls(self):
        pulls, self.pulls = self.pulls, []
        return pulls

    def push_fragment(
        self,
        fid,
        pull_step,
        base_version,
        local_step,
        c_steps,
        c_tokens,
        payload,
    ):
        self.pushes.append(
            {
                "fid": fid,
                "pull_step": pull_step,
                "base_version": base_version,
                "local_step": local_step,
                "c_steps": c_steps,
                "c_tokens": c_tokens,
            }
        )
        self.updates.append(
            SimpleNamespace(fragment_id=fid, version=pull_step, data=payload)
        )
        if pull_step % self.layout.num_fragments == 0:
            if pull_step < self.total_steps:
                self._queue_round(pull_step + 1)
            else:
                self.shutdown.value = True


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2
    torch.manual_seed(1234)
    model = TinyLM()
    params = dict(model.named_parameters())
    layout = build_layout(
        [(name, param.numel()) for name, param in params.items()],
        2,
        named_shapes={name: tuple(param.shape) for name, param in params.items()},
    )
    optimizer = torch.optim.AdamW(params.values(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    ids = torch.tensor([[1, 2, 3, 4]] * 8, dtype=torch.long)
    weights = torch.ones_like(ids, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(ids, weights), batch_size=1
    )
    client = FakeBarrierClient(layout, params, total_steps=4) if rank == 0 else None
    output_dir = os.environ["YETO_BARRIER_TEST_OUTPUT"] + f"/rank-{rank}"
    args = argparse.Namespace(
        learner_id=0,
        tuning="lora",
        shard="ddp",
        merge_alpha=0.5,
        barrier_sync=True,
        debug_broadcast_lag_commits=0,
        debug_step_sleep_ms=0.0,
        debug_push_delay_ms=0.0,
        debug_delay_jitter_ms=0.0,
        fixed_window_microsteps=2,
        fixed_window_tokens=16,
        fixed_window_schedule=None,
        allow_terminal_partial_fixed_window=False,
        max_local_steps=4,
        micro_batch_size=1,
        grad_accum=1,
        seq_len=4,
        loss_function="cross_entropy",
        probe_data=None,
        output_dir=output_dir,
    )
    run_inner_loop(
        args,
        model,
        params,
        layout,
        optimizer,
        scheduler,
        loader,
        client,
        rank,
        world,
        torch.device("cpu"),
        tokenizer=None,
    )
    flat = torch.cat([param.detach().reshape(-1) for param in params.values()])
    gathered = [torch.empty_like(flat) for _ in range(world)]
    dist.all_gather(gathered, flat)
    assert torch.equal(gathered[0], gathered[1])
    if rank == 0:
        assert [push["pull_step"] for push in client.pushes] == [1, 2, 3, 4]
        assert [push["local_step"] for push in client.pushes] == [2, 2, 4, 4]
        assert all(push["c_steps"] == 2 for push in client.pushes)
        assert all(push["c_tokens"] == 16 for push in client.pushes)
        print("MULTIRANK_BARRIER_PASS", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
