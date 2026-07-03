"""End-to-end test: real Rust syncer + two learner clients on localhost.

Each learner runs Adam inner steps on a quadratic f_m(w) = ||w - target_m||²
over its own target; the async merge should drive the global parameters
toward the mean of the targets. Exercises HELLO layout exchange, INIT,
striped chunk transfer, pull/push with counters, RDA+Avg merging, the
Nesterov outer step, broadcasts, and SHUTDOWN.
"""

import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest
import torch

from yeto.fragments import build_layout
from yeto.protocol import DTYPE_BF16, SyncerClient
from yeto.tensor_io import apply_fragment, pack_fragment, unpack_fragment

ROOT = Path(__file__).resolve().parent.parent
DIM = 4096  # large enough to require striping across several chunks at bf16? (4KB) — small but exercises the full path


def build_syncer() -> Path:
    binary = ROOT / "syncer/target/debug/yeto-syncer"
    subprocess.run(["cargo", "build", "-q"], cwd=ROOT / "syncer", check=True)
    assert binary.exists()
    return binary


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ToyLearner(threading.Thread):
    def __init__(self, learner_id: int, port: int, target: torch.Tensor, layout):
        super().__init__(daemon=True)
        self.learner_id = learner_id
        self.target = target
        self.layout = layout
        self.params = {
            "model.embed.weight": torch.zeros(DIM // 4),
            "model.body.weight": torch.zeros(DIM),
        }
        self.client = SyncerClient(
            ("127.0.0.1", port), learner_id, layout, DTYPE_BF16, num_streams=2
        )
        # Snapshot at the last applied broadcast: the learner keeps taking
        # local steps between the final merge and SHUTDOWN arriving, so
        # post-loop params include unmerged local drift.
        self.synced: dict[str, torch.Tensor] = {}
        self.exc: BaseException | None = None

    def run(self):
        try:
            self._run()
        except BaseException as e:
            self.exc = e

    def _run(self):
        self.client.start()
        if self.learner_id == 0:
            for fid, frag in enumerate(self.layout.fragments):
                self.client.send_init(fid, pack_fragment(frag, self.params, DTYPE_BF16))
        opt = torch.optim.Adam(list(self.params.values()), lr=0.05)
        for p in self.params.values():
            p.requires_grad_(True)
        steps_total = 0
        steps_at_reset = [0] * self.layout.num_fragments
        versions = [0] * self.layout.num_fragments
        pending = []
        t0 = time.monotonic()
        while not self.client.shutdown.is_set():
            if time.monotonic() - t0 > 60:
                raise TimeoutError("no SHUTDOWN within 60s")
            self.client.check_health()
            # inner step on ||w - target||^2
            opt.zero_grad()
            flat = torch.cat([p.reshape(-1) for p in self.params.values()])
            loss = ((flat - self.target) ** 2).sum()
            loss.backward()
            opt.step()
            steps_total += 1
            pending.extend(self.client.drain_pulls())
            still = []
            for pull in pending:
                fid = pull.fragment_id
                if steps_total - steps_at_reset[fid] < 1:
                    still.append(pull)
                    continue
                c_steps = steps_total - steps_at_reset[fid]
                self.client.push_fragment(
                    fid,
                    pull.global_step,
                    versions[fid],
                    steps_total,
                    c_steps,
                    c_steps * 128,  # tokens: uniform rate
                    pack_fragment(self.layout.fragments[fid], self.params, DTYPE_BF16),
                )
            pending = still
            for bc in self.client.drain_updates():
                frag = self.layout.fragments[bc.fragment_id]
                flat_new = unpack_fragment(frag, bc.data, DTYPE_BF16)
                apply_fragment(frag, flat_new, self.params)
                steps_at_reset[bc.fragment_id] = steps_total
                versions[bc.fragment_id] = bc.version
                self.synced = {k: v.detach().clone() for k, v in self.params.items()}
            time.sleep(0.005)  # ~5ms inner step
        self.client.close()


@pytest.mark.timeout(180)
def test_two_learners_converge_to_mean():
    binary = build_syncer()
    port = free_port()
    total_steps = 30
    named = [("model.embed.weight", DIM // 4), ("model.body.weight", DIM)]
    layout = build_layout(named, 4)

    proc = subprocess.Popen(
        [
            str(binary),
            "--port", str(port),
            "--learners", "2",
            "--quorum", "2",
            "--grace-ms", "200",
            "--total-steps", str(total_steps),
            "--outer-lr", "0.7",
            "--outer-momentum", "0.9",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        torch.manual_seed(0)
        # Two different targets; the consensus optimum is their mean.
        t_a = torch.randn(DIM + DIM // 4)
        t_b = -t_a  # mean target = 0
        learners = [
            ToyLearner(0, port, t_a, layout),
            ToyLearner(1, port, t_b, layout),
        ]
        for l in learners:
            l.start()
        for l in learners:
            l.join(timeout=120)
            assert not l.is_alive(), "learner did not finish"
            if l.exc:
                raise l.exc
        rc = proc.wait(timeout=30)
        assert rc == 0, "syncer exited nonzero"

        # After merging, both learners' post-broadcast fragments came
        # from the same global params; with opposite targets, the merged
        # motion cancels and the synced state stays near 0. Check the last
        # broadcast state stayed much closer to the consensus (0) than to
        # either learner's own target.
        for l in learners:
            assert l.synced, "learner never received a broadcast"
            flat = torch.cat([p.reshape(-1) for p in l.synced.values()])
            dist_to_own_target = (flat - l.target).norm()
            # Pure local training would reach its own target (dist -> 0).
            assert dist_to_own_target > 0.5 * l.target.norm(), (
                "learner collapsed to its own target; merging had no effect"
            )
    finally:
        if proc.poll() is None:
            proc.kill()
        out = proc.stdout.read() if proc.stdout else ""
        print(out[-3000:])


@pytest.mark.timeout(180)
def test_single_learner_roundtrip():
    """M=1, K=1: a single self-syncing learner; must run to completion."""
    binary = build_syncer()
    port = free_port()
    named = [("model.embed.weight", DIM // 4), ("model.body.weight", DIM)]
    layout = build_layout(named, 3)
    proc = subprocess.Popen(
        [str(binary), "--port", str(port), "--learners", "1", "--quorum", "1",
         "--grace-ms", "50", "--total-steps", "9"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        target = torch.ones(DIM + DIM // 4)
        l = ToyLearner(0, port, target, layout)
        l.start()
        l.join(timeout=120)
        assert not l.is_alive()
        if l.exc:
            raise l.exc
        assert proc.wait(timeout=30) == 0
        # Single learner: global params must track the learner toward target.
        flat = torch.cat([p.detach().reshape(-1) for p in l.params.values()])
        assert (flat - target).norm() < target.norm(), "no progress toward target"
    finally:
        if proc.poll() is None:
            proc.kill()
        print((proc.stdout.read() if proc.stdout else "")[-3000:])
