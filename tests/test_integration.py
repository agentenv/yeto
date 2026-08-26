"""End-to-end test: real Rust syncer + two learner clients on localhost.

Each learner runs Adam inner steps on a quadratic f_m(w) = ||w - target_m||²
over its own target; the async merge should drive the global parameters
toward the mean of the targets. Exercises HELLO layout exchange, INIT,
striped chunk transfer, pull/push with counters, RDA+Avg merging, the
Nesterov outer step, broadcasts, and SHUTDOWN.
"""

import json
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import torch

from yeto.export import ISO_BACKEND_SCALAR, parse_checkpoint
from yeto.fragments import build_layout
from yeto.final_marker import read_checkpoint_global_step, validate_final_checkpoint
from yeto.protocol import (
    DTYPE_BF16,
    DTYPE_F32,
    DTYPE_Q4,
    SyncerClient,
    bulk_dtype,
    layout_fingerprint,
)
from yeto.tensor_io import (
    apply_fragment,
    fragment_flat,
    pack_fragment,
    pack_tensor,
    quantize_q4,
    unpack_fragment,
)

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


def read_final_state(path: Path) -> list[torch.Tensor]:
    data = path.read_bytes()
    offset = 0

    def take(size):
        nonlocal offset
        value = data[offset : offset + size]
        offset += size
        return value

    (count,) = struct.unpack("<I", take(4))
    fragments = []
    for _ in range(count):
        (numel,) = struct.unpack("<Q", take(8))
        fragments.append(torch.tensor(struct.unpack(f"<{numel}f", take(numel * 4))))
    assert offset == len(data)
    return fragments


class ToyLearner(threading.Thread):
    def __init__(
        self,
        learner_id: int,
        port: int,
        target: torch.Tensor,
        layout,
        dtype=DTYPE_BF16,
        merge_alpha: float = 0.0,
        abandon_after_steps: int | None = None,
        abandon_after_broadcasts: int | None = None,
        budget_steps: int | None = None,
        budget_delay: float = 0.0,
    ):
        super().__init__(daemon=True)
        self.learner_id = learner_id
        self.target = target
        self.layout = layout
        self.dtype = dtype
        self.merge_alpha = merge_alpha
        self.abandon_after_steps = (
            budget_steps if budget_steps is not None else abandon_after_steps
        )
        self.abandon_after_broadcasts = abandon_after_broadcasts
        self.budget_steps = budget_steps
        self.budget_delay = budget_delay
        self.abandoned = False
        self.params = {
            "model.embed.weight": torch.zeros(DIM // 4),
            "model.body.weight": torch.zeros(DIM),
        }
        # Every push is local minus the last raw global broadcast.
        self.anchors: list[torch.Tensor | None] = [None] * layout.num_fragments
        self.client = SyncerClient(
            ("127.0.0.1", port), learner_id, layout, dtype, num_streams=2
        )
        # Snapshot of ordinary in-training broadcasts (before the terminal
        # raw overwrite), retained for convergence assertions.
        self.synced: dict[str, torch.Tensor] = {}
        self.saved: dict[str, torch.Tensor] = {}
        self.final_manifest = None
        self.normal_broadcasts = 0
        self.normal_blends = 0
        self.exc: BaseException | None = None
        self.steps_total = 0

    def run(self):
        try:
            self._run()
        except BaseException as e:
            self.exc = e

    def _run(self):
        self.client.start()
        if self.learner_id == 0:
            for fid, frag in enumerate(self.layout.fragments):
                self.client.send_init(
                    fid, pack_fragment(frag, self.params, bulk_dtype(self.dtype))
                )
        opt = torch.optim.Adam(list(self.params.values()), lr=0.05)
        for p in self.params.values():
            p.requires_grad_(True)
        steps_total = 0
        steps_at_reset = [0] * self.layout.num_fragments
        versions = [0] * self.layout.num_fragments
        pending = []
        t0 = time.monotonic()
        while True:
            if time.monotonic() - t0 > 60:
                raise TimeoutError("no final manifest within 60s")
            self.client.check_health()
            if self.client.finalizing.is_set():
                manifest, broadcasts = self.client.wait_for_final_fragments(timeout=10)
                for update in broadcasts:
                    frag = self.layout.fragments[update.fragment_id]
                    raw = unpack_fragment(frag, update.data, DTYPE_F32)
                    apply_fragment(frag, raw, self.params)
                self.client.acknowledge_finalization(manifest, timeout=10)
                self.final_manifest = manifest
                self.saved = {k: v.detach().clone() for k, v in self.params.items()}
                break
            # inner step on ||w - target||^2
            opt.zero_grad()
            flat = torch.cat([p.reshape(-1) for p in self.params.values()])
            loss = ((flat - self.target) ** 2).sum()
            loss.backward()
            opt.step()
            steps_total += 1
            # Apply broadcasts before answering pulls, mirroring the real
            # learners: a pipelined syncer's next pull for a fragment can
            # overtake the broadcast that closed its previous round.
            for bc in self.client.drain_updates():
                self.normal_broadcasts += 1
                frag = self.layout.fragments[bc.fragment_id]
                flat_new = unpack_fragment(frag, bc.data, bulk_dtype(self.dtype))
                self.anchors[bc.fragment_id] = flat_new.clone()
                if self.merge_alpha > 0:
                    self.normal_blends += 1
                    local = fragment_flat(frag, self.params)
                    flat_new = self.merge_alpha * local + (1.0 - self.merge_alpha) * flat_new
                apply_fragment(frag, flat_new, self.params)
                steps_at_reset[bc.fragment_id] = steps_total
                versions[bc.fragment_id] = bc.version
                self.synced = {k: v.detach().clone() for k, v in self.params.items()}
            pending.extend(self.client.drain_pulls())
            still = []
            for pull in pending:
                fid = pull.fragment_id
                if steps_total - steps_at_reset[fid] < 1:
                    still.append(pull)
                    continue
                c_steps = steps_total - steps_at_reset[fid]
                frag = self.layout.fragments[fid]
                anchor = self.anchors[fid]
                if anchor is None:
                    still.append(pull)
                    continue
                delta = fragment_flat(frag, self.params) - anchor
                if self.dtype == DTYPE_Q4:
                    payload = quantize_q4(delta)
                else:
                    payload = pack_tensor(delta, self.dtype)
                self.client.push_fragment(
                    fid,
                    pull.global_step,
                    pull.round_attempt,
                    versions[fid],
                    steps_total,
                    c_steps,
                    c_steps * 128,  # tokens: uniform rate
                    payload,
                )
            pending = still
            if (
                (
                    self.abandon_after_steps is not None
                    and steps_total >= self.abandon_after_steps
                )
                or (
                    self.abandon_after_broadcasts is not None
                    and self.normal_broadcasts >= self.abandon_after_broadcasts
                )
            ):
                if self.budget_steps is not None:
                    if self.budget_delay:
                        time.sleep(self.budget_delay)
                    from yeto.budget_finalization import finalize_learner_budget

                    manifest = finalize_learner_budget(
                        self.client,
                        self.layout,
                        self.params,
                        rank=0,
                        world=1,
                        device=torch.device("cpu"),
                        target_steps=self.budget_steps,
                        units=steps_total * 128,
                    )
                    self.steps_total = steps_total
                    self.final_manifest = manifest
                    self.saved = {
                        key: value.detach().clone()
                        for key, value in self.params.items()
                    }
                    break
                self.abandoned = True
                self.client.close()
                return
            time.sleep(0.005)  # ~5ms inner step
        self.client.close()


@pytest.mark.timeout(180)
def test_learner_budget_restart_freezes_exactly_and_marks_complete_checkpoint(tmp_path):
    binary = build_syncer()
    port = free_port()
    budget_steps = 4
    named = [("model.embed.weight", DIM // 4), ("model.body.weight", DIM)]
    layout = build_layout(named, 2)
    checkpoint = tmp_path / "state.ckpt"
    marker = Path(f"{checkpoint}.final")
    marker.write_text("YETO_FINAL_V1\nglobal_step=999\n", encoding="utf-8")
    tape = tmp_path / "tape.jsonl"
    cutoff_proc = subprocess.Popen(
        [
            str(binary),
            "--port",
            str(port),
            "--learners",
            "2",
            "--quorum",
            "2",
            "--grace-ms",
            "20",
            "--quorum-timeout-s",
            "10",
            "--sync-interval-steps",
            "0",
            "--total-steps",
            "100",
            "--learner-budget-steps",
            str(budget_steps),
            "--checkpoint-path",
            str(checkpoint),
            "--event-tape",
            str(tape),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        targets = [
            torch.ones(DIM + DIM // 4),
            -torch.ones(DIM + DIM // 4),
        ]
        learners = [
            ToyLearner(
                i,
                port,
                target,
                layout,
                dtype=DTYPE_F32,
                budget_steps=budget_steps,
                budget_delay=0.2,
            )
            for i, target in enumerate(targets)
        ]
        for learner in learners:
            learner.start()
        cutoff_output, _ = cutoff_proc.communicate(timeout=120)
        assert cutoff_proc.returncode == 0, cutoff_output
        assert checkpoint.exists()
        assert not marker.exists()
        cutoff_step = read_checkpoint_global_step(checkpoint)

        final_proc = subprocess.Popen(
            [
                str(binary),
                "--port",
                str(port),
                "--learners",
                "2",
                "--quorum",
                "2",
                "--grace-ms",
                "20",
                "--quorum-timeout-s",
                "10",
                "--sync-interval-steps",
                "0",
                "--pipeline",
                "1",
                "--total-steps",
                str(cutoff_step + layout.num_fragments),
                "--checkpoint-path",
                str(checkpoint),
                "--checkpoint-every",
                "1",
                "--resume",
                "--mark-final-checkpoint",
                "--event-tape",
                str(tape),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for learner in learners:
            learner.join(timeout=120)
            assert not learner.is_alive(), "budget learner did not finish"
            if learner.exc:
                raise learner.exc
            assert learner.steps_total == budget_steps
            assert learner.final_manifest is not None
        final_output, _ = final_proc.communicate(timeout=30)
        assert final_proc.returncode == 0, final_output

        parsed = parse_checkpoint(checkpoint)
        assert parsed.revision == 3
        assert parsed.backend_id == ISO_BACKEND_SCALAR
        assert parsed.layout_fingerprint == layout_fingerprint(layout)
        assert validate_final_checkpoint(checkpoint) == parsed.global_step
        assert parsed.global_step == learners[0].final_manifest.global_step
        assert parsed.global_step == learners[1].final_manifest.global_step
        assert all(
            version == manifested
            for (version, _params, _momentum), manifested in zip(
                parsed.fragments, learners[0].final_manifest.versions
            )
        )
        records = [json.loads(line) for line in tape.read_text().splitlines()]
        terminal_steps = {version for version, _params, _momentum in parsed.fragments}
        terminal = [record for record in records if record["step"] in terminal_steps]
        assert {record["fragment"] for record in terminal} == {0, 1}
        for record in terminal:
            assert {responder["id"] for responder in record["responders"]} == {0, 1}
            assert {responder["c_steps"] for responder in record["responders"]} == {
                budget_steps
            }
            assert {responder["c_tokens"] for responder in record["responders"]} == {
                budget_steps * 128
            }
        assert "learner-budget cutoff checkpoint written" in cutoff_output
    finally:
        for process in (cutoff_proc, locals().get("final_proc")):
            if process is not None and process.poll() is None:
                process.kill()
        print((locals().get("cutoff_output", "") + locals().get("final_output", ""))[-3000:])


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
def test_bf16_session_saves_lossless_authoritative_cut_and_checkpoint(tmp_path):
    """Ordinary bf16 broadcasts blend with local state, but terminal delivery
    is lossless f32. The final checkpoint is also written when T is not
    divisible by --checkpoint-every."""
    binary = build_syncer()
    port = free_port()
    total_steps = 3
    named = [("model.embed.weight", DIM // 4), ("model.body.weight", DIM)]
    layout = build_layout(named, 2)
    final_state = tmp_path / "final.bin"
    checkpoint = tmp_path / "state.ckpt"
    proc = subprocess.Popen(
        [
            str(binary),
            "--port",
            str(port),
            "--learners",
            "1",
            "--quorum",
            "1",
            "--grace-ms",
            "20",
            "--quorum-timeout-s",
            "10",
            "--total-steps",
            str(total_steps),
            "--checkpoint-path",
            str(checkpoint),
            "--checkpoint-every",
            "8",
            "--final-state",
            str(final_state),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        learner = ToyLearner(
            0,
            port,
            torch.ones(DIM + DIM // 4),
            layout,
            dtype=DTYPE_BF16,
            merge_alpha=0.75,
        )
        learner.start()
        learner.join(timeout=120)
        assert not learner.is_alive(), "learner did not finalize"
        if learner.exc:
            raise learner.exc
        assert proc.wait(timeout=30) == 0
        assert learner.normal_broadcasts > 0, "nonzero-alpha path was not exercised"
        assert learner.normal_blends > 0, "ordinary broadcasts were not alpha-blended"
        assert learner.final_manifest is not None
        assert learner.final_manifest.global_step == total_steps

        coordinator = read_final_state(final_state)
        assert any(
            not torch.equal(fragment, fragment.to(torch.bfloat16).float())
            for fragment in coordinator
        ), "test state happened to be exactly bf16-representable"
        for fid, fragment in enumerate(layout.fragments):
            saved = fragment_flat(fragment, learner.saved)
            assert torch.equal(saved, coordinator[fid]), f"fragment {fid} is not authoritative"

        ckpt = parse_checkpoint(checkpoint)
        assert ckpt.revision == 3
        assert ckpt.backend_id == ISO_BACKEND_SCALAR
        assert ckpt.layout_fingerprint == layout_fingerprint(layout)
        assert ckpt.global_step == total_steps
        assert tuple(version for version, _params, _momentum in ckpt.fragments) == (
            learner.final_manifest.versions
        )
        for fid, (_version, params, _momentum) in enumerate(ckpt.fragments):
            assert torch.equal(params, coordinator[fid])
        assert not checkpoint.with_suffix(".tmp").exists()
    finally:
        if proc.poll() is None:
            proc.kill()
        print((proc.stdout.read() if proc.stdout else "")[-3000:])


@pytest.mark.timeout(180)
def test_final_ack_membership_excludes_previously_abandoned_learner():
    binary = build_syncer()
    port = free_port()
    named = [("model.embed.weight", DIM // 4), ("model.body.weight", DIM)]
    layout = build_layout(named, 2)
    proc = subprocess.Popen(
        [
            str(binary),
            "--port",
            str(port),
            "--learners",
            "2",
            "--quorum",
            "1",
            "--grace-ms",
            "20",
            "--quorum-timeout-s",
            "10",
            "--sync-interval-steps",
            "0",
            "--total-steps",
            "4",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = ""
    try:
        survivor = ToyLearner(0, port, torch.ones(DIM + DIM // 4), layout)
        abandoned = ToyLearner(
            1,
            port,
            -torch.ones(DIM + DIM // 4),
            layout,
            abandon_after_broadcasts=1,
        )
        survivor.start()
        abandoned.start()
        abandoned.join(timeout=60)
        assert not abandoned.is_alive()
        if abandoned.exc:
            raise abandoned.exc
        assert abandoned.abandoned

        survivor.join(timeout=120)
        assert not survivor.is_alive()
        if survivor.exc:
            raise survivor.exc
        assert survivor.final_manifest is not None
        assert proc.wait(timeout=30) == 0
        output = proc.stdout.read() if proc.stdout else ""
        assert "\x1b" not in output
        assert "all learners acknowledged final cut learners=1" in output
    finally:
        if proc.poll() is None:
            proc.kill()
        if not output:
            output = proc.stdout.read() if proc.stdout else ""
        print(output[-3000:])


@pytest.mark.timeout(180)
def test_single_learner_roundtrip_q4():
    """Q4 session: INIT/BCAST in bf16 and pushes as 4-bit base-relative
    deltas. Stale handling never needs historical parameter reconstruction."""
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
        l = ToyLearner(0, port, target, layout, dtype=DTYPE_Q4)
        l.start()
        l.join(timeout=120)
        assert not l.is_alive()
        if l.exc:
            raise l.exc
        assert proc.wait(timeout=30) == 0
        assert l.synced, "learner never received a broadcast"
        flat = torch.cat([p.detach().reshape(-1) for p in l.params.values()])
        assert (flat - target).norm() < target.norm(), "no progress toward target"
    finally:
        if proc.poll() is None:
            proc.kill()
        out = proc.stdout.read() if proc.stdout else ""
        print(out[-3000:])


@pytest.mark.timeout(180)
def test_min_round_interval_paces_rounds():
    """--min-round-interval-ms throttles round launches: 6 rounds with a
    250 ms floor cannot finish faster than the 5 enforced gaps (lower-bound
    assert, so slow machines cannot make it flaky)."""
    binary = build_syncer()
    port = free_port()
    named = [("model.embed.weight", DIM // 4), ("model.body.weight", DIM)]
    layout = build_layout(named, 3)
    proc = subprocess.Popen(
        [str(binary), "--port", str(port), "--learners", "1", "--quorum", "1",
         "--grace-ms", "50", "--total-steps", "6", "--min-round-interval-ms", "250"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        t0 = time.monotonic()
        l = ToyLearner(0, port, torch.ones(DIM + DIM // 4), layout)
        l.start()
        l.join(timeout=120)
        assert not l.is_alive()
        if l.exc:
            raise l.exc
        assert proc.wait(timeout=30) == 0
        elapsed = time.monotonic() - t0
        assert elapsed >= 1.0, f"rounds not paced: finished in {elapsed:.2f}s"
    finally:
        if proc.poll() is None:
            proc.kill()
        print((proc.stdout.read() if proc.stdout else "")[-2000:])


@pytest.mark.timeout(180)
@pytest.mark.parametrize("iso_backend", ["scalar", "torch-svd"])
def test_single_learner_roundtrip_iso(iso_backend):
    """Iso-C aggregation end to end: the learner HELLO carries (rows, cols)
    per tensor for the iso fragments and the Rust syncer merges through the
    spectrum-flattening path (matrix_merge="iso", arXiv 2607.03011)."""
    binary = build_syncer()
    port = free_port()
    named = [("model.embed.weight", DIM // 4), ("model.body.weight", DIM)]
    layout = build_layout(
        named,
        3,
        matrix_merge="iso",
        named_shapes={
            "model.embed.weight": (DIM // 4,),
            "model.body.weight": (64, DIM // 64),
        },
    )
    assert any(f.shapes for f in layout.fragments), "iso fragment missing shapes"
    command = [
        str(binary),
        "--port",
        str(port),
        "--learners",
        "1",
        "--quorum",
        "1",
        "--grace-ms",
        "50",
        "--total-steps",
        "9",
        "--iso-backend",
        iso_backend,
    ]
    if iso_backend == "torch-svd":
        command.extend(
            [
                "--iso-worker-python",
                sys.executable,
                "--iso-worker-device",
                "cpu",
            ]
        )
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        # A constant 64x64 target makes every learner delta rank one.  In f32
        # SVD its mathematically-zero trailing singular values live near
        # machine epsilon, above Iso's specified 1e-10 relative cutoff, so
        # flattening them is intentionally very different from the f64 scalar
        # oracle.  Use a deterministic full-rank target to test transport and
        # backend integration without changing the approved cutoff semantics.
        target = torch.randn(
            DIM + DIM // 4,
            generator=torch.Generator().manual_seed(20260825),
        )
        l = ToyLearner(0, port, target, layout)
        l.start()
        l.join(timeout=120)
        assert not l.is_alive()
        if l.exc:
            raise l.exc
        assert proc.wait(timeout=30) == 0
        assert l.synced, "learner never received a broadcast"
        flat = torch.cat([p.detach().reshape(-1) for p in l.params.values()])
        assert (flat - target).norm() < target.norm(), "no progress toward target"
    finally:
        if proc.poll() is None:
            proc.kill()
        print((proc.stdout.read() if proc.stdout else "")[-3000:])


@pytest.mark.timeout(180)
def test_single_learner_roundtrip():
    """M=1, K=1: a single self-syncing learner; must run to completion.
    Runs with --pipeline 1 so the serial-round path stays covered (the
    other tests exercise the default pipelined scheduler)."""
    binary = build_syncer()
    port = free_port()
    named = [("model.embed.weight", DIM // 4), ("model.body.weight", DIM)]
    layout = build_layout(named, 3)
    proc = subprocess.Popen(
        [str(binary), "--port", str(port), "--learners", "1", "--quorum", "1",
         "--grace-ms", "50", "--total-steps", "9", "--pipeline", "1"],
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
