"""Turn each BF16 policy release into a W4A8 release for W4A8 samplers (S25 N7.3 + N8).

The trainer releases BF16 weights into ``policy-releases/`` (``ReleasingPublisher``).
:class:`RequantReleaser` follows that directory and, for each new BF16 release,
writes the re-quantized checkpoint (:func:`yeto.rl.quant.requant.requantize_checkpoint`)
as a release in ``policy-releases-w4a8/`` with the same commit order (weights,
``RELEASE.json``, then ``LATEST.json``). A W4A8 sampler follows that directory with
the usual ``PolicyFollower``; it never sees a partial W4A8 checkpoint.

The W4A8 record keeps the BF16 policy version (the version the sampler stamps on
its samples), the trainer's buffer positions, and ``derived_from`` (the BF16
release's directory, seq, version and hash, the scheme and the AWQ source). Its
policy hash is the BF16 hash plus the scheme, so a restarted releaser that sees
the same BF16 release writes nothing.

Where it runs: as a side process on the first trainer node
(``yeto.launch.ssh_trainer_island``), which reads the BF16 release from the
shared store. Per-rank quantization of the trainer's own shards (N8.2) would
replace the whole-checkpoint read; not done.

Status: CPU tests with a fake re-quantizer and with the real converter on a tiny
layout (tests/test_rl_quant_release.py); not run on a real release.

CLI (one pass or a loop)::

    python -m yeto.rl.quant.release --store DIR_OR_URI --template TEMPLATE_DIR \\
        (--recipe recipe.yaml --awq-scales DIR | --no-awq) [--loop --poll-seconds 10]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from yeto.rl.engine.policy_release import (
    FULL,
    RELEASE_DIR,
    W4A8_RELEASE_DIR,
    PolicyRelease,
    PolicyReleaseError,
    PolicyReleaseStore,
    ReleaseIncomplete,
)

SCHEME = "w4a8-int4-channel"

Requantize = Callable[[Path, Path], Any]  # (bf16 directory, output directory) -> stats


class RequantReleaser:
    """Writes one W4A8 release per BF16 release (module doc).

    ``requantize(source_dir, out_dir)`` writes a complete loadable checkpoint
    into ``out_dir`` (default: :func:`requantizer` over a template)."""

    def __init__(self, source: PolicyReleaseStore, target: PolicyReleaseStore, requantize: Requantize,
                 *, scheme: str = SCHEME, awq: str = "none",
                 log: Callable[[str], None] = lambda m: print(m, flush=True)) -> None:
        if source.release_dir == target.release_dir:
            raise ValueError(f"source and target are the same release directory {source.release_dir!r}")
        self.source = source
        self.target = target
        self.requantize = requantize
        self.scheme = scheme
        self.awq = awq
        self.log = log
        self.waiting: str | None = None  # why the newest BF16 release is not converted yet

    def done_for(self, release: PolicyRelease) -> bool:
        latest = self.target.latest()
        return (latest is not None and latest.derived_from is not None
                and latest.derived_from.get("policy_hash") == release.policy_hash
                and latest.version == release.version)

    def sync(self) -> PolicyRelease | None:
        """Convert the newest BF16 release once; returns the W4A8 release written,
        or None (no BF16 release, already converted, or not complete yet)."""
        src = self.source.latest()
        if src is None or self.done_for(src):
            return None
        if src.weights != FULL:
            raise PolicyReleaseError(f"release r{src.seq}: {src.weights!r} weights cannot be re-quantized "
                                     "(full-parameter BF16 releases only)")
        try:
            directory = self.source.resolve(src)
        except ReleaseIncomplete as error:
            self.waiting = str(error)
            return None
        self.waiting = None
        derived = {"release_dir": self.source.release_dir, "seq": src.seq, "version": src.version,
                   "policy_hash": src.policy_hash, "scheme": self.scheme, "awq": self.awq}
        stats: dict[str, Any] = {}

        def write(out: Path) -> None:
            stats["requant"] = self.requantize(directory, out)

        record = self.target.release(src.version, f"{src.policy_hash}+{self.scheme}", write,
                                     buffer_selected=src.buffer_selected, buffer_shards=src.buffer_shards,
                                     derived_from=derived)
        if record is not None:
            self.log(f"[requant] BF16 r{src.seq}-v{src.version} -> {self.target.release_dir}/{record.path} "
                     f"{json.dumps(stats.get('requant'), default=str)}")
        return record

    def run(self, stop: threading.Event, poll_s: float = 10.0) -> None:
        """:meth:`sync` until ``stop`` is set (a failed conversion ends the loop)."""
        while not stop.is_set():
            self.sync()
            stop.wait(poll_s)


def requantizer(template_dir: Path, smoothing: Any) -> Requantize:
    """The converter of :mod:`.requant` bound to one template and AWQ smoothing."""
    from .requant import requantize_checkpoint

    def run(source: Path, out: Path) -> Any:
        return requantize_checkpoint(Path(source), Path(template_dir), Path(out), smoothing)

    return run


def main(argv: list[str] | None = None) -> int:
    from yeto.rl.engine.resume import store_for

    from .awq import AwqSmoothing

    p = argparse.ArgumentParser(prog="python -m yeto.rl.quant.release", description=__doc__.splitlines()[0])
    p.add_argument("--store", required=True, help="the run's checkpoint store (as the trainer island mounts it)")
    p.add_argument("--template", required=True)
    p.add_argument("--recipe")
    p.add_argument("--awq-scales")
    p.add_argument("--no-awq", action="store_true")
    p.add_argument("--source-dir", default=RELEASE_DIR)
    p.add_argument("--target-dir", default=W4A8_RELEASE_DIR)
    p.add_argument("--keep", type=int, default=2)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--poll-seconds", type=float, default=10.0)
    a = p.parse_args(argv)
    if a.no_awq:
        smoothing, awq = AwqSmoothing.none(), "none"
    elif a.recipe and a.awq_scales:
        smoothing, awq = AwqSmoothing.from_artefact(Path(a.recipe), Path(a.awq_scales)), "fixed-artefact"
    else:
        p.error("give --recipe and --awq-scales (option R2) or --no-awq (option R1)")
    store = store_for(a.store)
    releaser = RequantReleaser(PolicyReleaseStore(store, keep=a.keep, release_dir=a.source_dir),
                               PolicyReleaseStore(store, keep=a.keep, release_dir=a.target_dir),
                               requantizer(Path(a.template), smoothing), awq=awq)
    if a.loop:
        releaser.run(threading.Event(), a.poll_seconds)
    else:
        releaser.sync()
    return 0


if __name__ == "__main__":
    sys.exit(main())
