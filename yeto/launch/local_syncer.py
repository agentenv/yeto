"""The syncer run as a local process on the head (``LocalSyncer``)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time

from .syncer import syncer_command, syncer_event_tape
from .fleet import _RlStrictFailure


class LocalSyncer:
    """The syncer as a subprocess of the head node's controller job.

    In head controller mode the syncer binary is file-mounted onto the head
    VM and runs right next to the controller, instead of on a separate
    cluster. Its stdout/stderr go to a log file (appended across restarts)
    that a background thread forwards into this process's stdout with a
    "[syncer]" prefix, so the head job's log stream carries the syncer's
    output. `probe`/`restart` plug into FleetController: a dead subprocess
    is simply restarted, and --resume (already in the command line) makes
    it pick up from its on-disk checkpoint.
    """

    def __init__(
        self,
        args,
        num_learners: int,
        binary: str = "~/yeto-syncer",
        log_file: str = "~/yeto-syncer.log",
    ):
        self.command = syncer_command(args, num_learners, binary=binary)
        self.binary = os.path.expanduser(binary)
        self.log_file = os.path.expanduser(log_file)
        self.event_tape = os.path.expanduser(syncer_event_tape(args))
        self.proc: subprocess.Popen | None = None
        self.tape_forwarder = None
        self.tape_run = None
        # The log file persists across controller jobs on a reused head;
        # forward only what this controller's syncer writes, not history.
        self._log_offset = os.path.getsize(self.log_file) if os.path.exists(self.log_file) else 0
        self._event_offset = (
            os.path.getsize(self.event_tape) if os.path.exists(self.event_tape) else 0
        )

    def start(self) -> None:
        if os.path.exists(self.binary):
            os.chmod(self.binary, 0o755)
        log_f = open(self.log_file, "ab")
        try:
            # shell=True so the ~ paths in the command expand. The shell may
            # fork rather than exec the binary, so terminate() on the Popen
            # pid alone can orphan the syncer, which then holds the port
            # across controller restarts; start_new_session gives the whole
            # tree its own process group for stop() to kill.
            self.proc = subprocess.Popen(
                self.command,
                shell=True,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        finally:
            log_f.close()  # Popen holds its own duplicate of the fd
        print(f"[launcher] syncer subprocess started (pid {self.proc.pid})", flush=True)

    def probe(self) -> str | _RlStrictFailure | None:
        """None if the subprocess is healthy, else a reason string.

        Exit code 0 means the syncer completed its total steps — terminal
        success, not a failure to recover from (restarting would resume at
        the final step, instantly re-complete, and loop until the learners
        report done)."""
        if self.proc is None:
            return "syncer subprocess was never started"
        code = self.proc.poll()
        if code is None or code == 0:
            return None
        strict_failure = self._strict_failure()
        if strict_failure is not None:
            return strict_failure
        return f"syncer subprocess exited with code {code}"

    def finished(self) -> bool:
        """0.26: the syncer ended its run normally (exit 0, or its elastic
        status.json says state=finished)."""
        if self.proc is not None and self.proc.poll() == 0:
            return True
        from ..rl.engine.island_status import read_syncer_status

        status = read_syncer_status(self.event_tape)
        return bool(status) and status.get("state") == "finished"

    def _strict_failure(self) -> "_RlStrictFailure | None":
        try:
            with open(self.event_tape, encoding="utf-8") as handle:
                handle.seek(self._event_offset)
                for line in handle:
                    event = json.loads(line)
                    if event.get("event") == "rl_strict_failure":
                        return _RlStrictFailure(
                            f"{event.get('metric', 'strict_failure')}: "
                            f"{event.get('error', 'syncer strict invariant failed')}"
                        )
        except (OSError, ValueError, TypeError):
            return None
        return None

    def restart(self) -> None:
        self.start()

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self._signal_tree(signal.SIGTERM)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._signal_tree(signal.SIGKILL)
        if self.tape_forwarder is not None:
            self.tape_forwarder.stop()
            self.tape_forwarder = None
        if self.tape_run is not None:
            self.tape_run.finish()
            self.tape_run = None

    def start_tape_forwarder(self, args) -> None:
        """Tail the syncer's event tape into a W&B run.

        The syncer is Rust and stays that way: its merge records already
        carry quorum/grace timings, per-island staleness, and per-island
        contribution, so a reader on this host turns them into the fleet's
        syncer-side curves with no change to the hot path. Survives a
        syncer restart because the tape is appended to, not rewritten.
        """
        if not getattr(args, "wandb", False):
            return
        from .. import wandb_logger
        from ..wandb_tape import OffsetStore, TapeForwarder, offset_path_for

        run = wandb_logger.init(
            args,
            job_type="syncer",
            name="syncer",
            step_metrics={"sync/*": "global_step", "learner/*": "global_step"},
            config_extra={"syncer_command": self.command},
        )
        if not run.enabled:
            return
        self.tape_run = run
        self.tape_forwarder = TapeForwarder(
            run,
            self.event_tape,
            # A reused head keeps the previous controller job's tape; the
            # stored offset stops this one from re-logging its merges.
            offsets=OffsetStore(offset_path_for(self.event_tape)),
        )
        self.tape_forwarder.start()

    def _signal_tree(self, sig: int) -> None:
        """Signal the syncer's whole process group (see start()), falling
        back to the direct child."""
        try:
            os.killpg(self.proc.pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                self.proc.send_signal(sig)
            except OSError:
                pass

    def start_log_forwarder(self) -> None:
        """Tail the syncer's log file into our stdout, forever (daemon)."""

        def _forward():
            while not os.path.exists(self.log_file):
                time.sleep(0.5)
            with open(self.log_file, "r", errors="replace") as f:
                f.seek(self._log_offset)
                while True:
                    line = f.readline()
                    if line:
                        print(f"[syncer] {line.rstrip()}", flush=True)
                    else:
                        time.sleep(0.5)

        threading.Thread(target=_forward, daemon=True).start()
