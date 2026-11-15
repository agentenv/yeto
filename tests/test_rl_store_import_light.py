"""The store-only path imports without torch: a host whose venv has no torch (an
Ascend trainer host that only reads and writes the object store, a head-side tool)
can still use the sample buffer, the dispense ledger and policy releases."""

from __future__ import annotations

import subprocess
import sys
import textwrap

PROBE = textwrap.dedent(
    """
    import builtins, sys

    real = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError("No module named 'torch'")
        return real(name, *args, **kwargs)

    builtins.__import__ = no_torch
    import yeto.cloud.s3_store  # noqa: F401
    import yeto.rl.engine.sample_buffer  # noqa: F401
    from yeto.rl.leave_code import LeaveReason
    from yeto.rl.engine.leave_reason import leave_reason_for

    assert leave_reason_for(KeyboardInterrupt()) is LeaveReason.STOPPED
    assert "torch" not in sys.modules
    print("ok")
    """
)


def test_sample_buffer_and_s3_store_import_without_torch() -> None:
    done = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip().endswith("ok")


def test_pool_client_reexports_leave_reason() -> None:
    from yeto.rl.leave_code import LeaveReason
    from yeto.rl.pool_client import LeaveReason as FromPoolClient

    assert FromPoolClient is LeaveReason
