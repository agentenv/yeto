"""`yeto-provision-solve`: one bounded JSON request, one bounded envelope.

Independent AgentEnv entry point (also runnable as
``python -m yeto.provision.json_cli``). It deliberately bypasses ``yeto.cli``
and its eager Torch imports; it never launches Sky and adds no
apply/status/delete command — Sky lifecycle belongs to the AE Sky wrapper and
orchestration worker.

Protocol:

- stdin:  exactly one ``SolveProvisionRequestV1`` JSON object (bounded).
- stdout: exactly one process envelope (bounded):
  ``{"schemaVersion":1,"ok":true,"value":FleetPlanV1}`` or
  ``{"schemaVersion":1,"ok":false,"error":{code,retryable,message}}``.
- stderr: diagnostics only. No NDJSON/event stream.
- exit 0: typed success *and* handled domain errors (the parent validates the
  typed envelope).
- nonzero: usage, signal, bootstrap, or protocol failure — structured stdout
  is then untrusted and ignored by the parent.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

from .contracts import (
    ContractError,
    SolveProvisionRequestV1,
    dump_envelope,
    error_envelope,
    success_envelope,
)
from .solver import solve_provision

__all__ = ["main", "MAX_REQUEST_BYTES", "MAX_RESPONSE_BYTES"]

# One curated snapshot holds at most a handful of offerings; these bounds are
# intentionally far above any legitimate request/response.
MAX_REQUEST_BYTES = 1_048_576  # 1 MiB
MAX_RESPONSE_BYTES = 65_536  # 64 KiB

_USAGE = "usage: yeto-provision-solve < SolveProvisionRequestV1.json (no arguments)"

_EXIT_OK = 0
_EXIT_PROTOCOL = 2


def _protocol_failure(reason: str) -> int:
    print(f"yeto-provision-solve: {reason}", file=sys.stderr)
    return _EXIT_PROTOCOL


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    if args:
        return _protocol_failure(_USAGE)

    raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        return _protocol_failure(f"request exceeds {MAX_REQUEST_BYTES} bytes")
    if not raw.strip():
        return _protocol_failure("empty request")

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return _protocol_failure("request is not valid UTF-8")

    # Exactly one JSON object: trailing data or a second record is a protocol
    # failure, not a handled domain error.
    try:
        data = json.loads(text)
    except ValueError:
        return _protocol_failure("request is not exactly one JSON document")
    if not isinstance(data, dict):
        return _protocol_failure("request must be one JSON object")

    try:
        request = SolveProvisionRequestV1.from_dict(data)
        plan = solve_provision(request, now=datetime.now(timezone.utc))
    except ContractError as err:
        # Handled domain error: typed ok:false envelope, exit zero.
        envelope = error_envelope(err.to_process_error())
    else:
        # An echo mismatch means the solver is broken, not that the request was
        # bad: structured stdout is then untrusted, so fail closed as a protocol
        # error (nonzero, ignored by the parent) rather than a typed envelope.
        if plan.items[0].plan_item_id != request.plan_item_id:
            return _protocol_failure("solver did not echo planItemId unchanged")
        envelope = success_envelope(plan)

    out = dump_envelope(envelope)
    if len(out.encode("utf-8")) > MAX_RESPONSE_BYTES:
        return _protocol_failure(f"response exceeds {MAX_RESPONSE_BYTES} bytes")

    sys.stdout.write(out + "\n")
    sys.stdout.flush()
    return _EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
