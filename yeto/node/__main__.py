"""``python -m yeto.node run --assignment <ticket> --store <run store>`` (:mod:`.runtime`);
``python -m yeto.node revoke --assignment <ticket> --store <run store>``: the control
plane's tombstone for an assignment (an unreadable one included) that keeps its generation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .runtime import add_run_args, run_cli


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m yeto.node", description="the yeto sampler node runtime")
    sub = parser.add_subparsers(dest="command", required=True)
    add_run_args(sub.add_parser("run", help="start this node's assigned sampler and report it in the store"))
    revoke = sub.add_parser("revoke", help="replace a ticket's assignment with a tombstone that keeps its generation")
    revoke.add_argument("--assignment", required=True, help="the ticket, or assignments/<ticket>.json")
    revoke.add_argument("--store", required=True, help="the run store: s3://BUCKET/runs/PREFIX, r2://..., or a directory")
    revoke.add_argument("--env-file", help="the store env file (AWS_*, R2_BUCKET)")
    args = parser.parse_args(argv)
    if args.command == "revoke":
        from .records import revoke_assignment, ticket_of_key  # noqa: PLC0415
        from .store import open_objects  # noqa: PLC0415

        ticket = ticket_of_key(args.assignment)
        revoke_assignment(open_objects(args.store, env_file=args.env_file), ticket)
        print(f"[yeto-node] assignment of {ticket} revoked (tombstone; the next assignment continues its generations)")
        return 0
    return run_cli(args)


if __name__ == "__main__":
    sys.exit(main())
