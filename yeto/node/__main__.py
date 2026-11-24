"""``python -m yeto.node run --assignment <ticket> --store <run store>`` (:mod:`.runtime`)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .runtime import add_run_args, run_cli


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m yeto.node", description="the yeto sampler node runtime")
    sub = parser.add_subparsers(dest="command", required=True)
    add_run_args(sub.add_parser("run", help="start this node's assigned sampler and report it in the store"))
    args = parser.parse_args(argv)
    return run_cli(args)


if __name__ == "__main__":
    sys.exit(main())
