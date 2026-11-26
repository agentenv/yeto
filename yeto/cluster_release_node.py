"""``yeto cluster release-node <name>``: give a quarantined spare-pool node back to the islands.

A resident ``ssh:`` trainer island with spare nodes (``roles: [trainer-spare]``) quarantines
a node it lost (no answer over SSH, or node health bad twice in a row) in the run store,
``fleet/nodes/<name>/quarantine.json`` (with a ``lease_store``: also in the lease store, where
every run's pick leaves it out). The node stays out of every pick until this command: it
shows the record, runs the node's health check (``--force`` skips it, with a WARN), runs the
stops the node still owes, gives up the run's cross-run lease (a live one only after a probe
found nothing of the run on the node) and removes the record.

Exit 0 released (or not quarantined), 1 the check, a stop or the lease release failed (the
record stays), 2 a usage or inventory error.
"""

from __future__ import annotations

import argparse
from typing import Any

COMMAND = "release-node"


def add_parser(csub: Any) -> argparse.ArgumentParser:
    """``release-node`` under the ``yeto cluster`` subcommands ``csub``."""
    doc = (__doc__ or "").split("\n\n")
    p = csub.add_parser(COMMAND, help="give a quarantined spare-pool node back to the islands",
                        description="\n\n".join(doc[1:]))
    p.add_argument("name", help="the node's name in the inventory")
    p.add_argument("--inventory", required=True, help="the SSH node inventory (YAML / JSON)")
    p.add_argument("--store", required=True,
                   help="the run store that holds the quarantine (the run's --rl-checkpoint-store)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the record and the stops that would run; no SSH, nothing removed")
    p.add_argument("--force", action="store_true", help="release without the node's health check")
    p.add_argument("--kill-grace-s", type=int, default=None,
                   help="seconds an owed stop waits after SIGTERM before SIGKILL")
    return p


def main(args: argparse.Namespace, **kw: Any) -> int:
    from .launch.island_spares import release_node_main

    return release_node_main(inventory=args.inventory, name=args.name, store=args.store,
                             dry_run=bool(args.dry_run), force=bool(args.force),
                             kill_grace_s=args.kill_grace_s, **kw)
