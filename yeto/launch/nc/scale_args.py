"""The flags of ``python -m yeto.launch.nc pool-scale`` and ``yeto run scale``
(no imports: ``yeto``'s parser builds them at every start)."""

from __future__ import annotations

from typing import Any


def add_scale_args(p: Any, *, target_flag: str) -> None:
    """The flags of ``pool-scale`` and ``yeto run scale`` (one of target / --add / --remove)."""
    how = p.add_mutually_exclusive_group(required=True)
    how.add_argument(target_flag, dest="target", type=int, help="samplers wanted (0 or more)")
    how.add_argument("--add", type=int, default=0, metavar="K", help="K more samplers than the current target")
    how.add_argument("--remove", action="append", default=[], metavar="TICKET",
                     help="release this node and lower the target by one (repeatable)")
    p.add_argument("--retire", action="append", default=[], metavar="TICKET",
                   help="with a lower target: the node ticket to release (repeatable; default: a node not "
                   "ready yet, else the newest, cheapest, shortest lease)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--drain", dest="mode", action="store_const", const="drain",
                      help="scale down: removed nodes claim no new task and finish their trials in flight first")
    mode.add_argument("--fence", dest="mode", action="store_const", const="fence",
                      help="scale down: fence removed nodes at once (their tasks go back to the others)")
    p.set_defaults(mode=None)
    p.add_argument("--force-static", action="store_true",
                   help="allow a scale with static data_slots (a removed node's slot pauses)")
    p.add_argument("--dry-run", action="store_true", help="check the request only; nothing changes")
    p.add_argument("--source", choices=("manual", "auto"), default="manual",
                   help="who asks: a person (default) or an automatic scaler; an automatic request is refused "
                   "for elastic.manual_hold_s after a manual one")
