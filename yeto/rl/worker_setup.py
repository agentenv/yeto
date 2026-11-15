"""One Ray ``worker_process_setup_hook`` that runs several yeto hooks.

Ray takes exactly one ``worker_process_setup_hook`` per runtime env, and yeto
already had two candidates for that slot -- the Ascend plugin stack
(``ascend_plugin_stack.SETUP_HOOK``) and the agentic advantage registration
(``adapters.verl.agentic_main.SETUP_HOOK``) -- so an agentic island on Ascend
was refused outright (``adapters/verl/verl_main.py``: "a verl island cannot
also install the Ascend plugin hook").  A third hook is now needed as well: a
trainer island built from nodes without RoCE has to install the cross-node
gloo shim in every worker *before* the worker builds its process groups, and
the shim only installs itself at import time.

This module is that one slot.  :func:`setup_worker` resolves each hook named in
:data:`HOOKS_ENV` (comma-separated) exactly the way Ray resolves a
``worker_process_setup_hook`` string -- ``module:function`` or the dotted
``module.function`` that the existing yeto hooks use -- and calls it, in the
order given.  A hook that fails to import or raises fails the worker, as it
would have when Ray called that hook directly: the plugin stack and the
``yeto_grpo`` registration are not optional.

The chain is only used when the cross-node gloo shim is switched on
(``YETO_CROSS_NODE_GLOO=1``); without it the runtime envs set a single hook
exactly as before.
"""

from __future__ import annotations

import importlib
import os
import sys

HOOKS_ENV = "YETO_WORKER_SETUP_HOOKS"  # forwarded to Ray workers by the YETO_ prefix
SETUP_HOOK = "yeto.rl.worker_setup.setup_worker"


def add_hook(env_vars: dict, hook: str) -> None:
    """Append ``hook`` to the chain in ``env_vars`` (the runtime env's own mapping).

    The process environment is left alone, so a second runtime env built in the
    same driver does not inherit this one's chain.
    """
    chain = [h for h in (env_vars.get(HOOKS_ENV) or "").split(",") if h]
    if hook not in chain:
        chain.append(hook)
    env_vars[HOOKS_ENV] = ",".join(chain)


def resolve(hook: str):
    """The callable named by ``hook``, as Ray's ``import_attr`` resolves it.

    ``module:attr`` splits at the colon; otherwise the name splits at its last
    dot (``yeto.rl.ascend_plugin_stack.setup_worker``).  Errors propagate.
    """
    if ":" in hook:
        module, _, attr = hook.partition(":")
    else:
        module, _, attr = hook.rpartition(".")
    if not module or not attr:
        raise ValueError(f"[yeto.worker_setup] hook {hook!r} is not 'module.attr' or 'module:attr'")
    return getattr(importlib.import_module(module), attr)


def setup_worker() -> None:
    for hook in (os.environ.get(HOOKS_ENV) or "").split(","):
        hook = hook.strip()
        if not hook:
            continue
        try:
            fn = resolve(hook)
        except Exception as exc:
            print(f"[yeto.worker_setup] hook {hook} failed to import: {exc!r}", file=sys.stderr, flush=True)
            raise
        fn()
