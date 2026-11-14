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

This module is that one slot.  :func:`setup_worker` imports each hook named in
:data:`HOOKS_ENV` (comma-separated dotted ``module:function`` or module names)
and calls it; a module name alone is imported for its import side effect.  A
failing hook is logged and skipped -- a worker that cannot register an
optional optimisation is still a usable worker -- except that the hooks are
run in the order given, so an ordering requirement is the caller's to express.
"""

from __future__ import annotations

import importlib
import os
import sys

HOOKS_ENV = "YETO_WORKER_SETUP_HOOKS"  # forwarded to Ray workers by the YETO_ prefix
SETUP_HOOK = "yeto.rl.worker_setup.setup_worker"


def add_hook(env_vars: dict, hook: str) -> None:
    """Append ``hook`` to the chain in ``env_vars`` (and keep the process in step).

    ``env_vars`` is the runtime env's own mapping, which Ray hands to the
    worker; the same value is written back into this process's environment so a
    later caller that reads ``os.environ`` sees the hooks already registered.
    """
    chain = [h for h in (env_vars.get(HOOKS_ENV) or os.environ.get(HOOKS_ENV) or "").split(",") if h]
    if hook not in chain:
        chain.append(hook)
    env_vars[HOOKS_ENV] = ",".join(chain)
    os.environ[HOOKS_ENV] = env_vars[HOOKS_ENV]


def setup_worker() -> None:
    for hook in (os.environ.get(HOOKS_ENV) or "").split(","):
        hook = hook.strip()
        if not hook:
            continue
        try:
            module, _, attr = hook.partition(":")
            obj = importlib.import_module(module)
            if attr:
                getattr(obj, attr)()
        except Exception as exc:  # pragma: no cover - a worker keeps running without it
            print(f"[yeto.worker_setup] hook {hook} failed: {exc!r}", file=sys.stderr, flush=True)
