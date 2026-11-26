"""The store client environment of a launch: what the S3 store client of the launcher, the
islands and the syncer VM reads (``--rl-object-store-endpoint`` and the store profile
applied). Its own module, depending on nothing else in ``yeto.launch``: both
:mod:`.storage` and :mod:`.syncer` use it, and :mod:`.storage` imports :mod:`.syncer`.
"""

from __future__ import annotations

import os


def store_client_environ(args, environ=None) -> dict[str, str]:
    """The environment the S3 store client of the head and the islands reads, with
    ``--rl-object-store-endpoint`` applied: the endpoint, and ``region=auto`` when an
    endpoint is set and no region is configured (what Cloudflare R2 expects; AWS S3
    ignores it). ``environ`` defaults to the environment the launch was parsed for
    (``args._store_environ``: the on-site syncer service's, run up's), else the process's."""
    from ..cloud.s3_store import ENDPOINT_ENV, PROFILE_ENV, REGION_ENV, store_endpoint, store_region

    if environ is None:
        environ = getattr(args, "_store_environ", None)
    env = dict(os.environ if environ is None else environ)
    if getattr(args, "rl_object_store_profile", None):
        env[PROFILE_ENV] = str(args.rl_object_store_profile)
    endpoint = getattr(args, "rl_object_store_endpoint", None) or env.get("AWS_ENDPOINT_URL")
    if not endpoint and not env.get(ENDPOINT_ENV) and env.get(PROFILE_ENV):
        endpoint = store_endpoint(env)  # the profile's endpoint_url
        if endpoint and store_region(env):
            env.setdefault(REGION_ENV, store_region(env))
    if endpoint and not env.get(ENDPOINT_ENV):
        env[ENDPOINT_ENV] = str(endpoint)
        env.setdefault(REGION_ENV, env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION") or "auto")
    return env
