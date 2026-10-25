"""Launcher side of peer recovery (S27 E-3, ``--rl-peer-recovery``).

Full-parameter asynchronous rounds only, with a fixed roster or an island
pool. The pool decides which island fills a slot; this option decides the
state source of that island (``--rl-island-state-source peer``; off is
``catch-up``: every fragment from the syncer, fresh optimizer state). With the
option on:

* every role syncer gets ``--syncer-epoch`` and the recovery options
  (:func:`yeto.launch.syncer.peer_recovery_syncer_flags`);
* every island gets ``--rl-syncer-epoch`` and the snapshot store
  (``--rl-peer-recovery-store``: the given path, else the run's
  ``--rl-checkpoint-store`` path on the island, so the store selection of the
  checkpoint store applies: shared path, bucket mount, Modal Volume, OSS);
* an island the launcher relaunches after a failure gets
  :data:`~yeto.rl.adapters.miles.island_entry.PEER_RECOVER_ENV` and recovers
  from a peer instead of starting from the initial weights.

With the option off every command is unchanged.
"""

from __future__ import annotations

import shlex

PEER_RECOVER_ENV = "YETO_RL_PEER_RECOVER"  # read by the island entry (same name there)


def peer_recovery_enabled(args) -> bool:
    return bool(getattr(args, "rl_peer_recovery", False))


def peer_recovery_store(args) -> str | None:
    """The snapshot store path on the island, or None when recovery is off."""
    return snapshot_store_path(args) if peer_recovery_enabled(args) else None


def snapshot_store_path(args) -> str | None:
    """The snapshot store path the island would use, whether or not recovery is on:
    ``--rl-peer-recovery-store`` mapped to the island's mount like every other store
    path (:func:`yeto.launch.storage.island_store_path`; raises ValueError when the
    island cannot reach it), else the ``--rl-checkpoint-store`` path on the island."""
    from .storage import island_store_path, rl_checkpoint_store_plan

    explicit = getattr(args, "rl_peer_recovery_store", None)
    if explicit:
        return island_store_path(args, str(explicit), flag="--rl-peer-recovery-store")
    plan = rl_checkpoint_store_plan(args)
    return None if plan is None else plan[0]


def check_peer_recovery(args, clouds=None) -> None:
    """Refuse ``--rl-peer-recovery`` options that do not apply. Every recovery
    option without ``--rl-peer-recovery`` is refused too. ``clouds``: the island
    clouds; on Modal the snapshot store must be on the mounted Volume."""
    from .syncer import rl_full_parameter_async, syncer_shard_count

    options = [flag for dest, flag in (
        ("rl_recovery_bound_steps", "--rl-recovery-bound-steps"),
        ("rl_recovery_max_donor_tries", "--rl-recovery-max-donor-tries"),
        ("rl_recovery_donor_timeout_s", "--rl-recovery-donor-timeout-s"),
        ("rl_peer_recovery_store", "--rl-peer-recovery-store"),
        ("rl_peer_recovery_ttl_s", "--rl-peer-recovery-ttl-s"),
        ("rl_peer_recovery_manifest_wait_s", "--rl-peer-recovery-manifest-wait-s"),
        ("rl_peer_recovery_donor_write", "--rl-peer-recovery-donor-write"),
    ) if getattr(args, dest, None) is not None]
    if not peer_recovery_enabled(args):
        if options:
            raise ValueError(", ".join(options) + " need --rl-peer-recovery")
        return
    if not rl_full_parameter_async(args):
        raise ValueError("--rl-peer-recovery needs --rl-sync-preset full-parameter with "
                         "--rl-sync-full-parameter-rounds async")
    if syncer_shard_count(args) > 1:
        # Recovery is one exchange with one syncer; shards would each need it.
        raise ValueError("--rl-peer-recovery needs one syncer process (--syncer-shards 1)")
    if getattr(args, "rl_sync_bcast_delta_full_every", 0):
        raise ValueError("--rl-peer-recovery resends whole fragments: it excludes broadcast deltas")
    store = peer_recovery_store(args)  # raises when the island cannot reach the explicit store
    if store is None:
        raise ValueError("--rl-peer-recovery needs --rl-peer-recovery-store or --rl-checkpoint-store "
                         "(a store every island reaches)")
    if clouds and "modal" in clouds and not getattr(args, "rl_checkpoint_store", None):
        # A Modal island mounts no store without --rl-checkpoint-store
        # modal-volume://... (the launch refuses any other store form on Modal), so
        # every path is local to one container. With a Volume store,
        # island_store_path already keeps the path under its mount.
        raise ValueError(f"--rl-peer-recovery snapshot store {store!r} on Modal islands: without "
                         "--rl-checkpoint-store modal-volume://NAME[/PREFIX] no Volume is mounted and "
                         "a container path is local to one container")
    for dest, flag in (("rl_recovery_bound_steps", "--rl-recovery-bound-steps"),
                       ("rl_recovery_max_donor_tries", "--rl-recovery-max-donor-tries"),
                       ("rl_recovery_donor_timeout_s", "--rl-recovery-donor-timeout-s")):
        value = getattr(args, dest, None)
        if value is not None and value < 1:
            raise ValueError(f"{flag} must be >= 1")
    ttl = getattr(args, "rl_peer_recovery_ttl_s", None)
    if ttl is not None and not ttl > 0:
        raise ValueError("--rl-peer-recovery-ttl-s must be positive")
    wait = getattr(args, "rl_peer_recovery_manifest_wait_s", None)
    if wait is not None and not wait >= 0:
        raise ValueError("--rl-peer-recovery-manifest-wait-s must be >= 0")
    if getattr(args, "rl_syncer_epoch", None) is not None and args.rl_syncer_epoch < 0:
        raise ValueError("--rl-syncer-epoch must be >= 0")


def island_flags(args) -> str:
    """Island flags; "" when recovery is off."""
    store = peer_recovery_store(args)
    if store is None:
        return ""
    from .syncer import island_pool, island_syncer_epoch

    # An island pool already sends the epoch with its membership flags.
    flags = "" if island_pool(args) else f" --rl-syncer-epoch {island_syncer_epoch(args)}"
    flags += f" --rl-peer-recovery-store {shlex.quote(store)}"
    ttl = getattr(args, "rl_peer_recovery_ttl_s", None)
    if ttl is not None:
        flags += f" --rl-peer-recovery-ttl-s {float(ttl)!r}"
    wait = getattr(args, "rl_peer_recovery_manifest_wait_s", None)
    if wait is not None:
        flags += f" --rl-peer-recovery-manifest-wait-s {float(wait)!r}"
    write = getattr(args, "rl_peer_recovery_donor_write", None)
    if write is not None:
        flags += f" --rl-peer-recovery-donor-write {write}"
    return flags


def relaunch_envs(args) -> dict[str, str]:
    """Environment of a relaunched island: recover from a peer."""
    return {PEER_RECOVER_ENV: "1"} if peer_recovery_enabled(args) else {}
