"""Provider-neutral rollout island pool.

* :mod:`.status` - island status, node probe, verdicts, breaker, budget, events;
* :mod:`.provider` - the :class:`~.provider.IslandProvider` contract (machines
  only), :class:`~.provider.MultiProvider` (several providers in one pool) and
  the provider registry (factories, ledger kinds for cleanup and billing);
* :mod:`.island_pool` - :class:`~.island_pool.IslandPool`, the policy: target,
  scale control file, evict -> fence -> claim release -> refill,
  ``continues_shard`` handoffs, lease-end replacement, breaker, adoption;
* :mod:`.controller` - :class:`~.controller.RolloutController`, the pools run
  in the launcher's controller process next to the trainer fleet (shared STOP
  flag, budget, event tape and cleanup).
"""
