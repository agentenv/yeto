"""Spot rollout islands on any cloud (S27 G-EA).

A spot (preemptible) island is only a rollout accelerator: it never trains.
It needs two links to the head, both on the syncer's port:

* weights: a receive-only syncer subscriber (:mod:`.subscriber`; syncer
  ``--max-subscribers``) gets every merged fragment version as bf16. It is
  never in the roster, quorum, lease or final cut, so its loss never
  touches a merge;
* samples: the sample relay (:mod:`.relay`, :mod:`.relay_buffer`; syncer
  ``--sample-relay``) carries its groups to a training island, which moves
  them into its own local sample buffer and reads them with the existing
  buffer reader (policy age gate, TIS log-probs unchanged).

The training island publishes its fragment layout on a relay board
(:mod:`.layout_manifest`); the sampler turns fragments into a HF directory
(:mod:`.materialize`) and loads it into its engine (:mod:`.sampler`).
Everything is off unless the syncer flags and the island settings are given.
"""
