"""The node runtime: one bootstrap for a sampler on any node (unified-sampler-pool D5).

``python -m yeto.node run --assignment <ticket>`` (:mod:`.runtime`) reads the
node's assignment from the run store with the ticket's token, runs the
preflight, fetches the yeto tree by hash, prepares the model, fills the spec's
machine paths from the node profile and starts the engine sampler; the node's
``ready`` / ``heartbeat`` / ``exit`` records go to the store (:mod:`.records`).
"""
