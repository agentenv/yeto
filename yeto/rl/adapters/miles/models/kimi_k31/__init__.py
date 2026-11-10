"""Kimi-K3.1 (``moonshotai/Kimi-K3.1-Pro``) on ports Miles: 128 of 896 routed
experts per MoE layer trained in place through an offline slot permutation.

* :mod:`.model` -- checkpoint constants, the partial-experts contract and the
  HF config check;
* :mod:`.ports_recipe` -- the Miles argv of the recipe (raw bridge mode,
  frozen router bias, the selection file, no VPP, zigzag CP constraint);
* :mod:`.release` -- release-manifest fields (``expert_selection_sha256``);
* :mod:`.provider` -- the read-only provider view of the checkpoint's
  ``config.json`` (AutoBridge cannot read kimi_k3 / kimi_k31);
* :mod:`.ports_runtime` -- the one trainer-side patch yeto adds (optimizer
  master validation that allows the distributed optimizer).

Freezing, slot permutation, export filtering and routing replay live in the
Miles fork (``--partial-experts-selection``, F2); the engine runs the
unmodified checkpoint and needs no patch.
"""
