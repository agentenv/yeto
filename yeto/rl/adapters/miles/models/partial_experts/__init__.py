"""Model-independent contract for training a subset of a MoE model's routed experts.

The package factors out what the DeepSeek V4 clone-expert runtime
(:mod:`..deepseek_v4`) and the Kimi-K3.1 in-place permuted runtime
(:mod:`..kimi_k31`) share, without any model constant:

* :mod:`.config` -- the per-model declaration (:class:`PartialExpertModelConfig`)
  and its registry; it decides, among other things, whether the Megatron
  distributed optimizer is allowed;
* :mod:`.attest` -- selection-file identity: file / canonical-JSON SHA256 and
  SHA256 string validation;
* :mod:`.freeze` -- per-parameter freezing of expert weights with marks and
  double-configuration detection;
* :mod:`.optimizer` -- ``ParamKey`` learning-rate groups for the trainable
  experts and the FP32 optimizer-master validation;
* :mod:`.export` -- filtering of HF export tasks / named weights down to the
  trainable experts;
* :mod:`.hooks` -- the import hook that patches a Miles module once it is
  imported (``install_or_defer``), shared by every model runtime;
* :mod:`.selection` -- the ``expert_selection.json`` generator (INTERFACES 2.3).

Nothing here imports torch, megatron or miles at module import time.
"""
