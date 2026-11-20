"""DeepSeek-V4.1-Flash (HF ``model_type`` ``deepseek_v41``) on ports Miles: LoRA only.

* :mod:`.provider` -- the read-only provider view of the checkpoint's
  ``config.json`` (AutoBridge cannot read ``deepseek_v41``; Miles trains it through
  its own plugin spec and mbridge converter);
* :mod:`.ports_recipe` -- the Miles argv of the recipe (Miles
  ``scripts/models/deepseek-v4.1.py`` model arguments and the
  ``scripts/run_deepseek_v4_1.py`` trainer flags), raw bridge mode, native LoRA on
  attention and the shared expert, routed experts / router / engram frozen.

The trainer loads a BF16 HF checkpoint directly (``--hf-checkpoint`` and
``--ref-load`` name the same directory, Miles ``--load-from-hf``).
"""
