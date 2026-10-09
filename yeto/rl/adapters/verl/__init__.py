"""verl backend adapter (rl-verl-backend; design D1-D11).

First step: verl FSDP2 + vLLM on NVIDIA, user fork pinned in ``pins``.  All
``verl`` / ``vllm`` imports live in the modules that run inside the verl image
(``verl_main``, ``trainer``, ``vllm_readback``); everything else is pure and
CPU-testable on a machine without verl.

Modules:

* ``pins``           fork commit, image recipe constants, version expectations
* ``identity``       BackendIdentity of this adapter (decoupling phase 5)
* ``param_names``    FSDP2/PEFT and vLLM LoRA names <-> yeto canonical names (1.5)
* ``config``         yeto run options -> checked verl hydra overrides (1.3)
* ``binding``        neutral name -> verl config key / function, three-way check (4.4a style)
* ``publish``        LoRA publication checksum + memory/disk transport switch (1.10)
* ``image``          Modal image recipe for the pinned fork (1.1)
* ``island_entry``   island process: data/model prep, runs ``verl_main``
* ``verl_main``      verl hydra entry with yeto's task runner (inside the image)
* ``trainer``        verl v1 sync trainer subclass + IslandRuntime (inside the image)
* ``vllm_readback``  runs in the vLLM worker: registered-adapter checksum (inside the image)
* ``reward_fn``      verl ``compute_score`` wrapper over yeto reward functions
* ``launch_flags`` / ``run_config_rules`` / ``rollout_meta_hook`` / ``entry``: registry roles
"""
