"""CPU end-to-end simulation of the shared object store of an async GRPO run.

The real yeto modules of the sampler, trainer and control plane run against a
fake boto3 S3 client that injects faults (:mod:`.fault_s3`) under a seeded
scheduler (:mod:`.sim`); only the GPU engine and the Miles training loop are
stubs (:mod:`.actors`). ``python -m tests.store_sim.run --help`` runs seeds at
scale; ``tests/test_store_fault_sim.py`` is the regression suite.
"""
