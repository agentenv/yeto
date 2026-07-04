"""Megatron-Core island backend: expert/tensor/pipeline parallelism for
LoRA fine-tuning MoE bases too large to shard with FSDP2.

A yeto.megatron.learner is a drop-in peer of yeto.learner — same DiLoCo
adapter sync to the Rust syncer, same LoRA/data/loss flags — but the frozen
base is distributed with Megatron-Core (experts sharded across EP ranks,
routed by all-to-all) instead of flat FSDP2 sharding. See docs/MEGATRON.md.
"""
