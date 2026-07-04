"""NAVA multimodal fine-tuning: a self-contained yeto task package.

Everything NAVA-specific lives here — the DiLoCo learner (`learner`), the
Gemini label prep (`labels`), the MM-DiT LoRA patcher (`lora`), the S3/URI
helpers (`utils`), and the task-backend adapter (`backend`) that plugs into
yeto.backends. The generic sync core (fragments, protocol, syncer, launcher)
stays task-agnostic and is imported from the parent package.
"""
