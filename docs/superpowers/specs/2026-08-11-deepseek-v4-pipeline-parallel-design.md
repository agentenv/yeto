# DeepSeek V4 Pipeline-Parallel Training Design

## Goal

Allow the pinned DeepSeek V4 E288 RL recipe, including attention LoRA plus clone-expert full tuning, to use pipeline parallelism greater than one without changing its canonical policy format, Miles revision, checkpoint identity, or expert-parallel ownership.

## Scope

The change removes the recipe-level PP1 restriction from the public launcher, learner, and SSH-plan validator. The existing TP8, EP8, eight-GPU rollout-engine, and DeepSeek V4 model-contract checks remain unchanged.

The expert-full runtime becomes pipeline-local:

- expert configuration validates the complete expert/branch grid only for layers present on the current pipeline stage;
- attention adapter conversion keeps global Bridge tasks, including remote tasks whose parameter is absent locally;
- applying a global policy broadcasts every canonical tensor collectively but writes only parameters owned by the current stage;
- expert tensor views derive canonical layer and projection names from Megatron-Bridge conversion tasks, rather than assuming a stage-local parameter index is the global HF layer index.

Miles already performs collective TP/PP adapter conversion at the pinned revision. The global Yeto canonical policy and its layout hash therefore remain unchanged.

## Alternatives Rejected

Removing only the PP1 guards would leave the expert-full runtime failing on remote adapter parameters and on the fixed 43-layer local assertion.

A new generic hybrid-shard abstraction would reduce duplication but would expand the change beyond the requested minimum and alter a production-sensitive synchronization boundary.

## Failure Handling

The runtime continues to fail closed when a local trainable parameter lacks a Bridge task, a conversion task produces a duplicate or out-of-contract canonical name, a pipeline stage has no expert layers, or the global policy layout differs from the attested contract.

## Verification

Tests cover:

- DeepSeek V4 clone-LoRA and expert-full CLI/SSH plans with PP2;
- uneven 43-layer PP2 argv generation;
- expert configuration on a pipeline-local subset of layers;
- remote attention adapter tasks with `param_weight=None`;
- conversion of stage-local expert parameters to nonzero global HF layer IDs;
- the existing expert-full, launcher, SSH harness, RL core, and DeepSeek V4 bridge suites.
