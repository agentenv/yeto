# Production source provenance and safe loading

Yeto resolves every remote model and dataset reference before it provisions a
learner. A branch or tag is convenient input, but it is not a production
identity: the launcher asks the Hugging Face control plane for the referenced
Git commit and sends that 40-character commit to every learner. Direct
learner, export, and sampling entry points apply the same policy. In a
distributed learner, rank zero resolves once and broadcasts the result so a
moving ref cannot select different snapshots on different ranks.

The security defaults are intentionally fail closed:

- `trust_remote_code=False` is passed to model config, tokenizer, and model
  loaders. `--trust-remote-code` is a deliberate opt-in and does not weaken
  revision pinning: executable Hub code is still loaded only from the resolved
  commit.
- Remote weights use safetensors where the loader supports it. Yeto's
  diffusion raw-state fallback writes safetensors. The remaining supported
  legacy `.pt` state inputs use `torch.load(..., weights_only=True)`; legacy
  binary adapter weights are rejected by the sampler.
- A pinned dataset failure never falls back to `refs/convert/parquet`, because
  that is a separately moving ref. Materialize schema-normalized data locally
  when a pinned source revision cannot be loaded directly.
- Artifact metadata cannot implicitly activate a Python diffusion adapter.
  Sampling a custom artifact requires the operator to pass
  `--diffusion-adapter` again after reviewing that code.
- Pickled loss callables are disabled unless
  `--allow-unsafe-pickled-loss` is present. Pickle can execute arbitrary code.
  The launcher stages an opted-in payload at a content-addressed per-payload
  filename, records its SHA-256 digest, quotes the learner argument, and every
  learner verifies the same digest before deserialization. Concurrent runs
  cannot overwrite one another's loss payload.

## Revision controls

Use a branch, tag, or commit at the public CLI:

```bash
yeto launch \
  --gpu gcp:1xa100@us-central1 \
  --model org/model \
  --model-revision release-2026-07 \
  --data org/training-data \
  --data-revision data-v4 \
  --loss-function cross_entropy
```

Before any cloud resource is created, both moving names are replaced with
their commits. The background prefetch, config, tokenizer, model, and dataset
loads receive those commits. If resolution fails or the Hub returns anything
other than a full commit SHA, launch stops.

The launcher also hashes the complete installed `yeto/**/*.py` tree. The head
and every learner compare their copy with that expected digest before loading
models or data; distributed ranks attest collectively so a mismatch fails the
whole island instead of stranding peers in a later collective.

The same controls exist on the standalone entry points:

```bash
python -m yeto.learner \
  --model org/model --model-revision <40-character-commit> \
  --data org/training-data --data-revision <40-character-commit> \
  --syncer none --learner-id 0 --num-learners 1

yeto-export \
  --checkpoint syncer.ckpt \
  --model org/model --model-revision <40-character-commit> \
  --output-dir exported

yeto-diffusion-export \
  --checkpoint syncer.ckpt \
  --model org/diffusion-model --model-revision <40-character-commit> \
  --output-dir exported-diffusion
```

Passing an already resolved commit does not require a Hub API request, so a
warm offline cache remains usable. Local model and dataset paths remain
supported and do not accept revision flags; accepting one would record a false
identity. Object-store data continues through the launcher's mounted local
path and likewise does not accept `--data-revision`. Version and lock those
objects at the storage layer when immutable local/object-store provenance is
required.

## Trusted code boundary

`--trust-remote-code` means the pinned model repository is inside the trusted
Python process boundary. It is not a sandbox. Review the repository commit and
the container/dependency image before enabling it.

`custom:path.py:function` and `--diffusion-adapter path.py:factory` are also
explicit Python-code execution. Custom diffusion loaders used with a remote
base must declare `supports_pinned_model_source = True` and either:

1. call `yeto.provenance.materialize_pinned_model(args)` and load only that
   local snapshot; or
2. pass `args.model_revision` and `args.trust_remote_code` through every
   underlying model/config/tokenizer loader.

Yeto rejects a custom remote loader that does not declare this contract. The
built-in non-standard adapter follows the local-snapshot form. An adapter that
uses additional repositories must independently pin each one; silently using
another `main` branch is not supported.

Cloud launch accepts adapter source only from inside the synced Yeto workdir.
The launcher records its SHA-256 digest, distributed learners attest one
shared digest collectively, and diffusion artifacts retain it. Module
discovery does not import parent packages, and the loader compiles the exact
attested source bytes instead of trusting `sys.modules` or bytecode caches.
Sampling requires an explicitly supplied adapter to match the artifact digest,
so a path that was edited after training is rejected instead of being silently
trusted.

Legacy custom diffusion artifacts that lack a complete training-time adapter
spec/digest binding fail closed. After separately verifying the historical
code-to-weights relationship, an operator may use
`--allow-unattested-legacy-adapter`; sample manifests mark that runtime binding
`legacy-unbound` rather than presenting it as attested.

## Legacy pickled losses

Prefer a reviewed built-in loss. Existing callable/custom launch workflows
that need by-value closure serialization must state the risk:

```bash
yeto launch ... \
  --loss-function custom:losses/my_loss.py:loss_fn \
  --allow-unsafe-pickled-loss
```

Direct `pickle:path` learner inputs also require the flag. A digest mismatch
is fatal. The digest proves that every learner received the same bytes; it
does not make those bytes safe.

## Artifact record

Causal training and checkpoint export write `yeto_provenance.json` next to the
saved model or adapter. Diffusion artifacts embed the same object under
`provenance` in `yeto_diffusion_adapter.json`. Records include:

- requested and resolved model identity;
- requested and resolved dataset identity when applicable;
- the resolved commits;
- whether remote code was trusted;
- a stable SHA-256 digest of the installed Yeto Python source tree;
- the loss artifact digest and unsafe-pickle state when applicable; and
- export checkpoint/global-step context where available.

Causal LoRA SFT outputs also record their training recipe and optional parent
adapter lineage. Strict `--resume-from` verifies the recorded immutable model,
dataset, trust setting, and recipe; `--branch-from` records the relationship
while permitting an intentional recipe or dataset change. Merged exports
record the exact parent directory digest and SafeTensors shard policy. See
[ADAPTER_LIFECYCLE.md](ADAPTER_LIFECYCLE.md).

Sampling uses the artifact's resolved base-model commit by default. A model
override is resolved independently. Batch sample manifests keep the original
artifact provenance and the actual runtime model provenance as separate
objects, record the artifact and runtime adapter specs/digests separately, and
also copy input-dataset provenance into every record.

Programmatic artifact writers that bypass provenance pinning mark their record
`attestation_status: unattested`; they preserve caller-supplied values only as
requests and never claim a moving or missing revision as resolved. Use the CLI
entry points, or call the pinning helpers first, for production artifacts.

Checkpoint exports record the SHA-256 digest of the exact byte buffer they
parsed. A concurrent atomic checkpoint replacement therefore cannot make an
export derived from one checkpoint claim the digest of another.

## Migration notes

- Old diffusion artifacts containing only `trainable_state.pt` can be handled
  only by an explicitly supplied custom adapter; Yeto loads such tensor state
  with `weights_only=True`. Re-export to `trainable_state.safetensors`.
- Old PEFT `.bin` adapter directories must be converted to safetensors before
  sampling.
- Old custom diffusion artifacts no longer auto-import the adapter named in
  metadata. Pass the reviewed adapter explicitly. If the artifact predates
  adapter digest records, also pass `--allow-unattested-legacy-adapter` and
  retain the resulting `legacy-unbound` manifest status in downstream audits.
- Existing custom/callable loss launches now need
  `--allow-unsafe-pickled-loss`. Remove the flag after migrating to a built-in
  or otherwise non-pickled production loss.
