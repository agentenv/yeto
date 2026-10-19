# Engine images

| Directory | Image | CPU arch | Device family | Roles | Status |
|---|---|---|---|---|---|
| `miles-ports/` | `yeto-miles-ports` | x86_64 | cuda | trainer + sampler | built and pushed (see the Dockerfile header) |
| `yeto-release/` | release layer on `yeto-miles-ports` | x86_64 | cuda | — | built once (2026-10-19) |
| `ppu-train/` | `yeto-ppu-train` | x86_64 | ppu | trainer + sampler | written, never built (S25 N0.2) |
| `npu-sampler/` | `yeto-npu-sampler` | aarch64 | npu | sampler | written, never built (S25 N0.2) |

`yeto/launch/images.py` is the catalog the launcher reads. A node of the SSH
inventory declares `cpu_arch`; a node with no `image` takes the inventory's
`images[cpu_arch]`, and a node whose image is a catalog image of another
architecture or device family is refused before any SSH (N1.6).

## Version locks

Each image directory has `versions.lock.json`. It is the single source of the
versions the image asserts, and it holds the decisions behind them (the
`decisions` list). `docker/image-assert/assert_image_versions.py` reads it:

* the image entrypoint (`docker/image-assert/entrypoint.sh`) runs it at every
  container start and exits 1 before the engine starts when a check fails;
* `yeto-ppu-train` also runs it at build time, so an image whose own lock does
  not hold is never pushed. `yeto-npu-sampler` cannot: `import torch_npu`
  needs the NPU driver, which no build host has.

Checked: CPU architecture, Python, package versions and imports, build-time
commits (`/opt/yeto/image/commits.json`), the CANN toolkit version, the host
driver version and required files. An entry with `"version": null` is recorded
in the report but not compared; `"optional": true` reports a missing package or
failed import without failing (TransformerEngine and tilelang on PPU, whose
support the first PPU probe decides).

## Open points (both new images)

1. Neither image was built: no base digest, no `pip check` output, no real
   version report. Every version in the two locks comes from a survey or from
   upstream documentation, not from a built image.
2. `yeto-ppu-train`: the base image's registry path and digest are unknown
   (N0.1 read the tag `inference-xpu-pytorch-opt:v0.5.17` on ppu1 only), and
   `MEGATRON_COMMIT` has no value yet.
3. `yeto-npu-sampler`: the ModelArts fixes come from a build context that does
   not exist in this repository yet, and the arm64 release layer is a follow-up.
