# Engine images

| Directory | Image | CPU arch | Device family | Roles | Status |
|---|---|---|---|---|---|
| `miles-ports/` | `yeto-miles-ports` | x86_64 | cuda | trainer + sampler | built and pushed (see the Dockerfile header) |
| `yeto-release/` | release layer on `yeto-miles-ports` | x86_64 | cuda | — | see [`yeto-release/README.md`](yeto-release/README.md) |
| `ppu-train/` | `yeto-ppu-train` | x86_64 | ppu | trainer + sampler | written, never built |

`yeto/launch/engine_images.py` is the catalog the launcher reads. A node of the
SSH inventory declares `cpu_arch`; a node with no `image` takes the inventory's
`images[cpu_arch]`, and a node whose image is a catalog image of another
architecture or device family is refused before any SSH.

## Version locks

`ppu-train/` has a `versions.lock.json`. It is the single
source of the versions the image asserts, and it holds the decisions behind them
(the `decisions` list). `docker/image-assert/assert_image_versions.py` reads it:

* the image entrypoint (`docker/image-assert/entrypoint.sh`) runs it at every
  container start and exits 1 before the engine starts when a check fails;
* `yeto-ppu-train` also runs it at build time, so an image whose own lock does
  not hold is never pushed.

Checked: CPU architecture, Python, package versions and imports, build-time
commits (`/opt/yeto/image/commits.json`), the CANN toolkit version, the host
driver version and required files. An entry with `"version": null` is recorded
in the report but not compared; `"optional": true` reports a missing package or
failed import without failing (TransformerEngine and tilelang on PPU).

## Open points (`ppu-train`)

1. The image has not been built: there is no base digest, no `pip check` output
   and no real version report. Every version in the lock comes from a survey
   of the nodes or from upstream documentation, not from a built image.
2. `yeto-ppu-train`: the base image's registry path and digest are not pinned,
   and `MEGATRON_COMMIT` has no value yet.
