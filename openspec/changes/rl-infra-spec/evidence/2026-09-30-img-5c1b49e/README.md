# ports image 5c1b49e-9f29303 (Agent IMG, 2026-09-30)
- Built with `CRANE=/tmp/img-tools/crane scripts/build_miles_ports_image.sh --push` (crane append; no docker daemon access).
- ghcr.io/michaellchung/yeto-miles-ports:5c1b49e-9f29303@sha256:17d428a2e955a1d43525b59b8785bb786b8e48852fe00c6e3e90dad798f0bcef (private: anonymous token request -> 401).
- File-level check on the pushed image's top (overlay) layer: /opt/yeto/image-manifest.json miles=5c1b49eb sglang=9f29303; /root/miles/.git HEAD = 5c1b49eb; `diff -r` of /root/miles/miles vs `git archive 5c1b49eb` empty.
- Parser check (check_flags.py / check_flags.log): the image's /root/miles on PYTHONPATH, miles-next-venv (no megatron.training), Miles' own argument provider + validate_policy_loss_variant_args. Full upstream parse_args + validate_args NOT run: that needs a container with libcuda (ALGO-1b ran it on a Modal T4) and this round allows no GPU/paid resources; no local docker daemon access.
