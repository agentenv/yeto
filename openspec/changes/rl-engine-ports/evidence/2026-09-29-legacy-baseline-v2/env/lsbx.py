import sys, os, modal
APP = "yeto-legacy-baseline"
# Exact MILES_IMAGE (ghcr.io/agentenv/miles@sha256:80c2...) is not pullable with available creds;
# base on the public radixark/miles v0.1.0 (sglang v0.5.16 era, same as agentenv fork base 6062afe)
BASE = "radixark/miles@sha256:cd40db923225c4146e90fdf4aa04bc000b71c1e980cc42df6f368de7545eaa09"
MILES_REPO = "https://github.com/agentenv/miles"
MILES_BASE = "6062afe0a9d5d6471e8395dedc81c78dd9f4a84f"
MILES = "ae475060fa670145aef75d678809039ae999cb97"
BUNDLE_SHA = "da3464d3c389f7e2cb3e390119b3c42c2f130d94a0711f9c95608f3243826cc6"
SGL_REPO = "https://github.com/agentenv/sglang"
SGL = "e1b57eb8e7749235c987cc6b1b2824ce3265369b"
BUNDLE = "/home/michael/work/gpu-legacy/yeto2/yeto/rl/vendor/miles-qwen38.bundle"
def image():
    return (modal.Image.from_registry(BASE).entrypoint([])
        .add_local_file(BUNDLE, "/opt/miles-qwen38.bundle", copy=True)
        .run_commands(
          f"printf '%s  %s\\n' {BUNDLE_SHA} /opt/miles-qwen38.bundle | sha256sum --check -",
          f"git clone --no-checkout {MILES_REPO} /opt/miles-legacy && cd /opt/miles-legacy && git fetch --depth 1 origin {MILES_BASE} && git checkout --detach {MILES_BASE} && git bundle verify /opt/miles-qwen38.bundle && git fetch /opt/miles-qwen38.bundle {MILES} && git checkout --detach {MILES} && test \"$(git rev-parse HEAD)\" = {MILES} && test -z \"$(git status --porcelain --untracked-files=all)\"",
          f"git clone --no-checkout {SGL_REPO} /opt/sglang-legacy && cd /opt/sglang-legacy && git fetch --depth 1 origin {SGL} && git checkout --detach {SGL}",
          "pip list 2>/dev/null | grep -i -E '^(sglang|miles|sgl-kernel|torch|megatron-core|peft|transformers|torch_memory_saver) ' || true",
          "cd /opt/sglang-legacy/python && pip install --no-deps -e . 2>&1 | tail -2",
          "cd /opt/miles-legacy && pip install --no-deps -e . 'peft==0.20.0' 2>&1 | tail -2",
          "cd /opt/miles-legacy && git status --porcelain --untracked-files=all | head",
          "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal 2>&1 | tail -1",
        ))
cmd = sys.argv[1]
app = modal.App.lookup(APP, create_if_missing=True)
if cmd == "create":
    sb = modal.Sandbox.create("sleep", "infinity", app=app, image=image(), gpu=sys.argv[2],
                              timeout=int(sys.argv[3]), cpu=16, memory=131072)
    print(sb.object_id)
elif cmd == "exec":
    sb = modal.Sandbox.from_id(sys.argv[2])
    p = sb.exec("bash", "-lc", sys.argv[3], timeout=int(os.environ.get("T", "3000")))
    for line in p.stdout: print(line, end="", flush=True)
    print(p.stderr.read(), end="", file=sys.stderr); sys.exit(p.wait())
elif cmd == "put":
    sb = modal.Sandbox.from_id(sys.argv[2]); sb.filesystem.write_bytes(open(sys.argv[3], "rb").read(), "/tmp/up.tgz")
    p = sb.exec("bash", "-lc", f"mkdir -p {sys.argv[4]} && tar xzf /tmp/up.tgz -C {sys.argv[4]}"); print(p.stderr.read()); sys.exit(p.wait())
elif cmd == "get":
    sb = modal.Sandbox.from_id(sys.argv[2]); open(sys.argv[4], "wb").write(sb.filesystem.read_bytes(sys.argv[3]))
elif cmd == "kill":
    modal.Sandbox.from_id(sys.argv[2]).terminate(); print("terminated")
elif cmd == "list":
    for s in modal.Sandbox.list(app_id=app.app_id): print(s.object_id)
