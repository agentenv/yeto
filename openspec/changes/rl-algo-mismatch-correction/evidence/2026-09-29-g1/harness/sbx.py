"""Modal sandbox control for algo1a G1/G2 (plan.md). Subcommands:
  create                      -> prints sandbox id (H100!, timeout 6000 s)
  exec <id> <cmd> [timeout]   stream a bash command
  put <id> <local.tgz> <dir>  upload+extract
  get <id> <remote> <local>   download a file
  kill <id>                   terminate
"""
import base64, json, os, sys
import modal
APP = "algo1a-g1"
IMG = "ghcr.io/michaellchung/yeto-miles-ports@sha256:5da40a07dabb3ea3fcf921efb4b2a21ca1178220bde40dc178c79b734fdaa540"
cmd = sys.argv[1]
if cmd == "create":
    auth = json.load(open(os.path.expanduser("~/.docker/config.json")))["auths"]["ghcr.io"]["auth"]
    user, token = base64.b64decode(auth).decode().split(":", 1)
    secret = modal.Secret.from_dict({"REGISTRY_USERNAME": user, "REGISTRY_PASSWORD": token})
    image = modal.Image.from_registry(IMG, secret=secret).entrypoint([])
    app = modal.App.lookup(APP, create_if_missing=True)
    sb = modal.Sandbox.create("sleep", "infinity", app=app, image=image, gpu="H100!", cpu=16,
                              memory=131072, timeout=6000)
    print(sb.object_id, flush=True)
elif cmd == "exec":
    sb = modal.Sandbox.from_id(sys.argv[2])
    p = sb.exec("bash", "-lc", sys.argv[3], timeout=int(sys.argv[4]) if len(sys.argv) > 4 else 3000)
    for line in p.stdout:
        print(line, end="", flush=True)
    print(p.stderr.read(), end="", file=sys.stderr)
    sys.exit(p.wait())
elif cmd == "put":
    sb = modal.Sandbox.from_id(sys.argv[2])
    sb.filesystem.write_bytes(open(sys.argv[3], "rb").read(), "/tmp/up.tgz")
    p = sb.exec("bash", "-lc", f"mkdir -p {sys.argv[4]} && tar xzf /tmp/up.tgz -C {sys.argv[4]}")
    print(p.stderr.read()); sys.exit(p.wait())
elif cmd == "get":
    sb = modal.Sandbox.from_id(sys.argv[2])
    open(sys.argv[4], "wb").write(sb.filesystem.read_bytes(sys.argv[3]))
elif cmd == "kill":
    modal.Sandbox.from_id(sys.argv[2]).terminate(); print("terminated", sys.argv[2])
