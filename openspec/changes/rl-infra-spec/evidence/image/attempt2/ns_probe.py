import base64, json, os, sys, modal
auth = json.load(open(os.path.expanduser("~/.docker/config.json")))["auths"]["ghcr.io"]["auth"]
u, t = base64.b64decode(auth).decode().split(":", 1)
app = modal.App.lookup("img-smoke", create_if_missing=True)
code = "import sys, miles; print(miles.__spec__); print(list(miles.__path__)); print([p for p in sys.path])"
for name, img in (("ports", modal.Image.from_registry(sys.argv[1], secret=modal.Secret.from_dict({"REGISTRY_USERNAME": u, "REGISTRY_PASSWORD": t}))),
                  ("base", modal.Image.from_registry(sys.argv[2]))):
    sb = modal.Sandbox.create("sleep", "infinity", app=app, image=img.entrypoint([]), cpu=2, memory=8192, timeout=1500)
    try:
        for cwd in ("/", "/root"):
            p = sb.exec("bash", "-lc", f"cd {cwd} && python3 -c '{code}' 2>&1 | grep -v -i -E 'warn|flax'", timeout=300)
            print(f"== {name} cwd={cwd}\n" + p.stdout.read(), flush=True); p.wait()
    finally:
        sb.terminate(); print("terminated", name, sb.object_id, flush=True)
