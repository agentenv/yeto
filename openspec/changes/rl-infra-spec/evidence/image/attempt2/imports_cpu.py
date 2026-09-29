import base64, json, os, sys, modal
auth = json.load(open(os.path.expanduser("~/.docker/config.json")))["auths"]["ghcr.io"]["auth"]
u, t = base64.b64decode(auth).decode().split(":", 1)
app = modal.App.lookup("img-smoke", create_if_missing=True)
img = modal.Image.from_registry(sys.argv[1], secret=modal.Secret.from_dict({"REGISTRY_USERNAME": u, "REGISTRY_PASSWORD": t})).entrypoint([])
sb = modal.Sandbox.create("sleep", "infinity", app=app, image=img, cpu=2, memory=8192, timeout=900)
try:
    sb.filesystem.write_bytes(open(sys.argv[2], "rb").read(), "/tmp/s.sh")
    # run only the imports check block from smoke_in_image.sh
    p = sb.exec("bash", "-lc", "cd / && awk '/^check\\(\\)/{print} /^check imports_point_at_forks/,/^PY.$/{print}' /tmp/s.sh > /tmp/i.sh && echo 'fails=0' | cat - /tmp/i.sh > /tmp/j.sh && bash /tmp/j.sh 2>&1 | grep -v -i -E 'warn|flax'", timeout=600)
    print(p.stdout.read(), flush=True); print("rc", p.wait())
finally:
    sb.terminate(); print("terminated", sb.object_id)
