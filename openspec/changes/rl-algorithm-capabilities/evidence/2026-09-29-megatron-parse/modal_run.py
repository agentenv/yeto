"""Modal CPU sandbox driver for the 2.6 Megatron parse (plan: ../2.6-plan.md)."""
import base64, io, json, os, sys, tarfile, modal
APP = "algocap-parse"
IMG = "ghcr.io/michaellchung/yeto-miles-ports@sha256:5da40a07dabb3ea3fcf921efb4b2a21ca1178220bde40dc178c79b734fdaa540"

def registry_secret():
    auth = json.load(open(os.path.expanduser("~/.docker/config.json")))["auths"]["ghcr.io"]["auth"]
    user, token = base64.b64decode(auth).decode().split(":", 1)
    return modal.Secret.from_dict({"REGISTRY_USERNAME": user, "REGISTRY_PASSWORD": token})

cmd = sys.argv[1]
app = modal.App.lookup(APP, create_if_missing=True)
if cmd == "run":
    root = sys.argv[2]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for part in ("yeto", "tests"):
            tar.add(os.path.join(root, part), arcname=part,
                    filter=lambda t: None if "__pycache__" in t.name else t)
    image = modal.Image.from_registry(IMG, secret=registry_secret()).entrypoint([])
    sb = modal.Sandbox.create("sleep", "infinity", app=app, image=image, timeout=1800,
                              cpu=4, memory=16384, gpu=os.environ.get("ALGOCAP_GPU") or None)
    print("SANDBOX", sb.object_id, flush=True)
    open(sys.argv[3], "w").write(sb.object_id)
    try:
        sb.filesystem.write_bytes(buf.getvalue(), "/tmp/y.tgz")
        sb.filesystem.write_bytes(open(sys.argv[4], "rb").read(), "/work/parse_all.py")
        p = sb.exec("bash", "-lc", "mkdir -p /work/yeto && tar xzf /tmp/y.tgz -C /work/yeto && "
                    "cd /work && nvidia-smi -L 2>&1 | head -1; "
                    "python -c 'import miles,megatron.training;print(miles.__file__)' && "
                    "git -C /root/miles rev-parse HEAD && "
                    "PYTHONPATH=/root/miles:$PYTHONPATH python /work/parse_all.py", timeout=1500)
        for line in p.stdout:
            print(line, end="", flush=True)
        print(p.stderr.read()[-4000:], file=sys.stderr)
        print("EXIT", p.wait())
        open(sys.argv[5], "wb").write(sb.filesystem.read_bytes("/work/results.json"))
    finally:
        sb.terminate()
        print("TERMINATED", sb.object_id)
