"""2.6 attempt 3 Modal driver (plan: attempt3-plan.md). Usage:
modal_run3.py <repo root> <yeto sha> <sandbox-id file> <remote results json>"""
import base64, io, json, os, sys, tarfile, modal
APP = "algocap-parse"
IMG = "ghcr.io/michaellchung/yeto-miles-ports@sha256:5da40a07dabb3ea3fcf921efb4b2a21ca1178220bde40dc178c79b734fdaa540"
HERE = os.path.dirname(os.path.abspath(__file__))

def registry_secret():
    auth = json.load(open(os.path.expanduser("~/.docker/config.json")))["auths"]["ghcr.io"]["auth"]
    user, token = base64.b64decode(auth).decode().split(":", 1)
    return modal.Secret.from_dict({"REGISTRY_USERNAME": user, "REGISTRY_PASSWORD": token})

root, sha, idfile, out = sys.argv[1:5]
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    for part in ("yeto", "tests"):
        tar.add(os.path.join(root, part), arcname=part, filter=lambda t: None if "__pycache__" in t.name else t)
    for f in ("parse3.py", "cases3.py"):
        tar.add(os.path.join(HERE, f), arcname="ev/" + f)
app = modal.App.lookup(APP, create_if_missing=True)
image = modal.Image.from_registry(IMG, secret=registry_secret()).entrypoint([])
sb = modal.Sandbox.create("sleep", "infinity", app=app, image=image, timeout=1800, cpu=4, memory=16384, gpu="T4")
print("SANDBOX", sb.object_id, "YETO_SHA", sha, flush=True)
open(idfile, "w").write(sb.object_id)
try:
    sb.filesystem.write_bytes(buf.getvalue(), "/tmp/y.tgz")
    p = sb.exec("bash", "-lc", "mkdir -p /work && tar xzf /tmp/y.tgz -C /work && cd /work && "
                "nvidia-smi -L | head -1 && git -C /root/miles rev-parse HEAD && "
                "PYTHONPATH=/root/miles:$PYTHONPATH python ev/parse3.py remote /work /work/remote3.json",
                timeout=1500)
    for line in p.stdout:
        print(line, end="", flush=True)
    print(p.stderr.read()[-4000:], file=sys.stderr)
    print("EXIT", p.wait())
    open(out, "wb").write(sb.filesystem.read_bytes("/work/remote3.json"))
finally:
    sb.terminate()
    print("TERMINATED", sb.object_id)
