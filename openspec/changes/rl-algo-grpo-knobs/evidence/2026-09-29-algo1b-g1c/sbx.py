import sys, os, io, tarfile, subprocess, modal
APP="algo1b-g1c"
IMG="radixark/miles@sha256:90940828dcd4d54fd907ff668b43537cbd94778047580e4160d6560af548b74d"
MILES="0394715083c91182b5eb0c526eeee4196ac694b9"; SGL="9f29303bef1eea38eb613e5f454a52db1326422d"
def image():
    return (modal.Image.from_registry(IMG)
        .entrypoint([])
        .run_commands(
          "git clone https://github.com/michaellchung/miles /opt/miles-next && cd /opt/miles-next && git checkout --detach "+MILES,
          "git clone --filter=blob:none https://github.com/michaellchung/sglang /opt/sglang-next && cd /opt/sglang-next && git checkout --detach "+SGL,
          "pip list 2>/dev/null | grep -i -E '^(sglang|miles|sgl-kernel|torch|megatron|peft) ' || true",
          "cd /opt/sglang-next/python && pip install --no-deps -e . 2>&1 | tail -3",
          "cd /opt/miles-next && pip install --no-deps -e . 2>&1 | tail -3",
          "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal 2>&1 | tail -1",
        ))
cmd=sys.argv[1]
app=modal.App.lookup(APP, create_if_missing=True)
if cmd=="create":
    gpu=sys.argv[2]
    sb=modal.Sandbox.create("sleep","infinity",app=app,image=image(),gpu=gpu,timeout=int(sys.argv[3]),cpu=16,memory=131072)
    print(sb.object_id)
elif cmd=="exec":
    sb=modal.Sandbox.from_id(sys.argv[2])
    p=sb.exec("bash","-lc",sys.argv[3],timeout=int(os.environ.get("T","3000")))
    for line in p.stdout: print(line,end="",flush=True)
    err=p.stderr.read(); print(err,end="",file=sys.stderr)
    sys.exit(p.wait())
elif cmd=="put":  # put <id> <localtar> <remote_dir>
    sb=modal.Sandbox.from_id(sys.argv[2]); data=open(sys.argv[3],"rb").read()
    sb.filesystem.write_bytes(data, "/tmp/up.tgz")
    p=sb.exec("bash","-lc",f"mkdir -p {sys.argv[4]} && tar xzf /tmp/up.tgz -C {sys.argv[4]}"); print(p.stderr.read()); sys.exit(p.wait())
elif cmd=="get":  # get <id> <remote_tgz> <local>
    sb=modal.Sandbox.from_id(sys.argv[2])
    open(sys.argv[4],"wb").write(sb.filesystem.read_bytes(sys.argv[3]))
elif cmd=="kill":
    modal.Sandbox.from_id(sys.argv[2]).terminate(); print("terminated")
elif cmd=="list":
    for s in modal.Sandbox.list(app_id=app.app_id): print(s.object_id)
