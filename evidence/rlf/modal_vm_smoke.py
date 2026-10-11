import modal, json, time
APP='yeto-rlf-vm-smoke'
app=modal.App.lookup(APP, create_if_missing=True)
image=modal.Image.debian_slim(python_version='3.12')
with modal.enable_output():
    sb=modal.Sandbox.create('bash','-lc','python -V && (command -v dockerd || true) && echo RLF_VM_READY',app=app,image=image,runtime='vm',cpu=1,memory=1024,timeout=180,idle_timeout=60,tags={'yeto-purpose':'rlf-vm-smoke','yeto-episode':'smoke-0'})
    out=''.join(list(sb.stdout)); err=''.join(list(sb.stderr)); rc=sb.returncode
    print(json.dumps({'sandbox_id':sb.object_id,'returncode':rc,'stdout':out,'stderr':err,'runtime':'vm'}))
    sb.terminate()
