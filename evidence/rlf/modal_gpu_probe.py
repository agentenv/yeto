import modal, json
app=modal.App.lookup('yeto-rlf-gpu-probe2', create_if_missing=True)
image=modal.Image.from_registry('nvidia/cuda:12.6.3-base-ubuntu24.04')
sb=modal.Sandbox.create('bash','-lc','nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader && echo RLF_GPU_READY',app=app,image=image,gpu='T4',cpu=1,memory=2048,runtime='gvisor',timeout=180,idle_timeout=60,tags={'yeto-purpose':'rlf-gpu-probe'})
try:
 print(json.dumps({'sandbox_id':sb.object_id,'stdout':''.join(list(sb.stdout)),'stderr':''.join(list(sb.stderr))}))
finally: sb.terminate()
