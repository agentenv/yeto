import modal, json
app=modal.App.lookup('yeto-rlf-policy-probe', create_if_missing=True)
image=modal.Image.from_registry('pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime')
sb=modal.Sandbox.create('bash','-lc',r'''python - <<"X"
import torch
m=torch.nn.Linear(8,2,device="cuda",dtype=torch.float32)
x=torch.randn(4,8,device="cuda"); y=torch.tensor([0,1,0,1],device="cuda")
o=torch.optim.SGD(m.parameters(),lr=.1)
before=m.weight.detach().clone(); loss=torch.nn.functional.cross_entropy(m(x),y); loss.backward(); g=float(m.weight.grad.abs().sum()); o.step(); delta=float((m.weight.detach()-before).abs().sum())
print({"cuda":torch.cuda.get_device_name(0),"loss":float(loss),"grad_abs_sum":g,"policy_update_abs_sum":delta,"passed":g>0 and delta>0})
X''',app=app,image=image,gpu='T4',cpu=2,memory=8192,runtime='gvisor',timeout=300,idle_timeout=60,tags={'yeto-purpose':'rlf-policy-probe'})
try: print(''.join(list(sb.stdout)))
finally: sb.terminate()
