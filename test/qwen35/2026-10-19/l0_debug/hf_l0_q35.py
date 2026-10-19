import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
p="/data/models/Qwen/Qwen3.8-27B"
tok=AutoTokenizer.from_pretrained(p)
m=AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, device_map="cuda:0").eval()
ids=tok("The capital of France is", return_tensors="pt").input_ids.cuda()
lm=m.model.language_model if hasattr(m.model,"language_model") else m.model
L0=lm.layers[0]
print("L0 type", type(L0.linear_attn).__name__, [n for n,_ in L0.linear_attn.named_children()])
la=L0.linear_attn
for n in ["A_log","dt_bias"]:
    v=getattr(la,n); print(n, v.dtype, tuple(v.shape), [round(x,4) for x in v.float().flatten()[:4].tolist()])
print("conv1d.weight", la.conv1d.weight.dtype, tuple(la.conv1d.weight.shape), la.conv1d.bias is not None, [round(x,5) for x in la.conv1d.weight.float().flatten()[:4].tolist()])
print("norm.weight", tuple(la.norm.weight.shape), [round(x,4) for x in la.norm.weight.float()[:4].tolist()], "norm type", type(la.norm).__name__)
rec={}
def hk(name):
    def f(mod, inp, out):
        o=out[0] if isinstance(out,tuple) else out
        rec[name]=o.detach()
    return f
hs=[]
for n,mod in L0.named_modules():
    if n in ["input_layernorm","linear_attn","post_attention_layernorm","mlp","linear_attn.in_proj_qkv","linear_attn.in_proj_z","linear_attn.in_proj_a","linear_attn.in_proj_b","linear_attn.conv1d","linear_attn.norm","linear_attn.out_proj"]:
        hs.append(mod.register_forward_hook(hk(n)))
with torch.no_grad():
    o=m(ids, output_hidden_states=True)
def nrm(x):
    x=x.float()
    if x.dim()==3: x=x[0,-1]
    elif x.dim()==2: x=x[-1]
    return round(float(x.norm()),4)
for k,v in rec.items(): print(k, tuple(v.shape), "last_tok_norm", nrm(v), "first", [round(t,4) for t in v.float().flatten()[:3].tolist()])
print("hs1 norm", round(float(o.hidden_states[1][0,-1].float().norm()),4))
import inspect; src=inspect.getsource(type(la).forward); print(src[:3800])
