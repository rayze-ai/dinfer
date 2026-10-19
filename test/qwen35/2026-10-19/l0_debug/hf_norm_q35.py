import inspect, torch, transformers.models.qwen3_5.modeling_qwen3_5 as M
src=inspect.getsource(M)
i=src.find("class Qwen3_5RMSNorm("); print(src[i:i+900])
print("---- '1.0 +' occurrences:", [l.strip() for l in src.splitlines() if "1.0 +" in l or "1 + self.weight" in l][:6])
from safetensors import safe_open
import json
p="/data/models/Qwen/Qwen3.8-27B"; wm=json.load(open(p+"/model.safetensors.index.json"))["weight_map"]
def stat(k):
    with safe_open(p+"/"+wm[k],"pt") as f: t=f.get_tensor(k).float()
    return k.split("language_model.")[-1], t.dtype, tuple(t.shape), round(t.mean().item(),4), round(t.abs().mean().item(),4), round(t.min().item(),3), round(t.max().item(),3)
for k in ["model.language_model.layers.0.input_layernorm.weight","model.language_model.layers.0.post_attention_layernorm.weight","model.language_model.norm.weight","model.language_model.layers.3.self_attn.q_norm.weight","model.language_model.layers.3.self_attn.k_norm.weight","model.language_model.layers.0.linear_attn.norm.weight"]:
    print(stat(k))
