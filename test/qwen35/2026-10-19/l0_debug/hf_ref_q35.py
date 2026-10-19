import torch, json, sys
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
p="/data/models/Qwen/Qwen3.8-27B"
tok=AutoTokenizer.from_pretrained(p)
cfg=AutoConfig.from_pretrained(p)
m=AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, device_map="cuda:0")
m.eval()
ids=tok("The capital of France is", return_tensors="pt").input_ids.cuda()
print("ids", ids.tolist())
with torch.no_grad():
    o=m(ids, output_hidden_states=True)
lg=o.logits[0,-1].float()
top=torch.topk(lg,5)
print("top5", [(tok.decode([i]), round(v.item(),3), int(i)) for v,i in zip(top.values, top.indices)])
hs=o.hidden_states
norms=[float(h[0,-1].float().norm()) for h in hs]
print("n_hidden_states", len(hs))
print("last_tok_norms", [round(x,2) for x in norms])
# 全 token 的首层/末层 norm，供逐层对拍
torch.save({"ids":ids.cpu(),"logits":o.logits[0].float().cpu(),"hs":[h[0].float().cpu() for h in hs]}, "/tmp/hf_ref_q35.pt")
print("saved")
