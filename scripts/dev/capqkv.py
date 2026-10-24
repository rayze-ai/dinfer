import os, torch, numpy as np, glob, safetensors.torch as st
os.environ.setdefault("VLLM_BATCH_INVARIANT","1"); os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING","0")
import vllm.utils.flashinfer as FIK
cap={}
orig=FIK.flashinfer_scaled_fp8_mm
def p(*a, **k):
    r=orig(*a,**k)
    A=list(a)+[None]*5
    b = k.get("b", A[1]); sb = k.get("scale_b", A[3] if len(a)>3 else None)
    if "b" not in cap and b is not None and b.shape[-1]==8192:
        cap["b"]=b.detach().view(torch.uint8).cpu().numpy().copy()
        cap["sb"]=float(sb) if sb is not None else float("nan")
    return r
FIK.flashinfer_scaled_fp8_mm=p
from vllm import LLM, SamplingParams
ids=[int(x) for x in open("/w/dinfer/test/kaon3/_ref_L1/cn1.ids").read().split()]
llm=LLM(model=os.environ["M"], enforce_eager=True, tensor_parallel_size=1, gpu_memory_utilization=0.85,
        dtype="auto", max_model_len=4096, enable_prefix_caching=False, block_size=32, max_num_seqs=1)
llm.generate(prompts=[{"prompt_token_idms": ids}] if False else [{"prompt_token_ids": ids}],
             sampling_params=SamplingParams(max_tokens=1, temperature=0.0))
W=cap["b"]                      # [2816, 8192]
np.save("/w/Wq_layer0.npy", W)
print("captured", W.shape, "scale_b", cap["sb"], flush=True)

snap=(glob.glob("/models/models--kaon*/snapshots/*/")+glob.glob("/hf_data/hub/models--kaon*/snapshots/*/"))[0]
P="model.language_model.layers.0.self_attn."
C={}; S={}
for f in sorted(glob.glob(snap+"*.safetensors")):
    d=st.load_file(f)
    for k in d:
        if k.startswith(P) and "proj" in k:
            nm=k[len(P):]
            if nm.endswith(".weight"): C[nm[:-7]]=d[k].view(torch.uint8).numpy()
            elif nm.endswith(".weight_scale"): S[nm[:-13]]=float(d[k])
            elif nm.endswith(".input_scale"): S[nm[:-12]+".in"]=float(d[k])
print("scales", S, flush=True)
sq,sk,sv = S["q_proj"], S["k_proj"], S["v_proj"]
smax=max(sq,sk,sv)

def e4m3_to_f32(b):
    sign = np.where((b>>7)&1, -1.0, 1.0).astype(np.float32)
    e=((b>>3)&0xF).astype(np.int32); m=(b&7).astype(np.float32)
    v=np.where(e==0, m/8.0*2.0**-6, (1.0+m/8.0)*2.0**np.clip(e-7,-20,20))
    return (sign*v).astype(np.float32)
def f32_to_e4m3(x):
    ax=np.abs(x); s=np.sign(x)
    out=np.zeros_like(ax, dtype=np.uint8)
    for i,(lo,hi,bits) in enumerate([(0,0.001953125,0)]):
        pass
    # 用 torch 做，保证与 vLLM 一致（RNE）
    t=torch.from_numpy(x.astype(np.float32)).cuda().clamp(-448,448)
    return t.to(torch.float8_e4m3fn).view(torch.uint8).cpu().numpy()

def rq(parts, scales):
    """parts: list of fp8 [N_i,K]; scales: list of float. 返回拼接后的 [sumN, K] fp8"""
    outs=[]
    for w,s in zip(parts, scales):
        if s == smax:
            outs.append(w)
        else:
            dq = e4m3_to_f32(w) * np.float32(s)
            q  = f32_to_e4m3(dq / np.float32(smax))
            outs.append(q)
    return np.concatenate(outs, 0), smax

Wt = W.T                              # [8192, 2816]
for tag, parts, scales in [
    ("q,k,v   ", [C["q_proj"],C["k_proj"],C["v_proj"]], [sq,sk,sv]),
    ("q,k,k   ", [C["q_proj"],C["k_proj"],C["k_proj"]], [sq,sk,sk]),
]:
    m, sm = rq(parts, scales)
    ok = np.array_equal(m, Wt)
    segs = [np.array_equal(m[a:b], Wt[a:b]) for a,b in ((0,4096),(4096,6144),(6144,8192))]
    print(f"{tag} max={sm:.17g} 全等={ok} 分段={segs}", flush=True)
