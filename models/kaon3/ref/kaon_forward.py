#!/usr/bin/env python3
# 已知语义错误（2026-10-22 核对 vLLM gemma4.py / gemma4_rope.py，逐层对拍前必须先修）：
#  1. full 层 proportional RoPE：应整头 neox 旋转、配对 (i, i+head_dim/2)、前 rope_dim/2 对频率 = theta^(-2i/head_dim)，
#     其余零频；本脚本 rope_apply 只转前 rope_dim 维、配对 (i, i+rope_dim/2)、分母 rope_dim —— 两处都错。
#  2. k_eq_v 层：V = rmsnorm_noweight(原始 K 投影)，不过 k_norm、不过 RoPE；本脚本写成 v=k（norm+rope 之后）。
#  3. SNAP 路径是旧挂载 /models/...，5090 上是 /hf_data/hub/...。
"""Kaon-V3 (Gemma4 MoE) 纯 torch 参考前向：CPU/GPU 均可，权重按需去量化。
用途：产生与基线服务可对拍的 logits，并作为 Rust 实现的语义基准。"""
import os,sys,json,glob,math,time
os.environ.setdefault("HF_HUB_OFFLINE","1")
import torch
from safetensors import safe_open

SNAP=glob.glob("/models/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/*")[0]
DT=torch.bfloat16
DEV=os.environ.get("DEV","cpu")

E2M1=torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6.],dtype=torch.float32)
def unpack_nvfp4(p):
    lo=(p & 0x0F).to(torch.int64); hi=(p>>4).to(torch.int64)
    o=torch.empty(p.shape[0],p.shape[1]*2,dtype=torch.float32)
    o[:,0::2]=E2M1[lo & 7]*torch.where(lo>=8,torch.tensor(-1.),torch.tensor(1.))
    o[:,1::2]=E2M1[hi & 7]*torch.where(hi>=8,torch.tensor(-1.),torch.tensor(1.))
    return o
def deq_nvfp4(packed,scale_f8,scale2):
    w=unpack_nvfp4(packed)
    s=scale_f8.to(torch.float32).repeat_interleave(16,dim=1)[:,:w.shape[1]]
    return (w*s*float(scale2)).to(DT)
def deq_fp8(w8,wscale):
    return (w8.to(torch.float32)*float(wscale)).to(DT)

class Store:
    def __init__(self):
        self.idx=json.load(open(f"{SNAP}/model.safetensors.index.json"))["weight_map"]
        self.files={}
    def fh(self,k):
        f=self.idx[k]
        if f not in self.files: self.files[f]=safe_open(f"{SNAP}/{f}","pt")
        return self.files[f]
    def has(self,k): return k in self.idx
    def raw(self,k): return self.fh(k).get_tensor(k)
    def shape(self,k): return self.fh(k).get_slice(k).get_shape()
    def lin(self,prefix):
        """按名称取线性层权重，自动处理 FP8 / NVFP4 / bf16。"""
        k=prefix+".weight"
        if not self.has(k):
            raise KeyError(k)
        if prefix+".weight_scale_2" in self.idx:
            return deq_nvfp4(self.raw(k),self.raw(prefix+".weight_scale"),self.raw(prefix+".weight_scale_2")).to(DEV)
        if prefix+".weight_scale" in self.idx:
            return deq_fp8(self.raw(k),self.raw(prefix+".weight_scale")).to(DEV)
        return self.raw(k).to(DEV,DT)

def rmsnorm(x,w,eps):
    v=x.to(torch.float32)
    v=v*torch.rsqrt(v.pow(2).mean(-1,keepdim=True)+eps)
    v=v.to(x.dtype)
    return v*w if w is not None else v

def rope_apply(x,pos,theta,partial,head_dim):
    """x [T,H,D] neox style, partial rotary on first `partial*D` dims."""
    rd=int(head_dim*partial)
    if rd==0: return x
    half=rd//2
    inv=1.0/(theta**(torch.arange(0,half,dtype=torch.float32,device=x.device)/half))
    f=torch.outer(pos.float(),inv)
    cos=f.cos().to(x.dtype)[:,None,:]; sin=f.sin().to(x.dtype)[:,None,:]
    x1=x[...,:half]; x2=x[...,half:rd]
    o1=x1*cos-x2*sin; o2=x1*sin+x2*cos
    return torch.cat([o1,o2,x[...,rd:]],dim=-1)

def main():
    ids_json=sys.argv[1] if len(sys.argv)>1 else None
    ids=json.loads(ids_json) if ids_json else [2]+[100+i for i in range(8)]
    st=Store()
    cfg=json.loads(open(f"{SNAP}/config.json").read())
    tc=cfg["text_config"]
    H=tc["hidden_size"]; L=tc["num_hidden_layers"]; NH=tc["num_attention_heads"]
    NE=tc["num_experts"]; TK=tc["top_k_experts"]; MOEI=tc["moe_intermediate_size"]
    I=tc["intermediate_size"]; eps=tc["rms_norm_eps"]; V=tc["vocab_size"]
    lt=tc["layer_types"]; sw=tc["sliding_window"]
    ghd=tc["global_head_dim"]; hd=tc["head_dim"]
    nkv=tc["num_key_value_heads"]; ngkv=tc["num_global_key_value_heads"]
    rope=tc["rope_parameters"]

    emb=st.raw("model.language_model.embed_tokens.weight").to(DEV,DT)
    normalizer=torch.tensor(float(H)**0.5,dtype=DT,device=DEV)
    x=emb[torch.tensor(ids,device=DEV)]*normalizer       # [T,H]
    T=x.shape[0]; pos=torch.arange(T,device=DEV)

    for li in range(L):
        p=f"model.language_model.layers.{li}."
        is_full = lt[li]=="full_attention"
        hd_= ghd if is_full else hd
        nkv_= ngkv if is_full else nkv
        k_eq_v = tc.get("attention_k_eq_v",False) and is_full
        rp = rope["full_attention"] if is_full else rope["sliding_attention"]

        residual=x
        h=rmsnorm(residual,st.raw(p+"input_layernorm.weight").to(DEV,DT),eps)
        # ---- attention ----
        q=st.lin(p+"self_attn.q_proj")@h.T          # [NH*hd, T]
        q=q.T.view(T,NH,hd_)
        q=rmsnorm(q,st.raw(p+"self_attn.q_norm.weight").to(DEV,DT),eps)
        k=st.lin(p+"self_attn.k_proj")@h.T
        k=k.T.view(T,nkv_,hd_)
        k=rmsnorm(k,st.raw(p+"self_attn.k_norm.weight").to(DEV,DT),eps)
        q=rope_apply(q,pos,rp["rope_theta"],rp.get("partial_rotary_factor",1.0),hd_)
        k=rope_apply(k,pos,rp["rope_theta"],rp.get("partial_rotary_factor",1.0),hd_)
        if k_eq_v:
            v=k
        else:
            v=st.lin(p+"self_attn.v_proj")@h.T
            v=v.T.view(T,nkv_,hd_)
            v=rmsnorm(v,None,eps)
        # GQA + causal (+ sliding)
        rep=NH//nkv_
        kk=k.repeat_interleave(rep,dim=1).transpose(0,1)   # [NH,T,hd]
        vv=v.repeat_interleave(rep,dim=1).transpose(0,1)
        qq=q.transpose(0,1).to(torch.float32)              # [NH,T,hd]
        sc=qq@kk.to(torch.float32).transpose(1,2)          # scaling=1.0
        mask=torch.full((T,T),float("-inf"),device=DEV)
        mask=torch.triu(mask,diagonal=1)
        if not is_full:
            idx=torch.arange(T,device=DEV)
            mask=torch.where((idx[None,:]>=idx[:,None]-sw+1)|(idx[None,:]>idx[:,None]),mask,torch.zeros((),device=DEV))
        sc=sc+mask
        w=torch.softmax(sc,dim=-1).to(DT)
        o=(w@vv).transpose(0,1).reshape(T,NH*hd_)
        o=st.lin(p+"self_attn.o_proj")@o.T
        h=rmsnorm(o.T,st.raw(p+"post_attention_layernorm.weight").to(DEV,DT),eps)+residual
        residual=h
        # ---- dense MLP ----
        h2=rmsnorm(h,st.raw(p+"pre_feedforward_layernorm.weight").to(DEV,DT),eps)
        gu=st.lin(p+"mlp.gate_up_proj")@h2.T if st.has(p+"mlp.gate_up_proj.weight") else None
        if gu is None:
            g=(st.lin(p+"mlp.gate_proj")@h2.T).T
            u=(st.lin(p+"mlp.up_proj")@h2.T).T
        else:
            g,u=gu.T.split(I,dim=-1)
        act=torch.nn.functional.gelu(g,approximate="tanh")*u
        m=(st.lin(p+"mlp.down_proj")@act.T).T
        h1=rmsnorm(m,st.raw(p+"post_feedforward_layernorm_1.weight").to(DEV,DT),eps)
        # ---- MoE ----
        h3=rmsnorm(residual,st.raw(p+"pre_feedforward_layernorm_2.weight").to(DEV,DT),eps)
        r=rmsnorm(residual,None,eps)*(H**-0.5)
        r=r*st.raw(p+"router.scale").to(DEV,DT)
        logits=(st.raw(p+"router.proj.weight").to(DEV,DT)@r.T).T.to(torch.float32)
        pr=torch.softmax(logits,dim=-1)
        tw,ti=torch.topk(pr,TK,dim=-1)
        tw=tw/tw.sum(-1,keepdim=True)
        pes=st.raw(p+"router.per_expert_scale").to(DEV,DT).to(torch.float32)
        moe_out=torch.zeros(T,H,dtype=DT,device=DEV)
        for t in range(T):
            for kk_ in range(TK):
                e=int(ti[t,kk_]); ww=tw[t,kk_].to(DT)
                ep=f"{p}experts.{e}."
                gp=st.lin(ep+"gate_proj"); up=st.lin(ep+"up_proj"); dp=st.lin(ep+"down_proj")
                a=torch.nn.functional.gelu(gp@h3[t],approximate="tanh")*(up@h3[t])
                moe_out[t]+=(dp@a)*ww*pes[e]
        h2n=rmsnorm(moe_out,st.raw(p+"post_feedforward_layernorm_2.weight").to(DEV,DT),eps)
        h=rmsnorm(h1+h2n,st.raw(p+"post_feedforward_layernorm.weight").to(DEV,DT),eps)+residual
        h=h*st.raw(p+"layer_scalar").to(DEV,DT)
        x=h
        print(f"layer {li} ok", flush=True)
    xn=rmsnorm(x,st.raw("model.language_model.norm.weight").to(DEV,DT) if st.has("model.language_model.norm.weight") else None,eps)
    lgt=(emb.to(torch.float32)@xn[-1].to(torch.float32))
    lgt=torch.tanh(lgt/30.0)*30.0
    top=torch.topk(lgt,10)
    print("TOP10:",[(int(i),round(float(v),4)) for i,v in zip(top.indices,top.values)])
    torch.save({"logits":lgt.float().cpu(),"ids":ids}, "/w/ref_out.pt")

if __name__=="__main__": main()
