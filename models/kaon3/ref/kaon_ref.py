#!/usr/bin/env python3
"""Kaon-V3 参考实现：CPU 上把 NVFP4 专家 + FP8 线性层去量化成 bf16，跑 Gemma4TextModel，
输出指定 prompt 的 logits 与 top-k，用于与基线服务对拍。"""
import os,sys,json,time
os.environ["HF_HUB_OFFLINE"]="1"
import torch
from safetensors import safe_open

SNAP="/models/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478"
DEV=os.environ.get("DEV","cpu")
DTYPE=torch.bfloat16

E2M1=torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6.],dtype=torch.float32)
def unpack_nvfp4(packed:torch.Tensor)->torch.Tensor:
    """U8 [rows, cols/2] -> fp32 [rows, cols]；低半字节 = 偶数列。"""
    lo=(packed & 0x0F).to(torch.int64)
    hi=(packed >> 4).to(torch.int64)
    out=torch.empty(packed.shape[0],packed.shape[1]*2,dtype=torch.float32)
    out[:,0::2]=E2M1[lo & 7]*torch.where((lo>>3)==0,1.,-1.) if False else E2M1[lo & 7]*torch.sign(1-2*(lo>>3).float())
    out[:,1::2]=E2M1[hi & 7]*torch.sign(1-2*(hi>>3).float())
    # 上面 sign 写法等价：指数最高位是符号
    return out

def dequant_nvfp4(packed,scale_f8,scale2):
    """packed U8[rows,cols/2], scale_f8 F8_E4M3 [rows, cols/16], scale2 fp32 标量。"""
    w=unpack_nvfp4(packed)                      # [r,c] fp32
    s=scale_f8.to(torch.float32)                # [r, c/16]
    s=s.repeat_interleave(16,dim=1)[:,:w.shape[1]]
    return (w*s*float(scale2)).to(DTYPE)

def load_all():
    idx=json.load(open(f"{SNAP}/model.safetensors.index.json"))["weight_map"]
    files=sorted(set(idx.values()))
    t={}
    for f in files:
        with safe_open(f"{SNAP}/{f}","pt") as fh:
            for k in fh.keys(): t[k]=fh.get_tensor(k)
    return t

if __name__=="__main__":
    st=load_all()
    print("tensors",len(st))
    k="model.language_model.layers.0.experts.0.gate_proj.weight"
    print(k, st[k].shape, st[k].dtype)
    w=dequant_nvfp4(st[k], st[k.replace(".weight",".weight_scale")], st[k.replace(".weight",".weight_scale_2")])
    print("dequant ok", w.shape, w.dtype, w.float().abs().mean().item())
