#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 定标 `ops.scaled_fp4_quant` 的语义：给定已知输入 a，读回 (q, sf)，判定
#   反量化公式到底是 q·sf  还是 q·sf/gs。
# 设计：a 全 = 1.0。若公式是 q·sf/gs，则 q·sf/gs ≈ 1；若是 q·sf，则 q·sf ≈ 1。
import os, numpy as np, torch
def main():
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING","0")
    os.environ.setdefault("VLLM_BATCH_INVARIANT","1")
    from vllm import LLM
    from vllm import _custom_ops as ops
    llm = LLM(model=os.environ["M"], enforce_eager=True, tensor_parallel_size=1,
              gpu_memory_utilization=0.85, dtype="auto", max_model_len=4096,
              enable_prefix_caching=False, block_size=32, max_num_seqs=1)
    def dec(b):
        sign=-1.0 if (b>>7)&1 else 1.0; e=(b>>3)&0xF; m=b&7
        if e==0: return sign*m/8.0*2**-6
        if e==15: return 448.0 if m==7 else (1+m/8.0)*2**8
        return sign*(1+m/8.0)*2.0**(e-7)
    FP4=np.array([0,0.5,1,1.5,2,3,4,6]); f=lambda n: FP4[n&7] if n<8 else -FP4[n&7]
    for gs_val in (1.0, 100.0, 462.4516):
        M,K = 8, 2816
        a = torch.ones((M,K), dtype=torch.bfloat16, device='cuda')
        gs = torch.tensor([gs_val], dtype=torch.float32, device='cuda')
        q, sf = ops.scaled_fp4_quant(a, gs, True, "none", padded_n=K)
        # 还原 sf（swizzled）
        Mp, Kp = 128, ((K//16)+3)//4*4
        s = sf.view(torch.uint8).cpu().numpy().reshape(Mp, Kp)
        sval = np.vectorize(dec)(s[:M, :K//16])
        qn = q.view(torch.uint8).cpu().numpy()
        qv = np.zeros((M,K))
        qv[:,0::2] = np.vectorize(lambda v: f(int(v)&0xF))(qn)
        qv[:,1::2] = np.vectorize(lambda v: f(int(v)>>4))(qn)
        for nm, deq in (("q*sf", qv*sval.repeat(16,1)),
                        ("q*sf/gs", qv*sval.repeat(16,1)/gs_val)):
            print(f"QCALIB gs={gs_val:9.3f} {nm:8s} mean={deq.mean():.6f} (期望 1.0)", flush=True)
        print(flush=True)
if __name__=='__main__':
    main()
