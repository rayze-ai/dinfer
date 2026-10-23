#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 极小规模定标：用手工构造的 fp4 数据调参照的 cutlass dense op，反推 alpha 的确切作用。
# 设计：A 的 fp4 值全 1.0（q=2 → e2m1 的 1.0），W 的 fp4 值全 1.0，scale 全 C。
# 那么 y = (1·C_a)·(1·C_w) 累加 k 次 · alpha  →  理论值 = k · C_a · C_w · alpha
# 用 k=64（一个 k-block 的 4 倍）与不同 alpha 各跑一次，直接读出乘子位置。
import os, numpy as np, torch
# ---- 第二段：钉死「块缩放排布映射」----
# 设计：A 的 fp4 只在第 0 个 16-元素块里为 1.0，其余为 0；A 的 4 个块缩放分别给 1/2/4/8。
# 那么 y = 16 · sf[0] · (W 侧 1·1) · alpha。读出 y/(16·alpha) 就知道「第 0 块」实际取到了哪个 sf。
def probe_scale_layout():
    from vllm import _custom_ops as ops
    import torch
    m, n, k = 1, 128, 64
    # q: 第 0 块（前 16 个元素 = 前 8 字节）全 1.0(0x22)，其余 0
    aq = torch.zeros((m, k // 2), dtype=torch.uint8, device='cuda')
    aq[:, :8] = 0x22
    wq = torch.full((n, k // 2), 0x22, dtype=torch.uint8, device='cuda')
    # A 的 4 个块缩放（自然序）= 1,2,4,8 的 e4m3 编码
    import numpy as np
    def e4(v):
        for b in range(0x38, 0x7f):
            pass
        return None
    # e4m3: 1.0=0x38, 2.0=0x40, 4.0=0x48, 8.0=0x50
    nat = np.array([[0x38, 0x40, 0x48, 0x50]], dtype=np.uint8)   # [m=1, k/16=4]
    padded = np.zeros((128, 4), dtype=np.uint8); padded[:1, :] = nat
    sw = padded.reshape(1, 1, 4, 32, 1, 4).transpose(0, 1, 4, 3, 2, 5).reshape(128, 4)
    asq = torch.from_numpy(sw.copy()).cuda().view(torch.float8_e4m3fn)
    wsf = torch.full((n, k // 16), 0x38, dtype=torch.uint8, device='cuda').view(torch.float8_e4m3fn)
    for alpha in (1.0,):
        out = ops.cutlass_scaled_fp4_mm(aq, wq, asq, wsf,
                                        torch.tensor([alpha], dtype=torch.float32, device='cuda'),
                                        torch.bfloat16)
        v = out[0, 0].float().item()
        got = v / (16 * alpha)
        print(f"LAYOUT alpha={alpha} y={v:.4f}  y/(16a)={got:.4f}  "
              f"(1.0→块0取sf[0]=1; 2/4/8→取到了别的块)", flush=True)

def main():
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING","0")
    os.environ.setdefault("VLLM_BATCH_INVARIANT","1")
    from vllm import LLM
    from vllm import _custom_ops as ops
    M=os.environ["M"]
    # 建个最小引擎拿到 ops（不需要真跑前向）
    llm=LLM(model=M, enforce_eager=True, tensor_parallel_size=1, gpu_memory_utilization=0.85,
            dtype="auto", max_model_len=4096, enable_prefix_caching=False, block_size=32, max_num_seqs=1)
    m,n,k=1,128,64
    # A: fp4 全 1.0 → e2m1 编码 1.0 = 0b0010 = 2（正），两个 nibble 打包成 0x22
    aq=torch.full((m,k//2), 0x22, dtype=torch.uint8, device='cuda')
    # A 的块缩放：k/16=4 个块，每个 1.0（e4m3 1.0 = 0x38）
    # 注意 cutlass 要 padded+swizzled：Mp=128, Kp=4
    # swizzle: (g, a4, r32, kO, k4) -> transpose(0,1,4,3,2,5)
    # 这里 M_pad=128, K_pad=4 → (1, 1, 4, 32, 1, 4)
    asf=np.zeros((128,4),dtype=np.uint8); asf[:m,:]=0x38
    sw=asf.reshape(1,1,4,32,1,4).transpose(0,1,4,3,2,5).reshape(128,4)
    # scale 必须是 float8_e4m3fn（用 uint8 视图装同样的位）
    asq=torch.from_numpy(sw.copy()).cuda().view(torch.float8_e4m3fn)
    # W: fp4 全 1.0
    wq=torch.full((n,k//2), 0x22, dtype=torch.uint8, device='cuda')
    wsf=torch.full((n,k//16), 0x38, dtype=torch.uint8, device='cuda').view(torch.float8_e4m3fn)  # 自然布局 [n,k/16]
    for alpha in (1.0, 2.0, 0.5):
        out=ops.cutlass_scaled_fp4_mm(aq, wq, asq, wsf, torch.tensor([alpha],dtype=torch.float32,device='cuda'), torch.bfloat16)
        v=out[0,0].float().item()
        print(f"CALIB alpha={alpha:5.2f}  y={v:12.4f}   y/alpha={v/alpha:10.4f}   (理论 k=64)", flush=True)
    print("CALIB 若 y/alpha 恒为 64 → alpha 是唯一乘子且作用在最后", flush=True)
    probe_scale_layout()
if __name__=='__main__':
    main()

