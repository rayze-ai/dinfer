#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证：按 vLLM 的 `requantize_with_max_scale` 复现 qkv 合并权重，与 vLLM 实际用的逐字节比。

算法（`vllm/.../utils/w8a8_utils.py:76`）：
    max_w_scale = max(q,k,v 的 weight_scale)
    for 每个分片（logical_widths = 4096/2048/2048）:
        weight_dq = fp8_to_bf16(weight[start:end]) * 原 scale
        weight[start:end] = fp8_quant(weight_dq, max_w_scale)
    weight_scale = max_w_scale
    input_scale  = max(input_scale)

判定：三步。
  1. 复现出来的 `[8192, 2816]` 权重 vs **vLLM 前向里真正用的那份**（从 modelopt 处理后的 layer 取）
     逐字节相同；
  2. 用复现权重跑 `flashinfer_scaled_fp8_mm`，与 vLLM 前向的真实 qkv 输出逐位相同；
  3. 顺带确认 `input_scale` 是否真的是 max（三个都 0.59375，应当不变）。

跑法（容器内）：
    OUT=/w/qkvq PYTHONPATH=/w/dinfer python3 -I /w/dinfer/scripts/dev/qkv_requant_verify.py
"""
import os
import sys
import glob
import numpy as np
import torch

os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def main():
    out = os.environ.get("OUT", "/w/qkvq")
    os.makedirs(out, exist_ok=True)
    ids = [int(x) for x in open(os.environ["IDS"]).read().split()]
    model = os.environ["M"]

    from vllm import LLM, SamplingParams
    import vllm.utils.flashinfer as FIK

    cap = {}

    # 抓 vLLM 前向里真正喂给 fp8 GEMM 的 qkv 权重（处理后的合并权重）与输出
    orig = FIK.flashinfer_scaled_fp8_mm

    def patched(*args, **kw):
        r = orig(*args, **kw)
        a = kw.get("a", args[0] if len(args) > 0 else None)
        b = kw.get("b", args[1] if len(args) > 1 else None)
        sa = kw.get("scale_a", args[2] if len(args) > 2 else None)
        sb = kw.get("scale_b", args[3] if len(args) > 3 else None)
        if a is None or b is None or sa is None or sb is None:
            return r
        if "w" not in cap:
            # 第一个 fp8 mm 就是 qkv（图里顺序：input_norm → qkv_proj）
            cap["w"] = b.detach().clone()
            cap["sa"] = sa.detach().clone()
            cap["sb"] = sb.detach().clone()
            cap["a"] = a.detach().clone()
            cap["out"] = r.detach().clone()
            print(f"CAP qkv: a={tuple(a.shape)} b={tuple(b.shape)} "
                  f"sb={float(sb)} sa={float(sa)} out={tuple(r.shape)}", flush=True)
        return r

    FIK.flashinfer_scaled_fp8_mm = patched

    llm = LLM(model=model, enforce_eager=True, tensor_parallel_size=1,
              gpu_memory_utilization=0.85, dtype="auto", max_model_len=4096,
              enable_prefix_caching=False, block_size=32, max_num_seqs=1)
    llm.generate(prompts=[{"prompt_token_ids": ids}],
                 sampling_params=SamplingParams(max_tokens=1, temperature=0.0))

    if "w" not in cap:
        print("NO_CAPTURE")
        return
    W_vllm = cap["w"]           # [2816, 8192] 物理 [K, N]（FlashInfer 视图）
    sb = float(cap["sb"])
    sa = float(cap["sa"])
    print(f"vLLM 实际使用: weight_scale={sb:.16g}  input_scale={sa:.16g}", flush=True)

    # ---- 1. 从 checkpoint 复现 ----
    import safetensors.torch as st
    cand = glob.glob("/models/models--kaon*/snapshots/*/") + glob.glob("/hf_data/hub/models--kaon*/snapshots/*/")
    assert cand, "找不到模型快照目录"
    snap = cand[0]
    P = "model.language_model.layers.0.self_attn."
    W = {}
    for f in sorted(glob.glob(snap + "*.safetensors")):
        d = st.load_file(f)
        for k in d:
            if k.startswith(P):
                W[k[len(P):]] = d[k]

    names = ["q_proj", "k_proj", "v_proj"]
    widths = [4096, 2048, 2048]
    scales = [W[n + ".weight_scale"].float() for n in names]
    max_ws = torch.stack([t.cuda() for t in scales]).max()
    print(f"checkpoint 三个 weight_scale = {[float(s) for s in scales]}", flush=True)
    print(f"max = {float(max_ws):.16g}", flush=True)

    parts = []
    start = 0
    for n, lw, s in zip(names, widths, scales):
        wq = W[n + ".weight"].cuda()               # [lw, 2816] fp8
        wdq = wq.to(torch.float32) * s.cuda()      # 反量化
        # 用 max_ws 重新量化（与 vLLM `ops.scaled_fp8_quant` 同口径）
        requant = (wdq / max_ws).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        parts.append(requant)
        start += lw
    merged = torch.cat(parts, 0)                    # [8192, 2816]

    # 转成 FlashInfer 视图 [K, N] 物理（= merged.t() 连续）
    merged_kn = merged.t().contiguous()

    same_bytes = int((merged_kn.view(torch.uint8) == W_vllm.view(torch.uint8)).sum())
    total = W_vllm.numel()
    print(f"[1] 复现权重 vs vLLM 实际权重: 相同字节 {same_bytes}/{total}", flush=True)
    if same_bytes != total:
        diff = (merged_kn.view(torch.uint8) != W_vllm.view(torch.uint8))
        print(f"    不同 {int(diff.sum())} 字节（占比 {float(diff.float().mean()):.4%}）", flush=True)
        # 按 N 分段看
        for i, (n, lw) in enumerate(zip(names, widths)):
            seg = diff[:, i * 0:0] if False else None
        dcol = diff.float().mean(0)
        print(f"    按列均值：前16列 {[round(float(x),3) for x in dcol[:8]]}", flush=True)
    else:
        print("    ✅ 逐字节相同 —— requantize 算法复现正确", flush=True)

    # ---- 2. 同输入跑一次，与 vLLM 前向输出比 ----
    a = cap["a"]                                     # [1, T, K] fp8
    out_ref = cap["out"]
    mm_ws = torch.tensor(max_ws, dtype=torch.float32, device="cuda")
    mm_sa = torch.tensor(sa, dtype=torch.float32, device="cuda")
    try:
        out_ours = orig(a, merged_kn, mm_sa, mm_ws, out_dtype=out_ref.dtype)
        nd = int((out_ours.bfloat16() != out_ref.bfloat16()).sum())
        print(f"[2] 复现权重 GEMM vs vLLM 输出: 不同 {nd}/{out_ref.numel()}", flush=True)
    except Exception as e:
        print(f"[2] GEMM 复现失败: {str(e)[:120]}", flush=True)

    # ---- 3. input_scale ----
    iscales = [float(W[n + ".input_scale"]) for n in names]
    print(f"[3] checkpoint input_scale = {iscales}  max = {max(iscales):.16g}  "
          f"{'✅ 与 vLLM 一致' if abs(max(iscales) - sa) < 1e-12 else '❌ 不一致'}", flush=True)

    # 落盘供 Rust 侧用
    np.ascontiguousarray(merged.view(torch.uint8).cpu().numpy()).tofile(f"{out}/W_qkv_requant.bin")
    np.ascontiguousarray(a.view(torch.uint8).cpu().numpy()).tofile(f"{out}/A_fp8.bin")
    np.ascontiguousarray(out_ref.bfloat16().view(torch.int16).cpu().numpy()).tofile(f"{out}/ref_out.bin")
    open(f"{out}/meta.txt", "w").write(
        f"scale_a={sa:.16g}\nscale_b={float(max_ws):.16g}\nshapeA={tuple(a.shape)}\n")
    print(f"已写 {out}", flush=True)


if __name__ == "__main__":
    main()
