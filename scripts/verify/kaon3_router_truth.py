#!/usr/bin/env python3
"""用 fp64 精确真值判定 router GEMM 的精度归属（谁更准）。

不是"参考实现"，是**真值校验**：把 bf16 原始权重与 router 的 fp32 输入直接用 float64 做矩阵乘，
得到该形状下不受累加顺序影响的精确 logits，然后比较各实现偏离真值多少。

用法（在 5090 的容器 / 宿主上均可，只需 numpy + 两个 .bin）：
  python3 -I scripts/verify/kaon3_router_truth.py <model_dir> <ours/router_logits.bin> <ref/router_logits.bin> <router_in.bin> [T] [E]
<router_in.bin> 是 router 的**输入**（norm+root_size+scale 之后、proj 之前），两侧必须逐位一致才有意义。

exit 0 = 我们的 router 不劣于参照实现（逐元素更接近真值的比例 >= 50% 且 top-8 错得更少）
"""
import sys, struct, json, os
import numpy as np

model_dir, ours_p, ref_p, rin_p = sys.argv[1:5]
E = int(sys.argv[6]) if len(sys.argv) > 6 else 128

def rd(p):
    b = open(p, 'rb').read()
    return np.frombuffer(b, dtype='<f4')

ours = rd(ours_p); ref = rd(ref_p); rin = rd(rin_p)
T = int(sys.argv[5]) if len(sys.argv) > 5 else int(rin.size // 2816)
H = int(rin.size // T)
K = 8

key = 'model.language_model.layers.0.router.proj.weight'
idx = json.load(open(os.path.join(model_dir, 'model.safetensors.index.json')))
with open(os.path.join(model_dir, idx['weight_map'][key]), 'rb') as fh:
    n = struct.unpack('<Q', fh.read(8))[0]
    hdr = json.loads(fh.read(n))
    e = hdr[key]
    fh.seek(8 + n + e['data_offsets'][0])
    raw = fh.read(e['data_offsets'][1] - e['data_offsets'][0])
W = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(E, H)

X = rin.reshape(T, H).astype(np.float64)
exact = X @ W.astype(np.float64).T
mo = np.abs(ours.reshape(T, E).astype(np.float64) - exact)
mr = np.abs(ref.reshape(T, E).astype(np.float64) - exact)

def top8(v):
    return set(np.argsort(-v)[:K])

dm = sum(1 for t in range(T) if top8(ours.reshape(T, E)[t]) != top8(exact[t]))
dr = sum(1 for t in range(T) if top8(ref.reshape(T, E)[t]) != top8(exact[t]))
frac = float((mo < mr).mean())

print(f"rows={T} experts={E}")
print(f"OURS vs fp64: max={mo.max():.3e} mean={mo.mean():.3e}  top8 错 {dm}/{T}")
print(f"REF  vs fp64: max={mr.max():.3e} mean={mr.mean():.3e}  top8 错 {dr}/{T}")
print(f"我们更接近真值的元素比例: {frac * 100:.1f}%")
ok = frac >= 0.5 and dm <= dr
print("ROUTER TRUTH " + ("PASS（我们不劣于参照）" if ok else "FAIL（我们更差，需要查）"))
sys.exit(0 if ok else 1)
