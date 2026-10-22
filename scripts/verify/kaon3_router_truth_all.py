#!/usr/bin/env python3
"""逐层 router fp64 真值扫描：两个实现各自对照**自己输入**的精确 logits，比谁更准、谁选错专家。

不是参考实现，是真值校验（同 `kaon3_router_truth.py`，扩到全部层、多个用例）。
每个实现的 router 输入可以不同（整跑各自累积）：判的是 router GEMM 本身的精度，
以及"以该实现自己的输入为准，top-8 专家集合选没选对"。

用法：python3 -I scripts/verify/kaon3_router_truth_all.py <model_dir> <ours_root> <ref_root> <case...>
  ours_root/<case>/sub_L{l}/router_{in,logits}.bin      （kaon3_probe SUB_LAYER=all --hidden-all）
  ref_root/<case>/L{l}_router_{in,logits}.bin           （scripts/dev/kaon3_vllm_router_all.py）
exit 0 = 每层我们的 max 偏差不大于参照，且全部用例合计我们选错的 token 数不多于参照。
"""
import sys, struct, json, os
import numpy as np

model_dir, ours_root, ref_root = sys.argv[1:4]
cases = sys.argv[4:]
E, H, K = 128, 2816, 8
idx = json.load(open(os.path.join(model_dir, 'model.safetensors.index.json')))['weight_map']

def router_w(l):
    key = f'model.language_model.layers.{l}.router.proj.weight'
    with open(os.path.join(model_dir, idx[key]), 'rb') as fh:
        n = struct.unpack('<Q', fh.read(8))[0]
        e = json.loads(fh.read(n))[key]
        fh.seek(8 + n + e['data_offsets'][0])
        raw = fh.read(e['data_offsets'][1] - e['data_offsets'][0])
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(E, H).astype(np.float64)

def rd(p):
    return np.fromfile(p, dtype='<f4').astype(np.float64)

def judge(x, y, W):
    T = x.size // H
    ex = x.reshape(T, H) @ W.T
    y = y.reshape(T, E)
    wrong = sum(set(np.argsort(-y[t])[:K]) != set(np.argsort(-ex[t])[:K]) for t in range(T))
    return np.abs(y - ex).max(), wrong, T

tot = {'ours': 0, 'ref': 0, 'tok': 0}
worse_layers = []
print(f"{'L':>3} {'ours max':>10} {'ref max':>10} {'ours错':>6} {'ref错':>6}  (tokens)")
for l in range(30):
    W = router_w(l)
    om = rm = 0.0; ow = rw = nt = 0
    for c in cases:
        po = os.path.join(ours_root, c, f'sub_L{l}')
        pr = os.path.join(ref_root, c)
        m1, w1, T = judge(rd(f'{po}/router_in.bin'), rd(f'{po}/router_logits.bin'), W)
        m2, w2, _ = judge(rd(f'{pr}/L{l}_router_in.bin'), rd(f'{pr}/L{l}_router_logits.bin'), W)
        om, rm, ow, rw, nt = max(om, m1), max(rm, m2), ow + w1, rw + w2, nt + T
    tot['ours'] += ow; tot['ref'] += rw; tot['tok'] += nt
    if om > rm:
        worse_layers.append(l)
    print(f"{l:>3} {om:>10.3e} {rm:>10.3e} {ow:>6} {rw:>6}  ({nt})")
print(f"TOTAL ours_wrong={tot['ours']} ref_wrong={tot['ref']} token_layers={tot['tok']} ours_worse_layers={worse_layers}")
ok = not worse_layers and tot['ours'] <= tot['ref']
print("ROUTER_TRUTH", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
