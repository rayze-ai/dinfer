#!/usr/bin/env python3
"""tp vs 单卡 MoE 定位：按"路由整行一致 / 不一致"拆行比 mlp_in / expert_sum / routed_out / iso。
用法：k3_tp_rows.py <single_dump> <tp_dump> --layers 1,2,3,4 [--hidden 7168 --latent 3584 --k 16]"""
import argparse, os, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--layers", default="1,2,3,4"); ap.add_argument("--hidden", type=int, default=7168)
ap.add_argument("--latent", type=int, default=3584); ap.add_argument("--k", type=int, default=16)
args = ap.parse_args()
ld = lambda d, n: np.fromfile(os.path.join(d, n + ".f32"), dtype="<f4").astype(np.float64)
def cosmax(x, y):
    if x.size == 0: return float("nan"), float("nan")
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30)), float(np.max(np.abs(x - y)))
for l in [int(x) for x in args.layers.split(",")]:
    ia = ld(args.a, f"iso{l}_topk_ids").reshape(-1, args.k); ib = ld(args.b, f"iso{l}_topk_ids").reshape(-1, args.k)
    T = ia.shape[0]
    same = np.array([set(ia[i]) == set(ib[i]) for i in range(T)])
    print(f"== layer {l}: 路由整行一致 {same.sum()}/{T}；不一致行 {np.where(~same)[0].tolist()}")
    for name, w in [("mlp_in", args.hidden), ("expert_sum", args.latent), ("routed_out", args.hidden), ("", args.hidden)]:
        key = f"iso{l}_{name}" if name else f"iso{l}"
        xa = ld(args.a, key).reshape(T, w); xb = ld(args.b, key).reshape(T, w)
        cm, mm = cosmax(xa[same].ravel(), xb[same].ravel()); cu, mu = cosmax(xa[~same].ravel(), xb[~same].ravel())
        nz = int((np.abs(xa[same] - xb[same]) > 0).sum())
        print(f"   {key:<18} 一致行 cos {cm:.7f} max_abs {mm:.3e} (非零差元素 {nz}/{xa[same].size})   不一致行 cos {cu:.6f} max_abs {mu:.3e}")
    # 不一致行里 top-16 分差：看是否平局（用 topk_w 代理：取两边权重最小值）
    wa = ld(args.a, f"iso{l}_topk_w").reshape(T, args.k); wb = ld(args.b, f"iso{l}_topk_w").reshape(T, args.k)
    for i in np.where(~same)[0]:
        da = set(ia[i]) - set(ib[i]); db = set(ib[i]) - set(ia[i])
        print(f"   行 {i}: 仅单卡选 {sorted(int(e) for e in da)} 仅tp选 {sorted(int(e) for e in db)}  最小权重 单卡 {wa[i].min():.4f} tp {wb[i].min():.4f}")
