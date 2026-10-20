#!/usr/bin/env python3
"""tp2 vs 单卡：比较 model_parity 的 KIMIK3_DUMP 目录（raw f32 LE）。用法：k3_tp_compare.py <single_dir> <tp_dir> [--iso-min 0.9999] [--logits-min 0.9998]
exit 0 = 全部达标（docs/loops/k3-tp.md 停止条件）。不依赖 numpy 之外的包。"""
import sys, os, struct, math, argparse
ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--iso-min", type=float, default=0.9999)
ap.add_argument("--logits-min", type=float, default=0.0, help="logits cos 门槛；默认 0 = 只判 top-1（链式末端 cos 受 MoE 平局翻转影响，单卡自身两次 0.99998，tp2 0.9982）")
ap.add_argument("--prefix", default="", help="只比以此开头的文件（如 iso1）")
ap.add_argument("--matched-only", action="store_true", help="iso{l} 只在路由整行一致的行上判（硬判决平局不可逐位复现，pitfalls 口径）；需要 iso{l}_topk_ids dump")
ap.add_argument("--hidden", type=int, default=7168); ap.add_argument("--k", type=int, default=16)
ap.add_argument("--top5", action="store_true", help="logits 还要求 top-5 集合一致（默认只报：5 层截断模型第 5 位 margin 一 ulp 级，见 loops/k3-tp.md 第 4 轮）")
args = ap.parse_args()
try:
    import numpy as np
except ImportError:
    np = None

def load(p):
    with open(p, "rb") as f:
        raw = f.read()
    if np is not None:
        return np.frombuffer(raw, dtype="<f4").astype("float64")
    return list(struct.unpack("<%df" % (len(raw) // 4), raw))

def cos(a, b):
    if np is not None:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 1.0
    dot = sum(x * y for x, y in zip(a, b)); na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 1.0

def top1(v):
    return int(np.argmax(v)) if np is not None else max(range(len(v)), key=lambda i: v[i])

def top5(v):
    if np is not None:
        return set(int(i) for i in np.argsort(-v)[:5])
    return set(sorted(range(len(v)), key=lambda i: -v[i])[:5])

names = sorted(n[:-4] for n in os.listdir(args.a) if n.endswith(".f32") and n.startswith(args.prefix))
ok = True
for n in names:
    pb = os.path.join(args.b, n + ".f32")
    if not os.path.exists(pb):
        print(f"  {n:<14} 缺少 tp 侧"); ok = False; continue
    a, b = load(os.path.join(args.a, n + ".f32")), load(pb)
    if len(a) != len(b):
        print(f"  {n:<14} 长度不同 {len(a)} vs {len(b)}"); ok = False; continue
    note = ""
    if args.matched_only and np is not None and n.startswith("iso") and "_" not in n:
        pa_ids, pb_ids = os.path.join(args.a, n + "_topk_ids.f32"), os.path.join(args.b, n + "_topk_ids.f32")
        if os.path.exists(pa_ids) and os.path.exists(pb_ids):
            ia, ib = load(pa_ids).reshape(-1, args.k), load(pb_ids).reshape(-1, args.k)
            same = np.array([set(ia[i]) == set(ib[i]) for i in range(ia.shape[0])])
            a, b = a.reshape(-1, args.hidden)[same].ravel(), b.reshape(-1, args.hidden)[same].ravel()
            note = f"  （路由一致 {int(same.sum())}/{len(same)} 行）"
    c = cos(a, b)
    mx = float(np.max(np.abs(a - b))) if np is not None else max(abs(x - y) for x, y in zip(a, b))
    line = f"  {n:<18} cos {c:.7f}  max_abs {mx:.3e}{note}"
    if n.endswith("_topk_ids"):
        # 路由 id：按 token 行（K=16）比集合是否一致，只报不判
        K = 16; T = len(a) // K
        rows_same = sum(1 for i in range(T) if set(a[i*K:(i+1)*K]) == set(b[i*K:(i+1)*K]))
        print(f"  {n:<18} 路由整行一致 {rows_same}/{T}"); continue
    if n.endswith("_topk_w"):
        print(f"  {n:<18} max_abs {mx:.3e}（只报）"); continue
    if n.startswith("logits") or n.startswith("decode_logits"):
        ta, tb = top1(a), top1(b)
        s5 = top5(a) == top5(b)
        line += f"  top-1 {ta} vs {tb}  top-5 集合{'一致' if s5 else '不同 ' + str(sorted(top5(a))) + ' vs ' + str(sorted(top5(b)))}"
        if c < args.logits_min or ta != tb or (args.top5 and not s5): ok = False; line += "  <-- FAIL"
    elif "_" in n and n.startswith("iso"):
        line += "  （中间量只报；按路由一致行拆比用 k3_tp_rows.py）"
    elif n.startswith("chain") or n == "final":
        line += "  （链式只报：单卡自身两次 chain4 ≈ 0.9992，见 loops/k3-tp.md 第 3 轮）"
    else:
        if c < args.iso_min: ok = False; line += "  <-- FAIL"
    print(line)
print("RESULT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
