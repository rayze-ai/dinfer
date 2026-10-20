#!/usr/bin/env python3
"""tp2 vs 单卡：比较 model_parity 的 KIMIK3_DUMP 目录（raw f32 LE）。用法：k3_tp_compare.py <single_dir> <tp_dir> [--iso-min 0.9999] [--logits-min 0.9998]
exit 0 = 全部达标（docs/loops/k3-tp.md 停止条件）。不依赖 numpy 之外的包。"""
import sys, os, struct, math, argparse
ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--iso-min", type=float, default=0.9999)
ap.add_argument("--logits-min", type=float, default=0.9998)
ap.add_argument("--prefix", default="", help="只比以此开头的文件（如 iso1）")
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

names = sorted(n[:-4] for n in os.listdir(args.a) if n.endswith(".f32") and n.startswith(args.prefix))
ok = True
for n in names:
    pb = os.path.join(args.b, n + ".f32")
    if not os.path.exists(pb):
        print(f"  {n:<14} 缺少 tp 侧"); ok = False; continue
    a, b = load(os.path.join(args.a, n + ".f32")), load(pb)
    if len(a) != len(b):
        print(f"  {n:<14} 长度不同 {len(a)} vs {len(b)}"); ok = False; continue
    c = cos(a, b)
    mx = float(np.max(np.abs(a - b))) if np is not None else max(abs(x - y) for x, y in zip(a, b))
    line = f"  {n:<18} cos {c:.7f}  max_abs {mx:.3e}"
    if n.endswith("_topk_ids"):
        # 路由 id：按 token 行（K=16）比集合是否一致，只报不判
        K = 16; T = len(a) // K
        rows_same = sum(1 for i in range(T) if set(a[i*K:(i+1)*K]) == set(b[i*K:(i+1)*K]))
        print(f"  {n:<18} 路由整行一致 {rows_same}/{T}"); continue
    if n.endswith("_topk_w"):
        print(f"  {n:<18} max_abs {mx:.3e}（只报）"); continue
    if n.startswith("logits") or n.startswith("decode_logits"):
        ta, tb = top1(a), top1(b)
        line += f"  top-1 {ta} vs {tb}"
        if c < args.logits_min or ta != tb: ok = False; line += "  <-- FAIL"
    else:
        if c < args.iso_min: ok = False; line += "  <-- FAIL"
    print(line)
print("RESULT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
