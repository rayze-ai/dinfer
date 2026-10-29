#!/usr/bin/env python3
"""两份 nsys sqlite 的 kernel 家族对比（稳态窗口）。用法：
  python3 -I nsys_family_compare.py <ours.sqlite> <vllm.sqlite> [--ours-stream N] [--skip-s 20]
两边都只取 graphId 为空（eager）的 kernel、且 start > 第一个 kernel + skip-s 秒；按家族汇总 ms 与 kernel 数，并给每边 top-10 kernel。"""
import sqlite3, sys, collections
args = sys.argv[1:]
def opt(name, default=None):
    if name in args:
        i = args.index(name); v = args[i + 1]; del args[i:i + 2]; return v
    return default
ours_stream = opt("--ours-stream"); skip = float(opt("--skip-s", "20"))
FAM = [("moe_gemm", ("grouped_gemm", "pipe4", "p64", "moe", "cutlass_fp4", "fp4_group", "nvfp4", "Fp4", "Nvfp4", "block_scaled")),
       ("lin_gemm", ("nvjet", "cutlass_80", "Kernel2", "sm120_xmma", "cutlass3", "gemm_fp8", "scaled_mm", "fp8")),
       ("attn", ("gqa", "flash", "fmha", "attention", "fa_", "paged", "FlashAttn", "flashinfer", "Attn")),
       ("route", ("route", "topk", "moe_align", "softmax")), ("norm", ("norm", "rms", "RMS")), ("quant", ("quant", "cvt", "scaled_fp")),
       ("elem", ("elementwise", "vectorized", "fill", "copy", "add", "mul", "act", "silu", "gelu", "rotary", "rope", "embedding", "index", "gather", "scatter", "cat"))]
def fam(n):
    for f, keys in FAM:
        if any(k.lower() in n.lower() for k in keys): return f
    return "other"
def load(db, stream=None):
    c = sqlite3.connect(db); K = "CUPTI_ACTIVITY_KIND_KERNEL"
    cols = {r[1] for r in c.execute(f"pragma table_info({K})")}
    g = "k.graphId" if "graphId" in cols else "NULL"
    rows = c.execute(f"select k.start, k.end, s.value, k.streamId, {g} from {K} k join StringIds s on k.shortName = s.id order by k.start").fetchall()
    t0 = rows[0][0] + int(skip * 1e9)
    out = [(a, b, n) for a, b, n, sid, gid in rows if a > t0 and gid in (0, None) and (stream is None or sid == int(stream))]
    return out, (rows[-1][1] - t0) / 1e9
for label, db, st in (("OURS", sys.argv[1], ours_stream), ("VLLM", sys.argv[2], None)):
    ks, span = load(db, st)
    tot = sum(b - a for a, b, _ in ks) / 1e6
    f = collections.defaultdict(lambda: [0, 0.0]); k = collections.defaultdict(lambda: [0, 0.0])
    for a, b, n in ks:
        f[fam(n)][0] += 1; f[fam(n)][1] += (b - a) / 1e6; k[n][0] += 1; k[n][1] += (b - a) / 1e6
    print(f"== {label}: window {span:.1f}s eager kernels={len(ks)} eager_ms={tot:.0f} ({100*tot/(span*1e3):.0f}% of window)")
    print("  family: " + " ".join(f"{n}={v[1]:.0f}ms({100*v[1]/max(tot,1e-9):.0f}%,n={v[0]})" for n, v in sorted(f.items(), key=lambda x: -x[1][1])))
    for n, (cnt, ms) in sorted(k.items(), key=lambda x: -x[1][1])[:10]:
        print(f"  {ms:7.1f}ms {100*ms/max(tot,1e-9):5.1f}% n={cnt:6d} {1e3*ms/cnt:8.1f}us  {n[:90]}")
