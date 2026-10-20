#!/usr/bin/env python3
"""按 prompt 长度分组的 prefill 步 kernel 分解（r11 第一步，engine-gap 8z）。

输入：nsys export 的 sqlite（`q35_nsys_prefill.sh` 产出，每长度 5 个单发 max_tokens=1 请求 → 5 个纯 prefill 步）。
做法：按 sample/argmax kernel 切步 → 只取含 prefill kernel（wy_/ho_/conv1d_prefill/gqa_fa/prep）的步 →
按步 span 聚成 K 簇（默认 3 = 1K / 3K / 6K）→ 每簇打印 kernel 之和、每 kernel 的 µs/步与占比、按家族汇总。
用法：python3 -I nsys_prefill_by_len.py <sqlite> [clusters=3] [top=14]
"""
import sqlite3, sys, collections

PRE = ("wy_", "ho_", "conv1d_prefill", "gqa_fa", "prep_kernel")
FAM = [("GEMM", ("nvjet", "cublas", "gemm", "splitK")), ("GDN wy", ("wy_",)), ("GDN ho", ("ho_",)),
       ("GDN conv", ("conv1d",)), ("GDN prep/norm/cumsum", ("prep_kernel", "norm_kernel", "cumsum")),
       ("attn gqa_fa", ("gqa_fa",)), ("rmsnorm", ("rmsnorm",)), ("silu/act", ("silu", "gate_kernel", "mul_sigmoid")),
       ("qkv_post/rope", ("qkv_post", "rope")), ("other", ())]


def fam(n):
    for f, keys in FAM:
        if any(k in n for k in keys):
            return f
    return "other"


def main():
    path = sys.argv[1]
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 14
    c = sqlite3.connect(path)
    t0 = c.execute("select max(end) from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=1 and bytes>=8388608").fetchone()[0] or 0
    ks = c.execute("select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id "
                   "where k.start>? order by k.start", (t0,)).fetchall()
    steps, cur = [], []
    for k in ks:
        cur.append(k)
        if "sample_fused" in k[2] or "argmax" in k[2]:
            steps.append(cur); cur = []
    pre = [s for s in steps if any(any(x in k[2] for x in PRE) for k in s)]
    if not pre:
        print("no prefill steps"); return
    spans = sorted((s[-1][1] - s[0][0]) / 1e6 for s in pre)
    # 简单 1-D K 均值（按 span），初值取分位点
    cents = [spans[int(len(spans) * (i + 0.5) / K)] for i in range(K)]
    for _ in range(20):
        groups = [[] for _ in range(K)]
        for s in pre:
            sp = (s[-1][1] - s[0][0]) / 1e6
            groups[min(range(K), key=lambda i: abs(sp - cents[i]))].append(s)
        cents = [sum((s[-1][1] - s[0][0]) / 1e6 for s in g) / len(g) if g else cents[i] for i, g in enumerate(groups)]
    order = sorted(range(K), key=lambda i: cents[i])
    for gi in order:
        g = groups[gi]
        if not g:
            continue
        n = len(g)
        span = sum((s[-1][1] - s[0][0]) for s in g) / n / 1e6
        tot = collections.Counter(); cnt = collections.Counter()
        for s in g:
            for st, en, nm in s:
                tot[nm] += en - st; cnt[nm] += 1
        busy = sum(tot.values()) / n / 1e6
        print(f"\n=== 簇 span≈{cents[gi]:.0f} ms：{n} 步，span {span:.1f} ms，kernel 之和 {busy:.1f} ms（GPU 忙 {busy/span*100:.0f}%），{sum(cnt.values())/n:.0f} kernel/步")
        famt = collections.Counter()
        for nm, d in tot.items():
            famt[fam(nm)] += d
        print("  按家族（ms/步，占 kernel 之和 %）：" + "；".join(f"{f} {d/n/1e6:.1f} ({d/n/1e6/busy*100:.0f}%)" for f, d in famt.most_common()))
        for nm, d in tot.most_common(top):
            print(f"  {d/n/1e3:8.0f}us {cnt[nm]/n:6.1f}/step {d/cnt[nm]/1e3:7.1f}us/call  {d/n/1e6/busy*100:4.1f}%  {nm[:70]}")


if __name__ == "__main__":
    main()
