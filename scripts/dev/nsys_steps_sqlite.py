#!/usr/bin/env python3
"""按步分类 nsys 结果（直接读 `nsys export --type sqlite` 的库，不经 csv），只统计**稳态窗口**。

engine-gap 第 6 轮的误读：按 sample kernel 切步后把加载期（权重 H2D、自调优、图捕获）的 memcpy / kernel 摊进了
"decode 步"，得出 "H2D 占 31%"。本脚本：
  * 稳态窗口 = 最后一次 ≥ 8 MB 的 H2D 之后（权重装载结束）→ 最后一个 kernel；
  * 步 = 相邻两个采样 kernel（sample_fused / argmax）之间的 GPU 活动；按 kernel 名判 prefill / decode / mixed；
  * 每类输出：步数、span（首 kernel 起到末 kernel 止）、kernel 之和、memcpy（H2D / D2H / D2D 次数与 GPU 时间）、
    步末到下一步首 kernel 的空档（p50 / p99）；
  * 另给窗口级账：busy / idle，以及 > 0.3 ms 空档叠 host API（cuMemAlloc* / cuLibraryLoadData / cuGraphInstantiate*）。
用法：nsys export --type sqlite -o x.sqlite x.nsys-rep && python3 nsys_steps_sqlite.py x.sqlite [topN]
"""
import sqlite3, sys, collections, bisect

PRE = ("wy_", "ho_", "conv1d_prefill", "gqa_fa", "prep_kernel")
DEC = ("recurrent_decode", "gqa_decode", "conv1d_update")
IDLE_MS = 10  # 步内空档超过它视为无请求空闲（不计入 span）

def main():
    path = sys.argv[1]
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    c = sqlite3.connect(path)
    t0 = c.execute("select max(end) from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=1 and bytes>=8388608").fetchone()[0] or 0
    t1 = c.execute("select max(end) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    print(f"steady window: {(t1 - t0) / 1e9:.2f} s (after last >=8MB H2D)")
    ks = c.execute(
        "select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id where k.start>? order by k.start",
        (t0,),
    ).fetchall()
    mc = c.execute("select start,end,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where start>? order by start", (t0,)).fetchall()
    # 切步
    steps, cur = [], []
    for k in ks:
        cur.append(k)
        if "sample_fused" in k[2] or "argmax" in k[2]:
            steps.append(cur)
            cur = []
    def cls(s):
        p = any(any(x in k[2] for x in PRE) for k in s)
        d = any(any(x in k[2] for x in DEC) for k in s)
        return "mixed" if p and d else ("prefill" if p else "decode")
    by = collections.defaultdict(list)
    for s in steps:
        by[cls(s)].append(s)
    mstarts = [m[0] for m in mc]
    for tag in ("decode", "prefill", "mixed"):
        ss = by.get(tag, [])
        if not ss:
            print(f"[{tag}] none")
            continue
        n = len(ss)
        span = busy = 0
        cnt = collections.Counter(); tot = collections.Counter()
        mem = collections.Counter(); memt = collections.Counter()
        gaps = []
        spans = []; idle_n = 0; idle_t = 0
        for i, s in enumerate(ss):
            a, b = s[0][0], s[-1][1]
            # 步内 > IDLE_MS 的空档 = 无请求空闲（sample 之后的尾 kernel 被归到下一步，使下一步"跨过"空闲期），
            # 不算 span；单独计数
            sp = b - a
            for j in range(1, len(s)):
                g = s[j][0] - s[j - 1][1]
                if g > IDLE_MS * 1e6:
                    sp -= g; idle_n += 1; idle_t += g
            span += sp; spans.append(sp)
            for k in s:
                nm = k[2].split("<")[0][:46]
                busy += k[1] - k[0]; cnt[nm] += 1; tot[nm] += k[1] - k[0]
            for j in range(bisect.bisect_left(mstarts, a), len(mc)):
                if mc[j][0] >= b:
                    break
                kind = {1: "H2D", 2: "D2H", 8: "D2D"}.get(mc[j][2], str(mc[j][2]))
                mem[kind] += 1; memt[kind] += mc[j][1] - mc[j][0]
        # 步末空档：到下一步（任意类）首 kernel
        idx = {id(s): i for i, s in enumerate(steps)}
        for s in ss:
            i = idx[id(s)]
            if i + 1 < len(steps):
                gaps.append(steps[i + 1][0][0] - s[-1][1])
        gaps.sort()
        g50 = gaps[len(gaps) // 2] / 1e3 if gaps else 0
        g99 = gaps[int(len(gaps) * 0.99)] / 1e3 if gaps else 0
        spans.sort()
        print(f"[{tag}] steps={n} span_ms={span / n / 1e6:.2f} (p50 {spans[n // 2] / 1e6:.2f}, p90 {spans[int(n * 0.9)] / 1e6:.2f}) "
              f"kernel_sum_ms={busy / n / 1e6:.2f} kernels/step={sum(cnt.values()) / n:.0f} gap_to_next_us p50={g50:.0f} p99={g99:.0f}"
              + (f"; excluded {idle_n} idle gaps >{IDLE_MS}ms = {idle_t / 1e6:.0f} ms" if idle_n else ""))
        print("   memcpy/step: " + ", ".join(f"{k} {mem[k] / n:.1f}x {memt[k] / n / 1e3:.0f}us" for k in sorted(mem)))
        for nm, d in tot.most_common(top):
            print(f"  {d / n / 1e3:7.0f}us {cnt[nm] / n:6.1f}/step {d / cnt[nm] / 1e3:7.1f}us {nm}")
    # 窗口级 busy / idle + 空档归因
    ev = sorted([(k[0], k[1]) for k in ks] + [(m[0], m[1]) for m in mc])
    busy = 0; gaps = []; cs, ce = ev[0]
    for s, e in ev[1:]:
        if s > ce:
            busy += ce - cs
            if s - ce > 300_000:
                gaps.append((ce, s))
            cs, ce = s, e
        else:
            ce = max(ce, e)
    busy += ce - cs
    W = t1 - t0
    print(f"window busy {busy / 1e6:.0f} ms / idle {(W - busy) / 1e6:.0f} ms; gaps>0.3ms: {len(gaps)} = {sum(e - s for s, e in gaps) / 1e6:.0f} ms")
    api = c.execute(
        "select a.start,a.end,s.value from CUPTI_ACTIVITY_KIND_RUNTIME a join StringIds s on a.nameId=s.id where a.start>? and a.end-a.start>100000 order by a.start",
        (t0,),
    ).fetchall()
    att = collections.Counter(); attn = collections.Counter()
    starts = [a[0] for a in api]
    for gs, ge in gaps:
        i = bisect.bisect_left(starts, gs - 500_000_000)
        while i < len(api) and api[i][0] < ge:
            s, e, nme = api[i]
            ov = min(e, ge) - max(s, gs)
            if ov > 0 and not nme.startswith(("cuEventSynchronize", "cuStreamSynchronize", "cuMemcpyDtoH")):
                att[nme] += ov; attn[nme] += 1
            i += 1
    print("  host API overlapping idle gaps (ms, n): " + ", ".join(f"{k} {v / 1e6:.0f}/{attn[k]}" for k, v in att.most_common(6)))

if __name__ == "__main__":
    main()
