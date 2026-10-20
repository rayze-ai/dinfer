#!/usr/bin/env python3
"""Prometheus 文本快照差值（bench 前后各抓一次 /metrics）→ 服务端自报口径，和客户端测得的放同一张表对账。
  python3 -I scripts/dev/metrics_delta.py <before> <after> [--json]
输出（一行 JSON 或人读文本）：
  dinfer：engine_step_execute_{prefill,decode,mixed}_seconds 的 count 增量与均值 ms（注意：async 调度下它量的是含 finish 等待的 host 侧步时，
          不等于 GPU 步时间）；time_to_first_token_seconds / time_per_output_token_seconds 直方图增量的 mean 与 p50/p99（桶内线性插值）；
          prefix_cache_{queries,hits}_total 增量 → 服务端命中率。
  vLLM：  vllm:time_to_first_token_seconds / vllm:time_per_output_token_seconds（旧版）或 vllm:inter_token_latency_seconds 直方图增量同上；
          vllm:prefix_cache_{queries,hits} 增量。
直方图分位数 = 增量桶计数的累积分布上线性插值（Prometheus histogram_quantile 同法）；精度受桶边界限制，只用于对账量级，不替代客户端分位数。
"""
import json
import re
import sys


def load(path):
    m = {}
    for ln in open(path):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        mm = re.match(r'^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+(\S+)', ln)
        if not mm:
            continue
        name, labels, val = mm.group(1), mm.group(2) or "", mm.group(3)
        try:
            v = float(val)
        except ValueError:
            continue
        m[(name, labels)] = v
    return m


def delta(a, b):
    return {k: b[k] - a.get(k, 0.0) for k in b}


def strip_prefix(name):
    return name.split(":", 1)[1] if ":" in name else name


def hist(d, base):
    """base 形如 'time_to_first_token_seconds'（不带前缀）；返回 {count, mean_ms, p50_ms, p99_ms} 或 None。"""
    buckets, count, total = [], None, None
    for (name, labels), v in d.items():
        n = strip_prefix(name)
        if n == base + "_bucket":
            le = re.search(r'le="([^"]+)"', labels)
            if le:
                buckets.append((float("inf") if le.group(1) == "+Inf" else float(le.group(1)), v))
        elif n == base + "_count":
            count = (count or 0) + v
        elif n == base + "_sum":
            total = (total or 0) + v
    if not count:
        return None
    # 同 le 多 label 组合求和
    acc = {}
    for le, v in buckets:
        acc[le] = acc.get(le, 0.0) + v
    bs = sorted(acc.items())

    def q(p):
        target = p * count
        prev_le, prev_c = 0.0, 0.0
        for le, c in bs:
            if c >= target:
                if le == float("inf"):
                    return prev_le * 1e3
                frac = (target - prev_c) / (c - prev_c) if c > prev_c else 1.0
                return (prev_le + (le - prev_le) * frac) * 1e3
            prev_le, prev_c = le, c
        return bs[-1][0] * 1e3 if bs else float("nan")

    return {"count": int(count), "mean_ms": (total / count) * 1e3 if total is not None else None, "p50_ms": q(0.5), "p99_ms": q(0.99)}


def counter(d, base):
    tot = [v for (name, labels), v in d.items() if strip_prefix(name) == base]
    return sum(tot) if tot else None


def summarize(before, after):
    d = delta(load(before), load(after))
    out = {}
    steps = {}
    for mode in ("prefill", "decode", "mixed"):
        c = counter(d, f"engine_step_execute_{mode}_seconds_count")
        s = counter(d, f"engine_step_execute_{mode}_seconds_sum")
        if c is not None:
            steps[mode] = {"count": int(c), "mean_ms": (s / c * 1e3) if c and s is not None else None}
    if steps:
        out["steps"] = steps
    for key, bases in (("ttft", ["time_to_first_token_seconds"]),
                       ("tpot", ["time_per_output_token_seconds", "inter_token_latency_seconds"]),
                       ("e2e", ["e2e_request_latency_seconds"])):
        for b in bases:
            h = hist(d, b)
            if h:
                h["metric"] = b
                out[key] = h
                break
    q = counter(d, "prefix_cache_queries_total")
    if q is None:
        q = counter(d, "prefix_cache_queries")
    h = counter(d, "prefix_cache_hits_total")
    if h is None:
        h = counter(d, "prefix_cache_hits")
    if q is not None and h is not None:
        out["prefix_cache"] = {"queries": int(q), "hits": int(h), "hit_ratio": (h / q) if q else None}
    return out


def fmt_steps(steps):
    if not steps:
        return "-"
    parts = []
    for mode in ("prefill", "decode", "mixed"):
        s = steps.get(mode)
        if not s:
            parts.append("?")
        elif s["count"] == 0:
            parts.append("0")
        else:
            parts.append(f"{s['count']} ({s['mean_ms']:.1f})")
    return " / ".join(parts)


def fmt_hist(h):
    if not h:
        return "-"
    mean = "-" if h.get("mean_ms") is None else f"{h['mean_ms']:.1f}"
    return f"{mean}/{h['p50_ms']:.1f}/{h['p99_ms']:.1f} (n={h['count']})"


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        print(__doc__)
        sys.exit(2)
    r = summarize(args[0], args[1])
    if "--json" in sys.argv:
        print(json.dumps(r, ensure_ascii=False))
    else:
        print("steps P/D/M count (mean ms):", fmt_steps(r.get("steps")))
        print("server TTFT mean/p50/p99 ms:", fmt_hist(r.get("ttft")))
        print("server TPOT mean/p50/p99 ms:", fmt_hist(r.get("tpot")))
        pc = r.get("prefix_cache")
        print("server prefix cache:", "-" if not pc else f"{pc['hits']}/{pc['queries']} = {pc['hit_ratio']:.3f}" if pc["queries"] else "0 queries")
