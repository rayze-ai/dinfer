#!/usr/bin/env python3
"""vLLM fla `chunk_gated_delta_rule`（Triton）per-kernel 分解（torch.profiler，按 kernel 名聚合）。

在 vLLM 容器里跑（B300：`nerdctl run --rm --gpus device=N ... vllm/vllm-openai:nightly python3 fla_prof.py`），
形状 HV=48 K=V=128（Qwen3.5-27B GDN），T ∈ {1×1024, 8×1024（cu_seqlens 8 段）, 1×8192}，含 / 不含
`use_qk_l2norm_in_kernel`。输出两份：人读表（stdout）+ 长表 CSV（`--csv 路径`，列同 test/<model>/<日期>/probe_*/probe.csv，
metric = `fla_<kernel>_us`、`fla_total_us`，side=vllm，group=gdn_scale）。

为什么要 l2norm 两档：dinfer 侧 prep（norm + cumsum）是单独一段，fla 把 l2norm 记成独立 kernel；关掉后剩下的才是
与 wy / ho 可逐段对照的部分。
"""
import argparse
import csv
import sys

import torch
from torch.profiler import ProfilerActivity, profile
from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule

SHORT = {
    "chunk_scaled_dot_kkt_fwd_kernel": "kkt",
    "merge_16x16_to_64x64_inverse_kernel": "tril_merge",
    "solve_tril_16x16_kernel": "tril_16",
    "recompute_w_u_fwd_kernel": "wu",
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64": "fwd_h",
    "chunk_fwd_kernel_o": "fwd_o",
    "l2norm_fwd_kernel2": "l2norm",
    "chunk_local_cumsum_scalar_kernel": "cumsum",
}


def short(name: str) -> str:
    for k, v in SHORT.items():
        if k in name:
            return v
    if "FillFunctor" in name:
        return "fill"
    return name.split("<")[0].split("(")[0].split("::")[-1][:40]


def one(nseq: int, L: int, l2: bool, n_iter: int = 20):
    HV, K, V = 48, 128, 128
    T = nseq * L
    dev = "cuda"
    q = torch.randn(1, T, HV, K, device=dev, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, T, HV, V, device=dev, dtype=torch.bfloat16)
    g = torch.randn(1, T, HV, device=dev, dtype=torch.float32) * -0.1
    beta = torch.rand(1, T, HV, device=dev, dtype=torch.bfloat16)
    cu = torch.arange(0, T + 1, L, device=dev, dtype=torch.int32)

    def run():
        return chunk_gated_delta_rule(
            q, k, v, g, beta, scale=None, initial_state=None, output_final_state=True, cu_seqlens=cu, use_qk_l2norm_in_kernel=l2
        )

    for _ in range(5):
        run()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(n_iter):
            run()
        torch.cuda.synchronize()
    agg: dict[str, list] = {}
    for e in p.events():
        if e.device_type.name != "CUDA":
            continue
        a = agg.setdefault(e.name, [0, 0.0])
        a[0] += 1
        a[1] += e.device_time
    tot = sum(a[1] for a in agg.values()) / n_iter
    rows = sorted(((a[1] / n_iter, a[0] // n_iter, name) for name, a in agg.items()), reverse=True)
    return tot, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--probe", default="probe_fla")
    ap.add_argument("--date", default="")
    ap.add_argument("--gpu", default="B300")
    ap.add_argument("--note", default="")
    ap.add_argument("--shapes", default="1x1024,8x1024,1x8192")
    a = ap.parse_args()
    out = []
    print(f"torch {torch.__version__} dev {torch.cuda.get_device_name()}", flush=True)
    for shp in a.shapes.split(","):
        nseq, L = (int(x) for x in shp.split("x"))
        for l2 in (True, False):
            tot, rows = one(nseq, L, l2)
            tag = "l2norm" if l2 else "nol2"
            print(f"=== fla {shp} {tag}: GPU total {tot:.0f} us/call ({len(rows)} kernels)", flush=True)
            for us, cnt, name in rows:
                print(f"{us:7.0f} us/call  x{cnt:<2} {us / tot * 100:4.1f}%  {short(name):12} {name[:70]}", flush=True)
            suffix = "" if l2 else "_nol2"
            out.append((f"fla_total{suffix}_us", shp, round(tot)))
            for us, cnt, name in rows:
                out.append((f"fla_{short(name)}{suffix}_us", shp, round(us)))
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["probe", "date", "gpu", "model", "group", "metric", "side", "x", "value", "unit", "note"])
            for i, (m, x, v) in enumerate(out):
                w.writerow([a.probe, a.date, a.gpu, "qwen35", "gdn_scale", m, "vllm", x, v, "us", a.note if i == 0 else ""])
        print(f"csv -> {a.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
