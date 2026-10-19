#!/usr/bin/env python3
"""真权重 MXFP4 的 e8m0 scale 统计（只读 weight_scale 张量，不解包、不占显存）。

用于 sm100 原生 FP4 路径的设计事实：指数范围、是否出现 u8=0 / u8=255 这类极值
（OCP e8m0：0 → 2^-127 但不进位、255 → NaN/inf 在 `__uint_as_float(s<<23)` 口径下分别为 0.0 与 inf/NaN）。

用法：python3 -I scripts/verify/k3_mxfp4_scale_stats.py --model <dir> [--layers 1,12,45,91] [--experts 0,447,895]
"""
import argparse
import json
import os
import struct

import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "realcheck", os.path.join(os.path.dirname(os.path.abspath(__file__)), "k3_mxfp4_realcheck.py")
)
_rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", default="1,12,45,91")
    ap.add_argument("--experts", default="0,447,895")
    ap.add_argument("--pattern", default="language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}.w{w}.weight_{kind}")
    args = ap.parse_args()

    hist_all = np.zeros(256, dtype=np.int64)
    mn, mx = 255, 0
    n0 = n255 = 0
    total = 0
    rows = []
    for layer in [int(x) for x in args.layers.split(",")]:
        for expert in [int(x) for x in args.experts.split(",")]:
            for w in (1, 2, 3):
                name = args.pattern.format(layer=layer, expert=expert, w=w, kind="scale")
                try:
                    shape, s = _rc.read_tensor(args.model, name)
                except KeyError:
                    rows.append(f"  L{layer} e{expert} w{w}: 缺失")
                    continue
                f = s.reshape(-1)
                c = np.bincount(f, minlength=256)
                hist_all += c
                mn = min(mn, int(f.min()))
                mx = max(mx, int(f.max()))
                n0 += int(c[0])
                n255 += int(c[255])
                total += f.size
                rows.append(
                    f"  L{layer:<3} e{expert:<4} w{w} shape {list(shape)}  min {int(f.min())} max {int(f.max())} "
                    f"中位 {int(np.median(f))}  众数 {int(np.argmax(c))}  u8=0 {int(c[0])} u8=255 {int(c[255])}"
                )
    print("\n".join(rows))
    used = np.nonzero(hist_all)[0]
    print(f"\n==> 合计 {total} 个 scale（{len(rows)} 张量）：u8 范围 [{mn}, {mx}]，"
          f"u8=0 共 {n0}（{100.0 * n0 / total:.3f}%），u8=255 共 {n255}（{100.0 * n255 / total:.3f}%）")
    print(f"==> 用到的指数 {len(used)} 个：{used.min()}..{used.max()}")
    print("==> 直方图（u8 值: 个数，仅非零项）：")
    for v in used:
        print(f"    {int(v):>3}: {int(hist_all[v]):>9}  (2^{int(v) - 127:.0f})")
    print(f"==> 值域 2^({int(mn) - 127}) .. 2^({int(mx) - 127})"
          f" = {2.0 ** (mn - 127):.3e} .. {2.0 ** (mx - 127):.3e}")


if __name__ == "__main__":
    main()
