#!/usr/bin/env python3
"""真权重 MXFP4 解包核对：从 safetensors 里取一张 `*.weight_packed` + `*.weight_scale`，
按 docs/loops/k3-moe.md 的解包口径用 numpy 解到 f32 作参考，并与 Rust 侧（mxfp4.cu）逐位比。

纯 numpy + safetensors header 解析（不依赖 torch / safetensors 包），可用 `python3 -I` 跑。
输出目录里每个 case 写三个裸二进制 + 一个 manifest，供 `crates/kernels/tests/mxfp4_realcheck.rs` 读：

    <name>.packed   u8  [rows, cols/2]        原样
    <name>.scale    u8  [rows, cols/group]    原样
    <name>.ref      f32 [rows, cols]          参考解包值（本脚本算）
    manifest.txt   每行 `name rows cols group`

口径（与 kernel 一致，逐位）：
  低半字节 = 偶数下标；E2M1 幅值表 [0,.5,1,1.5,2,3,4,6]，bit3 是符号；
  scale 按 OCP e8m0 的位构造 `__uint_as_float(s << 23)`（注意 s=0 得 0.0、s>=255 得 inf/NaN，
  与严格数学意义的 2^(s-127) 在两端不同——脚本会单独统计这两个极值是否出现）。

用法：
  python3 -I scripts/verify/k3_mxfp4_realcheck.py --model <模型目录> --out <输出目录>
        [--pattern 'language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}.w{w}.weight_{kind}']
        [--layer 1] [--expert 0] [--w 1,2,3]
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def read_header(path):
    """返回 (header dict, 数据区起始偏移)。"""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def find_shard(model_dir, tensor_name):
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            wm = json.load(f)["weight_map"]
        if tensor_name in wm:
            return os.path.join(model_dir, wm[tensor_name])
        raise KeyError(f"{tensor_name} 不在 index 的 weight_map 里")
    raise FileNotFoundError(f"缺 {idx}（本脚本只支持带 index 的分片模型）")


def read_tensor(model_dir, tensor_name):
    """返回 (shape, np.ndarray(u8, 行主序))。"""
    path = find_shard(model_dir, tensor_name)
    header, base = read_header(path)
    if tensor_name not in header:
        raise KeyError(f"{tensor_name} 不在 {os.path.basename(path)} 的头里")
    info = header[tensor_name]
    if info["dtype"] != "U8":
        raise ValueError(f"{tensor_name}: dtype {info['dtype']} 不是 U8")
    lo, hi = info["data_offsets"]
    shape = tuple(info["shape"])
    n = hi - lo
    with open(path, "rb") as f:
        f.seek(base + lo)
        buf = f.read(n)
    if len(buf) != n:
        raise IOError(f"{tensor_name}: 读到 {len(buf)} 字节，期望 {n}")
    return shape, np.frombuffer(buf, dtype=np.uint8).reshape(shape)


def dequant(packed, scale_u8, group):
    """[rows, cols] f32；与 mxfp4.cu 的位口径逐位一致。"""
    rows, half = packed.shape
    cols = half * 2
    nib = np.empty((rows, cols), dtype=np.uint8)
    nib[:, 0::2] = packed & 0x0F
    nib[:, 1::2] = (packed >> 4) & 0x0F
    val = E2M1[nib & 7]
    np.negative(val, out=val, where=(nib & 8) != 0)
    # e8m0：与 __uint_as_float(s << 23) 完全相同的位构造
    s = scale_u8.astype(np.uint32) << np.uint32(23)
    sval = s.view(np.float32)
    return val * np.repeat(sval, group, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pattern", default="language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}.w{w}.weight_{kind}")
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--expert", type=int, default=0)
    ap.add_argument("--w", default="1,2,3")
    ap.add_argument("--group", type=int, default=32)
    # 只取前 N 行（默认全量）。取一小片可以随仓库提交，让别的机器/无权重环境也能复核。
    ap.add_argument("--rows-limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    manifest = []
    summary = []
    for w in [int(x) for x in args.w.split(",") if x.strip()]:
        pname = args.pattern.format(layer=args.layer, expert=args.expert, w=w, kind="packed")
        sname = args.pattern.format(layer=args.layer, expert=args.expert, w=w, kind="scale")
        name = f"w{w}"
        pshape, packed = read_tensor(args.model, pname)
        sshape, scale = read_tensor(args.model, sname)
        rows, half = pshape
        cols = half * 2
        if args.rows_limit and args.rows_limit < rows:
            rows = args.rows_limit
            packed = packed[:rows]
            scale = scale[:rows]
        if sshape[0] != rows or sshape[1] * args.group != cols:
            raise ValueError(f"w{w}: scale {sshape} 与 packed {pshape} group {args.group} 不自洽")
        ref = dequant(packed, scale, args.group)

        packed.tofile(os.path.join(args.out, f"{name}.packed"))
        scale.tofile(os.path.join(args.out, f"{name}.scale"))
        ref.astype(np.float32).tofile(os.path.join(args.out, f"{name}.ref"))
        manifest.append(f"{name} {rows} {cols} {args.group}")

        # ---- 统计：解包后的量级 + scale 的极值（sm100 原生 FP4 路径要用的事实）----
        sflat = scale.reshape(-1)
        n0 = int((sflat == 0).sum())
        n255 = int((sflat == 255).sum())
        smax = int(sflat.max())
        smin = int(sflat.min())
        finite = ref[np.isfinite(ref)]
        summary.append(
            f"w{w}: shape [{rows},{cols}] scale u8 min {smin} max {smax} "
            f"u8=0 出现 {n0} / {sflat.size}   u8=255 出现 {n255}  "
            f"index 中位 {int(np.median(sflat))}  "
            f"|w|max {float(np.abs(finite).max()):.4g}  "
            f"非有限值 {int((~np.isfinite(ref)).sum())}  "
            f"零值占比 {float((ref == 0).mean()):.4f}"
        )

    with open(os.path.join(args.out, "manifest.txt"), "w") as f:
        f.write("\n".join(manifest) + "\n")
    print(f"==> 写出 {args.out}（{len(manifest)} 个 case）")
    for line in summary:
        print("  " + line)
    # 单独一行给 shell 汇总
    print("==> " + " | ".join(summary))


if __name__ == "__main__":
    main()
