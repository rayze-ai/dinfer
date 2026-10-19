#!/usr/bin/env python3
"""Qwen3 逐算子对拍：读 `crates/kernels/tests/ops_parity.rs` dump 的 .npy（target/ops_parity/<op>/<case>/），
用 torch（CPU, f32）算参考，按容差比，打印汇总表；全部通过 exit 0。

口径（docs/requirements.md 第 30 行 + docs/pitfalls.md "硬判决不可逐位复现"）：
  bf16 算子   atol 1e-2 / rtol 1e-2（gemm 3e-2/2e-2、qkv_post 2e-2/2e-2 与 ref_tests.rs 同）
  f32 算子    atol 1e-5（logprobs 1e-4：log_softmax 求和次序）
  attention   每行 cos 相似度 > 0.999 且 max abs < 2e-2
  硬判决      argmax / top-k / min_p 掩码 / 采样支持集：只在 "可判定"（分差 > 阈值）的位置断言
用法：python3 -I scripts/verify/ops_parity_ref.py [--dir target/ops_parity] [--ref-log /tmp/dinfer_kernels.log]
`--ref-log`：把 kernels.sh（ref_tests.rs）的 cargo test 输出也并进同一张表。
"""
import glob, json, math, os, re, sys

import numpy as np
import torch

torch.set_num_threads(max(1, os.cpu_count() or 1))
DIR = sys.argv[sys.argv.index("--dir") + 1] if "--dir" in sys.argv else "target/ops_parity"
REF_LOG = sys.argv[sys.argv.index("--ref-log") + 1] if "--ref-log" in sys.argv else None

ROWS = []  # (op, case, max_err, note, ok)


def load(d):
    """目录 → {name: tensor}；*_bf16.npy（u16 位）还原成 f32 tensor，去掉后缀。"""
    t = {}
    for f in glob.glob(os.path.join(d, "*.npy")):
        n = os.path.basename(f)[:-4]
        a = np.load(f)
        if n.endswith("_bf16"):
            a = (a.astype(np.uint32) << 16).view(np.float32)
            n = n[:-5]
        t[n] = torch.from_numpy(np.ascontiguousarray(a))
    m = os.path.join(d, "meta.json")
    meta = json.load(open(m)) if os.path.exists(m) else {}
    return t, meta


def bf(x):
    return x.to(torch.bfloat16).float()


def close(got, want, atol, rtol):
    """返回 (max_abs_err, n_bad)。"""
    got, want = got.float().reshape(-1), want.float().reshape(-1)
    assert got.numel() == want.numel(), f"len {got.numel()} vs {want.numel()}"
    d = (got - want).abs()
    bad = (d > atol + rtol * want.abs()).sum().item()
    return d.max().item() if d.numel() else 0.0, int(bad)


def rec(op, case, err, ok, note=""):
    ROWS.append((op, case, err, note, bool(ok)))


# ---------------------------------------------------------------- 参考实现

def rmsnorm(x, w, eps):
    xf = x.float()
    return xf * torch.rsqrt((xf * xf).mean(-1, keepdim=True) + eps) * w.float()


def chk_rmsnorm(t, m, case):
    want = bf(rmsnorm(t["x"], t["w"], m["eps"]))
    e, bad = close(t["out"], want, 1e-2, 1e-2)
    rec("rmsnorm", case, e, bad == 0, f"{bad} bad")


def chk_fused_add_rmsnorm(t, m, case):
    res = bf(t["x"] + t["res_in"])
    e1, b1 = close(t["res_out"], res, 1e-2, 1e-2)
    want = bf(rmsnorm(res, t["w"], m["eps"]))
    e2, b2 = close(t["out"], want, 1e-2, 1e-2)
    rec("fused_add_rmsnorm", case, max(e1, e2), b1 + b2 == 0, f"res {b1} bad / out {b2} bad")


def chk_gather_rmsnorm(t, m, case):
    want = bf(rmsnorm(t["x"][t["rows"].long()], t["w"], m["eps"]))
    e, bad = close(t["out"], want, 1e-2, 1e-2)
    rec("gather_rmsnorm", case, e, bad == 0, f"{bad} bad")


def chk_silu_mul(t, m, case):
    i = t["x"].shape[1] // 2
    want = bf(torch.nn.functional.silu(t["x"][:, :i]) * t["x"][:, i:])
    e, bad = close(t["out"], want, 1e-2, 1e-2)
    rec("silu_mul", case, e, bad == 0, f"{bad} bad")


def chk_embedding(t, m, case):
    want = t["table"][t["ids"].long()]
    e, bad = close(t["out"], want, 0.0, 0.0)
    rec("embedding", case, e, bad == 0, "逐位")


def chk_add_inplace(t, m, case):
    want = bf(t["x"] + t["y_in"])
    e, bad = close(t["y_out"], want, 1e-2, 1e-2)
    rec("add_inplace", case, e, bad == 0, f"{bad} bad")


def chk_bf16_to_f32(t, m, case):
    e, bad = close(t["y"], t["x"], 0.0, 0.0)
    rec("bf16_to_f32", case, e, bad == 0, "逐位")


def chk_gemm(t, m, case):
    want = bf(t["a"] @ t["w"].T)
    e, bad = close(t["out"], want, 3e-2, 2e-2)
    rec("gemm", case, e, bad == 0, f"{bad} bad (atol 3e-2 rtol 2e-2)")


def rope_neox(x, cs_row):
    """x [..., D] f32；cs_row [D] = cos(D/2) | sin(D/2)。"""
    half = x.shape[-1] // 2
    c, s = cs_row[..., :half], cs_row[..., half:]
    a, b = x[..., :half], x[..., half:]
    return torch.cat([a * c - b * s, b * c + a * s], -1)


def chk_qkv_post(t, m, case):
    h, hk, d, eps, bs = int(m["heads"]), int(m["kv_heads"]), int(m["head_dim"]), m["eps"], int(m["block_size"])
    x = t["qkv_in"].float()
    T = x.shape[0]
    cs = t["cos_sin"][t["pos"].long()]  # [T, D]
    q = x[:, : h * d].reshape(T, h, d)
    k = x[:, h * d : (h + hk) * d].reshape(T, hk, d)
    v = x[:, (h + hk) * d :].reshape(T, hk, d)
    qn = rope_neox(rmsnorm(q, t["wq"], eps), cs[:, None, :])
    kn = rope_neox(rmsnorm(k, t["wk"], eps), cs[:, None, :])
    want = bf(torch.cat([qn.reshape(T, -1), kn.reshape(T, -1), v.reshape(T, -1)], 1))
    e1, b1 = close(t["qkv_out"], want, 2e-2, 2e-2)
    # cache [slots, hk, 2, d]
    cache = torch.zeros_like(t["cache"])
    kb = bf(kn)
    for i in range(T):
        sl = int(t["slots"][i])
        if sl >= 0:
            cache[sl, :, 0] = kb[i]
            cache[sl, :, 1] = v[i]
    e2, b2 = close(t["cache"], cache, 2e-2, 2e-2)
    rec("qkv_post", case, max(e1, e2), b1 + b2 == 0, f"qkv {b1} bad / cache {b2} bad (atol 2e-2)")


def chk_qkv_post_ext(t, m, case):
    """Qwen3.5：partial RoPE（只旋前 rope_dim，pair (i, i+rope_dim/2)）、zero-centered norm（w + beta）、
    输出门行布局 [q | gate | k | v]（gate 段原样不动）。"""
    h, hk, d = int(m["heads"]), int(m["kv_heads"]), int(m["head_dim"])
    rd, eps, beta = int(m["rope_dim"]), m["eps"], m["norm_beta"]
    qmul = int(m.get("qmul", 1))
    x = t["qkv_in"].float()
    T = x.shape[0]
    cs = t["cos_sin"][t["pos"].long()]  # [T, rd]
    half = rd // 2
    c, s = cs[:, :half], cs[:, half:]

    def head_norm(v, w):
        n = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
        return n * (w.float() + beta)

    def rope_partial(v):
        # v [T, heads, d]：前 rd 维旋转，其余维不变
        r, tail = v[..., :rd], v[..., rd:]
        a, b = r[..., :half], r[..., half:]
        rot = torch.cat([a * c[:, None, :] - b * s[:, None, :], b * c[:, None, :] + a * s[:, None, :]], -1)
        return torch.cat([rot, tail], -1) if tail.shape[-1] else rot

    q_off = h * d
    qraw = x[:, : q_off].reshape(T, h, d)
    kraw = x[:, (h * qmul) * d : (h * qmul + hk) * d].reshape(T, hk, d)
    vraw = x[:, (h * qmul + hk) * d :].reshape(T, hk, d)
    qn = bf(rope_partial(head_norm(qraw, t["wq"])))
    kn = bf(rope_partial(head_norm(kraw, t["wk"])))
    parts = [qn.reshape(T, -1)]
    if qmul == 2:
        gate = x[:, q_off : h * qmul * d].reshape(T, h * d)
        parts.append(gate)
    parts += [kn.reshape(T, -1), vraw.reshape(T, -1)]
    want = bf(torch.cat(parts, 1))
    e1, b1 = close(t["qkv_out"], want, 2e-2, 2e-2)
    cache = torch.zeros_like(t["cache"])
    kb = bf(kn)
    for i in range(T):
        sl = int(t["slots"][i])
        if sl >= 0:
            cache[sl, :, 0] = kb[i]
            cache[sl, :, 1] = vraw[i]
    e2, b2 = close(t["cache"], cache, 2e-2, 2e-2)
    rec("qkv_post_ext", case, max(e1, e2), b1 + b2 == 0, f"qkv {b1} bad / cache {b2} bad (atol 2e-2)")


def chk_mul_sigmoid(t, m, case):
    want = bf(t["x_in"] * torch.sigmoid(t["gate"].float()))
    e, bad = close(t["out"], want, 2e-2, 2e-2)
    rec("mul_sigmoid", case, e, bad == 0, f"{bad} bad (atol 2e-2)")


def attn_ref(q, k, v, hist, scale):
    """q [tq, H, D]，k/v [tk, HK, D]（tk = hist + tq），causal：第 i 行看 k[: hist+i+1]。返回 [tq, H, D] f32。"""
    tq, H, D = q.shape
    HK = k.shape[1]
    g = H // HK
    out = torch.empty(tq, H, D)
    for hh in range(H):
        kh = k[:, hh // g].float()
        vh = v[:, hh // g].float()
        s = (q[:, hh].float() @ kh.T) * scale  # [tq, tk]
        pos_k = torch.arange(k.shape[0])[None, :]
        pos_q = (hist + torch.arange(tq))[:, None]
        s = s.masked_fill(pos_k > pos_q, float("-inf"))
        out[:, hh] = s.softmax(-1) @ vh
    return out


def attn_compare(op, case, got, want):
    got, want = got.float().reshape(-1, want.shape[-1]), want.reshape(-1, want.shape[-1])
    cos = torch.nn.functional.cosine_similarity(got, want, dim=-1)
    mx = (got - want).abs().max().item()
    ok = bool((cos > 0.999).all()) and mx < 2e-2
    rec(op, case, mx, ok, f"min cos {cos.min().item():.5f}")


def chk_gqa_prefill(t, m, case):
    h, hk, d, scale = int(m["heads"]), int(m["kv_heads"]), int(m["head_dim"]), m["scale"]
    x = t["qkv"]
    cu = t["cu_seqlens"].tolist()
    outs = []
    for i in range(len(cu) - 1):
        a, b = cu[i], cu[i + 1]
        q = x[a:b, : h * d].reshape(b - a, h, d)
        k = x[a:b, h * d : (h + hk) * d].reshape(b - a, hk, d)
        v = x[a:b, (h + hk) * d :].reshape(b - a, hk, d)
        outs.append(attn_ref(q, k, v, 0, scale))
    attn_compare("gqa_prefill", case, t["out"], torch.cat(outs))


def chk_paged_like(op, t, m, case):
    h, hk, d, bs, scale = int(m["heads"]), int(m["kv_heads"]), int(m["head_dim"]), int(m["block_size"]), m["scale"]
    cache, bt, seq, cu = t["cache"], t["block_table"].long(), t["seq_lens"].tolist(), t["cu_seqlens_q"].tolist()
    outs = []
    for i in range(len(seq)):
        L = seq[i]
        nq = cu[i + 1] - cu[i]
        slots = torch.cat([bt[i, p // bs : p // bs + 1] * bs + (p % bs) for p in range(L)]) if L else torch.zeros(0, dtype=torch.long)
        k = cache[slots, :, 0]  # [L, hk, d]
        v = cache[slots, :, 1]
        q = t["q"][cu[i] : cu[i + 1]]
        outs.append(attn_ref(q, k, v, L - nq, scale))
    attn_compare(op, case, t["out"], torch.cat(outs))


def chk_kv_pack(t, m, case):
    bs, hk, d = int(m["block_size"]), int(m["kv_heads"]), int(m["head_dim"])
    cache = t["cache_in"]  # [L, NB, bs, hk, 2, d] f32（由 bf16 位还原）
    L, NB = cache.shape[:2]
    blocks = t["blocks"].long()
    st = t["staging"]  # [n, L, per_layer_bytes] u8
    # 参考：每块每层 [bs, hk, 2, d] → head-major [hk, bs, 2, d]
    want_bits = cache[:, blocks].permute(1, 0, 3, 2, 4, 5).contiguous()  # [n, L, hk, bs, 2, d]
    want_u16 = (want_bits.numpy().view(np.uint32) >> 16).astype(np.uint16)
    got_u16 = st.numpy().reshape(len(blocks), L, -1).view(np.uint16)
    bad1 = int((want_u16.reshape(len(blocks), L, -1) != got_u16).sum())
    # unpack 回原位：被选块逐位相等，其余块全零
    # 按位比（随机 bf16 位里有 NaN，f32 比较 NaN != NaN 会误判）
    up = (t["cache_unpacked"].numpy().view(np.uint32) >> 16).astype(np.uint16)
    orig = (cache.numpy().view(np.uint32) >> 16).astype(np.uint16)
    sel_bad = int((up[:, blocks.numpy()] != orig[:, blocks.numpy()]).sum())
    mask = np.ones(NB, dtype=bool)
    mask[blocks.numpy()] = False
    rest_bad = int((up[:, mask] != 0).sum())
    rec("kv_pack", case, float(bad1), bad1 == 0, f"pack {bad1} 位错")
    rec("kv_unpack", case, float(sel_bad + rest_bad), sel_bad + rest_bad == 0, f"选中块 {sel_bad} / 其余非零 {rest_bad}")


def penalties_ref(logits, ids, offs, plen, rep, freq, pres):
    out = logits.clone()
    for b in range(logits.shape[0]):
        row = ids[offs[b] : offs[b + 1]].tolist()
        p_, o_ = row[: plen[b]], row[plen[b] :]
        for tok in set(row):
            x = out[b, tok].item()
            if rep[b] != 1.0:
                x = x / rep[b] if x > 0 else x * rep[b]
            cnt = o_.count(tok)
            if cnt > 0:
                x -= freq[b] * cnt + pres[b]
            out[b, tok] = x
    return out


def topk_topp_ref(row, kk, pp):
    r = row.clone()
    if 0 < kk < r.numel():
        thresh = torch.topk(r, kk).values[-1]
        r[r < thresh] = float("-inf")
    if 0 < pp < 1:
        srt, idx = torch.sort(r, descending=False)
        cum = srt.softmax(-1).cumsum(-1)
        drop = cum <= 1 - pp
        drop[-1] = False
        r[idx[drop]] = float("-inf")
    return r


def mask_compare(op, case, got, want, decidable):
    """掩码类：有限/−inf 模式只在 decidable 位置比较；有限值逐元素比。"""
    gi, wi = torch.isinf(got), torch.isinf(want)
    bad_pat = int(((gi != wi) & decidable).sum())
    both = ~gi & ~wi
    e = (got[both] - want[both]).abs().max().item() if both.any() else 0.0
    bad_val = int(((got[both] - want[both]).abs() > 1e-5).sum())
    und = int((~decidable).sum())
    rec(op, case, e, bad_pat + bad_val == 0, f"模式错 {bad_pat} / 值错 {bad_val} / 不可判定跳过 {und}")


def chk_sampler(t, m, case):
    logits = t["logits"]
    B, V = logits.shape
    ids, offs, plen = t["hist_ids"], t["hist_offsets"].tolist(), t["hist_prompt_len"].tolist()
    rep, freq, pres = t["rep"].tolist(), t["freq"].tolist(), t["pres"].tolist()
    temp, minp, topk, topp = t["temperature"], t["min_p"], t["top_k"].tolist(), t["top_p"].tolist()
    # 1. penalties（f32 逐元素）
    pen = penalties_ref(logits, ids, offs, plen, rep, freq, pres)
    e, bad = close(t["after_penalties"], pen, 1e-5, 1e-6)
    rec("penalties", case, e, bad == 0, f"{bad} bad")
    # 2. temperature：t=0 行不动
    tt = torch.where(temp > 1e-5, temp, torch.ones_like(temp))
    tmp = t["after_penalties"] / tt[:, None]
    e, bad = close(t["after_temperature"], tmp, 1e-5, 1e-6)
    rec("temperature", case, e, bad == 0, f"{bad} bad")
    # 3. min_p：p_i < min_p · max p → −inf（min_p=0 不动）。可判定 = |p_i − min_p·pmax| 相对 > 1e-5
    x = t["after_temperature"]
    probs = x.softmax(-1)
    thr = minp[:, None] * probs.max(-1, keepdim=True).values
    want = x.masked_fill((probs < thr) & (minp[:, None] > 0), float("-inf"))
    decid = ((probs - thr).abs() > 1e-5 * thr.clamp(min=1e-30)) | (minp[:, None] <= 0)
    mask_compare("min_p_mask", case, t["after_min_p"], want, decid)
    # 4. top-k / top-p（vLLM 口径）。可判定 = 与阈值的 logit 分差 > 1e-4
    x = t["after_min_p"]
    want = torch.stack([topk_topp_ref(x[b], int(topk[b]), float(topp[b])) for b in range(B)])
    decid = torch.ones_like(x, dtype=torch.bool)
    for b in range(B):
        fin = want[b][~torch.isinf(want[b])]
        if fin.numel() < V and fin.numel() > 0:
            thr_b = fin.min()
            decid[b] = (x[b] - thr_b).abs() > 1e-4
    mask_compare("topk_topp_mask", case, t["after_topk_topp"], want, decid)
    # 5. sample：greedy 行 = argmax(掩码后)；其余行必须落在支持集内（分布一致性 ref_tests 另测）
    masked = t["after_topk_topp"]
    sc = t["sample_chain"].long()
    bad = 0
    for b in range(B):
        if temp[b] <= 1e-5:
            top = masked[b].max()
            bad += int(masked[b, sc[b]] != top)  # 允许并列
        else:
            bad += int(torch.isinf(masked[b, sc[b]]))
    rec("sample", case, float(bad), bad == 0, "greedy=argmax / 支持集内")
    # 6. fused == chain（逐位：同一份 row 原语）
    e, bad = close(t["fused_work"], masked, 0.0, 0.0)
    same = int((t["sample_fused"] != t["sample_chain"]).sum())
    rec("sample_fused", case, e, bad == 0 and same == 0, f"掩码位差 {bad} / 采样差 {same}")
    # 7. argmax（原始 logits）：top1−top2 > 1e-4 才判定
    top2 = logits.topk(2, -1)
    decid = (top2.values[:, 0] - top2.values[:, 1]) > 1e-4
    am = t["argmax"].long()
    bad = int(((am != top2.indices[:, 0]) & decid).sum())
    rec("argmax", case, float(bad), bad == 0, f"不可判定行 {int((~decid).sum())}")
    # 8. logprobs：log_softmax；top-k 值按容差，id 只在相邻分差 > 1e-4 时比；rank = 严格大于 chosen 的个数
    k = int(m["logprobs_k"])
    lsm = logits.log_softmax(-1)
    chosen_lp = lsm[torch.arange(B), sc]
    e1, b1 = close(t["lp_val"][:, 0], chosen_lp, 1e-4, 1e-6)
    tk = lsm.topk(k, -1)
    e2, b2 = close(t["lp_val"][:, 1:], tk.values, 1e-4, 1e-6)
    gaps = tk.values[:, :-1] - tk.values[:, 1:]
    decid = torch.ones(B, k, dtype=torch.bool)
    decid[:, 1:] &= gaps > 1e-4
    decid[:, :-1] &= gaps > 1e-4
    b3 = int(((t["lp_ids"][:, 1:].long() != tk.indices) & decid).sum())
    rank = (lsm > chosen_lp[:, None]).sum(-1)
    b4 = int((t["lp_rank"].long() != rank).sum())
    b5 = int((t["lp_ids"][:, 0].long() != sc).sum())
    rec("logprobs", case, max(e1, e2), b1 + b2 + b3 + b4 + b5 == 0, f"val {b1 + b2} / id {b3} / rank {b4} / chosen {b5}")
    # 9. row_gather
    g = t["rg_idx"].long()
    want = t["rg_base"].clone()
    sel = g >= 0
    want[sel] = t["rg_prev"][g[sel]]
    bad = int((t["rg_out"] != want).sum())
    rec("row_gather", case, float(bad), bad == 0, "逐位")


CHECKS = {
    "rmsnorm": chk_rmsnorm, "fused_add_rmsnorm": chk_fused_add_rmsnorm, "gather_rmsnorm": chk_gather_rmsnorm,
    "silu_mul": chk_silu_mul, "embedding": chk_embedding, "add_inplace": chk_add_inplace, "bf16_to_f32": chk_bf16_to_f32,
    "gemm": chk_gemm, "qkv_post": chk_qkv_post, "qkv_post_ext": chk_qkv_post_ext,
    "mul_sigmoid": chk_mul_sigmoid, "gqa_prefill": chk_gqa_prefill,
    "gqa_paged": lambda t, m, c: chk_paged_like("gqa_paged", t, m, c),
    "gqa_decode": lambda t, m, c: chk_paged_like("gqa_decode", t, m, c),
    "gqa_mixed": lambda t, m, c: chk_paged_like("gqa_mixed", t, m, c),
    "kv_pack": chk_kv_pack, "sampler": chk_sampler,
}

# ---------------------------------------------------------------- 主流程

n_cases = 0
for op_dir in sorted(glob.glob(os.path.join(DIR, "*"))):
    op = os.path.basename(op_dir)
    if op not in CHECKS:
        rec(op, "*", float("nan"), False, "没有参考实现")
        continue
    for case_dir in sorted(glob.glob(os.path.join(op_dir, "*"))):
        case = os.path.basename(case_dir)
        n_cases += 1
        try:
            t, m = load(case_dir)
            CHECKS[op](t, m, case)
        except Exception as ex:  # noqa: BLE001
            rec(op, case, float("nan"), False, f"异常 {type(ex).__name__}: {ex}"[:80])

if n_cases == 0:
    sys.exit(f"{DIR}: 没有 dump（先跑 cargo test --test ops_parity -- --ignored）")

# ref_tests.rs 的结果并进表（已有覆盖不重写，但一起跑）
if REF_LOG and os.path.exists(REF_LOG):
    for ln in open(REF_LOG, errors="replace"):
        mm = re.match(r"^test (\w+) \.\.\. (ok|FAILED|ignored)", ln.strip())
        if mm and mm.group(2) != "ignored":
            rec("ref_tests::" + mm.group(1), "gen_kernel_refs", float("nan"), mm.group(2) == "ok", "torch 期望值 / Rust 断言")

# 汇总表
w0 = max(len(r[0]) for r in ROWS) + 2
w1 = min(max(len(r[1]) for r in ROWS), 44) + 2
print(f"{'算子':<{w0}}{'形状 / case':<{w1}}{'最大误差':>12}  {'通过':<6}说明")
print("-" * (w0 + w1 + 12 + 2 + 6 + 40))
for op, case, err, note, ok in ROWS:
    es = "—" if (isinstance(err, float) and math.isnan(err)) else (f"{err:.3e}" if err >= 1e-3 or err == 0 else f"{err:.2e}")
    print(f"{op:<{w0}}{case[:44]:<{w1}}{es:>12}  {'PASS' if ok else 'FAIL':<6}{note}")
n_ok = sum(1 for r in ROWS if r[4])
ops = sorted({r[0] for r in ROWS if not r[0].startswith("ref_tests::")})
print("-" * (w0 + w1 + 12 + 2 + 6 + 40))
print(f"{n_ok}/{len(ROWS)} 行通过；ops_parity 覆盖 {len(ops)} 个算子，{n_cases} 个 dump case")
sys.exit(0 if n_ok == len(ROWS) else 1)
