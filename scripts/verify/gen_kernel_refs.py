#!/usr/bin/env python3
"""生成 P2 算子对拍的输入与期望输出（.npy），供 `cargo test -p dinfer-kernels --features ref-tests` 读取。

只依赖 numpy + torch（CPU 即可）。bf16 用 torch.bfloat16 算再转 float32 存，
Rust 侧读 float32 再 cast 到 bf16，保证两边看到同一份 bf16 输入。

用法：python3 scripts/verify/gen_kernel_refs.py <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import torch

torch.manual_seed(1234)
out = Path(next((a for a in sys.argv[1:] if not a.startswith("--")), "test/kernel_refs"))  # 允许 --gqa 等开关（当前全部用例总是生成）
out.mkdir(parents=True, exist_ok=True)


def save(name: str, t):
    arr = t.detach().cpu()
    if arr.dtype == torch.bfloat16:
        arr = arr.float()
    np.save(out / f"{name}.npy", arr.numpy())


def bf(x):
    return x.to(torch.bfloat16)


# ---- rmsnorm / fused_add_rmsnorm ----
T, H = 13, 1536
x = bf(torch.randn(T, H) * 2)
res = bf(torch.randn(T, H))
w = bf(torch.rand(H) + 0.5)
eps = 1e-5


def rmsnorm(v):
    v32 = v.float()
    return bf(v32 * torch.rsqrt(v32.pow(2).mean(-1, keepdim=True) + eps) * w.float())


save("rmsnorm_x", x)
save("rmsnorm_w", w)
save("rmsnorm_out", rmsnorm(x))
save("fused_res_in", res)
res_new = bf(x.float() + res.float())
save("fused_res_out", res_new)
save("fused_out", rmsnorm(res_new))

# ---- rope（interleave 与 neox）----
T, QH, KH, HD, RD, MAXPOS, THETA = 9, 4, 2, 128, 64, 4096, 8e6
pos = torch.randint(0, MAXPOS, (T,), dtype=torch.int32)
inv_freq = 1.0 / (THETA ** (torch.arange(0, RD, 2).float() / RD))
freqs = torch.outer(torch.arange(MAXPOS).float(), inv_freq)  # [MAXPOS, RD/2]
cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1)  # [MAXPOS, RD]
q = bf(torch.randn(T, QH, HD))
k = bf(torch.randn(T, KH, HD))


def rope(t, neox):
    t32 = t.float().clone()
    c = cos_sin[pos.long(), : RD // 2][:, None, :]  # [T,1,RD/2]
    s = cos_sin[pos.long(), RD // 2 :][:, None, :]
    r = t32[..., :RD]
    if neox:
        a, b = r[..., : RD // 2], r[..., RD // 2 :]
        na, nb = a * c - b * s, b * c + a * s
        t32[..., : RD // 2], t32[..., RD // 2 : RD] = na, nb
    else:
        a, b = r[..., 0::2], r[..., 1::2]
        na, nb = a * c - b * s, b * c + a * s
        t32[..., 0:RD:2], t32[..., 1:RD:2] = na, nb
    return bf(t32)


save("rope_pos", pos)
save("rope_cos_sin", cos_sin)
save("rope_q", q)
save("rope_k", k)
save("rope_q_out_interleave", rope(q, False))
save("rope_k_out_interleave", rope(k, False))
save("rope_q_out_neox", rope(q, True))
save("rope_k_out_neox", rope(k, True))

# ---- silu_and_mul ----
T, I = 11, 2048
x2 = bf(torch.randn(T, 2 * I))
save("silu_x", x2)
save("silu_out", bf(torch.nn.functional.silu(x2[:, :I].float()) * x2[:, I:].float()))

# ---- embedding ----
V, H = 1000, 256
table = bf(torch.randn(V, H))
ids = torch.randint(0, V, (17,), dtype=torch.int32)
save("emb_table", table)
save("emb_ids", ids)
save("emb_out", table[ids.long()])

# ---- gemm ----
M, N, K = 33, 96, 512
a = bf(torch.randn(M, K))
wt = bf(torch.randn(N, K) * 0.05)
bias = bf(torch.randn(N) * 0.1)
save("gemm_a", a)
save("gemm_w", wt)
save("gemm_bias", bias)
save("gemm_out", bf(a.float() @ wt.float().T + bias.float()))

# ---- argmax / topk_topp_mask ----
B, V = 6, 163840
logits = torch.randn(B, V) * 3
save("argmax_logits", logits)
save("argmax_out", logits.argmax(-1).to(torch.int32))

top_k = torch.tensor([0, 1, 50, 0, 5, 163840], dtype=torch.int32)
top_p = torch.tensor([1.0, 1.0, 1.0, 0.9, 0.5, 0.95], dtype=torch.float32)


def mask_ref(row, kk, pp):
    """与 vLLM `apply_top_k_top_p_pytorch` 同口径：升序排序，top-p 在 **top-k 之后** 的分布上算，
    保留 (1 - cumsum) <= p 的尾部（即累计到 p 为止，含跨过 p 的那个）。"""
    r = row.clone()
    if 0 < kk < r.numel():
        thresh = torch.topk(r, kk).values[-1]
        r[r < thresh] = float("-inf")
    if 0 < pp < 1:
        srt, idx = torch.sort(r, descending=False)
        probs = srt.softmax(-1)
        cum = probs.cumsum(-1)
        drop = cum <= 1 - pp
        drop[-1] = False  # 至少保留最大的那个
        r[idx[drop]] = float("-inf")
    return r


masked = torch.stack([mask_ref(logits[b], int(top_k[b]), float(top_p[b])) for b in range(B)])
save("mask_logits", logits)
save("mask_top_k", top_k)
save("mask_top_p", top_p)
save("mask_out_isinf", torch.isinf(masked).to(torch.int32))

print(f"wrote refs to {out}")

# ---- GQA paged attention（Q1 Qwen3）：2 个请求，H=4 / Hkv=2 / D=32，block 8；prefill（无历史）+ paged（带历史）----
# 口径：fp32 SDPA 参考；输入 bf16。cache 布局 [num_blocks, block_size, Hkv, 2, D]（K 在 V 前）。
GH, GHK, GD, GBS = 4, 2, 32, 8
glens = [11, 6]                      # prefill 两个请求的长度
gq = bf(torch.randn(sum(glens), GH, GD))
gk = bf(torch.randn(sum(glens), GHK, GD))
gv = bf(torch.randn(sum(glens), GHK, GD))
gcu = torch.tensor([0, glens[0], glens[0] + glens[1]], dtype=torch.int32)
save("gqa_q", gq); save("gqa_k", gk); save("gqa_v", gv); save("gqa_cu", gcu)


def sdpa_causal(q, k, v, hist=0):
    """q [Tq,H,D]; k,v [Tk,Hkv,D]（Tk = hist + Tq）；query i 看 key [0, hist+i]。"""
    rep = GH // GHK
    kk = k.float().repeat_interleave(rep, dim=1); vv = v.float().repeat_interleave(rep, dim=1)
    s = torch.einsum("qhd,khd->hqk", q.float(), kk) * GD ** -0.5
    Tq, Tk = q.shape[0], k.shape[0]
    mask = torch.arange(Tk)[None, :] > (torch.arange(Tq)[:, None] + hist)
    s = s.masked_fill(mask[None], float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.einsum("hqk,khd->qhd", p, vv)


outs = [sdpa_causal(gq[a:b], gk[a:b], gv[a:b]) for a, b in zip(gcu[:-1].tolist(), gcu[1:].tolist())]
save("gqa_prefill_out", torch.cat(outs))

# paged：请求 0 历史 11（上面的 k/v）+ 新 3 行；请求 1 历史 6 + 新 1 行（decode）。block 表：req0 块 [0,1]（11+3=14 ≤ 16），req1 块 [2]（7 ≤ 8）。
gnew = [3, 1]
gq2 = bf(torch.randn(sum(gnew), GH, GD))
gk2 = bf(torch.randn(sum(gnew), GHK, GD)); gv2 = bf(torch.randn(sum(gnew), GHK, GD))
num_blocks = 3
cache = torch.zeros(num_blocks, GBS, GHK, 2, GD, dtype=torch.bfloat16)
bt = torch.tensor([[0, 1], [2, -1]], dtype=torch.int32)
slots_hist = []
for r, (a, b) in enumerate(zip(gcu[:-1].tolist(), gcu[1:].tolist())):
    for pos in range(b - a):
        blk = bt[r, pos // GBS].item(); slot = blk * GBS + pos % GBS; slots_hist.append(slot)
        cache[blk, pos % GBS, :, 0] = gk[a + pos]; cache[blk, pos % GBS, :, 1] = gv[a + pos]
slots_new = []
cur = 0
for r in range(2):
    for i in range(gnew[r]):
        pos = glens[r] + i; blk = bt[r, pos // GBS].item(); slot = blk * GBS + pos % GBS; slots_new.append(slot)
        cache[blk, pos % GBS, :, 0] = gk2[cur]; cache[blk, pos % GBS, :, 1] = gv2[cur]; cur += 1
save("gqa_hist_slots", torch.tensor(slots_hist, dtype=torch.int64))
save("gqa_new_slots", torch.tensor(slots_new, dtype=torch.int64))
save("gqa_q2", gq2); save("gqa_k2", gk2); save("gqa_v2", gv2)
save("gqa_cache_expect", cache.view(num_blocks, GBS, GHK * 2 * GD))
save("gqa_bt", bt); save("gqa_seq_lens", torch.tensor([glens[0] + gnew[0], glens[1] + gnew[1]], dtype=torch.int32))
save("gqa_cu_q", torch.tensor([0, gnew[0], gnew[0] + gnew[1]], dtype=torch.int32))
outs2 = []
cur = 0
for r, (a, b) in enumerate(zip(gcu[:-1].tolist(), gcu[1:].tolist())):
    kall = torch.cat([gk[a:b], gk2[cur:cur + gnew[r]]]); vall = torch.cat([gv[a:b], gv2[cur:cur + gnew[r]]])
    outs2.append(sdpa_causal(gq2[cur:cur + gnew[r]], kall, vall, hist=b - a)); cur += gnew[r]
save("gqa_paged_out", torch.cat(outs2))
print("gqa refs: prefill", tuple(torch.cat(outs).shape), "paged", tuple(torch.cat(outs2).shape))

# ---- GQA prefill，Qwen3 真实头形状 D=128 / H=16 / Hkv=8（走 flash tile kernel）：3 个请求 70 / 33 / 5（跨多片、尾片不满、
# 块内有不活跃 warp）。fp32 SDPA 参考，输入 bf16。
FH, FHK, FD = 16, 8, 128
flens = [70, 33, 5]
fq = bf(torch.randn(sum(flens), FH, FD)); fk = bf(torch.randn(sum(flens), FHK, FD)); fv = bf(torch.randn(sum(flens), FHK, FD))
fcu = torch.tensor([0] + list(torch.tensor(flens).cumsum(0).tolist()), dtype=torch.int32)
def sdpa_causal_g(q, k, v, H, HK, D):
    kk = k.float().repeat_interleave(H // HK, dim=1); vv = v.float().repeat_interleave(H // HK, dim=1)
    s_ = torch.einsum("qhd,khd->hqk", q.float(), kk) * D ** -0.5
    T = q.shape[0]
    s_ = s_.masked_fill((torch.arange(T)[None, :] > torch.arange(T)[:, None])[None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", torch.softmax(s_, -1), vv)
fouts = [sdpa_causal_g(fq[a:b], fk[a:b], fv[a:b], FH, FHK, FD) for a, b in zip(fcu[:-1].tolist(), fcu[1:].tolist())]
save("gqa128_q", fq); save("gqa128_k", fk); save("gqa128_v", fv); save("gqa128_cu", fcu); save("gqa128_prefill_out", torch.cat(fouts))
print("gqa128 refs: prefill", tuple(torch.cat(fouts).shape))

# ---- Q2 采样：penalties / temperature / min_p / logprobs（vLLM 语义）----
torch.manual_seed(77)
SB, SV = 4, 512
s_logits = torch.randn(SB, SV) * 3
# 历史 token（CSR）：prompt + output，含重复
hist = [([1, 5, 9, 5], [9, 9, 20]), ([3], []), ([7, 7, 7], [7, 100]), ([], [2, 4, 6, 4])]
ids, offs, plen = [], [0], []
for p_, o_ in hist:
    ids += p_ + o_; offs.append(len(ids)); plen.append(len(p_))
rep = torch.tensor([1.5, 1.0, 1.2, 1.0]); freq = torch.tensor([0.0, 0.0, 0.3, 0.5]); pres = torch.tensor([0.0, 0.0, 0.2, 0.1])
def penal_ref(row, p_, o_, r, f, pr):
    row = row.clone()
    seen = set(p_) | set(o_)
    for t in seen:
        v = row[t].item()
        if r != 1.0:
            v = v / r if v > 0 else v * r
        cnt = o_.count(t)
        if cnt > 0:
            v -= f * cnt + pr
        row[t] = v
    return row
pen_out = torch.stack([penal_ref(s_logits[b], hist[b][0], hist[b][1], rep[b].item(), freq[b].item(), pres[b].item()) for b in range(SB)])
save("pen_logits", s_logits); save("pen_ids", torch.tensor(ids, dtype=torch.int32)); save("pen_offsets", torch.tensor(offs, dtype=torch.int32))
save("pen_prompt_len", torch.tensor(plen, dtype=torch.int32)); save("pen_rep", rep); save("pen_freq", freq); save("pen_pres", pres); save("pen_out", pen_out)
# temperature
temps = torch.tensor([0.7, 1.0, 1.3, 0.0])
temp_out = s_logits.clone()
for b in range(SB):
    if temps[b] > 1e-5 and temps[b] != 1.0: temp_out[b] = s_logits[b] / temps[b]
save("temp_in", s_logits); save("temp_t", temps); save("temp_out", temp_out)
# min_p
minp = torch.tensor([0.1, 0.0, 0.3, 0.05])
mp_out = s_logits.clone()
for b in range(SB):
    if minp[b] > 0:
        pr = torch.softmax(s_logits[b], -1); mp_out[b][pr < minp[b] * pr.max()] = float("-inf")
save("minp_in", s_logits); save("minp_p", minp); save("minp_out_isinf", torch.isinf(mp_out).to(torch.int32))
# logprobs: log_softmax of raw logits, chosen token, top-5, rank
chosen = torch.tensor([3, 100, 0, 511], dtype=torch.int32)
lsm = torch.log_softmax(s_logits, -1)
k = 5
top = torch.topk(lsm, k, dim=-1)
lp_ids = torch.cat([chosen[:, None].long(), top.indices], 1).to(torch.int32)
lp_vals = torch.cat([lsm.gather(1, chosen[:, None].long()), top.values], 1)
rank = (lsm > lsm.gather(1, chosen[:, None].long())).sum(-1).to(torch.int32)
save("lp_logits", s_logits); save("lp_chosen", chosen); save("lp_ids", lp_ids); save("lp_vals", lp_vals); save("lp_rank", rank)
print("sampling refs: penalties/temperature/min_p/logprobs written")
