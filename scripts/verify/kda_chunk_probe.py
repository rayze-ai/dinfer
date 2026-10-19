"""KDA chunk 形式的代数自检（纯 torch，规格探针）。

用途：在写 CUDA 之前/之后，验证 `docs/loops/k3-kda.md` 的 v2 规格（inclusive cumsum、
A / Aq / u / w / M / kg 的定义）与逐 token 顺序递归数值等价，并给出 bf16 消融误差。

    /data/ref/venv-tf519/bin/python -I scripts/verify/kda_chunk_probe.py

输出：fp32 与各档 bf16 相对顺序递归参考的 max_err / cos。不做断言，供人读。
参考实现（gen_kda_refs.py）与 kernel 的对照由 kda_parity.rs 断言。
"""
import torch

torch.manual_seed(1)
K = V = 128
BT = 64
T = 256
H = 2
bf = lambda t: t.to(torch.bfloat16).to(torch.float32)

beta = torch.sigmoid(2.0 * torch.randn(T, H))
a = torch.tensor([1.0, 15.9])[:H]
# K3 的真实 gate 形状：-5·sigmoid(a·x)，a = exp(A_log) ∈ [1, 15.9]
g = -5.0 * torch.sigmoid(a.view(1, H, 1) * (1.5 * torch.randn(T, H, K)))
q = bf(torch.randn(T, H, K) * 0.8)
k = bf(torch.randn(T, H, K) * 0.8)
v = bf(torch.randn(T, H, V) * 0.8)
h0 = 0.05 * torch.randn(H, V, K)
qn = q / (q.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt() * (K ** -0.5)
kn = k / (k.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()


def seq():
    """逐 token 顺序递归（与 kda.cu 的 recurrent_kernel 同式）——判定基准。"""
    S = h0.clone()
    out = torch.zeros(T, H, V)
    for t in range(T):
        S = S * g[t].exp()[:, None, :]
        vp = (v[t] - (S * kn[t][:, None, :]).sum(-1)) * beta[t][:, None]
        S = S + vp[:, :, None] * kn[t][:, None, :]
        out[t] = (S * qn[t][:, None, :]).sum(-1)
    return out, S


def chunk(only=None, bc=16):
    """chunk 形式。only=None → 全 fp32；否则 only 是走 bf16 的环节集合（消融用）。"""
    def R(tag):
        return bf if (only is not None and tag in only) else (lambda x: x)
    out = torch.zeros(T, H, V)
    h = h0.clone()                                  # [H,V,K] fp32
    for c0 in range(0, T, BT):
        gc = torch.cumsum(g[c0:c0 + BT], 0)          # inclusive，[BT,H,K]
        be, kk, qq, vv = beta[c0:c0 + BT], kn[c0:c0 + BT], qn[c0:c0 + BT], v[c0:c0 + BT]
        A = torch.zeros(BT, BT, H)
        Aq = torch.zeros(BT, BT, H)
        for ib in range(0, BT, bc):                  # 16 行一块：衰减因子 exp(gc_i - gc_ref)
            fi = torch.exp(gc[ib:ib + bc] - gc[ib])
            fj = torch.exp(gc[ib] - gc)
            ki = R('A')(kk[ib:ib + bc] * fi)
            qf = R('A')(qq[ib:ib + bc] * fi)
            kj = R('A')(kk * fj)
            A[ib:ib + bc] = torch.einsum('ihk,jhk->ijh', ki, kj)
            Aq[ib:ib + bc] = torch.einsum('ihk,jhk->ijh', qf, kj)
        m = torch.tril(torch.ones(BT, BT), -1).bool()
        A = torch.where(m[:, :, None], A * be[:, None, :], torch.zeros_like(A))
        mq = torch.tril(torch.ones(BT, BT)).bool()
        Aq = torch.where(mq[:, :, None], Aq, torch.zeros_like(Aq))
        M = torch.linalg.inv(torch.eye(BT).unsqueeze(0) + A.permute(2, 0, 1)).permute(1, 2, 0)
        u = R('uw')(torch.einsum('ijh,jhv->ihv', M, R('uw')(be[:, :, None] * vv)))
        w = R('uw')(torch.einsum('ijh,jhk->ihk', M, R('uw')(be[:, :, None] * kk * gc.exp())))
        vnew = u - torch.einsum('ihk,hvk->ihv', w, h)          # fp32 保存
        o = R('o')(torch.einsum('ihk,hvk->ihv', R('o')(qq * gc.exp()), h))
        out[c0:c0 + BT] = o + torch.einsum('ijh,jhv->ihv', Aq, R('o')(vnew))
        glast = gc[-1]
        kg = kk * (glast[None] - gc).exp()
        # 形状核对：h ← h·exp(glast) + v'ᵀ·kg
        h = h * glast.exp()[:, None, :] + torch.einsum('ihv,ihk->hvk', vnew, kg)
    return out, h


def stat(name, x, y):
    e = (x - y).abs().max().item()
    rel = e / y.abs().max().item()
    cos = (x * y).sum() / ((x * x).sum().sqrt() * (y * y).sum().sqrt())
    print(f"  {name:34s} max_err {e:.3e}  rel {rel:.2e}  cos {cos:.7f}")


o0, s0 = seq()
print(f"T={T} H={H} K={K} BT={BT}  |out|max {o0.abs().max():.4f}  |state|max {s0.abs().max():.4f}")
for label, only in [("fp32", None), ("bf16 A", {'A'}), ("bf16 A+u/w", {'A', 'uw'}),
                    ("bf16 A+u/w+o", {'A', 'uw', 'o'}), ("bf16 全部", {'A', 'uw', 'o', 'h'})]:
    o, s = chunk(only if only is not None else None)
    stat(f"{label} out", o, o0)
    stat(f"{label} state", s, s0)
