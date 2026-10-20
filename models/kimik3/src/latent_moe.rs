//! Kimi-K3 latent MoE（`docs/design-kimik3.md` §1 "MoE" 行、`docs/loops/k3-moe.md`），HF `KimiSparseMoeBlock`：
//!
//! ```text
//! ids, w  = gate(x)                                  路由在 hidden=7168 上：fp32 logits → sigmoid → +bias → top-16 →
//!                                                    w = sigmoid 分数（不含 bias）/ Σ（renormalize）× scaling(1.0)
//! z       = routed_expert_down_proj(x)               7168 → 3584（latent），bf16
//! y_e     = w2_e( situ(w1_e z | w3_e z) )            896 专家在 3584 维工作（w1/w3: 3584→3072，w2: 3072→3584），MXFP4
//! y       = Σ_k w_k · y_{ids_k}                      bf16(y_e) × f32 权重，f32 累加（HF `.type(topk_weight.dtype).mul_().sum()`）
//! y       = routed_expert_up_proj( routed_expert_norm(y) )   3584 → 7168
//! out     = y + shared_experts(x)                    2 个 shared：普通 KimiMLP（intermediate 2×3072，situ），bf16
//! ```
//!
//! 专家权重**保持 MXFP4 packed 常驻**（每层 896 × (w13 packed 5.5 MB + scale 0.34 MB + w2 packed 5.5 MB + scale 0.34 MB)
//! ≈ 10.5 GB），GEMM 走 `mxfp4_gemm_grouped`（kernel 内解包 + bf16 乘加，`crates/kernels/cuda/mxfp4_gemm_sm8x.cu`）；
//! 不整层解包。w1/w3 装载时按输出维拼成 `w13 [E, 2·I, L/2]`，一条 grouped GEMM 出 `gate|up`，再 `situ_and_mul`。
//!
//! 路由 / 分桶 / gather / scatter 复用 `moe.cu`（GLM 线同一套）；每步一次 D2H 读 `offsets`（tile 计划在 host 展平，
//! 见 k3-moe.md TODO 2）。
use crate::config::KimiLinearConfig;
use crate::weights::Table;
use anyhow::{ensure, Context, Result};
use dinfer_kernels::cudarc::driver::CudaSlice;
use dinfer_kernels::ops as k;
use dinfer_layers::{bf16, Act, Gemm, Linear, Mlp, RmsNorm, Stream, Tensor, WeightSource};

/// `situ` 激活（HF `SituAndMul`）：`beta·tanh(g/beta)·sigmoid(g) · linear_beta·tanh(u/linear_beta)`。
/// 以 `Act` 形式接进 `dinfer_layers::Mlp`（shared experts / dense MLP 用），参数来自 config。
#[derive(Debug, Clone, Copy)]
pub struct Situ {
    pub beta: f32,
    pub linear_beta: Option<f32>,
}

impl Situ {
    pub fn from_config(cfg: &KimiLinearConfig) -> Result<Self> {
        ensure!(cfg.hidden_act == "situ", "hidden_act={} 不是 situ", cfg.hidden_act);
        Ok(Self {
            beta: cfg.activation_situ_beta.unwrap_or(1.0),
            linear_beta: cfg.activation_situ_linear_beta,
        })
    }
}

impl Act for Situ {
    /// 名字沿用 trait（gate|up 拼接布局相同），语义是 situ 而不是 silu。
    fn silu_and_mul(
        &self,
        stream: &Stream,
        x: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
        tokens: usize,
        inter: usize,
    ) -> Result<()> {
        k::situ_and_mul_bf16_n(stream, x, out, tokens, inter, self.beta, self.linear_beta)
    }
}

/// 一层的全部专家权重（packed 常驻）。布局 `[E, N, K/2]` u8 + `[E, N, K/group]` u8，与 `mxfp4_gemm_grouped` 一致。
pub struct ExpertPack {
    pub w13_packed: CudaSlice<u8>,
    pub w13_scale: CudaSlice<u8>,
    pub w2_packed: CudaSlice<u8>,
    pub w2_scale: CudaSlice<u8>,
    pub num_experts: usize,
    /// latent 维（专家输入 / w2 输出），3584。
    pub latent: usize,
    /// moe_intermediate_size，3072。
    pub inter: usize,
    pub group: usize,
    pub bytes: usize,
}

impl ExpertPack {
    /// 从 `layers.{l}.block_sparse_moe.experts.{e}.w{1,2,3}.weight_{packed,scale}` 装载（HF 每专家分开存；这里拼成大张量）。
    pub fn load(
        stream: &Stream,
        src: &WeightSource,
        t: &Table,
        prefix: &str,
        num_experts: usize,
        latent: usize,
        inter: usize,
    ) -> Result<Self> {
        let group = quant_group(src, t, &format!("{prefix}.experts.0.w1"), latent)?;
        let (w13_n, w13_k, w2_n, w2_k) = (2 * inter, latent, latent, inter);
        let w13_pk = w13_n * w13_k / 2;
        let w13_sc = w13_n * w13_k / group;
        let w2_pk = w2_n * w2_k / 2;
        let w2_sc = w2_n * w2_k / group;
        let mut w13_packed = stream.alloc_uninit::<u8>(num_experts * w13_pk)?;
        let mut w13_scale = stream.alloc_uninit::<u8>(num_experts * w13_sc)?;
        let mut w2_packed = stream.alloc_uninit::<u8>(num_experts * w2_pk)?;
        let mut w2_scale = stream.alloc_uninit::<u8>(num_experts * w2_sc)?;
        let half_pk = w13_pk / 2;
        let half_sc = w13_sc / 2;
        for e in 0..num_experts {
            let p = format!("{prefix}.experts.{e}");
            // w1 → w13 的前 inter 行，w3 → 后 inter 行（HF `cat([w1(x), w3(x)])`，gate 在前）。
            for (i, w) in ["w1", "w3"].into_iter().enumerate() {
                let pk = raw_u8(src, t, &format!("{p}.{w}.weight_packed"), &[inter, latent / 2])?;
                let sc = raw_u8(src, t, &format!("{p}.{w}.weight_scale"), &[inter, latent / group])?;
                let off_pk = e * w13_pk + i * half_pk;
                let off_sc = e * w13_sc + i * half_sc;
                h2d(stream, pk, &mut w13_packed, off_pk)?;
                h2d(stream, sc, &mut w13_scale, off_sc)?;
            }
            let pk = raw_u8(src, t, &format!("{p}.w2.weight_packed"), &[latent, inter / 2])?;
            let sc = raw_u8(src, t, &format!("{p}.w2.weight_scale"), &[latent, inter / group])?;
            h2d(stream, pk, &mut w2_packed, e * w2_pk)?;
            h2d(stream, sc, &mut w2_scale, e * w2_sc)?;
        }
        stream.synchronize()?;
        let bytes = num_experts * (w13_pk + w13_sc + w2_pk + w2_sc);
        tracing::info!(
            experts = num_experts,
            latent,
            inter,
            group,
            per_expert_mb = (w13_pk + w13_sc + w2_pk + w2_sc) as f64 / 1e6,
            total_gb = bytes as f64 / 1e9,
            "kimik3 latent MoE: 专家权重 MXFP4 packed 常驻"
        );
        Ok(Self { w13_packed, w13_scale, w2_packed, w2_scale, num_experts, latent, inter, group, bytes })
    }
}

/// `dst[off .. off+src.len()] = src`（H2D 到大张量的一段）。
fn h2d(stream: &Stream, src: &[u8], dst: &mut CudaSlice<u8>, off: usize) -> Result<()> {
    ensure!(off + src.len() <= dst.len(), "h2d: 越界");
    let mut view = dst.slice_mut(off..off + src.len());
    stream.inner_arc().memcpy_htod(src, &mut view)?;
    Ok(())
}

/// `weight_scale [N, K/group]` 推出 group（compressed-tensors `mxfp4-pack-quantized` 是 32，这里按文件反推再校验）。
fn quant_group(src: &WeightSource, t: &Table, key: &str, k: usize) -> Result<usize> {
    let info = src.info(t.name(&format!("{key}.weight_scale"))?).context("weight_scale")?;
    ensure!(info.shape.len() == 2 && k % info.shape[1] == 0, "{key}.weight_scale shape {:?}", info.shape);
    let g = k / info.shape[1];
    ensure!(g == 32, "{key}: group {g} ≠ 32（kernel 按 OCP MX group 32 写）");
    Ok(g)
}

fn raw_u8<'a>(src: &'a WeightSource, t: &Table, key: &str, shape: &[usize]) -> Result<&'a [u8]> {
    let name = t.name(key)?;
    let (info, b) = src.bytes(name)?;
    ensure!(info.dtype == "U8", "{key}: dtype {}（期望 U8）", info.dtype);
    ensure!(info.shape == shape, "{key}: shape {:?}（期望 {shape:?}）", info.shape);
    ensure!(b.len() == shape.iter().product::<usize>(), "{key}: 字节数");
    Ok(b)
}

/// 路由参数（HF `KimiMoEGate`）。
#[derive(Debug, Clone)]
pub struct RouterCfg {
    pub num_experts: usize,
    pub topk: usize,
    pub renormalize: bool,
    pub scale: f32,
}

impl RouterCfg {
    pub fn from_config(cfg: &KimiLinearConfig) -> Result<Self> {
        ensure!(
            cfg.moe_router_activation_func == "sigmoid",
            "router {} 未实现（kernel 只做 sigmoid）",
            cfg.moe_router_activation_func
        );
        ensure!(
            cfg.num_expert_group <= 1 || cfg.num_expert_group <= cfg.topk_group,
            "分组路由（n_group>{}）未实现",
            cfg.topk_group
        );
        ensure!(cfg.topk_method == "noaux_tc", "topk_method {} 未实现", cfg.topk_method);
        Ok(Self {
            num_experts: cfg.num_experts as usize,
            topk: cfg.num_experts_per_token as usize,
            renormalize: cfg.moe_renormalize,
            scale: cfg.routed_scaling_factor,
        })
    }
}

/// CPU 参考路由（单测 / 对拍用；与 `moe.cu::route_kernel` 同口径）：返回 `(ids[t*K], w[t*K])`。
pub fn route_cpu(
    logits: &[f32],
    bias: Option<&[f32]>,
    tokens: usize,
    cfg: &RouterCfg,
) -> (Vec<i32>, Vec<f32>) {
    let (e, kk) = (cfg.num_experts, cfg.topk);
    let mut ids = Vec::with_capacity(tokens * kk);
    let mut ws = Vec::with_capacity(tokens * kk);
    for t in 0..tokens {
        let sc: Vec<f32> = logits[t * e..(t + 1) * e].iter().map(|l| 1.0 / (1.0 + (-l).exp())).collect();
        let mut choice: Vec<f32> =
            sc.iter().enumerate().map(|(i, s)| s + bias.map_or(0.0, |b| b[i])).collect();
        let mut sel = Vec::with_capacity(kk);
        for _ in 0..kk {
            let (best, _) = choice.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |acc, (i, &v)| {
                if v > acc.1 {
                    (i, v)
                } else {
                    acc
                }
            });
            sel.push(best);
            choice[best] = f32::NEG_INFINITY;
        }
        let sum: f32 = sel.iter().map(|&i| sc[i]).sum();
        let denom = if cfg.renormalize { sum + 1e-20 } else { 1.0 };
        for &i in &sel {
            ids.push(i as i32);
            ws.push(sc[i] / denom * cfg.scale);
        }
    }
    (ids, ws)
}

pub struct LatentMoe {
    pub router: RouterCfg,
    pub hidden: usize,
    pub act: Situ,
    gate: Linear,              // [E, hidden] bf16
    bias: Option<Tensor<f32>>, // e_score_correction_bias [E]
    down: Linear,              // routed_expert_down_proj [latent, hidden]
    up: Linear,                // routed_expert_up_proj [hidden, latent]
    norm: Option<RmsNorm>,     // routed_expert_norm [latent]
    pub experts: ExpertPack,
    shared: Option<Mlp>,
    buckets: k::MoeBuckets,
    offsets_host: Vec<i32>,
    tile_bm: usize,
    tile_bn: usize,
    // scratch（按 max_tokens 预分配）
    logits: Tensor<f32>,
    topk_ids: Tensor<i32>,
    topk_w: Tensor<f32>,
    z: Tensor<bf16>,    // [T, latent] down_proj 输出
    zg: Tensor<bf16>,   // [T·K, latent] gather
    gu: Tensor<bf16>,   // [T·K, 2·inter]
    a: Tensor<bf16>,    // [T·K, inter]
    yg: Tensor<bf16>,   // [T·K, latent]
    acc: Tensor<f32>,   // [T, latent]
    ysum: Tensor<bf16>, // [T, latent] 加权和（norm 前）
    yn: Tensor<bf16>,   // [T, latent] norm 后
    yup: Tensor<bf16>,  // [T, hidden] up_proj 输出（routed 部分）
}

impl LatentMoe {
    /// `layer` 必须是 MoE 层。`max_tokens` 决定激活缓冲。
    pub fn load(
        stream: &Stream,
        src: &WeightSource,
        t: &Table,
        layer: usize,
        cfg: &KimiLinearConfig,
        max_tokens: usize,
    ) -> Result<Self> {
        ensure!(cfg.is_moe_layer(layer as u32), "第 {layer} 层不是 MoE 层");
        let router = RouterCfg::from_config(cfg)?;
        let act = Situ::from_config(cfg)?;
        let hidden = cfg.hidden_size as usize;
        let latent =
            cfg.routed_expert_hidden_size.context("routed_expert_hidden_size（latent MoE）缺失")? as usize;
        let inter = cfg.moe_intermediate_size as usize;
        let p = format!("layers.{layer}.block_sparse_moe");
        let e = router.num_experts;

        let gate = t.linear(stream, src, &format!("{p}.gate.weight"))?;
        ensure!(gate.w.shape == [e, hidden], "gate.weight shape {:?}", gate.w.shape);
        let bias_key = format!("{p}.gate.e_score_correction_bias");
        let bias = if t.contains(&bias_key) {
            let b = t.load_f32(stream, src, &bias_key)?;
            ensure!(b.numel() == e, "e_score_correction_bias {} ≠ E", b.numel());
            Some(b)
        } else {
            None
        };
        let down = t.linear(stream, src, &format!("{p}.routed_expert_down_proj.weight"))?;
        ensure!(down.w.shape == [latent, hidden], "routed_expert_down_proj shape {:?}", down.w.shape);
        let up = t.linear(stream, src, &format!("{p}.routed_expert_up_proj.weight"))?;
        ensure!(up.w.shape == [hidden, latent], "routed_expert_up_proj shape {:?}", up.w.shape);
        let norm = if cfg.latent_moe_use_norm {
            let n = t.rmsnorm(stream, src, &format!("{p}.routed_expert_norm.weight"), cfg.rms_norm_eps)?;
            ensure!(n.hidden == latent, "routed_expert_norm {} ≠ latent", n.hidden);
            Some(n)
        } else {
            None
        };
        let experts = ExpertPack::load(stream, src, t, &p, e, latent, inter)?;
        let shared = if cfg.num_shared_experts > 0 {
            let m = Mlp::load(
                stream,
                src,
                &format!("{}{p}.shared_experts", crate::weights::LM_PREFIX),
                max_tokens,
            )?;
            ensure!(
                m.inter == inter * cfg.num_shared_experts as usize,
                "shared_experts inter {} ≠ {}×{inter}",
                m.inter,
                cfg.num_shared_experts
            );
            Some(m)
        } else {
            None
        };
        let kk = router.topk;
        let pairs = max_tokens * kk;
        Ok(Self {
            buckets: k::MoeBuckets::new(stream, e, max_tokens, kk)?,
            offsets_host: vec![0; e + 1],
            tile_bm: 16,
            tile_bn: 64,
            logits: Tensor::zeros(stream, &[max_tokens, e])?,
            topk_ids: Tensor::zeros(stream, &[max_tokens, kk])?,
            topk_w: Tensor::zeros(stream, &[max_tokens, kk])?,
            z: Tensor::zeros(stream, &[max_tokens, latent])?,
            zg: Tensor::zeros(stream, &[pairs, latent])?,
            gu: Tensor::zeros(stream, &[pairs, 2 * inter])?,
            a: Tensor::zeros(stream, &[pairs, inter])?,
            yg: Tensor::zeros(stream, &[pairs, latent])?,
            acc: Tensor::zeros(stream, &[max_tokens, latent])?,
            ysum: Tensor::zeros(stream, &[max_tokens, latent])?,
            yn: Tensor::zeros(stream, &[max_tokens, latent])?,
            yup: Tensor::zeros(stream, &[max_tokens, hidden])?,
            router,
            hidden,
            act,
            gate,
            bias,
            down,
            up,
            norm,
            experts,
            shared,
        })
    }

    /// grouped GEMM 的 tile 尺寸（必须与 `mxfp4_gemm_sm8x.cu` host 侧模板参数一致；默认 16/64，按 `ArchProfile` 可改）。
    pub fn set_tiles(&mut self, bm: usize, bn: usize) {
        self.tile_bm = bm;
        self.tile_bn = bn;
    }

    /// `out[:t] = MoE(x[:t])`（覆盖写）。`x` 已过 `post_attention_layernorm`。
    pub fn forward(
        &mut self,
        stream: &Stream,
        gemm: &Gemm,
        t: usize,
        x: &Tensor<bf16>,
        out: &mut Tensor<bf16>,
    ) -> Result<()> {
        let (h, e, kk) = (self.hidden, self.router.num_experts, self.router.topk);
        let (latent, inter, group) = (self.experts.latent, self.experts.inter, self.experts.group);
        ensure!(t <= self.z.shape[0], "LatentMoe: {t} tokens > max {}", self.z.shape[0]);
        ensure!(x.numel() >= t * h && out.numel() >= t * h, "LatentMoe: x/out 缓冲");

        // 1. 路由（fp32 logits 在 hidden 维上）
        k::moe_gate_f32(stream, &x.data, &self.gate.w.data, &mut self.logits.data, t, e, h)?;
        k::moe_route_f32(
            stream,
            &self.logits.data,
            self.bias.as_ref().map(|b| &b.data),
            &mut self.topk_ids.data,
            &mut self.topk_w.data,
            t,
            e,
            kk,
            self.router.renormalize,
            self.router.scale,
        )?;
        // 2. latent 投影 + 分桶（一次 D2H 读 offsets，host 展平 tile 计划）
        self.down.forward_rows(gemm, x, &mut self.z, t)?;
        k::moe_bucket(stream, &self.topk_ids.data, &mut self.buckets, t, kk)?;
        self.offsets_host = stream.dtoh_sync_copy(&self.buckets.offsets)?;
        let pairs = self.offsets_host[e] as usize;
        ensure!(pairs == t * kk, "LatentMoe: pairs {pairs} ≠ t·K");
        // 3. gather 全部 (token, k) 行 → grouped GEMM w13 → situ → grouped GEMM w2
        k::moe_gather_bf16(
            stream,
            &self.z.data,
            &mut self.zg.data,
            &self.buckets.sorted_tok,
            0,
            pairs,
            latent,
        )?;
        let tiles13 =
            k::Mxfp4GemmTiles::new(stream, &self.offsets_host, 2 * inter, self.tile_bm, self.tile_bn)?;
        k::mxfp4_gemm_grouped(
            stream,
            &self.zg.data,
            &self.experts.w13_packed,
            &self.experts.w13_scale,
            &self.buckets.offsets,
            &tiles13,
            &mut self.gu.data,
            latent,
            2 * inter,
            group,
        )?;
        k::situ_and_mul_bf16_n(
            stream,
            &self.gu.data,
            &mut self.a.data,
            pairs,
            inter,
            self.act.beta,
            self.act.linear_beta,
        )?;
        let tiles2 = k::Mxfp4GemmTiles::new(stream, &self.offsets_host, latent, self.tile_bm, self.tile_bn)?;
        k::mxfp4_gemm_grouped(
            stream,
            &self.a.data,
            &self.experts.w2_packed,
            &self.experts.w2_scale,
            &self.buckets.offsets,
            &tiles2,
            &mut self.yg.data,
            inter,
            latent,
            group,
        )?;
        // 4. 加权散回（f32 累加）→ bf16 → norm → up_proj
        stream.memset_zeros_n(&mut self.acc.data, t * latent)?;
        k::moe_scatter_add_f32(
            stream,
            &self.yg.data,
            &mut self.acc.data,
            &self.buckets.sorted_tok,
            &self.buckets.sorted_k,
            &self.topk_w.data,
            0,
            pairs,
            kk,
            latent,
        )?;
        stream.memset_zeros_n(&mut self.ysum.data, t * latent)?;
        k::f32_add_to_bf16_n(stream, &self.acc.data, &mut self.ysum.data, t * latent)?;
        let y_in = if let Some(n) = &self.norm {
            k::rmsnorm_bf16_n(stream, &self.ysum.data, &n.w.data, &mut self.yn.data, t, latent, n.eps)?;
            &self.yn
        } else {
            &self.ysum
        };
        self.up.forward_rows(gemm, y_in, &mut self.yup, t)?;
        // 5. shared experts → out；out += routed
        if let Some(sh) = &mut self.shared {
            sh.forward(stream, gemm, &self.act, t, x, out)?;
            k::add_inplace_bf16_n(stream, &self.yup.data, &mut out.data, t * h)?;
        } else {
            stream.dtod_copy_n(&self.yup.data, &mut out.data, t * h)?;
        }
        Ok(())
    }

    /// 对拍：上一步路由结果 `(ids[t·K], w[t·K])`。
    pub fn last_routing(&self, stream: &Stream, t: usize) -> Result<(Vec<i32>, Vec<f32>)> {
        let kk = self.router.topk;
        let ids = self.topk_ids.to_vec(stream)?;
        let w = self.topk_w.to_vec(stream)?;
        Ok((ids[..t * kk].to_vec(), w[..t * kk].to_vec()))
    }
    /// 对拍：上一步 fp32 路由 logits `[t, E]`。
    pub fn last_logits(&self, stream: &Stream, t: usize) -> Result<Vec<f32>> {
        Ok(self.logits.to_vec(stream)?[..t * self.router.num_experts].to_vec())
    }
    /// 对拍：专家加权和（`routed_expert_norm` 前）`[t, latent]`。
    pub fn last_expert_sum(&self) -> &Tensor<bf16> {
        &self.ysum
    }
    /// 对拍：routed 分支 `up_proj` 输出（加 shared 前）`[t, hidden]`。
    pub fn last_routed_out(&self) -> &Tensor<bf16> {
        &self.yup
    }
    /// 每 token 的专家 GEMM 次数（grouped：每步 2 条 kernel；按 pair 算每 token K×2 个专家 GEMM）。
    pub fn gemms_per_token(&self) -> usize {
        self.router.topk * 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_cpu_sigmoid_bias_topk_renorm() {
        // 3 token × 8 专家，top-2；bias 把专家 7 抬高让它被选，但权重用不含 bias 的 sigmoid 分数。
        let cfg = RouterCfg { num_experts: 8, topk: 2, renormalize: true, scale: 1.0 };
        let mut logits = vec![0f32; 3 * 8];
        logits[0] = 3.0; // t0：专家 0 最高
        logits[1] = 1.0;
        logits[8 + 5] = 2.0; // t1：专家 5
        logits[16 + 2] = -1.0; // t2：全 0 除专家 2 更低 → 选 0、1（相等取小下标）
        let mut bias = vec![0f32; 8];
        bias[7] = 10.0;
        let (ids, w) = route_cpu(&logits, Some(&bias), 3, &cfg);
        assert_eq!(&ids[0..2], &[7, 0]);
        assert_eq!(&ids[2..4], &[7, 5]);
        assert_eq!(&ids[4..6], &[7, 0]);
        let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
        let (s7, s0) = (sig(0.0), sig(3.0));
        assert!((w[0] - s7 / (s7 + s0)).abs() < 1e-6 && (w[1] - s0 / (s7 + s0)).abs() < 1e-6);
        assert!((w[0] + w[1] - 1.0).abs() < 1e-6, "renormalize 后和为 1");
        // 不归一化 + scaling
        let cfg2 = RouterCfg { renormalize: false, scale: 2.5, ..cfg };
        let (_, w2) = route_cpu(&logits, None, 3, &cfg2);
        assert!((w2[0] - 2.5 * sig(3.0)).abs() < 1e-6);
    }

    #[test]
    fn situ_matches_reference_formula() {
        // 公式自检（kernel 与此同式）：beta=4, linear_beta=25
        let (beta, lb) = (4.0f32, 25.0f32);
        let situ =
            |g: f32, u: f32| beta * (g / beta).tanh() * (1.0 / (1.0 + (-g).exp())) * (lb * (u / lb).tanh());
        assert!((situ(0.0, 1.0)).abs() < 1e-7);
        assert!((situ(1.0, 1.0) - 4.0 * (0.25f32).tanh() * 0.7310586 * 25.0 * (0.04f32).tanh()).abs() < 1e-5);
        // 大 gate 饱和到 beta·up'
        assert!((situ(100.0, 2.0) - beta * 25.0 * (0.08f32).tanh()).abs() < 1e-4);
    }
}
