//! Gated DeltaNet 层（Qwen3.5 的 48 个线性注意力层）。数学规格见 `docs/loops/qwen35-gdn.md`（kernel 已三卡对拍）：
//!
//! `x → in_proj_qkv [q|k|v] → conv1d(silu, w=4) → in_proj_z / a / b → gate(g, beta) → recurrent(delta rule)
//!   → gated_rmsnorm(z, silu) → out_proj`
//!
//! 状态（每层每 slot）：conv `[slots, conv_dim, width-1]` bf16 + 递推 `[slots, HV, V, K]` f32，全部由本 crate 自管
//! （`SlotTable` 按 `RequestId` 分 slot，新请求首次出现时清零）；core 的 `KvCacheSpec::Mamba` 接法留 §3 待办。
//!
//! 本里程碑：prefill 也走逐 token 递归（与 chunk 形式数值等价，慢）；perf 项见设计 §3。
use crate::config::Qwen35Config;
use crate::weights::Table;
use anyhow::{ensure, Result};
use dinfer_core::runner::BatchMode;
use dinfer_kernels as k;
use dinfer_layers::{bf16, Gemm, Linear, Stream, Tensor, WeightSource};
use std::collections::HashMap;

/// 一格状态（第 `l` 层的状态张量）。
pub struct GdnState {
    /// `[slots, conv_dim, width-1]` bf16（列 0 最旧）。
    pub conv: Tensor<bf16>,
    /// `[slots, HV, V, K]` f32。
    pub rec: Tensor<f32>,
}

/// 步内元数据（executor 上传后的设备张量）。
pub struct GdnMeta<'a> {
    pub cu_seqlens: &'a Tensor<i32>,
    pub state_indices: &'a Tensor<i32>,
    pub has_initial: &'a Tensor<i32>,
    pub num_reqs: usize,
    pub mode: BatchMode,
}

pub struct GdnLayer {
    in_qkv: Linear,
    in_z: Linear,
    in_a: Linear,
    in_b: Linear,
    conv_w: Tensor<bf16>, // [conv_dim, width]
    a_log: Tensor<f32>,   // [HV]
    dt_bias: Tensor<f32>, // [HV]
    norm_w: Tensor<bf16>, // [V]
    out_proj: Linear,
    pub state: GdnState,
    // scratch
    qkv_buf: Tensor<bf16>,  // [T, conv_dim]
    conv_buf: Tensor<bf16>, // [T, conv_dim]
    z_buf: Tensor<bf16>,    // [T, HV·V]
    a_buf: Tensor<bf16>,    // [T, HV]
    b_buf: Tensor<bf16>,    // [T, HV]
    g_buf: Tensor<f32>,     // [T, HV]
    beta_buf: Tensor<f32>,  // [T, HV]
    o_buf: Tensor<bf16>,    // [T, HV·V]
    n_buf: Tensor<bf16>,    // [T, HV·V]
    /// `reset_slot` 的 dtod 源（一层一个 slot 大小的常驻零）。
    rec_zeros: Option<Tensor<f32>>,
    conv_zeros: Option<Tensor<bf16>>,
}

impl GdnLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        stream: &Stream,
        src: &WeightSource,
        t: &Table,
        layer: usize,
        cfg: &Qwen35Config,
        max_tokens: usize,
        num_slots: usize,
    ) -> Result<Self> {
        let p = format!("layers.{layer}.linear_attn");
        let hv = cfg.linear_num_value_heads as usize;
        let h = cfg.linear_num_key_heads as usize;
        let (kd, vd) = (cfg.linear_key_head_dim as usize, cfg.linear_value_head_dim as usize);
        let width = cfg.linear_conv_kernel_dim as usize;
        let conv_dim = 2 * h * kd + hv * vd;
        let hidden = cfg.hidden_size as usize;

        let in_qkv = t.linear(stream, src, &format!("{p}.in_proj_qkv.weight"), None)?;
        ensure!(in_qkv.w.shape == [conv_dim, hidden], "in_proj_qkv shape {:?}", in_qkv.w.shape);
        let in_z = t.linear(stream, src, &format!("{p}.in_proj_z.weight"), None)?;
        ensure!(in_z.w.shape == [hv * vd, hidden], "in_proj_z shape {:?}", in_z.w.shape);
        let a = t.linear(stream, src, &format!("{p}.in_proj_a.weight"), None)?;
        let b = t.linear(stream, src, &format!("{p}.in_proj_b.weight"), None)?;
        ensure!(a.w.shape == [hv, hidden] && b.w.shape == [hv, hidden], "in_proj_a/b shape {:?} {:?}", a.w.shape, b.w.shape);
        let (in_a, in_b) = (a, b);
        let conv_w = crate::weights::load_conv1d(stream, src, t, layer, conv_dim, width)?;
        let a_log = t.load_f32(stream, src, &format!("{p}.A_log"))?;
        let dt_bias = t.load_f32(stream, src, &format!("{p}.dt_bias"))?;
        ensure!(a_log.numel() == hv && dt_bias.numel() == hv, "A_log/dt_bias {}/{} != HV {hv}", a_log.numel(), dt_bias.numel());
        let norm_w = t.load_bf16(stream, src, &format!("{p}.norm.weight"))?;
        ensure!(norm_w.numel() == vd, "norm.weight {} != V {vd}", norm_w.numel());
        let out_proj = t.linear(stream, src, &format!("{p}.out_proj.weight"), None)?;
        ensure!(out_proj.w.shape == [hidden, hv * vd], "out_proj shape {:?}", out_proj.w.shape);

        let st = cfg.linear_conv_kernel_dim as usize - 1;
        let state = GdnState {
            conv: Tensor::zeros(stream, &[num_slots, conv_dim, st])?,
            rec: Tensor::zeros(stream, &[num_slots, hv, vd, kd])?,
        };
        let (tt, vv) = (max_tokens, hv * vd);
        Ok(Self {
            in_qkv,
            in_z,
            in_a,
            in_b,
            conv_w,
            a_log,
            dt_bias,
            norm_w,
            out_proj,
            state,
            qkv_buf: Tensor::zeros(stream, &[tt, conv_dim])?,
            conv_buf: Tensor::zeros(stream, &[tt, conv_dim])?,
            z_buf: Tensor::zeros(stream, &[tt, vv])?,
            a_buf: Tensor::zeros(stream, &[tt, hv])?,
            b_buf: Tensor::zeros(stream, &[tt, hv])?,
            g_buf: Tensor::zeros(stream, &[tt, hv])?,
            beta_buf: Tensor::zeros(stream, &[tt, hv])?,
            o_buf: Tensor::zeros(stream, &[tt, vv])?,
            n_buf: Tensor::zeros(stream, &[tt, vv])?,
            rec_zeros: Some(Tensor::zeros(stream, &[hv, vd, kd])?),
            conv_zeros: Some(Tensor::zeros(stream, &[conv_dim, st])?),
        })
    }

    /// 清零 slot `s` 的两份状态（新请求首次出现时调用）。
    pub fn reset_slot(&mut self, stream: &Stream, s: usize, cfg: &Qwen35Config) -> Result<()> {
        let hv = cfg.linear_num_value_heads as usize;
        let h = cfg.linear_num_key_heads as usize;
        let (kd, vd) = (cfg.linear_key_head_dim as usize, cfg.linear_value_head_dim as usize);
        let conv_dim = 2 * h * kd + hv * vd;
        let st = cfg.linear_conv_kernel_dim as usize - 1;
        // 只清本 slot（整块 memset 会抹掉其他请求的状态）：从常驻零张量 dtod 一份。
        // 每层每 slot ≈ 3.1 MB f32 + 60 KB bf16（真 27B），仅新请求首次出现时走一次。
        let _ = (conv_dim, st);
        let n = hv * vd * kd;
        {
            let zeros = self.rec_zeros.as_ref().expect("reset_slot: zero buffer");
            let mut dst = self.state.rec.data.slice_mut(s * n..(s + 1) * n);
            stream.inner_arc().memcpy_dtod(&zeros.data, &mut dst)?;
        }
        let cn = self.state.conv.numel() / self.state.conv.shape[0];
        let zeros_c = self.conv_zeros.as_ref().expect("reset_slot: zero buffer");
        let mut cd = self.state.conv.data.slice_mut(s * cn..(s + 1) * cn);
        stream.inner_arc().memcpy_dtod(&zeros_c.data, &mut cd)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        stream: &Stream,
        gemm: &Gemm,
        act: &impl dinfer_layers::Act,
        cfg: &Qwen35Config,
        t: usize,
        x: &Tensor<bf16>,
        meta: &GdnMeta<'_>,
        out: &mut Tensor<bf16>,
    ) -> Result<()> {
        let hv = cfg.linear_num_value_heads as usize;
        let h = cfg.linear_num_key_heads as usize;
        let (kd, vd) = (cfg.linear_key_head_dim as usize, cfg.linear_value_head_dim as usize);
        let width = cfg.linear_conv_kernel_dim as usize;
        let conv_dim = 2 * h * kd + hv * vd;
        let group = hv / h;
        ensure!(t <= self.qkv_buf.shape[0], "gdn: {t} tokens > max {}", self.qkv_buf.shape[0]);
        ensure!(meta.state_indices.numel() >= meta.num_reqs, "gdn: state_indices");

        // 1. 三个投影
        self.in_qkv.forward(gemm, x, &mut self.qkv_buf)?;
        self.in_z.forward(gemm, x, &mut self.z_buf)?;
        self.in_a.forward(gemm, x, &mut self.a_buf)?;
        self.in_b.forward(gemm, x, &mut self.b_buf)?;

        // 2. 短卷积（decode 单 token 走 update，其余走 varlen prefill）
        let single = meta.mode == BatchMode::Decode && meta.num_reqs == t;
        if single {
            k::gdn_conv1d_update(
                stream,
                &self.qkv_buf.data,
                &self.conv_w.data,
                &mut self.conv_buf.data,
                &mut self.state.conv.data,
                Some(&meta.state_indices.data),
                meta.num_reqs,
                conv_dim,
                width,
                width - 1,
                k::GdnAct::Silu,
            )?;
        } else {
            k::gdn_conv1d_prefill(
                stream,
                &self.qkv_buf.data,
                &self.conv_w.data,
                &mut self.conv_buf.data,
                &mut self.state.conv.data,
                &meta.cu_seqlens.data,
                Some(&meta.state_indices.data),
                Some(&meta.has_initial.data),
                meta.num_reqs,
                conv_dim,
                width,
                width - 1,
                k::GdnAct::Silu,
            )?;
        }

        // 3. gate / beta
        k::gdn_gate(
            stream,
            &self.a_buf.data,
            &self.b_buf.data,
            &self.a_log.data,
            &self.dt_bias.data,
            &mut self.g_buf.data,
            &mut self.beta_buf.data,
            t,
            hv,
        )?;

        // 4. delta rule 递归核心（q/k 的 l2norm 与 q 的 scale 在 kernel 内）
        k::gdn_recurrent(
            stream,
            &self.conv_buf.data,
            Some(&self.g_buf.data),
            Some(&self.beta_buf.data),
            None,
            None,
            &self.a_log.data,
            &self.dt_bias.data,
            &mut self.o_buf.data,
            &mut self.state.rec.data,
            &meta.cu_seqlens.data,
            Some(&meta.state_indices.data),
            meta.num_reqs,
            h,
            hv,
            kd,
            vd,
            (kd as f32).powf(-0.5),
        )?;

        // 5. 输出门 + RMSNorm（z 门，silu）
        k::gdn_gated_rmsnorm(
            stream,
            &self.o_buf.data,
            &self.norm_w.data,
            &self.z_buf.data,
            &mut self.n_buf.data,
            t * hv,
            vd,
            cfg.rms_norm_eps,
            k::GdnAct::Silu,
        )?;

        // 6. out_proj
        let _ = act;
        let _ = group;
        self.out_proj.forward(gemm, &self.n_buf, out)
    }
}

/// 请求 → slot 分配表。`RequestId` 首次出现分一个 slot（调用方负责清零该 slot 的状态）；
/// slot 不够时按"本步不在场且最久未见"的顺序回收。
pub struct SlotTable {
    slots: HashMap<u64, usize>,
    last_seen: HashMap<u64, u64>,
    free: Vec<usize>,
    cap: usize,
    step: u64,
}

impl SlotTable {
    pub fn new(cap: usize) -> Self {
        Self { slots: HashMap::new(), last_seen: HashMap::new(), free: (0..cap).rev().collect(), cap, step: 0 }
    }

    pub fn cap(&self) -> usize {
        self.cap
    }

    /// 请求结束（finish / abort）：归还 slot。不调用它也能工作（LRU 回收兜底），但兜底只能淘汰"本步不在场"的请求——
    /// 分块 prefill / 被抢占的活请求也可能不在场，会被误当受害者（2026-10-19 B300 c8-long bench 87 条 slot 复用告警）。
    pub fn release(&mut self, req_ids: &[u64]) {
        for r in req_ids {
            if let Some(s) = self.slots.remove(r) {
                self.last_seen.remove(r);
                self.free.push(s);
            }
        }
    }

    /// 开始一步：返回本步新分配的 slot（调用方需清零）。
    pub fn begin_step(&mut self, req_ids: &[u64]) -> Result<Vec<usize>> {
        self.step += 1;
        let mut fresh = Vec::new();
        for &r in req_ids {
            if self.slots.contains_key(&r) {
                self.last_seen.insert(r, self.step);
                continue;
            }
            let s = match self.free.pop() {
                Some(s) => s,
                None => {
                    // 回收：本步不在场、且最久未见的请求
                    let victim = self
                        .last_seen
                        .iter()
                        .filter(|(id, _)| !req_ids.contains(id))
                        .min_by_key(|(_, &st)| st)
                        .map(|(id, _)| *id);
                    match victim {
                        Some(v) => {
                            let s = self.slots.remove(&v).expect("victim slot");
                            self.last_seen.remove(&v);
                            tracing::debug!(req = v, slot = s, "qwen35: GDN slot 回收");
                            s
                        }
                        None => anyhow::bail!(
                            "qwen35: GDN slot 用尽（cap {}，本步 {} 请求）；提高 max_num_seqs 预算",
                            self.cap,
                            req_ids.len()
                        ),
                    }
                }
            };
            self.slots.insert(r, s);
            self.last_seen.insert(r, self.step);
            fresh.push(s);
        }
        Ok(fresh)
    }

    /// 本步每请求的 slot（按 `req_ids` 顺序）。
    pub fn indices(&self, req_ids: &[u64]) -> Vec<i32> {
        req_ids.iter().map(|r| *self.slots.get(r).expect("slot assigned in begin_step") as i32).collect()
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::SlotTable;

    #[test]
    fn slot_table_allocates_and_recycles() {
        let mut t = SlotTable::new(2);
        assert_eq!(t.begin_step(&[1, 2]).unwrap(), vec![0, 1]);
        assert_eq!(t.indices(&[2, 1]), vec![1, 0]);
        // 2 号不在场 → 被回收给 3（复用 2 的 slot 1）
        assert_eq!(t.begin_step(&[1, 3]).unwrap(), vec![1]);
        assert_eq!(t.indices(&[3, 1]), vec![1, 0]);
        // 两个都在场且要新 slot → 报错
        assert!(t.begin_step(&[1, 3, 4]).is_err());
    }
}
