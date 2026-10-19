//! `ArchProfile`：按卡（compute capability）给出的默认数值参数（规格 `docs/loops/arch-profile.md`）。
//!
//! 硬性规则：kernel 内不写卡型 / 模型相关调优数值；数值来自这里的默认值 + 启动自调优（按模型形状实测），
//! 环境变量只做覆盖并打 info 日志。本文件目前只承载 GEMM 自调优用到的字段；attention / 采样等后续迁入。

use crate::device::Device;
use anyhow::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    /// sm_80 / 86 / 89（A100 / 3090 / 4090）
    Sm8x,
    /// sm_90（H100 / H200）
    Sm90,
    /// sm_100 / 103（B200 / B300）
    Sm100,
}

/// decode 小 M GEMM 的自调优参数。全部是"怎么搜"的数值，不是某个 kernel 的最终配置。
#[derive(Clone, Debug)]
pub struct GemmTuneParams {
    /// 参与自调优的 M 桶（与 decode CUDA graph 的 batch 桶一致）。
    pub m_buckets: Vec<usize>,
    /// 走自写小 M kernel 的 M 上限（kernel 的 M_TILE 最大 8）。
    pub small_m_max: usize,
    /// cuBLASLt 启发式候选数（`cublasLtMatmulAlgoGetHeuristic` 的 requestedAlgoCount）。
    pub lt_candidates: usize,
    /// 小 M kernel split-K 的候选：block 总数 ≈ 目标倍数 × SM 数，每个倍数一个候选（去重）。
    pub small_m_wave_targets: Vec<usize>,
    /// 每个候选的计时次数（取平均）；首次先预热 1 次。
    pub timing_iters: usize,
    /// split-K 工作区上限（字节）：候选的 partial 超过它就不参与。
    pub split_workspace_max_bytes: usize,
}

#[derive(Clone, Debug)]
pub struct ArchProfile {
    pub arch: Arch,
    pub cc: (i32, i32),
    pub sm_count: usize,
    /// cuBLASLt 工作区大小（cudarc 同款策略：sm90+ 32 MiB，其余 4 MiB）。
    pub cublaslt_workspace_bytes: usize,
    pub gemm_tune: GemmTuneParams,
}

impl ArchProfile {
    pub fn detect(dev: &Device) -> Result<Self> {
        let cc = dev.compute_capability()?;
        let sm_count = dev.sm_count()?;
        Ok(Self::for_cc(cc, sm_count))
    }

    pub fn for_cc(cc: (i32, i32), sm_count: usize) -> Self {
        let arch = match cc.0 {
            ..=8 => Arch::Sm8x,
            9 => Arch::Sm90,
            _ => Arch::Sm100,
        };
        let cublaslt_workspace_bytes = if cc.0 >= 9 { 32 << 20 } else { 4 << 20 };
        let gemm_tune = GemmTuneParams {
            m_buckets: vec![1, 2, 4, 8, 16, 32, 64],
            small_m_max: 8,
            lt_candidates: 8,
            // 小卡（3090 82 SM）1–4 波就够；大卡（B300 148 SM）小 N 形状需要更多 block 才能把 HBM 喂满
            small_m_wave_targets: match arch {
                Arch::Sm8x => vec![1, 2, 4],
                Arch::Sm90 | Arch::Sm100 => vec![1, 2, 4, 8],
            },
            timing_iters: 5,
            split_workspace_max_bytes: 64 << 20,
        };
        Self { arch, cc, sm_count, cublaslt_workspace_bytes, gemm_tune }
    }

    /// `gdn_recurrent` decode 的 V 维切分：(seq, v 头) 共 `base_blocks` 个 block，不足 SM 数时按 V 切 2/4 份补齐；
    /// 结果必须整除 `v_dim`，上限 `GDN_VSPLIT_MAX`（每块再小则 q/k 归一化的重复计算占比上升）。
    pub fn gdn_recurrent_vsplit(sm_count: usize, base_blocks: usize, v_dim: usize) -> usize {
        const GDN_VSPLIT_MAX: usize = 4;
        let want = sm_count.div_ceil(base_blocks.max(1)).clamp(1, GDN_VSPLIT_MAX);
        (1..=want).rev().find(|s| v_dim % s == 0).unwrap_or(1)
    }

    /// `gdn_recurrent` decode 寄存器版 kernel 的 V 块大小 BV（kernel 支持 `GDN_DECODE_BV_CANDIDATES`）。
    /// 每 block 处理 (seq, v 头, BV 行)：BV 越大每 warp 持的状态行越多（访存并行度高、q/k 归一化重复算的占比低），
    /// 但 block 数 = `base_blocks · v_dim / BV` 越少；取最大的、仍能把 block 数铺到 ≥ `GDN_DECODE_WAVES` 波 SM 的 BV，
    /// 都铺不满（B=1）则取最小档。返回 0 = 形状不支持（K ∉ {64,128} 或 V 不被任何档整除），调用方回老 kernel。
    pub fn gdn_decode_bv(sm_count: usize, base_blocks: usize, k_dim: usize, v_dim: usize) -> usize {
        const GDN_DECODE_BV_CANDIDATES: [usize; 3] = [64, 32, 16];
        const GDN_DECODE_WAVES: usize = 2;
        if k_dim != 64 && k_dim != 128 {
            return 0;
        }
        let mut fallback = 0;
        for &bv in &GDN_DECODE_BV_CANDIDATES {
            if v_dim % bv != 0 {
                continue;
            }
            if base_blocks * (v_dim / bv) >= GDN_DECODE_WAVES * sm_count {
                return bv;
            }
            fallback = bv;
        }
        fallback
    }

    /// 小 M kernel 的 split-K 候选（去重、升序、≥1）。`blocks_per_split` = 不切时的 block 数（ceil(N / warps/block)）。
    pub fn small_m_split_candidates(&self, blocks_per_split: usize, k: usize) -> Vec<usize> {
        let mut v: Vec<usize> = vec![1];
        for &w in &self.gemm_tune.small_m_wave_targets {
            let want = w * self.sm_count;
            let s = want.div_ceil(blocks_per_split.max(1));
            // 每份至少 8 列（kernel 要求 8 对齐），份数不超过 k/8
            let s = s.clamp(1, (k / 8).max(1));
            v.push(s);
        }
        v.sort_unstable();
        v.dedup();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::ArchProfile;

    #[test]
    fn gdn_decode_bv_tiers() {
        // 27B 形状 HV=48 K=V=128：B=1 铺不满取小档，B≥4 都取 64；三种卡的 SM 数
        for (sm, b1) in [(82usize, 32usize), (128, 16), (148, 16)] {
            assert_eq!(ArchProfile::gdn_decode_bv(sm, 48, 128, 128), b1, "sm={sm} B=1");
            assert_eq!(ArchProfile::gdn_decode_bv(sm, 4 * 48, 128, 128), 64, "sm={sm} B=4");
            assert_eq!(ArchProfile::gdn_decode_bv(sm, 32 * 48, 128, 128), 64, "sm={sm} B=32");
        }
        // K 不支持 → 0；V 只被 16/32 整除时不选 64
        assert_eq!(ArchProfile::gdn_decode_bv(82, 48, 96, 128), 0);
        assert_eq!(ArchProfile::gdn_decode_bv(82, 4 * 48, 64, 32), 32);
    }
}
