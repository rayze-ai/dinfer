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
