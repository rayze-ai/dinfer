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

/// `gdn_recurrent` decode 寄存器版 kernel 的变体（`gdn.cu` 模板开关；C ABI 编码见 `code()`）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdnDecodeVariant {
    /// 第 4 轮 kernel：每行两次 warp_sum（p、o），两遍 FMA。
    Base,
    /// 转置归约：ROWS 行的 {S·k, S·q} 一次多值 warp 归约，一遍 FMA。
    Tr,
    /// 持久 block + cp.async 预取下一 tile 的状态（含 Tr）。
    Pipe,
    /// 持久 block + cp.async，但用 Base 的归约（隔离流水本身的收益）。
    PipeBase,
}

impl GdnDecodeVariant {
    pub fn code(self) -> i32 {
        match self {
            Self::Base => 0,
            Self::Tr => 1,
            Self::Pipe => 2,
            Self::PipeBase => 3,
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "base" => Some(Self::Base),
            "tr" => Some(Self::Tr),
            "pipe" => Some(Self::Pipe),
            "pipe_base" | "pipebase" => Some(Self::PipeBase),
            _ => None,
        }
    }
    pub const ALL: [Self; 4] = [Self::Base, Self::Tr, Self::Pipe, Self::PipeBase];
    pub fn name(self) -> &'static str {
        match self {
            Self::Base => "base",
            Self::Tr => "tr",
            Self::Pipe => "pipe",
            Self::PipeBase => "pipe_base",
        }
    }
}

/// `rmsnorm` / `fused_add_rmsnorm` 的线程布局（`norm.cu` 的 `layout` 参数）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RmsnormLayout {
    /// 一 block（256 线程 × VPT 向量）一行（B4 第 1 轮版）。
    Block256,
    /// 一 warp 一行、4 warp/block：block 数 = ceil(行/4)，每 lane 顺序持 hidden/256 个 16 B 向量。
    WarpPerRow,
    /// 一 block 1024 线程一行：hidden ≤ 8192 时每线程 ≤ 1 向量，归约两级。
    Block1024,
}

impl RmsnormLayout {
    pub fn code(self) -> i32 {
        match self {
            Self::Block256 => 0,
            Self::WarpPerRow => 1,
            Self::Block1024 => 2,
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "block256" | "block" => Some(Self::Block256),
            "warp" | "warp_per_row" => Some(Self::WarpPerRow),
            "block1024" | "big" => Some(Self::Block1024),
            _ => None,
        }
    }
    pub const ALL: [Self; 3] = [Self::Block256, Self::WarpPerRow, Self::Block1024];
    pub fn name(self) -> &'static str {
        match self {
            Self::Block256 => "block256",
            Self::WarpPerRow => "warp",
            Self::Block1024 => "block1024",
        }
    }
}

/// decode 小 M GEMM 的自调优参数。全部是"怎么搜"的数值，不是某个 kernel 的最终配置。
#[derive(Clone, Debug)]
pub struct GemmTuneParams {
    /// 参与自调优的 M 桶（与 decode CUDA graph 的 batch 桶一致）。
    pub m_buckets: Vec<usize>,
    /// 走自写小 M kernel 的 M 上限（kernel 的 M_TILE 最大 32；第 9 轮从 8 提到 32：M≥16 时让它与 Lt 的 splitK+reduce 比）。
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
            small_m_max: 32,
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

    /// mega kernel 结构参数：Sm100（B300，228 KB smem）用 4 组 · 3 级流水；sm8x（99 KB）只放得下 2 组 · 2 级。
    /// 数值来源：docs/loops/mega-sm100.md 记录表。
    pub fn mega_params(&self) -> MegaParams {
        let mut p = match self.arch {
            Arch::Sm100 => MegaParams { groups: 4, unroll: 4, stages: 3, ctas_per_sm: 1, gdn_waves: 2, gdn_bufs: 2 },
            Arch::Sm8x | Arch::Sm90 => MegaParams { groups: 2, unroll: 4, stages: 2, ctas_per_sm: 1, gdn_waves: 2, gdn_bufs: 1 },
        };
        if let Ok(s) = std::env::var("MEGA_PARAMS") {
            for kv in s.split(',') {
                if let Some((k, v)) = kv.split_once('=') {
                    if let Ok(v) = v.trim().parse::<usize>() {
                        match k.trim() {
                            "groups" => p.groups = v,
                            "unroll" => p.unroll = v,
                            "stages" => p.stages = v,
                            "ctas" => p.ctas_per_sm = v,
                            "gdn_waves" => p.gdn_waves = v,
                            "gdn_bufs" => p.gdn_bufs = v,
                            _ => {}
                        }
                    }
                }
            }
        }
        p
    }

    /// `gdn_recurrent` decode 寄存器版 kernel 的默认变体（第 7 轮）。3090 实测见 `docs/loops/qwen35-perf.md` 第 7 行；
    /// Sm100 的默认待 B300 实测后定（先与 Sm8x 一致，`GDN_DECODE_VARIANT=` 覆盖对比）。
    pub fn gdn_decode_variant(&self) -> GdnDecodeVariant {
        match self.arch {
            Arch::Sm8x | Arch::Sm90 | Arch::Sm100 => GdnDecodeVariant::Tr,
        }
    }

    /// `rmsnorm` / `fused_add_rmsnorm` 的线程布局：按 (行数, hidden) 选。规则由 3090 `rmsnorm_bench`（CUDA graph 内
    /// 净耗时）定，见 `docs/loops/qwen35-perf.md` 第 7 行：hidden ≤ 1024 时 128 线程 × 1 向量（Block256 档内）最快，
    /// warp-per-row 慢 1.8×（一 lane 顺序 4 次 16 B 读）；hidden ∈ (1024, 8192] 时 1024 线程 × 1 向量比 256 × VPT 快 5–8%
    /// （每线程只等一次访存）。行数目前不影响选择；B300 待实测后按 arch 分叉。
    /// hidden > 8192 或非 8 的倍数时 kernel 侧回标量版，这里的返回值无效。
    pub fn rmsnorm_layout(&self, rows: usize, hidden: usize) -> RmsnormLayout {
        const RMSNORM_ONE_VEC_MAX_HIDDEN: usize = 1024;
        const RMSNORM_BLOCK1024_MAX_HIDDEN: usize = 8192;
        let _ = rows;
        if hidden > RMSNORM_ONE_VEC_MAX_HIDDEN && hidden <= RMSNORM_BLOCK1024_MAX_HIDDEN {
            RmsnormLayout::Block1024
        } else {
            RmsnormLayout::Block256
        }
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
        Self::gdn_decode_bv_for(Arch::Sm8x, sm_count, base_blocks, k_dim, v_dim)
    }

    /// 按卡型：Sm100（B300 实测，`docs/loops/qwen35-perf.md` 7c）BV=64 的寄存器 / 占用组合比 32 慢 6%（B=32：34.1 vs 32.1 µs），
    /// 候选上限收到 32；Sm8x 两档差 < 1%，保持原规则（最大的仍能铺 ≥ 2 波的 BV）。
    pub fn gdn_decode_bv_for(arch: Arch, sm_count: usize, base_blocks: usize, k_dim: usize, v_dim: usize) -> usize {
        const GDN_DECODE_BV_CANDIDATES: [usize; 3] = [64, 32, 16];
        const GDN_DECODE_WAVES: usize = 2;
        if k_dim != 64 && k_dim != 128 {
            return 0;
        }
        let mut fallback = 0;
        for &bv in &GDN_DECODE_BV_CANDIDATES {
            if matches!(arch, Arch::Sm100) && bv > 32 {
                continue;
            }
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

    /// chunk prefill `ho`（tensor core 版）的 V 分片 BV：取能整除 V 的最大档。r9 第 1 轮装载重写后 BV=32 在 3090 与 B300 的五个
    /// 形状上都快于 16（B300 1×1024 498 vs 694、1×8192 660 vs 946 µs；装载 q / w / kg 的重读少一半 > block 数翻倍），
    /// 不再按"铺满 N 波"选小档。K ∉ {64,128,256} 或 V 不被任何档整除返回 0（调用方回老 kernel）。
    pub fn gdn_chunk_ho_bv(sm_count: usize, base_blocks: usize, k_dim: usize, v_dim: usize) -> usize {
        const GDN_CHUNK_HO_BV_CANDIDATES: [usize; 2] = [32, 16];
        let _ = (sm_count, base_blocks);
        if !matches!(k_dim, 64 | 128 | 256) {
            return 0;
        }
        GDN_CHUNK_HO_BV_CANDIDATES.iter().copied().find(|&bv| v_dim % bv == 0).unwrap_or(0)
    }

    /// chunk prefill `ho` 是否寄存器预取下一 chunk（r9 第 1 轮）。3090 实测：装载改成"每线程固定向量数进寄存器再转 bf16 写 smem"
    /// 本身就赢（bv32 1×8192 2961 → 2043 µs）；再把下一 chunk 的读提前到本 chunk 计算前（PF=1）反而慢 20–70%（多 ~130 个
    /// 寄存器 → 255 上限、occupancy 掉）。所以 Sm8x 默认不预取。
    /// **Sm100 开**（r10 第 2 轮，B300 GPU 7 实测）：r10 第 1 轮中间量 bf16 后预取寄存器减半，PF=1 由"全形状退"翻成
    /// 1×8192 bv32 690 → 520 µs（−25%）、8×1024 405 → 379；3090 同代码仍退（bv32pf 1798 vs 1686），按卡分叉。
    pub fn gdn_chunk_ho_prefetch(ho_bv: usize) -> bool {
        Self::gdn_chunk_ho_prefetch_for(Arch::Sm8x, ho_bv)
    }

    /// chunk prefill `wy`（tensor core 版）的 block warp 数（r10 第 3 轮）。B300 ncu（8×1024）：每 scheduler 可发射 warp 0.53、
    /// 64% 周期无可发射、DRAM 6%、smem / regs 都只限到 5 block/SM —— 延迟型，加 warp 掩 smem 依赖链；Sm100 取 8。
    /// Sm8x 待实测（3090 smem 限 2 block/SM，8 warp 可能反而降占用），先保持 4。
    pub fn gdn_chunk_wy_warps_for(arch: Arch) -> usize {
        match arch {
            Arch::Sm100 => 8,
            Arch::Sm8x | Arch::Sm90 => 4,
        }
    }

    pub fn gdn_chunk_ho_prefetch_for(arch: Arch, ho_bv: usize) -> bool {
        if ho_bv == 0 {
            return false;
        }
        matches!(arch, Arch::Sm100)
    }

    /// chunk prefill `ho` 是否拆两段（h-pass + o-pass，r9 第 2 轮）。融合版的并行度 = num_seqs · HV · V/BV 个 block，
    /// 每 block 串行走该序列全部 chunk；两段版把 o 的算量搬到 (chunk, head) 全并行的 o-pass，h-pass 每 chunk 少两个乘法，
    /// 代价是 h16 / v16 一写一读。平均每序列 chunk 数 ≥ `GDN_CHUNK_HO_SPLIT_MIN_CHUNKS`（串行链够长、o-pass 够多 block 铺满 SM）
    /// 才值得拆；短序列大批次（32×128）融合版已经并行度足够、拆了只多搬数据。
    /// 3090 实测（r9 第 2 轮）两段版在五个形状上都比融合版慢 10–30%（h16 一写一读 201 MB @ 1×8192，3090 带宽吃不下），
    /// 所以 3090 一律 false；B300 带宽 8× 待复测后再按 `sm_count` / 带宽分档开。候选门限：平均每序列 chunk ≥ 4 且
    /// num_chunks · HV ≥ sm_count（o-pass 铺满）。
    pub fn gdn_chunk_ho_split(sm_count: usize, num_seqs: usize, num_chunks: usize, heads_v: usize) -> bool {
        const GDN_CHUNK_HO_SPLIT_MIN_CHUNKS: usize = 4;
        const GDN_CHUNK_HO_SPLIT_ENABLED: bool = false;  // 待 B300 数字
        if !GDN_CHUNK_HO_SPLIT_ENABLED || num_seqs == 0 {
            return false;
        }
        num_chunks >= GDN_CHUNK_HO_SPLIT_MIN_CHUNKS * num_seqs && num_chunks * heads_v >= sm_count
    }

    /// 两段版 o-pass 的 V 分片 VT（r9 第 4 轮）：smem = h 分片 [VT][K] + q [BT][K] + Aq + v' [BT][VT]。3090 实测 VT=128
    /// （78 KB，1 block/SM，8 warp）比 VT=64（49 KB，2 block/SM）快 10%（q / Aq 只装一次）。取能整除 V 的最大档；都不整除返回 0。
    pub fn gdn_chunk_opass_vt(v_dim: usize) -> usize {
        const GDN_CHUNK_OPASS_VT_CANDIDATES: [usize; 4] = [128, 64, 32, 16];
        GDN_CHUNK_OPASS_VT_CANDIDATES.iter().copied().find(|&vt| v_dim % vt == 0).unwrap_or(0)
    }

    /// conv1d prefill 按 token 分块的块行数：block 数 = `ceil(tokens / rows) · ceil(dim / 2048)`（256 线程 × 8 通道），
    /// 取最大的仍能铺 ≥ `GDN_CONV_WAVES` 波 SM 的档（行多 → halo 重读占比小、块内串行链长；行少 → 反之），
    /// 都铺不满取最小档；下限 width-1（kernel 要求块 ≥ halo）。
    /// conv1d prefill 按 token 分块的块行数：block 数 = `ceil(tokens / rows) · ceil(dim / 2048)`（256 线程 × 8 通道），
    /// 取最大的仍能铺满每 SM `GDN_CONV_BLOCKS_PER_SM[arch]` 个常驻 block 的档；都铺不满取最小档；下限 width-1。
    /// 按卡分档的依据（r9 第 5–6 轮）：3090 rows 32–256 平、rows 8/16 慢（每块固定开销占比大），2 块/SM 即够；
    /// B300（148 SM，HBM 延迟型）8×1024 rows16 220 / rows32 222 / rows128 310 µs——老规则选 128 慢 40%，要 ≥ 8 块/SM 才喂饱。
    pub fn gdn_conv_rows(sm_count: usize, tokens: usize, dim: usize, width: usize) -> usize {
        Self::gdn_conv_rows_for(Arch::Sm8x, sm_count, tokens, dim, width)
    }

    pub fn gdn_conv_rows_for(arch: Arch, sm_count: usize, tokens: usize, dim: usize, width: usize) -> usize {
        const GDN_CONV_ROWS_CANDIDATES: [usize; 6] = [256, 128, 64, 32, 16, 8];
        let blocks_per_sm: usize = match arch {
            Arch::Sm100 => 8,
            Arch::Sm90 | Arch::Sm8x => 2,
        };
        let dim_blocks = dim.div_ceil(256 * 8).max(1);
        for &r in &GDN_CONV_ROWS_CANDIDATES {
            // 3090 实测 rows < 32 只会更慢：Sm8x 不下探到 16 / 8
            if matches!(arch, Arch::Sm8x | Arch::Sm90) && r < 32 {
                break;
            }
            if tokens.div_ceil(r) * dim_blocks >= blocks_per_sm * sm_count {
                return r.max(width - 1);
            }
        }
        let floor = if matches!(arch, Arch::Sm100) { 8 } else { 32 };
        floor.max(width - 1)
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

/// Qwen3.5 GDN 层 decode 单 persistent kernel（mega_sm100.cu）的结构参数。都是"按卡给"的数值，kernel 内不写。
/// `MEGA_PARAMS=groups=4,unroll=4,stages=3,ctas=1,gdn_waves=2` 覆盖（实验用）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MegaParams {
    /// 每 warp 的 8 行 N 组数（GEMV tile = 8 warp · groups · 8 行）。
    pub groups: usize,
    /// 每流水 stage 的 32 列 K 块数。
    pub unroll: usize,
    /// cp.async 流水深度（smem = stages · (tile + A 块)）。
    pub stages: usize,
    /// 每 SM 常驻 CTA 数（1 / 2，决定 launch_bounds 寄存器上限；实际再受 occupancy 查询约束）。
    pub ctas_per_sm: usize,
    /// GDN 阶段任务数至少铺几波常驻 CTA（决定每任务连续处理的 v 头数 gdn_hv）。
    pub gdn_waves: usize,
    /// GDN 状态 tile 的 smem 缓冲数：2 = 跨 tile 双缓冲预取（64 KB × 2，需 ≥ 128 KB smem），1 = 单缓冲。
    pub gdn_bufs: usize,
}

impl MegaParams {
    /// GEMV tile 的 N 行数（与 kernel 的 kWarps·G·8 同式；kernel 侧复核）。
    pub fn tile_n(&self) -> usize {
        8 * self.groups * 8
    }
    /// split-K 份数：任务数 = tiles · splits 尽量填满一波常驻 CTA（floor，避免第二波只有零头），
    /// 且每 split 至少一个流水 stage（k / (32·unroll) 个 stage）。
    /// GDN 任务 = (seq, 连续 gdn_hv 个 v 头)：取能整除 heads_v 的最大 d，使 batch·heads_v/d ≥ gdn_waves·grid。
    pub fn gdn_hv_for(&self, sm_count: usize, batch: usize, heads_v: usize) -> usize {
        let grid = sm_count * self.ctas_per_sm.max(1);
        let mut best = 1;
        for d in 1..=heads_v {
            if heads_v % d == 0 && batch * (heads_v / d) >= self.gdn_waves.max(1) * grid {
                best = d;
            }
        }
        best
    }

    pub fn splits_for(&self, sm_count: usize, n: usize, k: usize) -> usize {
        let grid = sm_count * self.ctas_per_sm.max(1);
        let tiles = n.div_ceil(self.tile_n());
        let total = k / (32 * self.unroll);
        (grid / tiles.max(1)).clamp(1, total.max(1))
    }
}

#[cfg(test)]
mod tests {
    use super::{Arch, ArchProfile};

    #[test]
    fn gdn_decode_bv_tiers() {
        // 27B 形状 HV=48 K=V=128：B=1 铺不满取小档，B≥4 都取 64；三种卡的 SM 数
        for (sm, b1) in [(82usize, 32usize), (128, 16), (148, 16)] {
            assert_eq!(ArchProfile::gdn_decode_bv(sm, 48, 128, 128), b1, "sm={sm} B=1");
            assert_eq!(ArchProfile::gdn_decode_bv(sm, 4 * 48, 128, 128), 64, "sm={sm} B=4");
            assert_eq!(ArchProfile::gdn_decode_bv(sm, 32 * 48, 128, 128), 64, "sm={sm} B=32");
            // Sm100：上限 32（B300 实测）
            assert_eq!(ArchProfile::gdn_decode_bv_for(Arch::Sm100, sm, 32 * 48, 128, 128), 32, "sm100 sm={sm} B=32");
            assert_eq!(ArchProfile::gdn_decode_bv_for(Arch::Sm100, sm, 8 * 48, 128, 128), 32, "sm100 sm={sm} B=8");
        }
        // K 不支持 → 0；V 只被 16/32 整除时不选 64
        assert_eq!(ArchProfile::gdn_decode_bv(82, 48, 96, 128), 0);
        assert_eq!(ArchProfile::gdn_decode_bv(82, 4 * 48, 64, 32), 32);
    }

    #[test]
    fn rmsnorm_layout_tiers() {
        use super::RmsnormLayout;
        let p = ArchProfile::for_cc((8, 6), 82);
        assert_eq!(p.rmsnorm_layout(1, 1024), RmsnormLayout::Block256);
        assert_eq!(p.rmsnorm_layout(32, 5120), RmsnormLayout::Block1024);
        assert_eq!(p.rmsnorm_layout(32, 8192), RmsnormLayout::Block1024);
        assert_eq!(p.rmsnorm_layout(32, 16384), RmsnormLayout::Block256);
    }

    #[test]
    fn gdn_chunk_ho_bv_tiers() {
        // r9：一律最大可整除档（B300 单序列也是 32）
        assert_eq!(ArchProfile::gdn_chunk_ho_bv(82, 48, 128, 128), 32);
        assert_eq!(ArchProfile::gdn_chunk_ho_bv(148, 48, 128, 128), 32);
        assert_eq!(ArchProfile::gdn_chunk_ho_bv(148, 8 * 48, 128, 128), 32);
        assert_eq!(ArchProfile::gdn_chunk_ho_bv(82, 48, 96, 128), 0);
        assert_eq!(ArchProfile::gdn_chunk_ho_bv(82, 4, 64, 16), 16);
        assert!(!ArchProfile::gdn_chunk_ho_prefetch(16));
        assert!(!ArchProfile::gdn_chunk_ho_prefetch(32));
        assert!(!ArchProfile::gdn_chunk_ho_prefetch(0));
        // 两段版默认关（3090 不赢，待 B300）；32×128（每序列 2 chunk）任何情况不拆
        assert!(!ArchProfile::gdn_chunk_ho_split(82, 1, 16, 48));
        assert!(!ArchProfile::gdn_chunk_ho_split(82, 32, 64, 48));
        assert_eq!(ArchProfile::gdn_chunk_opass_vt(128), 128);
        assert_eq!(ArchProfile::gdn_chunk_opass_vt(192), 64);
        assert_eq!(ArchProfile::gdn_chunk_opass_vt(96), 32);
        assert_eq!(ArchProfile::gdn_chunk_opass_vt(40), 0);
    }

    #[test]
    fn gdn_conv_rows_tiers() {
        // dim 10240 → 5 通道块。Sm8x（2 块/SM，不下探 <32）：3090 T=1024 → 兜底 32；T=8192：8192/64·5=640 ≥ 164 → 256? 8192/256·5=160 <164 → 128
        // Sm100（8 块/SM = 1184）：8800/32·5=1375 ≥ 1184 → 32；1024：1024/8·5=640 < 1184 → 兜底 8；T=3 → 8 ≥ width-1
        use super::Arch;
        assert_eq!(ArchProfile::gdn_conv_rows_for(Arch::Sm8x, 82, 1024, 10240, 4), 32);
        assert_eq!(ArchProfile::gdn_conv_rows_for(Arch::Sm8x, 82, 8192, 10240, 4), 128);
        assert_eq!(ArchProfile::gdn_conv_rows_for(Arch::Sm100, 148, 8800, 10240, 4), 32);
        assert_eq!(ArchProfile::gdn_conv_rows_for(Arch::Sm100, 148, 1024, 10240, 4), 8);
        assert_eq!(ArchProfile::gdn_conv_rows_for(Arch::Sm100, 148, 3, 10240, 4), 8);
    }
}
