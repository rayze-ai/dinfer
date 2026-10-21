//! Kaon-V3（Gemma4 MoE，MIXED 量化）权重加载。
//!
//! 三种落盘格式（2026-10-21 实测，见 `docs/loops/kaon-v3.md`）：
//! - **NVFP4**（128 专家）：`weight [N, K/2] U8` + `weight_scale [N, K/16] F8_E4M3`
//!   + `weight_scale_2 [] F32`（全局）+ `input_scale [] F32`。压缩态常驻，kernel 内解包。
//! - **FP8**（attn / dense MLP）：`weight [N, K] F8_E4M3` + `weight_scale [] F32`（**per-tensor**，
//!   与 `crates/layers` 里 block-quant 的 `_scale_inv` 约定不同，故不复用 `load_bf16`）。
//! - **BF16**（embed / router / layer_scalar / norms）：直读。
//!
//! 专家按 `[E, N, K]` 拼成连续大张量（`E*N` 行），decode 时一次 launch 覆盖全部选中专家
//! （实测比逐专家 launch 省一半以上时间，见 loop 文档「合并 launch」）。

use anyhow::{bail, ensure, Context, Result};
use dinfer_kernels::cudarc::driver::CudaSlice;
use dinfer_kernels::Stream;
use dinfer_layers::WeightSource;
use std::collections::HashMap;

/// 文本层前缀：`model.language_model.layers.{l}.`
pub const TEXT_PREFIX: &str = "model.language_model.";
pub const EMBED: &str = "model.language_model.embed_tokens.weight";
pub const NORM: &str = "model.language_model.norm.weight";

/// NVFP4 块大小（每 16 个 k 一个 e4m3 scale）。
pub const NVFP4_BLOCK: usize = 16;

/// 一张量化权重在显存里的形态。
pub enum QWeight {
    /// NVFP4：packed `[N, K/2]`、block scale `[N, K/16]`、全局 scale。
    NvFp4 { packed: CudaSlice<u8>, scale: CudaSlice<u8>, global: f32, n: usize, k: usize },
    /// FP8 e4m3 per-tensor：`weight [N,K]` + 标量 scale。
    Fp8 { w: CudaSlice<u8>, scale: f32, n: usize, k: usize },
    /// BF16。
    Bf16 { w: CudaSlice<u16>, n: usize, k: usize },
}

impl QWeight {
    pub fn shape(&self) -> (usize, usize) {
        match self {
            QWeight::NvFp4 { n, k, .. } | QWeight::Fp8 { n, k, .. } | QWeight::Bf16 { n, k, .. } => {
                (*n, *k)
            }
        }
    }
}

fn read_f32_scalar(src: &WeightSource, name: &str) -> Result<f32> {
    let (info, b) = src.bytes(name).with_context(|| format!("缺少标量 {name}"))?;
    ensure!(info.dtype == "F32" && b.len() == 4, "{name}: 期望单个 F32，实际 {} / {} B",
            info.dtype, b.len());
    Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn read_u8<'a>(src: &'a WeightSource, name: &str, want: usize) -> Result<&'a [u8]> {
    let (info, b) = src.bytes(name)?;
    ensure!(b.len() == want, "{name}: 期望 {want} B，实际 {}", b.len());
    ensure!(
        info.dtype == "U8" || info.dtype == "F8_E4M3",
        "{name}: 期望 U8/F8_E4M3，实际 {}",
        info.dtype
    );
    Ok(b)
}

/// 上传到设备。
fn h2d_u8(stream: &Stream, bytes: &[u8]) -> Result<CudaSlice<u8>> {
    let mut d = stream.alloc_uninit::<u8>(bytes.len())?;
    stream.inner_arc().memcpy_htod(bytes, &mut d)?;
    Ok(d)
}

/// 读一张线性层权重（按落盘格式自动分派）。返回的逻辑形状是 `(n, k)`，
/// `k` 是**元素**个数（NVFP4 的 `weight` 落盘列数是 `k/2`）。
pub fn load_linear(src: &WeightSource, stream: &Stream, name: &str) -> Result<QWeight> {
    let (info, _) = src.bytes(name)?;
    let shape = info.shape.clone();
    ensure!(shape.len() == 2, "{name}: 期望 2 维，实际 {shape:?}");
    let n = shape[0];
    let cols = shape[1];

    // NVFP4：有 2 维的 `weight_scale`（[N, K/16]）
    if let Some(sinfo) = src.info(&format!("{name}_scale")).cloned() {
        if sinfo.shape.len() == 2 {
            ensure!(sinfo.dtype == "F8_E4M3", "{name}_scale: 期望 F8_E4M3，实际 {}", sinfo.dtype);
            let k = cols * 2; // 每字节两个 fp4
            ensure!(
                sinfo.shape == vec![n, k / NVFP4_BLOCK],
                "{name}: scale {:?} 与 weight {n}x{cols}(K={k}) 不匹配（块大小应为 {NVFP4_BLOCK}）",
                sinfo.shape
            );
            let packed = read_u8(src, name, n * cols)?;
            let scale = read_u8(src, &format!("{name}_scale"), n * k / NVFP4_BLOCK)?;
            let global = read_f32_scalar(src, &format!("{name}_scale_2"))
                .with_context(|| format!("{name}: NVFP4 缺全局 scale_2"))?;
            return Ok(QWeight::NvFp4 {
                packed: h2d_u8(stream, packed)?,
                scale: h2d_u8(stream, scale)?,
                global,
                n,
                k,
            });
        }
    }
    let k = cols;
    match info.dtype.as_str() {
        "F8_E4M3" => {
            let w = read_u8(src, name, n * k)?;
            let scale = read_f32_scalar(src, &format!("{name}_scale"))
                .with_context(|| format!("{name}: FP8 per-tensor 缺标量 weight_scale"))?;
            Ok(QWeight::Fp8 { w: h2d_u8(stream, w)?, scale, n, k })
        }
        "BF16" => {
            let (_, b) = src.bytes(name)?;
            ensure!(b.len() == n * k * 2, "{name}: BF16 字节数不符");
            let mut d = stream.alloc_uninit::<u16>(n * k)?;
            stream.inner_arc().memcpy_htod(bytemuck_cast(b), &mut d)?;
            Ok(QWeight::Bf16 { w: d, n, k })
        }
        other => bail!("{name}: 不支持的 dtype {other}"),
    }
}

fn bytemuck_cast(b: &[u8]) -> &[u16] {
    // SAFETY: BF16 落盘就是 u16 位模式；长度已由调用方按 n*k*2 校验。
    unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u16, b.len() / 2) }
}

/// 把一个专家的三个矩阵读进预分配大缓冲的指定偏移（`[E, N, K]` 布局）。
pub struct ExpertSink<'a> {
    pub gate: &'a mut CudaSlice<u8>,
    pub up: &'a mut CudaSlice<u8>,
    pub down: &'a mut CudaSlice<u8>,
    pub gate_s: &'a mut CudaSlice<u8>,
    pub up_s: &'a mut CudaSlice<u8>,
    pub down_s: &'a mut CudaSlice<u8>,
}

fn copy_into(stream: &Stream, bytes: &[u8], dst: &mut CudaSlice<u8>, off: usize) -> Result<()> {
    ensure!(off + bytes.len() <= dst.len(), "专家权重写入越界");
    let mut view = dst.slice_mut(off..off + bytes.len());
    stream.inner_arc().memcpy_htod(bytes, &mut view)?;
    Ok(())
}

/// 读一个专家的 gate/up/down 到 sink 的第 `e` 个槽位。
pub fn load_expert_into(
    src: &WeightSource,
    stream: &Stream,
    prefix: &str,
    e: usize,
    inter: usize,
    hidden: usize,
    sink: &mut ExpertSink<'_>,
) -> Result<()> {
    // gate/up: [inter, hidden]；down: [hidden, inter]
    let (pk_gu, sc_gu) = (inter * hidden / 2, inter * hidden / NVFP4_BLOCK);
    let (pk_d, sc_d) = (hidden * inter / 2, hidden * inter / NVFP4_BLOCK);
    for (name, dst, dst_s, npk, nsc) in [
        ("gate_proj", &mut *sink.gate, &mut *sink.gate_s, pk_gu, sc_gu),
        ("up_proj", &mut *sink.up, &mut *sink.up_s, pk_gu, sc_gu),
        ("down_proj", &mut *sink.down, &mut *sink.down_s, pk_d, sc_d),
    ] {
        let w = format!("{prefix}.{name}.weight");
        let s = format!("{prefix}.{name}.weight_scale");
        let packed = read_u8(src, &w, npk)?;
        let scale = read_u8(src, &s, nsc)?;
        copy_into(stream, packed, dst, e * npk)?;
        copy_into(stream, scale, dst_s, e * nsc)?;
    }
    Ok(())
}

/// 所有层的路由表（`layers.{l}.router.*`）。
#[derive(Debug, Default)]
pub struct RouterNames(HashMap<usize, String>);

impl RouterNames {
    pub fn insert(&mut self, l: usize, prefix: String) {
        self.0.insert(l, prefix);
    }
    pub fn get(&self, l: usize) -> Option<&str> {
        self.0.get(&l).map(|s| s.as_str())
    }
}

pub fn layer_prefix(l: usize) -> String {
    format!("{TEXT_PREFIX}layers.{l}")
}
