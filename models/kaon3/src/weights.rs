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
    /// `a_scale` = 激活 input_scale（vLLM a1_gscale = 1/a_scale）。
    NvFp4 { packed: CudaSlice<u8>, scale: CudaSlice<u8>, global: f32, a_scale: f32, n: usize, k: usize },
    /// FP8 e4m3 per-tensor：`weight [N,K]` + 标量 scale；`a_scale` = 静态激活 scale（`input_scale`，没有为 1.0）。
    Fp8 { w: CudaSlice<u8>, scale: f32, a_scale: f32, n: usize, k: usize },
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

pub(crate) fn read_f32_scalar(src: &WeightSource, name: &str) -> Result<f32> {
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
                a_scale: input_scale(src, name),
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
            Ok(QWeight::Fp8 { w: h2d_u8(stream, w)?, scale, a_scale: input_scale(src, name), n, k })
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

/// 读一层的 `per_expert_scale`（`[E]` BF16，每层一份）。
pub fn per_expert_scales(src: &WeightSource, layer_prefix: &str) -> Result<Vec<f32>> {
    let (shape, pes) = src.load_f32_host(&format!("{layer_prefix}.router.per_expert_scale"))?;
    ensure!(!pes.is_empty(), "{layer_prefix}: per_expert_scale 为空（形状 {shape:?}）");
    Ok(pes)
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

/// 静态激活 scale（`<层>.input_scale`）；没有（BF16 / 动态量化）时 1.0。
fn input_scale(src: &WeightSource, name: &str) -> f32 {
    let base = name.strip_suffix(".weight").unwrap_or(name);
    read_f32_scalar(src, &format!("{base}.input_scale")).unwrap_or(1.0)
}

/// vLLM 把 q/k/v、gate/up 合并成一个线性层（`QKVParallelLinear` / `MergedColumnParallelLinear`）。FP8 per-tensor
/// 时各分片 scale 不同，`requantize_with_max_scale` 把每个分片按自己的 scale 反量化（fp16）再用组内最大 scale
/// 重新量化成 e4m3，之后整组共用 max weight scale 与 max input scale。这里逐位复现，保证与基线同一份权重。
/// 返回顺序同 `names`；任何一个不是 FP8 per-tensor 就退回逐个 `load_linear`（不重量化）。
pub fn load_fp8_group(src: &WeightSource, stream: &Stream, names: &[String]) -> Result<Vec<QWeight>> {
    let all_fp8 = names.iter().all(|nm| {
        src.info(nm).is_some_and(|i| i.dtype == "F8_E4M3" && i.shape.len() == 2)
            && src.info(&format!("{nm}_scale")).is_some_and(|i| i.shape.iter().product::<usize>() <= 1)
    });
    if !all_fp8 {
        return names.iter().map(|nm| load_linear(src, stream, nm)).collect();
    }
    let mut parts = Vec::with_capacity(names.len());
    for nm in names {
        let shape = src.info(nm).context("weight info")?.shape.clone();
        let (n, k) = (shape[0], shape[1]);
        let w = read_u8(src, nm, n * k)?.to_vec();
        let s = read_f32_scalar(src, &format!("{nm}_scale"))?;
        parts.push((w, s, input_scale(src, nm), n, k));
    }
    let smax = parts.iter().map(|p| p.1).fold(f32::MIN, f32::max);
    let amax = parts.iter().map(|p| p.2).fold(f32::MIN, f32::max);
    let inv = 1.0 / smax;
    parts
        .into_iter()
        .map(|(mut w, s, _, n, k)| {
            if s != smax {
                // per_tensor_dequantize：fp8 → fp16 再乘 0 维 fp32 scale（按 fp16 算，scale 先舍入成 fp16）；
                // scaled_fp8_quant：x · (1/max_scale) 后就近偶数、饱和转 e4m3。
                let s16 = half::f16::from_f32(s).to_f32();
                for b in w.iter_mut() {
                    let dq = half::f16::from_f32(e4m3_to_f32(*b) * s16).to_f32();
                    *b = f32_to_e4m3(dq * inv);
                }
            }
            Ok(QWeight::Fp8 { w: h2d_u8(stream, &w)?, scale: smax, a_scale: amax, n, k })
        })
        .collect()
}

/// e4m3fn 字节 → f32（精确；0x7f / 0xff 是 NaN）。
pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0xf) as i32;
    let m = (b & 7) as f32;
    if e == 15 && b & 7 == 7 {
        return f32::NAN;
    }
    sign * if e == 0 { m / 8.0 * 2f32.powi(-6) } else { (1.0 + m / 8.0) * 2f32.powi(e - 7) }
}

/// f32 → e4m3fn（就近偶数、饱和到 ±448），与 `__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3)` 一致。
pub fn f32_to_e4m3(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let a = x.abs();
    if a.is_nan() {
        return 0x7f;
    }
    if a >= 448.0 {
        return sign | 0x7e;
    }
    if a < 2f32.powi(-6) {
        // 次正规 m · 2^-9；m = 8 时恰好进位成最小正规数 0x08，编码连续
        return sign | (a * 512.0).round_ties_even() as u8;
    }
    let mut e = a.log2().floor() as i32;
    let mut m = ((a / 2f32.powi(e) - 1.0) * 8.0).round_ties_even() as i32;
    if m == 8 {
        m = 0;
        e += 1;
    }
    let eb = e + 7;
    if eb > 15 || (eb == 15 && m > 6) {
        return sign | 0x7e;
    }
    sign | ((eb as u8) << 3) | m as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_codec_roundtrip_and_ties() {
        for b in 0u8..=255 {
            if b & 0x7f == 0x7f {
                continue; // NaN
            }
            assert_eq!(f32_to_e4m3(e4m3_to_f32(b)), b, "byte {b:#04x}");
        }
        assert_eq!(e4m3_to_f32(0x38), 1.0);
        assert_eq!(e4m3_to_f32(0x23), 0.171875, "次正规之上的正规数锚点");
        assert_eq!(f32_to_e4m3(1.0625), 0x38, "1.0 与 1.125 正中间 → 偶数 1.0");
        assert_eq!(f32_to_e4m3(1.1875), 0x3a, "1.125 与 1.25 正中间 → 偶数 1.25");
        assert_eq!(f32_to_e4m3(1000.0), 0x7e, "饱和到 448");
        assert_eq!(f32_to_e4m3(2f32.powi(-10)), 0x00, "最小次正规的一半 → 偶数 0");
    }
}
