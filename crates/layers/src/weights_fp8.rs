//! FP8 e4m3（`float8_e4m3fn`）块量化权重的 CPU 侧解量化（P4-d 参考版）。
//!
//! GLM-5.3 权重格式（`quantization_config`：`fmt e4m3`、`weight_block_size [128,128]`）：
//! `X.weight` F8_E4M3 `[N,K]` + `X.weight_scale_inv` F32 `[⌈N/128⌉, ⌈K/128⌉]`，
//! `W[n,k] = f32(w8[n,k]) * scale_inv[n/128][k/128]`（与 vLLM `fp8_utils.py` 的 upcast 参考一致）。
//! sm86 无 FP8 张量核 → 加载期解量化到 bf16，`Linear` 层无感；真 FP8 GEMM 留 P10（4090 Pod）。

/// e4m3fn → f32：1 符号 / 4 指数（bias 7）/ 3 尾数；无 inf；`0x7f` / `0xff` = NaN；次正规 = m · 2^-9。
#[inline]
pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 3) & 0x0f) as i32;
    let man = (b & 0x07) as f32;
    if exp == 0x0f && (b & 0x07) == 0x07 {
        return f32::NAN;
    }
    let v = if exp == 0 { man * (1.0 / 512.0) } else { (1.0 + man / 8.0) * 2f32.powi(exp - 7) };
    sign * v
}

/// `[n, k]` 的块量化解码：`out[n][k] = f32(w8[n][k]) * scale_inv[n/bn][k/bk]`。
pub fn dequant_block(
    w8: &[u8],
    shape: &[usize],
    scale_inv: &[f32],
    scale_shape: &[usize],
    block: (usize, usize),
) -> Vec<f32> {
    assert_eq!(shape.len(), 2, "fp8 block dequant expects 2-D weight, got {shape:?}");
    let (n, k) = (shape[0], shape[1]);
    let (bn, bk) = block;
    let (sn, sk) = (n.div_ceil(bn), k.div_ceil(bk));
    assert_eq!(
        scale_shape,
        &[sn, sk],
        "scale_inv shape {scale_shape:?} != expected [{sn}, {sk}] for weight {shape:?}"
    );
    assert_eq!(w8.len(), n * k);
    let mut out = vec![0f32; n * k];
    for i in 0..n {
        let srow = &scale_inv[(i / bn) * sk..(i / bn + 1) * sk];
        let (wrow, orow) = (&w8[i * k..(i + 1) * k], &mut out[i * k..(i + 1) * k]);
        for j in 0..k {
            orow[j] = e4m3_to_f32(wrow[j]) * srow[j / bk];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_table() {
        assert_eq!(e4m3_to_f32(0x00), 0.0);
        assert_eq!(e4m3_to_f32(0x80), -0.0);
        assert_eq!(e4m3_to_f32(0x38), 1.0); // exp 7, man 0
        assert_eq!(e4m3_to_f32(0x3c), 1.5);
        assert_eq!(e4m3_to_f32(0xb8), -1.0);
        assert_eq!(e4m3_to_f32(0x7e), 448.0); // max finite
        assert_eq!(e4m3_to_f32(0x01), 1.0 / 512.0); // min subnormal 2^-9
        assert_eq!(e4m3_to_f32(0x08), 1.0 / 64.0); // min normal 2^-6
        assert!(e4m3_to_f32(0x7f).is_nan() && e4m3_to_f32(0xff).is_nan());
    }

    #[test]
    fn block_dequant_ragged() {
        // n=3, k=5, block (2,4) → scale [2,2]
        let w8 = vec![0x38u8; 15]; // 全 1.0
        let s = vec![1.0, 2.0, 3.0, 4.0];
        let out = dequant_block(&w8, &[3, 5], &s, &[2, 2], (2, 4));
        assert_eq!(out[0..4], [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(out[4], 2.0);
        assert_eq!(out[10..14], [3.0, 3.0, 3.0, 3.0]);
        assert_eq!(out[14], 4.0);
    }
}
