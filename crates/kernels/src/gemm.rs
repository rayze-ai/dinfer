//! bf16 GEMM：cuBLASLt 封装。`C[M,N] = A[M,K] · W[N,K]^T (+ bias[N])`，fp32 累加。
//!
//! 权重按 PyTorch `Linear` 的 `[out_features, in_features]`（即 `[N,K]`）行主序存，
//! 激活 `[M,K]` 行主序。cuBLAS 是列主序，等价于算 `C^T[N,M] = W[N,K]·A^T[K,M]`：
//! 以列主序视角，A 行主序 `[M,K]` 就是列主序 `[K,M]`（lda=K），W 行主序 `[N,K]` 就是列主序 `[K,N]`（ldb=K），
//! 要 `C^T = W^T_colmajor · A`，所以 transa=T（对 W），transb=N（对 A），输出列主序 `[N,M]`（ldc=N）即行主序 `[M,N]`。

use crate::device::Stream;
use anyhow::{ensure, Context, Result};
use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig};
use cudarc::driver::CudaSlice;
use half::bf16;

pub struct Gemm {
    lt: CudaBlasLT,
}

impl Gemm {
    pub fn new(stream: &Stream) -> Result<Self> {
        let lt = CudaBlasLT::new(stream.inner_arc()).context("cublasLt create")?;
        Ok(Self { lt })
    }

    /// out[M,N] = a[M,K] · w[N,K]^T，可选 bias[N]。
    #[allow(clippy::too_many_arguments)]
    pub fn forward_bf16(
        &self,
        a: &CudaSlice<bf16>,
        w: &CudaSlice<bf16>,
        bias: Option<&CudaSlice<bf16>>,
        out: &mut CudaSlice<bf16>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        // a / out 允许是按 max_tokens 预分配的更大缓冲（只用前 m 行）；权重必须精确。
        ensure!(a.len() >= m * k, "gemm: a.len {} < m*k {}", a.len(), m * k);
        ensure!(w.len() == n * k, "gemm: w.len {} != n*k {}", w.len(), n * k);
        ensure!(out.len() >= m * n, "gemm: out.len {} < m*n {}", out.len(), m * n);
        if let Some(b) = bias {
            ensure!(b.len() == n, "gemm: bias.len {} != n {n}", b.len());
        }
        if m == 0 {
            return Ok(());
        }
        let cfg = MatmulConfig {
            transa: true,
            transb: false,
            transc: false,
            m: n as u64,
            n: m as u64,
            k: k as u64,
            alpha: 1.0,
            lda: k as i64,
            ldb: k as i64,
            beta: 0.0,
            ldc: n as i64,
            stride_a: None,
            stride_b: None,
            stride_c: None,
            stride_bias: None,
            batch_size: None,
        };
        // SAFETY: 维度已校验；cudarc 内部按 cfg 建 layout，指针来自有效 CudaSlice。
        unsafe { self.lt.matmul(cfg, w, a, out, bias, None) }.context("cublasLt matmul bf16")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

    #[test]
    #[ignore = "needs a CUDA device"]
    fn gemm_matches_cpu_reference() {
        let dev = Device::new(0).unwrap();
        let s = dev.default_stream();
        let g = Gemm::new(&s).unwrap();
        let (m, n, k) = (5usize, 7usize, 64usize);
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 37) % 23) as f32 / 23.0 - 0.5).collect();
        let w: Vec<f32> = (0..n * k).map(|i| ((i * 53) % 29) as f32 / 29.0 - 0.5).collect();
        let bias: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let ab: Vec<bf16> = a.iter().map(|&v| bf16::from_f32(v)).collect();
        let wb: Vec<bf16> = w.iter().map(|&v| bf16::from_f32(v)).collect();
        let bb: Vec<bf16> = bias.iter().map(|&v| bf16::from_f32(v)).collect();
        let da = s.htod_copy(ab.clone()).unwrap();
        let dw = s.htod_copy(wb.clone()).unwrap();
        let db = s.htod_copy(bb.clone()).unwrap();
        let mut dout = s.alloc_zeros::<bf16>(m * n).unwrap();
        g.forward_bf16(&da, &dw, Some(&db), &mut dout, m, n, k).unwrap();
        let out = s.dtoh_sync_copy(&dout).unwrap();
        for i in 0..m {
            for j in 0..n {
                let mut acc = bb[j].to_f32();
                for t in 0..k {
                    acc += ab[i * k + t].to_f32() * wb[j * k + t].to_f32();
                }
                let got = out[i * n + j].to_f32();
                assert!((got - acc).abs() <= 2e-2 + 2e-2 * acc.abs(), "({i},{j}) got={got} want={acc}");
            }
        }
    }
}
