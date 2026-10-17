//! 最小设备张量：`CudaSlice<T>` + 形状。不做广播/视图，只做显式 reshape 校验。

use anyhow::{ensure, Result};
use dinfer_kernels::cudarc::driver::{CudaSlice, DeviceRepr, ValidAsZeroBits};
use dinfer_kernels::Stream;
use half::bf16;

pub struct Tensor<T> {
    pub data: CudaSlice<T>,
    pub shape: Vec<usize>,
}

impl<T: DeviceRepr> Tensor<T> {
    /// 不清零分配（见 `Stream::alloc_uninit` 的安全前提）。
    pub fn uninit(stream: &Stream, shape: &[usize]) -> Result<Self> {
        let n: usize = shape.iter().product();
        Ok(Self { data: stream.alloc_uninit::<T>(n)?, shape: shape.to_vec() })
    }
}

impl<T: DeviceRepr + ValidAsZeroBits> Tensor<T> {
    pub fn zeros(stream: &Stream, shape: &[usize]) -> Result<Self> {
        let n: usize = shape.iter().product();
        let t0 = std::time::Instant::now();
        let data = stream.alloc_zeros::<T>(n)?;
        ZEROS_STATS.with(|c| {
            let mut c = c.borrow_mut();
            c.0 += t0.elapsed().as_micros() as u64;
            c.1 += (n * std::mem::size_of::<T>()) as u64;
            c.2 += 1;
        });
        Ok(Self { data, shape: shape.to_vec() })
    }
}

impl<T: DeviceRepr> Tensor<T> {
    pub fn from_vec(stream: &Stream, v: Vec<T>, shape: &[usize]) -> Result<Self> {
        let n: usize = shape.iter().product();
        ensure!(v.len() == n, "Tensor::from_vec: {} elements vs shape {:?}", v.len(), shape);
        Ok(Self { data: stream.htod_copy(v)?, shape: shape.to_vec() })
    }
    #[inline]
    pub fn numel(&self) -> usize {
        self.data.len()
    }
    #[inline]
    pub fn dim(&self, i: usize) -> usize {
        self.shape[i]
    }
    pub fn to_vec(&self, stream: &Stream) -> Result<Vec<T>> {
        stream.dtoh_sync_copy(&self.data)
    }
    /// 只改形状解释，不动数据。
    pub fn reshape(&mut self, shape: &[usize]) -> Result<()> {
        ensure!(shape.iter().product::<usize>() == self.numel(), "reshape {:?} -> {:?}", self.shape, shape);
        self.shape = shape.to_vec();
        Ok(())
    }
}

impl Tensor<bf16> {
    pub fn from_f32(stream: &Stream, v: &[f32], shape: &[usize]) -> Result<Self> {
        Self::from_vec(stream, v.iter().map(|&x| bf16::from_f32(x)).collect(), shape)
    }
    pub fn to_vec_f32(&self, stream: &Stream) -> Result<Vec<f32>> {
        Ok(self.to_vec(stream)?.into_iter().map(|x| x.to_f32()).collect())
    }
}

thread_local! {
    /// (累计 µs, 累计字节, 次数)：`Tensor::zeros` 的显存分配 + 清零耗时（启动分段日志用）。
    static ZEROS_STATS: std::cell::RefCell<(u64, u64, u64)> = const { std::cell::RefCell::new((0, 0, 0)) };
}
/// 读出并清零 `Tensor::zeros` 的累计统计：(µs, bytes, count)。
pub fn take_zeros_stats() -> (u64, u64, u64) {
    ZEROS_STATS.with(|c| std::mem::take(&mut *c.borrow_mut()))
}
