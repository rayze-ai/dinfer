//! safetensors 读取（mmap，按名取张量）。自实现头解析，不依赖 safetensors crate（避免等版本）。
//!
//! 格式：8 字节小端 u64 = header 长度 N；接着 N 字节 JSON；之后是数据区。
//! JSON：`{"name": {"dtype": "BF16", "shape": [..], "data_offsets": [start, end]}, "__metadata__": {...}}`。

use anyhow::{bail, ensure, Context, Result};
use dinfer_kernels::Stream;
use half::bf16;
use memmap2::Mmap;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    /// 相对数据区起点的字节范围。
    pub start: usize,
    pub end: usize,
    pub file: usize,
}

pub struct WeightSource {
    maps: Vec<Mmap>,
    data_base: Vec<usize>,
    index: HashMap<String, TensorInfo>,
    /// fp8 块量化的 (block_n, block_k)。
    quant_block: (usize, usize),
}

/// 本 crate 唯一的 unsafe：只读 mmap。安全前提：文件在映射期间不被修改（权重文件只读，约定成立）。
#[allow(unsafe_code)]
fn map_readonly(file: &File) -> Result<Mmap> {
    // SAFETY: 见上。
    Ok(unsafe { Mmap::map(file)? })
}

impl WeightSource {
    /// `dir` 下的所有 `*.safetensors`（含分片）。
    pub fn open_dir(dir: &Path) -> Result<Self> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .with_context(|| format!("read_dir {}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect();
        files.sort();
        ensure!(!files.is_empty(), "no *.safetensors under {}", dir.display());
        Self::open_files(&files)
    }

    /// 设置 fp8 块量化的块大小（`quantization_config.weight_block_size`，GLM-5.3 为 [128,128]）。默认 128×128。
    pub fn with_quant_block(mut self, bn: usize, bk: usize) -> Self {
        self.quant_block = (bn, bk);
        self
    }

    pub fn open_files(files: &[PathBuf]) -> Result<Self> {
        let mut maps = Vec::new();
        let mut data_base = Vec::new();
        let mut index = HashMap::new();
        for (fi, f) in files.iter().enumerate() {
            let file = File::open(f).with_context(|| f.display().to_string())?;
            let map = map_readonly(&file)?;
            ensure!(map.len() >= 8, "{}: too short", f.display());
            let n = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
            ensure!(8 + n <= map.len(), "{}: bad header length", f.display());
            let header: Value =
                serde_json::from_slice(&map[8..8 + n]).with_context(|| f.display().to_string())?;
            let base = 8 + n;
            let obj = header.as_object().context("header not object")?;
            for (name, v) in obj {
                if name == "__metadata__" {
                    continue;
                }
                let dtype = v["dtype"].as_str().context("dtype")?.to_string();
                let shape: Vec<usize> = v["shape"]
                    .as_array()
                    .context("shape")?
                    .iter()
                    .map(|x| x.as_u64().unwrap() as usize)
                    .collect();
                let off = v["data_offsets"].as_array().context("data_offsets")?;
                let (start, end) = (off[0].as_u64().unwrap() as usize, off[1].as_u64().unwrap() as usize);
                ensure!(base + end <= map.len(), "{}: tensor {name} out of range", f.display());
                index.insert(name.clone(), TensorInfo { dtype, shape, start, end, file: fi });
            }
            maps.push(map);
            data_base.push(base);
        }
        Ok(Self { maps, data_base, index, quant_block: (128, 128) })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(String::as_str)
    }
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name)
    }
    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn bytes(&self, name: &str) -> Result<(&TensorInfo, &[u8])> {
        let info = self.index.get(name).with_context(|| format!("missing tensor {name}"))?;
        let base = self.data_base[info.file];
        Ok((info, &self.maps[info.file][base + info.start..base + info.end]))
    }

    /// 块量化 fp8 权重的解量化（CPU）；`name` 需为 `*.weight` 且存在同前缀 `*.weight_scale_inv`。
    fn dequant_fp8(&self, name: &str, info: &TensorInfo, raw: &[u8]) -> Result<Vec<f32>> {
        let scale_name = format!("{name}_scale_inv");
        let (sinfo, sraw) = self.bytes(&scale_name).with_context(|| {
            format!("{name}: F8_E4M3 weight needs {scale_name} (block-quantized); per-tensor fp8 unsupported")
        })?;
        ensure!(sinfo.dtype == "F32", "{scale_name}: expected F32 scale_inv, got {}", sinfo.dtype);
        let scales: Vec<f32> =
            sraw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let (bn, bk) = self.quant_block;
        Ok(crate::weights_fp8::dequant_block(raw, &info.shape, &scales, &sinfo.shape, (bn, bk)))
    }

    /// 读为 bf16（支持 BF16 直读、F32/F16 转换、F8_E4M3 块量化解码）。
    pub fn load_bf16(&self, stream: &Stream, name: &str) -> Result<Tensor<bf16>> {
        let (info, raw) = self.bytes(name)?;
        let n: usize = info.shape.iter().product();
        let v: Vec<bf16> = match info.dtype.as_str() {
            "F8_E4M3" => {
                ensure!(raw.len() == n, "{name}: size mismatch");
                self.dequant_fp8(name, info, raw)?.into_iter().map(bf16::from_f32).collect()
            }
            "BF16" => {
                ensure!(raw.len() == n * 2, "{name}: size mismatch");
                // mmap 切片按 u16 批量拷（小端机器上 bf16 位模式 == u16），比逐元素 collect 快一个量级
                let mut v: Vec<bf16> = Vec::with_capacity(n);
                let words: &[u16] = bytemuck::try_cast_slice(raw).unwrap_or(&[]);
                if words.len() == n {
                    v.extend(words.iter().map(|&w| bf16::from_bits(u16::from_le(w))));
                } else {
                    v.extend(raw.chunks_exact(2).map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]]))));
                }
                v
            }
            "F32" => {
                ensure!(raw.len() == n * 4, "{name}: size mismatch");
                raw.chunks_exact(4)
                    .map(|c| bf16::from_f32(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                    .collect()
            }
            "F16" => {
                ensure!(raw.len() == n * 2, "{name}: size mismatch");
                raw.chunks_exact(2)
                    .map(|c| bf16::from_f32(half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()))
                    .collect()
            }
            other => bail!("{name}: unsupported dtype {other} for bf16 load"),
        };
        Tensor::from_vec(stream, v, &info.shape)
    }

    /// 读为 host f32（用于 CPU 侧预处理，如 W_UK / W_UV 拆分）。fp8 块量化权重先解量化。
    pub fn load_f32_host(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let (info, raw) = self.bytes(name)?;
        let v: Vec<f32> = match info.dtype.as_str() {
            "F8_E4M3" => self.dequant_fp8(name, info, raw)?,
            "BF16" => {
                let words: &[u16] = bytemuck::try_cast_slice(raw).unwrap_or(&[]);
                if words.len() * 2 == raw.len() {
                    words.iter().map(|&w| bf16::from_bits(u16::from_le(w)).to_f32()).collect()
                } else {
                    raw.chunks_exact(2)
                        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                        .collect()
                }
            }
            "F32" => raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            "F16" => raw
                .chunks_exact(2)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            other => bail!("{name}: unsupported dtype {other}"),
        };
        Ok((info.shape.clone(), v))
    }
}
