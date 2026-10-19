//! lm_head 列切（`LMHEAD_COLSPLIT=1`，loop b-nvlink B2）的单卡形状测试：
//!
//! 1. `shards_concat_equals_full`：随机小 `lm_head.weight` [vocab, hidden] 写成 safetensors，world=2 分别
//!    `Linear::load_shard(dim 0, rank 0/1)` 后沿 vocab 拼回，与 `Linear::load` 全量逐位一致。
//! 2. `gather_blocks_matches_cpu_layout_fix`：模拟 all_gather 出来的 `[world][rows][vocab_local]` bf16 块布局，
//!    `gather_blocks_bf16_to_f32` 后等于 CPU 侧按 `[rows][vocab]` 拼接再升 f32 的结果（逐位）。
//!
//! 多卡端到端一致性（GEMM + NCCL all_gather + 采样同 token）留给 B300：`LMHEAD_COLSPLIT=1 scripts/verify/q5_tp.sh 2 0`。
//! 无 GPU 时直接跳过（打印 skip），不标 ignore。

use dinfer_layers::{bf16, Device, Linear, Stream, Tensor, WeightSource};
use std::path::PathBuf;

fn gpu() -> Option<(Device, Stream)> {
    let dev = Device::new(0).ok()?;
    let s = dev.default_stream();
    Some((dev, s))
}

/// 简单 LCG，避免引入 rand 依赖；覆盖 bf16 的正负 / 大小范围。
fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let u = ((*seed >> 40) as u32) as f32 / (1u64 << 24) as f32; // [0,1)
    (u - 0.5) * 8.0
}

/// 手写 safetensors（8 字节 header 长度 + JSON header + 数据），只写一个 BF16 张量。
fn write_safetensors(path: &PathBuf, name: &str, shape: &[usize], data: &[bf16]) {
    let nbytes = data.len() * 2;
    let header = format!(
        r#"{{"{name}":{{"dtype":"BF16","shape":[{}],"data_offsets":[0,{nbytes}]}}}}"#,
        shape.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")
    );
    // header 按 8 字节对齐补空格（safetensors 规范允许）
    let pad = (8 - header.len() % 8) % 8;
    let header = format!("{header}{}", " ".repeat(pad));
    let mut buf = Vec::with_capacity(8 + header.len() + nbytes);
    buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
    buf.extend_from_slice(header.as_bytes());
    for v in data {
        buf.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    std::fs::write(path, buf).unwrap();
}

#[test]
fn shards_concat_equals_full() {
    let Some((_dev, s)) = gpu() else {
        eprintln!("skip: no GPU");
        return;
    };
    let (vocab, hidden, world) = (64usize, 24usize, 2usize);
    let mut seed = 0x5eed_1234u64;
    let w: Vec<bf16> = (0..vocab * hidden).map(|_| bf16::from_f32(lcg(&mut seed))).collect();
    let dir = std::env::temp_dir().join(format!("lmhead_colsplit_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("model.safetensors");
    write_safetensors(&f, "lm_head.weight", &[vocab, hidden], &w);
    let src = WeightSource::open_files(&[f]).unwrap();

    let full = Linear::load(&s, &src, "lm_head").unwrap();
    assert_eq!((full.out_features, full.in_features), (vocab, hidden));
    let full_host = full.w.to_vec(&s).unwrap();
    assert_eq!(full_host.len(), w.len());
    assert!(full_host.iter().zip(&w).all(|(a, b)| a.to_bits() == b.to_bits()), "full load != source");

    let mut cat: Vec<bf16> = Vec::with_capacity(vocab * hidden);
    for rank in 0..world {
        let part = Linear::load_shard(&s, &src, "lm_head", 0, rank, world).unwrap();
        assert_eq!((part.out_features, part.in_features), (vocab / world, hidden), "rank {rank} shard shape");
        cat.extend(part.w.to_vec(&s).unwrap());
    }
    assert_eq!(cat.len(), full_host.len());
    let bad = cat.iter().zip(&full_host).position(|(a, b)| a.to_bits() != b.to_bits());
    assert!(bad.is_none(), "shard concat differs from full load at {bad:?}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn gather_blocks_matches_cpu_layout_fix() {
    let Some((_dev, s)) = gpu() else {
        eprintln!("skip: no GPU");
        return;
    };
    // 非整除 256 的尺寸，覆盖 kernel 的 grid-stride 尾巴
    for &(world, rows, vl) in &[(2usize, 3usize, 7usize), (4, 5, 33), (8, 32, 19), (2, 1, 1)] {
        let mut seed = (world * 1000 + rows * 10 + vl) as u64;
        // all_gather 布局：块 b = rank b 的 [rows][vl]
        let blocks: Vec<bf16> = (0..world * rows * vl).map(|_| bf16::from_f32(lcg(&mut seed))).collect();
        let mut expect = vec![0f32; rows * world * vl];
        for b in 0..world {
            for r in 0..rows {
                for c in 0..vl {
                    expect[r * (world * vl) + b * vl + c] = blocks[(b * rows + r) * vl + c].to_f32();
                }
            }
        }
        let x = Tensor::from_vec(&s, blocks, &[world, rows, vl]).unwrap();
        let mut y = Tensor::<f32>::zeros(&s, &[rows, world * vl]).unwrap();
        dinfer_kernels::gather_blocks_bf16_to_f32(&s, &x.data, &mut y.data, world, rows, vl).unwrap();
        let got = y.to_vec(&s).unwrap();
        let bad = got.iter().zip(&expect).position(|(a, b)| a.to_bits() != b.to_bits());
        assert!(bad.is_none(), "world={world} rows={rows} vl={vl}: mismatch at {bad:?}");
    }
}
