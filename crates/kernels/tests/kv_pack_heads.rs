//! kv_pack / kv_unpack head-major 布局（design §13.7 异构 tp）：
//! 1) pack 后 staging 每层 = `[kv_head][slot][K/V][D]`，与 CPU 参考逐字节相同；2) unpack 回原 cache 逐字节相同。
//! 跑法：cargo test -p dinfer-kernels --test kv_pack_heads -- --ignored
use dinfer_kernels::{ops, Device};

#[test]
#[ignore]
fn kv_pack_heads_roundtrip() {
    let dev = Device::new(0).unwrap();
    let s = dev.default_stream();
    let (layers, num_blocks, bs, hk, d) = (3usize, 7usize, 16usize, 4usize, 128usize);
    let per_layer = bs * hk * 2 * d * 2; // bytes, bf16
    // 每层 cache：随机字节
    let mut host: Vec<Vec<u8>> = (0..layers)
        .map(|l| (0..num_blocks * per_layer).map(|i| ((i * 31 + l * 7) % 251) as u8).collect())
        .collect();
    let caches: Vec<_> = host.iter().map(|h| s.htod_copy(h.clone()).unwrap()).collect();
    let ptrs: Vec<u64> = caches.iter().map(|c| s.ptr(c)).collect();
    let layer_ptrs = s.htod_copy(ptrs.clone()).unwrap();
    let blocks_v: Vec<i32> = vec![5, 0, 3];
    let blocks = s.htod_copy(blocks_v.clone()).unwrap();
    let mut staging = s.alloc_zeros::<u8>(blocks_v.len() * layers * per_layer).unwrap();
    ops::kv_pack(&s, &layer_ptrs, &blocks, &mut staging, blocks_v.len(), layers, per_layer, bs, hk).unwrap();
    let got = s.dtoh_sync_copy(&staging).unwrap();
    // CPU 参考：row = 2*d*2 字节（一个 slot 一个头的 K+V）
    let row = 2 * d * 2;
    let mut want = vec![0u8; got.len()];
    for (bi, &b) in blocks_v.iter().enumerate() {
        for l in 0..layers {
            let src = &host[l][b as usize * per_layer..(b as usize + 1) * per_layer];
            let dst = &mut want[(bi * layers + l) * per_layer..(bi * layers + l + 1) * per_layer];
            for sl in 0..bs {
                for h in 0..hk {
                    let so = (sl * hk + h) * row;
                    let doff = (h * bs + sl) * row;
                    dst[doff..doff + row].copy_from_slice(&src[so..so + row]);
                }
            }
        }
    }
    assert_eq!(got, want, "pack layout mismatch");
    // unpack 到清零的 cache，再比
    let zero_caches: Vec<_> = (0..layers).map(|_| s.alloc_zeros::<u8>(num_blocks * per_layer).unwrap()).collect();
    let zptrs: Vec<u64> = zero_caches.iter().map(|c| s.ptr(c)).collect();
    let zlayer_ptrs = s.htod_copy(zptrs.clone()).unwrap();
    ops::kv_unpack(&s, &zlayer_ptrs, &blocks, &staging, blocks_v.len(), layers, per_layer, bs, hk).unwrap();
    for l in 0..layers {
        let back = s.dtoh_sync_copy(&zero_caches[l]).unwrap();
        for &b in &blocks_v {
            let r = b as usize * per_layer..(b as usize + 1) * per_layer;
            assert_eq!(&back[r.clone()], &host[l][r.clone()], "unpack mismatch layer {l} block {b}");
        }
    }
    // 没被 pack 的块应仍为 0
    let back0 = s.dtoh_sync_copy(&zero_caches[0]).unwrap();
    assert!(back0[1 * per_layer..2 * per_layer].iter().all(|&x| x == 0));
    host.clear();
}

/// `Stream::copy2d`（PD 异构 tp 的 rank 列拷贝）：两个"rank"各把紧凑行写进 pinned canonical（pitch = 2 × 行宽）的自己列，
/// 再各自读回，逐字节相同；且 host 侧 pitch 行的两列互不覆盖。
#[test]
#[ignore]
fn copy2d_pinned_rank_columns() {
    use dinfer_kernels::PinnedPool;
    let dev = Device::new(0).unwrap();
    let s = dev.default_stream();
    let (width, height, world) = (4096usize, 9usize, 2usize);
    let pitch = width * world;
    let pool = PinnedPool::new(&s, 1, pitch * height).unwrap();
    let mut buf = pool.take().unwrap();
    buf.as_mut_cap().fill(0xEE);
    let base = buf.ptr() as u64;
    let srcs: Vec<Vec<u8>> = (0..world).map(|r| (0..width * height).map(|i| ((i + r * 97) % 253) as u8).collect()).collect();
    let dsrcs: Vec<_> = srcs.iter().map(|v| s.htod_copy(v.clone()).unwrap()).collect();
    for r in 0..world {
        s.copy2d(base + (r * width) as u64, true, pitch, s.ptr(&dsrcs[r]), false, width, width, height).unwrap();
    }
    s.synchronize().unwrap();
    let host = &buf.as_mut_cap()[..pitch * height];
    for row in 0..height {
        for r in 0..world {
            let got = &host[row * pitch + r * width..row * pitch + (r + 1) * width];
            assert_eq!(got, &srcs[r][row * width..(row + 1) * width], "row {row} rank {r}");
        }
    }
    // 读回：每 rank 只读自己列到紧凑设备缓冲
    for r in 0..world {
        let mut d = s.alloc_zeros::<u8>(width * height).unwrap();
        s.copy2d(s.ptr(&d), false, width, base + (r * width) as u64, true, pitch, width, height).unwrap();
        let _ = &mut d;
        let back = s.dtoh_sync_copy(&d).unwrap();
        assert_eq!(back, srcs[r], "readback rank {r}");
    }
}
