//! B4 诊断：decode attention 在 (batch, seq_len, splits) 网格上的 kernel 耗时与有效 KV 带宽（docs/loops/b4-sm100.md）。
//! 跑法：DECODE_SPLITS=<n> cargo test -p dinfer-kernels --test decode_sweep -- --ignored --nocapture
//! 只读 KV 的理论下限 = b·n·hk·2·d·2 B / 带宽（B300 ~8 TB/s，4090 ~1 TB/s）。
use dinfer_kernels::Device;
use half::bf16;
use std::time::Instant;

#[test]
#[ignore]
fn decode_sweep() {
    let dev = Device::new(0).unwrap();
    dev.disable_event_tracking();
    let s = dev.new_stream().unwrap();
    let (h, hk, d, bs) = (16usize, 8usize, 128usize, 16usize);
    let splits_env = std::env::var("DECODE_SPLITS").unwrap_or_default();
    println!("DECODE_SPLITS={splits_env:?}   b     n      v2(us)   paged_tc(us)   KV(MB)  v2 GB/s");
    for &(b, n) in &[(1usize, 128usize), (1, 1024), (8, 128), (8, 1024), (8, 4096), (32, 128), (32, 1024), (64, 1024)] {
        let blocks_per = n.div_ceil(bs);
        let cache = s.alloc_zeros::<bf16>(b * blocks_per * bs * hk * 2 * d).unwrap();
        // 乱序 block 表：模拟真实分配（相邻 key 不一定相邻块）
        let mut ids: Vec<i32> = (0..(b * blocks_per) as i32).collect();
        let mut x = 0x9e3779b9u32;
        for i in (1..ids.len()).rev() {
            x ^= x << 13; x ^= x >> 17; x ^= x << 5;
            ids.swap(i, (x as usize) % (i + 1));
        }
        let dbt = s.htod_copy(ids).unwrap();
        let seq = s.htod_copy(vec![n as i32; b]).unwrap();
        let dcu = s.htod_copy((0..=b as i32).collect::<Vec<_>>()).unwrap();
        let qd = s.alloc_zeros::<bf16>(b * h * d).unwrap();
        let mut od = s.alloc_zeros::<bf16>(b * h * d).unwrap();
        let mut od2 = s.alloc_zeros::<bf16>(b * h * d).unwrap();
        let mut scratch = s.alloc_zeros::<f32>(dinfer_kernels::gqa_decode_scratch_len(b, h, d)).unwrap();
        let scale = 1.0 / (d as f32).sqrt();
        let mut f_v2 = || {
            dinfer_kernels::gqa_decode_bf16(&s, &qd, &cache, &dbt, &seq, &mut od, &mut scratch, b, h, hk, d, bs, blocks_per, scale, 0, None).unwrap()
        };
        let mut f_paged = || {
            dinfer_kernels::gqa_paged_bf16(&s, &qd, &cache, &dbt, &seq, &dcu, &mut od2, b, 1, h, hk, d, bs, blocks_per, scale, 0, false).unwrap()
        };
        let time = |f: &mut dyn FnMut()| {
            for _ in 0..5 { f(); }
            s.synchronize().unwrap();
            let t = Instant::now();
            let iters = 200;
            for _ in 0..iters { f(); }
            s.synchronize().unwrap();
            t.elapsed().as_secs_f64() * 1e6 / iters as f64
        };
        let t_v2 = time(&mut f_v2);
        let t_paged = time(&mut f_paged);
        let kv_bytes = (b * n * hk * 2 * d * 2) as f64;
        println!("{:>22} {b:>3} {n:>6}   {t_v2:>7.2}   {t_paged:>9.2}   {:>6.2}  {:>7.0}", "", kv_bytes / 1e6, kv_bytes / t_v2 / 1e3);
    }
}
