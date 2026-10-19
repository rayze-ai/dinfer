//! 自定义 one-shot all-reduce（`comm.rs` + `allreduce.cu`）多卡对拍。需要 ≥ world 张有 P2P 的卡（NVLink 机器）。
//!
//! 运行（B300）：`CUDA_VISIBLE_DEVICES=0,1 cargo test --release --offline -p dinfer-kernels --test allreduce -- --ignored`
//! `AR_WORLD=4` 选 world（默认 2）。4090 Pod 无 P2P：custom 建不起来，测试会 skip（打印 "no P2P"）。
//!
//! 断言：
//! 1) one-shot 结果 == CPU 参考（固定 rank 顺序 fp32 累加后一次 bf16 round）**逐位**；world 2 时也 == NCCL 逐位
//!    （world ≥ 4 NCCL ring 每跳 bf16 舍入，与 one-shot 不同是预期的，只报告）。
//! 2) 融合版 `all_reduce_add_rmsnorm_bf16`：res == CPU 参考逐位、normed ≤ 1 ulp；vs（NCCL all-reduce + fused_add_rmsnorm）
//!    路径 world 2 res 逐位；world ≥ 4 只报告不断言（NCCL ring 每跳 bf16 舍入，与 fp32 累加后一次舍入不同，差 ≥ 1 ulp 常见）。
#![cfg(not(feature = "no-nvcc"))]

use dinfer_kernels::comm::{self, ArPeerInfo, Comm, CommId};
use dinfer_kernels::{bf16, Device, Stream};
use std::sync::{Arc, Barrier, Mutex};

fn world() -> usize {
    std::env::var("AR_WORLD").ok().and_then(|v| v.parse().ok()).unwrap_or(2)
}

struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn fill_bf16(&mut self, n: usize, scale: f32) -> Vec<bf16> {
        (0..n).map(|_| bf16::from_f32(self.next_f32() * scale)).collect()
    }
}

/// 进程内 "all-gather" ArPeerInfo：每 rank 放进自己的格子，barrier 后各自读全表。
struct Exchange {
    slots: Mutex<Vec<Option<ArPeerInfo>>>,
    bar: Barrier,
}

/// 各 rank 线程建 Comm + 自定义 all-reduce（握手与 TpExecutor::setup_custom_ar 同逻辑）；返回 ready 与否（全组一致）。
fn setup(world: usize, rank: usize, id: CommId, ex: &Exchange) -> (Device, Stream, Comm, bool) {
    let dev = Device::new(rank).unwrap();
    let s = dev.default_stream();
    s.bind_ctx().unwrap();
    let mut c = Comm::init(&s, rank, world, id).unwrap();
    let info = c.custom_ar_begin().unwrap();
    eprintln!("rank {rank}: custom_ar_begin -> {}", if info.is_some() { "ok" } else { "None (env / world / ipc alloc)" });
    ex.slots.lock().unwrap()[rank] = info;
    ex.bar.wait();
    let all: Vec<Option<ArPeerInfo>> = ex.slots.lock().unwrap().clone();
    let ready = if all.iter().all(|x| x.is_some()) {
        let peers: Vec<ArPeerInfo> = all.into_iter().map(|x| x.unwrap()).collect();
        let ok = c.custom_ar_finish(&peers).unwrap();
        eprintln!("rank {rank}: custom_ar_finish -> {ok} (peers dev {:?})", peers.iter().map(|p| p.device_ordinal).collect::<Vec<_>>());
        ok
    } else {
        false
    };
    // 全组一致：任一失败全 disable
    static OKS: Mutex<Vec<bool>> = Mutex::new(Vec::new());
    {
        let mut g = OKS.lock().unwrap();
        if g.len() < world {
            g.resize(world, false);
        }
        g[rank] = ready;
    }
    ex.bar.wait();
    let all_ok = OKS.lock().unwrap().iter().all(|&b| b);
    if !all_ok {
        c.custom_ar_disable();
    }
    ex.bar.wait();
    (dev, s, c, all_ok)
}

fn run_ranks<F>(world: usize, f: F)
where
    F: Fn(usize, &Stream, &Comm, &Barrier) + Send + Sync + 'static,
{
    // comm.rs 的回退原因走 tracing::info，测试里装一个 stderr subscriber 才看得见
    let _ = tracing_subscriber::fmt().with_env_filter("info").with_writer(std::io::stderr).try_init();
    let id = CommId::new().unwrap();
    let ex = Arc::new(Exchange { slots: Mutex::new(vec![None; world]), bar: Barrier::new(world) });
    let f = Arc::new(f);
    let ready_flags = Arc::new(Mutex::new(vec![false; world]));
    let hs: Vec<_> = (0..world)
        .map(|rank| {
            let ex = ex.clone();
            let f = f.clone();
            let rf = ready_flags.clone();
            std::thread::spawn(move || {
                let (_dev, s, c, ready) = setup(world, rank, id, &ex);
                rf.lock().unwrap()[rank] = ready;
                if !ready {
                    if rank == 0 {
                        eprintln!("custom all-reduce not available (no P2P / CUSTOM_AR=0): SKIP");
                    }
                    return;
                }
                f(rank, &s, &c, &ex.bar);
            })
        })
        .collect();
    for h in hs {
        h.join().expect("rank thread panicked");
    }
}

/// CPU 参考：固定 rank 顺序 fp32 累加，一次 bf16 round（与 kernel 同序）。
fn cpu_sum(parts: &[Vec<bf16>]) -> Vec<bf16> {
    let n = parts[0].len();
    (0..n)
        .map(|i| {
            let mut a = 0f32;
            for p in parts {
                a += p[i].to_f32();
            }
            bf16::from_f32(a)
        })
        .collect()
}

fn ulp_diff(a: bf16, b: bf16) -> u32 {
    (a.to_bits() as i32 - b.to_bits() as i32).unsigned_abs()
}

/// bf16 容差：|a-b| ≤ 1e-6 或 相对 1/128（1 bf16 ulp）。ulp 距离在 0 附近没意义（0 vs 1e-30 相差上千 ulp），所以用它。
fn close(a: bf16, b: bf16) -> bool {
    let (x, y) = (a.to_f32(), b.to_f32());
    let d = (x - y).abs();
    d <= 1e-6 || d <= x.abs().max(y.abs()) / 128.0
}

#[test]
#[ignore]
fn oneshot_matches_reference_and_nccl() {
    let w = world();
    // 元素数：1 KB … 512 KB（bf16，2 B）；含非 2 幂与 world-8 阈值边界
    let sizes: Vec<usize> = vec![512, 4096, 8 * 1024, 32 * 1024, 100 * 1024, 128 * 1024, 262144];
    run_ranks(w, move |rank, s, c, bar| {
        for (ci, &n) in sizes.iter().enumerate() {
            if n * 2 > (if w <= 4 { 512 << 10 } else { 256 << 10 }) {
                continue;
            }
            // 所有 rank 用同一组种子生成全部 rank 的输入，便于本地算参考
            let parts: Vec<Vec<bf16>> = (0..w).map(|r| Lcg(0x9E37_79B9 + (ci * 131 + r) as u64).fill_bf16(n, 4.0)).collect();
            let want = cpu_sum(&parts);
            let mut d_custom = s.htod_copy(parts[rank].clone()).unwrap();
            let mut d_nccl = s.htod_copy(parts[rank].clone()).unwrap();
            bar.wait();
            comm::all_reduce_bf16(c, &mut d_custom, n).unwrap();
            comm::all_reduce_bf16_nccl(c, &mut d_nccl, n).unwrap();
            s.synchronize().unwrap();
            let got = s.dtoh_sync_copy(&d_custom).unwrap();
            let via_nccl = s.dtoh_sync_copy(&d_nccl).unwrap();
            let bad = got.iter().zip(&want).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            assert_eq!(bad, 0, "rank {rank} n={n}: one-shot vs CPU ref {bad} mismatches");
            let nccl_bad = got.iter().zip(&via_nccl).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            let max_ulp = got.iter().zip(&via_nccl).map(|(a, b)| ulp_diff(*a, *b)).max().unwrap_or(0);
            if w == 2 {
                assert_eq!(nccl_bad, 0, "rank {rank} n={n}: one-shot vs NCCL {nccl_bad} mismatches (world 2 must be bitwise)");
            } else if rank == 0 {
                // world ≥ 4：NCCL ring 每一跳都 bf16 round（4 rank = 3 次中间舍入），与 fp32 累加后一次舍入不同，
                // 差 1 ulp 很常见、和在 0 附近抵消时相对差更大（B300 实测 4 rank n=512：53/512 超 1/128 相对容差）。
                // 这是 NCCL 的精度损失不是 kernel 错（正确性已由 CPU 参考逐位断言）；只报告。
                let far = got.iter().zip(&via_nccl).filter(|(a, b)| !close(**a, **b)).count();
                eprintln!("  (world {w}: NCCL ring 多次 bf16 舍入 → vs NCCL {far}/{n} 超 1-ulp 相对容差，预期内，不断言)");
            }
            if rank == 0 {
                eprintln!("n={n:>7} ({:>4} KB) world={w}: one-shot == ref (bitwise); vs NCCL mismatches={nccl_bad} max_ulp={max_ulp}", n * 2 / 1024);
            }
        }
        // 重复调用（flag 单调计数器）：同一缓冲连续 64 次仍一致
        let n = 16 * 1024;
        let parts: Vec<Vec<bf16>> = (0..w).map(|r| Lcg(77 + r as u64).fill_bf16(n, 1.0)).collect();
        let want = cpu_sum(&parts);
        for it in 0..64 {
            let mut d = s.htod_copy(parts[rank].clone()).unwrap();
            comm::all_reduce_bf16(c, &mut d, n).unwrap();
            let got = s.dtoh_sync_copy(&d).unwrap();
            let bad = got.iter().zip(&want).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            assert_eq!(bad, 0, "rank {rank} iter {it}: {bad} mismatches");
        }
        if rank == 0 {
            eprintln!("64 back-to-back one-shot calls: bitwise OK");
        }
    });
}

#[test]
#[ignore]
fn fused_add_rmsnorm_matches_nccl_path() {
    let w = world();
    let cases: Vec<(usize, usize)> = vec![(1, 1024), (8, 1024), (32, 1024), (37, 1024), (64, 1024), (16, 2048), (4, 4096)];
    let eps = 1e-6f32;
    run_ranks(w, move |rank, s, c, bar| {
        for (ci, &(t, h)) in cases.iter().enumerate() {
            let n = t * h;
            let parts: Vec<Vec<bf16>> = (0..w).map(|r| Lcg(1000 + (ci * 17 + r) as u64).fill_bf16(n, 2.0)).collect();
            let res0 = Lcg(5000 + ci as u64).fill_bf16(n, 3.0);
            let wv = Lcg(9000 + ci as u64).fill_bf16(h, 1.0);
            let d_w = s.htod_copy(wv.clone()).unwrap();
            // A：融合版
            let mut a_part = s.htod_copy(parts[rank].clone()).unwrap();
            let mut a_res = s.htod_copy(res0.clone()).unwrap();
            let mut a_norm = s.alloc_zeros::<bf16>(n).unwrap();
            // B：NCCL + fused_add_rmsnorm
            let mut b_part = s.htod_copy(parts[rank].clone()).unwrap();
            let mut b_res = s.htod_copy(res0.clone()).unwrap();
            let mut b_norm = s.alloc_zeros::<bf16>(n).unwrap();
            bar.wait();
            comm::all_reduce_add_rmsnorm_bf16(c, &mut a_part, &mut a_res, &d_w, &mut a_norm, t, h, eps).unwrap();
            comm::all_reduce_bf16_nccl(c, &mut b_part, n).unwrap();
            dinfer_kernels::fused_add_rmsnorm_bf16_n(s, &b_part, &mut b_res, &d_w, &mut b_norm, t, h, eps).unwrap();
            s.synchronize().unwrap();
            let (ar, an) = (s.dtoh_sync_copy(&a_res).unwrap(), s.dtoh_sync_copy(&a_norm).unwrap());
            let (br, bn) = (s.dtoh_sync_copy(&b_res).unwrap(), s.dtoh_sync_copy(&b_norm).unwrap());
            // CPU 参考（与 fused_add_rmsnorm 同序）：sum→bf16，+res→bf16；normed = bf16(res · rsqrt(mean(res²)+eps) · w)
            let sum = cpu_sum(&parts);
            let res_ref: Vec<bf16> = sum.iter().zip(&res0).map(|(a, b)| bf16::from_f32(a.to_f32() + b.to_f32())).collect();
            let mut norm_ref = vec![bf16::ZERO; n];
            for r in 0..t {
                let row = &res_ref[r * h..(r + 1) * h];
                let ss: f32 = row.iter().map(|v| v.to_f32() * v.to_f32()).sum();
                let inv = 1.0 / (ss / h as f32 + eps).sqrt();
                for i in 0..h {
                    norm_ref[r * h + i] = bf16::from_f32(row[i].to_f32() * inv * wv[i].to_f32());
                }
            }
            // 融合版 vs CPU 参考：res 逐位；normed ≤ 1 ulp（rsqrtf 近似 + 乘法顺序）
            let res_bad = ar.iter().zip(&res_ref).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            assert_eq!(res_bad, 0, "rank {rank} t={t} h={h}: fused res vs CPU ref {res_bad} mismatches");
            let norm_ulp = an.iter().zip(&norm_ref).map(|(a, b)| ulp_diff(*a, *b)).max().unwrap();
            let norm_far = an.iter().zip(&norm_ref).filter(|(a, b)| !close(**a, **b)).count();
            assert!(norm_ulp <= 1 || norm_far == 0, "rank {rank} t={t} h={h}: fused normed vs CPU ref max {norm_ulp} ulp, {norm_far} beyond tol");
            // 融合版 vs（NCCL + fused_add_rmsnorm）路径：world 2 res 逐位；world ≥ 4 只要求 close()
            let res_max_ulp = ar.iter().zip(&br).map(|(a, b)| ulp_diff(*a, *b)).max().unwrap();
            let res_far = ar.iter().zip(&br).filter(|(a, b)| !close(**a, **b)).count();
            let norm_far_nccl = an.iter().zip(&bn).filter(|(a, b)| !close(**a, **b)).count();
            let norm_max = an.iter().zip(&bn).map(|(a, b)| (a.to_f32() - b.to_f32()).abs()).fold(0f32, f32::max);
            if w == 2 {
                assert_eq!(res_max_ulp, 0, "rank {rank} t={t} h={h}: res not bitwise vs NCCL path");
                assert_eq!(norm_far_nccl, 0, "rank {rank} t={t} h={h}: normed vs NCCL path {norm_far_nccl} beyond tolerance (max |Δ| {norm_max})");
            } else if rank == 0 {
                eprintln!("  (world {w}: vs NCCL 路径 res {res_far}/{n} normed {norm_far_nccl}/{n} 超容差，NCCL 多次舍入所致，不断言)");
            }
            // part 不应被融合版改写（one-shot 走镜像区）
            let ap = s.dtoh_sync_copy(&a_part).unwrap();
            let untouched = ap.iter().zip(&parts[rank]).all(|(a, b)| a.to_bits() == b.to_bits());
            if rank == 0 {
                eprintln!("t={t:>3} h={h}: vs CPU ref: res bitwise, normed max_ulp={norm_ulp}; vs NCCL path: res max_ulp={res_max_ulp} normed max|Δ|={norm_max:.2e}; part_untouched={untouched}");
            }
        }
    });
}
