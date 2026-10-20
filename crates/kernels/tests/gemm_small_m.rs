//! decode 小 M GEMM 微基准（docs/loops/qwen35-perf.md 第 3 / 5B 轮）：Qwen3.5-27B 每层形状在 M 列表（GEMM_BENCH_M=1,7,8,9,…）下
//! cuBLASLt 默认 / cuBLASLt 最优候选 / 自写 small_m kernel 三者的 µs 与只读一遍权重的有效 GB/s。
//! 跑法：cargo test -p dinfer-kernels --test gemm_small_m -- --ignored --nocapture
//! 权重随机（bf16 GEMM 耗时与数值无关）；lm_head 2.5 GB，显存不够时设 GEMM_BENCH_SKIP_LM_HEAD=1。
use dinfer_kernels::{Device, Gemm, GemmPlan};
use half::bf16;

#[test]
#[ignore]
fn gemm_small_m_bench() {
    let dev = Device::new(0).unwrap();
    dev.disable_event_tracking();
    let s = dev.new_stream().unwrap();
    let g = Gemm::new(&s).unwrap();
    let p = g.profile();
    println!(
        "GPU cc={:?} sm={} arch={:?} lt_candidates={} split candidates(wave targets)={:?}",
        p.cc, p.sm_count, p.arch, p.gemm_tune.lt_candidates, p.gemm_tune.small_m_wave_targets
    );
    // GEMM_BENCH_M：逗号分隔的 M 列表（默认 1）。非桶 M 多一列 "bucket"：用上一个 2 的幂桶在本 M 实测选出的
    // plan（Lt algo 过 AlgoCheck；small_m 直接跑）在真实 M 上的耗时 —— 这就是 autotune 展开非桶 M 后 forward 实际走的路。
    let ms: Vec<usize> = std::env::var("GEMM_BENCH_M")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| vec![1]);
    let hid = 5120usize;
    // 第 9 轮：GDN 四投影融合成 in_proj [qkv 10240 | z 6144 | a 48 | b 48] = 16480，全注意力 qkv+gate 融合成 14336；
    // 旧的分开形状保留作对照（in_proj_a 独立 N=48 是 M=32 时 splitK+reduce 的典型）。
    let mut shapes: Vec<(&str, usize, usize)> = vec![
        ("gdn.in_proj", 16480, hid),
        ("gdn.in_proj_qkv", 10240, hid),
        ("gdn.in_proj_z", 6144, hid),
        ("gdn.in_proj_a", 48, hid),
        ("gdn.out_proj", hid, 6144),
        ("attn.qkv_gate", 14336, hid),
        ("attn.qkv", 8192, hid),
        ("attn.gate", 6144, hid),
        ("attn.o_proj", hid, 6144),
        ("mlp.gate_up", 34816, hid),
        ("mlp.down", hid, 17408),
    ];
    if std::env::var("GEMM_BENCH_SKIP_LM_HEAD").is_err() {
        shapes.push(("lm_head", 248320, hid));
    }
    let m_max = ms.iter().copied().max().unwrap().next_power_of_two();
    let k_max = shapes.iter().map(|x| x.2).max().unwrap();
    let n_max = shapes.iter().map(|x| x.1).max().unwrap();
    let a = s.alloc_zeros::<bf16>(m_max * k_max).unwrap();
    let mut out = s.alloc_zeros::<bf16>(m_max * n_max).unwrap();
    let mut partial = s.alloc_zeros::<f32>(p.gemm_tune.split_workspace_max_bytes / 4).unwrap();
    println!(
        "{:<4} {:<16} {:>6} {:>6} | {:>8} {:>5} | {:>8} {:>5} {:<26} | {:>8} {:>5} {:>4} | {:>20} | {:>9} | chosen",
        "M", "shape", "N", "K", "lt_def", "GB/s", "lt_best", "GB/s", "algo", "small_m", "GB/s", "spl", "mma us GB/s cfg", "bucket us"
    );
    for (name, n, k) in &shapes {
        let mut x = 0x9e3779b97f4a7c15u64;
        let w: Vec<bf16> = (0..n * k)
            .map(|_| {
                x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                bf16::from_f32(((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.05)
            })
            .collect();
        let dw = s.htod_copy(w).unwrap();
        s.synchronize().unwrap();
        for &m in &ms {
            let row = g.tune_shape(name, m, *n, *k, &dw, None, &a, &mut out, &mut partial).unwrap();
            let bucket = m.next_power_of_two();
            let bucket_us = if bucket != m {
                let brow = g.tune_shape(name, bucket, *n, *k, &dw, None, &a, &mut out, &mut partial).unwrap();
                match g.time_plan(brow.chosen, m, *n, *k, &dw, None, &a, &mut out, &mut partial).unwrap() {
                    Some(us) => format!("{us:9.1}"),
                    None => "  invalid".to_string(),
                }
            } else {
                "        =".to_string()
            };
            let (sm_us, sm_gb) = match row.small_m_us {
                Some(u) => (format!("{u:8.1}"), format!("{:5.0}", row.gbps(u))),
                None => ("       -".into(), "    -".into()),
            };
            let chosen = match row.chosen {
                GemmPlan::LtAlgo(_) => "lt".to_string(),
                GemmPlan::SmallM { splits } => format!("small_m(s{splits})"),
                GemmPlan::SmallMma { splits, warps } => format!("mma(s{splits}w{warps})"),
            };
            let (mma_us, mma_gb) = match row.mma_us {
                Some(u) => (format!("{u:8.1}"), format!("{:5.0}", row.gbps(u))),
                None => ("       -".into(), "    -".into()),
            };
            println!(
                "{:<4} {:<16} {:>6} {:>6} | {:8.1} {:5.0} | {:8.1} {:5.0} {:<26} | {} {} {:>4} | {} {} s{}w{} | {} | {}",
                m, name, n, k, row.lt_default_us, row.gbps(row.lt_default_us), row.lt_best_us, row.gbps(row.lt_best_us),
                row.lt_best_desc, sm_us, sm_gb, row.small_m_splits, mma_us, mma_gb, row.mma_splits, row.mma_warps, bucket_us, chosen
            );
        }
        drop(dw);
    }
}

/// 自调优后 forward_bf16 走表（Lt 指定 algo / small_m split）仍与 CPU 参考一致；覆盖桶 M、非桶 M（3 / 7 / 9 / 31 / 33：借桶 + AlgoCheck）、超小 M 上限的 M（16）。
#[test]
#[ignore]
fn gemm_autotuned_matches_cpu() {
    use dinfer_kernels::GemmTuneEntry;
    let dev = Device::new(0).unwrap();
    dev.disable_event_tracking();
    let s = dev.new_stream().unwrap();
    let g = Gemm::new(&s).unwrap();
    assert!(g.tune_enabled(), "GEMM_TUNE=off 时本测试无意义");
    let shapes = [(48usize, 5120usize, false), (5120, 6144, true), (6144, 5120, false)];
    let mut x = 0x2545f4914f6cdd1du64;
    let mut rnd = move || {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        (x >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    let mut ws = Vec::new();
    for &(n, k, bias) in &shapes {
        let w: Vec<bf16> = (0..n * k).map(|_| bf16::from_f32(rnd() * (1.0 / k as f32).sqrt() * 2.0)).collect();
        let b: Vec<bf16> = (0..n).map(|_| bf16::from_f32(rnd())).collect();
        ws.push((n, k, bias, w.clone(), b.clone(), s.htod_copy(w).unwrap(), s.htod_copy(b).unwrap()));
    }
    let entries: Vec<GemmTuneEntry> = ws
        .iter()
        .map(|(n, k, bias, _, _, dw, db)| GemmTuneEntry { name: format!("n{n}_k{k}"), n: *n, k: *k, w: dw, bias: bias.then_some(db) })
        .collect();
    let report = g.autotune(&entries).unwrap();
    println!("autotune {:.0} ms, partial {} B", report.elapsed_ms, report.partial_bytes);
    for l in report.summary_lines() {
        println!("  {l}");
    }
    for (n, k, bias, w, b, dw, db) in &ws {
        for &m in &[1usize, 3, 7, 8, 9, 16, 31, 33] {
            let a: Vec<bf16> = (0..m * k).map(|_| bf16::from_f32(rnd())).collect();
            let da = s.htod_copy(a.clone()).unwrap();
            let mut out = s.alloc_zeros::<bf16>(m * n).unwrap();
            g.forward_bf16(&da, dw, bias.then_some(db), &mut out, m, *n, *k).unwrap();
            let got = s.dtoh_sync_copy(&out).unwrap();
            // 抽查 64 个元素（CPU 全算 5120×6144 太慢）
            let mut bad = 0;
            for t in 0..64 {
                let (i, j) = ((t * 7) % m, (t * 2654435761usize) % n);
                let mut acc = if *bias { b[j].to_f32() } else { 0.0 };
                for kk in 0..*k {
                    acc += a[i * k + kk].to_f32() * w[j * k + kk].to_f32();
                }
                let gv = got[i * n + j].to_f32();
                if (gv - acc).abs() > 3e-2 + 2e-2 * acc.abs() {
                    bad += 1;
                    if bad <= 3 {
                        println!("n{n} k{k} m{m} ({i},{j}) got={gv} want={acc}");
                    }
                }
            }
            assert_eq!(bad, 0, "n{n} k{k} m{m}: {bad}/64 超容差");
        }
    }
}
