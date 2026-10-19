//! decode 小 M GEMM 微基准（docs/loops/qwen35-perf.md 第 3 轮）：Qwen3.5-27B 每层形状在 M=1（GEMM_BENCH_M 可改）下
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
    println!("GPU cc={:?} sm={} arch={:?} split candidates(wave targets)={:?}", p.cc, p.sm_count, p.arch, p.gemm_tune.small_m_wave_targets);
    let m: usize = std::env::var("GEMM_BENCH_M").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let hid = 5120usize;
    let mut shapes: Vec<(&str, usize, usize)> = vec![
        ("gdn.in_proj_qkv", 10240, hid),
        ("gdn.in_proj_z", 6144, hid),
        ("gdn.in_proj_a", 48, hid),
        ("gdn.out_proj", hid, 6144),
        ("attn.qkv", 8192, hid),
        ("attn.gate", 6144, hid),
        ("attn.o_proj", hid, 6144),
        ("mlp.gate_up", 34816, hid),
        ("mlp.down", hid, 17408),
    ];
    if std::env::var("GEMM_BENCH_SKIP_LM_HEAD").is_err() {
        shapes.push(("lm_head", 248320, hid));
    }
    let k_max = shapes.iter().map(|x| x.2).max().unwrap();
    let n_max = shapes.iter().map(|x| x.1).max().unwrap();
    let a = s.alloc_zeros::<bf16>(m * k_max).unwrap();
    let mut out = s.alloc_zeros::<bf16>(m * n_max).unwrap();
    let mut partial = s.alloc_zeros::<f32>(p.gemm_tune.split_workspace_max_bytes / 4).unwrap();
    println!("{:<18} {:>6} {:>6} | {:>9} {:>7} | {:>9} {:>7} {:<28} | {:>9} {:>7} {:>5} | chosen", "shape(m={m})", "N", "K", "lt_def us", "GB/s", "lt_best us", "GB/s", "algo", "small_m us", "GB/s", "split");
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
        let row = g.tune_shape(name, m, *n, *k, &dw, None, &a, &mut out, &mut partial).unwrap();
        let (sm_us, sm_gb) = match row.small_m_us {
            Some(u) => (format!("{u:9.1}"), format!("{:7.0}", row.gbps(u))),
            None => ("        -".into(), "      -".into()),
        };
        let chosen = match row.chosen {
            GemmPlan::LtAlgo(_) => "lt".to_string(),
            GemmPlan::SmallM { splits } => format!("small_m(s{splits})"),
        };
        println!(
            "{:<18} {:>6} {:>6} | {:9.1} {:7.0} | {:9.1} {:7.0} {:<28} | {} {} {:>5} | {}",
            name, n, k, row.lt_default_us, row.gbps(row.lt_default_us), row.lt_best_us, row.gbps(row.lt_best_us), row.lt_best_desc, sm_us, sm_gb, row.small_m_splits, chosen
        );
        drop(dw);
    }
}

/// 自调优后 forward_bf16 走表（Lt 指定 algo / small_m split）仍与 CPU 参考一致；覆盖桶 M、非桶 M（3）、超小 M 上限的 M（16）。
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
        for &m in &[1usize, 3, 8, 16] {
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
