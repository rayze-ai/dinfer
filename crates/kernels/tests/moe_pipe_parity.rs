//! MoE grouped NVFP4 GEMM 流水版逐位对拍（自洽，不读文件/不需 vLLM dump）。
//! 同一进程内：旧版（MOE_GEMM_PIPE 未设）vs 流水版（MOE_GEMM_PIPE=1），同随机输入，断言输出逐字节相等。
//! 跑法：cargo test -p dinfer-kernels --test moe_pipe_parity -- --ignored --nocapture
//! 门禁：bad=0（流水只改 cp.async 拷贝时机，不改落 smem 字节/mma 归约序 → 必须逐位一致）。
use dinfer_kernels as k;
use k::{bf16, Device, Stream};

// 确定性 LCG，产可复现随机字节。
fn lcg(seed: &mut u64) -> u8 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*seed >> 33) as u8
}

fn run(stream: &Stream, pipe: bool, aq: &[u8], a_s: &[u8], bbytes: &[Vec<u8>], sbytes: &[Vec<u8>],
       alpha: &[f32], offsets: &[i32], rows: usize, n: usize, kk: usize, e: usize) -> Vec<bf16> {
    if pipe { std::env::set_var("MOE_GEMM_PIPE", "1"); } else { std::env::remove_var("MOE_GEMM_PIPE"); }
    let daq = stream.htod_copy(aq.to_vec()).unwrap();
    let das = stream.htod_copy(a_s.to_vec()).unwrap();
    // per-expert B/S device buffer + 指针表
    let mut db = Vec::new();
    let mut ds = Vec::new();
    let mut bptr = Vec::<u64>::new();
    let mut sptr = Vec::<u64>::new();
    for ex in 0..e {
        let b = stream.htod_copy(bbytes[ex].clone()).unwrap();
        let s = stream.htod_copy(sbytes[ex].clone()).unwrap();
        bptr.push(stream.ptr(&b) as u64);
        sptr.push(stream.ptr(&s) as u64);
        db.push(b);
        ds.push(s);
    }
    let dbp = stream.htod_copy(bptr).unwrap();
    let dsp = stream.htod_copy(sptr).unwrap();
    let dal = stream.htod_copy(alpha.to_vec()).unwrap();
    let doff = stream.htod_copy(offsets.to_vec()).unwrap();
    let mut dout = stream.alloc_zeros::<bf16>(rows * n).unwrap();
    k::ops::moe_grouped_gemm(stream, &daq, &das, &dbp, &dsp, &dal, &doff, &mut dout, rows, n, kk, e).unwrap();
    stream.synchronize().unwrap();
    let out = stream.dtoh_sync_copy(&dout).unwrap();
    drop(db); drop(ds);
    out
}

#[test]
#[ignore]
fn moe_pipe_matches_baseline() {
    // Kaon 形状量级：E 专家、每专家若干行（模拟 decode 小 M 与一般 M 混合）、N=moe_inter、K。
    let (e, n, kk) = (8usize, 704usize, 2816usize);
    // 各专家行数：1,1,2,3,1,5,1,8 → 测 M=1 与 M>1 都覆盖。
    let lens = [1usize, 1, 2, 3, 1, 5, 1, 8];
    let rows: usize = lens.iter().sum();
    let mut offsets = vec![0i32; e + 1];
    for i in 0..e { offsets[i + 1] = offsets[i] + lens[i] as i32; }

    let mut seed = 0x1234_5678_9abc_def0u64;
    // A 量化：[rows, K/2] nibble + scale [rows, K/16]（e4m3 字节，避开 0/inf 用中间值）
    let aq: Vec<u8> = (0..rows * (kk / 2)).map(|_| lcg(&mut seed)).collect();
    let a_s: Vec<u8> = (0..rows * (kk / 16)).map(|_| 0x38u8.wrapping_add(lcg(&mut seed) & 0x0f)).collect();
    // per-expert B [N, K/2] + S [N, K/16]
    let mut bbytes = Vec::new();
    let mut sbytes = Vec::new();
    for _ in 0..e {
        bbytes.push((0..n * (kk / 2)).map(|_| lcg(&mut seed)).collect::<Vec<u8>>());
        sbytes.push((0..n * (kk / 16)).map(|_| 0x38u8.wrapping_add(lcg(&mut seed) & 0x0f)).collect::<Vec<u8>>());
    }
    let alpha: Vec<f32> = (0..e).map(|i| 1.0 + 0.01 * i as f32).collect();

    let dev = Device::new(0).expect("cuda");
    let stream = dev.default_stream();
    let base = run(&stream, false, &aq, &a_s, &bbytes, &sbytes, &alpha, &offsets, rows, n, kk, e);
    let pipe = run(&stream, true, &aq, &a_s, &bbytes, &sbytes, &alpha, &offsets, rows, n, kk, e);

    let mut bad = 0usize;
    let mut first = None;
    for i in 0..base.len() {
        if base[i].to_bits() != pipe[i].to_bits() {
            bad += 1;
            if first.is_none() { first = Some((i, base[i].to_f32(), pipe[i].to_f32())); }
        }
    }
    println!("MOE_PIPE_PARITY rows={rows} n={n} k={kk} e={e} | bad={bad}/{}", base.len());
    if let Some((i, b, p)) = first { println!("  first diff @ {i}: base={b:.6e} pipe={p:.6e}"); }
    assert_eq!(bad, 0, "流水版与旧版逐位不一致（应 bad=0）");
}

// 计时微基准：8 专家各 M=132（模拟 prefill 大 M），旧版 vs 流水版 kernel 墙钟。
// cargo test -p dinfer-kernels --test moe_pipe_parity -- --ignored --nocapture moe_pipe_timing
#[test]
#[ignore]
fn moe_pipe_timing() {
    let (e, n, kk) = (8usize, 704usize, 2816usize);
    let m_per = std::env::var("MOE_M").ok().and_then(|v| v.parse().ok()).unwrap_or(132usize);
    let lens: Vec<usize> = vec![m_per; e];
    let rows: usize = lens.iter().sum();
    let mut offsets = vec![0i32; e + 1];
    for i in 0..e { offsets[i + 1] = offsets[i] + lens[i] as i32; }

    let mut seed = 0xdead_beef_1234_5678u64;
    let aq: Vec<u8> = (0..rows * (kk / 2)).map(|_| lcg(&mut seed)).collect();
    let a_s: Vec<u8> = (0..rows * (kk / 16)).map(|_| 0x38u8.wrapping_add(lcg(&mut seed) & 0x0f)).collect();
    let mut bbytes = Vec::new();
    let mut sbytes = Vec::new();
    for _ in 0..e {
        bbytes.push((0..n * (kk / 2)).map(|_| lcg(&mut seed)).collect::<Vec<u8>>());
        sbytes.push((0..n * (kk / 16)).map(|_| 0x38u8.wrapping_add(lcg(&mut seed) & 0x0f)).collect::<Vec<u8>>());
    }
    let alpha: Vec<f32> = (0..e).map(|i| 1.0 + 0.01 * i as f32).collect();

    let dev = Device::new(0).expect("cuda");
    let stream = dev.default_stream();
    let daq = stream.htod_copy(aq).unwrap();
    let das = stream.htod_copy(a_s).unwrap();
    let mut db = Vec::new();
    let mut ds = Vec::new();
    let (mut bptr, mut sptr) = (Vec::<u64>::new(), Vec::<u64>::new());
    for ex in 0..e {
        let b = stream.htod_copy(bbytes[ex].clone()).unwrap();
        let s = stream.htod_copy(sbytes[ex].clone()).unwrap();
        bptr.push(stream.ptr(&b) as u64); sptr.push(stream.ptr(&s) as u64);
        db.push(b); ds.push(s);
    }
    let dbp = stream.htod_copy(bptr).unwrap();
    let dsp = stream.htod_copy(sptr).unwrap();
    let dal = stream.htod_copy(alpha).unwrap();
    let doff = stream.htod_copy(offsets).unwrap();
    let mut dout = stream.alloc_zeros::<bf16>(rows * n).unwrap();

    let iters = 200usize;
    let mut bench = |pipe: bool| -> f64 {
        if pipe { std::env::set_var("MOE_GEMM_PIPE", "1"); } else { std::env::remove_var("MOE_GEMM_PIPE"); }
        for _ in 0..20 { k::ops::moe_grouped_gemm(&stream,&daq,&das,&dbp,&dsp,&dal,&doff,&mut dout,rows,n,kk,e).unwrap(); }
        stream.synchronize().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..iters { k::ops::moe_grouped_gemm(&stream,&daq,&das,&dbp,&dsp,&dal,&doff,&mut dout,rows,n,kk,e).unwrap(); }
        stream.synchronize().unwrap();
        t0.elapsed().as_secs_f64() * 1e6 / iters as f64
    };
    let base_us = bench(false);
    let pipe_us = bench(true);
    // 单次 launch = 1 个 grouped GEMM（gate/up/down 中的一个）。有效带宽：e 专家 × N × K/2 字节权重。
    let wbytes = (e * n * (kk / 2)) as f64;
    let gbps = |us: f64| wbytes / (us * 1e-6) / 1e9;
    println!("MOE_PIPE_TIMING e={e} M/专家={m_per} N={n} K={kk} | base={base_us:.2}us ({:.0}GB/s) pipe={pipe_us:.2}us ({:.0}GB/s) | 加速 {:.2}x",
             gbps(base_us), gbps(pipe_us), base_us / pipe_us);
    drop(db); drop(ds);
}
