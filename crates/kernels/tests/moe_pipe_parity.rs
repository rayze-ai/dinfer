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
    let dtoff = stream.htod_copy(k::ops::moe_tile_off_cpu(offsets)).unwrap();
    let mut dout = stream.alloc_zeros::<bf16>(rows * n).unwrap();
    k::ops::moe_grouped_gemm(stream, &daq, &das, &dbp, &dsp, &dal, &doff, &dtoff, &mut dout, rows, n, kk, e).unwrap();
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

/// B16：STAGES=4 环形流水 vs 旧版（无流水）逐位。注意 host 的 MOE_GEMM_STAGES 是进程内 static 一次读取，
/// 本测试单独进程跑：`cargo test -p dinfer-kernels --test moe_pipe_parity -- --ignored --exact moe_pipe4_matches_baseline`
/// （与其它测试同进程时以第一个读到的值为准）。
#[test]
#[ignore]
fn moe_pipe4_matches_baseline() {
    std::env::set_var("MOE_GEMM_STAGES", "4");
    let (e, n, kk) = (8usize, 704usize, 2816usize);
    let lens = [1usize, 1, 2, 3, 1, 5, 1, 8];
    let rows: usize = lens.iter().sum();
    let mut offsets = vec![0i32; e + 1];
    for i in 0..e { offsets[i + 1] = offsets[i] + lens[i] as i32; }
    let mut seed = 0x0b16_5678_9abc_def0u64;
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
    let base = run(&stream, false, &aq, &a_s, &bbytes, &sbytes, &alpha, &offsets, rows, n, kk, e);
    let p4 = run(&stream, true, &aq, &a_s, &bbytes, &sbytes, &alpha, &offsets, rows, n, kk, e);
    let bad = base.iter().zip(&p4).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    println!("MOE_PIPE4_PARITY rows={rows} n={n} k={kk} e={e} STAGES=4 | bad={bad}/{}", base.len());
    assert_eq!(bad, 0, "4 级流水与旧版逐位不一致（应 bad=0）");
    // down 形状（K=704, BK=64 路径）
    let (n2, kk2) = (2816usize, 704usize);
    let aq2: Vec<u8> = (0..rows * (kk2 / 2)).map(|_| lcg(&mut seed)).collect();
    let a_s2: Vec<u8> = (0..rows * (kk2 / 16)).map(|_| 0x38u8.wrapping_add(lcg(&mut seed) & 0x0f)).collect();
    let mut bb2 = Vec::new(); let mut sb2 = Vec::new();
    for _ in 0..e {
        bb2.push((0..n2 * (kk2 / 2)).map(|_| lcg(&mut seed)).collect::<Vec<u8>>());
        sb2.push((0..n2 * (kk2 / 16)).map(|_| 0x38u8.wrapping_add(lcg(&mut seed) & 0x0f)).collect::<Vec<u8>>());
    }
    let base2 = run(&stream, false, &aq2, &a_s2, &bb2, &sb2, &alpha, &offsets, rows, n2, kk2, e);
    let p42 = run(&stream, true, &aq2, &a_s2, &bb2, &sb2, &alpha, &offsets, rows, n2, kk2, e);
    let bad2 = base2.iter().zip(&p42).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    println!("MOE_PIPE4_PARITY(down) rows={rows} n={n2} k={kk2} e={e} STAGES=4 | bad={bad2}/{}", base2.len());
    assert_eq!(bad2, 0, "4 级流水(down 形状)与旧版逐位不一致");
}
// 计时微基准：8 专家各 M=132（模拟 prefill 大 M），旧版 vs 流水版 kernel 墙钟。
// cargo test -p dinfer-kernels --test moe_pipe_parity -- --ignored --nocapture moe_pipe_timing
#[test]
#[ignore]
fn moe_pipe_timing() {
    // E=专家数(grid.z), ACT=激活专家数(前 ACT 个各 m_per 行, 其余 0 行 = serve decode 的空 block 场景)
    let e = std::env::var("MOE_E").ok().and_then(|v| v.parse().ok()).unwrap_or(8usize);
    let (n, kk) = (704usize, 2816usize);
    let m_per = std::env::var("MOE_M").ok().and_then(|v| v.parse().ok()).unwrap_or(132usize);
    let act = std::env::var("MOE_ACT").ok().and_then(|v| v.parse().ok()).unwrap_or(e).min(e);
    let lens: Vec<usize> = (0..e).map(|i| if i < act { m_per } else { 0 }).collect();
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
    let dtoff = stream.htod_copy(k::ops::moe_tile_off_cpu(&offsets)).unwrap();
    let doff = stream.htod_copy(offsets).unwrap();
    let mut dout = stream.alloc_zeros::<bf16>(rows * n).unwrap();

    let iters = 200usize;
    let mut bench = |pipe: bool| -> f64 {
        if pipe { std::env::set_var("MOE_GEMM_PIPE", "1"); } else { std::env::remove_var("MOE_GEMM_PIPE"); }
        for _ in 0..20 { k::ops::moe_grouped_gemm(&stream,&daq,&das,&dbp,&dsp,&dal,&doff,&dtoff,&mut dout,rows,n,kk,e).unwrap(); }
        stream.synchronize().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..iters { k::ops::moe_grouped_gemm(&stream,&daq,&das,&dbp,&dsp,&dal,&doff,&dtoff,&mut dout,rows,n,kk,e).unwrap(); }
        stream.synchronize().unwrap();
        t0.elapsed().as_secs_f64() * 1e6 / iters as f64
    };
    let base_us = bench(false);
    let pipe_us = bench(true);
    // 有效带宽按激活专家算（空 block 不搬权重）：act × N × K/2 字节。
    let wbytes = (act * n * (kk / 2)) as f64;
    let gbps = |us: f64| wbytes / (us * 1e-6) / 1e9;
    println!("MOE_PIPE_TIMING E={e}(grid.z) ACT={act} M/专家={m_per} N={n} K={kk} rows={rows} | base={base_us:.2}us ({:.0}GB/s) pipe={pipe_us:.2}us ({:.0}GB/s) | 加速 {:.2}x",
             gbps(base_us), gbps(pipe_us), base_us / pipe_us);
    drop(db); drop(ds);
}

// 第二刀 dual 逐位对拍:分开两次 moe_grouped_gemm(gate,up) vs 一次 moe_grouped_gemm_dual,断言两输出各逐字节相等。
// cargo test -p dinfer-kernels --test moe_pipe_parity -- --ignored --nocapture moe_dual_matches
#[test]
#[ignore]
fn moe_dual_matches_separate() {
    let (e, n, kk) = (8usize, 704usize, 2816usize);
    let lens = [1usize, 1, 2, 3, 1, 5, 1, 8];
    let rows: usize = lens.iter().sum();
    let mut offsets = vec![0i32; e + 1];
    for i in 0..e { offsets[i + 1] = offsets[i] + lens[i] as i32; }
    let mut seed = 0xabcd_1234_5678_9999u64;
    let aq: Vec<u8> = (0..rows * (kk / 2)).map(|_| lcg(&mut seed)).collect();
    let a_s: Vec<u8> = (0..rows * (kk / 16)).map(|_| 0x38u8.wrapping_add(lcg(&mut seed) & 0x0f)).collect();
    // gate / up 各一套 per-expert 权重
    let mkw = |seed: &mut u64| {
        let mut bb = Vec::new(); let mut ss = Vec::new();
        for _ in 0..e {
            bb.push((0..n * (kk / 2)).map(|_| lcg(seed)).collect::<Vec<u8>>());
            ss.push((0..n * (kk / 16)).map(|_| 0x38u8.wrapping_add(lcg(seed) & 0x0f)).collect::<Vec<u8>>());
        }
        (bb, ss)
    };
    let (bg, sg) = mkw(&mut seed);
    let (bu, su) = mkw(&mut seed);
    let alpha_g: Vec<f32> = (0..e).map(|i| 1.0 + 0.01 * i as f32).collect();
    let alpha_u: Vec<f32> = (0..e).map(|i| 1.0 + 0.02 * i as f32).collect();

    let dev = Device::new(0).expect("cuda");
    let st = dev.default_stream();
    std::env::remove_var("MOE_GEMM_PIPE");
    let daq = st.htod_copy(aq).unwrap();
    let das = st.htod_copy(a_s).unwrap();
    let dtoff = st.htod_copy(k::ops::moe_tile_off_cpu(&offsets)).unwrap();
    let doff = st.htod_copy(offsets).unwrap();
    // 上传 per-expert 权重 + 指针表
    let upload = |bb: &Vec<Vec<u8>>, ss: &Vec<Vec<u8>>| {
        let mut db = Vec::new(); let mut ds = Vec::new();
        let (mut bp, mut sp) = (Vec::<u64>::new(), Vec::<u64>::new());
        for ex in 0..e {
            let b = st.htod_copy(bb[ex].clone()).unwrap();
            let s = st.htod_copy(ss[ex].clone()).unwrap();
            bp.push(st.ptr(&b) as u64); sp.push(st.ptr(&s) as u64);
            db.push(b); ds.push(s);
        }
        (db, ds, st.htod_copy(bp).unwrap(), st.htod_copy(sp).unwrap())
    };
    let (_dbg, _dsg, dbgp, dsgp) = upload(&bg, &sg);
    let (_dbu, _dsu, dbup, dsup) = upload(&bu, &su);
    let dalg = st.htod_copy(alpha_g).unwrap();
    let dalu = st.htod_copy(alpha_u).unwrap();
    // baseline: 分开两次
    let mut og = st.alloc_zeros::<bf16>(rows * n).unwrap();
    let mut ou = st.alloc_zeros::<bf16>(rows * n).unwrap();
    k::ops::moe_grouped_gemm(&st, &daq, &das, &dbgp, &dsgp, &dalg, &doff, &dtoff, &mut og, rows, n, kk, e).unwrap();
    k::ops::moe_grouped_gemm(&st, &daq, &das, &dbup, &dsup, &dalu, &doff, &dtoff, &mut ou, rows, n, kk, e).unwrap();
    st.synchronize().unwrap();
    let bg_out = st.dtoh_sync_copy(&og).unwrap();
    let bu_out = st.dtoh_sync_copy(&ou).unwrap();
    // dual: 一次
    let mut og2 = st.alloc_zeros::<bf16>(rows * n).unwrap();
    let mut ou2 = st.alloc_zeros::<bf16>(rows * n).unwrap();
    k::ops::moe_grouped_gemm_dual(&st, &daq, &das, &dbgp, &dsgp, &dalg, &mut og2, &dbup, &dsup, &dalu, &mut ou2, &doff, &dtoff, rows, n, kk, e).unwrap();
    st.synchronize().unwrap();
    let dg_out = st.dtoh_sync_copy(&og2).unwrap();
    let du_out = st.dtoh_sync_copy(&ou2).unwrap();
    let cnt = |a: &[bf16], b: &[bf16]| a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    let bad_g = cnt(&bg_out, &dg_out);
    let bad_u = cnt(&bu_out, &du_out);
    println!("MOE_DUAL_PARITY rows={rows} n={n} k={kk} e={e} | gate bad={bad_g}/{} up bad={bad_u}/{}", bg_out.len(), bu_out.len());
    assert_eq!(bad_g, 0, "dual gate 与分开不一致");
    assert_eq!(bad_u, 0, "dual up 与分开不一致");
}
