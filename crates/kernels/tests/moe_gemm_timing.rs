//! B40 微基准（loop kaon3-decode-opt D180，nsys 外）：MoE grouped NVFP4 GEMM 在 decode 真实形状下的带宽效率。
//! 形状 = Kaon-V3：E=128，gate‖up K=2816/N=704，down K=704/N=2816；M=1（8 专家各 1 行）与 M=8（28 专家 64 行，
//! 行数分布照 expstats2：max 8 行/专家）。对比：dual pipe4<BM16>（默认）、`moe_grouped_gemm` 两次、B7 `moe_fp4_gateup_act` / `moe_fp4_gemv`。
//! 环境变量（static once，一进程一档）：MOE_DUAL_PIPE4 / MOE_GEMM_STAGES / MOE_GEMM_BM16 —— 由脚本多进程扫。
//! cargo test --release -p dinfer-kernels --test moe_gemm_timing -- --ignored --nocapture
use dinfer_kernels as k;
use k::{bf16, Device};

fn lcg(seed: &mut u64) -> u8 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*seed >> 33) as u8
}

#[test]
#[ignore]
fn moe_gemm_timing() {
    let dev = Device::new(0).expect("cuda");
    let st = dev.default_stream();
    let (e, n_gu, k_gu, n_dn, k_dn) = (128usize, 704usize, 2816usize, 2816usize, 704usize);
    let mut seed = 0x5eed_0b40u64;
    // per-expert 权重：gate / up（[704, 2816]）、down（[2816, 704]）
    let mkw = |seed: &mut u64, n: usize, kk: usize| {
        let mut bb = Vec::new(); let mut ss = Vec::new();
        for _ in 0..e {
            bb.push((0..n * (kk / 2)).map(|_| lcg(seed)).collect::<Vec<u8>>());
            ss.push((0..n * (kk / 16)).map(|_| 0x38u8.wrapping_add(lcg(seed) & 0x0f)).collect::<Vec<u8>>());
        }
        (bb, ss)
    };
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
    let (bg, sg) = mkw(&mut seed, n_gu, k_gu);
    let (bu, su) = mkw(&mut seed, n_gu, k_gu);
    let (bd, sd) = mkw(&mut seed, n_dn, k_dn);
    let (_g1, _g2, pbg, psg) = upload(&bg, &sg);
    let (_u1, _u2, pbu, psu) = upload(&bu, &su);
    let (_d1, _d2, pbd, psd) = upload(&bd, &sd);
    let alpha: Vec<f32> = (0..e).map(|i| 1.0 + 0.001 * i as f32).collect();
    let dal = st.htod_copy(alpha).unwrap();
    let per_expert_gu = 2.0 * (n_gu * k_gu / 2 + n_gu * k_gu / 16) as f64;   // gate+up 字节
    let per_expert_dn = (n_dn * k_dn / 2 + n_dn * k_dn / 16) as f64;
    let iters = 200usize;
    let time = |f: &mut dyn FnMut()| -> f64 {
        for _ in 0..20 { f(); }
        st.synchronize().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..iters { f(); }
        st.synchronize().unwrap();
        t0.elapsed().as_secs_f64() * 1e6 / iters as f64
    };
    // 两个场景：M=1（8 专家 ×1 行）、M=8（28 专家 64 行：4×5 + 4×4 + 8×2 + 12×1）
    let scenarios: Vec<(&str, Vec<usize>)> = vec![
        ("M=1", { let mut c = vec![0usize; e]; for i in 0..8 { c[(i * 17 + 3) % e] = 1; } c }),
        ("M=8", { let mut c = vec![0usize; e]; let mut lens = vec![5usize; 4]; lens.extend(vec![4; 4]); lens.extend(vec![2; 8]); lens.extend(vec![1; 12]);
                  for (i, l) in lens.iter().enumerate() { c[(i * 13 + 7) % e] = *l; } c }),
    ];
    // B48：prefill 混批形状（c16 一步 2148 tok × top-8 = 17184 行，全部 128 专家 ~134 行/专家，照 nsys 态 424 tok 片段也给一档）
    let mut scenarios = scenarios;
    for (name, tok) in [("P424", 424usize), ("P2148", 2148usize)] {
        let rows = tok * 8; let base = rows / e; let rem = rows % e;
        scenarios.push((name, (0..e).map(|i| base + if i < rem { 1 } else { 0 }).collect()));
    }
    for (name, counts) in &scenarios {
        let active = counts.iter().filter(|&&c| c > 0).count();
        let rows: usize = counts.iter().sum();
        let mut offsets = vec![0i32; e + 1];
        for i in 0..e { offsets[i + 1] = offsets[i] + counts[i] as i32; }
        let doff = st.htod_copy(offsets.clone()).unwrap();
        let dtoff = st.htod_copy(k::ops::moe_tile_off_cpu(&offsets)).unwrap();
        let aq_gu: Vec<u8> = (0..rows * (k_gu / 2)).map(|_| lcg(&mut seed)).collect();
        let as_gu: Vec<u8> = (0..rows * (k_gu / 16)).map(|_| 0x38u8.wrapping_add(lcg(&mut seed) & 0x0f)).collect();
        let aq_dn: Vec<u8> = (0..rows * (k_dn / 2)).map(|_| lcg(&mut seed)).collect();
        let as_dn: Vec<u8> = (0..rows * (k_dn / 16)).map(|_| 0x38u8.wrapping_add(lcg(&mut seed) & 0x0f)).collect();
        let (daq_gu, das_gu) = (st.htod_copy(aq_gu).unwrap(), st.htod_copy(as_gu).unwrap());
        let (daq_dn, das_dn) = (st.htod_copy(aq_dn).unwrap(), st.htod_copy(as_dn).unwrap());
        let mut og = st.alloc_zeros::<bf16>(rows * n_gu).unwrap();
        let mut ou = st.alloc_zeros::<bf16>(rows * n_gu).unwrap();
        let mut od = st.alloc_zeros::<bf16>(rows * n_dn).unwrap();
        let bytes_gu = active as f64 * per_expert_gu;
        let bytes_dn = active as f64 * per_expert_dn;
        let gbs = |bytes: f64, us: f64| bytes / us / 1e3;
        let us = time(&mut || { k::ops::moe_grouped_gemm_dual(&st, &daq_gu, &das_gu, &pbg, &psg, &dal, &mut og, &pbu, &psu, &dal, &mut ou, &doff, &dtoff, rows, n_gu, k_gu, e).unwrap(); });
        eprintln!("MOE_GEMM_TIMING {name} active={active} rows={rows} gate‖up dual:      {us:7.2} us  {:6.0} GB/s  (DRAM 下限 {:.2} us @1.79TB/s)", gbs(bytes_gu, us), bytes_gu / 1.79e6);
        let us = time(&mut || {
            k::ops::moe_grouped_gemm(&st, &daq_gu, &das_gu, &pbg, &psg, &dal, &doff, &dtoff, &mut og, rows, n_gu, k_gu, e).unwrap();
            k::ops::moe_grouped_gemm(&st, &daq_gu, &das_gu, &pbu, &psu, &dal, &doff, &dtoff, &mut ou, rows, n_gu, k_gu, e).unwrap();
        });
        eprintln!("MOE_GEMM_TIMING {name} active={active} rows={rows} gate‖up 2×grouped: {us:7.2} us  {:6.0} GB/s", gbs(bytes_gu, us));
        // B48 对拍口径：输出校验和（跨进程 MOE_PREFILL_P64=0/1 两次跑，同形状校验和必须逐位同）
        let hg: Vec<bf16> = st.dtoh_sync_copy(&og).unwrap();
        let hu: Vec<bf16> = st.dtoh_sync_copy(&ou).unwrap();
        let ck = |v: &[bf16]| v.iter().fold(0u64, |a, x| a.wrapping_mul(1099511628211).wrapping_add(x.to_bits() as u64));
        eprintln!("MOE_GEMM_TIMING {name} rows={rows} CHECKSUM gate={:016x} up={:016x}", ck(&hg), ck(&hu));
        let us = time(&mut || { k::ops::moe_grouped_gemm(&st, &daq_dn, &das_dn, &pbd, &psd, &dal, &doff, &dtoff, &mut od, rows, n_dn, k_dn, e).unwrap(); });
        eprintln!("MOE_GEMM_TIMING {name} active={active} rows={rows} down grouped:      {us:7.2} us  {:6.0} GB/s  (DRAM 下限 {:.2} us)", gbs(bytes_dn, us), bytes_dn / 1.79e6);
        let mut oact = st.alloc_zeros::<bf16>(rows * n_gu).unwrap();
        let us = time(&mut || { k::ops::moe_fp4_gateup_act(&st, &daq_gu, &das_gu, &pbg, &psg, &dal, &pbu, &psu, &dal, &doff, &mut oact, rows, n_gu, k_gu, e).unwrap(); });
        eprintln!("MOE_GEMM_TIMING {name} active={active} rows={rows} B7 gateup_act:     {us:7.2} us  {:6.0} GB/s", gbs(bytes_gu, us));
        let us = time(&mut || { k::ops::moe_fp4_gemv(&st, &daq_dn, &das_dn, &pbd, &psd, &dal, &doff, &mut od, rows, n_dn, k_dn, e).unwrap(); });
        eprintln!("MOE_GEMM_TIMING {name} active={active} rows={rows} B7 down gemv:      {us:7.2} us  {:6.0} GB/s", gbs(bytes_dn, us));
    }
}
