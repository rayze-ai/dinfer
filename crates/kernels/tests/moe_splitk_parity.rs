//! B44 split-K（loop kaon3-decode-opt D190）门禁：`moe_grouped_gemm_dual → moe_gelu_grouped_quant` vs
//! `moe_grouped_gemm_dual_splitk(ksplit) → moe_gelu_grouped_quant_splitk`。
//! ksplit=1：逐位同（部分和 ×alpha 再 bf16 与原 epilogue 同式）；ksplit=2/4：fp32 加法序不同，aq 码差 ≤ 1 且比例 ≤ 2%、a_s 同。
//! 形状 = Kaon decode：E=128、gate/up K=2816 N=704、rows=22（8 专家 1–8 行）。
//! cargo test --release -p dinfer-kernels --test moe_splitk_parity -- --ignored --nocapture
use dinfer_kernels as k;
use k::{bf16, ops, Device};

fn lcg(seed: &mut u64) -> u8 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*seed >> 33) as u8
}

#[test]
#[ignore]
fn moe_splitk_parity() {
    let dev = Device::new(0).expect("cuda");
    let st = dev.default_stream();
    let (e, n, kk) = (128usize, 704usize, 2816usize);
    let lens = [1usize, 1, 2, 3, 1, 5, 1, 8];
    let rows: usize = lens.iter().sum();
    let mut offsets = vec![0i32; e + 1];
    for (i, l) in lens.iter().enumerate() { let ex = (i * 13 + 7) % e; offsets[ex + 1] = *l as i32; }
    for i in 0..e { offsets[i + 1] += offsets[i]; }
    let doff = st.htod_copy(offsets.clone()).unwrap();
    let dtoff = st.htod_copy(ops::moe_tile_off_cpu(&offsets)).unwrap();
    let mut seed = 0xb44_5011u64 ^ 0x9e3779b97f4a7c15;
    let aq: Vec<u8> = (0..rows * (kk / 2)).map(|_| lcg(&mut seed)).collect();
    let a_s: Vec<u8> = (0..rows * (kk / 16)).map(|_| 0x38u8.wrapping_add(lcg(&mut seed) & 0x0f)).collect();
    let upload = |seed: &mut u64| {
        let mut db = Vec::new(); let mut ds = Vec::new();
        let (mut bp, mut sp) = (Vec::<u64>::new(), Vec::<u64>::new());
        for _ in 0..e {
            let b = st.htod_copy((0..n * (kk / 2)).map(|_| lcg(seed)).collect::<Vec<u8>>()).unwrap();
            let s = st.htod_copy((0..n * (kk / 16)).map(|_| 0x38u8.wrapping_add(lcg(seed) & 0x0f)).collect::<Vec<u8>>()).unwrap();
            bp.push(st.ptr(&b) as u64); sp.push(st.ptr(&s) as u64);
            db.push(b); ds.push(s);
        }
        (db, ds, st.htod_copy(bp).unwrap(), st.htod_copy(sp).unwrap())
    };
    let (_g1, _g2, pbg, psg) = upload(&mut seed);
    let (_u1, _u2, pbu, psu) = upload(&mut seed);
    let alpha_g: Vec<f32> = (0..e).map(|i| 0.8 + 0.003 * i as f32).collect();
    let alpha_u: Vec<f32> = (0..e).map(|i| 1.1 - 0.002 * i as f32).collect();
    let dalg = st.htod_copy(alpha_g).unwrap();
    let dalu = st.htod_copy(alpha_u).unwrap();
    let row2e: Vec<i32> = (0..e).flat_map(|ex| std::iter::repeat(ex as i32).take((offsets[ex + 1] - offsets[ex]) as usize)).collect();
    assert_eq!(row2e.len(), rows);
    let dr2e = st.htod_copy(row2e).unwrap();
    let agscale: Vec<f32> = (0..e).map(|i| 0.5 + 0.004 * i as f32).collect();
    let dag = st.htod_copy(agscale).unwrap();
    let daq = st.htod_copy(aq).unwrap();
    let das = st.htod_copy(a_s).unwrap();
    // 基线：dual → gelu_grouped_quant（K_down = n = 704）
    let mut og = st.alloc_zeros::<bf16>(rows * n).unwrap();
    let mut ou = st.alloc_zeros::<bf16>(rows * n).unwrap();
    ops::moe_grouped_gemm_dual(&st, &daq, &das, &pbg, &psg, &dalg, &mut og, &pbu, &psu, &dalu, &mut ou, &doff, &dtoff, rows, n, kk, e).unwrap();
    let mut aq0 = st.alloc_zeros::<u8>(rows * n / 2).unwrap(); let mut as0 = st.alloc_zeros::<u8>(rows * n / 16).unwrap();
    ops::moe_gelu_grouped_quant(&st, &og, &ou, &dr2e, &dag, &mut aq0, &mut as0, rows, n, false).unwrap();
    st.synchronize().unwrap();
    let (a0, s0) = (st.dtoh_sync_copy(&aq0).unwrap(), st.dtoh_sync_copy(&as0).unwrap());
    let mut fails = Vec::new();
    for ksplit in [1usize, 2, 4] {
        let mut part = st.alloc_zeros::<f32>(2 * ksplit * rows * n).unwrap();
        ops::moe_grouped_gemm_dual_splitk(&st, &daq, &das, &pbg, &psg, &dalg, &pbu, &psu, &dalu, &doff, &dtoff, &mut part, rows, n, kk, e, ksplit).unwrap();
        let mut aq1 = st.alloc_zeros::<u8>(rows * n / 2).unwrap(); let mut as1 = st.alloc_zeros::<u8>(rows * n / 16).unwrap();
        ops::moe_gelu_grouped_quant_splitk(&st, &part, ksplit, &dalg, &dalu, &dr2e, &dag, &mut aq1, &mut as1, rows, n, false).unwrap();
        st.synchronize().unwrap();
        let (a1, s1) = (st.dtoh_sync_copy(&aq1).unwrap(), st.dtoh_sync_copy(&as1).unwrap());
        // fp4 码差：按半字节比
        let mut nd = 0usize; let mut worst = 0i32;
        for (x, y) in a0.iter().zip(&a1) {
            for sh in [0u8, 4] {
                let (p, q) = (((x >> sh) & 0xf) as i32, ((y >> sh) & 0xf) as i32);
                if p != q { nd += 1; worst = worst.max((p - q).abs()); }
            }
        }
        let ns = s0.iter().zip(&s1).filter(|(p, q)| p != q).count();
        let total = a0.len() * 2;
        let frac = nd as f64 / total as f64;
        let ok = if ksplit == 1 { nd == 0 && ns == 0 } else { frac <= 0.02 && ns * 50 <= s0.len() && worst <= 1 };
        println!("MOE_SPLITK_PARITY ksplit={ksplit}: fp4 码差 {nd}/{total} ({:.3}%) worst={worst} | a_s 差 {ns}/{} => {}", frac * 100.0, s0.len(), if ok { "ok" } else { "FAIL" });
        if !ok { fails.push(ksplit); }
    }
    assert!(fails.is_empty(), "失败 ksplit: {fails:?}");
}
