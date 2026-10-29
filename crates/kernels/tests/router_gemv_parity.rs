//! R1（loop kaon3-decode-opt D171）：router 小 M bf16 gemv 对拍。
//! 判据：① 与 CPU fp32 参考（双精度累加后 bf16 化）比，|d| ≤ 1 bf16 ulp（fp32 累加序不同导致的最后一位差）；
//! ② **批不变**：同一行在 M=1 与 M=8 批里结果逐位同（kernel 累加序与 M/grid 无关）；③ M=1..16 全覆盖。
//! 跑法：cargo test -p dinfer-kernels --test router_gemv_parity -- --ignored --nocapture
use dinfer_kernels as k;
use k::{bf16, ops, Device};

fn lcg(seed: &mut u64) -> u32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*seed >> 33) as u32
}
fn rb(seed: &mut u64, scale: f32) -> bf16 {
    bf16::from_f32((lcg(seed) as f32 / u32::MAX as f32 - 0.5) * scale)
}

#[test]
#[ignore]
fn router_gemv_matches_reference_and_is_batch_invariant() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let (e, h) = (128usize, 2816usize);
    let mut seed = 0x5eed_u64;
    let w: Vec<bf16> = (0..e * h).map(|_| rb(&mut seed, 0.1)).collect();
    let x: Vec<bf16> = (0..16 * h).map(|_| rb(&mut seed, 2.0)).collect();
    let dw = st.htod_copy(w.clone()).unwrap();
    let dx = st.htod_copy(x.clone()).unwrap();
    // CPU 参考：f64 累加 → f32 → bf16
    let reference = |t: usize, ei: usize| -> bf16 {
        let mut acc = 0f64;
        for d in 0..h { acc += x[t * h + d].to_f32() as f64 * w[ei * h + d].to_f32() as f64; }
        bf16::from_f32(acc as f32)
    };
    let mut m1_rows: Vec<Vec<bf16>> = Vec::new();
    let mut worst_ulp = 0i32;
    let mut bad = 0usize;
    for m in 1..=16usize {
        let mut out = st.alloc_zeros::<bf16>(m * e).unwrap();
        ops::router_gemv_bf16(&st, &dx, &dw, &mut out, m, e, h).unwrap();
        st.synchronize().unwrap();
        let o = st.dtoh_sync_copy(&out).unwrap();
        for t in 0..m {
            for ei in 0..e {
                let got = o[t * e + ei];
                let want = reference(t, ei);
                let ulp = (got.to_bits() as i32 - want.to_bits() as i32).abs();
                worst_ulp = worst_ulp.max(ulp);
                if ulp > 1 { bad += 1; if bad < 5 { eprintln!("m={m} t={t} e={ei} got={} want={}", got.to_f32(), want.to_f32()); } }
            }
        }
        if m == 1 { m1_rows.push(o.clone()); }
        // 批不变：第 0 行在任意 M 下与 M=1 的结果逐位同
        let same = o[..e].iter().zip(&m1_rows[0][..e]).all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(same, "批不变失败：M={m} 的第 0 行与 M=1 不同");
    }
    eprintln!("router_gemv: worst |ulp|={worst_ulp} bad(>1 ulp)={bad}");
    assert_eq!(bad, 0, "router_gemv 与参考差 > 1 ulp 的元素数 = {bad}");
}
