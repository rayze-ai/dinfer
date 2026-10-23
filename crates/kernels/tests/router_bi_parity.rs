//! `router_gemm_bi_f32` vs vLLM `matmul_persistent` 的**逐位**对拍（sm_120）。
//!
//! 数据由 `scripts/dev/router_bi_dump.py` 在 vLLM 容器里产生（同一份 x/w，参照输出存 bin）。
//! 判定：逐位相同（0 差异）才算复刻成功；有差异就打印前几个位置方便定位。
//! 运行：`ROUTER_BI_DIR=/w/tmp/rbi CUDA_ARCHS=120 cargo test --release -p dinfer-kernels --test router_bi_parity -- --ignored --nocapture`
use dinfer_kernels::{bf16, Device};

fn rd_u16(p: &str) -> Vec<u16> {
    let b = std::fs::read(p).expect("读 bf16 bin");
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}
fn bf16s(v: &[u16]) -> Vec<bf16> {
    v.iter().map(|&u| bf16::from_bits(u)).collect()
}

#[test]
#[ignore]
fn router_bi_matches_vllm_bitwise() {
    let dir = std::env::var("ROUTER_BI_DIR").expect("设 ROUTER_BI_DIR");
    let meta = std::fs::read_to_string(format!("{dir}/meta.txt")).expect("meta.txt");
    let mut m = 0usize;
    let mut k = 0usize;
    let mut n = 0usize;
    for l in meta.lines() {
        let mut it = l.split('=');
        match (it.next(), it.next()) {
            (Some("M"), Some(v)) => m = v.trim().parse().unwrap(),
            (Some("K"), Some(v)) => k = v.trim().parse().unwrap(),
            (Some("N"), Some(v)) => n = v.trim().parse().unwrap(),
            _ => {}
        }
    }
    let x = bf16s(&rd_u16(&format!("{dir}/x.bin")));
    let w = bf16s(&rd_u16(&format!("{dir}/w.bin")));
    assert_eq!(x.len(), m * k);
    assert_eq!(w.len(), n * k);

    let dev = Device::new(0).expect("cuda");
    let st = dev.default_stream();
    let dx = st.htod_copy(x.clone()).unwrap();
    let dw = st.htod_copy(w.clone()).unwrap();
    let mut dout = st.alloc_zeros::<f32>(m * n).unwrap();
    dinfer_kernels::ops::router_gemm_bi_f32(&st, &dx, &dw, &mut dout, m, n, k).expect("kernel");
    let ours = st.dtoh_sync_copy(&dout).unwrap();

    let refb = std::fs::read(format!("{dir}/ref_f32.bin")).expect("ref");
    let rf: Vec<f32> = refb.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(rf.len(), m * n);

    let mut ndiff = 0usize;
    let mut maxd = 0f32;
    let mut first = Vec::new();
    for i in 0..m * n {
        if ours[i].to_bits() != rf[i].to_bits() {
            ndiff += 1;
            let d = (ours[i] - rf[i]).abs();
            if d > maxd {
                maxd = d;
            }
            if first.len() < 8 {
                first.push((i, ours[i], rf[i]));
            }
        }
    }
    println!("ROUTER_BI [M={m} K={k} N={n}] ndiff={ndiff}/{} max|d|={maxd:.3e}", m * n);
    for (i, a, b) in &first {
        println!("   [{i}] ours={a:.6e} ref={b:.6e}");
    }
    assert_eq!(ndiff, 0, "router BI matmul 与参照不是逐位相同");
}
