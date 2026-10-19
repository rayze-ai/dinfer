//! MXFP4 解包对拍。参考由 `scripts/verify/gen_mxfp4_refs.py` 生成（按位构造，不依赖 torch 的 fp4 dtype）。
//!
//! 运行：`MXFP4_REFS_DIR=<dir> cargo test -p dinfer-kernels --features ref-tests --test mxfp4_parity -- --ignored`
#![cfg(feature = "ref-tests")]

use dinfer_kernels::{bf16, Device, Stream};
use std::path::PathBuf;

fn root() -> PathBuf {
    let p = PathBuf::from(std::env::var("MXFP4_REFS_DIR").unwrap_or_else(|_| "test/kimik3/2026-10-19/mxfp4_refs".into()));
    if p.is_absolute() { p } else { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join(p) }
}

fn load_u8(case: &str, name: &str) -> (Vec<usize>, Vec<u8>) {
    let bytes = std::fs::read(root().join(case).join(format!("{name}.npy"))).unwrap();
    let arr = npyz::NpyFile::new(&bytes[..]).unwrap();
    let shape: Vec<usize> = arr.shape().iter().map(|&d| d as usize).collect();
    (shape, arr.into_vec::<u8>().unwrap())
}

fn load_f32(case: &str, name: &str) -> Vec<f32> {
    let bytes = std::fs::read(root().join(case).join(format!("{name}.npy"))).unwrap();
    let arr = npyz::NpyFile::new(&bytes[..]).unwrap();
    arr.into_vec::<f32>().unwrap()
}

#[test]
#[ignore]
fn mxfp4_dequant_all_cases() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let cases = ["moE_w1_small", "moE_w2_small", "group32_odd_rows"];
    for case in cases {
        let (ps, packed) = load_u8(case, "packed");
        let (ss, scale) = load_u8(case, "scale");
        let want = load_f32(case, "out");
        let rows = ps[0];
        let cols = ps[1] * 2;
        let group = cols / ss[1];
        let dp = st.htod_copy(packed).unwrap();
        let ds = st.htod_copy(scale).unwrap();
        let mut dout = st.alloc_uninit::<bf16>(rows * cols).unwrap();
        dinfer_kernels::ops::mxfp4_dequant_bf16(&st, &dp, &ds, &mut dout, rows, cols, group).unwrap();
        st.synchronize().unwrap();
        let got = st.dtoh_sync_copy(&dout).unwrap();
        assert_eq!(got.len(), want.len(), "{case}: 长度");
        let mut max_err = 0f32;
        for (g, w) in got.iter().zip(&want) {
            let e = (g.to_f32() - w).abs();
            let rel = e / w.abs().max(1e-6);
            assert!(rel <= 1e-3 || e <= 1e-6, "{case}: {} vs {w}", g.to_f32());
            max_err = max_err.max(e);
        }
        println!("  {case:<20} rows={rows} cols={cols} group={group} max_err {max_err:.3e}");
    }
}
