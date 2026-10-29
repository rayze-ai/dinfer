//! B40 门禁（loop kaon3-decode-opt D181）：pipe4 decode 路径在不同 BK（64/128/256）与旧 body（MOE_GEMM_PIPE=0）下输出逐位同。
//! 启动器开关是进程级 static once，所以本测试只负责"按当前 env 算一次并把输出 dump 到文件"，由脚本起多个进程后 cmp：
//!   MOE_GEMM_PIPE=0 → /tmp/moe_bk_ref.bin；MOE_GEMM_BK=64/128/256 → /tmp/moe_bk_<BK>.bin（文件名由 MOE_BK_TAG 指定）。
//! 形状 = Kaon decode：gate（K=2816,N=704）、down（K=704,N=2816）、dual（gate‖up）；rows=22（8 专家 1–8 行）。
use dinfer_kernels as k;
use k::{bf16, Device};

fn lcg(seed: &mut u64) -> u8 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*seed >> 33) as u8
}

#[test]
#[ignore]
fn moe_bk_dump() {
    let tag = std::env::var("MOE_BK_TAG").unwrap_or_else(|_| "x".into());
    let dev = Device::new(0).expect("cuda");
    let st = dev.default_stream();
    let e = 8usize;
    // D197：`MOE_BK_MULT=n` 把每专家行数 ×n（n=12 → 264 行，越过 128 的 decode 快路径门槛，用来对拍 MOE_DEC_RMAX 两侧）。
    let mult: usize = std::env::var("MOE_BK_MULT").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let lens: Vec<usize> = [1usize, 1, 2, 3, 1, 5, 1, 8].iter().map(|l| l * mult).collect();
    let rows: usize = lens.iter().sum();
    let mut offsets = vec![0i32; e + 1];
    for i in 0..e { offsets[i + 1] = offsets[i] + lens[i] as i32; }
    let doff = st.htod_copy(offsets.clone()).unwrap();
    let dtoff = st.htod_copy(k::ops::moe_tile_off_cpu(&offsets)).unwrap();
    let alpha: Vec<f32> = (0..e).map(|i| 1.0 + 0.01 * i as f32).collect();
    let dal = st.htod_copy(alpha).unwrap();
    let mut seed = 0xb40_0000_5eedu64;
    let mut out_all: Vec<u8> = Vec::new();
    for (n, kk) in [(704usize, 2816usize), (2816, 704)] {
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
        let (_b1, _s1, pb, ps) = upload(&mut seed);
        let (_b2, _s2, pbu, psu) = upload(&mut seed);
        let daq = st.htod_copy(aq).unwrap();
        let das = st.htod_copy(a_s).unwrap();
        let mut o = st.alloc_zeros::<bf16>(rows * n).unwrap();
        k::ops::moe_grouped_gemm(&st, &daq, &das, &pb, &ps, &dal, &doff, &dtoff, &mut o, rows, n, kk, e).unwrap();
        let mut og = st.alloc_zeros::<bf16>(rows * n).unwrap();
        let mut ou = st.alloc_zeros::<bf16>(rows * n).unwrap();
        k::ops::moe_grouped_gemm_dual(&st, &daq, &das, &pb, &ps, &dal, &mut og, &pbu, &psu, &dal, &mut ou, &doff, &dtoff, rows, n, kk, e).unwrap();
        st.synchronize().unwrap();
        for v in [st.dtoh_sync_copy(&o).unwrap(), st.dtoh_sync_copy(&og).unwrap(), st.dtoh_sync_copy(&ou).unwrap()] {
            for x in v { out_all.extend_from_slice(&x.to_bits().to_le_bytes()); }
        }
        // dual 的 gate 半与单路必须同（同进程、同 BK）
        let o1 = st.dtoh_sync_copy(&o).unwrap(); let o2 = st.dtoh_sync_copy(&og).unwrap();
        let bad = o1.iter().zip(&o2).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        eprintln!("MOE_BK_DUMP tag={tag} n={n} k={kk}: dual-vs-single bad={bad}/{}", o1.len());
        assert_eq!(bad, 0);
    }
    let path = format!("/tmp/moe_bk_{tag}.bin");
    std::fs::write(&path, &out_all).unwrap();
    eprintln!("MOE_BK_DUMP tag={tag} wrote {} bytes to {path}", out_all.len());
}
