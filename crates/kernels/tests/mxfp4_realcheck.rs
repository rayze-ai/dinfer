//! 真权重 MXFP4 解包核对：与 `scripts/verify/k3_mxfp4_realcheck.py` 产出的 numpy 参考**逐位**比。
//!
//! 脚本（模型相关：读哪个仓库、哪个张量）在 `scripts/verify/` 里；本测试只按目录约定读，
//! 因此 `crates/` 不出现任何模型信息：
//!
//!   <dir>/manifest.txt   每行 `name rows cols group`
//!   <dir>/<name>.packed  u8  [rows, cols/2]
//!   <dir>/<name>.scale   u8  [rows, cols/group]
//!   <dir>/<name>.ref     f32 [rows, cols]   numpy 侧解包值
//!
//! 解包是确定性的（表查 + 指数移位 + 一次 bf16 舍入），所以断言 **bit-exact**。
//! 环境变量 `MXFP4_REALCHECK_DIR` 未设置或目录不存在时 skip（远端 / 无权重机器上不失败）。
//!
//! 运行：`MXFP4_REALCHECK_DIR=/tmp/mxfp4_real cargo test -p dinfer-kernels --test mxfp4_realcheck -- --ignored --nocapture`

use dinfer_kernels::{bf16, Device};
use std::path::{Path, PathBuf};

fn dir() -> Option<PathBuf> {
    let d = PathBuf::from(std::env::var("MXFP4_REALCHECK_DIR").ok()?);
    if d.join("manifest.txt").is_file() { Some(d) } else { None }
}

struct Case {
    name: String,
    rows: usize,
    cols: usize,
    group: usize,
}

fn manifest(root: &Path) -> Vec<Case> {
    let text = std::fs::read_to_string(root.join("manifest.txt")).expect("manifest.txt");
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            assert_eq!(f.len(), 4, "manifest 行格式：`name rows cols group`，得到 {l:?}");
            Case {
                name: f[0].to_string(),
                rows: f[1].parse().unwrap(),
                cols: f[2].parse().unwrap(),
                group: f[3].parse().unwrap(),
            }
        })
        .collect()
}

fn read_u8(p: &Path, len: usize) -> Vec<u8> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    assert_eq!(b.len(), len, "{}: 字节数", p.display());
    b
}

fn read_f32(p: &Path, len: usize) -> Vec<f32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    assert_eq!(b.len(), len * 4, "{}: 字节数", p.display());
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// 解包是确定性的 → 与 numpy 参考逐位一致。返回 (最大绝对误差, 不一致元素数)。
fn check_case(st: &dinfer_kernels::Stream, root: &Path, c: &Case) -> (f32, usize) {
    let packed = read_u8(&root.join(format!("{}.packed", c.name)), c.rows * c.cols / 2);
    let scale = read_u8(&root.join(format!("{}.scale", c.name)), c.rows * c.cols / c.group);
    let want = read_f32(&root.join(format!("{}.ref", c.name)), c.rows * c.cols);

    let dp = st.htod_copy(packed).unwrap();
    let ds = st.htod_copy(scale).unwrap();
    let mut dout = st.alloc_uninit::<bf16>(c.rows * c.cols).unwrap();
    dinfer_kernels::ops::mxfp4_dequant_bf16(&st, &dp, &ds, &mut dout, c.rows, c.cols, c.group).unwrap();
    st.synchronize().unwrap();
    let got = st.dtoh_sync_copy(&dout).unwrap();

    let mut max_err = 0f32;
    let mut nbad = 0usize;
    let mut first = String::new();
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        // 参考是 f32；kernel 输出 bf16。比较口径：把参考按同一舍入（round-to-nearest-even）降到 bf16 后比位。
        let wb = bf16::from_f32(*w);
        if g.to_bits() != wb.to_bits() {
            nbad += 1;
            if first.is_empty() {
                first = format!("首个不一致 i={i} (行 {} 列 {}): got {} want {} (ref f32 {w})", i / c.cols, i % c.cols, g.to_f32(), wb.to_f32());
            }
        }
        max_err = max_err.max((g.to_f32() - w).abs());
    }
    assert_eq!(nbad, 0, "{}: {nbad} 个元素与 numpy 参考不一致；{first}", c.name);
    (max_err, nbad)
}

#[test]
#[ignore]
fn mxfp4_dequant_real_weights() {
    let Some(root) = dir() else {
        eprintln!("  skip：MXFP4_REALCHECK_DIR 未设置或没有 manifest.txt");
        return;
    };
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let cases = manifest(&root);
    assert!(!cases.is_empty(), "manifest 为空");
    for c in &cases {
        let t0 = std::time::Instant::now();
        let (max_err, _) = check_case(&st, &root, c);
        println!(
            "  {:<6} rows={:<5} cols={:<5} group={} 逐位一致（{} 个元素），|got-ref|max {max_err:.3e}，{:.0} ms",
            c.name,
            c.rows,
            c.cols,
            c.group,
            c.rows * c.cols,
            t0.elapsed().as_secs_f64() * 1e3
        );
    }
}
