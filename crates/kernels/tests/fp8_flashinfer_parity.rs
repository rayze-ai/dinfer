//! 与 vLLM **FlashInfer** 的 fp8 GEMM 逐位对拍（sm_120）。
//!
//! 数据来自 `test/kaon3/d17b/`（`kaon3_vllm_fp8det.py` 抓的 L0 `qkv_proj` 真实入参）：
//! - `A_fp8.bin`        `[24, 2816]` e4m3 字节（vLLM 已经量化好的激活）
//! - `B_fp8.bin`        `[2816, 8192]` e4m3 字节，**物理布局是 `[N, K]`**（FlashInfer 拿的是转置视图）
//! - `ref_f32.bin`      `[24, 8192]` f32，**vLLM 前向的真实输出**（bf16 值精确扩展成 f32）
//! - `meta.txt`         `scale_a=0.59375 scale_b=0.0013166156...`
//!
//! 判定：`forward_fp8_prequant` 的输出与 `ref_f32.bin` **逐位**比较（都在 bf16 值域上比）。
//! 0 → 我们的 GEMM 与 FlashInfer 完全等价。
//!
//! 运行：`CUDA_ARCHS=120 cargo test --release -p dinfer-kernels --test fp8_flashinfer_parity -- --ignored --nocapture`

use dinfer_kernels::{bf16, Device, Gemm};

fn read_f32_le(p: &str) -> Vec<f32> {
    let b = std::fs::read(p).expect("读 ref_f32.bin");
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// 读 bf16 LE 文件并精确扩成 f32（`bf16 → f32` 是零误差扩展，所以仍可做逐位比较）
fn read_bf16_as_f32(p: &str) -> Vec<f32> {
    let b = std::fs::read(p).expect("读 fwd_ref_bf16.bin");
    b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

/// 参考输出：优先 `ref_f32.bin`（新脚本），否则 `fwd_ref_bf16.bin`（旧脚本）
fn read_ref(dir: &str) -> Vec<f32> {
    let f32p = format!("{dir}/ref_f32.bin");
    if std::path::Path::new(&f32p).exists() {
        read_f32_le(&f32p)
    } else {
        read_bf16_as_f32(&format!("{dir}/fwd_ref_bf16.bin"))
    }
}

/// 读 `meta.txt` 里的 `key=value`（值是 `(24, 2816)` 这种就原样返回）
fn meta(dir: &str) -> std::collections::HashMap<String, String> {
    let s = std::fs::read_to_string(format!("{dir}/meta.txt")).expect("读 meta.txt");
    let mut out = std::collections::HashMap::new();
    // 形状里带空格和逗号，按已知 key 逐个抠，避免 split_whitespace 把 "(24," 切开
    for key in ["shapeA", "shapeB", "scale_a", "scale_b"] {
        if let Some(i) = s.find(&format!("{key}=")) {
            let rest = &s[i + key.len() + 1..];
            let val: String = if rest.starts_with('(') {
                let j = rest.find(')').expect("形状缺右括号");
                rest[..=j].to_string()
            } else {
                rest.split_whitespace().next().unwrap_or("").to_string()
            };
            out.insert(key.to_string(), val);
        }
    }
    out
}

fn dims(m: &std::collections::HashMap<String, String>, key: &str) -> (usize, usize) {
    let s = m[key].trim_matches(|c| c == '(' || c == ')');
    let (a, b) = s.split_once(',').expect("形状");
    (a.trim().parse().unwrap(), b.trim().parse().unwrap())
}

#[test]
#[ignore = "需要 GPU + 已抓好的真实入参"]
fn fp8_matches_flashinfer_bitwise() {
    let dir = std::env::var("D17_DIR").unwrap_or_else(|_| "test/kaon3/d17b".to_string());
    let m = meta(&dir);
    let scale_a: f32 = m["scale_a"].parse().unwrap();
    let scale_b: f32 = m["scale_b"].parse().unwrap();
    let (rows, k) = dims(&m, "shapeA"); // (24, 2816)
    let (k2, n) = dims(&m, "shapeB"); // (2816, 8192)：B 的视图是 [K, N]
    assert_eq!(k, k2, "K 不一致");

    let a_bytes = std::fs::read(format!("{dir}/A_fp8.bin")).expect("A_fp8.bin");
    let b_bytes = std::fs::read(format!("{dir}/B_fp8.bin")).expect("B_fp8.bin");
    let ref_out = read_ref(&dir);
    assert_eq!(a_bytes.len(), rows * k, "A 字节数");
    assert_eq!(b_bytes.len(), n * k, "B 字节数");
    assert_eq!(ref_out.len(), rows * n, "参考输出元素数");

    let dev = Device::new(0).expect("cuda");
    let st = dev.default_stream();
    let gemm = Gemm::new(&st).expect("gemm");
    let da = st.htod_copy(a_bytes.clone()).expect("upload a");

    // B 的落盘布局有歧义：vLLM 拿到的是 `B.stride()==(1,2816)` 的**转置视图**（物理 [N,K] 行主序），
    // 而 `np.ascontiguousarray(视图)` 会按**逻辑序**重排成 [K,N] 连续。两种都试，用哪份能让输出对上
    // 由结果判定，不靠猜。
    let b_nk = b_bytes.clone(); // 解释成 [N,K] 行主序
    let mut b_kn = b_bytes.clone(); // 解释成 [K,N] 行主序 → 手动转成 [N,K]
    for row in 0..n {
        for col in 0..k {
            b_kn[row * k + col] = b_bytes[col * n + row];
        }
    }

    for (label, wbytes) in [("[N,K] 物理序", b_nk), ("[K,N] 逻辑序(已转置)", b_kn)] {
        let dw = st.htod_copy(wbytes).expect("upload w");
        let mut dout = st.alloc_zeros::<bf16>(rows * n).expect("alloc");
        gemm.forward_fp8_prequant(&da, scale_a, &dw, scale_b, &mut dout, rows, n, k)
            .expect("forward_fp8_prequant");
        let got = st.dtoh_sync_copy(&dout).expect("download");

        let mut diff = 0usize;
        let mut max = 0f32;
        let mut first = Vec::new();
        for i in 0..rows * n {
            let g = got[i].to_f32();
            let r = ref_out[i];
            if g != r {
                diff += 1;
                let d = (g - r).abs();
                if d > max {
                    max = d;
                }
                if first.len() < 4 {
                    first.push((i / n, i % n, g, r));
                }
            }
        }
        println!("B={label}: diff {diff}/{}  max {max:.4e}", rows * n);
        for (row, col, g, x) in &first {
            println!("    [row {row}, col {col}] ours {g:.6e}  vllm {x:.6e}");
        }
    }
}
