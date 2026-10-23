//! 逐形状标定：对我们的 `forward_fp8_cublaslt_algo` 枚举 cuBLASLt 的启发式候选，
//! 找出与 vLLM 输出**逐位一致**的 algo 下标。
//!
//! 数据由 `scripts/dev/kaon3_d17_calib.py` 生成到 `/w/d17b/shapes/<n>x<k>/`：
//! `A_fp8.bin`(m×k)、`B_fp8.bin`(k×n 逻辑序)、`ref_f32.bin`(m×n)、`meta.txt`。
//!
//! 跑：`D17_SHAPES=/w/d17b/shapes CUDA_ARCHS=120 cargo test --release -p dinfer-kernels \
//!        --test fp8_algo_calib -- --ignored --nocapture`

use dinfer_kernels::{bf16, Device, Gemm};

fn read_f32_le(p: &str) -> Vec<f32> {
    let b = std::fs::read(p).expect("读 ref_f32.bin");
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn meta(dir: &str) -> std::collections::HashMap<String, String> {
    let s = std::fs::read_to_string(format!("{dir}/meta.txt")).expect("读 meta.txt");
    let mut out = std::collections::HashMap::new();
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
#[ignore = "需要 GPU + 标定数据"]
fn calibrate_algo_per_shape() {
    let root = std::env::var("D17_SHAPES").unwrap_or_else(|_| "/w/d17b/shapes".to_string());
    let dev = Device::new(0).expect("cuda");
    let st = dev.default_stream();
    let gemm = Gemm::new(&st).expect("gemm");

    let mut dirs: Vec<_> = std::fs::read_dir(&root)
        .expect("读 shapes 目录")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    println!("标定 {} 种形状", dirs.len());

    for d in dirs {
        let dir = d.to_str().unwrap().to_string();
        let m = meta(&dir);
        let (rows, k) = dims(&m, "shapeA");
        let (k2, n) = dims(&m, "shapeB");
        assert_eq!(k, k2);
        let scale_a: f32 = m["scale_a"].parse().unwrap();
        let scale_b: f32 = m["scale_b"].parse().unwrap();

        let a_bytes = std::fs::read(format!("{dir}/A_fp8.bin")).expect("A");
        let b_bytes = std::fs::read(format!("{dir}/B_fp8.bin")).expect("B");
        let ref_out = read_f32_le(&format!("{dir}/ref_f32.bin"));
        assert_eq!(a_bytes.len(), rows * k);
        assert_eq!(b_bytes.len(), n * k);

        // B 落盘是 [K,N] 逻辑序 → 转成物理 [N,K] 供 Lt 使用
        let mut w_nk = b_bytes.clone();
        for row in 0..n {
            for col in 0..k {
                w_nk[row * k + col] = b_bytes[col * n + row];
            }
        }

        let da = st.htod_copy(a_bytes).expect("a");
        let dw = st.htod_copy(w_nk).expect("w");
        let dsa = st.htod_copy(vec![scale_a]).expect("sa");
        let dsb = st.htod_copy(vec![scale_b]).expect("sb");

        let cnt = gemm.fp8_cublaslt_algo_count(&dsa, &dsb, rows, n, k).expect("count");
        let mut hits = Vec::new();
        let mut best = (usize::MAX, usize::MAX);
        for idx in 0..cnt {
            let mut o = st.alloc_zeros::<bf16>(rows * n).expect("alloc");
            if gemm.forward_fp8_cublaslt_algo(&da, &dsa, &dw, &dsb, &mut o, rows, n, k, idx).is_err() {
                continue;
            }
            let g = st.dtoh_sync_copy(&o).expect("dl");
            let mut d2 = 0usize;
            for i in 0..rows * n {
                if g[i].to_f32() != ref_out[i] {
                    d2 += 1;
                }
            }
            if d2 < best.0 {
                best = (d2, idx);
            }
            if d2 == 0 {
                hits.push(idx);
            }
        }
        if hits.is_empty() {
            println!("  n={n:<6} k={k:<6} 候选 {cnt} 个：**无 0 差异**，最好 algo[{}] diff {}",
                     best.1, best.0);
        } else {
            println!("  n={n:<6} k={k:<6} 候选 {cnt} 个：algo {:?} 为 0 差异", hits);
        }
    }
    println!("DONE");
}
