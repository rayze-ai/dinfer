//! gelu_tanh·mul 口径验证：把我们的 kernel 输出 dump 出来，交给 `/w/tmp/gelu_cmp.py` 与
//! `torch.compile(F.gelu(x, approximate='tanh') * up)` 逐位比。
//! 跑法：GPU=0 scripts/dev/remote_5090.sh ctr 'cd /w/dinfer && CUDA_ARCHS=120 cargo test --release \
//!        -p dinfer-kernels --test gelu_mode -- --ignored --nocapture'
use dinfer_kernels as k;

#[test]
#[ignore]
fn dump_gelu() {
    let dev = k::Device::new(0).unwrap();
    let s = dev.default_stream();
    let (rows, inter) = (24usize, 2112usize);
    // 固定伪随机，Python 侧用同一公式重建
    let mk = |n: usize, a: f32, b: f32, seed: u64| -> Vec<k::bf16> {
        (0..n)
            .map(|i| {
                let v = ((i as u64).wrapping_mul(1103515245).wrapping_add(seed) % 2000) as f32 / 1000.0 - 1.0;
                k::bf16::from_f32(v * a + b)
            })
            .collect()
    };
    let g = mk(rows * inter, 4.0, 0.1, 12345);
    let u = mk(rows * inter, 2.0, 0.0, 54321);
    let (dg, du) = (s.htod_copy(g.clone()).unwrap(), s.htod_copy(u.clone()).unwrap());
    let mut o = s.alloc_zeros::<k::bf16>(rows * inter).unwrap();
    k::ops::gelu_tanh_mul_bf16(&s, &dg, &du, &mut o, rows, inter).unwrap();
    let got = s.dtoh_sync_copy(&o).unwrap();
    let wr = |p: &str, v: &[k::bf16]| {
        std::fs::write(p, v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect::<Vec<u8>>()).unwrap()
    };
    wr("/w/tmp/gelu_got.bin", &got);
    wr("/w/tmp/gelu_g.bin", &g);
    wr("/w/tmp/gelu_u.bin", &u);
    println!("WROTE gelu_got/g/u  rows={rows} inter={inter}");
}
