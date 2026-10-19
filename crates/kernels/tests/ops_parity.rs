//! Qwen3 前向用到的全部 kernel：逐算子随机输入 → 跑 kernel → 把输入 / 输出 dump 成 `.npy`，
//! 由 `scripts/verify/ops_parity_ref.py` 用 torch 算参考并按容差比。入口 `scripts/verify/ops_parity.sh`。
//!
//! 与 `ref_tests.rs`（torch 先生成期望、Rust 侧断言）方向相反：这里 Rust 只负责生成 + 跑 + dump，
//! 判定全部在 Python 侧，便于一张表汇总"算子 / 形状 / 最大误差 / 通过"。
//! bf16 张量按 **原始 u16 位** 存（dtype `<u2`，名字带 `_bf16` 后缀），Python 侧 `<< 16` 还原 f32，逐位无损；
//! 这样 lm_head 权重（151936×1024）只占 311 MB 而不是 622 MB。
//! dump 目录：`OPS_PARITY_DIR`（默认 `target/ops_parity/`，相对工作区根）。
//!
//! 运行：`cargo test -p dinfer-kernels --test ops_parity -- --ignored --test-threads=1`
#![cfg(not(feature = "no-nvcc"))]

use dinfer_kernels::{bf16, Device, Gemm, Stream};
use std::fs;
use std::io::Write;
use std::path::PathBuf;

// ------------------------------------------------------------------ 公共：目录 / npy / 随机数

fn out_root() -> PathBuf {
    let p = PathBuf::from(std::env::var("OPS_PARITY_DIR").unwrap_or_else(|_| "target/ops_parity".into()));
    if p.is_absolute() {
        p
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join(p)
    }
}

/// 一个 case 一个目录：`<op>/<case>/`，里面若干 .npy + meta.json。
struct Case {
    dir: PathBuf,
    meta: Vec<(String, String)>,
}

impl Case {
    fn new(op: &str, case: &str) -> Self {
        let dir = out_root().join(op).join(case);
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        Case { dir, meta: vec![] }
    }
    fn npy(&self, name: &str, descr: &str, shape: &[usize], bytes: &[u8]) {
        let shape_s = match shape.len() {
            0 => "()".to_string(),
            1 => format!("({},)", shape[0]),
            _ => format!("({})", shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")),
        };
        let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_s}, }}");
        // 总头长（magic 6 + ver 2 + len 2 + header）对齐 64，末尾 '\n'
        let pad = 64 - (10 + header.len() + 1) % 64;
        header.push_str(&" ".repeat(pad % 64));
        header.push('\n');
        let mut f = fs::File::create(self.dir.join(format!("{name}.npy"))).unwrap();
        f.write_all(b"\x93NUMPY\x01\x00").unwrap();
        f.write_all(&(header.len() as u16).to_le_bytes()).unwrap();
        f.write_all(header.as_bytes()).unwrap();
        f.write_all(bytes).unwrap();
    }
    fn f32(&self, name: &str, shape: &[usize], v: &[f32]) {
        assert_eq!(shape.iter().product::<usize>(), v.len(), "{name}: shape");
        self.npy(name, "<f4", shape, bytemuck::cast_slice(v));
    }
    fn i32(&self, name: &str, shape: &[usize], v: &[i32]) {
        assert_eq!(shape.iter().product::<usize>(), v.len(), "{name}: shape");
        self.npy(name, "<i4", shape, bytemuck::cast_slice(v));
    }
    fn i64(&self, name: &str, shape: &[usize], v: &[i64]) {
        assert_eq!(shape.iter().product::<usize>(), v.len(), "{name}: shape");
        self.npy(name, "<i8", shape, bytemuck::cast_slice(v));
    }
    fn u64(&self, name: &str, shape: &[usize], v: &[u64]) {
        assert_eq!(shape.iter().product::<usize>(), v.len(), "{name}: shape");
        self.npy(name, "<u8", shape, bytemuck::cast_slice(v));
    }
    /// bf16 原始位；文件名自动带 `_bf16`。
    fn bf16(&self, name: &str, shape: &[usize], v: &[bf16]) {
        assert_eq!(shape.iter().product::<usize>(), v.len(), "{name}: shape");
        let bits: Vec<u16> = v.iter().map(|x| x.to_bits()).collect();
        self.npy(&format!("{name}_bf16"), "<u2", shape, bytemuck::cast_slice(&bits));
    }
    fn u8(&self, name: &str, shape: &[usize], v: &[u8]) {
        assert_eq!(shape.iter().product::<usize>(), v.len(), "{name}: shape");
        self.npy(name, "|u1", shape, v);
    }
    fn meta<T: std::fmt::Display>(&mut self, k: &str, v: T) {
        self.meta.push((k.to_string(), v.to_string()));
    }
    fn finish(&self) {
        let body: Vec<String> = self.meta.iter().map(|(k, v)| format!("\"{k}\": {v}")).collect();
        fs::write(self.dir.join("meta.json"), format!("{{{}}}\n", body.join(", "))).unwrap();
    }
}

/// xorshift64*：确定性，每个 case 自己 seed。
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn uniform(&mut self) -> f32 {
        // [0,1)，24 位
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn normal(&mut self) -> f32 {
        // Box–Muller
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    fn normals_bf16(&mut self, n: usize, scale: f32) -> Vec<bf16> {
        (0..n).map(|_| bf16::from_f32(self.normal() * scale)).collect()
    }
    fn normals_f32(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.normal() * scale).collect()
    }
    /// 打乱 0..n
    fn perm(&mut self, n: usize) -> Vec<usize> {
        let mut v: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
        v
    }
}

fn dev() -> (Device, Stream) {
    let d = Device::new(0).unwrap();
    let s = d.default_stream();
    (d, s)
}

/// Qwen3-0.6B 真实形状
const HID: usize = 1024;
const H: usize = 16;
const HK: usize = 8;
const D: usize = 128;
const INTER: usize = 3072;
const VOCAB: usize = 151_936;
const EPS: f32 = 1e-6;
const BS: usize = 16; // kv block_size

/// rope cos_sin 表 [max_pos, D]：前 D/2 cos、后 D/2 sin（与 model rope 表一致，theta = 1e6）
fn cos_sin_table(max_pos: usize, d: usize, theta: f64) -> Vec<f32> {
    let half = d / 2;
    let mut t = vec![0f32; max_pos * d];
    for p in 0..max_pos {
        for i in 0..half {
            let inv = 1.0 / theta.powf((2 * i) as f64 / d as f64);
            let ang = p as f64 * inv;
            t[p * d + i] = ang.cos() as f32;
            t[p * d + half + i] = ang.sin() as f32;
        }
    }
    t
}

// ------------------------------------------------------------------ norm / act / embedding / add

#[test]
#[ignore]
fn parity_rmsnorm() {
    let (_d, s) = dev();
    // 第 7 轮：三种线程布局都对一遍（h=64/1024 走 warp 版 VPL 1/4，h=5120 走 block1024 / block256 VPT=4）
    let mut cases = vec![];
    for l in dinfer_kernels::arch::RmsnormLayout::ALL {
        cases.extend([(3usize, 64usize, l), (37, HID, l), (9, 5120, l)]);
    }
    for (ci, &(t, h, l)) in cases.iter().enumerate() {
        let mut c = Case::new("rmsnorm", &format!("t{t}_h{h}_{}", l.name()));
        let mut r = Rng::new(100 + ci as u64);
        let x = r.normals_bf16(t * h, 1.0);
        let w: Vec<bf16> = (0..h).map(|_| bf16::from_f32(r.uniform() + 0.5)).collect();
        let dx = s.htod_copy(x.clone()).unwrap();
        let dw = s.htod_copy(w.clone()).unwrap();
        let mut out = s.alloc_zeros::<bf16>(t * h).unwrap();
        dinfer_kernels::rmsnorm_bf16_layout(&s, &dx, &dw, &mut out, t, h, EPS, Some(l)).unwrap();
        c.bf16("x", &[t, h], &x);
        c.bf16("w", &[h], &w);
        c.bf16("out", &[t, h], &s.dtoh_sync_copy(&out).unwrap());
        c.meta("eps", EPS);
        c.finish();
    }
}

#[test]
#[ignore]
fn parity_fused_add_rmsnorm() {
    let (_d, s) = dev();
    let mut cases = vec![];
    for l in dinfer_kernels::arch::RmsnormLayout::ALL {
        cases.extend([(5usize, 64usize, l), (33, HID, l), (9, 5120, l)]);
    }
    for (ci, &(t, h, l)) in cases.iter().enumerate() {
        let mut c = Case::new("fused_add_rmsnorm", &format!("t{t}_h{h}_{}", l.name()));
        let mut r = Rng::new(200 + ci as u64);
        let x = r.normals_bf16(t * h, 1.0);
        let res = r.normals_bf16(t * h, 1.0);
        let w: Vec<bf16> = (0..h).map(|_| bf16::from_f32(r.uniform() + 0.5)).collect();
        let dx = s.htod_copy(x.clone()).unwrap();
        let mut dres = s.htod_copy(res.clone()).unwrap();
        let dw = s.htod_copy(w.clone()).unwrap();
        let mut out = s.alloc_zeros::<bf16>(t * h).unwrap();
        dinfer_kernels::fused_add_rmsnorm_bf16_layout(&s, &dx, &mut dres, &dw, &mut out, t, h, EPS, Some(l))
            .unwrap();
        c.bf16("x", &[t, h], &x);
        c.bf16("res_in", &[t, h], &res);
        c.bf16("w", &[h], &w);
        c.bf16("res_out", &[t, h], &s.dtoh_sync_copy(&dres).unwrap());
        c.bf16("out", &[t, h], &s.dtoh_sync_copy(&out).unwrap());
        c.meta("eps", EPS);
        c.finish();
    }
}

#[test]
#[ignore]
fn parity_gather_rmsnorm() {
    let (_d, s) = dev();
    for (ci, &(t, nr, h)) in [(9usize, 3usize, 64usize), (300, 8, HID)].iter().enumerate() {
        let mut c = Case::new("gather_rmsnorm", &format!("t{t}_rows{nr}_h{h}"));
        let mut r = Rng::new(250 + ci as u64);
        let x = r.normals_bf16(t * h, 1.0);
        let w: Vec<bf16> = (0..h).map(|_| bf16::from_f32(r.uniform() + 0.5)).collect();
        let rows: Vec<i32> = r.perm(t)[..nr].iter().map(|&i| i as i32).collect();
        let dx = s.htod_copy(x.clone()).unwrap();
        let dw = s.htod_copy(w.clone()).unwrap();
        let drows = s.htod_copy(rows.clone()).unwrap();
        let mut out = s.alloc_zeros::<bf16>(nr * h).unwrap();
        dinfer_kernels::gather_rmsnorm_bf16(&s, &dx, &drows, &dw, &mut out, nr, h, EPS).unwrap();
        c.bf16("x", &[t, h], &x);
        c.bf16("w", &[h], &w);
        c.i32("rows", &[nr], &rows);
        c.bf16("out", &[nr, h], &s.dtoh_sync_copy(&out).unwrap());
        c.meta("eps", EPS);
        c.finish();
    }
}

#[test]
#[ignore]
fn parity_silu_mul() {
    let (_d, s) = dev();
    for (ci, &(t, i)) in [(4usize, 32usize), (29, INTER)].iter().enumerate() {
        let c = Case::new("silu_mul", &format!("t{t}_i{i}"));
        let mut r = Rng::new(300 + ci as u64);
        let x = r.normals_bf16(t * 2 * i, 2.0);
        let dx = s.htod_copy(x.clone()).unwrap();
        let mut out = s.alloc_zeros::<bf16>(t * i).unwrap();
        dinfer_kernels::silu_and_mul_bf16_n(&s, &dx, &mut out, t, i).unwrap();
        c.bf16("x", &[t, 2 * i], &x);
        c.bf16("out", &[t, i], &s.dtoh_sync_copy(&out).unwrap());
        c.finish();
    }
}

#[test]
#[ignore]
fn parity_embedding() {
    let (_d, s) = dev();
    for (ci, &(t, v, h)) in [(6usize, 100usize, 32usize), (64, VOCAB, HID)].iter().enumerate() {
        let c = Case::new("embedding", &format!("t{t}_v{v}_h{h}"));
        let mut r = Rng::new(400 + ci as u64);
        let table = r.normals_bf16(v * h, 0.05);
        let mut ids: Vec<i32> = (0..t).map(|_| r.below(v) as i32).collect();
        ids[0] = 0;
        ids[t - 1] = (v - 1) as i32;
        let dt = s.htod_copy(table.clone()).unwrap();
        let dids = s.htod_copy(ids.clone()).unwrap();
        let mut out = s.alloc_zeros::<bf16>(t * h).unwrap();
        dinfer_kernels::embedding_bf16_n(&s, &dids, &dt, &mut out, t, h).unwrap();
        c.bf16("table", &[v, h], &table);
        c.i32("ids", &[t], &ids);
        c.bf16("out", &[t, h], &s.dtoh_sync_copy(&out).unwrap());
        c.finish();
    }
}

#[test]
#[ignore]
fn parity_add_inplace() {
    let (_d, s) = dev();
    for (ci, &n) in [(7usize * 64usize), 41 * HID].iter().enumerate() {
        let c = Case::new("add_inplace", &format!("n{n}"));
        let mut r = Rng::new(500 + ci as u64);
        let x = r.normals_bf16(n, 1.0);
        let y = r.normals_bf16(n, 1.0);
        let dx = s.htod_copy(x.clone()).unwrap();
        let mut dy = s.htod_copy(y.clone()).unwrap();
        dinfer_kernels::add_inplace_bf16_n(&s, &dx, &mut dy, n).unwrap();
        c.bf16("x", &[n], &x);
        c.bf16("y_in", &[n], &y);
        c.bf16("y_out", &[n], &s.dtoh_sync_copy(&dy).unwrap());
        c.finish();
    }
}

#[test]
#[ignore]
fn parity_bf16_to_f32() {
    let (_d, s) = dev();
    for (ci, &n) in [(1000usize), 3 * VOCAB].iter().enumerate() {
        let c = Case::new("bf16_to_f32", &format!("n{n}"));
        let mut r = Rng::new(600 + ci as u64);
        let x = r.normals_bf16(n, 5.0);
        let dx = s.htod_copy(x.clone()).unwrap();
        let mut dy = s.alloc_zeros::<f32>(n).unwrap();
        dinfer_kernels::bf16_to_f32_n(&s, &dx, &mut dy, n).unwrap();
        c.bf16("x", &[n], &x);
        c.f32("y", &[n], &s.dtoh_sync_copy(&dy).unwrap());
        c.finish();
    }
}

// ------------------------------------------------------------------ gemm（cuBLASLt）

#[test]
#[ignore]
fn parity_gemm() {
    let (_d, s) = dev();
    let g = Gemm::new(&s).unwrap();
    // (m, n, k)：小；qkv 投影；down 投影；lm_head（m 小一点，权重 311 MB）
    let shapes = [(5usize, 48usize, 32usize), (37, (H + 2 * HK) * D, HID), (37, HID, INTER), (8, VOCAB, HID)];
    for (ci, &(m, n, k)) in shapes.iter().enumerate() {
        let c = Case::new("gemm", &format!("m{m}_n{n}_k{k}"));
        let mut r = Rng::new(700 + ci as u64);
        let a = r.normals_bf16(m * k, 1.0);
        let w = r.normals_bf16(n * k, (1.0 / k as f32).sqrt());
        let da = s.htod_copy(a.clone()).unwrap();
        let dw = s.htod_copy(w.clone()).unwrap();
        let mut out = s.alloc_zeros::<bf16>(m * n).unwrap();
        g.forward_bf16(&da, &dw, None, &mut out, m, n, k).unwrap();
        c.bf16("a", &[m, k], &a);
        c.bf16("w", &[n, k], &w);
        c.bf16("out", &[m, n], &s.dtoh_sync_copy(&out).unwrap());
        c.finish();
    }
}

// ------------------------------------------------------------------ gemm_small_m（自写 decode 小 M kernel）

#[test]
#[ignore]
fn parity_gemm_small_m() {
    let (_d, s) = dev();
    // (m, n, k, splits, bias)：Qwen3.5-27B 真形状（hidden 5120）+ M∈{1,4,8}；splits 覆盖不切 / 切；一例带 bias。
    let hid = 5120usize;
    let cases: [(usize, usize, usize, usize, bool); 6] = [
        (1, 34816, hid, 1, false), // mlp.gate_up
        (1, 48, hid, 16, false),   // gdn.in_proj_a（小 N 必须 split 才能铺满）
        (4, hid, 6144, 4, false),  // gdn.out_proj / attn.o_proj
        (8, 10240, hid, 2, false), // gdn.in_proj_qkv
        (8, hid, 17408, 8, true),  // mlp.down + bias
        (3, 6144, hid, 1, true),   // 非 2 的幂 M（M_TILE=4 只算 3 行）+ bias
    ];
    for (ci, &(m, n, k, splits, bias)) in cases.iter().enumerate() {
        let c = Case::new(
            "gemm_small_m",
            &format!("m{m}_n{n}_k{k}_s{splits}{}", if bias { "_bias" } else { "" }),
        );
        let mut r = Rng::new(760 + ci as u64);
        let a = r.normals_bf16(m * k, 1.0);
        let w = r.normals_bf16(n * k, (1.0 / k as f32).sqrt());
        let b = r.normals_bf16(n, 1.0);
        let da = s.htod_copy(a.clone()).unwrap();
        let dw = s.htod_copy(w.clone()).unwrap();
        let db = s.htod_copy(b.clone()).unwrap();
        let mut out = s.alloc_zeros::<bf16>(m * n).unwrap();
        let mut partial = s.alloc_zeros::<f32>(splits * m * n).unwrap();
        dinfer_kernels::gemm_small_m_bf16(
            &s,
            &da,
            &dw,
            bias.then_some(&db),
            &mut out,
            Some(&mut partial),
            m,
            n,
            k,
            splits,
        )
        .unwrap();
        c.bf16("a", &[m, k], &a);
        c.bf16("w", &[n, k], &w);
        if bias {
            c.bf16("bias", &[n], &b);
        }
        c.bf16("out", &[m, n], &s.dtoh_sync_copy(&out).unwrap());
        c.finish();
    }
}

// ------------------------------------------------------------------ qkv_post（q/k norm + neox rope + KV 写 cache）

#[test]
#[ignore]
fn parity_qkv_post() {
    let (_d, s) = dev();
    // (t, h, hk)；d 固定 128（kernel 要求）
    for (ci, &(t, h, hk)) in [(6usize, 4usize, 2usize), (45, H, HK)].iter().enumerate() {
        let mut c = Case::new("qkv_post", &format!("t{t}_h{h}_hk{hk}"));
        let mut r = Rng::new(800 + ci as u64);
        let w = (h + 2 * hk) * D;
        let qkv = r.normals_bf16(t * w, 1.0);
        let wq: Vec<bf16> = (0..D).map(|_| bf16::from_f32(r.uniform() + 0.5)).collect();
        let wk: Vec<bf16> = (0..D).map(|_| bf16::from_f32(r.uniform() + 0.5)).collect();
        let max_pos = 4096;
        let pos: Vec<i32> = (0..t).map(|_| r.below(max_pos) as i32).collect();
        let cs = cos_sin_table(max_pos, D, 1e6);
        let nblocks = (t + BS - 1) / BS + 1;
        let perm = r.perm(nblocks * BS);
        let mut slots: Vec<i64> = perm[..t].iter().map(|&x| x as i64).collect();
        slots[t / 2] = -1; // 一行不写 cache
        let mut dqkv = s.htod_copy(qkv.clone()).unwrap();
        let dwq = s.htod_copy(wq.clone()).unwrap();
        let dwk = s.htod_copy(wk.clone()).unwrap();
        let dpos = s.htod_copy(pos.clone()).unwrap();
        let dcs = s.htod_copy(cs.clone()).unwrap();
        let mut cache = s.alloc_zeros::<bf16>(nblocks * BS * hk * 2 * D).unwrap();
        let dslots = s.htod_copy(slots.clone()).unwrap();
        dinfer_kernels::qkv_post_bf16(
            &s, &mut dqkv, &dwq, &dwk, &dpos, &dcs, &mut cache, &dslots, t, h, hk, D, EPS, BS,
        )
        .unwrap();
        c.bf16("qkv_in", &[t, w], &qkv);
        c.bf16("wq", &[D], &wq);
        c.bf16("wk", &[D], &wk);
        c.i32("pos", &[t], &pos);
        c.f32("cos_sin", &[max_pos, D], &cs);
        c.i64("slots", &[t], &slots);
        c.bf16("qkv_out", &[t, w], &s.dtoh_sync_copy(&dqkv).unwrap());
        c.bf16("cache", &[nblocks * BS, hk, 2, D], &s.dtoh_sync_copy(&cache).unwrap());
        c.meta("heads", h);
        c.meta("kv_heads", hk);
        c.meta("head_dim", D);
        c.meta("eps", EPS);
        c.meta("block_size", BS);
        c.finish();
    }
}

// ------------------------------------------------------------------ attention

/// 把每请求的 k/v（[len, hk, D]）按 block_table 写进 paged cache；返回 slot 表。
fn alloc_blocks(r: &mut Rng, lens: &[usize], bs: usize) -> (Vec<Vec<i32>>, usize, usize) {
    let need: Vec<usize> = lens.iter().map(|&l| (l + bs - 1) / bs).collect();
    let total: usize = need.iter().sum::<usize>() + 1;
    let perm = r.perm(total);
    let mut bt = vec![];
    let mut off = 0;
    for &n in &need {
        bt.push(perm[off..off + n].iter().map(|&b| b as i32).collect::<Vec<i32>>());
        off += n;
    }
    let max_blocks = *need.iter().max().unwrap();
    (bt, total, max_blocks)
}

fn flat_bt(bt: &[Vec<i32>], max_blocks: usize) -> Vec<i32> {
    let mut v = vec![0i32; bt.len() * max_blocks];
    for (i, row) in bt.iter().enumerate() {
        v[i * max_blocks..i * max_blocks + row.len()].copy_from_slice(row);
    }
    v
}

#[test]
#[ignore]
fn parity_gqa_prefill() {
    let (_d, s) = dev();
    let shapes: [(usize, usize, Vec<usize>); 2] = [(4, 2, vec![5, 3]), (H, HK, vec![200, 57, 1])];
    for (ci, (h, hk, lens)) in shapes.iter().enumerate() {
        let (h, hk) = (*h, *hk);
        let mut c = Case::new(
            "gqa_prefill",
            &format!("h{h}_hk{hk}_lens{}", lens.iter().map(|l| l.to_string()).collect::<Vec<_>>().join("-")),
        );
        let mut r = Rng::new(900 + ci as u64);
        let t: usize = lens.iter().sum();
        let w = (h + 2 * hk) * D;
        let qkv = r.normals_bf16(t * w, 1.0);
        let mut cu = vec![0i32];
        for &l in lens {
            cu.push(cu.last().unwrap() + l as i32);
        }
        let dqkv = s.htod_copy(qkv.clone()).unwrap();
        let dcu = s.htod_copy(cu.clone()).unwrap();
        let mut out = s.alloc_zeros::<bf16>(t * h * D).unwrap();
        let qv = dqkv.slice(..);
        let kv = dqkv.slice(h * D..);
        let vv = dqkv.slice((h + hk) * D..);
        let scale = (D as f32).powf(-0.5);
        dinfer_kernels::gqa_prefill_bf16(
            &s,
            &qv,
            &kv,
            &vv,
            &mut out,
            &dcu,
            lens.len(),
            *lens.iter().max().unwrap(),
            h,
            hk,
            D,
            scale,
            w,
            w,
        )
        .unwrap();
        c.bf16("qkv", &[t, w], &qkv);
        c.i32("cu_seqlens", &[cu.len()], &cu);
        c.bf16("out", &[t, h, D], &s.dtoh_sync_copy(&out).unwrap());
        c.meta("heads", h);
        c.meta("kv_heads", hk);
        c.meta("head_dim", D);
        c.meta("scale", scale);
        c.finish();
    }
}

/// paged / decode / mixed 共用：hist + new 全部写 cache，q 只有 new 行（紧凑 [tq, h, D]）。
fn paged_case(
    s: &Stream,
    op: &str,
    seed: u64,
    h: usize,
    hk: usize,
    d: usize,
    hist: &[usize],
    new: &[usize],
    mode: &str,
) {
    let name = if hist.len() > 8 {
        // 大 batch：只写 B 与 hist 范围，避免目录名过长
        format!(
            "h{h}_hk{hk}_d{d}_b{}_hist{}-{}_new1",
            hist.len(),
            hist.iter().min().unwrap(),
            hist.iter().max().unwrap()
        )
    } else {
        format!(
            "h{h}_hk{hk}_d{d}_hist{}_new{}",
            hist.iter().map(|l| l.to_string()).collect::<Vec<_>>().join("-"),
            new.iter().map(|l| l.to_string()).collect::<Vec<_>>().join("-")
        )
    };
    let mut c = Case::new(op, &name);
    let mut r = Rng::new(seed);
    let n = hist.len();
    let lens: Vec<usize> = hist.iter().zip(new).map(|(a, b)| a + b).collect();
    let (bt, total_blocks, max_blocks) = alloc_blocks(&mut r, &lens, BS);
    let tq: usize = new.iter().sum();
    let tk: usize = lens.iter().sum();
    // 全部 k/v（hist+new）按请求顺序平铺 [tk, hk, d]，slot 由 block_table 决定
    let k = r.normals_bf16(tk * hk * d, 1.0);
    let v = r.normals_bf16(tk * hk * d, 1.0);
    let mut slots = Vec::with_capacity(tk);
    for (i, &l) in lens.iter().enumerate() {
        for p in 0..l {
            slots.push(bt[i][p / BS] as i64 * BS as i64 + (p % BS) as i64);
        }
    }
    let q = r.normals_bf16(tq * h * d, 1.0);
    let mut cu_q = vec![0i32];
    for &l in new {
        cu_q.push(cu_q.last().unwrap() + l as i32);
    }
    let seq_lens: Vec<i32> = lens.iter().map(|&l| l as i32).collect();
    let btf = flat_bt(&bt, max_blocks);

    let mut cache = s.alloc_zeros::<bf16>(total_blocks * BS * hk * 2 * d).unwrap();
    let dk = s.htod_copy(k).unwrap();
    let dv = s.htod_copy(v).unwrap();
    let dsl = s.htod_copy(slots.clone()).unwrap();
    dinfer_kernels::gqa_cache_write_bf16(s, &dk, &dv, &mut cache, &dsl, tk, hk, d, BS).unwrap();
    let dq = s.htod_copy(q.clone()).unwrap();
    let dbt = s.htod_copy(btf.clone()).unwrap();
    let dseq = s.htod_copy(seq_lens.clone()).unwrap();
    let dcu = s.htod_copy(cu_q.clone()).unwrap();
    let mut out = s.alloc_zeros::<bf16>(tq * h * d).unwrap();
    let scale = (d as f32).powf(-0.5);
    let max_q = *new.iter().max().unwrap();
    match mode {
        "paged" => dinfer_kernels::gqa_paged_bf16(
            s, &dq, &cache, &dbt, &dseq, &dcu, &mut out, n, max_q, h, hk, d, BS, max_blocks, scale, 0, false,
        )
        .unwrap(),
        "decode" => {
            let mut scratch = s.alloc_zeros::<f32>(dinfer_kernels::gqa_decode_scratch_len(n, h, d)).unwrap();
            dinfer_kernels::gqa_decode_bf16(
                s,
                &dq,
                &cache,
                &dbt,
                &dseq,
                &mut out,
                &mut scratch,
                n,
                h,
                hk,
                d,
                BS,
                max_blocks,
                scale,
                0,
                None,
            )
            .unwrap()
        }
        "mixed" => {
            let mut scratch = s.alloc_zeros::<f32>(dinfer_kernels::gqa_decode_scratch_len(n, h, d)).unwrap();
            dinfer_kernels::gqa_mixed_bf16(
                s,
                &dq,
                &cache,
                &dbt,
                &dseq,
                &dcu,
                &mut out,
                &mut scratch,
                n,
                max_q,
                h,
                hk,
                d,
                BS,
                max_blocks,
                scale,
                0,
            )
            .unwrap()
        }
        _ => unreachable!(),
    }
    c.bf16("cache", &[total_blocks * BS, hk, 2, d], &s.dtoh_sync_copy(&cache).unwrap());
    c.bf16("q", &[tq, h, d], &q);
    c.i32("block_table", &[n, max_blocks], &btf);
    c.i32("seq_lens", &[n], &seq_lens);
    c.i32("cu_seqlens_q", &[n + 1], &cu_q);
    c.bf16("out", &[tq, h, d], &s.dtoh_sync_copy(&out).unwrap());
    c.meta("heads", h);
    c.meta("kv_heads", hk);
    c.meta("head_dim", d);
    c.meta("block_size", BS);
    c.meta("scale", scale);
    c.finish();
}

#[test]
#[ignore]
fn parity_gqa_paged() {
    let (_d, s) = dev();
    paged_case(&s, "gqa_paged", 1000, 4, 2, D, &[11, 6], &[3, 1], "paged");
    paged_case(&s, "gqa_paged", 1001, H, HK, D, &[40, 70, 0, 130], &[5, 1, 33, 64], "paged");
}

#[test]
#[ignore]
fn parity_gqa_decode() {
    let (_d, s) = dev();
    paged_case(&s, "gqa_decode", 1100, 4, 2, D, &[7, 20], &[1, 1], "decode");
    // B=32（Hkv=8 → B×Hkv=256 → splits=1 直接写出路径）与 B=4（split-K 多段）
    let hist32: Vec<usize> = (0..32).map(|i| 1 + (i * 37) % 400).collect();
    paged_case(&s, "gqa_decode", 1101, H, HK, D, &hist32, &vec![1; 32], "decode");
    paged_case(&s, "gqa_decode", 1102, H, HK, D, &[300, 15, 1023, 64], &[1, 1, 1, 1], "decode");
}

#[test]
#[ignore]
fn parity_gqa_mixed() {
    let (_d, s) = dev();
    paged_case(&s, "gqa_mixed", 1200, 4, 2, D, &[9, 5, 12], &[1, 4, 1], "mixed");
    paged_case(&s, "gqa_mixed", 1201, H, HK, D, &[100, 0, 250, 30, 7], &[1, 48, 1, 17, 1], "mixed");
}

// ------------------------------------------------------------------ Qwen3.5（D=256 / GROUP 6 / partial RoPE / 输出门）

/// Qwen3.5 全注意力层形状：24 q 头 / 4 kv 头（group 6）、head_dim 256。
const H35: usize = 24;
const HK35: usize = 4;
const D35: usize = 256;

#[test]
#[ignore]
fn parity_gqa_prefill_qwen35() {
    let (_d, s) = dev();
    let shapes: [(usize, usize, usize, Vec<usize>); 2] =
        [(H35, HK35, D35, vec![1, 37, 200]), (H35, HK35, D35, vec![512])];
    for (ci, (h, hk, d, lens)) in shapes.iter().enumerate() {
        let (h, hk, d) = (*h, *hk, *d);
        let mut c = Case::new(
            "gqa_prefill",
            &format!(
                "h{h}_hk{hk}_d{d}_lens{}",
                lens.iter().map(|l| l.to_string()).collect::<Vec<_>>().join("-")
            ),
        );
        let mut r = Rng::new(1900 + ci as u64);
        let t: usize = lens.iter().sum();
        let w = (h + 2 * hk) * d;
        let qkv = r.normals_bf16(t * w, 1.0);
        let mut cu = vec![0i32];
        for &l in lens {
            cu.push(cu.last().unwrap() + l as i32);
        }
        let dqkv = s.htod_copy(qkv.clone()).unwrap();
        let dcu = s.htod_copy(cu.clone()).unwrap();
        let mut out = s.alloc_zeros::<bf16>(t * h * d).unwrap();
        let qv = dqkv.slice(..);
        let kv = dqkv.slice(h * d..);
        let vv = dqkv.slice((h + hk) * d..);
        let scale = (d as f32).powf(-0.5);
        dinfer_kernels::gqa_prefill_bf16(
            &s,
            &qv,
            &kv,
            &vv,
            &mut out,
            &dcu,
            lens.len(),
            *lens.iter().max().unwrap(),
            h,
            hk,
            d,
            scale,
            w,
            w,
        )
        .unwrap();
        c.bf16("qkv", &[t, w], &qkv);
        c.i32("cu_seqlens", &[cu.len()], &cu);
        c.bf16("out", &[t, h, d], &s.dtoh_sync_copy(&out).unwrap());
        c.meta("heads", h);
        c.meta("kv_heads", hk);
        c.meta("head_dim", d);
        c.meta("scale", scale);
        c.finish();
    }
}

#[test]
#[ignore]
fn parity_gqa_paged_qwen35() {
    let (_d, s) = dev();
    paged_case(&s, "gqa_paged", 1910, H35, HK35, D35, &[11, 6], &[3, 1], "paged");
    paged_case(&s, "gqa_paged", 1911, H35, HK35, D35, &[40, 70, 0, 130], &[5, 1, 33, 64], "paged");
}

#[test]
#[ignore]
fn parity_gqa_decode_qwen35() {
    let (_d, s) = dev();
    // 小 batch（splits > 1，走 partial + merge）与 B=32（Hkv=4 → B×Hkv=128 → splits > 1 也可能成立）
    paged_case(&s, "gqa_decode", 1920, H35, HK35, D35, &[7, 20], &[1, 1], "decode");
    paged_case(&s, "gqa_decode", 1921, H35, HK35, D35, &[300, 15, 1023, 64], &[1, 1, 1, 1], "decode");
    let hist32: Vec<usize> = (0..32).map(|i| 1 + (i * 37) % 400).collect();
    paged_case(&s, "gqa_decode", 1922, H35, HK35, D35, &hist32, &vec![1; 32], "decode");
}

#[test]
#[ignore]
fn parity_gqa_mixed_qwen35() {
    let (_d, s) = dev();
    paged_case(&s, "gqa_mixed", 1930, H35, HK35, D35, &[100, 0, 250, 30, 7], &[1, 48, 1, 17, 1], "mixed");
}

/// Qwen3.5 qkv_post：D=256、partial RoPE（rope_dim=64）、zero-centered norm（beta=1）、输出门行布局。
/// 另加一个 D=128 的 partial RoPE 独立 case（Qwen3 整头路径仍由 parity_qkv_post 覆盖）。
#[test]
#[ignore]
fn parity_qkv_post_qwen35() {
    let (_d, s) = dev();
    // (t, h, hk, d, rope_dim, beta, qmul)
    let cases: [(usize, usize, usize, usize, usize, f32, usize); 3] =
        [(6, H35, HK35, D35, 64, 1.0, 2), (45, H35, HK35, D35, 64, 1.0, 2), (17, H, HK, D, 64, 1.0, 1)];
    for (ci, &(t, h, hk, d, rd, beta, qmul)) in cases.iter().enumerate() {
        let mut c = Case::new("qkv_post_ext", &format!("t{t}_h{h}_hk{hk}_d{d}_rd{rd}_qm{qmul}"));
        let mut r = Rng::new(1950 + ci as u64);
        let w = (h * qmul + 2 * hk) * d;
        let qkv = r.normals_bf16(t * w, 1.0);
        let wq: Vec<bf16> = (0..d).map(|_| bf16::from_f32(r.uniform() + 0.5)).collect();
        let wk: Vec<bf16> = (0..d).map(|_| bf16::from_f32(r.uniform() + 0.5)).collect();
        let max_pos = 4096;
        let pos: Vec<i32> = (0..t).map(|_| r.below(max_pos) as i32).collect();
        let cs = cos_sin_table(max_pos, rd, 1e6);
        let nblocks = (t + BS - 1) / BS + 1;
        let perm = r.perm(nblocks * BS);
        let mut slots: Vec<i64> = perm[..t].iter().map(|&x| x as i64).collect();
        slots[t / 2] = -1; // 一行不写 cache
        let mut dqkv = s.htod_copy(qkv.clone()).unwrap();
        let dwq = s.htod_copy(wq.clone()).unwrap();
        let dwk = s.htod_copy(wk.clone()).unwrap();
        let dpos = s.htod_copy(pos.clone()).unwrap();
        let dcs = s.htod_copy(cs.clone()).unwrap();
        let mut cache = s.alloc_zeros::<bf16>(nblocks * BS * hk * 2 * d).unwrap();
        let dslots = s.htod_copy(slots.clone()).unwrap();
        dinfer_kernels::qkv_post_ext_bf16(
            &s, &mut dqkv, &dwq, &dwk, &dpos, &dcs, &mut cache, &dslots, t, h, hk, d, rd, EPS, beta, qmul, BS,
        )
        .unwrap();
        c.bf16("qkv_in", &[t, w], &qkv);
        c.bf16("wq", &[d], &wq);
        c.bf16("wk", &[d], &wk);
        c.i32("pos", &[t], &pos);
        c.f32("cos_sin", &[max_pos, rd], &cs);
        c.i64("slots", &[t], &slots);
        c.bf16("qkv_out", &[t, w], &s.dtoh_sync_copy(&dqkv).unwrap());
        c.bf16("cache", &[nblocks * BS, hk, 2, d], &s.dtoh_sync_copy(&cache).unwrap());
        c.meta("heads", h);
        c.meta("kv_heads", hk);
        c.meta("head_dim", d);
        c.meta("rope_dim", rd);
        c.meta("eps", EPS);
        c.meta("norm_beta", beta);
        c.meta("qmul", qmul);
        c.meta("block_size", BS);
        c.finish();
    }
}

/// 输出门：out = x * sigmoid(gate)（原地）。
#[test]
#[ignore]
fn parity_mul_sigmoid() {
    let (_d, s) = dev();
    for (ci, &(t, n)) in [(1usize, 6144usize), (37, 12288), (5, 256)].iter().enumerate() {
        let mut c = Case::new("mul_sigmoid", &format!("t{t}_n{n}"));
        let mut r = Rng::new(1980 + ci as u64);
        let x = r.normals_bf16(t * n, 1.0);
        let g = r.normals_bf16(t * n, 2.0);
        let mut dx = s.htod_copy(x.clone()).unwrap();
        let dg = s.htod_copy(g.clone()).unwrap();
        dinfer_kernels::mul_sigmoid_bf16(&s, &mut dx, &dg, t * n).unwrap();
        c.bf16("x_in", &[t, n], &x);
        c.bf16("gate", &[t, n], &g);
        c.bf16("out", &[t, n], &s.dtoh_sync_copy(&dx).unwrap());
        c.finish();
    }
}

// ------------------------------------------------------------------ kv_pack / kv_unpack（PD / host tier）

#[test]
#[ignore]
fn parity_kv_pack() {
    let (_d, s) = dev();
    for (ci, &(layers, hk, nblocks_total, npack)) in
        [(2usize, 2usize, 6usize, 3usize), (28, HK, 40, 9)].iter().enumerate()
    {
        let mut c = Case::new("kv_pack", &format!("L{layers}_hk{hk}_blocks{npack}"));
        let mut r = Rng::new(1300 + ci as u64);
        let per_block_elems = BS * hk * 2 * D;
        let per_layer_bytes = per_block_elems * 2;
        // 每层一块独立 cache [nblocks_total, BS, hk, 2, D]，随机 bf16 位
        let mut caches = vec![];
        let mut host = vec![];
        for _ in 0..layers {
            let v: Vec<u16> =
                (0..nblocks_total * per_block_elems).map(|_| (r.next_u64() >> 48) as u16).collect();
            let vb: Vec<bf16> = v.iter().map(|&b| bf16::from_bits(b)).collect();
            caches.push(s.htod_copy(vb).unwrap());
            host.push(v);
        }
        let ptrs: Vec<u64> = caches.iter().map(|cc| s.ptr(cc)).collect();
        let dptrs = s.htod_copy(ptrs).unwrap();
        let blocks: Vec<i32> = r.perm(nblocks_total)[..npack].iter().map(|&b| b as i32).collect();
        let dblocks = s.htod_copy(blocks.clone()).unwrap();
        let mut staging = s.alloc_zeros::<u8>(npack * layers * per_layer_bytes).unwrap();
        dinfer_kernels::kv_pack(&s, &dptrs, &dblocks, &mut staging, npack, layers, per_layer_bytes, BS, hk)
            .unwrap();
        let st = s.dtoh_sync_copy(&staging).unwrap();
        // unpack 到全零 cache，回读
        let mut caches2 = vec![];
        for _ in 0..layers {
            caches2.push(s.alloc_zeros::<bf16>(nblocks_total * per_block_elems).unwrap());
        }
        let ptrs2: Vec<u64> = caches2.iter().map(|cc| s.ptr(cc)).collect();
        let dptrs2 = s.htod_copy(ptrs2).unwrap();
        dinfer_kernels::kv_unpack(&s, &dptrs2, &dblocks, &staging, npack, layers, per_layer_bytes, BS, hk)
            .unwrap();
        let mut all_in: Vec<bf16> = vec![];
        let mut all_out: Vec<bf16> = vec![];
        for l in 0..layers {
            all_in.extend(host[l].iter().map(|&b| bf16::from_bits(b)));
            all_out.extend(s.dtoh_sync_copy(&caches2[l]).unwrap());
        }
        c.bf16("cache_in", &[layers, nblocks_total, BS, hk, 2, D], &all_in);
        c.i32("blocks", &[npack], &blocks);
        c.u8("staging", &[npack, layers, per_layer_bytes], &st);
        c.bf16("cache_unpacked", &[layers, nblocks_total, BS, hk, 2, D], &all_out);
        c.meta("block_size", BS);
        c.meta("kv_heads", hk);
        c.meta("head_dim", D);
        c.finish();
    }
}

// ------------------------------------------------------------------ sampler

#[test]
#[ignore]
fn parity_sampler() {
    let (_d, s) = dev();
    for (ci, &(b, v)) in [(8usize, 512usize), (8, VOCAB)].iter().enumerate() {
        let mut c = Case::new("sampler", &format!("b{b}_v{v}"));
        let mut r = Rng::new(1400 + ci as u64);
        let logits = r.normals_f32(b * v, 3.0);
        // 行参数：0 = greedy 行；-1/0 top_k = 关
        let temperature: Vec<f32> = [0.0, 0.7, 1.0, 1.3, 0.0, 0.9, 1.0, 0.5][..b].to_vec();
        let temp01: Vec<f32> = temperature.iter().map(|&t| if t > 1e-5 { 1.0 } else { 0.0 }).collect();
        let min_p: Vec<f32> = [0.0, 0.05, 0.0, 0.1, 0.0, 0.0, 0.2, 0.0][..b].to_vec();
        let top_k: Vec<i32> = [0, 40, -1, 5, 0, 1, 0, 100][..b].to_vec();
        let top_p: Vec<f32> = [1.0, 0.9, 0.5, 1.0, 1.0, 1.0, 0.95, 0.3][..b].to_vec();
        let rep: Vec<f32> = [1.0, 1.5, 1.0, 1.2, 1.0, 1.0, 1.1, 1.0][..b].to_vec();
        let freq: Vec<f32> = [0.0, 0.0, 0.3, 0.5, 0.0, 0.0, 0.0, 0.2][..b].to_vec();
        let pres: Vec<f32> = [0.0, 0.0, 0.2, 0.1, 0.0, 0.0, 0.3, 0.0][..b].to_vec();
        // 历史 CSR：prompt + output，含重复
        let mut ids = vec![];
        let mut offs = vec![0i32];
        let mut plen = vec![];
        for i in 0..b {
            let np = 1 + (i * 3) % 7;
            let no = i % 5;
            let mut row: Vec<i32> = (0..np + no).map(|_| r.below(v) as i32).collect();
            if np + no >= 3 {
                row[np + no - 1] = row[0]; // 重复一个
            }
            ids.extend(row);
            offs.push(ids.len() as i32);
            plen.push(np as i32);
        }
        let seeds: Vec<u64> = (0..b as u64).map(|i| 4242 + i).collect();
        let offsets: Vec<u64> = (0..b as u64).map(|i| 11 * i).collect();

        let dt = s.htod_copy(temperature.clone()).unwrap();
        let dt01 = s.htod_copy(temp01.clone()).unwrap();
        let dmp = s.htod_copy(min_p.clone()).unwrap();
        let dk = s.htod_copy(top_k.clone()).unwrap();
        let dp = s.htod_copy(top_p.clone()).unwrap();
        let drep = s.htod_copy(rep.clone()).unwrap();
        let dfreq = s.htod_copy(freq.clone()).unwrap();
        let dpres = s.htod_copy(pres.clone()).unwrap();
        let dids = s.htod_copy(ids.clone()).unwrap();
        let doffs = s.htod_copy(offs.clone()).unwrap();
        let dplen = s.htod_copy(plen.clone()).unwrap();
        let dseeds = s.htod_copy(seeds.clone()).unwrap();
        let doffsets = s.htod_copy(offsets.clone()).unwrap();

        // 链：penalties → temperature → min_p → topk/topp → sample（每步 dump）
        let mut w = s.htod_copy(logits.clone()).unwrap();
        dinfer_kernels::penalties_f32(&s, &mut w, &dids, &doffs, &dplen, &drep, &dfreq, &dpres, v).unwrap();
        let after_pen = s.dtoh_sync_copy(&w).unwrap();
        dinfer_kernels::temperature_f32(&s, &mut w, &dt, v).unwrap();
        let after_temp = s.dtoh_sync_copy(&w).unwrap();
        dinfer_kernels::min_p_mask_f32(&s, &mut w, &dmp, v).unwrap();
        let after_minp = s.dtoh_sync_copy(&w).unwrap();
        dinfer_kernels::topk_topp_mask_f32(&s, &mut w, &dk, &dp, v).unwrap();
        let after_mask = s.dtoh_sync_copy(&w).unwrap();
        let mut o_chain = s.alloc_zeros::<i32>(b).unwrap();
        dinfer_kernels::sample_f32(&s, &w, &dt01, &dseeds, &doffsets, &mut o_chain, v).unwrap();
        // 融合：从 penalties 之后的 logits 起（sample_fused 不含 penalties）
        let mut w2 = s.htod_copy(after_pen.clone()).unwrap();
        let mut o_fused = s.alloc_zeros::<i32>(b).unwrap();
        dinfer_kernels::sample_fused_f32(
            &s,
            &mut w2,
            &dt,
            &dmp,
            &dk,
            &dp,
            &dt01,
            &dseeds,
            &doffsets,
            &mut o_fused,
            b,
            v,
        )
        .unwrap();
        // argmax（原始 logits）
        let dl = s.htod_copy(logits.clone()).unwrap();
        let mut o_arg = s.alloc_zeros::<i32>(b).unwrap();
        dinfer_kernels::argmax_f32_n(&s, &dl, &mut o_arg, b, v).unwrap();
        // logprobs（原始 logits，chosen = 链采样结果）
        let kk = 5usize;
        let mut lp_ids = s.alloc_zeros::<i32>(b * (kk + 1)).unwrap();
        let mut lp_val = s.alloc_zeros::<f32>(b * (kk + 1)).unwrap();
        let mut lp_rank = s.alloc_zeros::<i32>(b).unwrap();
        dinfer_kernels::logprobs_f32(&s, &dl, &o_chain, kk, &mut lp_ids, &mut lp_val, &mut lp_rank, v)
            .unwrap();
        // row_gather：ids[i] = prev[g[i]]（g<0 不动）
        let prev: Vec<i32> = (0..b).map(|i| 1000 + i as i32).collect();
        let mut g: Vec<i32> = r.perm(b).iter().map(|&x| x as i32).collect();
        g[1] = -1;
        let base: Vec<i32> = (0..b).map(|i| -(i as i32) - 1).collect();
        let dprev = s.htod_copy(prev.clone()).unwrap();
        let dg = s.htod_copy(g.clone()).unwrap();
        let mut dout = s.htod_copy(base.clone()).unwrap();
        dinfer_kernels::row_gather_i32(&s, &dprev, &dg, &mut dout, b).unwrap();

        c.f32("logits", &[b, v], &logits);
        c.f32("temperature", &[b], &temperature);
        c.f32("min_p", &[b], &min_p);
        c.i32("top_k", &[b], &top_k);
        c.f32("top_p", &[b], &top_p);
        c.f32("rep", &[b], &rep);
        c.f32("freq", &[b], &freq);
        c.f32("pres", &[b], &pres);
        c.i32("hist_ids", &[ids.len()], &ids);
        c.i32("hist_offsets", &[b + 1], &offs);
        c.i32("hist_prompt_len", &[b], &plen);
        c.u64("seeds", &[b], &seeds);
        c.u64("offsets", &[b], &offsets);
        c.f32("after_penalties", &[b, v], &after_pen);
        c.f32("after_temperature", &[b, v], &after_temp);
        c.f32("after_min_p", &[b, v], &after_minp);
        c.f32("after_topk_topp", &[b, v], &after_mask);
        c.f32("fused_work", &[b, v], &s.dtoh_sync_copy(&w2).unwrap());
        c.i32("sample_chain", &[b], &s.dtoh_sync_copy(&o_chain).unwrap());
        c.i32("sample_fused", &[b], &s.dtoh_sync_copy(&o_fused).unwrap());
        c.i32("argmax", &[b], &s.dtoh_sync_copy(&o_arg).unwrap());
        c.i32("lp_ids", &[b, kk + 1], &s.dtoh_sync_copy(&lp_ids).unwrap());
        c.f32("lp_val", &[b, kk + 1], &s.dtoh_sync_copy(&lp_val).unwrap());
        c.i32("lp_rank", &[b], &s.dtoh_sync_copy(&lp_rank).unwrap());
        c.i32("rg_prev", &[b], &prev);
        c.i32("rg_idx", &[b], &g);
        c.i32("rg_base", &[b], &base);
        c.i32("rg_out", &[b], &s.dtoh_sync_copy(&dout).unwrap());
        c.meta("logprobs_k", kk);
        c.finish();
    }
}

/// rmsnorm / fused_add_rmsnorm 三种线程布局计时（第 7 轮）：B ∈ {1, 8, 32, 64}，hidden 5120（Qwen3.5-27B）与 1024（Qwen3-0.6B）。
/// 数字写进 `docs/loops/qwen35-perf.md` 第 7 行，选法进 `ArchProfile::rmsnorm_layout`。不写 dump，不参与对拍。
#[test]
#[ignore]
fn rmsnorm_bench() {
    use dinfer_kernels::arch::RmsnormLayout;
    // 3090 上单次 launch ≈ 4.5 µs 全是启动开销，看不出 kernel 差异：把 50 次 launch 捕获进一个 CUDA graph 回放，
    // 报告的是每次 kernel 的净耗时（含 graph 内节点间隔 ~1 µs）。
    let (d, _s0) = dev();
    d.disable_event_tracking(); // cudarc 的 device_ptr 事件记录在捕获中报 901
    let s = d.new_stream().unwrap(); // 非默认流（capture 要求）
    let iters = 50;
    let replays = 20;
    for h in [5120usize, 1024] {
        for t in [1usize, 8, 32, 64] {
            let mut r = Rng::new(7);
            let x = r.normals_bf16(t * h, 1.0);
            let w: Vec<bf16> = (0..h).map(|_| bf16::from_f32(r.uniform() + 0.5)).collect();
            let dx = s.htod_copy(x.clone()).unwrap();
            let dw = s.htod_copy(w).unwrap();
            let mut dres = s.htod_copy(x).unwrap();
            let mut out = s.alloc_zeros::<bf16>(t * h).unwrap();
            let mut line = format!("  h={h:>4} B={t:>2}:");
            for fused in [false, true] {
                for l in RmsnormLayout::ALL {
                    let once = |s: &Stream, dres: &mut _, out: &mut _| {
                        if fused {
                            dinfer_kernels::fused_add_rmsnorm_bf16_layout(
                                s,
                                &dx,
                                dres,
                                &dw,
                                out,
                                t,
                                h,
                                EPS,
                                Some(l),
                            )
                            .unwrap();
                        } else {
                            dinfer_kernels::rmsnorm_bf16_layout(s, &dx, &dw, out, t, h, EPS, Some(l))
                                .unwrap();
                        }
                    };
                    once(&s, &mut dres, &mut out);
                    s.synchronize().unwrap();
                    s.begin_capture().unwrap();
                    for _ in 0..iters {
                        once(&s, &mut dres, &mut out);
                    }
                    let graph = s.end_capture().unwrap();
                    graph.launch().unwrap();
                    s.synchronize().unwrap();
                    let t0 = std::time::Instant::now();
                    for _ in 0..replays {
                        graph.launch().unwrap();
                    }
                    s.synchronize().unwrap();
                    let us = t0.elapsed().as_secs_f64() * 1e6 / (iters * replays) as f64;
                    line.push_str(&format!(" {}{} {us:5.2}", if fused { "add+" } else { "" }, l.name()));
                }
            }
            println!("{line}  (µs)");
        }
    }
}
