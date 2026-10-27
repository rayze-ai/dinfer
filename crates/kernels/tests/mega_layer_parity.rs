//! kaon3-mega B2：整层 megakernel 对拍 + 计时（sliding 档 GROUP=2, D=256）。
//!
//! 两条路径走**同一份 device 代码**（mega_layer_forward），区别只在 launch 方式：
//!   - 融合（fused）：一次 launch 跑全 15 stage（stage_lo=0, stage_hi=14, setup=1）。
//!   - 分离（sep）  ：每 stage 一次 launch（(s,s)，首个 setup=1，其余 0；流序列化 + ready 累加 = 等价分别 launch）。
//! 判据：最终 out + 关键中间量（qkv_buf / attn_buf / o_buf / x2）**逐位一致 0/N**。
//! 这测的是"融合本身不破坏数值"（省 launch + 片上衔接不改结果）；与 vLLM/eager 的 serve 级对拍留到接 serve。
//!
//! 跑：`CUDA_ARCHS=120 cargo test --release -p dinfer-kernels --features sm120 --test mega_layer_parity -- --ignored --nocapture`
use dinfer_kernels::{bf16, ffi, Device};
use ffi::MegaLayerParams;
use std::ffi::c_void;

const H: usize = 2816;
const I: usize = 2112;
const HEADS: usize = 16;
const NKV: usize = 8;
const D: usize = 256;
const EPS: f32 = 1e-6;

fn sigma_n() -> usize { HEADS * D + 2 * NKV * D } // 8192

fn rand(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s % 2000) as f32 / 1000.0) - 1.0
        })
        .collect()
}

fn f32_to_e4m3(y: f32) -> u8 {
    let s = if y < 0.0 { 0x80u8 } else { 0 };
    let a = y.abs();
    if a >= 448.0 { return s | 0x7E; }
    if a == 0.0 { return s; }
    let e = a.log2().floor() as i32;
    if e < -6 {
        let m = (a / (1.0f32 / 8.0) * 2f32.powi(-6)).round_ties_even() as i32;
        return s | (m as u8).min(7);
    }
    let e = e.clamp(-6, 8);
    let mut m = ((a / 2f32.powi(e) - 1.0) * 8.0).round_ties_even() as i32;
    if m >= 8 { m = 0; } else { m = m.clamp(0, 7); }
    s | (((e + 7) as u8) << 3) | (m as u8 & 7)
}

fn q_fp8(x: &[f32], inv: f32) -> Vec<u8> {
    x.iter().map(|&v| f32_to_e4m3((v * inv).clamp(-448.0, 448.0))).collect()
}

#[test]
#[ignore]
fn mega_layer_fused_matches_separate() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();

    let m: usize = std::env::var("MEGA_M").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    let n: usize = std::env::var("MEGA_N").ok().and_then(|v| v.parse().ok()).unwrap_or(48); // seq_len
    let block_size = 16usize;
    let max_blocks = n.div_ceil(block_size) + 1;
    let sn = sigma_n();

    // ---- 权重 ----
    let input_w = rand(H, 1);
    let q_norm_w = rand(D, 2);
    let k_norm_w = rand(D, 3);
    let post_attn_w = rand(H, 4);
    let pre_ff_w = rand(H, 5);
    let post_ff1_w = rand(H, 6);
    let qkv_w = rand(sn * H, 11);
    let o_w = rand(H * HEADS * D, 12);
    let gate_w = rand(I * H, 13);
    let up_w = rand(I * H, 14);
    let dn_w = rand(H * I, 15);
    let (qkv_ws, qkv_as) = (1.2e-3f32, 0.6f32);
    let (o_ws, o_as) = (1.0e-3f32, 0.6f32);
    let (gate_ws, gate_as) = (1.2e-3f32, 0.59f32);
    let (up_ws, up_as) = (1.1e-3f32, 0.59f32);
    let (dn_ws, dn_as) = (1.7e-4f32, 0.59f32);

    let dbf = |v: &[f32]| st.htod_copy(v.iter().map(|&f| bf16::from_f32(f)).collect::<Vec<_>>()).unwrap();
    let d_input = dbf(&input_w);
    let d_qn = dbf(&q_norm_w);
    let d_kn = dbf(&k_norm_w);
    let d_pan = dbf(&post_attn_w);
    let d_pff = dbf(&pre_ff_w);
    let d_pff1 = dbf(&post_ff1_w);
    let d_qkv_w = st.htod_copy(q_fp8(&qkv_w, 1.0 / qkv_ws)).unwrap();
    let d_o_w = st.htod_copy(q_fp8(&o_w, 1.0 / o_ws)).unwrap();
    let d_gate_w = st.htod_copy(q_fp8(&gate_w, 1.0 / gate_ws)).unwrap();
    let d_up_w = st.htod_copy(q_fp8(&up_w, 1.0 / up_ws)).unwrap();
    let d_dn_w = st.htod_copy(q_fp8(&dn_w, 1.0 / dn_ws)).unwrap();

    // ---- 输入 / 位置 / paged KV ----
    let x = rand(m * H, 21);
    let d_x = dbf(&x);
    // 每请求独占 max_blocks 个块，互不重叠。cache 布局 [slot][NKV][2][D]。
    let num_slots = m * max_blocks * block_size;
    let cache0 = rand(num_slots * NKV * 2 * D, 31);
    let d_cache_bytes: Vec<bf16> = cache0.iter().map(|&f| bf16::from_f32(f)).collect();
    let d_cache = st.htod_copy(d_cache_bytes.clone()).unwrap();

    let mut block_table = vec![0i32; m * max_blocks];
    for r in 0..m {
        for j in 0..max_blocks {
            block_table[r * max_blocks + j] = (r * max_blocks + j) as i32;
        }
    }
    let d_bt = st.htod_copy(block_table.clone()).unwrap();
    let seq_lens: Vec<i32> = (0..m).map(|_| n as i32).collect();
    let d_sl = st.htod_copy(seq_lens).unwrap();
    // current token 写在位置 n-1 的 slot
    let slot_mapping: Vec<i32> = (0..m)
        .map(|r| {
            let pos = n - 1;
            block_table[r * max_blocks + pos / block_size] * block_size as i32 + (pos % block_size) as i32
        })
        .collect();
    let d_sm = st.htod_copy(slot_mapping).unwrap();
    let positions: Vec<i32> = (0..m).map(|_| (n - 1) as i32).collect();
    let d_pos = st.htod_copy(positions).unwrap();
    // rope 表 [max_pos, rope_dim]；rope_dim=D（整头旋转），neox，span=D/2。cos/sin 各 half。
    let rope_dim = D;
    let max_pos = n + 8;
    let half = rope_dim / 2;
    let mut cos_sin = vec![0f32; max_pos * rope_dim];
    for p in 0..max_pos {
        for i in 0..half {
            let freq = (p as f32) / 10000f32.powf(2.0 * i as f32 / rope_dim as f32);
            cos_sin[p * rope_dim + i] = freq.cos();
            cos_sin[p * rope_dim + half + i] = freq.sin();
        }
    }
    let d_cs = st.htod_copy(cos_sin).unwrap();

    // ---- 中间缓冲（两路共享：顺序跑，跑完 snapshot 再跑下一路）----
    let mut norm_bf = st.alloc_zeros::<bf16>(m * H).unwrap();
    let mut norm_q = st.alloc_zeros::<u8>(m * H).unwrap();
    let mut qkv_buf = st.alloc_zeros::<bf16>(m * sn).unwrap();
    let mut attn_buf = st.alloc_zeros::<bf16>(m * HEADS * D).unwrap();
    let mut attn_q = st.alloc_zeros::<u8>(m * HEADS * D).unwrap();
    let mut o_buf = st.alloc_zeros::<bf16>(m * H).unwrap();
    let mut x2 = st.alloc_zeros::<bf16>(m * H).unwrap();
    let mut norm1_bf = st.alloc_zeros::<bf16>(m * H).unwrap();
    let mut norm1_q = st.alloc_zeros::<u8>(m * H).unwrap();
    let mut gate_buf = st.alloc_zeros::<bf16>(m * I).unwrap();
    let mut up_buf = st.alloc_zeros::<bf16>(m * I).unwrap();
    let mut act_q = st.alloc_zeros::<u8>(m * I).unwrap();
    let mut down_out = st.alloc_zeros::<bf16>(m * H).unwrap();
    let mut act_scratch = st.alloc_zeros::<bf16>(m * I).unwrap();
    let mut out = st.alloc_zeros::<bf16>(m * H).unwrap();
    let mut attn_scratch = st.alloc_zeros::<f32>(m * HEADS * (D + 2) * 2).unwrap();

    // split-K 全 1（最简；分离/融合都走 splits=1 原路径）
    let n_qkv = sn.div_ceil(128);
    let n_o = H.div_ceil(128);
    let n_gu = I.div_ceil(128);
    let n_dn = H.div_ceil(128);
    let mut part_qkv = st.alloc_zeros::<f32>(m * sn).unwrap();
    let mut part_o = st.alloc_zeros::<f32>(m * H).unwrap();
    let mut part_gate = st.alloc_zeros::<f32>(m * I).unwrap();
    let mut part_up = st.alloc_zeros::<f32>(m * I).unwrap();
    let mut part_dn = st.alloc_zeros::<f32>(m * H).unwrap();
    let mut sem_qkv = st.alloc_zeros::<i32>(n_qkv).unwrap();
    let mut sem_o = st.alloc_zeros::<i32>(n_o).unwrap();
    let mut sem_gate = st.alloc_zeros::<i32>(n_gu).unwrap();
    let mut sem_up = st.alloc_zeros::<i32>(n_gu).unwrap();
    let mut sem_dn = st.alloc_zeros::<i32>(n_dn).unwrap();

    // stage 调度表
    let mut dep = st.alloc_zeros::<i32>(16).unwrap();
    let mut cnt = st.alloc_zeros::<i32>(16).unwrap();
    let mut sta = st.alloc_zeros::<i32>(16).unwrap();
    let mut rdy = st.alloc_zeros::<i32>(16).unwrap();
    let mut dep2 = st.alloc_zeros::<i32>(16).unwrap();
    let mut wc = st.alloc_zeros::<u64>(16).unwrap();
    let mut ec = st.alloc_zeros::<u64>(16).unwrap();

    let build_params = |st: &dinfer_kernels::Stream| MegaLayerParams {
        x: st.ptr(&d_x) as *const c_void,
        out: std::ptr::null_mut(),
        input_w: st.ptr(&d_input) as *const c_void,
        q_norm_w: st.ptr(&d_qn) as *const c_void,
        k_norm_w: st.ptr(&d_kn) as *const c_void,
        post_attn_w: st.ptr(&d_pan) as *const c_void,
        pre_ff_w: st.ptr(&d_pff) as *const c_void,
        post_ff1_w: st.ptr(&d_pff1) as *const c_void,
        qkv_w: st.ptr(&d_qkv_w) as *const c_void,
        qkv_wscale: qkv_ws,
        qkv_a_inv: 1.0 / qkv_as,
        o_w: st.ptr(&d_o_w) as *const c_void,
        o_wscale: o_ws,
        o_a_inv: 1.0 / o_as,
        gate_w: st.ptr(&d_gate_w) as *const c_void,
        gate_wscale: gate_ws,
        gate_a_inv: 1.0 / gate_as,
        up_w: st.ptr(&d_up_w) as *const c_void,
        up_wscale: up_ws,
        up_a_inv: 1.0 / up_as,
        dn_w: st.ptr(&d_dn_w) as *const c_void,
        dn_wscale: dn_ws,
        dn_a_inv: 1.0 / dn_as,
        norm_bf: std::ptr::null_mut(),
        norm_q: std::ptr::null_mut(),
        qkv_buf: std::ptr::null_mut(),
        attn_buf: std::ptr::null_mut(),
        attn_q: std::ptr::null_mut(),
        o_buf: std::ptr::null_mut(),
        x2: std::ptr::null_mut(),
        norm1_bf: std::ptr::null_mut(),
        norm1_q: std::ptr::null_mut(),
        gate_buf: std::ptr::null_mut(),
        up_buf: std::ptr::null_mut(),
        act_q: std::ptr::null_mut(),
        down_out: std::ptr::null_mut(),
        act_scratch: std::ptr::null_mut(),
        kv_cache: std::ptr::null_mut(),
        block_table: st.ptr(&d_bt) as *const i32,
        seq_lens: st.ptr(&d_sl) as *const i32,
        slot_mapping: st.ptr(&d_sm) as *const i32,
        attn_scratch: std::ptr::null_mut(),
        cos_sin: st.ptr(&d_cs) as *const f32,
        positions: st.ptr(&d_pos) as *const i32,
        rope_dim: rope_dim as i32,
        rope_span: (rope_dim / 2) as i32,
        rope_neox: 1,
        m: m as i32,
        h: H as i32,
        i: I as i32,
        heads: HEADS as i32,
        nkv: NKV as i32,
        d: D as i32,
        block_size: block_size as i32,
        max_blocks: max_blocks as i32,
        eps: EPS,
        attn_scale: 1.0,
        kv_fp8: 1,
        splits: 1,
        keys_per_split: 32,
        sp_qkv: 1,
        sp_o: 1,
        sp_gu: 1,
        sp_dn: 1,
        part_qkv: std::ptr::null_mut(),
        part_o: std::ptr::null_mut(),
        part_gate: std::ptr::null_mut(),
        part_up: std::ptr::null_mut(),
        part_dn: std::ptr::null_mut(),
        sem_qkv: std::ptr::null_mut(),
        sem_o: std::ptr::null_mut(),
        sem_gate: std::ptr::null_mut(),
        sem_up: std::ptr::null_mut(),
        sem_dn: std::ptr::null_mut(),
        fuse_glue: 0,
        fuse_ff: 0,
    };

    // 每次运行前把可变缓冲指针填进去（借用检查：集中在一个闭包里）
    macro_rules! fill_bufs {
        ($p:expr, $cache:expr) => {{
            $p.out = st.ptr_mut(&mut out) as *mut c_void;
            $p.norm_bf = st.ptr_mut(&mut norm_bf) as *mut c_void;
            $p.norm_q = st.ptr_mut(&mut norm_q) as *mut c_void;
            $p.qkv_buf = st.ptr_mut(&mut qkv_buf) as *mut c_void;
            $p.attn_buf = st.ptr_mut(&mut attn_buf) as *mut c_void;
            $p.attn_q = st.ptr_mut(&mut attn_q) as *mut c_void;
            $p.o_buf = st.ptr_mut(&mut o_buf) as *mut c_void;
            $p.x2 = st.ptr_mut(&mut x2) as *mut c_void;
            $p.norm1_bf = st.ptr_mut(&mut norm1_bf) as *mut c_void;
            $p.norm1_q = st.ptr_mut(&mut norm1_q) as *mut c_void;
            $p.gate_buf = st.ptr_mut(&mut gate_buf) as *mut c_void;
            $p.up_buf = st.ptr_mut(&mut up_buf) as *mut c_void;
            $p.act_q = st.ptr_mut(&mut act_q) as *mut c_void;
            $p.down_out = st.ptr_mut(&mut down_out) as *mut c_void;
            $p.act_scratch = st.ptr_mut(&mut act_scratch) as *mut c_void;
            $p.kv_cache = st.ptr_mut($cache) as *mut c_void;
            $p.attn_scratch = st.ptr_mut(&mut attn_scratch) as *mut f32;
            $p.part_qkv = st.ptr_mut(&mut part_qkv) as *mut f32;
            $p.part_o = st.ptr_mut(&mut part_o) as *mut f32;
            $p.part_gate = st.ptr_mut(&mut part_gate) as *mut f32;
            $p.part_up = st.ptr_mut(&mut part_up) as *mut f32;
            $p.part_dn = st.ptr_mut(&mut part_dn) as *mut f32;
            $p.sem_qkv = st.ptr_mut(&mut sem_qkv) as *mut i32;
            $p.sem_o = st.ptr_mut(&mut sem_o) as *mut i32;
            $p.sem_gate = st.ptr_mut(&mut sem_gate) as *mut i32;
            $p.sem_up = st.ptr_mut(&mut sem_up) as *mut i32;
            $p.sem_dn = st.ptr_mut(&mut sem_dn) as *mut i32;
        }};
    }

    let mut d_cache_buf = d_cache;
    // ================= 分离路径（每 stage 一次 launch） =================
    let mut p = build_params(&st);
    fill_bufs!(p, &mut d_cache_buf);
    for s in 0..15i32 {
        let rc = unsafe {
            ffi::mega_layer_forward(
                &p as *const MegaLayerParams, s, s, if s == 0 { 1 } else { 0 },
                st.ptr_mut(&mut dep) as *mut i32, st.ptr_mut(&mut cnt) as *mut i32,
                st.ptr_mut(&mut sta) as *mut i32, st.ptr_mut(&mut rdy) as *mut i32,
                st.ptr_mut(&mut dep2) as *mut i32, std::ptr::null_mut(), std::ptr::null_mut(), 0,
                st.raw() as _,
            )
        };
        assert_eq!(rc, 0, "分离 stage {s} 返回 {rc}");
    }
    st.synchronize().unwrap();
    let out_sep = st.dtoh_sync_copy(&out).unwrap();
    let qkv_sep = st.dtoh_sync_copy(&qkv_buf).unwrap();
    let attn_sep = st.dtoh_sync_copy(&attn_buf).unwrap();
    let o_sep = st.dtoh_sync_copy(&o_buf).unwrap();
    let x2_sep = st.dtoh_sync_copy(&x2).unwrap();
    let cache_sep = st.dtoh_sync_copy(&d_cache_buf).unwrap();
    let norm1_sep = st.dtoh_sync_copy(&norm1_bf).unwrap();

    // ================= 片上融合路径（一次 launch 全链；fuse_glue=1：QKNORM 吸收 rope+kv 写）=================
    // 重置 cache（KVWRITE 改了 cache；fp8 roundtrip 幂等但重置更干净）与 ready
    d_cache_buf = st.htod_copy(d_cache_bytes.clone()).unwrap();
    st.memset_zeros_n(&mut rdy, 16).unwrap();
    let mut p2 = build_params(&st);
    p2.fuse_glue = 1;  // R40：片上流水（norm→rope→roundtrip→写 全程 head 留 smem，不落 global 中间量）
    p2.fuse_ff = 1;    // R41：S_RES 吸收 S_PRE_FF（pre_ff_norm+量化），省 barrier + 再读一遍 x2
    fill_bufs!(p2, &mut d_cache_buf);
    let rc = unsafe {
        ffi::mega_layer_forward(
            &p2 as *const MegaLayerParams, 0, 14, 1,
            st.ptr_mut(&mut dep) as *mut i32, st.ptr_mut(&mut cnt) as *mut i32,
            st.ptr_mut(&mut sta) as *mut i32, st.ptr_mut(&mut rdy) as *mut i32,
            st.ptr_mut(&mut dep2) as *mut i32, st.ptr_mut(&mut wc) as *mut u64,
            st.ptr_mut(&mut ec) as *mut u64, 1, st.raw() as _,
        )
    };
    assert_eq!(rc, 0, "融合返回 {rc}");
    st.synchronize().unwrap();
    let out_fus = st.dtoh_sync_copy(&out).unwrap();
    let qkv_fus = st.dtoh_sync_copy(&qkv_buf).unwrap();
    let attn_fus = st.dtoh_sync_copy(&attn_buf).unwrap();
    let o_fus = st.dtoh_sync_copy(&o_buf).unwrap();
    let x2_fus = st.dtoh_sync_copy(&x2).unwrap();
    let cache_fus = st.dtoh_sync_copy(&d_cache_buf).unwrap();
    let norm1_fus = st.dtoh_sync_copy(&norm1_bf).unwrap();

    // ---- 逐位比较（退化检查优先）----
    let cmp = |name: &str, a: &[bf16], b: &[bf16]| -> usize {
        let (mut nd, mut na, mut nb, mut mx) = (0usize, 0f64, 0f64, 0f32);
        for i in 0..a.len() {
            let (x, y) = (a[i].to_f32(), b[i].to_f32());
            na += (x as f64) * (x as f64);
            nb += (y as f64) * (y as f64);
            if a[i].to_bits() != b[i].to_bits() { nd += 1; }
            mx = mx.max((x - y).abs());
        }
        println!("  {name:9} ‖sep‖={:.3} ‖fus‖={:.3}  不同 {nd}/{} max|d|={mx:.3e}", na.sqrt(), nb.sqrt(), a.len());
        assert!(na.sqrt() > 0.1, "{name}: sep 范数异常（可能全零）");
        assert!(nb.sqrt() > 0.1, "{name}: fus 范数异常（可能全零）");
        nd
    };
    println!("[R40 片上融合对拍] m={m} n={n} sigmaN={sn}（基线 fuse_glue=0 15launch vs 片上融合 fuse_glue=1 1launch）");
    let nd_qkv = cmp("qkv_buf", &qkv_sep, &qkv_fus);
    let nd_cache = cmp("kv_cache", &cache_sep, &cache_fus);
    let nd_attn = cmp("attn_buf", &attn_sep, &attn_fus);
    let nd_o = cmp("o_buf", &o_sep, &o_fus);
    let nd_x2 = cmp("x2", &x2_sep, &x2_fus);
    let nd_norm1 = cmp("norm1", &norm1_sep, &norm1_fus);
    let nd_out = cmp("out", &out_sep, &out_fus);
    let total = nd_qkv + nd_cache + nd_attn + nd_o + nd_x2 + nd_norm1 + nd_out;
    println!("VERDICT: {}", if total == 0 { "融合逐位一致（0/N）" } else { "有差异（逐 stage 定位）" });

    // ---- 分阶段 exec/wait 周期（融合路径）----
    let ecv = st.dtoh_sync_copy(&ec).unwrap();
    let wcv = st.dtoh_sync_copy(&wc).unwrap();
    let names = ["IN","QKV","QKNORM","ROPE","KVWR","ATTN","OQNT","OPROJ","RES","PREFF","GATE","UP","ACT","DOWN","POST"];
    let tot: u64 = ecv[..15].iter().sum();
    println!("[融合 分阶段 exec/wait 周期]");
    for i in 0..15 {
        let pct = if tot > 0 { ecv[i] as f64 / tot as f64 * 100.0 } else { 0.0 };
        println!("  {:7} exec={:>11} ({:4.1}%)  wait={:>11}", names[i], ecv[i], pct, wcv[i]);
    }

    assert_eq!(total, 0, "片上融合与分离基线不逐位一致：qkv={nd_qkv} cache={nd_cache} attn={nd_attn} o={nd_o} x2={nd_x2} norm1={nd_norm1} out={nd_out}");

    // ---- 计时：融合（一次 launch）vs 分离（15 次 launch） ----
    const ITERS: usize = 200;
    st.synchronize().unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..ITERS {
        st.memset_zeros_n(&mut rdy, 16).unwrap();
        for s in 0..15i32 {
            unsafe {
                ffi::mega_layer_forward(
                    &p as *const MegaLayerParams, s, s, if s == 0 { 1 } else { 0 },
                    st.ptr_mut(&mut dep) as *mut i32, st.ptr_mut(&mut cnt) as *mut i32,
                    st.ptr_mut(&mut sta) as *mut i32, st.ptr_mut(&mut rdy) as *mut i32,
                    st.ptr_mut(&mut dep2) as *mut i32, std::ptr::null_mut(), std::ptr::null_mut(), 0,
                    st.raw() as _,
                );
            }
        }
    }
    st.synchronize().unwrap();
    let sep_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    // 逐 stage 墙钟拆解：**按依赖顺序 0..15 跑完整序列**（deps 满足不卡），每 stage launch 前后 sync 计时。
    // （不能孤立重复跑单 stage——依赖只在前序 signal，孤立跑会死自旋，见 R12。）
    {
        let sn = ["IN","QKV","QKNORM","ROPE","KVWR","ATTN","OQNT","OPROJ","RES","PREFF","GATE","UP","ACT","DOWN","POST"];
        const ROUNDS: usize = 10;
        const ITW: usize = 50;
        let mut rounds: Vec<[f64; 15]> = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            let mut per = [0f64; 15];
            for _ in 0..ITW {
                st.memset_zeros_n(&mut rdy, 16).unwrap();
                for s in 0..15i32 {
                    st.synchronize().unwrap();
                    let ts = std::time::Instant::now();
                    unsafe {
                        ffi::mega_layer_forward(
                            &p as *const MegaLayerParams, s, s, if s == 0 { 1 } else { 0 },
                            st.ptr_mut(&mut dep) as *mut i32, st.ptr_mut(&mut cnt) as *mut i32,
                            st.ptr_mut(&mut sta) as *mut i32, st.ptr_mut(&mut rdy) as *mut i32,
                            st.ptr_mut(&mut dep2) as *mut i32, std::ptr::null_mut(), std::ptr::null_mut(), 0,
                            st.raw() as _,
                        );
                    }
                    st.synchronize().unwrap();
                    per[s as usize] += ts.elapsed().as_secs_f64() * 1e6;
                }
            }
            for v in per.iter_mut() { *v /= ITW as f64; }
            rounds.push(per);
        }
        // 每 stage 10 轮取中位 + min/max 看稳定性
        let mut med = [0f64; 15];
        let mut lo = [0f64; 15];
        let mut hi = [0f64; 15];
        for s in 0..15 {
            let mut v: Vec<f64> = rounds.iter().map(|r| r[s]).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            med[s] = v[ROUNDS / 2];
            lo[s] = v[0];
            hi[s] = v[ROUNDS - 1];
        }
        // 明细表：每轮每 stage 的墙钟（10 轮全列）
        println!("[逐stage墙钟明细 us(10轮×50iter,每格=该轮50iter均值)]");
        print!("{:8}", "round");
        for i in 0..15 { print!("{:>8}", sn[i]); }
        println!();
        for (r, per) in rounds.iter().enumerate() {
            print!("r{:<7}", r);
            for i in 0..15 { print!("{:8.2}", per[i]); }
            println!();
        }
        // 汇总：中位 + min/max/抖动，由大到小
        let stot: f64 = med.iter().sum();
        let mut order: Vec<usize> = (0..15).collect();
        order.sort_by(|&a, &b| med[b].partial_cmp(&med[a]).unwrap());
        println!("[逐stage墙钟 汇总(10轮中位,由大到小)]  合计中位={stot:.1}us");
        for &i in &order {
            println!("  {:7} 中位{:7.2}us ({:4.1}%)  [min{:.2} max{:.2} 抖{:.1}%]",
                sn[i], med[i], med[i] / stot * 100.0, lo[i], hi[i],
                if med[i] > 0.0 { (hi[i] - lo[i]) / med[i] * 100.0 } else { 0.0 });
        }
    }

    st.synchronize().unwrap();
    let t1 = std::time::Instant::now();
    for _ in 0..ITERS {
        st.memset_zeros_n(&mut rdy, 16).unwrap();
        unsafe {
            ffi::mega_layer_forward(
                &p2 as *const MegaLayerParams, 0, 14, 1,
                st.ptr_mut(&mut dep) as *mut i32, st.ptr_mut(&mut cnt) as *mut i32,
                st.ptr_mut(&mut sta) as *mut i32, st.ptr_mut(&mut rdy) as *mut i32,
                st.ptr_mut(&mut dep2) as *mut i32, std::ptr::null_mut(), std::ptr::null_mut(), 0,
                st.raw() as _,
            );
        }
    }
    st.synchronize().unwrap();
    let fus_us = t1.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    // 中间量：融合 launch 但 **不** 片上融 glue（p.fuse_glue=0，一次 launch 全链）——隔离"省 launch" vs "片上融 glue"
    st.synchronize().unwrap();
    let t2 = std::time::Instant::now();
    for _ in 0..ITERS {
        st.memset_zeros_n(&mut rdy, 16).unwrap();
        unsafe {
            ffi::mega_layer_forward(
                &p as *const MegaLayerParams, 0, 14, 1,
                st.ptr_mut(&mut dep) as *mut i32, st.ptr_mut(&mut cnt) as *mut i32,
                st.ptr_mut(&mut sta) as *mut i32, st.ptr_mut(&mut rdy) as *mut i32,
                st.ptr_mut(&mut dep2) as *mut i32, std::ptr::null_mut(), std::ptr::null_mut(), 0,
                st.raw() as _,
            );
        }
    }
    st.synchronize().unwrap();
    let fus_noglue_us = t2.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    // 隔离 fuse_ff：fuse_glue=1 但 fuse_ff=0（一次 launch 全链，只少 R41 这一处融合）
    let mut p3 = build_params(&st);
    p3.fuse_glue = 1;
    p3.fuse_ff = 0;
    fill_bufs!(p3, &mut d_cache_buf);
    st.synchronize().unwrap();
    let t3 = std::time::Instant::now();
    for _ in 0..ITERS {
        st.memset_zeros_n(&mut rdy, 16).unwrap();
        unsafe {
            ffi::mega_layer_forward(
                &p3 as *const MegaLayerParams, 0, 14, 1,
                st.ptr_mut(&mut dep) as *mut i32, st.ptr_mut(&mut cnt) as *mut i32,
                st.ptr_mut(&mut sta) as *mut i32, st.ptr_mut(&mut rdy) as *mut i32,
                st.ptr_mut(&mut dep2) as *mut i32, std::ptr::null_mut(), std::ptr::null_mut(), 0,
                st.raw() as _,
            );
        }
    }
    st.synchronize().unwrap();
    let fus_glueonly_us = t3.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    println!("[计时] m={m} 分离(15launch)={sep_us:.1}us  融合launch无glue(1launch)={fus_noglue_us:.1}us  仅glue(1launch)={fus_glueonly_us:.1}us  glue+ff(1launch)={fus_us:.1}us");
    println!("[计时] 省launch={:.2}x  片上融glue再降={:.2}x  R41融ff再降={:.2}x  合计={:.2}x",
             sep_us / fus_noglue_us, fus_noglue_us / fus_glueonly_us, fus_glueonly_us / fus_us, sep_us / fus_us);
}
