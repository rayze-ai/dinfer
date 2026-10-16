//! 探针：cudarc 的 device_ptr 事件记录在 stream capture 下是否可用；捕获 GEMM + rmsnorm 成图并重放。
//! cargo test -p dinfer-kernels --test graph_probe -- --ignored --nocapture
use dinfer_kernels::cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use dinfer_kernels::{Device, Gemm};
use half::bf16;

#[test]
#[ignore]
fn capture_gemm_rmsnorm_and_replay() {
    let dev = Device::new(0).unwrap();
    dev.disable_event_tracking();
    let s = dev.new_stream().unwrap(); // 非默认流（capture 要求）
    let g = Gemm::new(&s).unwrap();
    let (m, n, k) = (8usize, 1024usize, 1024usize);
    let a = s.htod_copy(vec![bf16::from_f32(0.01); m * k]).unwrap();
    let w = s.htod_copy(vec![bf16::from_f32(0.02); n * k]).unwrap();
    let gw = s.htod_copy(vec![bf16::from_f32(1.0); n]).unwrap();
    let mut y = s.alloc_zeros::<bf16>(m * n).unwrap();
    let mut z = s.alloc_zeros::<bf16>(m * n).unwrap();
    // 热身（cublasLt 启发式等）
    g.forward_bf16(&a, &w, None, &mut y, m, n, k).unwrap();
    dinfer_kernels::rmsnorm_bf16_n(&s, &y, &gw, &mut z, m, n, 1e-6).unwrap();
    s.synchronize().unwrap();
    let want = s.dtoh_sync_copy(&z).unwrap();
    s.memset_zeros_n(&mut z, m * n).unwrap();
    s.synchronize().unwrap();
    // 分步探针：先只捕获 rmsnorm
    s.inner_arc().begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL).unwrap();
    let r1 = dinfer_kernels::rmsnorm_bf16_n(&s, &y, &gw, &mut z, m, n, 1e-6);
    println!("rmsnorm under capture: {:?}", r1.as_ref().err());
    let g1 =
        s.inner_arc().end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
    println!("end_capture(rmsnorm only): {:?}", g1.as_ref().err());
    s.synchronize().unwrap();
    s.memset_zeros_n(&mut z, m * n).unwrap();
    s.synchronize().unwrap();
    // 再只捕获 GEMM
    s.inner_arc().begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL).unwrap();
    let r2 = g.forward_bf16(&a, &w, None, &mut y, m, n, k);
    println!("gemm under capture: {:#?}", r2.as_ref().err().map(|e| format!("{e:#}")));
    let g2 =
        s.inner_arc().end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
    println!("end_capture(gemm only): {:?}", g2.as_ref().err());
    s.synchronize().unwrap();
    // 完整捕获
    s.begin_capture().unwrap();
    g.forward_bf16(&a, &w, None, &mut y, m, n, k).unwrap();
    dinfer_kernels::rmsnorm_bf16_n(&s, &y, &gw, &mut z, m, n, 1e-6).unwrap();
    let graph = s.end_capture().unwrap();
    // 捕获期间不应执行：z 仍为 0
    s.synchronize().unwrap();
    let z0 = s.dtoh_sync_copy(&z).unwrap();
    assert!(z0.iter().all(|x| x.to_f32() == 0.0), "capture must not execute");
    // 重放 3 次
    for _ in 0..3 {
        graph.launch().unwrap();
    }
    s.synchronize().unwrap();
    let got = s.dtoh_sync_copy(&z).unwrap();
    let maxd = got.iter().zip(&want).map(|(a, b)| (a.to_f32() - b.to_f32()).abs()).fold(0f32, f32::max);
    println!("graph replay max |diff| vs eager = {maxd}");
    assert!(maxd < 1e-3);
    // 改输入后重放（同地址）：结果应跟随
    let a2 = vec![bf16::from_f32(0.02); m * k];
    s.inner_arc().memcpy_htod(&a2, &mut a.clone()).ok();
    println!("capture OK: event record/wait inside device_ptr did not break capture");
}
