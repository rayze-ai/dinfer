//! E4：带优先级的流能建、能用、能释放（Device::new_stream_with_priority 的布局自检）。
//! cargo test --release -p dinfer-kernels --test stream_prio_smoke -- --ignored --nocapture
use dinfer_kernels::Device;
#[test]
#[ignore]
fn stream_prio_smoke() {
    let dev = Device::new(0).expect("cuda");
    let (least, greatest) = dev.stream_priority_range().unwrap();
    eprintln!("prio range least={least} greatest={greatest}");
    let hi = dev.new_stream_with_priority(greatest).unwrap();
    let lo = dev.new_stream().unwrap();
    let a = hi.alloc_zeros::<f32>(1 << 20).unwrap();
    let b = lo.alloc_zeros::<f32>(1 << 20).unwrap();
    hi.synchronize().unwrap();
    lo.synchronize().unwrap();
    let ev = lo.record_event().unwrap();
    hi.wait_event(&ev).unwrap();
    hi.synchronize().unwrap();
    let va = hi.dtoh_sync_copy(&a).unwrap();
    let vb = lo.dtoh_sync_copy(&b).unwrap();
    assert_eq!(va.len(), 1 << 20);
    assert_eq!(vb.len(), 1 << 20);
    drop(hi);
    drop(lo);
    eprintln!("stream_prio_smoke ok");
}
