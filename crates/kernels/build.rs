//! 用 nvcc 编译 `cuda/*.cu` 为静态库并链接。
//!
//! 不走 `cc::Build::cuda(true)`：它会把 gcc 专有参数（-ffunction-sections 等）塞给 nvcc 导致失败。
//! 这里直接调用 nvcc 生成 .o，再用 `cc::Build` 只做归档（`ar`）。
//!
//! 环境变量：
//! - `CUDA_ARCHS`：逗号分隔的 sm 列表，默认 `86,89,90`（3090 / 4090 / H100）；B300 用 `103`（Blackwell Ultra，远端 env.sh 设）。
//! - `CUDA_HOME` / `CUDA_PATH`：CUDA 安装目录，默认 `/usr/local/cuda`。
//! - feature `no-nvcc`：跳过编译，kernel 入口以 stub 形式存在（调用即 panic）。

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cuda");
    println!("cargo:rerun-if-env-changed=CUDA_ARCHS");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rustc-check-cfg=cfg(no_nvcc_stub)");
    println!("cargo:rustc-check-cfg=cfg(mega_sm100)");
    println!("cargo:rerun-if-env-changed=MEGA_SM100_DEV");

    if env::var_os("CARGO_FEATURE_NO_NVCC").is_some() {
        println!("cargo:rustc-cfg=no_nvcc_stub");
        return;
    }

    let cuda_home =
        env::var("CUDA_HOME").or_else(|_| env::var("CUDA_PATH")).unwrap_or_else(|_| "/usr/local/cuda".into());
    let nvcc = PathBuf::from(&cuda_home).join("bin/nvcc");
    if !nvcc.exists() {
        panic!("nvcc not found at {}. Set CUDA_HOME, or build with `--features no-nvcc`.", nvcc.display());
    }

    let archs = env::var("CUDA_ARCHS").unwrap_or_else(|_| "86,89,90".into());
    // Blackwell 专用文件（`*_sm100.cu` 且声明只在 sm100 有意义的）只在目标含 sm_100 / sm_103 时编；
    // 其它卡用 ffi.rs 的 stub（返回 cudaErrorNotSupported）。MEGA_SM100_DEV=1 在开发机（sm86）也编，只做对拍。
    let has_sm100 = archs.split(',').any(|a| matches!(a.trim(), "100" | "100a" | "103" | "103a"))
        || env::var_os("MEGA_SM100_DEV").is_some();
    if has_sm100 {
        println!("cargo:rustc-cfg=mega_sm100");
    }
    let sm100_only = ["mega_sm100", "gqa_fa_sm100", "umma_peak_sm100"];
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let profile = env::var("PROFILE").unwrap_or_default();
    let opt = if profile == "release" { "-O3" } else { "-O2" };

    // 收集源文件（排序，保证 .a 内对象顺序稳定）
    let mut srcs: Vec<PathBuf> = std::fs::read_dir("cuda")
        .expect("cuda/ dir")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "cu"))
        .filter(|p| {
            let stem = p.file_stem().unwrap().to_string_lossy();
            has_sm100 || !sm100_only.contains(&stem.as_ref())
        })
        .collect();
    srcs.sort();
    let arch_flags: Vec<String> = archs
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|a| {
            // tcgen05 等 Blackwell 架构专用指令只在 `a` 变体里可用（compute_103a / sm_103a）。
            // sm_103a 的 cubin 只能跑在 sm_103 上，与 sm_103 的可运行范围相同，故对含 100/103 的目标全局换。
            let code = match a {
                "100" | "103" => format!("{a}a"),
                _ => a.to_string(),
            };
            format!("-gencode=arch=compute_{code},code=sm_{code}")
        })
        .collect();
    // 并行 nvcc（2026-10-20，用户："构建慢？"）：之前 23 个 .cu × 3 arch 串行 ≈ 2 min / 轮，r10 每轮都付。
    // 并发数 = cargo 给的 NUM_JOBS（受 -j 约束）或 nproc，上限 = 文件数；nvcc 自身单线程，并行无共享状态。
    let jobs = env::var("NUM_JOBS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
        .unwrap_or(4)
        .clamp(1, srcs.len().max(1));
    let objs: Vec<PathBuf> = srcs.iter().map(|src| out_dir.join(format!("{}.o", src.file_stem().unwrap().to_string_lossy()))).collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let nvcc_s = nvcc.clone();
    std::thread::scope(|sc| {
        for _ in 0..jobs {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= srcs.len() {
                    break;
                }
                let (src, obj) = (&srcs[i], &objs[i]);
                let mut cmd = Command::new(&nvcc_s);
                cmd.args(["-c", "-std=c++17", opt, "--expt-relaxed-constexpr", "-Xcompiler", "-fPIC", "-I", "cuda"]);
                // 与 cargo:rustc-cfg=mega_sm100 同步：`.cu` 侧用它决定是否调用 sm100 专用入口
                // （弱符号在归档里靠不住，见 gqa.cu 的说明）。只在编入 sm100 文件时定义。
                if has_sm100 {
                    cmd.arg("-DMEGA_SM100=1");
                }
                if profile != "release" {
                    cmd.arg("-lineinfo");
                }
                cmd.args(&arch_flags);
                cmd.arg("-o").arg(obj).arg(src);
                run(&mut cmd, src);
            });
        }
    });
    assert!(!objs.is_empty(), "no .cu files under cuda/");

    // 只归档，不编译：cc 在没有源文件时直接把 .o 打成 libdinfer_kernels.a。
    let mut ar = cc::Build::new();
    ar.cargo_metadata(true);
    for o in &objs {
        ar.object(o);
    }
    ar.compile("dinfer_kernels");

    println!("cargo:rustc-link-search=native={cuda_home}/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    // nvcc 13.0 的 host 侧 stub 用了函数内 static 的线程安全初始化（__cxa_guard_*）与 __gxx_personality_v0，
    // 需要 C++ 运行时；12.4 的 stub 不需要但多链一个 libstdc++ 无害。
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

fn run(cmd: &mut Command, src: &Path) {
    let out = cmd.output().unwrap_or_else(|e| panic!("failed to spawn nvcc for {}: {e}", src.display()));
    for line in String::from_utf8_lossy(&out.stderr).lines() {
        if !line.trim().is_empty() {
            println!("cargo:warning=nvcc[{}]: {line}", src.display());
        }
    }
    if !out.status.success() {
        panic!("nvcc failed on {} ({})", src.display(), out.status);
    }
}
