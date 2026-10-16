//! 用 nvcc 编译 `cuda/*.cu` 为静态库并链接。
//!
//! 不走 `cc::Build::cuda(true)`：它会把 gcc 专有参数（-ffunction-sections 等）塞给 nvcc 导致失败。
//! 这里直接调用 nvcc 生成 .o，再用 `cc::Build` 只做归档（`ar`）。
//!
//! 环境变量：
//! - `CUDA_ARCHS`：逗号分隔的 sm 列表，默认 `86,89,90`（3090 / 4090 / H100）。
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
    println!("cargo:rustc-check-cfg=cfg(dinfer_no_nvcc)");

    if env::var_os("CARGO_FEATURE_NO_NVCC").is_some() {
        println!("cargo:rustc-cfg=dinfer_no_nvcc");
        return;
    }

    let cuda_home =
        env::var("CUDA_HOME").or_else(|_| env::var("CUDA_PATH")).unwrap_or_else(|_| "/usr/local/cuda".into());
    let nvcc = PathBuf::from(&cuda_home).join("bin/nvcc");
    if !nvcc.exists() {
        panic!("nvcc not found at {}. Set CUDA_HOME, or build with `--features no-nvcc`.", nvcc.display());
    }

    let archs = env::var("CUDA_ARCHS").unwrap_or_else(|_| "86,89,90".into());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let profile = env::var("PROFILE").unwrap_or_default();
    let opt = if profile == "release" { "-O3" } else { "-O2" };

    let mut objs = Vec::new();
    for entry in std::fs::read_dir("cuda").expect("cuda/ dir") {
        let src = entry.unwrap().path();
        if src.extension().is_none_or(|e| e != "cu") {
            continue;
        }
        let obj = out_dir.join(format!("{}.o", src.file_stem().unwrap().to_string_lossy()));
        let mut cmd = Command::new(&nvcc);
        cmd.args(["-c", "-std=c++17", opt, "--expt-relaxed-constexpr", "-Xcompiler", "-fPIC", "-I", "cuda"]);
        if profile != "release" {
            cmd.arg("-lineinfo");
        }
        for a in archs.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            cmd.arg(format!("-gencode=arch=compute_{a},code=sm_{a}"));
        }
        cmd.arg("-o").arg(&obj).arg(&src);
        run(&mut cmd, &src);
        objs.push(obj);
    }
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
