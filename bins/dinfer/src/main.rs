//! dinfer 入口。`serve`：加载模型 + tokenizer，起引擎线程与 OpenAI 兼容 HTTP 服务；`device-info`：打印 GPU 信息。

use anyhow::Result;
use clap::{Parser, Subcommand};

#[cfg(any(
    all(feature = "kimik3", feature = "glm53"),
    all(feature = "kimik3", feature = "qwen3"),
    all(feature = "glm53", feature = "qwen3")
))]
compile_error!("enable exactly one model feature: `kimik3`, `glm53` or `qwen3`");

#[derive(Parser)]
#[command(name = "dinfer", version, about = "Rust LLM inference engine for Kimi-K3 / GLM-5.3")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// 启动 HTTP 服务（无鉴权：默认只绑 127.0.0.1，对外必须放网关后）
    Serve {
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8000)]
        port: u16,
        /// 模型目录（config.json + safetensors）
        #[arg(long)]
        model: String,
        /// tokenizer / chat 模板目录（默认同 --model）
        #[arg(long)]
        tokenizer: Option<String>,
        #[arg(long)]
        served_model_name: Option<String>,
        #[arg(long, default_value_t = 4096)]
        max_model_len: u32,
        #[arg(long, default_value_t = 16)]
        max_num_seqs: u32,
        #[arg(long, default_value_t = 2048)]
        max_num_batched_tokens: u32,
        /// KV block 数（P6 固定分配；按显存自动算留 P10）
        #[arg(long, default_value_t = 512)]
        num_blocks: u32,
        #[arg(long, default_value_t = 16)]
        block_size: u32,
        #[arg(long, default_value_t = 0)]
        device: usize,
        /// 开启 MTP 投机解码（nextn=1）
        #[arg(long, default_value_t = false)]
        mtp: bool,
    },
    /// 打印 CUDA 设备信息
    DeviceInfo {
        #[arg(long, default_value_t = 0)]
        ordinal: usize,
    },
}

fn model_name() -> &'static str {
    #[cfg(feature = "kimik3")]
    {
        dinfer_kimik3::MODEL_NAME
    }
    #[cfg(feature = "glm53")]
    {
        dinfer_glm53::MODEL_NAME
    }
    #[cfg(feature = "qwen3")]
    {
        dinfer_qwen3::MODEL_NAME
    }
    #[cfg(not(any(feature = "kimik3", feature = "glm53", feature = "qwen3")))]
    {
        "none"
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Serve {
            host,
            port,
            model,
            tokenizer,
            served_model_name,
            max_model_len,
            max_num_seqs,
            max_num_batched_tokens,
            num_blocks,
            block_size,
            device,
            mtp,
        } => {
            let name = served_model_name.unwrap_or_else(|| model_name().to_string());
            let addr = format!("{host}:{port}");
            #[cfg(feature = "glm53")]
            {
                let opt = dinfer_glm53::service::ServeOptions {
                    model_dir: model,
                    tokenizer_dir: tokenizer,
                    max_model_len,
                    max_num_seqs,
                    max_num_batched_tokens,
                    num_blocks,
                    block_size,
                    enable_chunked_prefill: true,
                    device,
                    mtp,
                };
                // 模型加载是阻塞的 CUDA 工作，放到 blocking 线程，避免卡 tokio runtime
                let svc = tokio::task::spawn_blocking(move || {
                    dinfer_glm53::service::Glm53Service::start(opt, name)
                })
                .await??;
                tracing::info!(%addr, "dinfer serving (no auth; bind to localhost unless fronted by a gateway)");
                dinfer_server::serve(&addr, std::sync::Arc::new(svc)).await?;
            }
            #[cfg(feature = "qwen3")]
            {
                let _ = mtp;
                let opt = dinfer_qwen3::service::ServeOptions {
                    model_dir: model,
                    tokenizer_dir: tokenizer,
                    max_model_len,
                    max_num_seqs,
                    max_num_batched_tokens,
                    num_blocks,
                    block_size,
                    enable_chunked_prefill: true,
                    device,
                };
                let svc = tokio::task::spawn_blocking(move || {
                    dinfer_qwen3::service::Qwen3Service::start(opt, name)
                })
                .await??;
                tracing::info!(%addr, "dinfer serving (no auth; bind to localhost unless fronted by a gateway)");
                dinfer_server::serve(&addr, std::sync::Arc::new(svc)).await?;
            }
            #[cfg(not(any(feature = "glm53", feature = "qwen3")))]
            {
                let _ = (
                    &addr,
                    model,
                    tokenizer,
                    max_model_len,
                    max_num_seqs,
                    max_num_batched_tokens,
                    num_blocks,
                    block_size,
                    device,
                    mtp,
                );
                anyhow::bail!(
                    "serve for model `{name}` is not implemented yet (build with --features glm53 or qwen3)"
                );
            }
        }
        Cmd::DeviceInfo { ordinal } => {
            let dev = dinfer_layers::Device::new(ordinal)?;
            let (maj, min) = dev.compute_capability()?;
            println!("device {ordinal}: {} sm_{maj}{min}", dev.name()?);
        }
    }
    Ok(())
}
