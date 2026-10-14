//! dinfer 入口。P0：`serve` 起一个只有 /health /v1/models 的服务；`device-info` 打印 GPU 信息。

use anyhow::Result;
use clap::{Parser, Subcommand};

#[cfg(all(feature = "kimik3", feature = "glm53"))]
compile_error!("enable exactly one model feature: `kimik3` or `glm53`");

#[derive(Parser)]
#[command(name = "dinfer", version, about = "Rust LLM inference engine for Kimi-K3 / GLM-5.3")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// 启动 HTTP 服务
    Serve {
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8000)]
        port: u16,
        #[arg(long)]
        served_model_name: Option<String>,
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
    #[cfg(not(any(feature = "kimik3", feature = "glm53")))]
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
        Cmd::Serve { host, port, served_model_name } => {
            let name = served_model_name.unwrap_or_else(|| model_name().to_string());
            let addr = format!("{host}:{port}");
            tracing::info!(%addr, model = %name, "dinfer serving (no auth; bind to localhost unless fronted by a gateway)");
            dinfer_server::serve(&addr, name).await?;
        }
        Cmd::DeviceInfo { ordinal } => {
            let dev = dinfer_layers::Device::new(ordinal)?;
            let (maj, min) = dev.compute_capability()?;
            println!("device {ordinal}: {} sm_{maj}{min}", dev.name()?);
        }
    }
    Ok(())
}
