//! dinfer 入口。`serve`：加载模型 + tokenizer，起引擎线程与 OpenAI 兼容 HTTP 服务；`device-info`：打印 GPU 信息。

use anyhow::Result;
use clap::{Parser, Subcommand};

#[cfg(any(
    all(feature = "kimik3", feature = "glm53"),
    all(feature = "kimik3", feature = "qwen3"),
    all(feature = "glm53", feature = "qwen3"),
    all(feature = "kimik3", feature = "qwen35"),
    all(feature = "glm53", feature = "qwen35"),
    all(feature = "qwen3", feature = "qwen35")
))]
compile_error!("enable exactly one model feature: `kimik3`, `glm53`, `qwen3` or `qwen35`");

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
        /// KV block 数；不传 = 按显存自动算（qwen3；glm53 暂为固定 512）
        #[arg(long)]
        num_blocks: Option<u32>,
        #[arg(long, default_value_t = 16)]
        block_size: u32,
        /// 自动算 num_blocks 时可用显存比例
        #[arg(long, default_value_t = 0.9)]
        gpu_memory_utilization: f32,
        /// 关闭 prefix cache（默认开）
        #[arg(long, default_value_t = false)]
        no_prefix_caching: bool,
        /// 关闭 "decode 行不占 token 预算"（默认开：每步装下全部 decode + 预算内 prefill，使跨流并行生效）
        #[arg(long, default_value_t = false)]
        no_decode_exempt: bool,
        /// KV host tier：用这么多 GiB 的 pinned 主机内存做 prefix cache 的第二层（0 = 关）
        #[arg(long, default_value_t = 0.0)]
        kv_host_gib: f64,
        /// PD 分离角色：single（默认）| prefill | decode
        #[arg(long, default_value = "single")]
        role: String,
        /// PD：本实例收 KV 的端口（decode 必填；0 = 不监听）
        #[arg(long, default_value_t = 0)]
        kv_port: u16,
        /// PD prefill 侧：对端 decode 的 host:kv_port
        #[arg(long)]
        peer: Option<String>,
        /// PD decode 侧：等 KV 超时（ms），超时本地重算 prefill
        #[arg(long, default_value_t = 2000)]
        pd_wait_ms: u64,
        /// PD：pinned 缓冲个数（每块 = max_model_len 全层 KV，Qwen3 4k ≈ 448 MB；= 导出 / 导入在飞上限）
        #[arg(long, default_value_t = 8)]
        pd_pool_bufs: usize,
        /// PD KV 传输后端：tcp（默认，payload 走 TCP）| shm（D 建 /dev/shm 槽区，P 直接 D2H 写入，跨卡可用）| cuda-ipc（同卡 D2D）
        #[arg(long, default_value = "tcp")]
        kv_transport: String,
        /// PD 共享槽区槽数（0 = 自动：shm 4 / cuda-ipc 2；每槽 = max_model_len 全层 KV）
        #[arg(long, default_value_t = 0)]
        kv_slots: usize,
        /// 关闭 decode CUDA graph（默认开；qwen3）
        #[arg(long, default_value_t = false)]
        no_cuda_graph: bool,
        #[arg(long, default_value_t = 0)]
        device: usize,
        /// 张量并行组大小（qwen3；design §13）：用卡 device..device+tp
        #[arg(long, default_value_t = 1)]
        tp: usize,
        /// 数据并行副本数（qwen3）：副本 i 用卡 device+i*tp..；同进程按在飞数路由
        #[arg(long, default_value_t = 1)]
        dp: usize,
        /// 开启 MTP 投机解码（nextn=1）
        #[arg(long, default_value_t = false)]
        mtp: bool,
        /// 投机解码方法（qwen3）：none（默认）| ngram（无需额外权重，docs/loops/ngram-spec.md）
        #[arg(long, default_value = "none")]
        speculative: String,
        /// 投机解码每步 draft 数 K
        #[arg(long, default_value_t = 3)]
        num_speculative_tokens: usize,
        /// n-gram 最短匹配长度
        #[arg(long, default_value_t = 2)]
        ngram_min: usize,
        /// n-gram 最长匹配长度
        #[arg(long, default_value_t = 4)]
        ngram_max: usize,
    },
    /// PD 分离入口（design §10.3）：收 OpenAI 请求 → 打 prefill 实例出首 token + 导出 KV → 打 decode 实例接力。
    /// 任一侧失败退化为 decode 实例单机跑（不丢请求）。无鉴权。
    Proxy {
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8100)]
        port: u16,
        /// prefill 实例 base URL，如 http://127.0.0.1:8101
        #[arg(long)]
        prefill: String,
        /// decode 实例 base URL，如 http://127.0.0.1:8102
        #[arg(long)]
        decode: String,
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
    #[cfg(feature = "qwen35")]
    {
        dinfer_qwen35::MODEL_NAME
    }
    #[cfg(not(any(feature = "kimik3", feature = "glm53", feature = "qwen3", feature = "qwen35")))]
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
            gpu_memory_utilization,
            no_prefix_caching,

            no_decode_exempt,


            kv_host_gib,
            role,
            kv_port,
            peer,
            pd_wait_ms,
            pd_pool_bufs,
            kv_transport,
            kv_slots,
            no_cuda_graph,
            device,
            tp,
            dp,
            mtp,
            speculative,
            num_speculative_tokens,
            ngram_min,
            ngram_max,
        } => {
            let name = served_model_name.unwrap_or_else(|| model_name().to_string());
            let addr = format!("{host}:{port}");
            #[cfg(feature = "glm53")]
            {
                let _ = (gpu_memory_utilization, no_prefix_caching, no_cuda_graph, &speculative, num_speculative_tokens, ngram_min, ngram_max); // glm53：固定 num_blocks，prefix cache / graph 待接
                let opt = dinfer_glm53::service::ServeOptions {
                    model_dir: model,
                    tokenizer_dir: tokenizer,
                    max_model_len,
                    max_num_seqs,
                    max_num_batched_tokens,
                    num_blocks: num_blocks.unwrap_or(512),
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
                let _ = (mtp, &role, kv_port, &peer, pd_wait_ms, pd_pool_bufs, &kv_transport, kv_slots);
                let opt = dinfer_qwen3::service::ServeOptions {
                    model_dir: model,
                    tokenizer_dir: tokenizer,
                    max_model_len,
                    max_num_seqs,
                    max_num_batched_tokens,
                    num_blocks,
                    block_size,
                    gpu_memory_utilization,
                    enable_chunked_prefill: true,
                    enable_prefix_caching: !no_prefix_caching,
                    decode_exempt_from_budget: !no_decode_exempt,
                    kv_host_gib,
                    role: role.parse().map_err(anyhow::Error::msg)?,
                    kv_port,
                    peer,
                    pd_wait_ms,
                    pd_pool_bufs,
                    kv_transport,
                    kv_slots,
                    enable_cuda_graph: !no_cuda_graph,
                    device,
                    tp: tp.max(1),
                    speculative,
                    num_speculative_tokens,
                    ngram_min,
                    ngram_max,
                };
                let dp = dp.max(1);
                // DP（design §13.3）：N 个独立副本并行装载，副本 i 用卡 device + i·tp
                let mut joins = Vec::with_capacity(dp);
                for i in 0..dp {
                    let mut o = opt.clone();
                    o.device = device + i * opt.tp;
                    let name = name.clone();
                    joins.push(tokio::task::spawn_blocking(move || dinfer_qwen3::service::Qwen3Service::start(o, name)));
                }
                let mut shards: Vec<std::sync::Arc<dyn dinfer_server::backend::ChatBackend>> = Vec::with_capacity(dp);
                for j in joins {
                    shards.push(std::sync::Arc::new(j.await??));
                }
                tracing::info!(%addr, dp, tp = opt.tp, "dinfer serving (no auth; bind to localhost unless fronted by a gateway)");
                if shards.len() == 1 {
                    dinfer_server::serve(&addr, shards.pop().unwrap()).await?;
                } else {
                    dinfer_server::serve(&addr, std::sync::Arc::new(dinfer_server::dp::DpBackend::new(shards))).await?;
                }
            }
            #[cfg(feature = "qwen35")]
            {
                let _ = (
                    &role, kv_port, &peer, pd_wait_ms, pd_pool_bufs, &kv_transport, kv_slots, tp, dp, mtp, no_cuda_graph,
                    &speculative, num_speculative_tokens, ngram_min, ngram_max, kv_host_gib,
                );
                let opt = dinfer_qwen35::service::ServeOptions {
                    model_dir: model,
                    tokenizer_dir: tokenizer,
                    max_model_len,
                    max_num_seqs,
                    max_num_batched_tokens,
                    num_blocks,
                    block_size,
                    gpu_memory_utilization,
                    enable_chunked_prefill: true,
                    enable_prefix_caching: !no_prefix_caching,
                    decode_exempt_from_budget: !no_decode_exempt,
                    device,
                };
                let name = name.clone();
                let svc = tokio::task::spawn_blocking(move || {
                    dinfer_qwen35::service::Qwen35Service::start(opt, name)
                })
                .await??;
                tracing::info!(%addr, "dinfer serving (no auth; bind to localhost unless fronted by a gateway)");
                dinfer_server::serve(&addr, std::sync::Arc::new(svc)).await?;
            }
            #[cfg(not(any(feature = "glm53", feature = "qwen3", feature = "qwen35")))]
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
                    gpu_memory_utilization,
                    no_prefix_caching,
                    no_cuda_graph,
                    device,
                    mtp,
                    &speculative,
                    num_speculative_tokens,
                    ngram_min,
                    ngram_max,
                );
                anyhow::bail!(
                    "serve for model `{name}` is not implemented yet (build with --features glm53, qwen3 or qwen35)"
                );
            }
        }
        Cmd::Proxy { host, port, prefill, decode } => {
            let addr = format!("{host}:{port}");
            tracing::info!(%addr, %prefill, %decode, "dinfer PD proxy (no auth)");
            dinfer_server::proxy::serve(&addr, prefill, decode).await?;
        }
        Cmd::DeviceInfo { ordinal } => {
            let dev = dinfer_layers::Device::new(ordinal)?;
            let (maj, min) = dev.compute_capability()?;
            println!("device {ordinal}: {} sm_{maj}{min}", dev.name()?);
        }
    }
    Ok(())
}
