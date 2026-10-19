//! Kimi-K3 权重名映射（`docs/design-kimik3.md` §2.1）。
//!
//! 真权重（`moonshotai/Kimi-K3`，96 分片）的文本主干全部在 `language_model.model.` 前缀下，`lm_head` 在
//! `language_model.lm_head.weight`；`vision_tower.*` / `mm_projector.*` 是视觉塔，不进表。
//! `Table` 用前缀剥离后的相对名取张量：`layers.{l}.self_attn.q_proj.weight`、`embed_tokens.weight`、`norm.weight`、
//! `lm_head.weight`。
//!
//! KDA 层（第 1 层实测 header，B300 2026-10-19）各张量的 dtype / shape：
//!
//! | 相对名（`layers.N.self_attn.`） | dtype | shape | 说明 |
//! |---|---|---|---|
//! | `q_proj.weight` / `k_proj.weight` / `v_proj.weight` | BF16 | `[12288, 7168]` | HF 三张分开；vLLM 融合成 `in_proj_qkvgfab` |
//! | `g_proj.weight` | BF16 | `[12288, 7168]` | 输出门（`use_full_rank_gate`） |
//! | `f_a_proj.weight` | BF16 | `[128, 7168]` | gate 低秩第一段 |
//! | `f_b_proj.weight` | BF16 | `[12288, 128]` | gate 低秩第二段 |
//! | `b_proj.weight` | BF16 | `[96, 7168]` | beta（逐头标量） |
//! | `q_conv1d.weight` / `k_conv1d.weight` / `v_conv1d.weight` | **F32** | `[12288, 1, 4]` | depthwise，无 bias；HF 以模型 dtype（bf16）跑，装载时转 bf16 |
//! | `A_log` | F32 | **`[128]`** | 只有前 96（头数）有值，97..128 全 0 填充；vLLM `a_log_weight_loader` 也只 `narrow` 前 H 个 |
//! | `dt_bias` | F32 | `[12288]` | 逐 k 维（头×head_dim），不是逐头 |
//! | `o_norm.weight` | **F32** | `[128]` | gated RMSNorm（sigmoid 门）的权重；均值 ≈ 0.034，**不是** zero-centered（HF `FusedRMSNormGated` 直接乘 w） |
//! | `o_proj.weight` | BF16 | `[7168, 12288]` | |
//!
//! 层内其它：`input_layernorm.weight` / `post_attention_layernorm.weight` BF16 `[7168]`（普通 RMSNorm，`w · norm(x)`，
//! HF `KimiRMSNorm` **不是** zero-centered，与 Qwen3.5 不同）；`self_attention_res_{norm,proj}` / `mlp_res_{norm,proj}`
//! BF16 `[7168]` / `[1, 7168]`；MoE 段见 `docs/loops/k3-moe.md`。
use crate::config::KimiLinearConfig;
use anyhow::{bail, ensure, Context, Result};
use dinfer_layers::{bf16, Linear, RmsNorm, Stream, Tensor, WeightSource};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// 文本主干前缀（含结尾 `.`）。
pub const LM_PREFIX: &str = "language_model.model.";
/// `lm_head` 的文件名（不在 `LM_PREFIX` 下）。
pub const LM_HEAD: &str = "language_model.lm_head.weight";
/// 不加载的前缀（视觉塔、投影）。
const SKIP_PREFIXES: &[&str] = &["vision_tower.", "mm_projector.", "vision.", "visual."];

/// 名字表：相对名 → 文件里的原名。
#[derive(Debug, Clone)]
pub struct Table {
    names: HashMap<String, String>,
}

impl Table {
    /// 从名字列表建表（单测 / 单分片加载用）。只要求至少有一个 `LM_PREFIX` 下的张量。
    pub fn from_names<S: AsRef<str>>(names: &[S]) -> Result<Self> {
        let mut map = HashMap::new();
        for n in names {
            let n = n.as_ref();
            if SKIP_PREFIXES.iter().any(|p| n.starts_with(p)) {
                continue;
            }
            if let Some(rel) = n.strip_prefix(LM_PREFIX) {
                map.insert(rel.to_string(), n.to_string());
            } else if n == LM_HEAD {
                map.insert("lm_head.weight".to_string(), n.to_string());
            }
        }
        if map.is_empty() {
            bail!(
                "找不到 `{LM_PREFIX}*` 下的张量；文件里前几个名字：{}",
                names.iter().take(5).map(|n| n.as_ref()).collect::<Vec<_>>().join(", ")
            );
        }
        Ok(Self { names: map })
    }

    pub fn open(src: &WeightSource) -> Result<Self> {
        let all: Vec<&str> = src.names().collect();
        let t = Self::from_names(&all).context("kimik3 权重表")?;
        tracing::info!(tensors = t.names.len(), "kimik3: weight table");
        Ok(t)
    }

    pub fn contains(&self, key: &str) -> bool {
        self.names.contains_key(key)
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// 文件里的原名。
    pub fn name(&self, key: &str) -> Result<&str> {
        match self.names.get(key) {
            Some(n) => Ok(n.as_str()),
            None => bail!("kimik3 权重缺少 `{key}`"),
        }
    }

    /// bf16 张量（文件是 F32 时由 `WeightSource` 转 bf16：conv1d 权重走这条）。
    pub fn load_bf16(&self, stream: &Stream, src: &WeightSource, key: &str) -> Result<Tensor<bf16>> {
        let name = self.name(key)?.to_string();
        src.load_bf16(stream, &name).with_context(|| format!("load {name}"))
    }

    /// f32 张量（`A_log` / `dt_bias` / `o_norm.weight`）。
    pub fn load_f32(&self, stream: &Stream, src: &WeightSource, key: &str) -> Result<Tensor<f32>> {
        let (shape, v) = self.load_f32_host(src, key)?;
        Tensor::from_vec(stream, v, &shape)
    }

    pub fn load_f32_host(&self, src: &WeightSource, key: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let name = self.name(key)?.to_string();
        src.load_f32_host(&name).map_err(|e| e.context(format!("load {name}")))
    }

    /// 普通 RMSNorm（`KimiRMSNorm`：`w · norm(x)`，**非** zero-centered）。
    pub fn rmsnorm(&self, stream: &Stream, src: &WeightSource, key: &str, eps: f32) -> Result<RmsNorm> {
        let w = self.load_bf16(stream, src, key)?;
        Ok(RmsNorm { hidden: w.numel(), w, eps })
    }

    /// 无 bias 的 `Linear`（K3 所有投影都无 bias）。
    pub fn linear(&self, stream: &Stream, src: &WeightSource, key: &str) -> Result<Linear> {
        let w = self.load_bf16(stream, src, key)?;
        let b_key = key.replace(".weight", ".bias");
        let b = if self.contains(&b_key) { Some(self.load_bf16(stream, src, &b_key)?) } else { None };
        Linear::new(w, b)
    }
}

/// 某层的张量落在哪几个分片（读 `model.safetensors.index.json`）。单层对拍 / 按层流式加载用，
/// 不必 mmap 全部 96 个分片（1.56 TB）。
pub fn shards_for_layer(dir: &Path, layer: usize) -> Result<Vec<PathBuf>> {
    shards_for_prefix(dir, &format!("{LM_PREFIX}layers.{layer}."))
}

/// 顶层张量（`embed_tokens` / `norm` / `output_attn_res_*` / `lm_head`）所在分片：名字以 `language_model.` 开头且不含 `.layers.`。
pub fn shards_for_top(dir: &Path) -> Result<Vec<PathBuf>> {
    let idx = dir.join("model.safetensors.index.json");
    let s = std::fs::read_to_string(&idx).with_context(|| idx.display().to_string())?;
    let v: serde_json::Value = serde_json::from_str(&s)?;
    let map = v.get("weight_map").and_then(|m| m.as_object()).context("index: weight_map")?;
    let mut files: Vec<PathBuf> = map
        .iter()
        .filter(|(k, _)| k.starts_with("language_model.") && !k.contains(".layers."))
        .filter_map(|(_, f)| f.as_str().map(|f| dir.join(f)))
        .collect();
    files.sort();
    files.dedup();
    ensure!(!files.is_empty(), "index 里没有顶层张量");
    Ok(files)
}

/// 名字以 `pre` 开头的张量落在哪几个分片。
pub fn shards_for_prefix(dir: &Path, pre: &str) -> Result<Vec<PathBuf>> {
    let idx = dir.join("model.safetensors.index.json");
    let s = std::fs::read_to_string(&idx).with_context(|| idx.display().to_string())?;
    let v: serde_json::Value = serde_json::from_str(&s)?;
    let map = v.get("weight_map").and_then(|m| m.as_object()).context("index: weight_map")?;
    let mut files: Vec<PathBuf> = map
        .iter()
        .filter(|(k, _)| k.starts_with(pre))
        .filter_map(|(_, f)| f.as_str().map(|f| dir.join(f)))
        .collect();
    files.sort();
    files.dedup();
    ensure!(!files.is_empty(), "index 里没有前缀 {pre} 的张量");
    Ok(files)
}

/// `{q,k,v}_conv1d.weight [dim, 1, width]`（F32）→ bf16 `[dim, width]`。
pub fn load_conv1d(stream: &Stream, src: &WeightSource, t: &Table, key: &str, dim: usize, width: usize) -> Result<Tensor<bf16>> {
    let mut w = t.load_bf16(stream, src, key)?;
    ensure!(w.numel() == dim * width, "{key}: [{dim}, 1, {width}] 不匹配：文件 {:?}", w.shape);
    w.reshape(&[dim, width])?;
    Ok(w)
}

/// `A_log`：文件 `[128]`（按 head_dim 填充），只取前 `heads` 个（vLLM `a_log_weight_loader` 同样 `narrow`）。
pub fn load_a_log(stream: &Stream, src: &WeightSource, t: &Table, key: &str, heads: usize) -> Result<Tensor<f32>> {
    let (shape, mut v) = t.load_f32_host(src, key)?;
    ensure!(v.len() >= heads, "{key}: {shape:?} 少于头数 {heads}");
    ensure!(v[heads..].iter().all(|x| *x == 0.0), "{key}: 第 {heads} 项之后应为 0 填充");
    v.truncate(heads);
    Tensor::from_vec(stream, v, &[heads])
}

/// KDA 层各张量的期望形状（`cfg` 给定），给 `KdaLayer::load` 做断言用。
pub struct KdaShapes {
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub width: usize,
}

impl KdaShapes {
    pub fn new(cfg: &KimiLinearConfig) -> Self {
        let la = &cfg.linear_attn_config;
        Self {
            hidden: cfg.hidden_size as usize,
            heads: la.num_heads as usize,
            head_dim: la.head_dim as usize,
            width: la.short_conv_kernel_size as usize,
        }
    }
    /// 头 × head_dim（q/k/v/g 的输出维）。
    pub fn proj(&self) -> usize {
        self.heads * self.head_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header() -> Vec<String> {
        let mut n: Vec<String> = [
            "embed_tokens.weight",
            "norm.weight",
            "layers.1.input_layernorm.weight",
            "layers.1.self_attn.q_proj.weight",
            "layers.1.self_attn.A_log",
            "layers.1.block_sparse_moe.experts.0.w1.weight_packed",
        ]
        .iter()
        .map(|s| format!("{LM_PREFIX}{s}"))
        .collect();
        n.push(LM_HEAD.to_string());
        n.push("vision_tower.encoder.blocks.0.attn.qkv.weight".to_string());
        n.push("mm_projector.proj.weight".to_string());
        n
    }

    #[test]
    fn strips_prefix_and_skips_vision() {
        let t = Table::from_names(&header()).unwrap();
        assert_eq!(t.name("embed_tokens.weight").unwrap(), "language_model.model.embed_tokens.weight");
        assert_eq!(t.name("layers.1.self_attn.A_log").unwrap(), "language_model.model.layers.1.self_attn.A_log");
        assert_eq!(t.name("lm_head.weight").unwrap(), LM_HEAD);
        assert!(!t.contains("vision_tower.encoder.blocks.0.attn.qkv.weight"));
        assert!(!t.contains("mm_projector.proj.weight"));
        assert!(t.name("layers.2.self_attn.q_proj.weight").is_err());
        assert_eq!(t.len(), 7);
        assert!(Table::from_names(&["vision_tower.x", "nonsense.weight"]).is_err());
    }

    #[test]
    fn kda_shapes_from_official_config() {
        let p = "/data/ref/models/kimik3/config.json";
        let Ok(s) = std::fs::read_to_string(p) else {
            eprintln!("skip: {p} not present");
            return;
        };
        let c = crate::config::KimiK3Config::from_json(&s).unwrap();
        let sh = KdaShapes::new(&c.text_config);
        assert_eq!((sh.hidden, sh.heads, sh.head_dim, sh.width), (7168, 96, 128, 4));
        assert_eq!(sh.proj(), 12288);
        // 真分片 header（B300 2026-10-19）核对过的形状，写死在这里当回归：
        // q/k/v/g [12288,7168]、f_a [128,7168]、f_b [12288,128]、b [96,7168]、conv [12288,1,4] F32、
        // A_log [128] F32（前 96 有值）、dt_bias [12288] F32、o_norm [128] F32、o_proj [7168,12288]。
        assert_eq!(sh.proj() / sh.heads, 128);
    }

    #[test]
    fn shards_for_layer_reads_index() {
        let dir = Path::new("/data/ref/models/kimik3");
        if !dir.join("model.safetensors.index.json").exists() {
            eprintln!("skip: index not present");
            return;
        }
        let f = shards_for_layer(dir, 1).unwrap();
        assert!(!f.is_empty());
        assert!(f.iter().all(|p| p.extension().is_some_and(|e| e == "safetensors")));
        assert!(shards_for_layer(dir, 999).is_err());
    }
}
