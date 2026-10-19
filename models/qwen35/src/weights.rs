//! Qwen35 权重名映射与融合（`docs/design-qwen35.md` §2.1）。
//!
//! 真权重是 `Qwen3_5ForConditionalGeneration`（多模态壳），safetensors 里的名字有两种可能的前缀：
//! - VL 训练栈：`model.language_model.layers.{l}...`（`lm_head.weight` 在顶层，不带前缀）；
//! - vLLM 侧 `Qwen3_5ForCausalLM`：`language_model.model.layers.{l}...`（`weigths_mapper` 把前者映射成后者）。
//!
//! `Table` 从文件里探测实际前缀，之后所有层都按"剥离前缀后的相对名"取：`layers.{l}.self_attn.q_proj.weight`、
//! `embed_tokens.weight`、`norm.weight`、`lm_head.weight`。`visual.*`（视觉）与 `mtp.*`（MTP 层）不进表。
//!
//! 融合（HF 里是分开的张量，kernel 要连续的）：
//! - **q 与输出门交织**：`q_proj [H·2D, hidden]` 的行是 `h·2D .. h·2D+D` = q、`+D .. +2D` = gate（vLLM
//!   `qwen3_next.py` / HF `modeling_qwen3_5.py` 的 `chunk(2, -1)`）。我们拼成 `[q(H·D) | gate(H·D) | k | v]`，
//!   `qkv_post_ext(qmul=2)` 只碰 q/k/v 段。
//! - `gate_proj` / `up_proj` 的合并由 `dinfer_layers::Mlp::load_tp` 做（同 Qwen3）。
//! - `conv1d.weight [dim, 1, width]` → `[dim, width]`（depthwise，无 bias）。

use crate::config::Qwen35Config;
use anyhow::{bail, ensure, Context, Result};
use dinfer_layers::{bf16, Linear, Stream, Tensor, WeightSource};
use std::collections::HashMap;

/// 真权重里的语言模型前缀（含结尾的 `.`）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Layout {
    /// `model.language_model.`（HF `Qwen3_5ForConditionalGeneration`）
    Vl,
    /// `language_model.model.`（vLLM `Qwen3_5ForCausalLM`）
    Llm,
    /// `model.`（裸文本模型 / tiny 权重）
    Plain,
}

impl Layout {
    pub fn prefix(&self) -> &'static str {
        match self {
            Layout::Vl => "model.language_model.",
            Layout::Llm => "language_model.model.",
            Layout::Plain => "model.",
        }
    }
}

/// 名字表：`key`（前缀剥离后的相对名）→ 文件里的原名。
#[derive(Debug, Clone)]
pub struct Table {
    pub layout: Layout,
    names: HashMap<String, String>,
}

/// 视觉塔与 MTP 层：列进 header 但我们不加载。
const SKIP_PREFIXES: &[&str] = &["visual.", "vision.", "mtp.", "model.visual.", "model.mtp."];

impl Table {
    /// 从名字列表建表（单测用）。
    pub fn from_names<S: AsRef<str>>(names: &[S]) -> Result<Self> {
        let has = |p: &str| names.iter().any(|n| n.as_ref() == format!("{p}embed_tokens.weight"));
        let layout = if has("model.language_model.") {
            Layout::Vl
        } else if has("language_model.model.") {
            Layout::Llm
        } else if has("model.") {
            Layout::Plain
        } else {
            bail!(
                "找不到语言模型的 embed_tokens.weight（试过 model.language_model. / language_model.model. / model.）；\
                 文件里前几个名字：{}",
                names.iter().take(5).map(|n| n.as_ref()).collect::<Vec<_>>().join(", ")
            );
        };
        Ok(Self::with_layout(layout, names.iter().map(|n| n.as_ref().to_string())))
    }

    fn with_layout<I: Iterator<Item = String>>(layout: Layout, names: I) -> Self {
        let pre = layout.prefix();
        let mut map = HashMap::new();
        for n in names {
            if SKIP_PREFIXES.iter().any(|p| n.starts_with(p)) {
                continue;
            }
            if let Some(rel) = n.strip_prefix(pre).map(str::to_string) {
                map.insert(rel.clone(), n.clone());
                let _ = rel;
            } else if n == "lm_head.weight" {
                // 顶层 lm_head（VL 布局下不在语言模型前缀里）
                map.insert(n.clone(), n);
            }
        }
        Self { layout, names: map }
    }

    pub fn open(src: &WeightSource) -> Result<Self> {
        let all: Vec<&str> = src.names().collect();
        let t = Self::from_names(&all).context("qwen35 权重表")?;
        tracing::info!(layout = ?t.layout, tensors = t.names.len(), "qwen35: weight layout");
        Ok(t)
    }

    pub fn contains(&self, key: &str) -> bool {
        self.names.contains_key(key)
    }

    /// 文件里的原名（`WeightSource::load_*` 用）。
    pub fn name(&self, key: &str) -> Result<&str> {
        match self.names.get(key) {
            Some(n) => Ok(n.as_str()),
            None => bail!("qwen35 权重缺少 `{}`（layout {:?}）", key, self.layout),
        }
    }

    pub fn load_bf16(&self, stream: &Stream, src: &WeightSource, key: &str) -> Result<Tensor<bf16>> {
        let name = self.name(key)?.to_string();
        src.load_bf16(stream, &name).with_context(|| format!("load {name}"))
    }

    fn load_bf16_shard(&self, stream: &Stream, src: &WeightSource, key: &str, dim: usize, rank: usize, world: usize) -> Result<Tensor<bf16>> {
        let name = self.name(key)?.to_string();
        src.load_bf16_shard(stream, &name, dim, rank, world).with_context(|| format!("load {name}"))
    }

    /// 给 `Mlp::load_tp` 这类"按前缀拼 `.gate_proj.weight`"的装载器用：返回文件里对应的前缀
    /// （如 `layers.0.mlp` → `model.language_model.layers.0.mlp`）。表里没有就退回 key 本身。
    pub fn prefix_of(&self, key: &str) -> String {
        match self.names.get(&format!("{key}.down_proj.weight")) {
            Some(n) => n.trim_end_matches(".down_proj.weight").to_string(),
            None => key.to_string(),
        }
    }

    /// embedding 表（`dinfer_layers::Embedding::load` 签名要的是不带 `.weight` 的前缀）。
    pub fn embedding(&self, stream: &Stream, src: &WeightSource, key: &str) -> Result<dinfer_layers::Embedding> {
        let name = self.name(key)?;
        let table = src.load_bf16(stream, name).with_context(|| format!("load {name}"))?;
        Ok(dinfer_layers::Embedding {
            vocab: table.shape[0],
            hidden: table.shape[1],
            table,
        })
    }

    /// RMSNorm（同 `RmsNorm::load` 的前缀约定）。
    pub fn rmsnorm(&self, stream: &Stream, src: &WeightSource, key: &str, eps: f32) -> Result<dinfer_layers::RmsNorm> {
        let w = self.load_bf16(stream, src, key)?;
        Ok(dinfer_layers::RmsNorm { hidden: w.numel(), w, eps })
    }

    /// f32 张量（`A_log` / `dt_bias`，真权重是 f32）。
    pub fn load_f32(&self, stream: &Stream, src: &WeightSource, key: &str) -> Result<Tensor<f32>> {
        let name = self.name(key)?.to_string();
        let (shape, v) = src.load_f32_host(&name).map_err(|e| e.context(format!("load {name}")))?;
        Ok(Tensor::from_vec(stream, v, &shape)?)
    }

    /// `Linear`：按输出行切（`dim = 0`）或输入列切（`dim = 1`）的 TP 分片。
    pub fn linear(&self, stream: &Stream, src: &WeightSource, key: &str, dim: Option<(usize, (usize, usize))>) -> Result<Linear> {
        let w = match dim {
            None => self.load_bf16(stream, src, key)?,
            Some((d, (rank, world))) => self.load_bf16_shard(stream, src, key, d, rank, world)?,
        };
        let b_key = key.replace(".weight", ".bias");
        let b = if self.contains(&b_key) { Some(self.load_bf16(stream, src, &b_key)?) } else { None };
        Linear::new(w, b)
    }
}

/// q_proj 的 per-head 交织→分段：第 `h` 头在 q 段的行起点。
pub fn q_src_row(h: usize, head_dim: usize) -> usize {
    h * 2 * head_dim
}
/// 第 `h` 头在 gate 段的源行起点。
pub fn gate_src_row(h: usize, head_dim: usize) -> usize {
    h * 2 * head_dim + head_dim
}

/// 拼 `[q(H·D) | k(Hkv·D) | v(Hkv·D)]` 的融合 qkv 权重 `[(H + 2Hkv)·D, hidden]`，`qmul = 1` 路径。
/// TP：`tp = Some((rank, world))` 时按头切——本 rank 拿 `H/world` 个 q 头与 `Hkv/world` 个 kv 头。
///
/// 输出门单独一条 GEMM（`load_gate`），不塞进 qkv 行里：`qkv_post_ext(qmul=2)` 的 gate 段在行内偏移固定、
/// 每行有 stride，attention 之后要用 `mul_sigmoid` 就得按行搬——拆成两条紧凑 GEMM 省掉这一步（FLOPs 相同）。
pub fn load_qkv(
    stream: &Stream,
    src: &WeightSource,
    t: &Table,
    layer: usize,
    cfg: &Qwen35Config,
    tp: (usize, usize),
) -> Result<Linear> {
    let (rank, world) = tp;
    let (h_all, hk_all, d) =
        (cfg.num_attention_heads as usize, cfg.num_key_value_heads as usize, cfg.head_dim as usize);
    ensure!(world >= 1 && rank < world, "bad tp shard {rank}/{world}");
    let (h, hk) = (h_all / world, hk_all / world);
    let hidden = cfg.hidden_size as usize;
    let p = format!("layers.{layer}.self_attn");
    let q = t.load_bf16(stream, src, &format!("{p}.q_proj.weight"))?;
    let k = t.load_bf16(stream, src, &format!("{p}.k_proj.weight"))?;
    let v = t.load_bf16(stream, src, &format!("{p}.v_proj.weight"))?;
    ensure!(q.shape == [h_all * 2 * d, hidden], "q_proj shape {:?}（期望 [{}, {hidden}]）", q.shape, h_all * 2 * d);
    ensure!(k.shape == [hk_all * d, hidden] && v.shape == k.shape, "k/v_proj shape {:?} {:?}", k.shape, v.shape);
    let row = (h + 2 * hk) * d;
    let mut w = Tensor::<bf16>::zeros(stream, &[row, hidden])?;
    let m = stream.inner_arc();
    // q：本 rank 的头 = [rank·h, (rank+1)·h)，源行按 per-head 交错取（第 hh 头 q 段起点 = hh·2D）
    for hh in 0..h {
        let src_row = q_src_row(rank * h + hh, d);
        let n = d * hidden;
        let mut dst = w.data.slice_mut(hh * d * hidden..hh * d * hidden + n);
        m.memcpy_dtod(&q.data.slice(src_row * hidden..src_row * hidden + n), &mut dst)?;
    }
    // k / v：按 kv 头切
    for (dst_head, part) in [(h, &k), (h + hk, &v)] {
        let lo = rank * hk * d * hidden;
        let n = hk * d * hidden;
        let mut dst = w.data.slice_mut(dst_head * d * hidden..dst_head * d * hidden + n);
        m.memcpy_dtod(&part.data.slice(lo..lo + n), &mut dst)?;
    }
    stream.synchronize()?;
    Linear::new(w, None)
}

/// 输出门权重 `[H·D, hidden]`：从 `q_proj` 的 per-head 交织里取 gate 段（与 `load_qkv` 的 q 段互补）。
pub fn load_gate(
    stream: &Stream,
    src: &WeightSource,
    t: &Table,
    layer: usize,
    cfg: &Qwen35Config,
    tp: (usize, usize),
) -> Result<Linear> {
    let (rank, world) = tp;
    let (h_all, d) = (cfg.num_attention_heads as usize, cfg.head_dim as usize);
    ensure!(cfg.attn_output_gate, "load_gate: attn_output_gate = false");
    let h = h_all / world;
    let hidden = cfg.hidden_size as usize;
    let q = t.load_bf16(stream, src, &format!("layers.{layer}.self_attn.q_proj.weight"))?;
    let mut w = Tensor::<bf16>::zeros(stream, &[h * d, hidden])?;
    let m = stream.inner_arc();
    for hh in 0..h {
        let src_row = gate_src_row(rank * h + hh, d);
        let n = d * hidden;
        let mut dst = w.data.slice_mut(hh * d * hidden..hh * d * hidden + n);
        m.memcpy_dtod(&q.data.slice(src_row * hidden..src_row * hidden + n), &mut dst)?;
    }
    stream.synchronize()?;
    Linear::new(w, None)
}

/// `conv1d.weight [dim, 1, width]` → `[dim, width]`（depthwise，无 bias）。
pub fn load_conv1d(
    stream: &Stream,
    src: &WeightSource,
    t: &Table,
    layer: usize,
    dim: usize,
    width: usize,
) -> Result<Tensor<bf16>> {
    let w = t.load_bf16(stream, src, &format!("layers.{layer}.linear_attn.conv1d.weight"))?;
    ensure!(
        w.numel() == dim * width,
        "conv1d [{dim}, 1, {width}] 不匹配：文件 {:?}",
        w.shape
    );
    let mut w = w;
    w.reshape(&[dim, width])?;
    Ok(w)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vl_header() -> Vec<String> {
        let mut n = vec![
            "model.language_model.embed_tokens.weight".to_string(),
            "model.language_model.layers.0.self_attn.q_proj.weight".to_string(),
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight".to_string(),
            "model.language_model.layers.0.linear_attn.A_log".to_string(),
            "model.language_model.norm.weight".to_string(),
            "lm_head.weight".to_string(),
            "visual.patch_embed.proj.weight".to_string(),
            "model.visual.blocks.0.attn.qkv.weight".to_string(),
            "mtp.layers.0.mlp.down_proj.weight".to_string(),
        ];
        n.sort();
        n
    }

    #[test]
    fn detect_vl_layout_and_strip() {
        let t = Table::from_names(&vl_header()).unwrap();
        assert_eq!(t.layout, Layout::Vl);
        assert_eq!(t.name("embed_tokens.weight").unwrap(), "model.language_model.embed_tokens.weight");
        assert_eq!(
            t.name("layers.0.self_attn.q_proj.weight").unwrap(),
            "model.language_model.layers.0.self_attn.q_proj.weight"
        );
        // 顶层 lm_head 保留原名
        assert_eq!(t.name("lm_head.weight").unwrap(), "lm_head.weight");
        // 视觉 / MTP 不进表
        assert!(!t.contains("visual.patch_embed.proj.weight"));
        assert!(!t.contains("model.visual.blocks.0.attn.qkv.weight"));
        assert!(!t.contains("layers.0.mlp.down_proj.weight"));
        assert!(t.name("layers.9.mlp.down_proj.weight").is_err());
    }

    #[test]
    fn detect_llm_and_plain_layouts() {
        let llm = ["language_model.model.embed_tokens.weight", "language_model.model.norm.weight"];
        let t = Table::from_names(&llm).unwrap();
        assert_eq!(t.layout, Layout::Llm);
        assert_eq!(t.name("norm.weight").unwrap(), "language_model.model.norm.weight");

        let plain = ["model.embed_tokens.weight", "model.layers.1.mlp.gate_proj.weight", "lm_head.weight"];
        let t = Table::from_names(&plain).unwrap();
        assert_eq!(t.layout, Layout::Plain);
        assert_eq!(t.name("layers.1.mlp.gate_proj.weight").unwrap(), "model.layers.1.mlp.gate_proj.weight");

        assert!(Table::from_names(&["nonsense.weight"]).is_err());
    }

    /// per-head 交织：源行 h·2D 是 q、h·2D+D 是 gate；D=4、3 个头时 q 段 = 4·(3·4)。
    #[test]
    fn q_gate_deinterleave_rows() {
        let d = 4;
        let src: Vec<usize> = (0..3).map(|h| q_src_row(h, d)).collect();
        assert_eq!(src, vec![0, 8, 16]);
        let gate: Vec<usize> = (0..3).map(|h| gate_src_row(h, d)).collect();
        assert_eq!(gate, vec![4, 12, 20]);
        // 每个头 2D 行，24 头 × 2 × 256 = 12288 行 = q_proj 的输出维
        assert_eq!(q_src_row(23, 256) + 2 * 256, 12288);
    }
}
