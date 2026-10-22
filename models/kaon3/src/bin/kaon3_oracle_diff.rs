//! 两份 vLLM 格式 oracle（`/v1/completions` 带 `prompt_logprobs` 的响应）互比：标定数值噪声下限。
//! 用法：kaon3_oracle_diff <a.json> <b.json>   # a 当参照
//! 指标与 `kaon3_probe --oracle` 同口径：a 的可判定行（top1 − top2 ≥ 0.5）上 top-1 是否一致；a 的 top-5 在 b 里也出现的
//! token 的 |Δlogprob| 均值。用途：同一模型两种合法实现（如 vLLM 开 / 关 batch-invariant）之间的差异 = 我们能逼近的极限。
use anyhow::{Context, Result};
use serde_json::Value;

fn rows(path: &str) -> Result<Vec<Vec<(String, f64)>>> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).with_context(|| format!("读 {path}"))?)?;
    let ch = &v["choices"][0];
    let mut out = Vec::new();
    for e in ch["prompt_logprobs"].as_array().context("prompt_logprobs")?.iter().skip(1) {
        let mut r: Vec<(String, f64)> = e
            .as_object()
            .context("行")?
            .iter()
            .filter(|(_, x)| x["rank"].as_u64().unwrap_or(99) <= 5)
            .map(|(k, x)| (k.clone(), x["logprob"].as_f64().unwrap_or(f64::NAN)))
            .collect();
        r.sort_by(|a, b| b.1.total_cmp(&a.1));
        out.push(r);
    }
    let mut last: Vec<(String, f64)> = ch["logprobs"]["top_logprobs"][0]
        .as_object()
        .context("top_logprobs")?
        .iter()
        .map(|(k, x)| (k.trim_start_matches("token_id:").to_string(), x.as_f64().unwrap_or(f64::NAN)))
        .collect();
    last.sort_by(|a, b| b.1.total_cmp(&a.1));
    out.push(last);
    Ok(out)
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    anyhow::ensure!(a.len() == 2, "usage: kaon3_oracle_diff <a.json> <b.json>");
    let (ra, rb) = (rows(&a[0])?, rows(&a[1])?);
    anyhow::ensure!(ra.len() == rb.len(), "行数不同 {} vs {}", ra.len(), rb.len());
    let (mut dec, mut miss, mut sd, mut nd) = (0, 0, 0.0, 0usize);
    for (x, y) in ra.iter().zip(&rb) {
        if x.len() < 2 || y.is_empty() {
            continue;
        }
        if x[0].1 - x[1].1 >= 0.5 {
            dec += 1;
            miss += (x[0].0 != y[0].0) as usize;
        }
        for (id, l) in x {
            if let Some((_, m)) = y.iter().find(|(j, _)| j == id) {
                sd += (l - m).abs();
                nd += 1;
            }
        }
    }
    println!(
        "ORACLE-DIFF rows {} | 可判定 {dec} 行 top1 不一致 {miss}（{:.1}%）| 共有 top-5 token |Δlp| 均值 {:.4}（{nd} 对）",
        ra.len(),
        100.0 * miss as f64 / dec.max(1) as f64,
        sd / nd.max(1) as f64
    );
    Ok(())
}
