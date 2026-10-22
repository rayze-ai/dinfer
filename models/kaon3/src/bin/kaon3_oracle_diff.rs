//! 两份 vLLM 格式 oracle（`/v1/completions` 带 `prompt_logprobs` 的响应）互比：标定数值噪声下限 / 判定一致性。
//!
//! 用法：
//!   kaon3_oracle_diff <a.json> <b.json> [a.ids]     # 单个用例；a 当参照
//!   kaon3_oracle_diff <dir_a> <dir_b> [dir_filter]  # 目录汇总（只取三处都有的用例；>512 行的长序列单列 LONG、不进汇总）
//!                                                  # 原：目录汇总：dir_a 里每个 <name>.json（配 <name>.ids）对 dir_b 同名文件
//!
//! 指标（与 `kaon3_probe --oracle` 同口径）：a 的可判定行（top1 − top2 ≥ 0.5）上 top-1 是否一致；a 的 top-5 在 b 里也出现的
//! token 的 |Δlogprob| 均值；以及双方**实际下一个 token** 的平均负对数似然（NLL）。NLL 对局部离散翻转不敏感，
//! 是量化推理（W4A4 / W8A8）下比逐行 top-1 更稳的质量指标。汇总模式最后打一行 `POOLED k=v ...` 供 verify 脚本解析。
use anyhow::{Context, Result};
use serde_json::Value;
use std::path::Path;

/// 每行：top-5（降序）+ 实际 token 的 logprob（若 ids 给出且条目里有）。
struct Row {
    top: Vec<(String, f64)>,
    actual: Option<f64>,
}

fn rows(path: &Path, ids: Option<&[String]>) -> Result<Vec<Row>> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).with_context(|| format!("读 {}", path.display()))?)?;
    let ch = &v["choices"][0];
    let pl = ch["prompt_logprobs"].as_array().context("prompt_logprobs")?;
    let mut out = Vec::new();
    for (p, e) in pl.iter().enumerate().skip(1) {
        let obj = e.as_object().context("行")?;
        let mut top: Vec<(String, f64)> = obj
            .iter()
            .filter(|(_, x)| x["rank"].as_u64().unwrap_or(99) <= 5)
            .map(|(k, x)| (k.clone(), x["logprob"].as_f64().unwrap_or(f64::NAN)))
            .collect();
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
        let actual = ids.and_then(|ids| ids.get(p)).and_then(|id| obj.get(id)).and_then(|x| x["logprob"].as_f64());
        out.push(Row { top, actual });
    }
    let mut last: Vec<(String, f64)> = ch["logprobs"]["top_logprobs"][0]
        .as_object()
        .context("top_logprobs")?
        .iter()
        .map(|(k, x)| (k.trim_start_matches("token_id:").to_string(), x.as_f64().unwrap_or(f64::NAN)))
        .collect();
    last.sort_by(|a, b| b.1.total_cmp(&a.1));
    out.push(Row { top: last, actual: None });
    Ok(out)
}

#[derive(Default)]
struct Acc {
    rows: usize,
    dec: usize,
    miss: usize,
    sd: f64,
    nd: usize,
    nll_a: f64,
    nll_b: f64,
    na: usize,
}

impl Acc {
    fn add(&mut self, ra: &[Row], rb: &[Row]) {
        self.rows += ra.len();
        for (x, y) in ra.iter().zip(rb) {
            if x.top.len() >= 2 && !y.top.is_empty() && x.top[0].1 - x.top[1].1 >= 0.5 {
                self.dec += 1;
                self.miss += (x.top[0].0 != y.top[0].0) as usize;
            }
            for (id, l) in &x.top {
                if let Some((_, m)) = y.top.iter().find(|(j, _)| j == id) {
                    self.sd += (l - m).abs();
                    self.nd += 1;
                }
            }
            if let (Some(a), Some(b)) = (x.actual, y.actual) {
                self.nll_a -= a;
                self.nll_b -= b;
                self.na += 1;
            }
        }
    }
    fn line(&self, tag: &str) -> String {
        let n = self.na.max(1) as f64;
        format!(
            "{tag} rows={} decisive={} flips={} flip_pct={:.2} mean_common_d={:.4} nll_a={:.4} nll_b={:.4} dnll={:+.4} nll_tokens={}",
            self.rows,
            self.dec,
            self.miss,
            100.0 * self.miss as f64 / self.dec.max(1) as f64,
            self.sd / self.nd.max(1) as f64,
            self.nll_a / n,
            self.nll_b / n,
            (self.nll_b - self.nll_a) / n,
            self.na
        )
    }
}

fn read_ids(p: &Path) -> Option<Vec<String>> {
    std::fs::read_to_string(p).ok().map(|s| s.split_whitespace().map(str::to_string).collect())
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    anyhow::ensure!(a.len() >= 2, "usage: kaon3_oracle_diff <a.json> <b.json> [a.ids] | <dir_a> <dir_b>");
    let (pa, pb) = (Path::new(&a[0]), Path::new(&a[1]));
    if pa.is_dir() {
        let mut names: Vec<String> = std::fs::read_dir(pa)?
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .filter_map(|f| f.strip_suffix(".json").map(str::to_string))
            .filter(|n| pb.join(format!("{n}.json")).exists())
            .filter(|n| a.get(2).is_none_or(|f| Path::new(f).join(format!("{n}.json")).exists()))
            .collect();
        names.sort();
        let mut tot = Acc::default();
        for n in &names {
            let ids = read_ids(&pa.join(format!("{n}.ids")));
            let (ra, rb) = (rows(&pa.join(format!("{n}.json")), ids.as_deref())?, rows(&pb.join(format!("{n}.json")), ids.as_deref())?);
            anyhow::ensure!(ra.len() == rb.len(), "{n}: 行数不同 {} vs {}", ra.len(), rb.len());
            let mut one = Acc::default();
            one.add(&ra, &rb);
            // 长序列行间强相关（前面一偏后面跟着偏），等效独立样本很少：单独打印，不进汇总
            if ra.len() > 512 {
                println!("{}", one.line(&format!("LONG {n:<10}")));
                continue;
            }
            println!("{}", one.line(&format!("CASE {n:<10}")));
            tot.add(&ra, &rb);
        }
        println!("{}", tot.line(&format!("POOLED cases={}", names.len())));
    } else {
        let ids = a.get(2).and_then(|p| read_ids(Path::new(p)));
        let (ra, rb) = (rows(pa, ids.as_deref())?, rows(pb, ids.as_deref())?);
        anyhow::ensure!(ra.len() == rb.len(), "行数不同 {} vs {}", ra.len(), rb.len());
        let mut one = Acc::default();
        one.add(&ra, &rb);
        println!("{}", one.line("ORACLE-DIFF"));
    }
    Ok(())
}
