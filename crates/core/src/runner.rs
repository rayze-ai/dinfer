//! 模型无关的 runner：把 `SchedulerOutput` 组成扁平 batch（`BatchPlan`），交给 `ModelExecutor` 执行，
//! 回填 `ModelRunnerOutput`。对应 vLLM `gpu_model_runner.py` 的 `_update_states` + `_prepare_inputs` 的 CPU 部分。
//!
//! batch 形态（`BatchMode`）由本模块判定，executor 据此选 attention 路径：
//! - `Prefill`：全部请求从 0 开始 prefill（无历史 KV）→ 展开 K/V 的 causal attention；
//! - `Decode`：全部请求单 token decode → 吸收路径读 paged cache；
//! - `Paged`：其余（chunked prefill 带历史 / prefill+decode 混合）→ 吸收路径、任意 query 长、对 cache 做 causal。
//!
//! P4 约束（后续阶段放开）：只支持单 KV group（MLA）；采样由 executor 决定（P4 为 greedy），本模块不做 logits 后处理；
//! 无投机解码（每请求每 step 恰好采样 1 个 token；prefill 中间 chunk 的采样结果由 scheduler 忽略）。

use crate::kv::BlockId;
use crate::request::RequestId;
use crate::sample::SamplingParams;
use crate::sched::{ModelRunnerOutput, SchedulerOutput};
use crate::util::{id_hash_map, IdHashMap};
use anyhow::{ensure, Context, Result};

/// worker 侧每请求的持久状态（跨 step），对应 vLLM `CachedRequestState`。
#[derive(Debug, Clone)]
pub struct ReqState {
    pub req_id: RequestId,
    /// prompt + 已采样 token。
    pub token_ids: Vec<u32>,
    pub num_prompt: usize,
    pub sampling: SamplingParams,
    /// 已在 KV 中算好的 token 数（本 step 开始时）。
    pub num_computed: usize,
    /// 全量 block 表（单 group）。
    pub block_ids: Vec<BlockId>,
}

/// 本 step 要执行的扁平 batch，全部 host 侧；executor 负责上传。
#[derive(Debug, Default, Clone)]
pub struct BatchPlan {
    /// 按 batch 顺序的请求 id。
    pub req_ids: Vec<RequestId>,
    /// 每请求本 step 的 token 数（与 req_ids 平行）。
    pub num_tokens: Vec<u32>,
    pub input_ids: Vec<i32>,
    pub positions: Vec<i32>,
    /// 每 token 写入 KV 的槽位 = block_id * block_size + offset。
    pub slot_mapping: Vec<i64>,
    /// prefill：`[0, n_0, n_0+n_1, …]`，长度 = num_reqs + 1。
    pub cu_seqlens: Vec<i32>,
    /// 每请求本 step 结束后的序列长度（decode 的 seq_lens）。
    pub seq_lens: Vec<i32>,
    /// `[num_reqs, max_blocks]`，-1 填充。
    pub block_table: Vec<i32>,
    pub max_blocks: usize,
    /// 要取 logits 的 token 行（每请求最后一个 token，按 batch 顺序）。
    pub logits_rows: Vec<i32>,
    pub mode: BatchMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BatchMode {
    /// 全部请求从位置 0 开始 prefill。
    #[default]
    Prefill,
    /// 全部请求单 token decode。
    Decode,
    /// 混合 / 带历史的 chunked prefill。
    Paged,
}

impl BatchPlan {
    pub fn num_reqs(&self) -> usize {
        self.req_ids.len()
    }
    pub fn total_tokens(&self) -> usize {
        self.input_ids.len()
    }
}

/// GPU 执行器：由模型 crate 实现。`forward_sample` 对每个请求产出恰好一个 token（P4 无投机解码）。
pub trait ModelExecutor {
    fn forward_sample(&mut self, plan: &BatchPlan, sampled: &mut Vec<u32>) -> Result<()>;
}

pub struct Runner<E: ModelExecutor> {
    pub exec: E,
    pub block_size: usize,
    states: IdHashMap<ReqState>,
    plan: BatchPlan,
    sampled: Vec<u32>,
}

impl<E: ModelExecutor> Runner<E> {
    pub fn new(exec: E, block_size: usize) -> Self {
        Self { exec, block_size, states: id_hash_map(64), plan: BatchPlan::default(), sampled: Vec::new() }
    }

    pub fn num_states(&self) -> usize {
        self.states.len()
    }
    pub fn state(&self, id: RequestId) -> Option<&ReqState> {
        self.states.get(&id)
    }
    /// 上一 step 的 batch（测试 / 诊断用）。
    pub fn last_plan(&self) -> &BatchPlan {
        &self.plan
    }

    /// 执行一个 step：更新状态 → 组 batch → forward+sample → 回填状态。
    pub fn execute(&mut self, so: &SchedulerOutput) -> Result<ModelRunnerOutput> {
        self.update_states(so)?;
        if so.num_scheduled_tokens.is_empty() {
            return Ok(ModelRunnerOutput::default());
        }
        self.build_plan(so)?;
        self.sampled.clear();
        self.exec.forward_sample(&self.plan, &mut self.sampled)?;
        ensure!(
            self.sampled.len() == self.plan.num_reqs(),
            "executor returned {} tokens for {} requests",
            self.sampled.len(),
            self.plan.num_reqs()
        );
        // 回填：computed 推进；只有 "本步算到了序列末尾"（prefill 末块 / decode）的采样才是真 token，
        // prefill 中间 chunk 的采样无意义：不追加、也不上报（scheduler 侧同样忽略，见 vLLM `_prepare_inputs`）。
        let mut out = ModelRunnerOutput {
            req_ids: self.plan.req_ids.clone(),
            sampled: Vec::with_capacity(self.plan.num_reqs()),
        };
        for (i, &rid) in self.plan.req_ids.iter().enumerate() {
            let st = self.states.get_mut(&rid).unwrap();
            st.num_computed += self.plan.num_tokens[i] as usize;
            if st.num_computed == st.token_ids.len() {
                let tok = self.sampled[i];
                st.token_ids.push(tok);
                out.sampled.push(vec![tok]);
            } else {
                out.sampled.push(Vec::new());
            }
        }
        Ok(out)
    }

    fn update_states(&mut self, so: &SchedulerOutput) -> Result<()> {
        for rid in &so.finished_req_ids {
            self.states.remove(rid);
        }
        for n in &so.new_reqs {
            ensure!(n.block_ids.groups.len() <= 1, "runner: only one KV group supported");
            let st = ReqState {
                req_id: n.req_id,
                num_prompt: n.prompt_ids.len(),
                token_ids: n.prompt_ids.clone(),
                sampling: n.sampling.clone(),
                num_computed: n.num_computed_tokens as usize,
                block_ids: n.block_ids.group(0).to_vec(),
            };
            self.states.insert(n.req_id, st);
        }
        let c = &so.cached_reqs;
        for (i, rid) in c.req_ids.iter().enumerate() {
            let st = self.states.get_mut(rid).with_context(|| format!("cached req {rid} unknown"))?;
            if let Some(nb) = &c.new_block_ids[i] {
                if c.resumed[i] {
                    st.block_ids.clear();
                    // 抢占恢复：scheduler 的 computed 已含 prefix 命中数；序列本身不变。
                }
                st.block_ids.extend_from_slice(nb.group(0));
            }
            if c.resumed[i] {
                // 恢复后 scheduler 给的是 "本 step 结束后" 的 computed；减掉本步 token 得起点。
                let n = so.scheduled_tokens_of(*rid).unwrap_or(0) as usize;
                st.num_computed = (c.num_computed_tokens[i] as usize).saturating_sub(n);
            }
        }
        Ok(())
    }

    fn build_plan(&mut self, so: &SchedulerOutput) -> Result<()> {
        let bs = self.block_size;
        let p = &mut self.plan;
        p.req_ids.clear();
        p.num_tokens.clear();
        p.input_ids.clear();
        p.positions.clear();
        p.slot_mapping.clear();
        p.cu_seqlens.clear();
        p.seq_lens.clear();
        p.block_table.clear();
        p.logits_rows.clear();
        p.cu_seqlens.push(0);

        let mut any_prefill = false;
        let mut any_decode = false;
        let mut any_paged = false;
        let mut max_blocks = 0usize;
        for &(rid, n) in &so.num_scheduled_tokens {
            let st = self.states.get(&rid).with_context(|| format!("scheduled req {rid} unknown"))?;
            let n = n as usize;
            let start = st.num_computed;
            ensure!(start + n <= st.token_ids.len(), "req {rid}: scheduled {n} tokens past end");
            if start == 0 {
                any_prefill = true;
            } else if n == 1 {
                any_decode = true;
            } else {
                any_paged = true; // 带历史的 chunked prefill
            }
            p.req_ids.push(rid);
            p.num_tokens.push(n as u32);
            for pos in start..start + n {
                p.input_ids.push(st.token_ids[pos] as i32);
                p.positions.push(pos as i32);
                let b = st
                    .block_ids
                    .get(pos / bs)
                    .with_context(|| format!("req {rid}: no block for pos {pos}"))?;
                p.slot_mapping.push(*b as i64 * bs as i64 + (pos % bs) as i64);
            }
            let total = p.input_ids.len();
            p.cu_seqlens.push(total as i32);
            p.seq_lens.push((start + n) as i32);
            p.logits_rows.push(total as i32 - 1);
            max_blocks = max_blocks.max(st.block_ids.len());
        }
        p.mode = match (any_prefill, any_decode, any_paged) {
            (true, false, false) => BatchMode::Prefill,
            (false, true, false) => BatchMode::Decode,
            _ => BatchMode::Paged,
        };
        p.max_blocks = max_blocks;
        p.block_table.resize(p.req_ids.len() * max_blocks, -1);
        for (i, rid) in p.req_ids.iter().enumerate() {
            let st = &self.states[rid];
            for (j, &b) in st.block_ids.iter().enumerate() {
                p.block_table[i * max_blocks + j] = b as i32;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CacheConfig, SchedulerConfig};
    use crate::kv::{GroupInfo, KvCacheSpec};
    use crate::request::Request;
    use crate::sched::{Scheduler, SchedulerParams};

    /// 假执行器：记录每步 plan，返回 "位置 + 1000" 作为 token。
    struct Echo {
        plans: Vec<BatchPlan>,
    }
    impl ModelExecutor for Echo {
        fn forward_sample(&mut self, plan: &BatchPlan, sampled: &mut Vec<u32>) -> Result<()> {
            self.plans.push(plan.clone());
            for &s in &plan.seq_lens {
                sampled.push(1000 + s as u32);
            }
            Ok(())
        }
    }

    fn sched(budget: u32) -> Scheduler {
        let p = SchedulerParams {
            sched: SchedulerConfig {
                max_num_seqs: 8,
                max_num_batched_tokens: budget,
                enable_chunked_prefill: false,
                long_prefill_token_threshold: 0,
            },
            cache: CacheConfig { enable_prefix_caching: false, ..Default::default() },
            max_model_len: 256,
            num_spec_tokens: 0,
            async_scheduling: false,
            eos_token_ids: vec![],
        };
        let g = vec![GroupInfo {
            spec: KvCacheSpec::Mla { block_size: 16, head_size: 64, dtype_bytes: 2 },
            block_size: 16,
        }];
        Scheduler::new(p, g, 64)
    }

    #[test]
    fn two_requests_prefill_then_decode_then_finish() {
        let mut s = sched(1000);
        let sp = |n| SamplingParams { max_tokens: n, temperature: 0.0, ..Default::default() };
        s.add_request(Request::new(1, (0..20).collect(), sp(3), 0));
        s.add_request(Request::new(2, (0..5).collect(), sp(2), 0));
        let mut r = Runner::new(Echo { plans: vec![] }, 16);

        // step 1：两个 prefill 组成一个 batch
        let so = s.schedule();
        let mro = r.execute(&so).unwrap();
        let res = s.update_from_output(&so, &mro);
        let p = &r.exec.plans[0];
        assert_eq!(p.mode, BatchMode::Prefill);
        assert_eq!(p.req_ids, vec![1, 2]);
        assert_eq!(p.cu_seqlens, vec![0, 20, 25]);
        assert_eq!(p.logits_rows, vec![19, 24]);
        assert_eq!(p.positions[20..25], [0, 1, 2, 3, 4]);
        // req1 占块 0,1；req2 占块 2 → slot = block*16 + off
        assert_eq!(p.slot_mapping[16], 16);
        assert_eq!(p.slot_mapping[20], 32);
        assert_eq!(p.max_blocks, 2);
        assert_eq!(&p.block_table, &[0, 1, 2, -1]);
        assert_eq!(res.tokens_of(0), &[1020]);
        assert_eq!(res.tokens_of(1), &[1005]);

        // step 2：两个 decode
        let so = s.schedule();
        let mro = r.execute(&so).unwrap();
        let res = s.update_from_output(&so, &mro);
        let p = &r.exec.plans[1];
        assert_eq!(p.mode, BatchMode::Decode);
        assert_eq!(p.input_ids, vec![1020, 1005]);
        assert_eq!(p.positions, vec![20, 5]);
        assert_eq!(p.seq_lens, vec![21, 6]);
        assert_eq!(p.slot_mapping, vec![20, 37]);
        assert_eq!(p.logits_rows, vec![0, 1]);
        // req2 到 max_tokens=2 → 完成
        assert_eq!(res.items[1].finish, Some(crate::request::FinishReason::Length));

        // step 3：只剩 req1；完成后 runner 状态被清理
        let so = s.schedule();
        assert_eq!(so.finished_req_ids, vec![2]);
        let mro = r.execute(&so).unwrap();
        assert_eq!(r.num_states(), 1);
        let res = s.update_from_output(&so, &mro);
        assert_eq!(res.items[0].finish, Some(crate::request::FinishReason::Length));
        assert_eq!(r.state(1).unwrap().token_ids.len(), 23);
        let so = s.schedule();
        let _ = r.execute(&so).unwrap();
        assert_eq!(r.num_states(), 0);
    }

    #[test]
    fn mixed_and_chunked_batches_use_paged_mode() {
        // 预算 12：req1 prompt 20 分两块 prefill（12 + 8），第二块带历史 → Paged；
        // 同时 req2（prompt 4）在第 2 步进来 → 与 req1 的 chunk 混合。
        let mut s = sched(12);
        let sp = SamplingParams { max_tokens: 4, temperature: 0.0, ..Default::default() };
        s.add_request(Request::new(1, (0..20).collect(), sp.clone(), 0));
        let mut r = Runner::new(Echo { plans: vec![] }, 16);
        let so = s.schedule();
        assert_eq!(so.num_scheduled_tokens, vec![(1, 12)]);
        let mro = r.execute(&so).unwrap();
        s.update_from_output(&so, &mro);
        assert_eq!(r.exec.plans[0].mode, BatchMode::Prefill);
        // 中间 chunk 的采样结果被 scheduler 忽略，但 runner 也不应把它追加到序列
        assert_eq!(r.state(1).unwrap().token_ids.len(), 20);

        s.add_request(Request::new(2, (0..4).collect(), sp, 0));
        let so = s.schedule();
        assert_eq!(so.num_scheduled_tokens, vec![(1, 8), (2, 4)]);
        let mro = r.execute(&so).unwrap();
        let p = &r.exec.plans[1];
        assert_eq!(p.mode, BatchMode::Paged);
        assert_eq!(p.cu_seqlens, vec![0, 8, 12]);
        assert_eq!(p.positions[..8], [12, 13, 14, 15, 16, 17, 18, 19]);
        assert_eq!(p.seq_lens, vec![20, 4]);
        assert_eq!(p.logits_rows, vec![7, 11]);
        let res = s.update_from_output(&so, &mro);
        assert_eq!(res.tokens_of(0), &[1020]);
        assert_eq!(res.tokens_of(1), &[1004]);
        assert_eq!(r.state(1).unwrap().token_ids.len(), 21);

        // 之后两者都 decode
        let so = s.schedule();
        let mro = r.execute(&so).unwrap();
        assert_eq!(r.exec.plans[2].mode, BatchMode::Decode);
        assert_eq!(r.exec.plans[2].input_ids, vec![1020, 1004]);
        s.update_from_output(&so, &mro);
    }
}
