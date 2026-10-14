//! 模型无关的 runner：把 `SchedulerOutput` 组成扁平 batch（`BatchPlan`），交给 `ModelExecutor` 执行，
//! 回填 `ModelRunnerOutput`。对应 vLLM `gpu_model_runner.py` 的 `_update_states` + `_prepare_inputs` 的 CPU 部分。
//!
//! P4 最小版约束（后续阶段放开）：
//! - 一个 step 内要么全是 prefill（每请求本步 token 数 > 1 或 `num_computed == 0`），要么全是单 token decode；
//!   混合 batch / 带历史 KV 的 chunked prefill 返回错误（P4-b 加 `mla_prefill_paged` 后放开）。
//! - 只支持单 KV group（MLA）。
//! - 采样由 executor 决定（P4 为 greedy）；本模块不做 logits 后处理。

use crate::kv::BlockId;
use crate::request::RequestId;
use crate::sample::SamplingParams;
use crate::sched::{ModelRunnerOutput, SchedulerOutput};
use crate::util::{id_hash_map, IdHashMap};
use anyhow::{bail, ensure, Context, Result};

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
    pub is_prefill: bool,
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
        // 回填：computed 推进；prefill 末块 / decode 都把采样 token 追加到序列。
        let mut out = ModelRunnerOutput {
            req_ids: self.plan.req_ids.clone(),
            sampled: Vec::with_capacity(self.plan.num_reqs()),
        };
        for (i, &rid) in self.plan.req_ids.iter().enumerate() {
            let st = self.states.get_mut(&rid).unwrap();
            st.num_computed += self.plan.num_tokens[i] as usize;
            let tok = self.sampled[i];
            st.token_ids.push(tok);
            out.sampled.push(vec![tok]);
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
        let mut max_blocks = 0usize;
        for &(rid, n) in &so.num_scheduled_tokens {
            let st = self.states.get(&rid).with_context(|| format!("scheduled req {rid} unknown"))?;
            let n = n as usize;
            let start = st.num_computed;
            ensure!(start + n <= st.token_ids.len(), "req {rid}: scheduled {n} tokens past end");
            let is_prefill = start == 0 || n > 1;
            if is_prefill {
                ensure!(
                    start == 0,
                    "req {rid}: chunked prefill with history (computed={start}) not supported yet"
                );
                any_prefill = true;
            } else {
                any_decode = true;
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
        if any_prefill && any_decode {
            bail!("mixed prefill/decode batch not supported yet (P4-b)");
        }
        p.is_prefill = any_prefill;
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
        assert!(p.is_prefill);
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
        assert!(!p.is_prefill);
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
    fn mixed_batch_is_rejected_for_now() {
        let mut s = sched(1000);
        let sp = SamplingParams { max_tokens: 10, temperature: 0.0, ..Default::default() };
        s.add_request(Request::new(1, (0..8).collect(), sp.clone(), 0));
        let mut r = Runner::new(Echo { plans: vec![] }, 16);
        let so = s.schedule();
        let mro = r.execute(&so).unwrap();
        s.update_from_output(&so, &mro);
        s.add_request(Request::new(2, (0..8).collect(), sp, 0));
        let so = s.schedule();
        assert_eq!(so.num_scheduled_tokens.len(), 2);
        assert!(r.execute(&so).is_err());
    }
}
