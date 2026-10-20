//! 模型无关的 runner：把 `SchedulerOutput` 组成扁平 batch（`BatchPlan`），交给 `ModelExecutor` 执行，
//! 回填 `ModelRunnerOutput`。对应 vLLM `gpu_model_runner.py` 的 `_update_states` + `_prepare_inputs` 的 CPU 部分。
//!
//! batch 形态（`BatchMode`）由本模块判定，executor 据此选 attention 路径：
//! - `Prefill`：全部请求从 0 开始 prefill（无历史 KV）→ 展开 K/V 的 causal attention；
//! - `Decode`：全部请求单 token decode → 吸收路径读 paged cache；
//! - `Paged`：其余（chunked prefill 带历史 / prefill+decode 混合）→ 吸收路径、任意 query 长、对 cache 做 causal。
//!
//! 约束：只支持单 KV group（MLA）；采样由 executor 决定（greedy），本模块不做 logits 后处理。
//!
//! 投机解码（P6-d）：scheduler 给每请求的 draft（`scheduled_spec_tokens`）追加到该请求本步的 token 之后；executor 对
//! 每请求返回 `generated = [新采样…]`（长度 1 + 接受数）与下一步的 `draft`。本模块把 `num_computed` 回退被拒绝的
//! draft 数，并把 draft 通过 `ModelRunnerOutput::drafts` 交还 scheduler（`update_draft_tokens`）。
//! A4：n-gram 提案（`spec`）在 `finish` 里对 executor 没给 draft 的 greedy 请求用 host 侧 token_ids 提案（CPU）。
//! 带 draft 的请求在 `plan.sample` 里占 `1 + 草稿数` 行（与 `logits_rows` 平行），executor 逐行采样后在 host 侧验证。

use crate::kv::BlockId;
use crate::request::RequestId;
use crate::sample::SamplingParams;
use crate::sampler::{SampleBatch, TokenLogprobs};
use crate::sched::{ModelRunnerOutput, SchedulerOutput};
use crate::spec::NgramProposer;
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
    /// Philox 种子：用户给的 `sampling.seed`，否则入队时随机生成一次（整个请求固定）。
    pub seed: u64,
    /// 已在 KV 中算好的 token 数（本 step 开始时）。
    pub num_computed: usize,
    /// 异步调度：该请求在上一步 plan 里的行号（上一步采样结果在 prev_chosen 的这一行），无则 None。
    pub last_row: Option<u32>,
    /// 全量 block 表（单 group）。
    pub block_ids: Vec<BlockId>,
}

/// 本 step 要执行的扁平 batch，全部 host 侧；executor 负责上传。
#[derive(Debug, Default, Clone)]
pub struct BatchPlan {
    /// 按 batch 顺序的请求 id。
    pub req_ids: Vec<RequestId>,
    /// 本 step 之前已结束（finish / abort）的请求：executor 用来释放按请求持有的状态（如 GDN 递归状态 slot）。
    /// KV 块由 scheduler 释放，这里只是通知。
    pub finished_req_ids: Vec<RequestId>,
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
    /// 要取 logits 的 token 行（按 batch 顺序）：无 draft 的请求取最后一行；带 draft 的请求取本步全部 `1 + 草稿数` 行
    /// （verify 要逐行 argmax）。`logits_row_start[i]..logits_row_start[i+1]` 是请求 i 的行区间。
    pub logits_rows: Vec<i32>,
    pub logits_row_start: Vec<usize>,
    /// 每请求本步的 draft token（已追加进 input_ids 尾部）；空 = 无投机。
    pub spec_tokens: Vec<Vec<u32>>,
    /// 每请求：本步最后一行的 "下一个 token" 若已知（prefill 中间 chunk：序列里已有）则给出，否则 -1（要采样）。
    /// MTP 用它做中间 chunk 行的输入。
    pub next_known: Vec<i32>,
    pub mode: BatchMode,
    /// 每 logits 行一行的采样参数（与 `logits_rows` 平行；无 draft 时 = 每请求一行）。
    pub sample: SampleBatch,
    /// 异步调度：按 token 行，`>= 0` 表示该行的 input_id 还在飞（上一步采样），值 = 上一步 plan 里的行号，
    /// executor 在设备上用上一步的 chosen 填；`-1` = 用 `input_ids` 里的值。全 -1 时可忽略。
    pub gather_idx: Vec<i32>,
    pub any_gather: bool,
    /// 插队 prefill 步（Q3-o）：在飞的 decode 步 n 还没 finish 时同步跑的、只含新请求的步。
    /// 它的采样结果**不能**覆盖 `prev_chosen`（n+1 要从那里 gather 步 n 的结果）。
    pub skip_prev_chosen: bool,
    /// 递归状态按块缓存（design §15.7；来自 `SchedulerOutput`，只保留本 plan 内的请求）：
    /// forward 前从快照槽恢复进活槽的 (req, 快照槽)；本步内把状态保存到快照槽的 (req, 对齐位置 token 数, 快照槽)
    /// （位置 = 该请求算到多少 token 时的状态；可在本步中间，由 prefill kernel 沿 chunk 边界写出）。
    pub recurrent_resume: Vec<(RequestId, u32)>,
    pub recurrent_snapshot: Vec<(RequestId, u32, u32)>,
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
    /// 能否异步执行（结果回来前就调度下一步）：全 decode、无 draft、无惩罚 / logprobs（这些要 host 看 token）。
    pub fn async_eligible(&self) -> bool {
        self.mode == BatchMode::Decode
            && self.spec_tokens.iter().all(|d| d.is_empty())
            && !self.sample.any_penalty
            && !self.sample.any_mask // 受限解码：mask 要看 host 侧完整输出序列（guided-decoding loop）
            && self.sample.max_logprobs.is_none()
    }
    pub fn total_tokens(&self) -> usize {
        self.input_ids.len()
    }
}

/// GPU 执行器：由模型 crate 实现。
/// `forward_sample`：对每个请求 push 一个 `generated`（verify 后的新 token：长度 1 + 接受的 draft 数；无 draft 时恰 1 个）
/// 与一个 `draft`（下一步的投机 token，无 MTP 时为空）。prefill 中间 chunk 也要 push（内容由 runner 忽略）。
/// `forward_sample_pair` 的返回：(decode 批的 logprobs, prefill 批的 logprobs)，各与对应 plan 的 req_ids 平行。
pub type PairLogprobs = (Vec<Vec<TokenLogprobs>>, Vec<Vec<TokenLogprobs>>);

/// executor 一步的 host 侧分段（engine-gap 第 7 轮）：upload = 上传 plan / 采样参数（含状态 slot 处理）；
/// launch = forward + 采样入队（eager 时含所有 kernel launch、工作区分配、懒加载）；wait = 等 GPU 结果（同步步的
/// D2H 或 async ring 的 `finish`）。单位秒。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StepPhases {
    pub upload: f64,
    pub launch: f64,
    pub wait: f64,
}

pub trait ModelExecutor {
    /// 上一步（forward_sample 或 finish 刚返回的那步）的分段计时；不支持的 executor 返回 None。
    fn take_step_phases(&mut self) -> Option<StepPhases> {
        None
    }
    fn forward_sample(
        &mut self,
        plan: &BatchPlan,
        generated: &mut Vec<Vec<u32>>,
        drafts: &mut Vec<Vec<u32>>,
    ) -> Result<()>;

    /// 是否支持 decode 批与 prefill 批并行执行（跨流）。默认否 → runner 不拆 plan。
    fn supports_pair(&self) -> bool {
        false
    }
    /// decode 批 + prefill 批并行执行；返回两边的 logprobs。默认实现：顺序执行。
    fn forward_sample_pair(
        &mut self,
        decode: &BatchPlan,
        prefill: &BatchPlan,
        gen_d: &mut Vec<Vec<u32>>,
        dr_d: &mut Vec<Vec<u32>>,
        gen_p: &mut Vec<Vec<u32>>,
        dr_p: &mut Vec<Vec<u32>>,
    ) -> Result<PairLogprobs> {
        self.forward_sample(decode, gen_d, dr_d)?;
        let lp_d = self.take_logprobs(decode.num_reqs());
        self.forward_sample(prefill, gen_p, dr_p)?;
        let lp_p = self.take_logprobs(prefill.num_reqs());
        Ok((lp_d, lp_p))
    }

    /// KV host tier（design §10.2）：forward 前把 host 槽拉进 GPU 块（主流，同步排在前面）；
    /// step 入队后把 GPU 块推到 host 槽（copy stream，异步）。默认：不支持（忽略）。
    fn kv_fetch(&mut self, _orders: &[(crate::kv::BlockId, u32)]) -> Result<()> {
        Ok(())
    }
    fn kv_offload(&mut self, _orders: &[(crate::kv::BlockId, u32)]) -> Result<()> {
        Ok(())
    }
    /// PD 分离（design §10.3）：按位置把 `blocks` 的全部层 KV 拷成一段连续 host 字节（P 侧，同步，主流）。
    /// 布局 = `kv_pack` 的 `[block][layer][...]`，每块 `unit_bytes`；`kv_import` 反向。默认：不支持。
    /// 异步：入队 pack + D2H 后立即返回句柄；GPU 侧用事件保证下一步 forward 不会在 pack 之前复用这些块。
    /// 发送线程 `handle.wait()` 拿字节。
    fn kv_export(&mut self, _blocks: &crate::kv::KvBlocks) -> Result<Box<dyn crate::pd::KvExportHandle>> {
        anyhow::bail!("kv_export: executor does not support PD")
    }
    /// 异步：payload（最好是本 executor `kv_host_pool` 给的 pinned 缓冲）H2D + unpack 入队，主流事件保证 forward 前完成；
    /// 缓冲由 executor 持有到 H2D 结束后归还。
    fn kv_import(&mut self, _blocks: &crate::kv::KvBlocks, _payload: Box<dyn crate::pd::HostBuf>) -> Result<()> {
        anyhow::bail!("kv_import: executor does not support PD")
    }
    /// 共享槽区后端（design §10.3 b）P 侧：映射对端区（shm mmap+register / cuda-ipc open）。失败 → 调用方退回 payload 路径。
    fn pd_map_region(&mut self, _info: &crate::pd::RegionInfo) -> Result<()> {
        anyhow::bail!("pd_map_region: executor does not support shared regions")
    }
    /// P 侧：把 `blocks` 的全层 KV 直接写进已映射对端区的槽 `slot`（D2H 到 shm / D2D 到 IPC 区）；`wait()` 后数据已落。
    fn kv_export_slot(&mut self, _blocks: &crate::kv::KvBlocks, _slot: u32) -> Result<Box<dyn crate::pd::KvExportHandle>> {
        anyhow::bail!("kv_export_slot: executor does not support shared regions")
    }
    /// 每步调一次：回收已完成在飞导入的缓冲 / 槽（槽 drop = credit 回 P）。默认空。
    fn pd_poll(&mut self) {}
    /// 接收线程落 payload 用的 pinned 池；None = 用普通内存（导入时多一次 memcpy）。
    fn kv_host_pool(&self) -> Option<std::sync::Arc<dyn crate::pd::HostBufPool>> {
        None
    }
    /// 一个块（全层）导出后占多少字节；0 = 不支持 PD。
    fn kv_unit_bytes(&self) -> usize {
        0
    }

    /// 异步两段式：`begin` 上传 + 入队 + 异步 D2H（不等 GPU），`finish` 等结果。只在 `supports_async()` 且
    /// plan 满足 `BatchPlan::async_eligible()` 时被 runner 调用；默认实现 = 同步 forward_sample 后缓存结果。
    fn supports_async(&self) -> bool {
        false
    }
    fn begin(&mut self, plan: &BatchPlan) -> Result<()> {
        let (mut g, mut d) = (Vec::new(), Vec::new());
        self.forward_sample(plan, &mut g, &mut d)?;
        self.stash_sync_result(g, d);
        Ok(())
    }
    fn finish(&mut self, generated: &mut Vec<Vec<u32>>, drafts: &mut Vec<Vec<u32>>) -> Result<()> {
        let (g, d) = self.take_sync_result();
        *generated = g;
        *drafts = d;
        Ok(())
    }
    /// 默认 begin/finish 的暂存（不支持异步的 executor 可不实现：默认 begin 直接走 forward_sample 并暂存）。
    fn stash_sync_result(&mut self, _g: Vec<Vec<u32>>, _d: Vec<Vec<u32>>) {}
    fn take_sync_result(&mut self) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
        (Vec::new(), Vec::new())
    }

    /// 上一次 `forward_sample` 的 logprobs（与 req_ids 平行；不要 logprobs 的请求为空 Vec）。默认：不支持 → 全空。
    fn take_logprobs(&mut self, num_reqs: usize) -> Vec<Vec<TokenLogprobs>> {
        vec![Vec::new(); num_reqs]
    }
}

/// `Runner::begin` 的返回：finish 需要的每行信息。
#[derive(Default)]
pub struct Pending {
    rows: Vec<PendingRow>,
    req_ids: Vec<RequestId>,
    is_async: bool,
    lps_override: Option<Vec<Vec<TokenLogprobs>>>,
}
struct PendingRow {
    rid: RequestId,
    n: usize,
    num_draft: usize,
    end_of_seq: bool,
}

pub struct Runner<E: ModelExecutor> {
    pub exec: E,
    last_async: bool,
    pub block_size: usize,
    states: IdHashMap<ReqState>,
    plan: BatchPlan,
    generated: Vec<Vec<u32>>,
    drafts: Vec<Vec<u32>>,
    /// n-gram 提案器（None = 不做 n-gram 投机；executor 自带的 draft（MTP）不受影响）。
    spec: Option<NgramProposer>,
    /// 诊断：step 构成计数（每 1000 步打一行日志）。mixed = 同步里既有 decode 行又有 prefill 行。
    steps_total: u64,
    steps_mixed: u64,
    steps_pair: u64,
    steps_prefill_only: u64,
}

impl<E: ModelExecutor> Runner<E> {
    pub fn new(exec: E, block_size: usize) -> Self {
        Self {
            exec,
            last_async: false,
            block_size,
            states: id_hash_map(64),
            plan: BatchPlan::default(),
            generated: Vec::new(),
            drafts: Vec::new(),
            spec: None,
            steps_total: 0,
            steps_mixed: 0,
            steps_pair: 0,
            steps_prefill_only: 0,
        }
    }

    /// 启用 n-gram 提案（`--speculative ngram`）。
    pub fn with_ngram(mut self, p: NgramProposer) -> Self {
        self.spec = Some(p);
        self
    }
    pub fn ngram(&self) -> Option<&NgramProposer> {
        self.spec.as_ref()
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
    /// 同步执行一步（= begin + finish）。
    pub fn execute(&mut self, so: &SchedulerOutput) -> Result<ModelRunnerOutput> {
        let pending = self.begin(so)?;
        self.finish(pending)
    }

    /// 本步是否会走异步路径（engine 据此决定要不要在 finish 前先调度下一步）。只有 begin 之后才准确。
    pub fn last_begin_was_async(&self) -> bool {
        self.last_async
    }

    /// 第一段：更新状态、建 plan、入队。异步合格时不等 GPU；返回的 `Pending` 交给 `finish`。
    /// 状态里的 `num_computed` 在这里就推进（下一步 build_plan 需要），token_ids 在 finish 时补。
    pub fn begin(&mut self, so: &SchedulerOutput) -> Result<Pending> {
        self.update_states(so)?;
        if so.num_scheduled_tokens.is_empty() {
            self.last_async = false;
            return Ok(Pending::default());
        }
        self.build_plan(so)?;
        self.generated.clear();
        self.drafts.clear();
        if !so.kv_fetch.is_empty() {
            self.exec.kv_fetch(&so.kv_fetch)?;
        }
        let use_async = self.exec.supports_async() && self.plan.async_eligible();
        self.last_async = use_async;
        let mut lps_override = None;
        if use_async {
            self.exec.begin(&self.plan)?;
        } else {
            // 跨流并行（Q3-j）：混批且 executor 支持 → decode 批 / prefill 批分别执行，结果按原顺序散回
            lps_override = self.try_execute_pair(so)?;
            if lps_override.is_none() {
                self.exec.forward_sample(&self.plan, &mut self.generated, &mut self.drafts)?;
            }
        }
        // 记录每行的 (rid, n, num_draft, end_of_seq, row)，并推进 num_computed（token 值稍后补）
        let mut rows = Vec::with_capacity(self.plan.num_reqs());
        for (i, &rid) in self.plan.req_ids.iter().enumerate() {
            let st = self.states.get_mut(&rid).unwrap();
            let n = self.plan.num_tokens[i] as usize;
            let num_draft = self.plan.spec_tokens[i].len();
            let base = n - num_draft;
            let end_of_seq = st.num_computed + base >= st.token_ids.len();
            // 异步合格的 plan 全是 decode（n=1，无 draft），这里可以提前推进；非异步时 finish 会用同一 rows 回填
            if use_async {
                st.num_computed += base; // accepted=0（无 draft）
                st.last_row = Some(i as u32);
            }
            rows.push(PendingRow { rid, n, num_draft, end_of_seq });
        }
        if !so.kv_offload.is_empty() {
            self.exec.kv_offload(&so.kv_offload)?;
        }
        Ok(Pending { rows, req_ids: self.plan.req_ids.clone(), is_async: use_async, lps_override })
    }

    /// 插队 prefill（Q3-o）：在飞 decode 步未 finish 时，同步执行一个只含新请求的步。
    /// 与 `begin` 的区别：强制同步、plan 标记 `skip_prev_chosen`、要求 so 里没有 decode 行（running 行都被
    /// placeholder 挡住，scheduler 保证）。返回 Pending 直接交给 `finish`。
    pub fn begin_cut_in(&mut self, so: &SchedulerOutput) -> Result<Pending> {
        self.update_states(so)?;
        if so.num_scheduled_tokens.is_empty() {
            self.last_async = false;
            return Ok(Pending::default());
        }
        self.build_plan(so)?;
        ensure!(
            !self.plan.any_gather,
            "cut-in step contains an in-flight decode row (scheduler should have held it back)"
        );
        self.plan.skip_prev_chosen = true;
        self.last_async = false;
        self.generated.clear();
        self.drafts.clear();
        if !so.kv_fetch.is_empty() {
            self.exec.kv_fetch(&so.kv_fetch)?;
        }
        self.exec.forward_sample(&self.plan, &mut self.generated, &mut self.drafts)?;
        if !so.kv_offload.is_empty() {
            self.exec.kv_offload(&so.kv_offload)?;
        }
        let mut rows = Vec::with_capacity(self.plan.num_reqs());
        for (i, &rid) in self.plan.req_ids.iter().enumerate() {
            let st = self.states.get(&rid).unwrap();
            let n = self.plan.num_tokens[i] as usize;
            let num_draft = self.plan.spec_tokens[i].len();
            let base = n - num_draft;
            rows.push(PendingRow { rid, n, num_draft, end_of_seq: st.num_computed + base >= st.token_ids.len() });
        }
        Ok(Pending { rows, req_ids: self.plan.req_ids.clone(), is_async: false, lps_override: None })
    }

    /// 第二段：取结果、补 token、产出 ModelRunnerOutput。
    pub fn finish(&mut self, pending: Pending) -> Result<ModelRunnerOutput> {
        if pending.req_ids.is_empty() {
            return Ok(ModelRunnerOutput::default());
        }
        if pending.is_async {
            self.generated.clear();
            self.drafts.clear();
            self.exec.finish(&mut self.generated, &mut self.drafts)?;
        }
        ensure!(
            self.generated.len() == pending.req_ids.len() && self.drafts.len() == pending.req_ids.len(),
            "executor returned {}/{} entries for {} requests",
            self.generated.len(),
            self.drafts.len(),
            pending.req_ids.len()
        );
        let mut out = ModelRunnerOutput {
            req_ids: pending.req_ids.clone(),
            sampled: Vec::with_capacity(pending.req_ids.len()),
            drafts: Vec::with_capacity(pending.req_ids.len()),
            logprobs: Vec::with_capacity(pending.req_ids.len()),
            mode: Some(self.plan.mode),
            phases: self.exec.take_step_phases(),
        };
        let mut lps = match pending.lps_override {
            Some(l) => l,
            None => self.exec.take_logprobs(pending.req_ids.len()),
        };
        ensure!(lps.len() == pending.req_ids.len(), "executor returned {} logprob entries", lps.len());
        for (i, row) in pending.rows.iter().enumerate() {
            let Some(st) = self.states.get_mut(&row.rid) else {
                // 请求在飞行期间已结束并被清理（异步下一步已算了它一行）：丢弃
                out.sampled.push(Vec::new());
                out.drafts.push(Vec::new());
                out.logprobs.push(Vec::new());
                continue;
            };
            let base = row.n - row.num_draft;
            if row.end_of_seq {
                let gen = &self.generated[i];
                ensure!(
                    !gen.is_empty() && gen.len() <= 1 + row.num_draft,
                    "req {}: executor returned {} tokens",
                    row.rid,
                    gen.len()
                );
                let accepted = gen.len() - 1;
                if !pending.is_async {
                    st.num_computed += base + accepted;
                }
                st.token_ids.extend_from_slice(gen);
                out.sampled.push(gen.clone());
                let mut draft = std::mem::take(&mut self.drafts[i]);
                // n-gram 提案：executor 没给 draft、同步步、greedy 且无惩罚（惩罚要逐 token 的历史，draft 行给不了，
                // 随机采样的拒绝采样未做 → 这些请求不提案，退化为普通 decode；TODO 随机采样按 vLLM rejection_sampler）
                if draft.is_empty() && !pending.is_async {
                    if let Some(ng) = &self.spec {
                        // 受限解码（grammar）也不提案：draft 行的 mask 要文法逐 token 前进 / 回退（vLLM 的
                        // accept/rollback），guided-decoding loop 未做 → 这些请求走普通 decode
                        if st.sampling.is_greedy() && !st.sampling.has_penalties() && st.sampling.grammar.is_none() {
                            // 还能产出的 token 数（含本步之后的 bonus）：多于它的 draft 一定被 max_tokens 截掉
                            let budget = (st.num_prompt + st.sampling.max_tokens as usize)
                                .saturating_sub(st.token_ids.len() + 1);
                            draft = ng.propose(&st.token_ids, budget);
                        }
                    }
                }
                out.drafts.push(draft);
                out.logprobs.push(std::mem::take(&mut lps[i]));
            } else {
                ensure!(row.num_draft == 0, "req {}: draft on a mid-prefill chunk", row.rid);
                if !pending.is_async {
                    st.num_computed += row.n;
                }
                out.sampled.push(Vec::new());
                out.drafts.push(Vec::new());
                out.logprobs.push(Vec::new());
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
                token_ids: n.prompt_ids.iter().copied().chain(n.known_output.iter().copied()).collect(),
                seed: n.sampling.seed.unwrap_or_else(|| crate::sampler::random_seed(n.req_id)),
                sampling: n.sampling.clone(),
                num_computed: n.num_computed_tokens as usize,
                last_row: None,
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

    /// 一行是否是 "纯 decode"：本步只算 1 个 token，且已有历史（首 token 的 prefill 不算）。
    fn is_decode_row(&self, rid: RequestId, n: u32) -> bool {
        n == 1 && self.states.get(&rid).is_some_and(|st| st.num_computed > 0)
    }

    /// 只保留 `keep(i)` 为真的行的 SchedulerOutput 副本（其余字段原样）。
    fn filter_output(so: &SchedulerOutput, keep: impl Fn(usize) -> bool) -> SchedulerOutput {
        let mut out = so.clone();
        out.num_scheduled_tokens =
            so.num_scheduled_tokens.iter().enumerate().filter(|(i, _)| keep(*i)).map(|(_, x)| *x).collect();
        out
    }

    /// 跨流并行：`self.plan`（已 build）是 decode 行与 prefill 行的混批，且 executor 支持 pair 执行时，
    /// 拆成两个 plan 交给 `forward_sample_pair`，把 generated / drafts / logprobs 按原顺序散回。
    /// 返回 Some(logprobs) 表示走了 pair 路径；None 表示调用方按单 plan 执行。
    fn try_execute_pair(&mut self, so: &SchedulerOutput) -> Result<Option<Vec<Vec<TokenLogprobs>>>> {
        let rows = &so.num_scheduled_tokens;
        let decode_mask: Vec<bool> = rows.iter().map(|&(rid, n)| self.is_decode_row(rid, n)).collect();
        let num_decode = decode_mask.iter().filter(|&&d| d).count();
        let mixed = num_decode > 0 && num_decode < rows.len();
        self.steps_total += 1;
        if mixed {
            self.steps_mixed += 1;
        } else if num_decode == 0 {
            self.steps_prefill_only += 1;
        }
        if self.steps_total % 1000 == 0 {
            tracing::info!(
                steps = self.steps_total,
                mixed = self.steps_mixed,
                pair = self.steps_pair,
                prefill_only = self.steps_prefill_only,
                "runner step mix (cumulative)"
            );
        }
        if !(self.exec.supports_pair() && mixed) {
            return Ok(None);
        }
        self.steps_pair += 1;

        // 用同一个 build_plan 分别构造两个子 plan；主 plan 先挪开再放回
        let full = std::mem::take(&mut self.plan);
        self.build_plan(&Self::filter_output(so, |i| decode_mask[i]))?;
        let plan_d = std::mem::take(&mut self.plan);
        self.build_plan(&Self::filter_output(so, |i| !decode_mask[i]))?;
        let plan_p = std::mem::take(&mut self.plan);
        self.plan = full;

        let (mut gen_d, mut dr_d, mut gen_p, mut dr_p) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut lp_d, mut lp_p) =
            self.exec.forward_sample_pair(&plan_d, &plan_p, &mut gen_d, &mut dr_d, &mut gen_p, &mut dr_p)?;
        ensure!(
            gen_d.len() == plan_d.num_reqs() && gen_p.len() == plan_p.num_reqs(),
            "pair executor returned {}/{} entries for {}/{} requests",
            gen_d.len(),
            gen_p.len(),
            plan_d.num_reqs(),
            plan_p.num_reqs()
        );
        lp_d.resize(plan_d.num_reqs(), Vec::new());
        lp_p.resize(plan_p.num_reqs(), Vec::new());

        // 按原顺序散回：decode 行从 *_d 取，prefill 行从 *_p 取
        let mut d = (gen_d.into_iter(), dr_d.into_iter(), lp_d.into_iter());
        let mut p = (gen_p.into_iter(), dr_p.into_iter(), lp_p.into_iter());
        let mut lps = Vec::with_capacity(rows.len());
        for &is_decode in &decode_mask {
            let src = if is_decode { &mut d } else { &mut p };
            self.generated.push(src.0.next().unwrap());
            self.drafts.push(src.1.next().unwrap());
            lps.push(src.2.next().unwrap());
        }
        Ok(Some(lps))
    }

    fn build_plan(&mut self, so: &SchedulerOutput) -> Result<()> {
        let bs = self.block_size;
        let p = &mut self.plan;
        p.req_ids.clear();
        p.finished_req_ids.clear();
        p.finished_req_ids.extend_from_slice(&so.finished_req_ids);
        p.num_tokens.clear();
        p.input_ids.clear();
        p.positions.clear();
        p.slot_mapping.clear();
        p.cu_seqlens.clear();
        p.seq_lens.clear();
        p.block_table.clear();
        p.logits_rows.clear();
        p.logits_row_start.clear();
        p.spec_tokens.clear();
        p.next_known.clear();
        p.sample.clear();
        p.gather_idx.clear();
        p.any_gather = false;
        p.skip_prev_chosen = false;
        p.recurrent_resume.clear();
        p.recurrent_snapshot.clear();
        p.cu_seqlens.push(0);
        p.logits_row_start.push(0);
        for &(rid, s) in &so.recurrent_resume {
            if so.num_scheduled_tokens.iter().any(|(r, _)| *r == rid) {
                p.recurrent_resume.push((rid, s));
            }
        }
        for &(rid, pos, s) in &so.recurrent_snapshot {
            if so.num_scheduled_tokens.iter().any(|(r, _)| *r == rid) {
                p.recurrent_snapshot.push((rid, pos, s));
            }
        }

        let mut any_prefill = false;
        let mut any_decode = false;
        let mut any_paged = false;
        let mut max_blocks = 0usize;
        for &(rid, n) in &so.num_scheduled_tokens {
            let st = self.states.get(&rid).with_context(|| format!("scheduled req {rid} unknown"))?;
            let n = n as usize;
            let start = st.num_computed;
            let spec: &[u32] = so
                .scheduled_spec_tokens
                .iter()
                .find(|(id, _)| *id == rid)
                .map(|(_, s)| s.as_slice())
                .unwrap_or(&[]);
            let base = n
                .checked_sub(spec.len())
                .with_context(|| format!("req {rid}: {n} tokens < {} drafts", spec.len()))?;
            // 异步调度：最后 1 个 token 可能还在飞（上一步采样未回）：允许 start + base == token_ids.len() + 1，
            // 该行 input_id 由设备从上一步 chosen gather（st.last_row）。
            let in_flight = start + base == st.token_ids.len() + 1;
            ensure!(
                start + base <= st.token_ids.len() + 1,
                "req {rid}: scheduled {base} tokens past end (have {}, start {start})",
                st.token_ids.len()
            );
            ensure!(!in_flight || (base == 1 && st.last_row.is_some()), "req {rid}: in-flight token without last_row");
            if start == 0 {
                any_prefill = true;
            } else if n == 1 {
                any_decode = true;
            } else {
                any_paged = true; // 带历史的 chunked prefill / 带 draft 的 decode
            }
            p.req_ids.push(rid);
            p.num_tokens.push(n as u32);
            let row0 = p.input_ids.len();
            for (k, pos) in (start..start + n).enumerate() {
                let (tok, gather) = if k < base {
                    match st.token_ids.get(pos) {
                        Some(&t) => (t, -1),
                        None => (0, st.last_row.unwrap() as i32), // 在飞：设备上 gather
                    }
                } else {
                    (spec[k - base], -1)
                };
                p.input_ids.push(tok as i32);
                p.gather_idx.push(gather);
                p.any_gather |= gather >= 0;
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
            if spec.is_empty() {
                p.logits_rows.push(total as i32 - 1);
            } else {
                p.logits_rows.extend((row0 + base - 1..total).map(|r| r as i32));
            }
            p.logits_row_start.push(p.logits_rows.len());
            p.spec_tokens.push(spec.to_vec());
            // 中间 chunk：最后一行的下一个 token 已在序列里
            let next_known =
                if start + base < st.token_ids.len() { st.token_ids[start + base] as i32 } else { -1 };
            p.next_known.push(next_known);
            // 采样行：历史 = 到本步末尾为止的 prompt + 已产出（中间 chunk 的行不会被采样，参数无意义但保持平行）
            // 带 draft 的请求每个 logits 行一行采样参数（历史同为 base 末尾；draft 行只用于 greedy 验证）
            for _ in 0..(1 + spec.len()) {
                p.sample.push(
                    &st.sampling,
                    st.seed,
                    &st.token_ids[..(start + base).min(st.token_ids.len())],
                    st.num_prompt,
                );
            }
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
        fn forward_sample(
            &mut self,
            plan: &BatchPlan,
            generated: &mut Vec<Vec<u32>>,
            drafts: &mut Vec<Vec<u32>>,
        ) -> Result<()> {
            self.plans.push(plan.clone());
            for &s in &plan.seq_lens {
                generated.push(vec![1000 + s as u32]);
                drafts.push(Vec::new());
            }
            Ok(())
        }
    }

    /// 假执行器（投机）：无 draft 的行返回 7；带 draft 的请求接受前 `accept` 个 draft，bonus = 7。
    struct Verify {
        accept: usize,
        plans: Vec<BatchPlan>,
    }
    impl ModelExecutor for Verify {
        fn forward_sample(
            &mut self,
            plan: &BatchPlan,
            generated: &mut Vec<Vec<u32>>,
            drafts: &mut Vec<Vec<u32>>,
        ) -> Result<()> {
            self.plans.push(plan.clone());
            for spec in &plan.spec_tokens {
                let mut g: Vec<u32> = spec.iter().take(self.accept).copied().collect();
                g.push(7);
                generated.push(g);
                drafts.push(Vec::new());
            }
            Ok(())
        }
    }

    /// n-gram 提案 → 下一步 plan 带 1+K 个 token / logits 行 / 采样行 → 接受 1 拒 2 → num_computed 回退。
    #[test]
    fn ngram_drafts_flow_through_plan_and_rollback() {
        let mut s = sched_spec(1000, 3);
        let sp = SamplingParams { max_tokens: 20, temperature: 0.0, ..Default::default() };
        // prompt 尾部 [1,2] 在前面出现过，其后 [3,4,5]
        s.add_request(Request::new(1, vec![1, 2, 3, 4, 5, 9, 1, 2], sp, 0));
        let mut r = Runner::new(Verify { accept: 1, plans: vec![] }, 16).with_ngram(NgramProposer::new(2, 4, 3));

        // step 1：prefill，输出 7；finish 里提案（序列 [..,1,2,7]：尾部 [2,7] 无匹配 → 无 draft）
        let so = s.schedule();
        let mro = r.execute(&so).unwrap();
        assert_eq!(mro.sampled[0], vec![7]);
        assert!(mro.drafts[0].is_empty());
        s.update_from_output(&so, &mro);
        s.update_draft_tokens(&[]);

        // 用执行器固定输出 7 造出重复：step 2 → 序列 [1,2,3,4,5,9,1,2,7,7]（尾部 [7,7] 无更早匹配）；
        // step 3 → [..,7,7,7]，尾部 [7,7] 在 idx 8 匹配 → 后继 [7]（截断到序列末尾）→ draft [7]
        for _ in 0..2 {
            let so = s.schedule();
            let mro = r.execute(&so).unwrap();
            s.update_from_output(&so, &mro);
            if !mro.drafts[0].is_empty() {
                s.update_draft_tokens(&[(1, mro.drafts[0].clone())]);
            }
        }
        assert_eq!(r.state(1).unwrap().token_ids.len(), 11);
        assert_eq!(s.request(1).unwrap().spec_ids, vec![7], "runner must propose [7] via ngram");

        // step 4：带 1 个 draft → plan 2 token / 2 logits 行 / 2 采样行；Verify accept=1 → 全接受 + bonus
        let so = s.schedule();
        assert_eq!(so.num_scheduled_tokens, vec![(1, 2)]);
        let mro = r.execute(&so).unwrap();
        let p = r.exec.plans.last().unwrap();
        assert_eq!(p.mode, BatchMode::Paged);
        assert_eq!(p.num_tokens, vec![2]);
        assert_eq!(p.spec_tokens, vec![vec![7]]);
        assert_eq!(p.logits_rows, vec![0, 1]);
        assert_eq!(p.logits_row_start, vec![0, 2]);
        assert_eq!(p.sample.num_rows(), 2, "one sample row per logits row");
        assert_eq!(mro.sampled[0], vec![7, 7]);
        s.update_from_output(&so, &mro);
        // 本步算了 base(1) + 接受(1) = 2 个位置：10 → 12；序列 11 → 13（最后一个 token 下一步才算）
        assert_eq!(r.state(1).unwrap().num_computed, 12);
        assert_eq!(s.request(1).unwrap().num_computed_tokens, 12);
        assert_eq!(r.state(1).unwrap().token_ids.len(), 13);
        // 序列 [..,7,7,7,7,7]：尾部 4-gram [7,7,7,7] 最近一次匹配在倒数第 5 起 → 后继只剩尾部最后 1 个 → draft [7]
        // （vLLM 同口径：最长 n-gram、最近位置、尾部不够 k 个截断）
        assert_eq!(mro.drafts[0], vec![7]);
        s.update_draft_tokens(&[(1, mro.drafts[0].clone())]);

        // step 5：1 个 draft（2 token），执行器拒绝它（accept=0）→ generated [7]，num_computed = 12 + 1 + 0 = 13（不是 14）
        r.exec.accept = 0;
        let so = s.schedule();
        assert_eq!(so.num_scheduled_tokens, vec![(1, 2)]);
        let mro = r.execute(&so).unwrap();
        assert_eq!(mro.sampled[0], vec![7]);
        s.update_from_output(&so, &mro);
        assert_eq!(r.state(1).unwrap().num_computed, 13);
        assert_eq!(s.request(1).unwrap().num_computed_tokens, 13);
        assert_eq!(r.state(1).unwrap().token_ids.len(), 14);
    }

    fn sched_spec(budget: u32, num_spec: u32) -> Scheduler {
        let mut s = sched(budget);
        s.set_num_spec_tokens_for_test(num_spec);
        s
    }

    fn sched(budget: u32) -> Scheduler {
        let p = SchedulerParams {
            sched: SchedulerConfig {
                max_num_seqs: 8,
                max_num_batched_tokens: budget,
                enable_chunked_prefill: false,
                long_prefill_token_threshold: 0,
            decode_exempt_from_budget: false,
            },
            cache: CacheConfig { enable_prefix_caching: false, ..Default::default() },
            max_model_len: 256,
            num_spec_tokens: 0,
            async_scheduling: false,
            eos_token_ids: vec![],
            recurrent_snapshot_tokens: 0,
            recurrent_snapshot_slots: 0,
            recurrent_snapshot_chunk: 0,
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
