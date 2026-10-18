//! DP 路由（design §13.3）：同进程 N 个独立引擎副本（各自 scheduler / KV / TP 组），按在飞请求数选副本。
//! RequestId 高 8 位编码副本号，abort / pd_wait_for 解码后转发。dense 模型副本之间无需波次同步；
//! MoE+EP 的 dummy step 同步在 engine 层做（§13.3），不在这里。

use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::Arc;

use anyhow::Result;
use dinfer_core::engine::EngineEvent;
use dinfer_core::request::RequestId;
use dinfer_core::sample::SamplingParams;

use crate::backend::{ChatBackend, OutputStream, Prompt, RenderRequest};

const SHARD_SHIFT: u32 = 56;

pub struct DpBackend {
    shards: Vec<Arc<dyn ChatBackend>>,
    inflight: Vec<Arc<AtomicUsize>>,
}

impl DpBackend {
    pub fn new(shards: Vec<Arc<dyn ChatBackend>>) -> Self {
        assert!(!shards.is_empty() && shards.len() < 256, "dp shards must be 1..=255");
        let inflight = (0..shards.len()).map(|_| Arc::new(AtomicUsize::new(0))).collect();
        Self { shards, inflight }
    }
    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }
    fn pick(&self) -> usize {
        let mut best = 0;
        let mut best_n = usize::MAX;
        for (i, c) in self.inflight.iter().enumerate() {
            let n = c.load(Relaxed);
            if n < best_n {
                best = i;
                best_n = n;
            }
        }
        best
    }
    fn split(id: RequestId) -> (usize, RequestId) {
        ((id >> SHARD_SHIFT) as usize, id & ((1u64 << SHARD_SHIFT) - 1))
    }
    fn join(shard: usize, id: RequestId) -> RequestId {
        assert!(id >> SHARD_SHIFT == 0, "engine RequestId uses top 8 bits");
        ((shard as u64) << SHARD_SHIFT) | id
    }
}

impl ChatBackend for DpBackend {
    fn model_name(&self) -> &str {
        self.shards[0].model_name()
    }
    fn max_model_len(&self) -> u32 {
        self.shards[0].max_model_len()
    }
    fn default_sampling(&self) -> SamplingParams {
        self.shards[0].default_sampling()
    }
    fn render(&self, req: &RenderRequest<'_>) -> Result<Prompt> {
        self.shards[0].render(req)
    }
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.shards[0].encode(text)
    }
    fn decode(&self, ids: &[u32]) -> Result<String> {
        self.shards[0].decode(ids)
    }
    fn token_strs(&self, ids: &[u32]) -> Option<Vec<String>> {
        self.shards[0].token_strs(ids)
    }
    fn submit(
        &self,
        prompt_ids: Vec<u32>,
        sampling: SamplingParams,
        pd: Option<dinfer_core::engine::PdSubmit>,
    ) -> (RequestId, tokio::sync::mpsc::UnboundedReceiver<EngineEvent>) {
        let i = self.pick();
        self.inflight[i].fetch_add(1, Relaxed);
        let (id, mut rx) = self.shards[i].submit(prompt_ids, sampling, pd);
        // 转发事件；接收端结束（流关闭 / 被 drop）时在飞数 -1。
        let (tx, out) = tokio::sync::mpsc::unbounded_channel();
        let counter = self.inflight[i].clone();
        tokio::spawn(async move {
            while let Some(ev) = rx.recv().await {
                if tx.send(ev).is_err() {
                    break;
                }
            }
            counter.fetch_sub(1, Relaxed);
        });
        (Self::join(i, id), out)
    }
    fn abort(&self, id: RequestId) {
        let (i, id) = Self::split(id);
        if let Some(s) = self.shards.get(i) {
            s.abort(id);
        }
    }
    fn new_output_stream(&self) -> Box<dyn OutputStream> {
        self.shards[0].new_output_stream()
    }
    fn pd_wait_for(&self, remote_id: RequestId, timeout: std::time::Duration) -> bool {
        let (i, id) = Self::split(remote_id);
        self.shards.get(i).is_some_and(|s| s.pd_wait_for(id, timeout))
    }
    fn metrics(&self) -> Option<Arc<dinfer_core::metrics::Metrics>> {
        self.shards[0].metrics()
    }
}
