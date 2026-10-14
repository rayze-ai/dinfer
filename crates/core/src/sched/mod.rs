//! 调度器。对应 vLLM `v1/core/sched/scheduler.py`。
//!
//! P0 只定义输出结构与一个最小 FCFS 调度器，用于跑通 benchmark 骨架；
//! chunked prefill、抢占、prefix cache、spec token 在 P1 实现。

pub mod output;

pub use output::SchedulerOutput;

use crate::config::SchedulerConfig;
use crate::request::{Request, RequestId, RequestStatus};
use std::collections::VecDeque;

#[derive(Debug)]
pub struct Scheduler {
    cfg: SchedulerConfig,
    waiting: VecDeque<Request>,
    running: Vec<Request>,
}

impl Scheduler {
    pub fn new(cfg: SchedulerConfig) -> Self {
        Self { cfg, waiting: VecDeque::new(), running: Vec::new() }
    }

    pub fn add_request(&mut self, req: Request) {
        self.waiting.push_back(req);
    }

    #[inline]
    pub fn has_requests(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty()
    }

    /// 最小版本：先 running 每个 1 token，再从 waiting 取请求做整段 prefill，直到预算用尽。
    pub fn schedule(&mut self) -> SchedulerOutput {
        let mut out = SchedulerOutput::default();
        let mut budget = self.cfg.max_num_batched_tokens;

        for r in &self.running {
            if budget == 0 {
                break;
            }
            out.scheduled.push((r.id, 1));
            budget -= 1;
        }

        while let Some(front) = self.waiting.front() {
            if self.running.len() as u32 + out.new_reqs.len() as u32 >= self.cfg.max_num_seqs {
                break;
            }
            let need = front.num_prompt_tokens() - front.num_computed_tokens;
            if need > budget {
                break;
            }
            let mut r = self.waiting.pop_front().unwrap();
            r.status = RequestStatus::Running;
            out.scheduled.push((r.id, need));
            out.new_reqs.push(r.id);
            budget -= need;
            self.running.push(r);
        }

        out.total_tokens = self.cfg.max_num_batched_tokens - budget;
        out
    }

    /// 最小版本：每个被调度请求追加一个 token；达到 max_tokens 则完成。
    pub fn update_from_output(
        &mut self,
        sched: &SchedulerOutput,
        sampled: &[(RequestId, u32)],
    ) -> Vec<RequestId> {
        let mut finished = Vec::new();
        for &(id, tok) in sampled {
            if let Some(r) = self.running.iter_mut().find(|r| r.id == id) {
                if let Some(&(_, n)) = sched.scheduled.iter().find(|(rid, _)| *rid == id) {
                    r.num_computed_tokens += n;
                }
                r.output_ids.push(tok);
                if r.output_ids.len() as u32 >= r.sampling.max_tokens {
                    r.status = RequestStatus::Finished(crate::request::FinishReason::Length);
                    finished.push(id);
                }
            }
        }
        self.running.retain(|r| !r.is_finished());
        finished
    }

    pub fn num_running(&self) -> usize {
        self.running.len()
    }
    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::SamplingParams;

    fn req(id: u64, prompt: u32, max_tokens: u32) -> Request {
        let sp = SamplingParams { max_tokens, ..Default::default() };
        Request::new(id, vec![1; prompt as usize], sp, 0)
    }

    #[test]
    fn fcfs_prefill_then_decode() {
        let cfg = SchedulerConfig { max_num_seqs: 8, max_num_batched_tokens: 64, ..Default::default() };
        let mut s = Scheduler::new(cfg);
        s.add_request(req(1, 10, 2));
        s.add_request(req(2, 60, 1));
        let o = s.schedule();
        assert_eq!(o.new_reqs, vec![1], "second request exceeds budget");
        assert_eq!(o.total_tokens, 10);
        let fin = s.update_from_output(&o, &[(1, 7)]);
        assert!(fin.is_empty());
        let o2 = s.schedule();
        assert!(o2.scheduled.contains(&(1, 1)));
        assert_eq!(o2.new_reqs, vec![2]);
        assert_eq!(o2.total_tokens, 61);
        let fin = s.update_from_output(&o2, &[(1, 8), (2, 9)]);
        assert_eq!(fin.len(), 2);
        assert!(!s.has_requests());
    }
}
