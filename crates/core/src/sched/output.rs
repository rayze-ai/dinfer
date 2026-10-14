//! 调度输出。对应 vLLM `v1/core/sched/output.py::SchedulerOutput` 的最小子集。

use crate::request::RequestId;

#[derive(Debug, Default, Clone)]
pub struct SchedulerOutput {
    /// 本 step 首次调度的请求（worker 需拿全量数据）。
    pub new_reqs: Vec<RequestId>,
    /// (request, 本 step 调度的 token 数)。
    pub scheduled: Vec<(RequestId, u32)>,
    pub total_tokens: u32,
}
