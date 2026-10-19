//! n-gram 提案器（纯 CPU）：用序列尾部的最后 n ∈ [min_n, max_n] 个 token 在 prompt + 已生成序列里找**最近**一次
//! 出现（不含尾部自身），取其后至多 k 个 token 作 draft；长 n 优先；都不匹配则不提案（返回空）。
//! 对齐 vLLM `_find_longest_matched_ngram_and_propose_tokens`：最长 n-gram、最靠后的位置、尾部不够 k 个截断。
//! 实现用朴素倒序扫描（每请求 O(len × max_n)，2k 序列 × 16 请求每步 ≪ 1 ms）；vLLM 用 KMP，结果一致。

#[derive(Debug, Clone)]
pub struct NgramProposer {
    pub min_n: usize,
    pub max_n: usize,
    /// 每步提案的 draft 数上限。
    pub k: usize,
}

impl NgramProposer {
    pub fn new(min_n: usize, max_n: usize, k: usize) -> Self {
        let min_n = min_n.max(1);
        Self { min_n, max_n: max_n.max(min_n), k }
    }

    /// 对 `tokens`（prompt + 已生成）提案至多 `min(self.k, k_cap)` 个 draft。
    pub fn propose(&self, tokens: &[u32], k_cap: usize) -> Vec<u32> {
        let k = self.k.min(k_cap);
        let len = tokens.len();
        if k == 0 || len < self.min_n + 1 {
            return Vec::new();
        }
        for n in (self.min_n..=self.max_n.min(len - 1)).rev() {
            let tail = &tokens[len - n..];
            // 候选起点 s：tokens[s..s+n] == tail 且 s + n < len（不与尾部重叠到自身末尾之后）
            let mut s = len - n;
            while s > 0 {
                s -= 1;
                if tokens[s..s + n] == *tail {
                    let start = s + n;
                    let end = (start + k).min(len);
                    if end > start {
                        return tokens[start..end].to_vec();
                    }
                    break; // 匹配紧贴尾部（start == len）：这个 n 没有可取的后继，试更短的 n
                }
            }
        }
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_match_returns_empty() {
        let p = NgramProposer::new(2, 4, 3);
        assert!(p.propose(&[1, 2, 3, 4, 5, 6], 8).is_empty());
        assert!(p.propose(&[1, 2], 8).is_empty(), "too short");
        assert!(p.propose(&[], 8).is_empty());
    }

    #[test]
    fn basic_match_takes_following_k() {
        let p = NgramProposer::new(2, 4, 3);
        // 尾部 [1,2] 在开头出现过，其后 [3,4,5]
        assert_eq!(p.propose(&[1, 2, 3, 4, 5, 9, 1, 2], 8), vec![3, 4, 5]);
    }

    #[test]
    fn prefers_most_recent_match() {
        let p = NgramProposer::new(2, 2, 2);
        // [1,2] 出现两次：后一次后面是 [7,8]，前一次是 [3,4] → 取最近
        assert_eq!(p.propose(&[1, 2, 3, 4, 1, 2, 7, 8, 1, 2], 8), vec![7, 8]);
    }

    #[test]
    fn longer_ngram_wins() {
        let p = NgramProposer::new(2, 3, 2);
        // 尾部 [5,1,2]：3-gram 只在 idx 0 出现（后继 [3,4]）；2-gram [1,2] 最近出现在 idx 6（后继 [9,9]）
        // 长 n 优先 → [3,4]
        assert_eq!(p.propose(&[5, 1, 2, 3, 4, 0, 1, 2, 9, 9, 5, 1, 2], 8), vec![3, 4]);
    }

    #[test]
    fn truncates_when_tail_short() {
        let p = NgramProposer::new(2, 4, 5);
        // [1,2] 后面只剩 [3] 再到尾部 [1,2]：取 [3,1,2]（到序列末尾截断，允许含尾部本身）
        assert_eq!(p.propose(&[1, 2, 3, 1, 2], 8), vec![3, 1, 2]);
        // k_cap 截断
        assert_eq!(p.propose(&[1, 2, 3, 4, 5, 6, 1, 2], 2), vec![3, 4]);
        assert!(p.propose(&[1, 2, 3, 4, 5, 6, 1, 2], 0).is_empty());
    }

    #[test]
    fn repeated_text_proposes_full_k() {
        let p = NgramProposer::new(2, 4, 3);
        let seq: Vec<u32> = [10, 11, 12, 13, 14, 15].iter().cycle().take(14).copied().collect();
        // 序列 10..15 循环 14 个：尾部 4-gram [14,15,10,11] 在 idx 4 出现 → 后继 [12,13,14]
        assert_eq!(p.propose(&seq, 8), vec![12, 13, 14]);
    }
}
