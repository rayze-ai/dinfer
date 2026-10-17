//! PD 分离（design.md §10.3）v1：prefill 实例 P 算完首 token 后，把该请求的 KV 单元（一块 × 全层）打包推给
//! decode 实例 D；D 收进 pinned 槽、登记 `IncomingKv`，scheduler 对带 `request_id` 的新请求当作 host 命中接力 decode。
//! 本文件只放与 GPU 无关的部分：线协议、TCP 传输、D 侧收件表。GPU 搬运复用 host tier 的 pack / arena 路径。
//!
//! 线协议（小端）：`magic u32 | header_len u32 | header(JSON) | payload`，payload = `units.len() × unit_bytes` 连续字节。
//! 不引入 NIXL / side channel；介质可替换：`KvTransport` trait，v2 同机 CUDA IPC，v3 RDMA。
use crate::kv::BlockHash;
use crate::request::RequestId;
use crate::sample::SamplingParams;
use anyhow::{bail, Context, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::sync::Arc;
use std::time::{Duration, Instant};

const MAGIC: u32 = 0x5044_4b56; // "PDKV"
const MAX_HEADER: u32 = 16 << 20;

/// 迁移消息头：D 据此重建请求并把 payload 里的单元登记为已计算块。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MigrationHeader {
    /// P 侧 request id；proxy 转发给 D 的请求带同一个 id 用来配对。
    pub request_id: RequestId,
    /// 权重版本 / dtype / block_size 标识，与 `BlockHash` 的盐一致；不等直接拒收。
    pub model_tag: String,
    pub prompt_ids: Vec<u32>,
    pub sampling: SamplingParams,
    /// P 采样出的首 token（已含在 P 的流式首 chunk 里；D 从它继续）。
    pub first_token: u32,
    /// 单元 = (kv group, 块哈希)，顺序 = 前缀顺序；payload 按此顺序排列。
    pub units: Vec<(u8, BlockHash)>,
    pub unit_bytes: u32,
}

impl MigrationHeader {
    pub fn payload_len(&self) -> usize {
        self.units.len() * self.unit_bytes as usize
    }
}

/// 一次完整的迁移：头 + 连续 payload（D 侧收完后交给 executor `kv_unpack`）。
#[derive(Debug)]
pub struct Migration {
    pub header: MigrationHeader,
    pub payload: Vec<u8>,
    pub received_at: Instant,
}

pub fn encode(header: &MigrationHeader, payload: &[u8], w: &mut impl Write) -> Result<()> {
    anyhow::ensure!(
        payload.len() == header.payload_len(),
        "pd: payload {} != units×unit_bytes {}",
        payload.len(),
        header.payload_len()
    );
    let h = serde_json::to_vec(header)?;
    w.write_all(&MAGIC.to_le_bytes())?;
    w.write_all(&(h.len() as u32).to_le_bytes())?;
    w.write_all(&h)?;
    w.write_all(payload)?;
    w.flush()?;
    Ok(())
}

pub fn decode(r: &mut impl Read) -> Result<(MigrationHeader, Vec<u8>)> {
    let mut u = [0u8; 4];
    r.read_exact(&mut u).context("pd: read magic")?;
    if u32::from_le_bytes(u) != MAGIC {
        bail!("pd: bad magic {:#x}", u32::from_le_bytes(u));
    }
    r.read_exact(&mut u)?;
    let hl = u32::from_le_bytes(u);
    anyhow::ensure!(hl <= MAX_HEADER, "pd: header too large {hl}");
    let mut h = vec![0u8; hl as usize];
    r.read_exact(&mut h)?;
    let header: MigrationHeader = serde_json::from_slice(&h).context("pd: header json")?;
    let mut payload = vec![0u8; header.payload_len()];
    r.read_exact(&mut payload).context("pd: read payload")?;
    Ok((header, payload))
}

/// 传输介质抽象。v1 TCP；同机 CUDA IPC / 跨机 RDMA 只换实现。
pub trait KvTransport: Send + Sync {
    fn send(&self, header: &MigrationHeader, payload: &[u8]) -> Result<()>;
}

pub struct TcpTransport {
    pub addr: String,
    pub connect_timeout: Duration,
}

impl KvTransport for TcpTransport {
    fn send(&self, header: &MigrationHeader, payload: &[u8]) -> Result<()> {
        let addr = self.addr.to_socket_addrs()?.next().context("pd: resolve")?;
        let mut s = TcpStream::connect_timeout(&addr, self.connect_timeout)
            .with_context(|| format!("pd: connect {}", self.addr))?;
        s.set_nodelay(true)?;
        encode(header, payload, &mut s)
    }
}

/// D 侧收件表：`request_id → Migration`。接收线程写入，scheduler 首次调度时 `take`；超时未被领走的条目由 `gc` 丢弃。
#[derive(Default)]
pub struct IncomingKv {
    inner: Mutex<HashMap<RequestId, Migration>>,
    stats: Mutex<IncomingStats>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct IncomingStats {
    pub received: u64,
    pub taken: u64,
    pub rejected: u64,
    pub expired: u64,
}

impl IncomingKv {
    pub fn put(&self, m: Migration) {
        self.stats.lock().received += 1;
        self.inner.lock().insert(m.header.request_id, m);
    }
    pub fn take(&self, id: RequestId) -> Option<Migration> {
        let m = self.inner.lock().remove(&id);
        if m.is_some() {
            self.stats.lock().taken += 1;
        }
        m
    }
    pub fn contains(&self, id: RequestId) -> bool {
        self.inner.lock().contains_key(&id)
    }
    pub fn reject(&self) {
        self.stats.lock().rejected += 1;
    }
    /// 丢掉超过 `max_age` 没被领走的迁移（proxy 没把请求发过来 / D 已回退本地重算）。返回丢弃数。
    pub fn gc(&self, max_age: Duration) -> usize {
        let now = Instant::now();
        let mut g = self.inner.lock();
        let before = g.len();
        g.retain(|_, m| now.duration_since(m.received_at) < max_age);
        let n = before - g.len();
        self.stats.lock().expired += n as u64;
        n
    }
    pub fn stats(&self) -> IncomingStats {
        *self.stats.lock()
    }
}

/// D 侧接收线程：每条连接一次迁移。`model_tag` 不匹配直接丢（计 rejected）。返回实际绑定端口（测试用 0）。
pub fn spawn_receiver(bind: &str, model_tag: String, table: Arc<IncomingKv>) -> Result<u16> {
    let l = TcpListener::bind(bind).with_context(|| format!("pd: bind {bind}"))?;
    let port = l.local_addr()?.port();
    std::thread::Builder::new().name("pd-recv".into()).spawn(move || {
        for conn in l.incoming() {
            let Ok(mut s) = conn else { continue };
            let _ = s.set_nodelay(true);
            match decode(&mut s) {
                Ok((header, payload)) if header.model_tag == model_tag => {
                    tracing::debug!(id = header.request_id, units = header.units.len(), "pd: migration received");
                    table.put(Migration { header, payload, received_at: Instant::now() });
                }
                Ok((header, _)) => {
                    tracing::warn!(id = header.request_id, got = %header.model_tag, want = %model_tag, "pd: model_tag mismatch, dropped");
                    table.reject();
                }
                Err(e) => {
                    tracing::warn!(error = %e, "pd: bad migration");
                    table.reject();
                }
            }
        }
    })?;
    Ok(port)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hdr(id: RequestId, tag: &str) -> MigrationHeader {
        MigrationHeader {
            request_id: id,
            model_tag: tag.into(),
            prompt_ids: vec![1, 2, 3, 4],
            sampling: SamplingParams::default(),
            first_token: 42,
            units: vec![(0, BlockHash(7)), (0, BlockHash(8))],
            unit_bytes: 4,
        }
    }

    #[test]
    fn roundtrip_encode_decode() {
        let h = hdr(9, "qwen3");
        let payload: Vec<u8> = (0..8).collect();
        let mut buf = Vec::new();
        encode(&h, &payload, &mut buf).unwrap();
        let (h2, p2) = decode(&mut &buf[..]).unwrap();
        assert_eq!(h, h2);
        assert_eq!(payload, p2);
        assert!(encode(&h, &payload[..7], &mut Vec::new()).is_err());
        assert!(decode(&mut &b"xxxx\0\0\0\0"[..]).is_err());
    }

    #[test]
    fn tcp_receiver_accepts_matching_tag_and_rejects_other() {
        let table = Arc::new(IncomingKv::default());
        let port = spawn_receiver("127.0.0.1:0", "qwen3".into(), table.clone()).unwrap();
        let t = TcpTransport { addr: format!("127.0.0.1:{port}"), connect_timeout: Duration::from_secs(2) };
        t.send(&hdr(1, "qwen3"), &[0u8; 8]).unwrap();
        t.send(&hdr(2, "glm53"), &[0u8; 8]).unwrap();
        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline && table.stats().received + table.stats().rejected < 2 {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(table.contains(1));
        assert!(!table.contains(2));
        let m = table.take(1).unwrap();
        assert_eq!(m.header.first_token, 42);
        assert!(table.take(1).is_none());
        let s = table.stats();
        assert_eq!((s.received, s.taken, s.rejected), (1, 1, 1));
    }

    #[test]
    fn gc_drops_stale() {
        let table = IncomingKv::default();
        let mut m = Migration { header: hdr(5, "q"), payload: vec![0; 8], received_at: Instant::now() };
        m.received_at -= Duration::from_secs(10);
        table.put(m);
        assert_eq!(table.gc(Duration::from_secs(2)), 1);
        assert!(!table.contains(5));
    }
}
