//! PD 分离（design.md §10.3）v1：prefill 实例 P 算完首 token 后，把该请求的 KV 单元（一块 × 全层）按位置推给
//! decode 实例 D；D 收进 pinned 槽、登记 `IncomingKv`，scheduler 对带 `remote_id` 的新请求当作已算好接力 decode。
//! 本文件只放与 GPU 无关的部分：传输抽象、线协议、TCP 后端、D 侧收件表。GPU 搬运复用 host tier 的 pack / arena 路径。
//!
//! ## 传输抽象（为 NIXL / RDMA 预留，接口形状对齐 NIXL agent 语义）
//! - `KvTransport::register(region)`：一次注册 pinned 区（NIXL `registerMem`；TCP 忽略）。
//! - `KvTransport::send(header, payload)`：payload 是**已注册区内的切片**（`MemDesc { ptr, len }`）或任意 host 字节；
//!   TCP 后端直接 write；NIXL 后端拿 (ptr, len) 发 WRITE 描述符。
//! - `send` 返回 `Transfer`（NIXL xfer handle），`poll()` 查完成；TCP 后端同步完成后立即 Done。
//! - 控制面（header：request_id / prompt / sampling / 首 token）**始终走 TCP 小消息**：NIXL 只搬 payload，
//!   header 作为 NIXL `notif` 或仍由本 TCP 通道携带（v2 决定）。
//!
//! 线协议（小端）：`magic u32 | header_len u32 | header(JSON) | payload`，payload = `num_units × unit_bytes` 连续字节。
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

/// 迁移消息头：D 据此重建请求并把 payload 里的单元按位置填进新分配的块。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MigrationHeader {
    /// P 侧 request id；proxy 转发给 D 的请求带同一个 id 用来配对。
    pub request_id: RequestId,
    /// 权重版本 / dtype / block_size 标识；不等直接拒收。
    pub model_tag: String,
    pub prompt_ids: Vec<u32>,
    pub sampling: SamplingParams,
    /// P 采样出的首 token（已在 P 的流式首 chunk 里返回；D 从它继续）。
    pub first_token: u32,
    /// 单元数 = 该请求的块数（按位置，含未满尾块）；payload = `num_units × unit_bytes`，布局与 executor `kv_export` 一致。
    pub num_units: u32,
    pub unit_bytes: u32,
    /// 共享槽区后端（design §10.3 b）：数据已由 P 直接写进 D 的槽 `slot`，TCP 上不带 payload。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slot: Option<u32>,
}

impl MigrationHeader {
    /// TCP 上跟在 header 后面的字节数（槽模式 = 0）。
    pub fn payload_len(&self) -> usize {
        if self.slot.is_some() { 0 } else { self.data_len() }
    }
    /// KV 数据总字节（`num_units × unit_bytes`），与 payload 走哪条路无关。
    pub fn data_len(&self) -> usize {
        self.num_units as usize * self.unit_bytes as usize
    }
}

/// 连接被对端正常关闭（持久连接读到 EOF）。
#[derive(Debug)]
pub struct Eof;
impl std::fmt::Display for Eof {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("pd: connection closed")
    }
}
impl std::error::Error for Eof {}

/// 一次完整的迁移：头 + 连续 payload（D 侧收完后交给 executor `kv_import`）。
pub struct Migration {
    pub header: MigrationHeader,
    pub payload: Box<dyn HostBuf>,
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
    let (h, b) = decode_into(r, &VecPool)?;
    let mut v = vec![0u8; b.as_slice().len()];
    v.copy_from_slice(b.as_slice());
    Ok((h, v))
}
/// 头解析后向 `pool` 要落点直接读 payload（pinned 时零拷贝进 DMA 源）；池给不出就退化 Vec。
pub fn decode_into(r: &mut impl Read, pool: &dyn HostBufPool) -> Result<(MigrationHeader, Box<dyn HostBuf>)> {
    let mut u = [0u8; 4];
    if let Err(e) = r.read_exact(&mut u) {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            return Err(Eof.into());
        }
        return Err(e).context("pd: read magic");
    }
    if u32::from_le_bytes(u) != MAGIC {
        bail!("pd: bad magic {:#x}", u32::from_le_bytes(u));
    }
    r.read_exact(&mut u)?;
    let hl = u32::from_le_bytes(u);
    anyhow::ensure!(hl <= MAX_HEADER, "pd: header too large {hl}");
    let mut h = vec![0u8; hl as usize];
    r.read_exact(&mut h)?;
    let header: MigrationHeader = serde_json::from_slice(&h).context("pd: header json")?;
    let n = header.payload_len();
    if n == 0 {
        return Ok((header, Box::new(Vec::new())));
    }
    let mut buf: Box<dyn HostBuf> = pool.take(n).unwrap_or_else(|| Box::new(vec![0u8; n]));
    r.read_exact(&mut buf.as_mut_cap()[..n]).context("pd: read payload")?;
    buf.set_len(n)?;
    Ok((header, buf))
}

/// 一块 host 字节缓冲（payload 落点）。pinned（executor 提供，可直接 DMA）或普通 `Vec<u8>`（退化路径）。
pub trait HostBuf: Send + Sync {
    fn as_slice(&self) -> &[u8];
    /// 整个容量的可写视图（接收方往里读满后 `set_len`）。
    fn as_mut_cap(&mut self) -> &mut [u8];
    fn set_len(&mut self, len: usize) -> Result<()>;
    fn capacity(&self) -> usize;
    /// true = pinned，executor 可直接异步 H2D；false = 需要先拷进 pinned 中转。
    fn is_pinned(&self) -> bool;
    fn desc(&self) -> MemDesc {
        MemDesc::of(self.as_slice())
    }
    /// 下转回具体类型（executor 认回自己池里的 pinned 缓冲，零拷贝 H2D）。
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send + Sync>;
    /// 有效数据字节数（槽模式下 `as_slice()` 可能为空：数据在设备侧槽里）。
    fn len(&self) -> usize {
        self.as_slice().len()
    }
    /// 数据在接收方共享槽区的哪个槽；None = 普通 host 字节。
    fn slot(&self) -> Option<u32> {
        None
    }
}
impl HostBuf for Vec<u8> {
    fn as_slice(&self) -> &[u8] {
        self
    }
    fn as_mut_cap(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
    fn set_len(&mut self, len: usize) -> Result<()> {
        anyhow::ensure!(len <= self.len(), "Vec payload: len {len} > cap {}", self.len());
        self.truncate(len);
        Ok(())
    }
    fn capacity(&self) -> usize {
        self.len()
    }
    fn is_pinned(&self) -> bool {
        false
    }
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send + Sync> {
        self
    }
}
/// 缓冲池：接收线程 / 导出路径找它要落点。executor 实现为 pinned 池；没有池时用 `VecPool`。
pub trait HostBufPool: Send + Sync {
    /// 拿一块容量 ≥ `min_bytes` 的缓冲；池空 / 太小返回 None（调用方退化到 Vec）。
    fn take(&self, min_bytes: usize) -> Option<Box<dyn HostBuf>>;
}
#[derive(Default)]
pub struct VecPool;
impl HostBufPool for VecPool {
    fn take(&self, min_bytes: usize) -> Option<Box<dyn HostBuf>> {
        Some(Box::new(vec![0u8; min_bytes]))
    }
}
/// 一次异步导出的句柄（executor 返回）：GPU → host 拷贝可能还在飞；`wait()` 只阻塞调用线程（发送线程），
/// 返回 payload 字节。drop 归还缓冲。
pub trait KvExportHandle: Send {
    fn wait(&mut self) -> Result<&dyn HostBuf>;
}
/// 同步已就绪的导出（mock / 退化路径）。
pub struct ReadyExport(pub Box<dyn HostBuf>);
impl KvExportHandle for ReadyExport {
    fn wait(&mut self) -> Result<&dyn HostBuf> {
        Ok(self.0.as_ref())
    }
}

/// 一段 host 内存（NIXL 的 memory descriptor 形状）。`Send`：调用方保证指向的内存在 transfer 完成前有效。
#[derive(Debug, Clone, Copy)]
pub struct MemDesc {
    pub ptr: *const u8,
    pub len: usize,
}
unsafe impl Send for MemDesc {}
unsafe impl Sync for MemDesc {}
impl MemDesc {
    pub fn of(b: &[u8]) -> Self {
        Self { ptr: b.as_ptr(), len: b.len() }
    }
    /// SAFETY：调用方保证 (ptr, len) 有效且在返回切片生命周期内不被写。
    pub unsafe fn as_slice(&self) -> &[u8] {
        std::slice::from_raw_parts(self.ptr, self.len)
    }
}

/// 一次在飞传输（NIXL xfer handle 形状）。TCP 后端同步发完即 Done。
pub trait Transfer: Send {
    fn poll(&mut self) -> Result<TransferState>;
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferState {
    InFlight,
    Done,
}
pub struct DoneTransfer;
impl Transfer for DoneTransfer {
    fn poll(&mut self) -> Result<TransferState> {
        Ok(TransferState::Done)
    }
}

/// 传输后端。v1 TCP；NIXL / CUDA IPC / RDMA 只换实现。
pub trait KvTransport: Send + Sync {
    fn name(&self) -> &'static str;
    /// 注册一块 pinned 区（NIXL `registerMem`）。TCP 后端 no-op。幂等。
    fn register(&self, _region: MemDesc) -> Result<()> {
        Ok(())
    }
    /// 发一次迁移：header 控制面 + payload 数据面。payload 可以是已注册区的切片。
    fn send(&self, header: &MigrationHeader, payload: MemDesc) -> Result<Box<dyn Transfer>>;
    /// 共享槽区后端：对端已握手到的区描述与连接代数（代数变了 = 对端重启，调用方重新映射）。TCP 后端 None。
    fn region(&self) -> Option<(RegionInfo, u64)> {
        None
    }
    /// 拿一个空闲槽（credit）；None = 没有空闲 / 未连接 → 调用方走 payload 路径。
    fn acquire_slot(&self) -> Option<u32> {
        None
    }
    /// 本地放弃一个刚拿到但没用的槽。
    fn release_slot(&self, _slot: u32) {}
}

// ---------------------------------------------------------------- 共享槽区后端（design §10.3 b：shm / cuda-ipc）
/// D 侧区描述，握手时发给 P。`kind` = "shm" | "cuda-ipc"。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegionInfo {
    pub kind: String,
    pub slots: u32,
    pub slot_bytes: u64,
    pub shm_name: Option<String>,
    pub ipc_handle: Option<Vec<u8>>,
    pub device_uuid: Vec<u8>,
}
const MAGIC_HELLO: u32 = 0x5044_4849; // "PDHI"

/// D 侧本地区（core 只认 host 基址；设备区基址为 null，unpack 时由 executor 按槽号取地址）。
pub struct LocalRegion {
    pub info: RegionInfo,
    pub host_base: *const u8,
}
unsafe impl Send for LocalRegion {}
unsafe impl Sync for LocalRegion {}

/// 落在共享槽里的 payload：drop 时把槽号作为 credit 回给 P（经该连接的写线程）。
pub struct SlotPayload {
    slot: u32,
    len: usize,
    host: *const u8,
    credit: crossbeam_channel::Sender<u32>,
}
unsafe impl Send for SlotPayload {}
unsafe impl Sync for SlotPayload {}
impl HostBuf for SlotPayload {
    fn as_slice(&self) -> &[u8] {
        // SAFETY: host 非空时指向 D 本地 shm 槽，长度 len 在槽内（接收时已校验）。
        if self.host.is_null() { &[] } else { unsafe { std::slice::from_raw_parts(self.host, self.len) } }
    }
    fn as_mut_cap(&mut self) -> &mut [u8] {
        &mut []
    }
    fn set_len(&mut self, len: usize) -> Result<()> {
        anyhow::ensure!(len == self.len, "SlotPayload: fixed len");
        Ok(())
    }
    fn capacity(&self) -> usize {
        self.len
    }
    fn is_pinned(&self) -> bool {
        true
    }
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send + Sync> {
        self
    }
    fn len(&self) -> usize {
        self.len
    }
    fn slot(&self) -> Option<u32> {
        Some(self.slot)
    }
}
impl Drop for SlotPayload {
    fn drop(&mut self) {
        let _ = self.credit.send(self.slot);
    }
}

struct SharedConn {
    conn: Option<TcpStream>,
    region: Option<RegionInfo>,
    credits: Vec<u32>,
    gen: u64,
    last_attempt: Option<Instant>,
}
/// P 侧：一条到 D 的持久连接。连上先收 D 的 `RegionInfo`（hello），之后：槽模式 header（无 payload）走这条连接，
/// D 用完槽后在同一连接上回 4 字节槽号作为 credit；无槽 / 未连上时退回一次性 TCP（`TcpTransport` 路径）。
pub struct SharedTransport {
    addr: String,
    connect_timeout: Duration,
    inner: Arc<Mutex<SharedConn>>,
}
impl SharedTransport {
    pub fn new(addr: String, connect_timeout: Duration) -> Arc<Self> {
        let t = Arc::new(Self {
            addr,
            connect_timeout,
            inner: Arc::new(Mutex::new(SharedConn { conn: None, region: None, credits: Vec::new(), gen: 0, last_attempt: None })),
        });
        t.try_connect();
        t
    }
    fn try_connect(&self) {
        let mut g = self.inner.lock();
        if g.conn.is_some() || g.last_attempt.is_some_and(|t| t.elapsed() < Duration::from_secs(1)) {
            return;
        }
        g.last_attempt = Some(Instant::now());
        let r = (|| -> Result<(TcpStream, RegionInfo)> {
            let addr = self.addr.to_socket_addrs()?.next().context("pd: resolve")?;
            let mut s = TcpStream::connect_timeout(&addr, self.connect_timeout)?;
            s.set_nodelay(true)?;
            s.set_read_timeout(Some(Duration::from_secs(2)))?;
            let mut u = [0u8; 4];
            s.read_exact(&mut u)?;
            anyhow::ensure!(u32::from_le_bytes(u) == MAGIC_HELLO, "pd: peer did not send region hello (peer runs tcp backend?)");
            s.read_exact(&mut u)?;
            let n = u32::from_le_bytes(u);
            anyhow::ensure!(n <= 1 << 20, "pd: hello too large");
            let mut b = vec![0u8; n as usize];
            s.read_exact(&mut b)?;
            let info: RegionInfo = serde_json::from_slice(&b)?;
            s.set_read_timeout(None)?;
            Ok((s, info))
        })();
        match r {
            Ok((s, info)) => {
                g.gen += 1;
                let gen = g.gen;
                g.credits = (0..info.slots).rev().collect();
                tracing::info!(peer = %self.addr, kind = %info.kind, slots = info.slots, slot_mib = info.slot_bytes >> 20, "pd: shared region handshake");
                let mut rd = s.try_clone().expect("tcp try_clone");
                g.conn = Some(s);
                g.region = Some(info);
                let inner = self.inner.clone();
                std::thread::Builder::new()
                    .name("pd-credit".into())
                    .spawn(move || {
                        let mut u = [0u8; 4];
                        while rd.read_exact(&mut u).is_ok() {
                            let mut g = inner.lock();
                            if g.gen == gen {
                                g.credits.push(u32::from_le_bytes(u));
                            }
                        }
                        let mut g = inner.lock();
                        if g.gen == gen {
                            tracing::warn!("pd: shared connection closed by peer");
                            g.conn = None;
                            g.region = None;
                            g.credits.clear();
                        }
                    })
                    .expect("spawn pd-credit");
            }
            Err(e) => tracing::warn!(peer = %self.addr, error = %e, "pd: shared handshake failed, will retry / fall back to tcp"),
        }
    }
}
impl KvTransport for SharedTransport {
    fn name(&self) -> &'static str {
        "shared"
    }
    fn send(&self, header: &MigrationHeader, payload: MemDesc) -> Result<Box<dyn Transfer>> {
        if header.slot.is_some() {
            let mut g = self.inner.lock();
            let s = g.conn.as_mut().context("pd: shared connection down")?;
            if let Err(e) = encode(header, &[], s) {
                g.conn = None;
                g.region = None;
                g.credits.clear();
                return Err(e);
            }
            return Ok(Box::new(DoneTransfer));
        }
        TcpTransport { addr: self.addr.clone(), connect_timeout: self.connect_timeout }.send(header, payload)
    }
    fn region(&self) -> Option<(RegionInfo, u64)> {
        self.try_connect();
        let g = self.inner.lock();
        g.region.clone().map(|r| (r, g.gen))
    }
    fn acquire_slot(&self) -> Option<u32> {
        self.inner.lock().credits.pop()
    }
    fn release_slot(&self, slot: u32) {
        self.inner.lock().credits.push(slot);
    }
}

pub struct TcpTransport {
    pub addr: String,
    pub connect_timeout: Duration,
}

impl KvTransport for TcpTransport {
    fn name(&self) -> &'static str {
        "tcp"
    }
    fn send(&self, header: &MigrationHeader, payload: MemDesc) -> Result<Box<dyn Transfer>> {
        let addr = self.addr.to_socket_addrs()?.next().context("pd: resolve")?;
        let mut s = TcpStream::connect_timeout(&addr, self.connect_timeout)
            .with_context(|| format!("pd: connect {}", self.addr))?;
        s.set_nodelay(true)?;
        // SAFETY: 调用方保证 payload 在本调用期间有效（同步发送）。
        encode(header, unsafe { payload.as_slice() }, &mut s)?;
        Ok(Box::new(DoneTransfer))
    }
}

/// D 侧收件表：`request_id → Migration`。接收线程写入，engine 配对时 `take`；超时未被领走的条目由 `gc` 丢弃。
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

/// D 侧 TCP 接收线程：每条连接一次迁移。`model_tag` 不匹配直接丢（计 rejected）。返回实际绑定端口（测试用 0）。
/// NIXL 后端的接收方是 agent 回调 + notif，不走这里；但落点同样是 `IncomingKv::put`。
pub fn spawn_receiver(bind: &str, model_tag: String, table: Arc<IncomingKv>) -> Result<u16> {
    spawn_receiver_with_pool(bind, model_tag, table, Arc::new(VecPool))
}
/// 同上，payload 落进 `pool`（executor 的 pinned 池 → D 侧零拷贝 H2D）。
pub fn spawn_receiver_with_pool(bind: &str, model_tag: String, table: Arc<IncomingKv>, pool: Arc<dyn HostBufPool>) -> Result<u16> {
    spawn_receiver_shared(bind, model_tag, table, pool, None)
}
/// 同上，并可带本地共享槽区：每条连接先发 hello(RegionInfo)，之后循环收消息（持久连接）；槽模式消息不带 payload，
/// 构造 `SlotPayload` 指向槽；payload drop → 槽号经本连接的写线程回给 P。
pub fn spawn_receiver_shared(
    bind: &str,
    model_tag: String,
    table: Arc<IncomingKv>,
    pool: Arc<dyn HostBufPool>,
    region: Option<Arc<LocalRegion>>,
) -> Result<u16> {
    let l = TcpListener::bind(bind).with_context(|| format!("pd: bind {bind}"))?;
    let port = l.local_addr()?.port();
    std::thread::Builder::new().name("pd-recv".into()).spawn(move || {
        for conn in l.incoming() {
            let Ok(s) = conn else { continue };
            let _ = s.set_nodelay(true);
            let (model_tag, table, pool, region) = (model_tag.clone(), table.clone(), pool.clone(), region.clone());
            let _ = std::thread::Builder::new().name("pd-conn".into()).spawn(move || serve_conn(s, model_tag, table, pool, region));
        }
    })?;
    Ok(port)
}
fn serve_conn(mut s: TcpStream, model_tag: String, table: Arc<IncomingKv>, pool: Arc<dyn HostBufPool>, region: Option<Arc<LocalRegion>>) {
    let (ctx, crx) = crossbeam_channel::unbounded::<u32>();
    if let Some(r) = &region {
        let hello = serde_json::to_vec(&r.info).unwrap();
        let mut w = Vec::with_capacity(8 + hello.len());
        w.extend_from_slice(&MAGIC_HELLO.to_le_bytes());
        w.extend_from_slice(&(hello.len() as u32).to_le_bytes());
        w.extend_from_slice(&hello);
        if s.write_all(&w).is_err() {
            return;
        }
        if let Ok(mut wr) = s.try_clone() {
            std::thread::Builder::new()
                .name("pd-credit-w".into())
                .spawn(move || {
                    for slot in crx {
                        if wr.write_all(&slot.to_le_bytes()).is_err() {
                            break;
                        }
                    }
                })
                .ok();
        }
    }
    loop {
        match decode_into(&mut s, pool.as_ref()) {
            Ok((header, payload)) if header.model_tag == model_tag => {
                let payload: Box<dyn HostBuf> = match header.slot {
                    None => payload,
                    Some(slot) => {
                        let Some(r) = region.as_ref() else {
                            tracing::warn!(id = header.request_id, "pd: slot message but no local region, dropped");
                            table.reject();
                            continue;
                        };
                        let len = header.data_len();
                        if slot as u64 >= r.info.slots as u64 || len as u64 > r.info.slot_bytes {
                            tracing::warn!(id = header.request_id, slot, len, "pd: slot out of range, dropped");
                            table.reject();
                            continue;
                        }
                        // SAFETY: slot 已校验在区内。
                        let host = if r.host_base.is_null() { std::ptr::null() } else { unsafe { r.host_base.add(slot as usize * r.info.slot_bytes as usize) } };
                        Box::new(SlotPayload { slot, len, host, credit: ctx.clone() })
                    }
                };
                tracing::debug!(id = header.request_id, units = header.num_units, slot = ?header.slot, "pd: migration received");
                table.put(Migration { header, payload, received_at: Instant::now() });
            }
            Ok((header, _)) => {
                tracing::warn!(id = header.request_id, got = %header.model_tag, want = %model_tag, "pd: model_tag mismatch, dropped");
                table.reject();
            }
            Err(e) if e.is::<Eof>() => break,
            Err(e) => {
                tracing::warn!(error = %e, "pd: bad migration");
                table.reject();
                break;
            }
        }
    }
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
            num_units: 2,
            unit_bytes: 4,
            slot: None,
        }
    }
    /// 共享槽区握手 + 槽模式消息 + credit 回传（host_base 用一段普通内存模拟 shm）+ 无槽退回 TCP payload。
    #[test]
    fn shared_transport_slot_roundtrip_and_credit() {
        let table = Arc::new(IncomingKv::default());
        let mem = vec![7u8; 64];
        let region = Arc::new(LocalRegion {
            info: RegionInfo { kind: "shm".into(), slots: 2, slot_bytes: 32, shm_name: Some("x".into()), ipc_handle: None, device_uuid: vec![1; 16] },
            host_base: mem.as_ptr(),
        });
        let port = spawn_receiver_shared("127.0.0.1:0", "qwen3".into(), table.clone(), Arc::new(VecPool), Some(region)).unwrap();
        let t = SharedTransport::new(format!("127.0.0.1:{port}"), Duration::from_secs(2));
        let (info, gen) = t.region().expect("handshake");
        assert_eq!((info.slots, info.slot_bytes, gen), (2, 32, 1));
        let s0 = t.acquire_slot().unwrap();
        let s1 = t.acquire_slot().unwrap();
        assert!(t.acquire_slot().is_none(), "2 credits only");
        let mut h = hdr(1, "qwen3");
        h.slot = Some(s1);
        assert_eq!(h.payload_len(), 0);
        t.send(&h, MemDesc::of(&[])).unwrap();
        let wait = |f: &dyn Fn() -> bool| {
            let deadline = Instant::now() + Duration::from_secs(3);
            while Instant::now() < deadline && !f() {
                std::thread::sleep(Duration::from_millis(5));
            }
        };
        wait(&|| table.contains(1));
        let m = table.take(1).unwrap();
        assert_eq!((m.payload.slot(), m.payload.len()), (Some(s1), 8));
        assert_eq!(m.payload.as_slice(), &mem[s1 as usize * 32..s1 as usize * 32 + 8]);
        drop(m); // → credit 回到 P
        let deadline = Instant::now() + Duration::from_secs(3);
        let mut got = None;
        while Instant::now() < deadline && got.is_none() {
            got = t.acquire_slot();
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(got, Some(s1), "credit must come back after payload drop");
        t.release_slot(s0);
        assert!(t.acquire_slot().is_some());
        let z = [0u8; 8];
        t.send(&hdr(2, "qwen3"), MemDesc::of(&z)).unwrap();
        wait(&|| table.contains(2));
        assert_eq!(table.take(2).unwrap().payload.len(), 8);
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
        let z = [0u8; 8];
        let mut x = t.send(&hdr(1, "qwen3"), MemDesc::of(&z)).unwrap();
        assert_eq!(x.poll().unwrap(), TransferState::Done);
        t.send(&hdr(2, "glm53"), MemDesc::of(&z)).unwrap();
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
        let mut m = Migration { header: hdr(5, "q"), payload: Box::new(vec![0u8; 8]), received_at: Instant::now() };
        m.received_at -= Duration::from_secs(10);
        table.put(m);
        assert_eq!(table.gc(Duration::from_secs(2)), 1);
        assert!(!table.contains(5));
    }
}
