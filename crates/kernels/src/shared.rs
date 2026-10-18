//! PD 分离的共享槽区（design §10.3 后端 b）：D 侧建一块 `slots × slot_bytes` 的区，P 侧映射后直接把 KV 写进槽里，
//! 控制面只传槽号。两种介质：
//! - `Shm`：POSIX 共享内存（/dev/shm）+ `cuMemHostRegister` 钉住 → 两个进程都能当 pinned 用（P D2H 写、D H2D 读），
//!   不依赖 GPU P2P，跨卡 / 同卡都可用（4090 之间 `nvidia-smi topo -p2p r` 全 CNS，IPC 跨卡 open 直接报错）。
//! - `CudaIpc`：D 侧 `cuMemAlloc` + `cuIpcGetMemHandle`，P 侧 `cuIpcOpenMemHandle` → 同一张卡上 P 的 D2D 写
//!   （探针 64 MB 0.15 ms ≈ 446 GB/s），不过 host。跨卡无 P2P 时 open 失败，调用方回退 Shm / TCP。
use anyhow::{bail, ensure, Context, Result};
use cudarc::driver::sys;
use crate::device::{Device, Stream};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionKind {
    Shm,
    CudaIpc,
}
impl RegionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            RegionKind::Shm => "shm",
            RegionKind::CudaIpc => "cuda-ipc",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "shm" => Some(RegionKind::Shm),
            "cuda-ipc" | "ipc" => Some(RegionKind::CudaIpc),
            _ => None,
        }
    }
}

enum Mem {
    /// mmap 的 host 区；`owner` = 创建方（Drop 时 unlink）。
    Host { base: *mut u8, owner: bool, name: String },
    /// 本进程 cuMemAlloc 的设备区（D 侧，导出 handle）。
    DevLocal { ptr: u64, handle: [u8; 64] },
    /// IPC 打开的对端设备区（P 侧）。
    DevRemote { ptr: u64 },
}

pub struct SharedRegion {
    mem: Mem,
    slots: usize,
    slot_bytes: usize,
    bytes: usize,
    uuid: [u8; 16],
}
unsafe impl Send for SharedRegion {}
unsafe impl Sync for SharedRegion {}

fn shm_path(name: &str) -> std::ffi::CString {
    std::ffi::CString::new(format!("/{}", name.trim_start_matches('/'))).unwrap()
}
fn mmap_shm(path: &std::ffi::CString, bytes: usize, create: bool) -> Result<*mut u8> {
    // SAFETY: libc 调用，参数已构造；失败走 errno。
    unsafe {
        let flags = if create { libc::O_RDWR | libc::O_CREAT | libc::O_EXCL } else { libc::O_RDWR };
        let fd = libc::shm_open(path.as_ptr(), flags, 0o600);
        ensure!(fd >= 0, "shm_open {:?}: {}", path, std::io::Error::last_os_error());
        if create && libc::ftruncate(fd, bytes as libc::off_t) != 0 {
            let e = std::io::Error::last_os_error();
            libc::close(fd);
            libc::shm_unlink(path.as_ptr());
            bail!("ftruncate {bytes} B: {e}");
        }
        let p = libc::mmap(std::ptr::null_mut(), bytes, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, fd, 0);
        libc::close(fd);
        ensure!(p != libc::MAP_FAILED, "mmap {bytes} B: {}", std::io::Error::last_os_error());
        Ok(p as *mut u8)
    }
}

impl SharedRegion {
    /// D 侧：建区。`Shm`：/dev/shm 文件 `kv-slots-<pid>`，mmap 后 `cuMemHostRegister`（写 0 触发缺页，注册前把页真的分出来）。
    /// `CudaIpc`：cuMemAlloc + 取 IPC handle。
    pub fn create(stream: &Stream, kind: RegionKind, slots: usize, slot_bytes: usize) -> Result<Self> {
        ensure!(slots > 0 && slot_bytes > 0, "SharedRegion: slots/slot_bytes must be > 0");
        let bytes = slots * slot_bytes;
        stream.bind_ctx()?;
        let uuid = stream.device().uuid()?;
        let mem = match kind {
            RegionKind::Shm => {
                let name = format!("kv-slots-{}", std::process::id());
                let path = shm_path(&name);
                let base = mmap_shm(&path, bytes, true)?;
                // SAFETY: 区已映射，长度 bytes。
                unsafe {
                    std::ptr::write_bytes(base, 0, bytes);
                    sys::cuMemHostRegister_v2(base as *mut _, bytes, sys::CU_MEMHOSTREGISTER_PORTABLE)
                        .result()
                        .with_context(|| format!("cuMemHostRegister {bytes} B"))?;
                }
                Mem::Host { base, owner: true, name }
            }
            RegionKind::CudaIpc => {
                let mut ptr: sys::CUdeviceptr = 0;
                let mut h = sys::CUipcMemHandle { reserved: [0; 64] };
                // SAFETY: ctx 已绑定；失败走 result。
                unsafe {
                    sys::cuMemAlloc_v2(&mut ptr, bytes).result().with_context(|| format!("cuMemAlloc {bytes} B for IPC region"))?;
                    if let Err(e) = sys::cuIpcGetMemHandle(&mut h, ptr).result() {
                        let _ = sys::cuMemFree_v2(ptr);
                        return Err(e).context("cuIpcGetMemHandle");
                    }
                }
                let mut handle = [0u8; 64];
                for (d, s) in handle.iter_mut().zip(h.reserved.iter()) {
                    *d = *s as u8;
                }
                Mem::DevLocal { ptr, handle }
            }
        };
        Ok(Self { mem, slots, slot_bytes, bytes, uuid })
    }
    /// P 侧：映射对端的区。`Shm` 用 name；`CudaIpc` 要求同一张卡（uuid 相等），否则 open 会因无 P2P 失败。
    pub fn open_remote(
        stream: &Stream,
        kind: RegionKind,
        slots: usize,
        slot_bytes: usize,
        shm_name: Option<&str>,
        ipc_handle: Option<&[u8]>,
        peer_uuid: &[u8],
    ) -> Result<Self> {
        let bytes = slots * slot_bytes;
        stream.bind_ctx()?;
        let uuid = stream.device().uuid()?;
        let mem = match kind {
            RegionKind::Shm => {
                let name = shm_name.context("shm region without name")?.to_string();
                let base = mmap_shm(&shm_path(&name), bytes, false)?;
                // SAFETY: 区已映射。
                unsafe {
                    sys::cuMemHostRegister_v2(base as *mut _, bytes, sys::CU_MEMHOSTREGISTER_PORTABLE)
                        .result()
                        .with_context(|| format!("cuMemHostRegister (remote shm) {bytes} B"))?;
                }
                Mem::Host { base, owner: false, name }
            }
            RegionKind::CudaIpc => {
                ensure!(peer_uuid == uuid, "cuda-ipc region: peer is on a different GPU (no P2P on this box); use shm");
                let hb = ipc_handle.context("cuda-ipc region without handle")?;
                ensure!(hb.len() == 64, "ipc handle must be 64 B");
                let mut h = sys::CUipcMemHandle { reserved: [0; 64] };
                for (d, s) in h.reserved.iter_mut().zip(hb.iter()) {
                    *d = *s as _;
                }
                let mut ptr: sys::CUdeviceptr = 0;
                // SAFETY: handle 来自对端 cuIpcGetMemHandle。
                unsafe {
                    sys::cuIpcOpenMemHandle_v2(&mut ptr, h, sys::CUipcMem_flags_enum::CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS as u32)
                        .result()
                        .context("cuIpcOpenMemHandle")?;
                }
                Mem::DevRemote { ptr }
            }
        };
        Ok(Self { mem, slots, slot_bytes, bytes, uuid })
    }
    pub fn kind(&self) -> RegionKind {
        match self.mem {
            Mem::Host { .. } => RegionKind::Shm,
            _ => RegionKind::CudaIpc,
        }
    }
    pub fn slots(&self) -> usize {
        self.slots
    }
    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn uuid(&self) -> [u8; 16] {
        self.uuid
    }
    pub fn shm_name(&self) -> Option<&str> {
        match &self.mem {
            Mem::Host { name, .. } => Some(name),
            _ => None,
        }
    }
    pub fn ipc_handle(&self) -> Option<[u8; 64]> {
        match &self.mem {
            Mem::DevLocal { handle, .. } => Some(*handle),
            _ => None,
        }
    }
    pub fn is_host(&self) -> bool {
        matches!(self.mem, Mem::Host { .. })
    }
    /// 槽 `slot` 的起始地址：Shm = host 指针（pinned），CudaIpc = 设备地址。
    pub fn slot_addr(&self, slot: usize) -> Result<u64> {
        ensure!(slot < self.slots, "slot {slot} >= {}", self.slots);
        let off = (slot * self.slot_bytes) as u64;
        Ok(match &self.mem {
            Mem::Host { base, .. } => *base as u64 + off,
            Mem::DevLocal { ptr, .. } | Mem::DevRemote { ptr } => *ptr + off,
        })
    }
}
impl Drop for SharedRegion {
    fn drop(&mut self) {
        // SAFETY: 析构，独占；错误忽略。
        unsafe {
            match &self.mem {
                Mem::Host { base, owner, name } => {
                    let _ = sys::cuMemHostUnregister(*base as *mut _);
                    libc::munmap(*base as *mut _, self.bytes);
                    if *owner {
                        libc::shm_unlink(shm_path(name).as_ptr());
                    }
                }
                Mem::DevLocal { ptr, .. } => {
                    let _ = sys::cuMemFree_v2(*ptr);
                }
                Mem::DevRemote { ptr } => {
                    let _ = sys::cuIpcCloseMemHandle(*ptr);
                }
            }
        }
    }
}

impl Device {
    /// 设备 UUID（cuDeviceGetUuid_v2）；判断两个进程是否在同一张卡。
    pub fn uuid(&self) -> Result<[u8; 16]> {
        let mut u = sys::CUuuid { bytes: [0; 16] };
        // SAFETY: 合法 CUdevice。
        unsafe { sys::cuDeviceGetUuid_v2(&mut u, self.cu_device()).result().context("cuDeviceGetUuid")? };
        let mut out = [0u8; 16];
        for (d, s) in out.iter_mut().zip(u.bytes.iter()) {
            *d = *s as u8;
        }
        Ok(out)
    }
}
