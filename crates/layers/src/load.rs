//! 装载 / 启动骨架公共件（用户 2026-10-20 "executor 胶水必须是 core 公共件，不按模型复制" 的装载侧对应物）：
//! 四个模型 crate 原各写一遍 per-layer `Instant` + `mem_info` 前后差 + `layer loaded` 日志，这里统一成一份。
//!
//! - [`LoadProgress`]：逐层装载进度（每层一行 `layer loaded layer=l/n kind=… secs=… gb=… gbs=…`）+ 汇总一行
//!   （`model loaded layers=n secs=… gb=… avg_gbs=… rank=… world=…`）；顶层张量用 [`LoadProgress::timed`]。
//!   模型 crate 只提供装层闭包（开分片 / 建表 / `XxxLayer::load_tp`）并返回一个 `&'static str` 标签（kda/moe 之类），
//!   模型名 / 字段不进本 crate。
//! - [`StartupTimer`]：服务启动三段计时 `startup done model_load_ms=… graph_capture_ms=… warmup_ms=… total_ms=…`
//!   （graph / warmup 两段由 executor 自报，见 `exec::DecodeGraphs::capture_secs`）。
//!
//! 显存口径：`gb` = 装该层前后 `cuMemGetInfo` free 的差（权重 + 该层私有缓冲），`gbs` = gb / secs（mmap 读盘 + H2D +
//! 分片切片的合速）。cudarc `alloc` 是同步 `cuMemAlloc`，所以 gb 不依赖同步；`secs` 默认在每层后 `synchronize` 一次让
//! H2D 真落地（`sync_per_layer(false)` 只量入队时间，给不想打断 H2D 流水的装载用）。
use anyhow::Result;
use dinfer_kernels::Stream;
use std::time::Instant;

/// `mem_info` / `synchronize` 的最小抽象：让进度统计能在无 GPU 的单测里跑（`Stream` 实现它）。
pub trait MemProbe {
    /// 当前设备 free 字节。
    fn free_bytes(&self) -> Result<usize>;
    fn sync(&self) -> Result<()>;
}

impl MemProbe for Stream {
    fn free_bytes(&self) -> Result<usize> {
        Ok(self.mem_info()?.0)
    }
    fn sync(&self) -> Result<()> {
        self.synchronize()
    }
}

/// 一次模型装载的进度与汇总。`new` 起表；`load_layers` 装层；`finish` 打汇总并返回统计。
pub struct LoadProgress {
    t0: Instant,
    rank: usize,
    world: usize,
    sync_per_layer: bool,
    layers: usize,
    layers_secs: f64,
    layers_bytes: u64,
}

/// `LoadProgress::finish` 的返回：给调用方写进自己的 startup 汇总 / 测试断言。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoadStatsSummary {
    pub layers: usize,
    pub layers_secs: f64,
    pub layers_bytes: u64,
    pub total_secs: f64,
}

impl LoadProgress {
    /// `(rank, world)`：TP 分片坐标（单卡 `(0, 1)`），只用于日志。
    pub fn new(tp: (usize, usize)) -> Self {
        Self { t0: Instant::now(), rank: tp.0, world: tp.1, sync_per_layer: true, layers: 0, layers_secs: 0.0, layers_bytes: 0 }
    }

    /// 每层装完是否 `synchronize`（默认 true：`secs` 含 H2D 落地；false 只量入队，`gb` 不受影响）。
    pub fn sync_per_layer(mut self, on: bool) -> Self {
        self.sync_per_layer = on;
        self
    }

    /// 顶层张量 / 其它非逐层步骤：一行 `loaded what=… secs=…`。
    pub fn timed<T>(&self, what: &'static str, f: impl FnOnce() -> Result<T>) -> Result<T> {
        let t = Instant::now();
        let r = f()?;
        tracing::info!(
            rank = self.rank,
            what,
            secs = format!("{:.2}", t.elapsed().as_secs_f64()),
            "loaded"
        );
        Ok(r)
    }

    /// 逐层装载：`load(l)` 返回 `(层, 标签)`，标签是模型 crate 给的 `&'static str`（如 `"kda/moe"`、`"gdn"`），
    /// 每层打一行、累计显存与耗时。`probe` 通常就是 `Stream`。
    pub fn load_layers<L>(
        &mut self,
        probe: &impl MemProbe,
        n: usize,
        mut load: impl FnMut(usize) -> Result<(L, &'static str)>,
    ) -> Result<Vec<L>> {
        let mut out = Vec::with_capacity(n);
        for l in 0..n {
            let t = Instant::now();
            let free0 = probe.free_bytes()?;
            let (layer, kind) = load(l)?;
            if self.sync_per_layer {
                probe.sync()?;
            }
            let free1 = probe.free_bytes()?;
            let rec = self.record(l, n, kind, t.elapsed().as_secs_f64(), free0.saturating_sub(free1) as u64);
            tracing::info!(
                rank = self.rank,
                layer = format!("{}/{n}", l + 1),
                kind,
                secs = format!("{:.2}", rec.secs),
                gb = format!("{:.2}", rec.gb),
                gbs = format!("{:.2}", rec.gbs),
                "layer loaded"
            );
            out.push(layer);
        }
        Ok(out)
    }

    /// 纯统计（无 GPU 可测）：记一层，返回该层的 (secs, gb, gbs)。
    fn record(&mut self, _l: usize, _n: usize, _kind: &str, secs: f64, bytes: u64) -> LayerRec {
        self.layers += 1;
        self.layers_secs += secs;
        self.layers_bytes += bytes;
        let gb = bytes as f64 / 1e9;
        LayerRec { secs, gb, gbs: gb / secs.max(1e-6) }
    }

    /// 汇总一行并返回统计。`total_secs` 从 `new` 起算（含顶层与 `timed` 的部分）。
    pub fn finish(&self) -> LoadStatsSummary {
        let total_secs = self.t0.elapsed().as_secs_f64();
        let gb = self.layers_bytes as f64 / 1e9;
        tracing::info!(
            rank = self.rank,
            world = self.world,
            layers = self.layers,
            secs = format!("{:.1}", self.layers_secs),
            gb = format!("{gb:.2}"),
            avg_gbs = format!("{:.2}", gb / self.layers_secs.max(1e-6)),
            total_secs = format!("{total_secs:.1}"),
            "model loaded"
        );
        LoadStatsSummary { layers: self.layers, layers_secs: self.layers_secs, layers_bytes: self.layers_bytes, total_secs }
    }
}

struct LayerRec {
    secs: f64,
    gb: f64,
    gbs: f64,
}

/// 服务启动计时：`start()` 在 `Service::start` 第一行；`model_loaded()` 在 executor 建好后；
/// `done(graph_capture_secs, warmup_secs)` 在 engine 起来后打 `startup done` 一行。
/// `model_load_ms` = 模型阶段墙钟 − graph − warmup（graph 捕获 / warmup 发生在 `Executor::new` 内，要扣掉才是纯装载）；
/// `engine_ms` = 模型阶段之后到 `done` 的其余启动步骤（tokenizer join / scheduler / spawn）。
pub struct StartupTimer {
    t0: Instant,
    model_phase_secs: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StartupTiming {
    pub model_load_ms: u64,
    pub graph_capture_ms: u64,
    pub warmup_ms: u64,
    pub engine_ms: u64,
    pub total_ms: u64,
}

impl StartupTimer {
    pub fn start() -> Self {
        Self { t0: Instant::now(), model_phase_secs: None }
    }

    /// 模型 + executor 建好（含 TP 其它 rank join）。
    pub fn model_loaded(&mut self) {
        self.model_phase_secs = Some(self.t0.elapsed().as_secs_f64());
    }

    /// 纯计算（可测）：按三段拆。
    pub fn split(&self, graph_capture_secs: f64, warmup_secs: f64) -> StartupTiming {
        let total = self.t0.elapsed().as_secs_f64();
        let model_phase = self.model_phase_secs.unwrap_or(total);
        let ms = |s: f64| (s.max(0.0) * 1e3) as u64;
        StartupTiming {
            model_load_ms: ms(model_phase - graph_capture_secs - warmup_secs),
            graph_capture_ms: ms(graph_capture_secs),
            warmup_ms: ms(warmup_secs),
            engine_ms: ms(total - model_phase),
            total_ms: ms(total),
        }
    }

    /// 打 `startup done` 一行并返回拆分。
    pub fn done(&self, graph_capture_secs: f64, warmup_secs: f64) -> StartupTiming {
        let t = self.split(graph_capture_secs, warmup_secs);
        tracing::info!(
            model_load_ms = t.model_load_ms,
            graph_capture_ms = t.graph_capture_ms,
            warmup_ms = t.warmup_ms,
            engine_ms = t.engine_ms,
            total_ms = t.total_ms,
            "startup done"
        );
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// 假显存：每次 `free_bytes` 调用后 free 下降固定字节，模拟"装一层吃 1 GB"。
    struct FakeMem {
        free: Cell<usize>,
        per_layer: usize,
        syncs: Cell<usize>,
    }
    impl MemProbe for FakeMem {
        fn free_bytes(&self) -> Result<usize> {
            Ok(self.free.get())
        }
        fn sync(&self) -> Result<()> {
            self.syncs.set(self.syncs.get() + 1);
            Ok(())
        }
    }

    #[test]
    fn load_layers_accumulates_bytes_and_syncs_per_layer() {
        let mem = FakeMem { free: Cell::new(10_000_000_000), per_layer: 1_000_000_000, syncs: Cell::new(0) };
        let mut p = LoadProgress::new((1, 4));
        let layers = p
            .load_layers(&mem, 3, |l| {
                mem.free.set(mem.free.get() - mem.per_layer);
                Ok((l * 10, if l % 2 == 0 { "a" } else { "b" }))
            })
            .unwrap();
        assert_eq!(layers, vec![0, 10, 20]);
        assert_eq!(mem.syncs.get(), 3);
        let s = p.finish();
        assert_eq!(s.layers, 3);
        assert_eq!(s.layers_bytes, 3_000_000_000);
        assert!(s.total_secs >= s.layers_secs);
    }

    #[test]
    fn load_layers_no_sync_option_and_error_propagates() {
        let mem = FakeMem { free: Cell::new(1 << 30), per_layer: 0, syncs: Cell::new(0) };
        let mut p = LoadProgress::new((0, 1)).sync_per_layer(false);
        let r: Result<Vec<()>> = p.load_layers(&mem, 2, |l| if l == 1 { anyhow::bail!("boom") } else { Ok(((), "x")) });
        assert!(r.is_err());
        assert_eq!(mem.syncs.get(), 0);
        assert_eq!(p.finish().layers, 1);
    }

    #[test]
    fn startup_split_subtracts_graph_and_warmup_from_model_phase() {
        let mut t = StartupTimer::start();
        t.model_phase_secs = Some(10.0);
        // 伪造 t0 让 total 可控：直接测 split 的算术
        let s = StartupTiming { ..t.split(2.0, 0.5) };
        assert_eq!(s.model_load_ms, 7_500);
        assert_eq!(s.graph_capture_ms, 2_000);
        assert_eq!(s.warmup_ms, 500);
        // total 是真实 elapsed（≈0），engine = total − model_phase 被夹到 0
        assert_eq!(s.engine_ms, 0);
        t.model_loaded();
        assert!(t.split(0.0, 0.0).model_load_ms < 1_000);
    }
}
