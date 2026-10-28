# loop: kaon3-decode-opt —— 用户定的 step A/B/C 排查优化流程（2026-10-28）

> 前情（必读教训）:两刀 MoE kernel 优化(cp.async流水 R77 1.83× / gate+up融合 R91 grid翻倍)graph态端到端都 <1%。
> 根因:靠 NO_GRAPH 分段定靶子(说 MoE 占 63%)是假象,被 cudagraph 摊薄。**改任何 kernel 前必须先验证那段是 graph 态真大头**。
> 用户批评:"做之前反复强调验证,方案是屁股想出来的" —— 不再未验证就动手。

## 用户定的流程（严格按此）
- **Step A（测）**:跑 5 轮,测 dinfer 的:①启动时间 ②TTFT ③TPOT ④每一步的 exec 时间 ⑤wait 时间 ⑥带宽 ⑦GPU 利用率。
- **Step B（改）**:针对**耗时最长**的那段,深入分析+优化,提交,编译最新代码运行。
- **Step C（验+回滚）**:端到端测试。收益差则**回滚**,回 Step B。重复 5 次无进展 → 回 Step A 重新分析。

## 用户的两个想法（待验证,不臆测）
1. **wait 不该有,高效计算要 compute overlap**:之前说 dense mega 的 wait 高(R55: GATE/UP/DOWN wait≈4×exec,CTA 自旋空等)。用户认为高效计算就不该有 wait,要让计算 overlap 起来(像第一刀 cp.async 在 MoE 做的,但 dense mega 没做)。→ Step A 测清 wait 到底多大、在哪段;若真大且是 overlap 缺失,Step B 做 overlap。
2. **"256/block 有 128 空转,两步填满有没有收益,128 改 256"**:待查清用户指哪个参数(MEGA_CTAS? BN? 线程数? grid?),查到用数据回答"128改256"的收益,不臆测。

## 测量手段现状（诚实）
- 启动时间:serve 日志 `model_load_ms` ✓
- TTFT/TPOT/吞吐:`bench`(绑核 60-91,seed42/全共享1024/128out) ✓
- GPU 利用率:nvidia-smi 采样(graph 态可测,decode 实测 ~100%) ✓
- **每步 exec/wait**:dense mega 有 `KAON3_DENSE_TIME`(wait_cyc/exec_cyc)、MoE 有 `KAON3_MOE_TIME`、各段 `KAON3_STAGE_TIME`——**但都 NO_GRAPH 态**(graph 态插桩与 cudagraph 冲突)。graph 态 kernel 级:nsys 在此环境抓不到(6轮失败)。→ exec/wait 只能 NO_GRAPH 测,结果标注"NO_GRAPH,graph 态会摊薄",与 graph 态 GPU util/TPOT 交叉判断。
- 带宽:kernel 时间 + 字节算(微基准),或 stage 时间反推。

## 环境
5090 GPU4(复用前确认 nvidia-smi 空),host 网络 detached 容器跑 serve(bench HTTP 通);NO_GRAPH 插桩态单独跑。生产态:graph 开 + MEGA_SPGU/DN=4 + ROUTER_GEMM_BI=0 + MOE_GROUPED=1 + KAON3_MEGA=1。

## 记录
每轮 Step 结果续写本文件 R 编号(接 kaon3-fp8gemm/moe-pipe 的编号体系,本 loop 用 D1/D2...)。

## D1 2026-10-28 两想法都查清=都指向 dense mega(非MoE):但按step A先测再改,不扎进去
**想法2定位**:用户"256/block 128空转"=`mega_mlp_core.cuh:103` 注释——dense mega kernel 用 **256 线程(kThreads),但加载 A 只 16行×(BK/16=8)=128 槽,前 128 线程参与,后 128 空转**。我第二刀做的是 MoE grid.z 翻倍(R91),**没碰这个 dense mega 128 空转**。用户记得对,是另一处。
**想法1定位**:wait 高也在 dense mega(R55 GATE/UP/DOWN wait≈4×exec,CTA 自旋空等),缺 compute overlap(第一刀 cp.async 只在 MoE 做了,dense mega 没做)。
**两想法都指向 dense mega 这一段**。但 dense NO_GRAPH 测只占 decode 8-15%(不可信,graph 摊薄)。
**按用户 step A/B/C,先不扎进 dense**:Step A 跑 5 轮测全套(启动/TTFT/TPOT/exec/wait/带宽/GPU util),确认 dense 是不是 graph 态耗时最长的段,再 Step B 改。不再未验证就动手。

## D2 2026-10-28 Step A graph态指标(5轮稳定):
| 指标 | 值 |
|---|---|
| 单流TTFT | 58ms |
| 单流TPOT | **19.33ms** |
| rate8吞吐 | 624 tok/s |
| rate8 TPOT | **43ms** |
| rate8 TTFT | 98-101ms |
| GPU util(decode) | **~100%**(159×100%/27×99%) |
对比vLLM非eager TPOT 7.57ms:单流19.33慢2.5×,rate8 43慢5.7×。
**GPU ~100%满载**:不是在"等"GPU空闲(否则util掉),但util 100%≠高效(CTA自旋也算忙)。分清"真算vs自旋空转"需kernel级exec/wait(step A还缺,下面NO_GRAPH插桩补,标注会摊薄)。
启动model_load_ms≈42000(前轮日志,此轮grep没中,已知~42s)。

## ★★★★★ D3 2026-10-28 Step A 完整(exec/wait补齐):dense wait=3.95×exec铁证,用户想法1正确
单层decode分段(NO_GRAPH,graph会摊薄):MoE 0.209ms(41%)/attn 0.15(29%)/dense 0.12(24%)/oproj 0.025(5%)。
**dense exec/wait(用户想法1核心)**:
| stage | exec | wait | wait/exec |
|---|---|---|---|
| GATE | 1.68M | 6.91M | 4.1× |
| UP | 1.66M | 6.91M | 4.2× |
| DOWN | 1.36M | 5.32M | 3.9× |
| 合计 | 4.9M | 19.3M | **3.95×** |
**用户想法1正确(数据铁证)**:dense mega wait=exec的3.95×,GPU花在等(CTA自旋空等signal)是真算的4倍=缺compute overlap。GPU util 100%满载但~80%是自旋、20%真算。
moe-time单流:gemm3=0.101ms(两刀已证改了端到端<1%,不碰)。
**Step A定论**:NO_GRAPH占比MoE最大但两刀证MoE改端到端没用(graph摊薄);**真靶子=dense wait 3.95×**(结构性比例,持久kernel窄stage自旋决定,graph改绝对值不改这比例,未被摊薄论推翻,有明确机理)。
**Step B方向**:攻dense mega wait,让GATE/UP/DOWN计算overlap(非CTA自旋等signal)。先读kernel看wait机制=为何自旋,再定overlap改法。改前仍按规矩:确认这段graph态真占比(skip差分)+改后A/B,<3%回滚。
