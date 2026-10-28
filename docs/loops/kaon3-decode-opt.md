# loop: kaon3-decode-opt —— 用户定的 step A/B/C 排查优化流程（2026-10-28）

> 前情（必读教训）:两刀 MoE kernel 优化(cp.async流水 R77 1.83× / gate+up融合 R91 grid翻倍)graph态端到端都 <1%。
> 根因:靠 NO_GRAPH 分段定靶子(说 MoE 占 63%)是假象,被 cudagraph 摊薄。**改任何 kernel 前必须先验证那段是 graph 态真大头**。
> 用户批评:"做之前反复强调验证,方案是屁股想出来的" —— 不再未验证就动手。

## 用户定的流程（严格按此）
- **Step A（测）**:跑 5 轮,测 dinfer 的:①启动时间 ②TTFT ③TPOT ④每一步的 exec 时间 ⑤wait 时间 ⑥带宽 ⑦GPU 利用率。
- **Step B（改）**:针对**耗时最长**的那段,深入分析+优化,提交,编译最新代码运行。
- **Step C（验+回滚，用户 2026-10-28 完善）**:端到端测试。收益差则**回滚**,回 Step B。**但每次结果（含回滚的负结果）都要进 CSV 和 dashboard**（`test/kaon3/perf-trend.*`，带 CHANGE 说明）。Step B↔C 重复 **5 次无进展** → 回 Step A 重新分析,继续跑 loop。
  - CSV/dashboard:`bench_compare` 或直接写 perf-trend.csv 一行(round/change/指标),重生成 html,本机 8090 可看。回滚的轮次也记(change 注明"回滚:端到端 X%<门槛")。

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

## ★★★★ D4 2026-10-28 dense wait机理读清(mega_common.cuh)+回答用户两想法(有数据/机理)
**wait机理**:持久CTA主循环(mega_common:140)每task=wait_stage(自旋等依赖`while(flag<need)__nanosleep(40)`)→run(真算)→signal。wait_cyc=自旋clock累加。dense decode每stage task极少(tokens=1一个tile),但stage真依赖链(GATE等NORM,ACT等GATE+UP,DOWN等ACT)→CTA领到后段task时前段没signal完就自旋空等。
**wait高根因**:非"算得慢",是**stage依赖链串行+decode task太少喂不满CTA**→大量CTA领不到活或领到就等→自旋。=用户说的缺compute overlap(stage间)。
**回答想法1(对,但overlap类型不同)**:第一刀是K维load/compute overlap(cp.async);dense fp8_gemm_tile已有cp.async(:135 wait_group 0全等)。dense的wait是**stage维**(GATE/UP/DOWN间)串行。可overlap的只有GATE+UP(互相独立,ACT的dep2已等两支路=本想并行),wait高=并行没真发生(task少CTA没同时跑GATE和UP)。
**回答想法2(需诚实纠正)**:"256线程128空转"是fp8_gemm_tile内部加载A的线程利用(线程维,mega_mlp_core:103),与stage wait(CTA/task维)**是两回事**。128→256对wait 3.95×没用,且只影响A加载(decode A仅1-8行本就小),收益极小。不是真靶子。
**真靶子**:decode dense mega task太少喂不满持久CTA→大量CTA自旋等=**decode小batch下持久megakernel调度模型本身不适配**。
**Step B验证先行(不直接改kernel)**:NO_GRAPH下dense占24%+wait3.95×,graph会摊薄→先skip差分证dense graph态真占比(加DENSE_SKIP纯计时),占比够大再改。不重蹈两刀覆辙。

## ★★★★★★ D5 2026-10-28 颠覆性:dense/MoE整条删掉 TPOT 都不变→两段都不在decode关键路径!
graph态skip差分(base/DENSE_SKIP整条dense不跑/MOE_SKIP整条MoE不跑):
| | c1 TPOT | rate8 TPOT | 吞吐 |
|---|---|---|---|
| base | 18.80 | 51.84 | 584 |
| DENSE_SKIP | 18.83 | 51.89 | 586 |
| MOE_SKIP | 18.83 | 51.96 | 584 |
**删整条dense→TPOT不变。删整条MoE→TPOT不变。两个都0影响。**
**颠覆性结论**:dense和MoE根本**不在decode关键路径**,与别的(attn/host/调度)并行或被覆盖,省掉不影响总时间。NO_GRAPH分段"MoE 63%/dense 24%"**全是假象**,graph态这俩不决定TPOT。
**彻底解释两刀<1%**:MoE/dense不在关键路径,优化它们端到端当然没用(第一刀cp.async/第二刀融合都白做)。也解释D3 wait 3.95×对端到端无感——wait在非关键路径段。
**真关键路径**:删了dense+MoE还18.8/51.8,只剩 attn+2×norm+残差+oproj+采样/调度/host。真大头在这里。
**Step A定论改写**:之前所有"MoE/dense是大头"作废。Step B靶子转向关键路径(attn/host/调度)。
**下一步**:验关键路径谁大——ATTN_SKIP(attn跳过会坏KV,改成跳过部分/测;或测关attn的qkv/paged各段)、或测host(纯空转serve看每步固定开销)。先ATTN_SKIP+纯host步时。
**方法教训(写knowledge)**:判断某段是否瓶颈,最硬的是"整段skip看端到端动不动",比任何分段计时(NO_GRAPH/nsys)都可信。应该一开始就做这个,省掉前面所有MoE弯路。

## D6 2026-10-28 双skip(dense+moe都删)+GPU util:分清关键路径是attn计算还是host/串行
base vs DENSE_SKIP+MOE_SKIP(两段都删),带GPU util采样。判据:
- 两段删后TPOT仍不掉 + util掉→瓶颈=attn/host(GPU空出但总时没变=host/串行开销主导);
- util仍高→attn占满GPU。
结果落tmp_bothskip_RESULT.txt,等。

## ★★★★★★ D7 2026-10-28 双删定论:dense+MoE一起删 TPOT仍51.9不变+GPU仍~100%→关键路径=attn主干
| | TPOT | GPU util |
|---|---|---|
| base | 51.92 | 100% |
| dense+MoE都删 | 51.94 | ~100%(37×100/33×99) |
**两条支路一起删,TPOT纹丝不动+GPU还满**→删掉后GPU被剩下的段(attn+norm+oproj+采样)填满,总时不变。dense/MoE本就藏在attn时间里(并行/被覆盖)。
**Step B真靶子=attn段**(qkv→q/k norm→rope→paged gqa→oproj):decode关键路径,占满GPU决定TPOT。之前整轮(MoE两刀)全打偏,attn才是。
**attn不能简单skip(坏KV崩)**→需attn内部分段看谁大(qkv GEMM/paged gqa kernel/q_norm k_norm v_norm逐头/rope/oproj)。stage-time的"attn段"是整个合一(0.15ms NO_GRAPH)。
**下一步**:attn内部细分计时(加KAON3_ATTN_TIME或逐kernel event),定位qkv/paged/norm/rope哪个大。paged gqa decode(gqa_decode)和qkv(cuBLASLt)是热点候选。对标vLLM decode attn(flashinfer/flash-attn)。
**方法沉淀**:整段skip(D5删dense/MoE各0影响,D7双删0影响)是定位关键路径最硬手段,比NO_GRAPH/nsys分段可信。本轮教训=该最先做skip定关键路径,而非信NO_GRAPH的"MoE63%"做两刀。

## ★★★★★ D8 2026-10-28 plen扫描:TPOT线性=固定~13ms+paged~4.6ms/1K KV,固定开销已超vLLM全部
单流c1 TPOT vs prompt_len:
| plen | TPOT |
|---|---|
| 128 | 13.43 |
| 512 | 16.06 |
| 1024 | 19.33 |
| 4096 | 31.90 |
线性拟合:**固定~13ms(与KV无关) + ~4.6ms/1K KV(paged attention读KV)**。
**两大头现形**:
1. **固定每步~13ms**(plen128的TPOT)=qkv/norm/rope/oproj+host/launch,与KV无关。**vLLM整个TPOT才7.57ms,我们光固定开销13ms已超vLLM全部**→最该先攻。
2. paged attention ~4.6ms/1K(随KV线性)=读KV kernel。
对标vLLM 7.57(约1024):我们19.33=固定13+paged~6,两头都慢。
**Step B靶子优先级**:①固定13ms(占比最大、plen短时是全部)——先分清GPU算vs host/launch:MOE_SKIP+DENSE_SKIP+plen128压最小GPU活,看TPOT剩多少=纯host/launch地板;②paged attention(长上下文才显著)。
**下一步**:测 DENSE_SKIP+MOE_SKIP+plen128 的TPOT(最小GPU活)=固定开销里GPU vs host拆分。若还接近13=host/launch主导(调度/同步,非kernel);若大降=固定里GPU算(qkv/norm/oproj)主导。
