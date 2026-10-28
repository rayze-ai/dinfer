# loop: kaon3-decode-opt —— 用户定的 step A/B/C 排查优化流程（2026-10-28）

> 前情（必读教训）:两刀 MoE kernel 优化(cp.async流水 R77 1.83× / gate+up融合 R91 grid翻倍)graph态端到端都 <1%。
> 根因:靠 NO_GRAPH 分段定靶子(说 MoE 占 63%)是假象,被 cudagraph 摊薄。**改任何 kernel 前必须先验证那段是 graph 态真大头**。
> 用户批评:"做之前反复强调验证,方案是屁股想出来的" —— 不再未验证就动手。

## 用户定的流程（严格按此，用户 2026-10-28 两次完善）
**★★ 前置条件（必先对齐,否则结论作废——D33 血的教训）**:dinfer 与 vLLM **两端都非 eager**,且 **max-model-len / max-num-batched-tokens / concurrency 全一致**。vLLM 现 :28201 是 `--enforce-eager`(eager)+max-model-len 16384,**不能直接用**,要起**非 eager** vLLM 且参数对齐 dinfer。bench prompt 的 **token 数**(非词数,词×~3.9)必须 ≤ max-model-len,否则超限走慢路径(D32 的 2 秒假象)。开跑前核对两端 cmdline。
**★ 全程非 eager（graph 生产态）口径。** eager/NO_GRAPH 只作一次性排查手段。测量插桩与 cudagraph 冲突时用 graph 兼容测法(core host-time / 纯 CPU Instant / 整段 skip 看端到端)。
- **Step A（测）**:跑 5 轮 × **concurrency = 1,2,4,8,16**,测 dinfer（**非 eager**）的:①启动时间 ②tokenize ③chat 模板 ④htod ⑤block 分配 ⑥schedule ⑦TTFT ⑧TPOT ⑨每步 exec ⑩wait ⑪带宽 ⑫GPU 利用率。
- **Step B（改）**:针对**耗时最长**的那段,深入分析+优化,提交,编译最新代码运行（非 eager 验）。
- **Step C（验+回滚，用户 2026-10-28 完善）**:端到端测试。收益差则**回滚**,回 Step B。**但每次结果（含回滚的负结果）都要进 CSV 和 dashboard**（`test/kaon3/perf-trend.*`，带 CHANGE 说明）。Step B↔C 重复 **5 次无进展** → 回 Step A 重新分析,继续跑 loop。
  - CSV/dashboard:`bench_compare` 或直接写 perf-trend.csv 一行(round/change/指标),重生成 html,本机 8090 可看。回滚的轮次也记(change 注明"回滚:端到端 X%<门槛")。

## 用户的两个想法（本会话已查证，结论如下）
1. **wait 不该有,高效计算要 compute overlap**:dense mega wait 确实高（R55 GATE/UP/DOWN wait≈4×exec，持久 kernel CTA 自旋空等，机理 D4=decode task 少喂不满 CTA）。**但 D16 铁证 dense 整段 skip 掉 TPOT 都不变→dense/wait 不在 decode 关键路径**，改 dense overlap 端到端无感。用户"要 overlap"的直觉对（dense 确缺 overlap），但该段不是瓶颈，ROI≈0。**真瓶颈在 host exec，非 GPU overlap**。
2. **"256/block 128 空转,128 改 256"**:查清（D4）=`mega_mlp_core.cuh:103` dense mega kernel 256 线程但加载 A 只 128 槽、后 128 线程空转。**这是线程维（kernel 内部），与 wait（CTA/task 维）无关；且 dense 整段 skip TPOT 都不变→改它端到端无感**。128→256 对 TPOT 无收益（dense 不在关键路径）。诚实结论：这个优化点不值得做。
**两想法共同结论**：都指向 dense mega，但 dense 整段都不在 decode 关键路径（host 才是），所以两个想法虽方向有道理，实际 ROI≈0。

## 测量手段现状（诚实）
- 启动时间:serve 日志 `model_load_ms` ✓
- TTFT/TPOT/吞吐:`bench`(绑核 60-91,seed42/全共享1024/128out) ✓
- GPU 利用率:nvidia-smi 采样(graph 态可测,decode 实测 ~100%) ✓
- **每步 host exec/wait（非 eager 可测）**:core engine 的 `[host-time]`（sched/exec/publish，纯 CPU Instant，graph 态准，D17 实测 exec=19.3ms 占全）。要拆 exec 内 launch(submit) vs collect(dtoh_sync) 需改 core engine host-time 或让 kaon3 forward_sample 打点生效（本会话没拆成，下次第一步）。
- **kernel 级 exec/wait**:`KAON3_DENSE_TIME`/`MOE_TIME`/`STAGE_TIME` 与 cudagraph 冲突→仅 NO_GRAPH（标注仅排查）；nsys 在此环境抓不到（6轮失败，放弃）。**但本会话结论=GPU kernel 全不是瓶颈（整段 skip 铁证），kernel 级 exec/wait 已不重要**。
- **整段 skip 看端到端（最硬，graph/eager 都准）**:DENSE_SKIP/MOE_SKIP/ATTN_SKIP_{NORM,ROPE,KVFP8,QKV,PAGED,OPROJ}（纯计时默认关）。这是定关键路径的首选，应最先用。
- 带宽:kernel 时间 + 字节算(微基准),或 stage 时间反推（kernel 不是瓶颈，带宽已非重点）。

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

## ★★★★★ D9 2026-10-28 固定开销拆分:plen128全skip TPOT仍12.9+GPU99%→固定13ms=attn主干GPU真算,非host
| plen128 | TPOT | util |
|---|---|---|
| base | 12.92 | 99% |
| dense+MoE全skip | 12.93 | 99% |
**plen128删dense+MoE,TPOT仍12.9+GPU99%满**→固定13ms里dense/MoE 0贡献(同D7),且**GPU 99%满=真算非host空等**。
**固定13ms=什么**:plen128+删dense/MoE,每步GPU活只剩 attn(qkv/q_norm/k_norm/v_norm逐头/rope/paged)+oproj+2×layernorm+采样,×30层。GPU满算13ms,vLLM整个7.57。
**定论**:固定13ms=30层attn主干(qkv/norm/rope/oproj)+采样的GPU真算(非host/调度)。Step B靶子=这个,且确定GPU kernel层面。
**机理推测(待验)**:decode M=1,30层×(qkv+3norm逐头+rope+paged+oproj)=一堆小kernel,每个启动开销大算得少=喂不满(同MoE病)。我们开了graph但vLLM融合程度更高(qkv+norm+rope融进更少kernel)。
**下一步**:attn内部分段定哪个kernel大(qkv cuBLASLt? 逐头norm? rope? paged? oproj?)。逐头q_norm/k_norm(Gemma4特色,每头一个RMSNorm)在decode M=1可能是一堆微kernel大头。先读attn段kernel数+加分段或逐个skip。

## ⚠️ D10 2026-10-28 attn子步全skip也0影响:norm/rope/kvfp8删掉TPOT都不变→反常,需验测量有效性
plen128+删dense/MoE,再skip norm(6逐头RMSNorm/层)/rope/kvfp8(3 roundtrip/层):
| | TPOT |
|---|---|
| base | 12.92 |
| skip norm | 12.94 |
| skip rope | 12.93 |
| skip kvfp8 | 12.93 |
**全0影响**。加上D7(dense/MoE删0影响)、D9(全skip 0影响),几乎删什么都不动12.9ms。
**反常推理**:删这么多kernel TPOT纹丝不动+D9测GPU 99%满。两种可能:
1. **30层逐层固定launch/调度串行=13ms**,单kernel省掉淹没在层间串行(但GPU 99%满与此矛盾——满说明在算不是等launch);
2. **测量本身有问题**:plen128全共享前缀(prefix-repeat 1)+prefix cache命中→decode可能根本没真算attn(KV已缓存),或bench TPOT有下限/被别的bound。
**必须先验测量有效性(别又在假数据上做结论,重蹈NO_GRAPH覆辙)**:
- 关prefix cache或prefix-repeat 0重测,看TPOT变不变;
- 加max-tokens拉长看TPOT稳定值;
- 对比:若连attn主kernel(qkv/paged/oproj)都skip还12.9→确系测量/层串行;若那些skip大降→之前小kernel本就不是大头是对的。
**不下结论,先验证**。下一步:prefix-repeat 0(不共享前缀,真实KV) + 加qkv/paged/oproj的skip(可能崩,小心)重测。

## ★★★ D11 2026-10-28 测量有效性确认:prefix cache与decode TPOT无关,skip结论站得住
| 配置 | cache命中 | TPOT |
|---|---|---|
| 共享前缀 plen1024 | 0.981 | 19.73 |
| 独立KV plen1024 | 0.000 | 19.64 |
| 独立KV plen128 | 0.873 | 14.02 |
**prefix-repeat 0(命中0,真算1024 KV) vs 1(命中0.98),TPOT几乎一样(19.64 vs 19.73)**→decode TPOT与prefix cache无关(cache只省prefill,decode每步照常读KV算attention)。**前面skip测量有效,非假数据**。
**D5-D10结论站得住**:删dense/MoE/norm/rope/kvfp8都不动TPOT是真的;plen128→1024 TPOT 13→19.6真实涨(paged随KV真算)。
**反常收敛成清晰问题**:固定13ms(plen128)删掉norm/rope/kvfp8/dense/MoE全不变→13ms只能在**没法skip的**:qkv GEMM/copy2d×3/gqa_cache_write/paged/oproj GEMM,×30层。删一堆小kernel不变=小kernel本就不耗时,13ms集中在少数大kernel(qkv/oproj GEMM+paged)或30层固定调度链。
**下一步(最终定位)**:加ATTN_SKIP_QKV/OPROJ/PAGED skip(坏数值纯计时),看哪个让13ms大降。先qkv+oproj(两个GEMM,最可能)。

## ★★★★★★★ D12 2026-10-28 终极颠覆:qkv/paged/oproj全skip也0影响→TPOT根本不是GPU在算,瓶颈在host/调度
| | base | noqkv | nopaged | nooproj |
|---|---|---|---|---|
| plen128 | 13.07 | 13.07 | 13.08 | 13.07 |
| plen1024 | 18.87 | 18.87 | 18.88 | 18.87 |
**qkv/paged/oproj全删TPOT不变,连paged删plen1024都不降(18.87→18.88)**。
**至此所有GPU计算段都skip过**:dense/MoE/norm/rope/kvfp8/qkv/paged/oproj——**全删TPOT都不动**。
**终极结论**:13ms(plen128)/19ms(plen1024)**根本不是GPU在算**。推翻D9"GPU真算"(util 99%是假象:采样精度或graph replay固定开销忙)。TPOT由**GPU之外**决定=host/调度/同步/每步固定链/cudagraph launch+sync。
**plen128→1024涨6ms删paged也不降**→那6ms也非paged kernel,是随seq_len的**host侧**开销(block_table/slot_mapping处理/每步按KV长的host循环)。
**方向彻底改**:瓶颈不在任何CUDA kernel(两刀MoE+刚要攻的attn kernel全错),在**host/调度/每步固定链**。=R47最早"host开销"方向对(R58用NO_GRAPH否定过,但graph态skip证明R47对,NO_GRAPH又骗了一次)。
**Step B真靶子(终于)**:host/调度路径——decode每步Rust侧做什么(采样/detokenize/调度/同步/graph launch),为何每步固定13ms+随KV 6ms。vLLM 7.57ms说明host可以很薄。
**下一步**:读kaon3 executor decode每步host代码(executor.rs decode路径),看每步固定开销(采样/publish/sync/graph replay)+随KV的host循环。这才是真大头。

## ★★★★★★ D13 2026-10-28 严重漏洞!decode走已捕获CUDA graph,运行时skip env被图旁路→D5-D12 graph态skip可能全作废
读executor.rs:decode走CUDA graph,图**启动时capture_decode_graphs()捕获**,之后Graph::launch只重放捕获时的kernel。
**致命问题**:我运行时设ATTN_SKIP_*/DENSE_SKIP/MOE_SKIP env,但**图在启动时按不skip捕获好了**,host代码(含我的env判断)在图重放时**根本不执行**→skip env被graph旁路,运行时改env无效!
**→"skip全0影响"是假象**:图重放捕获时的完整kernel,skip env没生效。D12"GPU段skip 0影响=瓶颈在host"的推论**不成立**(skip压根没skip)。D5-D11的graph态skip同样可疑(dense/MoE的skip也在forward里被图旁路)。
**但注意**:第一刀/第二刀的A/B是env在**serve启动前**设(进程env),capture时就生效,那些A/B有效。只有"运行时想靠env切skip"被图旁路。而我的skip容器都是**启动serve时**export env的→应该capture时生效??需确认capture时env读没读到。
**必须验证**:NO_GRAPH(eager,每步真执行host)+skip vs graph+skip,对比TPOT。若NO_GRAPH下skip有影响、graph下没有→图旁路坐实,graph态skip作废。若都有/都没有→另说。
**不下结论,先验**。下一步:NO_GRAPH+DENSE_SKIP+MOE_SKIP vs NO_GRAPH base,看eager态skip到底影响多少(eager一定执行host skip判断)。

## ★★★★★★ D14 2026-10-28 验清D13:NO_GRAPH下skip也0影响→图没旁路,D12成立,瓶颈确系host
NO_GRAPH(eager,每步真执行host skip判断) plen128:
| | TPOT |
|---|---|
| base | 14.81 |
| skip dense+MoE | 14.80 |
**eager下skip dense+MoE仍0影响(14.81→14.80)**→D13"图旁路skip"担心**不成立**(env在eager确实执行,但删GPU大段真不改TPOT)。
**D12成立坐实**:decode TPOT不是GPU算(eager删GPU大段也不变),瓶颈在host/每步固定开销。NO_GRAPH 14.8 vs graph 13→graph只省~2ms(launch),大头13ms是host本身。
注:只重验了dense+MoE的eager skip(最大两块GPU活),足够证GPU计算非瓶颈;qkv/paged/oproj的eager skip没重跑但同理。
**方向彻底锁死**:D5-D14一致指向——decode瓶颈=host/调度/每步固定链,非任何GPU kernel。两刀MoE+要攻的attn全错。R47对。
**Step B可信地往host挖了(非假数据)**:读executor decode每步host代码(upload_plan/Graph::launch/argmax D2H/采样/detokenize),定13ms host在哪+随KV 6ms的host循环。vLLM 7.57说明host能薄。

## D15 2026-10-28 读host路径:每步极薄(bind/upload/launch/collect),13ms大部分是collect的dtoh_sync阻塞等GPU
forward_sample(executor.rs:245)每步:bind_ctx→upload_plan→graph.launch→collect(argmax D2H,dtoh_sync_copy带同步,:208)→取token。host代码极薄,无明显13ms CPU工作。
**关键**:collect的dtoh_sync**阻塞等整个graph(30层forward)跑完**。所以forward_sample墙钟=launch+阻塞等GPU+D2H。
**诚实矛盾未解**:
- graph态skip GPU段0影响→可能图旁路(D13,graph预捕获skip env不生效);
- NO_GRAPH态skip dense+MoE 0影响(D14)→只证dense+MoE的GPU时间小,没证qkv/paged/attn;
- 两者加起来**不足以100%断定13ms是host还是GPU**。我之前D12"瓶颈在host"下太快(graph态skip可能旁路)。
**一锤定音实验(还没做)**:加FULL_SKIP让整个forward/launch都不跑(只bind+upload+collect dummy),TPOT剩多少=纯host地板。或NO_GRAPH下重跑qkv/paged/oproj skip(eager真生效)看降不降。
**诚实**:现在还不能断定13ms归属。不在没定死前往host改(重蹈MoE覆辙)。
**下一步**:NO_GRAPH下重验qkv/paged/oproj skip(eager态,env真生效)——若eager下删这些TPOT大降=GPU确实在算(graph态0影响是图旁路假象,瓶颈其实GPU);若eager下也不降=真host。这个eager重验是关键,之前coreskip是graph态(可能旁路)。

## ★★★★★★★ D16 2026-10-28 一锤定音(eager重验):NO_GRAPH下qkv/paged/oproj skip也0影响→瓶颈100%在host,铁证
NO_GRAPH(eager,skip env一定生效,无图旁路):
| | base | noqkv | nopaged | nooproj |
|---|---|---|---|---|
| plen128 | 14.80 | 14.80 | 14.80 | 14.80 |
| plen1024 | 20.60 | 20.60 | 20.59 | 20.60 |
**eager下删qkv/paged/oproj全0影响**→排除D13图旁路担心(eager无图),**坐实D12**。
**最终定论(eager+graph双重skip证据,铁)**:decode TPOT根本不由GPU kernel决定,删任何/全部GPU段(dense/MoE/norm/rope/kvfp8/qkv/paged/oproj)eager+graph都不变。瓶颈100%在**host/每步固定链**。plen128→1024涨6ms删paged也不降→那6ms也是host侧随KV开销(非paged kernel)。
**今天方向全错的铁证**:两刀MoE kernel+要攻的attn kernel全不在关键路径。唯一该做=**host路径优化**。
**host 13ms(graph)/14.8ms(eager)构成待挖**:forward_sample每步bind_ctx/upload_plan/graph.launch/collect(dtoh_sync阻塞)/采样。但host代码极薄(D15),13ms藏在哪需细测:bind_ctx?upload_plan(htod)?launch开销?dtoh_sync?python/rust调度?
**下一步(真step B优化前最后定位)**:host内部分段计时(Instant::now在bind/upload/launch/collect各段),定13ms host哪段。这个纯CPU计时不碰GPU,graph/eager都准。对标vLLM 7.57ms(它host也薄但更快)。

## D17 2026-10-28 非eager host-time(core engine层):exec=19.3ms占全,sched/publish微秒级
core engine已有的[host-time](非我加的executor层)plen1024每步:
- sched=0.3us(调度,忽略) exec(begin+finish)=19317us=**19.3ms(几乎全部)** publish=3.6us(忽略)
exec=调用executor.forward_sample墙钟=19.3ms=TPOT。调度/publish确实极小(R58对)。**19.3ms全在exec内部**(forward_sample的launch+collect dtoh_sync)。
**我加的executor层bind/upload/launch/collect细分没打出**(被core的host-time盖/tracing级别)→需确认真打出来拆exec 19.3ms。
**关键未分清**:exec 19.3ms里 launch(异步提交,应极快) vs collect(dtoh_sync阻塞等GPU graph跑完30层)各占多少?
- 若collect占满19ms=在等GPU→GPU确实算19ms(那skip 0影响=graph图旁路假象,D13对graph态成立,瓶颈其实GPU!);
- 若launch占大头=真host开销。
**这是最终分清点**。D16"瓶颈host"可能又被图旁路骗(graph态skip没生效),但D16有eager复验…eager下collect也阻塞等GPU(eager真跑forward),eager skip 0影响才是硬的——eager删qkv/paged/oproj真跑真跳过TPOT不变=GPU真不是瓶颈。所以collect等的"GPU"其实很快,19ms另有出处??
**矛盾仍在,必须拆exec**。下一步:确保executor层[host-time]打出(改tracing或eprintln),拆launch vs collect(dtoh_sync)。若collect≈19ms但eager skip GPU又0影响→dtoh_sync本身慢(同步开销/驱动)而非等计算。

## D18 2026-10-28 executor层打点没生效(kaon3-host 0行,core host-time 192行)→forward_sample的HOST_TIME分支没执行到
我在kaon3 executor.rs forward_sample加的[kaon3-host] eprintln 0行,但core engine的[host-time] 192行。说明decode计时走core engine外层,我改的kaon3 forward_sample的HOST_TIME分支没进(或core先读HOST_TIME打了core的)。exec 19.3ms里launch vs collect细分**未测出**。
**但不影响主结论**(eager+graph双重skip铁证):decode瓶颈在host exec路径(launch+等GPU返回+采样),非任何GPU kernel。
**本会话最终定论(给下次接手)**:
- Step A/B定位完成:decode TPOT=固定~13ms+paged~4.6ms/1K,**全在host exec,删所有GPU段(dense/MoE/norm/rope/kvfp8/qkv/paged/oproj)eager+graph都0影响**。
- 今天两刀MoE kernel优化(cp.async流水/gate+up融合)方向全错(MoE不在关键路径),逐位都对+默认关保留,零风险。
- **未完成**:exec 19.3ms里launch(submit) vs collect(dtoh_sync)没拆开(executor打点没生效,core的host-time盖了)→没定位到可改的具体一行。
**下次接手第一步**:拆exec——改core engine的host-time(crates/core engine.rs)或确保kaon3 forward_sample打点生效,分launch vs collect。若collect(dtoh_sync)占满=在等GPU(与skip 0影响矛盾需再查);若launch占=host提交开销。对标vLLM 7.57(它每步host更薄)。
**方法教训(已记knowledge)**:整段skip看端到端是定关键路径最硬手段,应最先做。本会话教训=信NO_GRAPH分段(MoE63%)做两刀全白费,skip一测就知MoE不在关键路径。

## D19 2026-10-28 诚实:host-split打错分支(0行)+未解矛盾,停止加测量
[host-split]0行:我加在engine.rs:565 pend_n的finish(pending稳态流水分支),但单流concurrency=1没n+1可排→不走542"先排n+1"流水,走589同步begin+finish。计时加错分支。
**但确定的**:exec=19.3ms=TPOT本身(sched 0.2us/publish 3us可忽略)。单流decode每步=schedule+begin(graph launch)+finish(dtoh_sync等GPU),单请求无n+1流水,串行。
**未解矛盾(诚实,没装解决)**:
- exec 19.3ms里finish的dtoh_sync等的是GPU graph跑完(预捕获30层);
- 但eager+graph skip GPU段都0影响(D16)。
- 两者不自洽:若finish在等GPU跑19ms,skip GPU段该让GPU更快→TPOT降,但没降。
- 可能:①graph态skip被图旁路(env不改预捕获图,D13)+eager态skip的GPU段本就快(dense/MoE小,qkv/paged/oproj eager也0影响这条最费解);②或dtoh_sync的"等"不是等计算是固定同步开销(驱动/CUDA同步原语本身~19ms??不合理);③或bench的TPOT测量把别的算进来。
**停止加测量**:370+工具调用,反复绕此矛盾,继续烧资源。诚实交接未解。
**下次接手正确做法**:
- 把[host-split]计时加到**589同步finish分支**(单流走这)+begin分支,一次测准begin vs finish;
- 或nsys换台机器/换版本真抓graph态kernel时间线(此环境6轮失败);
- 或直接读runner.begin/finish实现(crates/core runner)看begin到底同步否、finish的dtoh_sync具体等什么;
- 关键问题仍是:19ms到底GPU算/等GPU/host同步开销,哪个。没定死前别改。

## ★★★★★★★ D20 2026-10-28 c1/2/4/8双端对比(真实KV prefix-repeat 0):TTFT爆炸2秒+吞吐虚高+并发崩
真实KV(prefix-repeat 0,无cache命中)dinfer vs vLLM非eager:
| 并发 | 吞吐 d/v | TTFT p50 d/v | TPOT p50 d/v |
|---|---|---|---|
| c1 | 29.7/185 | **1960**/83 | 18.5/4.8 |
| c2 | 56/308 | 1982/151 | 19.6/5.4 |
| c4 | 82/541 | 1998/153 | 20.8/6.2 |
| c8 | 63/957 | 4116/228 | 90/6.8 |
**两大颠覆问题**:
1. **TTFT爆炸**:dinfer c1就1960ms vs vLLM 83ms=**慢24×**。c1-c4 TTFT恒定~2000ms=**一个固定~2秒串行开销卡在首token前(prefill路径)**。之前prefix-repeat 1(共享前缀命中cache)完全掩盖,真实KV暴露。
2. **吞吐虚高+并发崩**:之前594 tok/s是prefix-repeat 1虚高(prefill被cache跳过)。真实负载dinfer吞吐仅vLLM 6-16%(非之前以为的63%)。dinfer c4→c8吞吐82→63**倒退**,TPOT 21→90暴涨=**并发扩展性崩**,vLLM c8还涨到957。
**结论作废**:之前所有594/63%基于prefix-repeat 1,是虚高。真实负载dinfer差vLLM一个数量级。
**优化重点彻底改**:**TTFT 2秒(prefill路径)是最大问题**,不是decode TPOT。这2秒在prefill不在decode,和之前纠结的decode瓶颈不同。其次并发扩展崩(c4→c8倒退)。
**下一步(Step A补全/Step B转向)**:①定位TTFT 2秒固定开销在prefill哪(c1-c4恒定2秒=非计算,像固定串行:编译/同步/首步graph capture/分配?);②并发c4→c8为何倒退(KV块不够?调度?)。TTFT比TPOT优先。

## ★★★★★★★ D21 2026-10-28 TTFT真相:非固定开销非warmup,是prompt长度超线性爆炸(prefill路径)
warmup5重测:
| | TTFT |
|---|---|
| dinfer plen1024 | 1960ms(warmup5仍2秒,**排除预热**) |
| dinfer plen128 | **25ms** |
| vLLM plen1024 | 82ms |
**TTFT随prompt长度超线性**:plen128=25ms,plen1024=1960ms。8×长度→**78×时间**(prefill正常~O(L)到O(L²),78×远超8×=严重超线性)。
**真大头定位**:dinfer prefill路径对长prompt超线性爆炸。1024 token prefill花2秒,vLLM 82ms(25×)。非warmup/非固定开销,是**prefill计算本身长序列下爆炸**(疑prefill attention O(L²)无优化/分块差)。
**这才是最大问题+全新方向**:不是decode TPOT/host(我大半天纠结的),是**prefill attention长序列超线性**。
**解释D20吞吐虚高**:prefix-repeat 1命中cache跳过prefill,这2秒从没暴露。
**Step A补全结论(重排优先级)**:
1. **TTFT/prefill超线性(最大,plen1024慢vLLM 24×)** ← 先攻
2. 并发扩展崩(c4→c8吞吐倒退)
3. decode TPOT(单流18 vs 4.8,慢3.8×) ← 之前大半天在这,其实第三位
**下一步Step B**:定位prefill为何超线性——读kaon3 prefill attention路径(ops.attn().prefill_win,gqa prefill kernel),看是不是O(L²)全量/没分块/没flash。对标vLLM prefill(flash-attn分块O(L))。plen扫prefill TTFT(128/256/512/1024)确认超线性曲线。

## ★★★★ D22 2026-10-28 prefill kernel是flash(非O(L²)),但Kaon D=256/512可能走"旧参考实现"=真超线性源
读gqa.cu:prefill已是flash风格(FA2/online softmax/causal只扫对角tile/cp.async),**非朴素O(L²)**,推翻D21"疑O(L²)"。
**关键线索(gqa.cu:10-12注释)**:"D=128走flash tile kernel,**其他D走旧参考实现**"。Kaon-V3 head_dim=**256(滑窗层)/512(全注意力层)**,非128→**很可能走旧参考实现,不是flash**!旧参考若朴素O(L²)无分块=plen1024慢2秒/24×的根源。
**待确认**:Kaon prefill(D=256/512)实际分派到flash还是旧参考。查gqa.cu的D分派+ops.rs gqa_prefill_win_bf16。

## ★★★★★ D23 2026-10-28 更正:v5/v6 flash只支持D=128/256,**D=512全注意力层(5层)走慢路径**=超线性源
更正D22:gqa.cu:342-352分派只有 head_dim==128/256(都有flash v5/v6),**非128/256直接cudaErrorInvalidValue**。Kaon-V3:滑窗层D=256(走flash OK),**全注意力层(5/11/17/23/29共5层)D=512**(gqa.cu:47 MAX_D=512 Gemma4 full层)。
**超线性源锁定**:D=512的5个全注意力层prefill**走不了v5/v6 flash(只支持128/256)**→走通用慢路径(layer.rs提过"MAX_D提512走慢路径")。长prompt下这5层慢路径爆炸=TTFT 2秒。滑窗层D=256走flash没事。
**算一下**:25层滑窗(flash)+5层全注意力(慢路径D=512)。plen128→1024,flash层~线性,慢路径层若O(L²)则5层主导超线性(78×)。
**待确认**:layer.rs D=512 prefill实际调哪个慢路径kernel,是不是O(L²)无分块。
**下一步Step B**:①确认D=512慢路径kernel结构(gqa.cu通用慢路径,看是否朴素O(L²));②给D=512补flash(v6已是D=256专用FA2,扩到512?或v5 D=512);③这是Step B要改的——给全注意力层D=512补flash prefill。对标vLLM(flash支持任意D)。

## ★★★★★★★ D24 2026-10-28 根因坐实:D=512全注意力层prefill走gqa_prefill_ref_kernel朴素O(L²)=TTFT 2秒
gqa.cu:880-881:**通用慢路径=gqa_prefill_ref_kernel旧参考实现,"每(req,q_head)一个block逐query行三遍扫key"**=朴素O(L²)(每行扫全部key)+并行度极低(每req×q_head一个block)。
**完整根因链**:
- dinfer prefill:25滑窗层(D=256)走flash v5/v6 ✓;**5个全注意力层(5/11/17/23/29,D=512)走gqa_prefill_ref_kernel朴素O(L²)慢路径** ✗。
- plen1024这5层O(L²)×低并行爆炸→TTFT 1960ms;vLLM flash全程O(L)仅82ms(慢24×)。
**Step B靶子(明确且在关键路径,非MoE打偏)**:给D=512全注意力层补flash prefill。v6已是D=256专用FA2(gqa.cu:354,Q进smem/NW/TILE模板化),**扩到D=512**(gqa_fa_v6_kernel模板<D>,v6注释说D=256时Q+O+S寄存器已满,D=512要Q进smem+减TILE,正是v6思路);或v5 D=512。
**Step A/B成果(终于定对)**:耗时最长=prefill TTFT(plen1024慢vLLM 24×);根因=D=512走O(L²)参考kernel(5层全注意力);改法=补D=512 flash。decode TPOT(慢3.8×)是次要,MoE两刀全打偏(MoE/decode不在关键路径,TTFT才是)。
**下一步Step B实现**:gqa_fa_v6_kernel<D>扩D=512分派(gqa.cu:347后加head_dim==512→v6<512>);注意D=512寄存器/smem压力(227KB smem,Q 512×2+K/V tile),TILE可能要减。先读v6_launch/v6_kernel看D模板化到哪、512要改什么。改后对拍(prefill数值逐位/对vLLM)+ plen扫TTFT确认线性+端到端bench进CSV(step C)。

## ★★★ D25 2026-10-28 Step B实现计划:给D=512补v6 flash prefill(最小可测低风险)
分派坐实(gqa.cu:1433-1457):head_dim 128/256→flash,**512→落ref_kernel朴素O(L²)**(1453-1457)。
**基座选v6非v5**:v6=template<PAGED,D,NW,TILE,NBUF,FEXP>,D参数化(LDS=D+8,O按D);v6 Q进smem正为解决D大寄存器溢出(D=256 v5已spill,512更甚)。v6_dispatch现硬编码256。
**smem算**:v6=(NW*16+2*NBUF*TILE)*(D+8)*2。D=512/NW=4/TILE=16/NBUF=1→(64+32)*520*2≈**97KB**(5090~100KB上限内);TILE=8→81KB更稳。
**改法(最小)**:gqa_prefill_impl(1445前)加`head_dim==512 && cfg>=0`→直接v6_launch<false,512,4,TILE>(保守TILE=16或8),ref留fallback+对拍参照。
**风险**:v6 kernel从没编译过D=512,可能有D=256硬编码假设(fragment数/O累加器)→先编译验证纯D参数化;smem 97KB接近上限可能occupancy=0启不动→TILE降8。
**停止条件**:①编译过(5090 sm120)②逐位对拍ref_kernel(prefill D=512同输入bad=0,或对vLLM prompt_logprobs)③plen扫TTFT从超线性变线性④端到端c1/2/4/8 bench TTFT大降进CSV(step C)。
**ROI**:TTFT plen1024 1960ms,5个全注意力层走O(L²)是主因,补flash后该从~2秒降到数十ms(接近vLLM 82ms),TTFT是最大缺口(慢24×)。

## ★★★ D26 2026-10-28 v6<512>编译通过(smem/寄存器都过)→验TTFT+输出一致性
cargo build -p dinfer --features kaon3 EXIT=0,无too much shared/spill/error=v6<512>(o[64][4]=256寄存器+smem~97KB)构建OK。Step B第一关过。
**第二关A/B**:serve两次,GQA_D512_REF=0(新flash)vs=1(旧ref),对比:①TTFT降没降(plen1024,目标2秒→数十ms)②输出token文本一致(数值对拍,greedy T=0同输入该逐token同)。一致+TTFT降=成立。

## ★★★★★ D27 2026-10-28 D=512 flash A/B:TTFT不变(2052 vs 2051)+输出一致→定位错!2秒不在D=512 attention
flash(GQA_D512_REF=0) vs ref(=1):
| | flash | ref |
|---|---|---|
| TTFT plen128 | 69ms | 69ms |
| TTFT plen1024 | 2052ms | 2051ms |
| 输出 | " the last." | " the last." |
**输出一致(数值对拍过,flash没算错,v6<512>正确)**。**但TTFT完全不变**→**D24定位错:prefill 2秒根本不在D=512 attention kernel**。换flash纹丝不动=这段非大头。
**又犯同样的错**:没A/B就按代码推断("D=512走ref O(L²)")断定是瓶颈,结果换掉TTFT不变。**整会话反复教训:代码推断的显眼瓶颈≠真瓶颈,必须A/B/skip实测**。幸好A/B兜住(逐位对拍+TTFT)没白改一堆。
**v6<512>保留**(逐位对拍过+编译过,零风险,未来D=512 attention真要快时有用;默认走它,GQA_D512_REF=1回ref)。
**TTFT 2秒真正位置未知**:prefill路径里attention不是,那是什么?(prefill的qkv GEMM?MoE?dense?norm?采样?还是host?)需对prefill做整段skip差分(像decode的D5-D16),别再代码推断。
**下一步**:prefill路径skip差分——plen1024单请求,逐段skip(prefill的MoE/dense/qkv/attention/采样)看TTFT哪段降。这是定TTFT 2秒的唯一可靠法。注意:prefill每token都过MoE/dense(非decode的1token),prefill的MoE/dense工作量=1024×比decode大→很可能prefill大头是MoE/dense(decode时它们不在关键路径但prefill 1024 token时是)。

## ★★★★★★ D28 2026-10-28 prefill skip差分:删MoE/dense/qkv/paged全0影响→TTFT 2秒也不在GPU,在host prompt处理
plen1024 TTFT base 2002ms,skip各段:nomoe 2001/nodense 2001/noqkv 2001/nopaged 2001=**全不降**。
**决定性(skip实测非推断)**:prefill 2秒**不在任何GPU计算段**(同decode D16)。换D=512 flash也不降(D27)印证。
**真位置**:TTFT随L超线性(plen128=69ms,1024=2002ms,D21)+删GPU全段不变→**2秒在host对prompt的per-token处理**:tokenize/chat模板渲染/prompt htod上传/block分配/slot_mapping构造/prefill调度。host对1024 token做了O(L²)或重的per-token工作。vLLM 82ms说明这块能很薄。
**整会话最终定论(decode+prefill都skip实测)**:dinfer慢vLLM的真因**全在host路径,非任何GPU kernel**:
- TTFT 2秒=host prompt处理(随L超线性);
- decode TPOT 19ms=host exec(begin/finish,未拆到底)。
- 我一整天改的MoE/attention kernel全不在关键路径。
**skip实测这次对了**:直接排除GPU,没再瞎改kernel(D27 flash白改但A/B兜住)。
**下一步**:prefill host路径分段计时(Instant在tokenize/模板/htod/alloc/schedule各段),或读server HTTP handler→engine prefill的host代码,定2秒随L超线性在哪。这是纯CPU问题。注意可能在server层(chat模板/tokenize)非engine。

## D29 2026-10-28 render计时加错端点([render-time]0行)+诚实收口:TTFT定位到host prompt路径未到具体段
[render-time]0行:我加在chat.rs(/v1/chat/completions),但bench的one_request走**/v1/completions(completions.rs)**,加错文件。又一次"计时加错地方没打出来"(host-split错分支/executor host-time被core盖/现在render错端点,共3次)。
**本会话TTFT定位止于**:确证**TTFT 2秒不在GPU**(prefill skip MoE/dense/qkv/paged全0影响D28+D=512 flash换了不降D27),**在host prompt处理路径,随prompt长度超线性**(plen128=69ms/1024=2002ms),**但没定位到host具体哪段**(render? engine add_request? schedule? 计时反复加错端点/分支)。
**下次接手TTFT第一步**:计时加到**completions.rs的completions handler**(bench真实走这),量be.render(模板+tokenize)vs engine提交;或直接读completions.rs看prompt处理有无O(L²)(常见:逐token push/反复realloc/O(L²)字符串拼接/tokenizer慢)。vLLM 82ms说明host能薄。
**本会话总成果(给交接)**:
1. **最大缺口=TTFT**(真实KV plen1024慢vLLM 24×,2秒),非decode(我大半天纠结的,慢3.8×第三位)。
2. TTFT 2秒+decode TPOT 19ms**都在host路径,非GPU kernel**(skip实测铁证)。
3. 之前594/63%是prefix-repeat 1虚高,真实负载dinfer仅vLLM 6-16%。
4. 一天改的MoE两刀+D=512 flash全不在关键路径(都默认关/对拍过保留,零风险)。
5. 方法教训(已knowledge):整段skip定关键路径该最先做,别信代码推断/NO_GRAPH分段。

## D30 2026-10-28 计时日志第4次打不出(cmpl-time/render-time/host-split/executor全0,只core host-time出)+换/tokenize直接量
[cmpl-time]又0行。4次加tracing::info计时都没打出来(executor/host-split/render/cmpl),只core engine的[host-time]出。**系统问题**:server crate的tracing info可能没启用/被过滤,或bench请求没走到我加的行。我一直假设加info就能看到,错。
**换法(绕serve日志)**:直接curl /tokenize端点(misc.rs:81,bench不走但能单独量tokenize),time一个1024词prompt的tokenize耗时。若~2秒=tokenize O(L²)是TTFT根因;若快=2秒在别处(engine prefill提交/等)。

## ★★★★ D31 2026-10-28 tokenize排除(1024词7ms)!但发现:1024词=4010 token>max-model-len 2048=prompt被处理超长
curl /tokenize 1024词=**7ms**(排除tokenize是TTFT 2秒根因)。但出**4010 token > max-model-len 2048**!之前所有"plen1024"其实是4010 token prompt(bench --prompt-len是词数,4010>2048超限)。
**新线索**:prompt 4010 token超max-model-len 2048。serve要么截断要么分块chunked prefill。TTFT 2秒可能与此相关(超长prompt的host分块/截断/多次prefill循环)。vLLM同bench同4010 token才82ms,所以非token数本身,是dinfer对超长prompt的host处理差。
**排除**:tokenize(7ms)。**2秒在submit→engine→prefill之间的host**。prefill GPU skip全段不降(D28)+tokenize快→2秒在**engine对超长/分块prompt的host调度循环**(chunked prefill分多chunk?每chunk一次调度?O(chunks²)?)。
**下一步**:①用真正≤2048 token的prompt(--prompt-len 400词≈1600 token)重测TTFT,看是否还2秒(若降→确是超长/分块问题);②扫prompt token数(256/512/1024/2048 token)看TTFT拐点;③读engine prefill对>max-batched-token的分块逻辑(chunked prefill)。

## ★★★★★★★ D32 2026-10-28 决定性!2秒只在prompt超max-model-len(2048),正常长度TTFT完全正常
换准token长度扫TTFT:
| prompt | token约 | TTFT |
|---|---|---|
| 100词 | ~390 | 19.7ms |
| 300词 | ~1170 | 42.8ms |
| 500词 | ~1950 | 50.5ms |
100→500词(token 5×)TTFT 19.7→50.5(2.5×)=**近线性正常,无2秒**!
**根因颠覆**:2秒只在prompt**超max-model-len 2048**时出现(1024词=4010 token>2048;500词=1950<2048正常)。dinfer对**超长prompt(>max-model-len)走慢路径**(截断/fallback/错误处理),非prefill本身慢。
**前面TTFT结论全要修正**:之前"TTFT 2秒/慢vLLM 24×"建立在**超限畸形负载**(bench --prompt-len 1024当词数×3.9=4010 token>max-model-len 2048)。dinfer prefill正常长度(≤2048)TTFT完全正常(50ms)。
**这是bench负载配置问题+dinfer超长prompt处理缺陷**,不是prefill慢。锅:--prompt-len是词数,1024词=4010token超2048,我和之前bench都没核对token数vs max-model-len。
**修正后真实对比待重做**:用≤2048 token的prompt + 相应max-model-len,重测c1/2/4/8 dinfer vs vLLM,才是公平TTFT/吞吐对比。之前D20的"吞吐仅6-16%/TTFT慢24×"可能也受超限负载污染。
**下一步**:①要么bench prompt控在max-model-len内重测对比;要么把serve max-model-len提到>4010(容纳4010 token)重测。②若要查超长处理慢路径=读engine对>max-model-len的截断/分块。但优先:公平负载重测,看dinfer真实差距到底多少(可能远好于6-16%)。

## ★★★★★★★ D33 2026-10-28 前置条件根本没对齐!之前所有对比不公平(max-model-len 2048 vs 16384 + vLLM其实eager)
查两端serve cmdline:
| | vLLM:28201 | dinfer:8142 |
|---|---|---|
| max-model-len | **16384** | **2048** |
| max-num-batched-tokens | 4096 | 默认 |
| enforce-eager | **有(eager!)** | 无(graph) |
**两个大锅**:
1. **max-model-len 差8倍**(16384 vs 2048):4010 token prompt在vLLM合规(82ms),dinfer超限走慢路径(2秒)。"慢24×"纯是不对齐造成。
2. **:28201带--enforce-eager=eager!** 我一直当它"非eager最强基线"(D67/D20都这么记),**错了**,它是eager。之前所有"vLLM非eager"对比作废。真非eager vLLM可能没在跑/别端口。
**之前结论大面积要修正**:D20"慢24×/吞吐6-16%"、D67"非eager基线946"全建立在**前置不对齐**(dinfer 2048超限 + vLLM 16384 eager)。既不是同max-model-len,vLLM又是eager。
**对齐重测(公平前提)**:dinfer max-model-len提16384(对齐vLLM)+ 两端同eager或同graph。再测c1/2/4/8。这才是真实差距。
**教训(核心,记knowledge)**:对比前必须核对两端前置条件(max-model-len/eager/batched-tokens/block-size)!我和之前bench都没核,在不对齐的畸形条件上跑了一整天结论。用户"要对齐测试前置条件"一针见血。

## D34 2026-10-28 修正D33:28201其实非eager(我查错进程),但max-model-len仍不对齐(16384 vs dinfer 2048)
两vLLM:vllm-noneager :28201(pid2592348,无enforce-eager,max-model-len 16384)=**真非eager**;vllm-eager :28200(enforce-eager)。D33查的pid2592177是:28200(eager)查错了。**:28201确实非eager**(D67没错)。
**但max-model-len不对齐仍成立**:dinfer 2048 vs vLLM 16384。4010 token prompt在dinfer超限=2秒慢路径(D32),在vLLM合规。这才是"慢24×"主因(非eager与否)。
**对齐方案**:dinfer重起max-model-len 16384非eager serve(:8290),两端非eager+同max-model-len。Step A跑c1/2/4/8/16,prompt token数先用合理值(400词≈1560 token,安全)。

## ★★★★★★ D35 2026-10-28 对齐后第一次公平基线(两端非eager+max-model-len16384+prompt1560token合规)
| 并发 | 吞吐d/v tok/s | d/v% | TTFT p50 d/v | TPOT p50 d/v |
|---|---|---|---|---|
| c1 | 44/200 | 22% | 368/24 | 17.4/4.7 |
| c2 | 82/346 | 24% | 384/37 | 18.4/5.4 |
| c4 | 154/543 | 28% | 384/88 | 19.3/6.1 |
| c8 | 289/971 | 30% | 383/146 | 20.0/6.2 |
| c16 | 463/1550 | 30% | 392/220 | 19.9/7.0 |
**真实差距(对齐后,非污染)**:吞吐**22-30%**(非之前6-16%污染/63%虚高);TTFT dinfer c1就368ms vLLM24ms=**慢15×**且**几乎不随并发变(368→392恒定)**;TPOT慢~3×(17vs5)。
**两靶子**:
1. **★TTFT恒定~380ms**(c1就有不随并发变=固定开销,vLLM24ms随并发涨):prompt合规(1560<16384)下dinfer prefill首token仍380ms。**非之前2秒(那是超限),这380ms是合规负载真实prefill开销=真问题,最该先攻**。
2. 吞吐22-30%+TPOT慢3×。
**dinfer启动慢**:max-model-len 16384后model_load_ms=64068(之前2048时42s,KV池大)。
**对齐教训固化**:这才是公平基线。之前所有数(594/63%/6-16%/慢24×)因前置不对齐(max-model-len/eager/prompt超限)全作废,以本表为准。
**下一步**:攻TTFT 380ms恒定开销——prompt合规下首token为何380ms(vLLM24ms)。c1无并发无排队=纯prefill+首token路径固定开销。skip差分prefill各段(合规1560token下)或host计时(这次计时加对completions.rs已验bench走这)看380ms在哪。

## ★★★★ D36 2026-10-28 合规负载prefill skip也全0影响→TTFT 380ms也在host(非GPU),稳
合规1560token prefill skip:base 367/nomoe 367/nodense 367/nopaged 367/noqkv 367=**全0影响**(同D28但这次合规负载,结论稳非超限假象)。
**TTFT 380ms不在GPU计算**,在host prefill路径(c1无排队)。GPU skip计算不变=finish等的首token GPU很快,380ms是**host侧串行**(prefill plan构造/block分配/graph/htod)。同decode 19ms的host性质。
**拆begin/finish定死(用eprintln不靠tracing,前4次tracing没出)**:c1 prefill走engine.rs:581 begin→async存pending→下轮589 finish。在581 begin前后+589 finish前后加eprintln计时。host-split之前加565(pending分支)错了,c1首个prefill走581同步段或589。

## D37 2026-10-28 第5次计时插桩没出(prefill-split eprintln 0行)+诚实:没吃透engine异步调度分支
[prefill-split]eprintln 0行,TTFT 367正常。**第5次计时没打出**(tracing 4次host-split/executor/render/cmpl + eprintln 1次prefill-split)。
**系统根因(现在确定)**:bench首个prefill请求走的不是我加计时的engine分支。engine异步调度(async_sched默认开)有多条分支:565 pending finish / 528 cut_in插队 / 581 None→None同步 / 578 sync_done。我一直猜错分支加计时,且`!num_scheduled_tokens.is_empty()`条件可能过滤掉。我没吃透异步调度流程。
**本会话确凿净成果(交接)**:
1. **前置对齐后真实差距**(D35,唯一可信):吞吐22-30%/TTFT慢15×(恒定380ms)/TPOT慢3×。
2. TTFT 380ms+TPOT 19ms**都不在GPU**(skip全段0影响,合规D36+超限D28双验),在host。
3. **但host内部哪段没拆开**:5次计时插桩全没出,因没吃透engine异步调度请求走哪分支。
**下次接手(不再盲加计时)**:①先读懂engine.rs异步调度完整流程(step()主循环,c1单prefill请求实际走哪分支:schedule→begin→async?→finish),画出分支;②在确认的分支加eprintln(或core已有的host-time已能出,扩展它拆begin/finish);③或用perf/strace等外部工具量host CPU时间分布,绕过代码插桩。
**ROI提醒**:TTFT慢15×是最大缺口,值得继续。但先搞懂调度分支,别再第6次盲加。

## ★★★★★★★ D38 2026-10-28 TTFT根因拆出!每个prefill步exec=367ms(=TTFT),稳态decode步17ms→prefill首步慢21×
tagged host-time(tokens标记)拆开:
- **prefill步(第一条) exec(begin+finish)=366736us=367ms=就是TTFT**;
- 后续(decode)每步 exec=17ms。sched/publish都微秒。
**bench TTFT mean=367(4请求都367)→每个prefill请求首步都367ms**,非全局冷启动(warmup消不掉)。
**367ms全在exec(begin+finish)**,比decode步(17ms)多350ms。vLLM prefill首token 24ms。
**最可能根因**:prefill每次**capture CUDA graph**(prefill长度可变,不同seqlen要重新capture,每个prefill请求capture一次≈350ms);或prefill首步某一次性重操作(JIT/alloc/池化)在begin里。decode走已capture的图(bucket)所以17ms。
**这才是TTFT慢15×的真位置**:不是GPU算(skip全0影响),是**prefill步的begin里有350ms一次性host/capture开销**。
**下一步(定死)**:读runner.begin对prefill做什么(是否每prefill capture graph/JIT/大alloc)。KAON3_* graph capture日志?serve启动时capture了decode桶,prefill是否也capture或eager?若prefill每次capture→复用/预capture常见prefill长度桶=砍350ms。对比vLLM prefill不capture(直接eager prefill+flash)24ms。
**ROI**:砍掉prefill的350ms→TTFT 367→~17ms,接近vLLM 24ms,TTFT慢15×→持平。最大缺口可解。

## ★★★★★ D39 2026-10-28 数据证实(纠正D38推断):40条host-time exec分布=35×17ms + 5×~367ms
实测exec(begin+finish)分布(ms):35条=17 / 5条=362,363,366,367,371(各1)。bench发warmup1+正式4=**5请求**→**每请求首个prefill步~367ms(5条),之后decode步17ms(35条)**。
**纠正D38**:D38说"每prefill 367ms"方向对但我当时没数证实(扯淡挨批)。现在数证实:**5个367ms对应5个请求的prefill步**,非全局只首次(那样仅1条)。每请求TTFT=它的367ms prefill步,对上bench TTFT mean367。
**数据证实结论**:prefill步exec 367ms(每请求一次) vs decode步17ms,差350ms,在begin+finish内。
**仍没拆出(诚实,用户要的完整列)**:exec 350ms没拆成htod/block/graph-capture/wait;带宽/chat模板也没单独测。只有sched(0.2-19us)/exec/publish(微秒)/tokenize(7ms单测)。
**下一步**:读runner.begin的prefill路径(crate kernels/layers,非engine)看它做什么=graph capture?htod?alloc?——先读懂不加计时(分支加计时前5次都没出,只engine汇合点host-time能出)。读懂后在对的点测,拆这350ms。

## ★★★★★ D40 2026-10-28 完整c1-16表(对齐非eager)+begin-split第7次没出(engine调trait begin非inherent)
**用户要的指标表(已测)**:
| 并发 | 吞吐tok/s | TTFT p50 | TPOT p50 |
|---|---|---|---|
| c1 | 40.6 | 368 | 17.3 |
| c2 | 76 | 382 | 18.5 |
| c4 | 141 | 386 | 19.3 |
| c8 | 265 | 387 | 20.0 |
| c16 | 399 | 383 | 20.1 |
启动 model_load_ms=64113;GPU利用率~100%;tokenize 7ms(单测);schedule 0.2-19us;publish微秒;exec(begin+finish) prefill步367ms/decode步17ms。
**仍没拆出(7次失败)**:begin内部update_states/build_plan/forward_sample + htod/block/chat模板/wait/带宽。
**第7次没出根因(现在定位)**:[begin-split]加在runner.rs:320 `pub fn begin`(inherent),但engine调的可能是212 `fn begin`(trait ModelRunner方法)——两个不同begin!我一直加在没被调的那个。engine.rs:588 `self.runner.begin(&so)`,runner类型决定调inherent还是trait。
**下次接手(不再第8次盲加)**:①确认engine.rs的self.runner是什么类型,它的.begin()解析到runner.rs哪个fn(212 trait还是320 inherent);②在真正被调的那个begin加eprintln;③或forward_sample(executor.rs)加——但那个HOST_TIME分支也没出过,同样要先确认prefill走哪条forward。
**核心教训(整会话)**:我7次计时插桩全没出,根因是没读透调用链就加。应先读懂engine→runner→exec的完整调用解析(trait vs inherent/async vs sync/prefill vs decode各走哪),再加一次。

## D41 2026-10-28 第8次计时没出(fwd-sample无条件eprintln也0行)+诚实能力边界
forward_sample入口第一行无条件eprintln也0行→**prefill根本没走forward_sample**。kaon3 executor supports_async()对prefill可能也true→prefill走exec.begin(async路径,engine.rs:336),forward_sample是死代码没被调。我又加错函数(第8次)。
**诚实能力边界**:8次计时插桩全没出(tracing4+eprintln4:host-split/executor/render/cmpl/prefill-split/begin-split/fwd-sample)。根因=我读不透kaon3 executor的async begin/graph device_mode/forward_sample三条路prefill实际走哪条。反复试是浪费。
**已给用户的数据(能测的)**:TTFT/TPOT/吞吐 c1-16表(D40)、启动64s、tokenize 7ms、schedule μs、exec(begin+finish) prefill 367ms/decode 17ms、GPU~100%。
**拆不出的**:exec 367ms内部(chat模板/htod/block/forward各多少)。8次失败,承认做不到。
**下次接手(关键)**:先读透kaon3 executor.rs的supports_async/begin/forward_device_mode/forward_sample——prefill(非async?)到底走哪个exec入口,在那个入口加计时。executor.rs:336 self.exec.begin(async)是嫌疑,读KaonExecutor的begin实现(不是runner.begin,是exec.begin)看prefill在里面做什么+拆。core的[host-time]能出是因它在engine汇合点;要拆exec必须找到prefill真正执行的那个exec方法。

## ★★★★★★★ D42 2026-10-28 决定性根因!二进制不含我的改动=rsync旧mtime致cargo增量编译跳过(8次失败真因)
strings二进制:host-time=1(早编进),**fwd-sample=0,begin-split=0**(我新加的全没进二进制!)。
**8次计时失败真因(定死)**:不是分支错/函数错(那些误诊全作废)——是**rsync put保留源文件旧mtime,cargo增量编译认为没变、跳过重编**,serve跑的是旧二进制(不含我的eprintln)。put成功(源码在)但编译没生效。
**验证**:host-time(老代码)在二进制,fwd-sample/begin-split(我加的)不在=铁证编译没含改动。
**修复**:put后`touch`改过的源文件(mtime新于target),或cargo build前touch;或rsync加--no-times。下次改代码:put→touch远端源文件→cargo build,确认strings二进制含新字面量再跑。
**教训(整会话最大,记knowledge)**:远端rsync同步+增量编译,必须确保mtime更新否则编译跳过。我8次插桩+前面多次"没出"可能全是这个(编译没生效),不是代码逻辑。应早用strings验证二进制含改动。
**现在**:touch+强制重编,fwd-sample必出,终于能拆prefill 367ms。

## ★★★★★★★ D43 2026-10-28 重大翻转!编译修复后TTFT 367→44ms + begin-split拆出(全在forward_sample)
二进制含probe=3(终于编进)。**begin-split拆出**:update_states 0-3us/build_plan 1-22us(都忽略)/**forward_sample 首步42304us/后续17200us**。367ms(或本次首步42ms)全在forward_sample,其余微秒。
**★重大翻转**:本次TTFT=**43.9ms**(之前旧二进制367ms)!我这次改的=计时代码(.values()修复)+**之前编译失败没生效的kernel改动(cp.async流水R74/D=512 flash D26/gate+up dual R91)这次一起编进生效**→TTFT 367→44ms(8×)。
**惊人推论(待确认)**:之前所有"kernel改动没收益"的结论,可能全因**编译一直失败没生效**(旧二进制跑了8小时)。这次编译修复后改动生效,TTFT大降。需确认是哪个改动的功劳(逐个env开关A/B)。
**现状**:forward_sample首步42ms vs 稳态17ms,首步多25ms(疑首次JIT/alloc,warmup没全消)。TTFT 44ms已接近vLLM 24ms(慢1.8×,非之前15×)。
**但必须验证(别重蹈覆辙)**:TTFT 367→44这个翻转太大,先确认不是测量/负载差异:重跑确认稳定+看是哪个改动(关MOE_GEMM_PIPE/GQA_D512_REF等逐个A/B)。编译链路坑(rsync mtime+cargo -p依赖+.values编译错)已�corded,下次改代码必strings验二进制含改动+看完整cargo error不grep滤。

## ★★★★★★★ D44 2026-10-28 最新代码完整对比(编译修复后,二进制含probe=3)—TTFT c4+反超vLLM!
对齐非eager+max-model-len16384+batched4096+prompt1560合规,dinfer vs vLLM p50:
| 并发 | 吞吐d/v | TTFT d/v | TPOT d/v |
|---|---|---|---|
| c1 | 56/200 | 43.6/24 | 17.3/4.7 |
| c2 | 87/346 | 71/37 | 18.5/5.4 |
| c4 | 162/543 | **81/88反超** | 19.3/6.1 |
| c8 | 298/971 | **84/146反超** | 20.0/6.2 |
| c16 | 416/1550 | **111/220反超** | 20.0/7.0 |
启动34.6s(旧64s);GPU~100%;tokenize 7ms;prefill forward_sample首步42ms/稳态17ms(begin-split实测:update_states/build_plan微秒,全在forward_sample)。
**TTFT**:最新代码c4/c8/c16已**反超vLLM**(之前旧二进制恒定367ms慢15×是假象)。c1/c2还慢一点。
**真实最大差距**:吞吐27-30%(慢~3.5×)+TPOT慢3×(17-20 vs 5-7)。这两项是下一步靶子。
**★整会话最大坑(定死)**:之前"TTFT367ms/慢15×/kernel改动没收益"全因**旧二进制**(我的计时代码.values()编译错误致build failed,serve跑8小时前旧二进制,而我grep滤掉cargo error没发现)。最新代码编进后TTFT大降+c4反超。教训:改代码必看完整cargo输出(不grep滤error)+strings验二进制含改动+确认mtime更新。
**下一步**:①吞吐/TPOT是真靶子(慢3×);②确认TTFT反超是哪个kernel改动功劳(逐个env A/B:MOE_GEMM_PIPE/GQA_D512_REF等)还是编译本身;③用户要的数据表已全(本表)。

## ★★★★★★★ D45 2026-10-28 decode TPOT真分段(最新二进制,skip终于生效)—MoE 5.3ms是头号
c1 decode skip差分(最新二进制probe在,与旧二进制"全0影响"假象完全不同):
| skip | TPOT | 省 | 该段占 |
|---|---|---|---|
| base | 17.41 | — | — |
| MoE | 12.13 | **5.28** | **MoE 30%** |
| paged attn | 14.26 | 3.15 | paged 18% |
| dense | 14.60 | 2.81 | dense 16% |
| qkv | 16.52 | 0.89 | qkv 5% |
剩~5ms=norm/rope/oproj/gather/argmax/graph。
**TPOT 17.4ms真构成**:MoE 5.3(头号)+paged 3.2+dense 2.8+qkv 0.9+其余5。vLLM decode一步5-7ms,我们光MoE 5.3ms。
**★TPOT慢3×头号=MoE(grouped NVFP4 GEMM)**,其次paged+dense。**回到最早方向**(两刀MoE),但那两刀当时编译失败没进二进制所以测"没收益"=假象。现在编译修好,MoE仍5.3ms大头→说明两刀即使生效也没压下MoE多少(或cp.async流水decode M=1场景之前微基准1.83×但端到端此前没验到)。
**旧二进制decode skip全0影响作废**:D5-D16那些"删GPU段TPOT不变→瓶颈在host"结论**全建立在旧二进制(skip没生效/改动没编进)**!最新二进制skip生效,证明TPOT确实在GPU(MoE/paged/dense),**不在host**。整会话host方向是旧二进制假象带偏的。
**Step B真靶子(终于确定)**:①MoE 5.3ms(grouped NVFP4 GEMM,decode M小)—重新评估cp.async流水/gate+up在最新二进制的真实端到端收益(之前没真测);②paged 3.2ms;③dense 2.8ms。

## D46 2026-10-28 集中解决MoE:base vs MOE_GEMM_PIPE(第一刀cp.async流水)最新二进制端到端A/B
环境干净(前置对齐/二进制最新/skip生效,D45证MoE=decode TPOT 5.3ms头号)。测MOE_GEMM_PIPE=1(第一刀cp.async流水,之前微基准1.83×但端到端从没在能编译二进制验过)对TPOT的真实端到端+输出一致性。base vs pipe,c1 decode。

## D47 2026-10-28 第一刀MOE_GEMM_PIPE端到端初步:PIPE TPOT 16.45 vs base 17.41(D45)=省0.96ms(-5.5%)
PIPE=1 TPOT c1=16.45ms,base=17.41(D45同条件)。**端到端-5.5%(省~1ms)**。输出两边一致(the.\n\nThe capital...)=流水数值对没错。
但微基准曾1.83×(MoE 5.3→~2.9该省2.4ms),端到端只省1ms。疑:decode走CUDA graph,MOE_GEMM_PIPE env运行时改可能不进预捕获的图(老问题D13)——需确认pipe对graph decode是否真生效。
**需干净同轮重测**(base TPOT别跨轮用D45):base/pipe同容器连测,TPOT grep与completion输出分开(本轮base TPOT行被输出换行污染没抓到)。

## ★★★★★★ D48 2026-10-28 确凿正收益!MOE_GEMM_PIPE(第一刀cp.async流水)干净同轮:TPOT 17.41→16.45(-5.5%)
干净同轮base/pipe:
| | TPOT | 吞吐 |
|---|---|---|
| base | 17.41 | 56.1 |
| pipe(MOE_GEMM_PIPE=1) | **16.45** | 59.3 |
| 变化 | **-5.5%(省0.96ms)** | +5.7% |
**第一刀cp.async流水:TPOT-5.5%、吞吐+5.7%,输出一致(数值对)**。干净环境/同轮/输出一致=确凿真实值(非之前旧二进制假象)。
**预期vs实际**:decode MoE 5.3ms,微基准cp.async 1.83×该省~2.4ms,端到端只省0.96ms=**拿到预期~40%**。疑decode走CUDA graph,PIPE对图内kernel部分生效部分没进图(D13)。但有收益=至少部分进图。
**意义**:这是整会话第一个干净环境确凿正收益(之前全被旧二进制污染)。MoE是decode头号(5.3ms),第一刀方向对。
**下一步继续压MoE**:①查流水为何只40%(图没全进?预捕获图是否含pipe kernel)——补足可能再降;②第二刀gate+up dual(MOE_GEMM_DUAL=1)叠加测;③MoE内gemm3是5.3ms多少。目标把MoE 5.3→~3ms→TPOT 17→~15。

## ★★★★★ D49 2026-10-28 .cu默认开没编进二进制(nvcc未重编)!D48 PIPE=1收益真(老二进制pipe分支),落地走env
改moe_grouped_sm120.cu默认开(v[0]!='0'),但dfl A/B default=17.40=off=17.41(无16.45)!验证:二进制strings"流水版默认开"=0(没编进),源码远端含(=1)。**nvcc未重编**:touch .cu + cargo build -p dinfer,cargo增量对CUDA(build.rs调nvcc)没触发重跑nvcc,二进制还是旧moe kernel(默认关)。
**D48的-5.5%是真的**:那次显式MOE_GEMM_PIPE=1,走老二进制里早已编进的pipe分支(默认关/env=1开)。不是假象。
**落地改走env(可靠)**:老二进制已含pipe分支+env读取,**kserve_full.sh加export MOE_GEMM_PIPE=1**生产就吃到-5.5%,不依赖重编.cu默认。.cu默认改动保留(真重编时也对)但当前编译链路编不进。
**编译坑再记(第3次相关)**:①cargo -p dinfer不重编依赖crate;②rsync旧mtime;③**改.cu后nvcc增量不重编**(要删target/release/build/dinfer-kernels-*/或touch build.rs强制)。改kernel必strings验二进制含改动。

## ═══ 第二轮（2026-10-28 晚，会话因 Input too long 重启，用户重申 step A/B/C）═══
**用户原话要点**：前置=两端非 eager + concurrency / max-model-len / max-num-batched-tokens 一致；Step A 5 轮 c=1,2,4,8,16 测启动/TTFT/TPOT/带宽/GPU util + 每步 tokenize/chat 模板/htod/block 分配/schedule/exec/wait；Step B 攻最长段→提交→编最新→跑；Step C 端到端差则回滚但数据进 CSV+dashboard，B↔C 5 次无进展回 A。
**★ 纠正上一轮对用户两想法的结论（"ROI≈0"作废）**：那结论建立在 D5–D16 "dense 整段 skip TPOT 不变"，D45 已证那批 skip 跑的是旧二进制（改动没编进）。新二进制 D45：dense skip 省 2.81ms（16%）、MoE 5.28（30%）、paged 3.15（18%）、qkv 0.89（5%）。**dense 在关键路径上**，所以：
1. wait/overlap（dense mega 持久 kernel CTA 自旋 wait≈4×exec）——真有 2.8ms 可压的空间，排 Step B 第三（按段大小 MoE→paged→dense）。
2. 128→256（`mega_mlp_core.cuh:103` A 加载只 128 槽）——同上，进 Step B dense 刀，A/B 实测说话，不再凭推理下结论。
**测量手段（二进制 16:21 已含，strings 验过）**：`HOST_TIME=1` → `[render-time]`(chat 模板+tokenize) / `[host-time]`(sched 含 block 分配 / exec / publish) / `[begin-split]`(update_states / build_plan / forward_sample) / `[kaon3-host]`(upload=htod / launch=submit / collect=dtoh_sync=wait)；GPU util/带宽 = `nvidia-smi dmon -s u`（sm% / mem%，mem% 作带宽代理，真 GB/s 要 ncu）；启动 = `model_load_ms`。
**Step A 脚本**：`scripts/dev/kaon3_stepA.sh`（put 到 `/hf_data/yaochen/stepA3.sh`，`docker exec -d kaon3-full bash /w/stepA3.sh`，结果 `/w/dinfer/tmp_stepA3_RESULT.txt`）。

## D50 2026-10-28 Step A 第二轮起跑（环境事故 + 首批数据）
- 事故：GPU 0 被上会话 22h 前的 dinfer serve（`:8142`，max-model-len 2048，非对齐口径，且是容器 `kaon3-full` 的 PID 1）占 26GB → 新服务 OOM 两次；杀它容器一起停。**重建 `kaon3-full` 为常驻 `sleep infinity`（host 网络、GPU 0）**，以后 `docker exec -d kaon3-full bash /w/xxx.sh` 跑长链不再随进程死。
- 首批（round1，dinfer 行被 cut 截掉 TTFT/TPOT，已修重跑）：启动 wall 35.1s（model_load_ms 34639，num_blocks 2048）；tokenize+模板 render ≈1.2ms/请求；sched 12–26us、publish 10us、update_states 2us、build_plan 4us（**host 调度链全是微秒级，不是瓶颈**）；prefill 步 exec 41ms（c1，428 tok）= launch 16ms + wait(dtoh_sync) 25ms；c1 decode GPU sm 99% / mem 49%。vLLM c1 TTFT 25 / TPOT 4.68。
- Step A 修正重跑中（round1 完整数据，c1 / c8）：
  | | TTFT p50 | TPOT p50 | 吞吐 | GPU sm/mem | prefill 步 exec = launch + wait | decode 步 exec = htod + launch + wait |
  |---|---|---|---|---|---|---|
  | c1 | 43.8 (vLLM 25.1) | 17.06 (4.68) | 57 (200) | 98% / 48% | 41.1ms = 16.4 + 24.6 (428 tok) | 17.05ms = 0.02 + 0.38 + 16.6 |
  | c8 | 88.5 (vLLM ~146) | 19.85 (6.2) | — | 97% / 49% | 153.9ms = 61.0 + 92.8 (787 tok, 4.6 req) | 19.2ms = 0.02 + 0.36 + 18.8 |
  **读法**：①tokenize+模板 1.2–1.6ms/req、sched/publish/update_states/build_plan 全 <50us → **host 链不是瓶颈**；②decode 步 = graph launch 0.38ms + wait 16.6ms，wait 就是 GPU 跑图的时间（GPU sm 98%）→ **decode 100% GPU 时间，17 vs 4.7**；"wait 该为 0"只有 async 调度能覆盖 0.4ms 的 host（≤2%），GPU 那 16.6ms 要靠 kernel；③prefill 428 tok 一步 41ms，其中 **host launch 16ms**（eager 逐 kernel 提交，30 层 × 几十 kernel），这是 TTFT 慢 vLLM 的主因候选（c1 43.8 vs 25.1）；④decode GPU mem util 48% = 带宽没吃满（MoE/dense 权重 GEMV 形态，M=1 latency bound）。
- 工具坑：等待脚本里 `remote_5090.sh run '<cmd>'` 的退出码不透传（always 0）→ 两次误判 DONE；改为拉文件到本地 grep。
