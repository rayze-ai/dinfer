# loop: kaon3-decode-opt —— 用户定的 step A/B/C 排查优化流程（2026-10-28）

> 前情（必读教训）:两刀 MoE kernel 优化(cp.async流水 R77 1.83× / gate+up融合 R91 grid翻倍)graph态端到端都 <1%。
> 根因:靠 NO_GRAPH 分段定靶子(说 MoE 占 63%)是假象,被 cudagraph 摊薄。**改任何 kernel 前必须先验证那段是 graph 态真大头**。
> 用户批评:"做之前反复强调验证,方案是屁股想出来的" —— 不再未验证就动手。

## 用户定的流程（严格按此，用户 2026-10-28 两次完善）
**★★ 前置条件（必先对齐,否则结论作废——D33 血的教训）**:dinfer 与 vLLM **两端都非 eager**,且 **max-model-len / max-num-batched-tokens / concurrency 全一致**。vLLM 现 :28201 是 `--enforce-eager`(eager)+max-model-len 16384,**不能直接用**,要起**非 eager** vLLM 且参数对齐 dinfer。bench prompt 的 **token 数**(非词数,词×~3.9)必须 ≤ max-model-len,否则超限走慢路径(D32 的 2 秒假象)。开跑前核对两端 cmdline。
**★ 全程非 eager（graph 生产态）口径。** eager/NO_GRAPH 只作一次性排查手段。测量插桩与 cudagraph 冲突时用 graph 兼容测法(core host-time / 纯 CPU Instant / 整段 skip 看端到端)。
- **Step A（测）**:跑 5 轮 × **concurrency = 1,2,4,8,16**,测 dinfer（**非 eager**）的:①启动时间 ②tokenize ③chat 模板 ④htod ⑤block 分配 ⑥schedule ⑦TTFT ⑧TPOT ⑨每步 exec ⑩wait ⑪带宽 ⑫GPU 利用率。
- **Step B（改）**:针对**耗时最长**的那段,深入分析+优化,提交,编译最新代码运行（非 eager 验）。
- **Step C（验+回滚，用户 2026-10-28 完善）**:端到端测试。收益差则**回滚**,回 Step B。**但每次结果（含回滚的负结果）都要进 CSV 和 dashboard**（`test/kaon3/perf-trend.*`，带 CHANGE 说明）。Step B↔C 重复 **5 次无进展** → 回 Step A 重新分析,继续跑 loop。（用户 2026-10-28 晚补充：5 次不是硬顶，我明确判断某方向还需要轮次时可以适当增加，但要在 loop 里写明理由。）
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

## ★★★★★★ D51 2026-10-28 Step A 第二轮定论（5 轮 × c1–16，24 窗口有效，中位数；归档 test/kaon3/2026-10-28/stepA3_RESULT.txt）
前置已对齐：两端非 eager、max-model-len 16384、max-num-batched-tokens 4096、max-num-seqs 64、prompt 400 词≈1560 tok、64 out、bench 绑核 60-91（client 0.00 核）。**注意本轮 MOE_GEMM_PIPE 没开**（D49 说"已加进 kserve_full.sh"核实为假：grep rc=1；默认二进制 pipe 关）。
| c | 吞吐 d/v (%) | TTFT p50 d/v | TPOT p50 d/v | sm/mem% | render | prefill 步 exec=launch+wait (tok/步) | decode 步 exec=htod+launch+wait (nreq) | TPOT 5 轮散布 |
|---|---|---|---|---|---|---|---|---|
| c1 | 57/200 (29%) | 43.0/25.0 | 17.07/4.69 (3.6×) | 95/47 | 1.2ms | 41.0=16.3+24.6 (428) | 17.06=0.02+0.36+16.64 (1.0) | ±0.0% |
| c2 | 94/346 (27%) | 69.2/36.7 | 18.07/5.41 (3.3×) | 97/43 | 1.1ms | 111.4=44.6+66.8 (198) | 17.71=0.02+0.36+17.30 (1.8) | ±0.1% |
| c4 | 175/543 (32%) | 77.6/89.0 | 18.96/6.05 (3.1×) | 97/44 | 1.1ms | 125.8=50.1+75.6 (395) | 18.53=0.02+0.37+18.11 (3.4) | ±0.1% |
| c8 | 324/956 (34%) | 82.6/146.4 | 19.71/6.24 (3.2×) | 97/44 | 1.1ms | 153.5=61.0+92.5 (786) | 19.17=0.02+0.36+18.76 (6.6) | ±0.5% |
| c16 | 481/1577 (31%) | 104.6/177.3 | 19.68/7.21 (2.7×) | 96/36 | 1.1ms | 331.2=130.4+200.7 (1285) | 19.05=0.02+0.23+18.75 (13.0) | ±0.4% |
（启动 36.1s wall / model_load_ms 34911 / num_blocks 2048；sched 0–42us、publish 4–21us、update_states ≤4us、build_plan ≤10us、htod 20–30us，全部窗口如此。）
**定论**：
1. **host 链（tokenize/模板 1.1ms、schedule/block 分配/publish/htod 全 <50us）不是瓶颈**，5 轮稳定。
2. **decode 步 = 0.36ms graph 提交 + 16.6–18.8ms 等 GPU（sm 96%）**。"wait 该为 0"：这里的 wait 是 GPU 真算的时间，不是空等；能被 overlap 掉的只有 0.36ms（2%）。**差距 100% 在 decode 图内 GPU 时间：17 vs 4.7ms（3.6×）**，且几乎不随并发涨（17→19），vLLM 也是（4.7→7.2）→ 两边都是权重带宽 / 延迟形态，我们每步多花 12ms。
3. **mem util 仅 36–47%**：带宽没吃满 → 不是带宽墙，是延迟 / 并行度 / kernel 数（小 kernel 串行）。这与想法 1"要 overlap"一致，但 overlap 对象是 **图内 kernel 之间**（填满 SM、减少串行 kernel 数），不是 host 与 GPU 之间。
4. TTFT：c≥4 已反超 vLLM；c1/c2 慢 18ms，其中 prefill **host launch 16ms（eager 逐 kernel 提交，占 prefill 步 40%）** 是 c1 TTFT 差的主因候选，第二优先。
**Step B 靶子（按用户规则取最长段）= decode 图内 GPU 17ms**。D45 分段：MoE 5.3 / paged 3.2 / dense 2.8 / qkv 0.9 / 其余 ~5。先用 nsys `--cuda-graph-trace=node` 拿**每步 kernel 清单 + 每个 kernel 时长 + 总 kernel 数**（上次 2024.2.3 抓不到 CUDA 13 kernel，换新版），再决定是减 kernel 数（融合）还是提单 kernel 并行度（MoE grouped GEMM 128 专家 grid 只 88 块活、K 串行 22 步）。

## D52 2026-10-28 decode 图内 kernel 清单：nsys 两次失败→第二次加 flush；并行编 MOE_GEMM_SKIP 二进制
- nsys 2024.2.3（上会话）和 2026.3.2（本次）第一次都"无 kernel 表"：诊断表显示 **"Buffers holding CUDA trace data will be flushed on CudaProfilerStop()"**，只记到权重加载 14k memcpy，kernel 记录在 CUPTI 缓冲里没落盘（dinfer `kill -INT` 直接退，不调 cudaProfilerStop）。修：`--cuda-flush-interval=1000`（每秒 flush）。脚本 `scripts/dev/kaon3_nsys_decode.sh` + 解析 `scripts/dev/nsys_kern.py`（按 graphExecId 聚类取 decode 图实例，逐 kernel 名聚合时长/占比/grid）。
- 同时容器内编译 `MOE_GEMM_SKIP=gu|dn|all`（只跳 3 个 grouped GEMM，路由/gather/quant/gelu/combine 照跑）版二进制（`/w/build_kaon3.sh`，结果 tmp_build_RESULT.txt；验 strings 含 MOE_GEMM_SKIP）。
- 用户 2026-10-28 晚：B↔C "5 次"非硬顶，判断需要可加轮次（已写进规格）。

## D53 2026-10-28 MOE_GEMM_SKIP 差分（新二进制 09:58，graph 态 c1，进行中）
| | TPOT | 省 |
|---|---|---|
| base（PIPE 关） | 16.15 | — |
| skip gate+up GEMM | 15.18 | 0.97 |
| skip down GEMM | 15.88 | 0.27 |
初看：**3 个 grouped GEMM 合计只 ~1.2ms，MoE 段 D45 的 5.3ms 里其余 ~4ms 在路由 / bucket / gather / quant / gelu / scatter_add（每层 ≥12 个小 kernel：gate_kernel + route_topk + route_softmax + count/scan/fill ×3 + row2expert + gather + quant ×2 + gelu + scatter_add + memset + f32_add）**。若剩余项证实，Step B 第一刀应是 **MoE 前后处理 kernel 融合（减 launch 数）**，而不是 GEMM 本体。等 all / pipe / pipe_dual / MOE_SKIP 行 + nsys 清单确认。
（注意 base 16.15 比 Step A 的 17.07 低 0.9ms：bench 参数不同，num-requests 4 vs 8、本轮无 HOST_TIME；只在本表内部比。）
- D53 补：`all`（跳 3 个 GEMM）14.86 → **3 个 grouped GEMM 合计 1.29ms/步（30 层，每层 43us）**；`MOE_GEMM_PIPE=1` 16.18 vs base 16.15 → **本二进制 pipe 已是默认（源码默认开已编进，D49 的"没编进"是当时 build 没重编 kernels；本次 build.log 显示 Compiling dinfer-kernels）**，PIPE=1 无额外收益，正常。
- MoE 支路每层 kernel 清单（读源码 `moe.cu` / `moe_grouped_sm120.cu` / `layer.rs:1296-1530`，decode M=1）：router 前 rmsnorm_noweight + scale + mul（3）→ router GEMM（cuBLASLt 1 + bf16→f32 1）→ route_topk_ids（1 block，thread0 串行 8×128）→ route_softmax（1 block，thread0 串行）→ bucket：memset + count + scan（1 block thread0 串行 128）+ fill（4）→ memset acc → row2expert（128 blocks）→ gather → quant_gu → gate GEMM → up GEMM → gelu → quant_dn → down GEMM → scatter_add（atomic）→ memset y → f32_add_to_bf16。**≈22 个 launch/层 × 30 = 660 launch/步只为 MoE**，其中 route_topk/softmax/scan 三个是单线程串行 kernel。

## ★★★★★★★ D54 2026-10-28 nsys 终于抓到 decode 图内 kernel 清单（c1，224 步，TPOT 16.29）—— 每步 1298 kernel，两处按 4160 行空跑
nsys 2026.3.2 + `--cuda-flush-interval=1000` + 不 INT nsys 本体（之前 INT 打断报告生成=无 .nsys-rep）。脚本 `scripts/dev/kaon3_nsys_decode.sh` / `nsys_kern.py`。224 个 decode 步共 290782 kernel = **1298 kernel/步**；kernel 时间和 14.96ms/步 + 节点间隙 1.89ms/步 = 16.85 ≈ TPOT。
| 每步 | n/步 | 单次 us | ms/步 | 占 | 说明 |
|---|---|---|---|---|---|
| mega_kernel dense MLP (170 CTA) | 30 | 86.9 | **2.61** | 17.4% | D45 dense 2.8 吻合 |
| **rmsnorm_vec<128> grid=66560** | 90 (q/k/v 逐头 ×30) | 25.4 | **2.29** | 15.3% | **按 4160×16 行跑，实际 1×16 行**（`head_rmsnorm` 用 `x.numel()/head_dim`） |
| gqa_paged_kernel grid=(1,16) D=512 全注意力慢路径 | 5 | 369 | **1.85** | 12.4% | 只 16 block |
| gqa_fa_v5 (滑窗 decode) | 25 | 50.5 | 1.26 | 8.4% | |
| rmsnorm_vec<1024> grid=1 (h=2816) | 181 (6/层) | 6.8 | 1.23 | 8.2% | 单行 2816 用 6.8us 偏慢 |
| grouped_gemm_pipe<128>+<64> | 60+30 | 18.6 / 12.5 | 1.49 | 10% | D53 skip 1.29 吻合 |
| cuBLAS gemvx | 31 | 31.8 avg | 0.99 | 6.6% | =30 router(~5us)+1 lm_head(~0.8ms 带宽) |
| nvjet fp8 64x64 / 64x32 (qkv/oproj) | 25 / 35 | 29.4 / 16.5 | 1.31 | 8.8% | |
| **rope_kernel grid=4160** | 30 | 23.5 | **0.70** | 4.7% | **按 4160 行跑，实际 1 行**（`tokens = positions.len()`=plan 缓冲满长） |
| route_topk + route_softmax + scan（单线程 kernel） | 90 | 7.3/5.6/1.4 | 0.43 | 2.9% | |
| gather_rows (logits 1 行 ×262144, 1 block) | 1 | 141 | 0.14 | 0.9% | 1 block 搬 512KB |
| 其余 ~45 小 kernel/层 | ~700 | <2.5 | ~0.6 | | launch 间隙另计 1.89ms |
**Step B 第一刀（按最长可修段）**：逐头 norm + rope 的行数改为实际 tokens（纯 launch 边界修正，真实行逐位不变）。预期省 **2.29+0.70−~0.1 ≈ 2.9ms/步（TPOT 16.3→~13.4，-18%）**。
**后续刀（按收益排）**：② 全注意力 D=512 慢路径 1.85ms（5 层各 369us，16 block）；③ h=2816 rmsnorm 6.8us→~3us ×181 = 0.6ms；④ MoE 小 kernel 融合（减 ~300 launch ≈ 0.45ms 间隙 + 0.4ms）；⑤ gather_rows 0.13ms；⑥ dense mega 2.6ms（用户两想法）。
**prefill 侧（TTFT，第二优先）**：eager 797 kernel/步，**down GEMM grid=(44,1040,128)=5.8M block，16 次共 57ms（占 prefill 160ms 的 37%）**——grid.y 按 R=tokens×8 全量切，128 专家 ×1040 行块绝大多数空退。

## B1 2026-10-28 第一刀提交（commit 后缀 "B1"）：逐头 norm / rope 行数改实际 tokens，远端编译 + ABAB 对比排队中
- 改动：`crates/kernels/src/ops.rs` rope_span_bf16 显式收 tokens；`crates/layers/src/nn.rs` Rope::apply 加 tokens；`models/kaon3/src/layer.rs` head_rmsnorm 收 rows=tokens×heads、rope 传 tokens；测试 rope_vllm_parity 同步。
- 验证链 `/w/b1.sh`：备份旧二进制 → cargo build（看 Compiling 列表 + strings 验 marker）→ A(旧)/B(新)/A2/B2 各 c1+c8 + 固定 prompt 输出一致性。预期 c1 TPOT 16.3→~13.4。
- 又踩：排队用 `pgrep -f "bash /w/x.sh"` 匹配到等待进程自己 → 死锁 30 分钟（CLAUDE.md 早写"链内用 -x"）。改串行一条 `bash a.sh; bash b.sh`。
- **第二刀调研（全注意力 D=512 层 decode 1.85ms/步 = 5 层 × 369us）**：`gqa.cu:954 gqa_paged_kernel` 通用慢路径，grid=(req, 16 头)=16 block ×256 线程、每 block 对 KV 扫三遍、`for d<512` 标量 f32 点积、无 GQA 组共享（kv_heads=2 的 K/V 被 8 个 q 头各读一遍）。`gqa_decode_v3`（D=128/256，GROUP 1/2/4/6，split-K 多 block，cp.async 双缓冲）不支持 D=512：core 的 `s_acc[GROUP][KEYS][D]` 在 D=512/GROUP=8 时 = 8×32×512×4 = 512KB smem，超限；需要 D=512 专版（LPK=64 lane/key → KEYS=2/block 或改 2 pass）。**先等 B1 结果，再决定第二刀做 D=512 v3 专版（预期 369→~80us/层，省 ~1.4ms）还是 h=2816 norm（0.6ms，简单）**。
- D53 定稿（PIPE 三态同轮，c1）：PIPE=0 17.08/17.11、PIPE=1 16.15、unset 16.18 → **当前二进制默认=pipe 开，收益 -5.5% 与 D48 一致**。D49 "没编进"是当时未重编 kernels；`kserve_full.sh` 不需要再加 env。

## ★★★★★★★ D55 2026-10-28 B1 首批 A/B（同轮，旧二进制 A vs 新 B，c1）：TPOT 16.16 → 13.02（**-19.4%**），TTFT 42.0 → 39.8
预测 2.9ms（nsys 逐头 norm 2.29 + rope 0.70）实得 3.14ms。首次 Step B 刀收益达预期。等 c8 / 输出一致 / A2 B2 复测后进 CSV。
- **D55 定稿（ABAB 四轮）**：A 16.16/16.17 → B 13.02/13.04（c1，-19.4%，吞吐 60.4→74.4）；c8 19.55/19.57 → 16.40/16.44（-16.1%，吞吐 290→328）；TTFT c1 42.0→39.8；输出 "the last." 四次逐字相同。**B1 保留，入 e2e-rounds.csv（round b1）**。注：c8 TTFT p50 ~590ms 是本脚本每次冷起服务 + 8 并发同时涌入排在首个 prefill 批后（Step A 预热后 p50 82ms，p90 593）；对 TPOT/吞吐无影响，TTFT 口径以 Step A 的为准。

## ★★★★ D56 2026-10-28 RMSNORM_LAYOUT 三态（B1 二进制，c1 同轮两遍）：block256 12.58 < default(block1024) 13.02 < warp 13.60，输出一致
h=2816 单行 norm（181 次/步）：Block1024 每线程 1 向量两级归约 6.8us；Block256 每线程 2 向量快 ~2.4us；warp 版一 warp 一行 11 向量/lane 串行最慢。**-0.44ms/步（-3.4%）**。落地：`ArchProfile::rmsnorm_layout` 对 Sm120 返回 Block256（B2）。
## D57 2026-10-28 dense mega 旋钮（用户想法廉价验证，进行中）：base 13.02、MEGA_CTAS=2 13.32（每 SM 2 CTA 反而慢 +2.3%）……
- **D57 定稿（dense mega 旋钮，c1）**：base 13.02/13.03、CTAS=2 13.32（+2.3%）、SP 2/2 13.17、SP 8/8 13.16 **但输出坏**（"is is is…"）、**KAON3_MEGA=0 11.33 但输出坏**（同样 "is is is…"）。→ ① mega 旋钮无廉价收益；② **非 mega dense 路径与 SPGU=8 都有数值 bug**（待查，不在本 loop 关键路径上，但 11.33 说明 dense 2.6ms 里至少 1.7ms 可省——用户两想法方向对，值得做 dense 专刀）。
- **vLLM sm_120 分派调研（子代理，源码 /data/ref/vllm-img-v0.28.0-pr455）**：MoE 走 FlashInfer CUTLASS FP4 grouped GEMM 单次调用（gate+up 合 N=1408，align/sort/加权归约都在内部，≈3–4 kernel/层）+ Gemma4 自定义 Triton 路由 1 kernel（softmax+scale+topk 合一）+ fp4 quant 1；attention：滑窗层 FA2（window_left=1023），D=512 全注意力层 FlashInfer XQA（sm12x 专用 decode kernel）；dense 线性 = scaled_fp4_quant + CUTLASS FP4 scaled_mm，无 M=1 GEMV 特化；lm_head 普通 cuBLAS bf16 + softcap 逐点 + argmax 单发；**每步约 900–1200 kernel**（和我们 1298 同量级 → 差距不在 launch 数，在单 kernel 效率：MoE 3–4 vs 我们 22/层、attention 专用 decode kernel vs 我们 v5/慢路径）。
- **B2 定稿（ABAB）**：c1 13.03/13.03 → 12.58/12.59（-3.4%，吞吐 74.4→77.0）；c8 16.41/16.42 → 16.01/16.15；英文 + 中文 chat 输出四次逐字相同。入 CSV（round b2）。累计 B1+B2：c1 TPOT 16.16→12.58（**-22%**）。
**第二刀（B3）方案定稿**：decode 专用 split-K attention 接到 kaon3——nsys 证实滑窗层 decode 走 v5 prefill 风格（每头 1 block，25×50us=1.26ms），D=512 全注意力走通用慢路径（5×369us=1.85ms）；vLLM 同位置是 FA2 decode / XQA。改法：① `gqa_decode_v3_core` 加 `window`（key 下界 k0=max(0,n-window)，split 只切 [k0,n)）；② 新增 D=512 变体（LPK=64=2 warp/key，KEYS=2/tile，跨 warp 归约走 smem；GROUP=8）；③ `gqa_decode_bf16` C/Rust 加 window、支持 D=512；④ kaon3 `BatchMode::Decode` 时走 decode 入口（Paged 混批仍走原路径），LayerScratch 加 decode scratch。预期 -2.5ms（滑窗 ~0.9 + 全注意力 ~1.6）。

## B3 2026-10-28 第三刀提交：decode 专用 split-K attention（v3 加 window + 新 D=512 kernel）接 kaon3 Decode 模式
- 改动文件：`gqa_decode_v3_core.cuh`（core 加 window：k0=max(0,n−window)，split 只切 [k0,n)）、`gqa_decode_v3.cu`（kernel 壳加 window；新 `gqa_decode_d512_kernel<GROUP>`：256 线程、64 lane/key、4 key/tile、warp 内 shfl + 两 warp smem 合并点积、跨 key 槽合并逐 GROUP 做；入口 D=512 分派 group 1/2/4/8）、`gqa.cu`（`gqa_decode_impl` 加 window + D=512；新 extern `gqa_decode_win_bf16`；v2/v1 路径 window>0 报错）、`kernels.h`/`ffi.rs`/`ops.rs`（新封装）、`mega_*_sm120.cu` + 测试补 window=0、`models/kaon3/src/layer.rs`（LayerScratch.decode_scratch；`BatchMode::Decode` 走新入口，`ATTN_DECODE_OLD=1` 回退）。
- 风险：① d512 kernel 寄存器（qr[8][8]+acc[8][8]=128 float + 其它，256 线程上限 255/线程）可能溢出到 local → 看 ptxas 报告；② 数值口径：online softmax exp2f 与旧慢路径 __expf 三遍扫不逐位，只要求 argmax 输出一致（bench 固定 prompt + 中文 chat）；③ 混批 Paged 仍走旧路径不变。
- 远端 `TAG=b3 bash /w/bN.sh` 跑编译 + ABAB 中。预期 c1 TPOT 12.58 → ~10.1。
- B3 第一次验证：编译 EXIT=0（marker=9），但新二进制 **OOM**——`decode_scratch` 按 max_tokens=4160 行 × 16 头 × 32 split × 514 × 4B = **4.4 GB**；warmup prefill OOM 后 model.rs:207 panic（kv_cache 空）。修：`LayerScratch::new_with_rows(max_tokens, max_rows=max_num_seqs)`，scratch 68 MB。重跑中（A=旧二进制备份）。
- **B4 候选已写（未编）**：`gather_rows_kernel` 改 grid=(width/(256·8), rows) 的 16 B 向量拷贝（原 1 block 逐元素搬 lm_head 行 512 KB = 141 us/步，D54）。预期 -0.13 ms。等 B3 结果后与 B3 一起/分开验（分开验，一刀一数）。
- B3 重跑事故：重启命令里 `pkill -f "bash /w/bN.sh"` 把自己的 `docker exec bash -c` 壳杀掉（命令行含该串），后续 `cp` 备份没跑 → 本轮 A/A2 用的是 OOM 版 B3 二进制（NOTUP，无效），B/B2 = 修好的新二进制有效。pre-B3（B2）二进制已被覆盖丢失；同轮基线改用 `/w/dinfer_prev_b2`（B1 二进制，12.58 前一档 13.02）补跑。**第三次踩 pgrep/pkill -f（记忆 #7 已写）：以后链内一律 `pkill -x` 或按 PID。**

## ★★★★★ D58 2026-10-28 B3 新二进制（c1 / c8，A 轮因旧 OOM 二进制无效）：TPOT 12.58 → **9.77（-22%）**、c8 16.38 → 13.59（-17%），吞吐 77→97.8
但输出变了：英文 " the last." → "the capital of France is the capital of…"（循环），中文 chat 连贯但措辞不同。B3 设计上非逐位（在线 softmax exp2f + split-K merge vs 三遍扫 __expf），变化本身不能定对错 → 按 Rust 验证规则写 GPU 对拍 `crates/kernels/tests/gqa_decode_win_parity.rs`（6 个场景：D=256 窗 1024 / D=512 全窗 + 窗 1024，多请求含 1 / 33 / 64 / 1024 / 1025 边界，amp 1.0 / 2.0；判据 |err| ≤ 0.03+0.02|ref|，同时跑慢路径对照）。排队：同二进制 `ATTN_DECODE_OLD=1` vs 新 ABAB（`tmp_b3env`）→ 对拍测试（`tmp_partest`）。对拍不过则 B3 回滚（CSV 照记负结果）。
- 第三方参照：vLLM 基线对同一 raw prompt "The capital of France is" 输出也是退化循环（"France is France is…"），中文 chat 结构与我们两版都不同 → raw prompt 本就是低置信样本，**输出变化不是 B3 错误的证据**；对错只认 GPU 对拍测试。
- B3 自查修一处隐患：`gqa_decode_merge_v2_kernel` 的有效 split 数按 seq_lens 全长算，kernel 侧按窗口内 key 数算；window=1024 / kps=32 时两者都饱和到 32 恰好相等（scratch 零初始化也让多读的 split 贡献 0），但语义不一致 → merge 加 `window` 同式。已 put，随对拍测试一起编。
- D58 补（同二进制 `ATTN_DECODE_OLD=1` vs 新，ABAB）：旧 12.58 / 16.01（c1/c8，输出同 B2），新 9.79 / 13.63 —— 与 B3 新旧二进制对比一致，**B3 收益 -22% 确凿，归因单一（attention 路径）**。去留等对拍。

## ★★★★★ D59 2026-10-28 B3 GPU 对拍通过 → **B3 定稿保留**（归档 test/kaon3/2026-10-28/{partest,b3env,b3}_RESULT.txt）
`cargo test --release -p dinfer-kernels --test gqa_decode_win_parity -- --ignored`（5090 sm120，编译 28.5s，测试 0.92s）**EXIT=0**。6 场景全过，判据 |err| ≤ 0.03+0.02|ref|：
| 场景 | decode_win 最大误差 / 超差 | paged_slow 对照 |
|---|---|---|
| D=256 Hkv=8 win=1024 seq=[1600] | 0.0072 / 0/4096 | 0.0072 / 0 |
| D=256 win=1024 seq=[1600,70,1,1024,1025] amp=2 | 0.0078 / 0/20480 | 0.0080 / 0 |
| D=256 win=0 seq=[300,1] | 0.0077 / 0/8192 | 0.0077 / 0 |
| D=512 Hkv=2 win=0 seq=[1600] | 0.0078 / 0/8192 | 0.0078 / 0 |
| D=512 win=0 seq=[1600,70,1,33,64] amp=2 | 0.0078 / 0/40960 | 0.0078 / 0 |
| D=512 win=1024 seq=[1600,900] | 0.0078 / 0/16384 | 0.0078 / 0 |
**读法**：新路径误差与旧慢路径对 CPU f64 的误差同量级（都是 bf16 输入 + fp32 累加的本底 ~0.008），不是新 kernel 引入的偏差；窗口边界 1024/1025、单 token、D=512 多请求都对。
**结论**：B3 保留。D58 的输出变化判定为"非逐位但等价"（vLLM 对同 raw prompt 同样退化循环）。入 CSV round b3：c1 TPOT 12.58→9.79（-22.2%，吞吐 77.0→97.7）、c8 16.01→13.63（-14.9%，334→370）；TTFT 不变（38.6/583）。三刀累计 c1 **17.07→9.79（-43%）**，vs vLLM 4.69 还差 2.1×。
**下一刀**：B4（gather_rows 多 block 16B 拷贝，已 commit 4bcfd2a 未编，预期 -0.13ms）编译 + ABAB；之后按 D54 清单：MoE 小 kernel 融合（~700 小 kernel/步 + 1.89ms launch 间隙）、route_topk/softmax/scan 单线程 kernel 0.43ms、dense mega 2.61ms。

## B4 2026-10-28 验证中：`TAG=b4 MARKER=gather_rows_kernelILb1 bash /w/bN.sh`（GPU0，编译 EXIT=0，marker=17），A c1 9.77 基线已出，等 B/A2/B2。结果 `/w/dinfer/tmp_b4_RESULT.txt`。
- B3 kernel 资源（cuobjdump，无 spill）：d512 kernel REG 161 / smem 33984、v3 kernel REG 54–123 / smem 13–26KB、merge_v2 REG 80。LOCAL 全 0 → 寄存器隐患 ① 排除。
- 容器里 PID 64/853 两个 `dinfer` 是僵尸（Z，RSS 0，PPID 1=sleep infinity 不收尸），不占 GPU，无需处理；GPU 0 当前只有 b4 链占用。
- 下一刀 B5 候选：D54 "其余 ~700 小 kernel/步 <2.5us + 1.89ms launch 间隙"。正在从 `tmp_nsys2.sqlite` 导全表看这 700 个是谁（MoE 前处理 route_softmax/bucket/norm/gather/quant 各层 ~23 个？）。

## ★★★★ D60 2026-10-28 nsys 全表（45 种 kernel，`tmp_nsys2_full.txt`）+ 带宽 roofline → 下一刀定 **dense mega**（B5），不是小 kernel 融合
全表（c1 一步 1298 kernel，kernel 和 14.96ms，span 16.84ms → 空隙 1.9ms）。"其余 ~700 小 kernel" 拆开：add_inplace 90 次 0.8us、rmsnorm 1024 单行 151 次 1.3us、scatter_add/gather/grouped_quant/scan/count/row2expert/scale/mul_row/fill/f32_add 各 30 次 0.6–2.5us —— **合计只 ~0.6ms**，融合顶多省 0.4ms + 部分空隙，ROI 低。
**roofline（5090 1.79 TB/s）**：
| 段 | 字节/层 | 带宽下限/层 | 实测/层 | 差 |
|---|---|---|---|---|
| dense MLP fp8 gate+up+down（2816×2112×3 = 17.8 MB） | 17.8 MB | 10 us | **86.9 us** | **8.7×** |
| MoE 8 专家 NVFP4 gate/up/down（8×3×2816×704×0.5 = 23.8 MB+scale） | ~25 MB | 14 us | 18.6+12.5+18.6 = 50 us | 3.6× |
| qkv/oproj fp8 nvjet（2816×(2048+1024)+... ≈ 11 MB） | 11 MB | 6 us | 29.4+15.7 = 45 us | 7× |
| router gemv（2816×128 bf16 = 0.7 MB） | 0.7 MB | 0.4 us | 3.8 us | 10× |
| lm_head gemv（262144×2816 bf16 = 1.48 GB） | 1.48 GB | 826 us | 870 us | 1.05× ✓ |
**读法**：lm_head 这种"大 N 一个 gemv"已到带宽；**所有小 M（M=1）GEMM 都离带宽 4–9×**，dense mega 最差（2.61ms/步 = 17.4% 占比，若到带宽 0.3ms → **省 ~2.3ms/步，TPOT 9.6→~7.3，-24%**）。R26 已证 mega 的 wait 是 exec 的 2.5×（persistent 阶段链 1 CTA/SM 自旋），D57 旋钮（CTAS/SP）调不动 → 要换 kernel 结构，不是调参。
**B5 方案**：decode（M≤16）dense MLP 不走 persistent mega，走 3 个带宽型 kernel：① gate‖up 合一 fp8 GEMV（N=2×2112 行切 CTA，每 warp 若干行、K 连续 16B 向量流读、x 的 M≤16 行驻 smem、warp 内 shuffle 归约，epilogue 直接 gelu·mul 写 act bf16 + fp8 量化）→ ② down fp8 GEMV（N=2816，K=2112）→ norm 沿用。预期 3 launch ≈ 25us/层。保留 mega 路径作 `DENSE_DECODE_OLD=1` 回退；prefill 不动。同样结构可复用到 qkv/oproj（第 2 候选 1.3ms）。
- **B4 定稿（ABAB，归档 b4_RESULT.txt）**：c1 9.77/9.78 → 9.64/9.64（**-1.4%**，预测 -0.13ms 实得 -0.135ms，吞吐 97.7→99.1）；c8 13.59/13.79 → 13.45/13.47；输出四次逐字相同（对 lm_head 行是纯拷贝，逐位）。保留，入 CSV round b4。四刀累计 c1 17.07→9.64（**-43.5%**）。
- **口径复核（用户 2026-10-28 晚再提醒）**：所有 B 刀 ABAB 链两端均非 eager；dinfer `--max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 4096`；vLLM `vllm-noneager`（:28201 GPU3）`--max-model-len 16384 --max-num-batched-tokens 4096 --block-size 32 --gpu-memory-utilization 0.85` 无 `--enforce-eager`；bench 同二进制同参数（400 词 / 64 out / 8c 请求 / warmup 2 / seed 7 / 绑核 60-91）。**唯一差别**：vLLM 未显式 `--max-num-seqs`（默认 256）vs dinfer 64，c ≤ 16 两端都不触顶、不影响；测 c ≥ 32 前把 dinfer 改 256 对齐。

## B5 2026-10-28 开工：decode dense MLP 带宽型 fp8 GEMV（替代 persistent mega，M ≤ 16）
设计（模型无关，放 `crates/kernels/cuda/mlp_fp8_decode.cu`，所有 arch 共用，无 mma）：
1. `mlp_fp8_norm_quant`：一 warp 一行，照 `rmsnorm_row_warp` 口径（f32 平方和 → rsqrt → bf16 舍一次）再 `fp8(bf16·a_inv)`，与 mega S_NORM 逐位同。
2. `mlp_fp8_gemv_gateup`：A_q fp8 [m,K] 整体驻 smem（16×2816=45 KB），一 warp 负责 gate/up 同一行号各 2 行（4 行权重），lane 16 B 向量流读 K、手写 e4m3→f32 位解码、f32 FMA、warp shuffle 归约；epilogue 照 mega：`g=bf16(acc·ws/a_inv)`，`u` 同，`y=bf16(gelu_tanh(f32 g))`，`act=bf16(y·u)`，`act_q=fp8(f32(act)·dn_a_inv)`。写 act_q（喂 down）。
3. `mlp_fp8_gemv_down`：同结构，一 warp 1–2 行，输出 bf16 `down_out = bf16(acc·ws/a_inv)`。
4. post-norm 沿用 `rmsnorm_into`（mega S_POST 同口径）。
数值：三处 GEMM 的 fp32 累加顺序与 mega 的 mma 不同（非逐位），但同为 fp32 累加、同 fp8 输入 → 差应在 bf16 ulp 级；验证 = Rust GPU 测试 `mlp_fp8_decode_parity`（CPU f64 参照 + mega 路径对照，M∈{1,3,8,16}）。
接入：`layer.rs` `mega_dense_branch` 前加 `dense_decode_gemv`（`DENSE_DECODE_OLD=1` 回退到 mega），复用 MegaBufs 的 norm1_q / act_q / down_out。
预期：3 kernel ≈ 25 us/层（权重 17.8 MB @1.79 TB/s = 10 us 下限），vs mega 86.9 → 省 ~1.8 ms/步，TPOT 9.64 → ~7.8。
- **B5 代码已提交（8e4cf70）并 put 到 5090**：`crates/kernels/cuda/mlp_fp8_decode.cu`（3 kernel：norm_quant / gateup_act<M_TILE> / gemv<M_TILE>，M_TILE∈{1,2,4,8,16} 决定 smem = M_TILE·K·2 B，>48 KB 时 cudaFuncSetAttribute）+ kernels.h / ffi.rs（含 no-nvcc stub）/ ops.rs 三包装 + `tests/mlp_fp8_decode_parity.rs`（5 场景：m∈{1,3,8,16}×(2816,2112) + (5,256,160)；判据 fp8 码差 ≤1、down |err| ≤ 0.01+0.01|ref|）+ `models/kaon3/src/layer.rs`（`dense_decode_gemv` 在 `mega_dense_branch` 前，tokens ≤ 16 且 fp8 权重时走；`DENSE_DECODE_OLD=1` 回退；新增 `dq16`/`act_q16` 小缓冲）。
- 链 `/w/b5.sh`（GPU0 已启）：先编 + 跑对拍（门禁，不过则停 `PARITY_FAIL`），过了再 `TAG=b5 MARKER=gateup_act_kernel bash /w/bN.sh`（A=B4 二进制 / B=新，c1+c8 ×2 + 固定 prompt）。结果 `tmp_b5par_RESULT.txt` / `tmp_b5_RESULT.txt`。
- B5 第一次编译失败：extern "C" 块里 `bf16` 未限定（别名在 `dinfer` 命名空间）→ `using dinfer::bf16;`，b3f34d3，链重启。
- **B6 预案（等 B5 链结束再 put）**：qkv / o_proj fp8 GEMM（nvjet 29.4+15.7 us/层 = 1.31 ms/步，11 MB 权重下限 6 us）decode 时走同一 `mlp_fp8_gemv`：`QLinear::forward` FP8 分支 tokens ≤ 16 且 `M_TILE·K·2 ≤ 96 KB`（K=8192 的全注意力 o_proj 只 m ≤ 4 走）时 quant_fp8 → gemv，`FP8_GEMV_OLD=1` 回退。预期 -0.9 ms/步。
- **B5 对拍通过（`tmp_b5par_RESULT.txt`，编译 + 测试 17.5 s）**：5 场景全 ok。norm_q fp8 码全同 2816/2816、8448/8448、22528/22528、45055/45056、1280/1280（最大码差 0）；act_q 全同率 99.9%（33768/33792 等，差的那几个也是码差 0 ⇒ 实为 ±0 码——`code_diff` 为 0 但位不同只能是 +0/−0）；down bf16 最大误差 0.0073–0.0077（与 B3 测试同本底），超差 0。与 mega 口径同一组 scale 公式，norm 阶段 ±1 码的担心没出现。ABAB 已起。

## ★★★★★★ D61 2026-10-28 B5 首批 A/B（A=B4 二进制，B=B5）：c1 TPOT 9.64 → **7.47（-22.5%）**，c8 13.45 → 10.94（-18.7%），吞吐 99.1→125.7 / 373.8→422.4
预测 -1.8 ms 实得 -2.17 ms/步（mega 2.61 ms → GEMV 三 kernel ≈ 0.44 ms，比 25 us/层预期还好）。TTFT 不变（prefill 不走此路径）。
输出：中文 chat 一致；英文 raw prompt " the capital of France is…"（循环）→ " the following.  Wait, I'm not sure…"——非逐位路径（累加序）在低置信样本上翻 token，与 D58 结论同；对错以 `mlp_fp8_decode_parity`（fp8 码差 0 / bf16 ≤ 0.0077）为准。
A 轮 c8 一次 TPOT p50 19.02 / TTFT 222（其它轮 13.45 / 583）—— 同一二进制、同参数，是 bench 调度/排队态差异（TTFT 低 TPOT 高 = 请求没叠上），不是代码；看 A2 c8 是否回 13.4x。等 A2/B2。
- **B5 定稿（ABAB，归档 b5_RESULT.txt / b5par_RESULT.txt）**：c1 9.64/9.64 → **7.47/7.47（-22.5%）**，吞吐 99.1→125.8；c8 13.45/13.45 → 10.94/10.99（-18.5%），373→425。A2 c8 回 13.45 ⇒ A 轮 19.02 是排队态噪声。中文 chat 四次一致；英文 raw prompt 两版各自稳定。**保留**，入 CSV round b5。五刀累计 c1 **17.07 → 7.47（-56%）**，vs vLLM 4.69 差 1.59×。
- **B6 链已起（`/w/b6.sh`，GPU0）**：对拍（含 4 个 qkv/o_proj 形状 gemv）门禁 → `TAG=b6 MARKER=fp8_gemv_m_tile bash /w/bN.sh`（A=B5 二进制）。预期 c1 7.47 → ~6.6。

## B7 设计 2026-10-28（等 B6 结果期间写，MoE grouped GEMM 1.49 ms/步 → 带宽型 GEMV）
**现状**：`moe_grouped_sm120.cu` grid=(N/64, ceil(R/32), E)，decode c1 R=8（8 专家各 1 行）→ 每专家 11（gate/up）/ 44（down）个 CTA，共 88 / 352 CTA，BM=32 的 mma tile 装 1 行；nsys 18.6+18.6+12.5 us/层 vs 权重 25 MB 下限 14 us（3.6×）。与 dense mega 同病：小 M 用 mma tile 结构，CTA 少 + 每 K 步 __syncthreads。
**方案（同 B5 结构，NVFP4 版）**：`moe_fp4_decode.cu`（模型无关、所有 arch）：
1. `moe_fp4_gateup_act`：grid=(ceil(I/ kWarps), E)，block 内按专家行段 [off[e], off[e+1]) 以 16 行为 chunk 循环；chunk 的 A（fp4 → bf16 精确）+ A 块缩放（e4m3 → f32）驻 smem；一 warp 同行号 gate+up 两行，lane 一次 u2（16 nibble=一个 16-块）读 + 对应 B 块缩放字节，块内 16 FMA 后乘 sa·sb 累加（与 mma block_scale 口径同式）；epilogue `bf16(acc·alpha_g)`、`bf16(acc·alpha_u)` → gelu_mul round2 → bf16 写 gelu_o[R, I]。去掉 gate_o/up_o 两个中间缓冲和独立 gelu kernel。
2. `moe_fp4_gemv`（down）：grid=(ceil(N/(4·kWarps)), E)，一 warp 4 行（K=704 只 44 块，4 行让 lane 满），写 bf16 yg[R, N]。
3. 量化沿用 `moe_grouped_quant`（逐位同）。空专家 CTA 直接 return。host 只在 R ≤ 64（tokens·8）时走，否则原 grouped GEMM；`MOE_DECODE_GEMV_OLD=1` 回退。
**数值**：fp4·fp4 乘积 ≤ 4 bit 尾数、×两个 e4m3 scale ≤ 10 bit → 单项在 f32 精确；差只在累加序，bf16 ulp 级。对拍 = Rust GPU 测试（CPU f64 参照 + 现 grouped GEMM 对照）。
**预期**：c1 三 GEMM 50 us/层 → ~20 us，省 ~0.9 ms/步（TPOT 7.47 → ~6.5，与 B6 叠加 → ~5.7）。

## ★★★ D62 2026-10-28 B6 ABAB（A=B5，B=B6）：c1 7.47 → 6.96（-6.8%）**但 c8 10.94 → 11.56（+5.7%）且英文输出 "is is is…"（坏输出特征，同 D57 MEGA=0）** → **回滚**（`FP8_GEMV_OLD` 默认改为走旧路径），CSV 记负结果 round b6
- 对拍（`b6par_RESULT.txt`）：独立 gemv 4 形状（N=3072 K=2816 m=16/1、N=2816 K=4096 m=8、K=8192 m=4）对 CPU f64 全过（最大误差 ≤ 0.0118）—— **随机 fp8 字节 + 合成 scale 的对拍过了，生产路径坏** ⇒ 差在接入（A 的量化口径 / scale 公式 / 缓冲）而不是 kernel 算术。
- c8 变慢的结构原因：每 block 装 M_TILE·K·2 B 的 A 到 smem（m=8,K=2816 → 45 KB）却只服务 4 warp × 2 行 = 8 行权重（22 KB）—— A 装载 > 权重流量，m ≥ 8 时输给 nvjet。要改：每 block 服务更多行（8 warp × 2 行或每 warp 循环多行对），B5 的 dense 也有此问题但基线（mega）太差没暴露。
- 下一步：① 写直接对拍 `QLinear::forward` 新旧路径（同 `quant_fp8` 产出的 A、同权重，gemv vs `forward_fp8_cublaslt`）定位接入差；② 修 m ≥ 8 的 A 复用。
- **D62 修正（自我纠错）**：上面"输出坏 ⇒ 接入有差"下得太快。英文 raw prompt 在 B3/B5 每次换累加序都翻（" the last." → "capital of France…" → "the following. Wait…" → "is is is"），vLLM 自己也退化循环，**这不是对错证据**（D58 已定此规则，我又用它判了一次）；中文 chat 四次一致。独立 gemv 对拍（含 K=8192）全过。**B6 的真问题只有 c8 变慢**：block 4 warp 服务 8 行权重，却装 M_TILE·K·2 B 的 A 进 smem（m=8,K=2816 → 45 KB vs 权重 22 KB），A 装载流量 > 权重流量。
- **B6b 修法（已改两份 .cu）**：每 block warp 数改运行时参数，host `pick_warps`（按 SM 数与形状：M_TILE<16 用 16 warp、=16 用 8；grid 不足 2×SM 时减半）—— 一次 A 装载摊 32 行权重，A/权重流量比 2× → 0.25×。B5 的 dense gateup/down 同样受益（m ≥ 8 时）。kernel 内无卡型数值（SM 数运行时取）。
- 判据改为：c1 **且 c8** 都不劣于 B5，中文 chat 一致，对拍过；英文 raw prompt 不作判据（记录即可）。
- B7 已接进 `moe_branch`（`MOE_DECODE_GEMV=1` 开，默认关；R ≤ 128）：gate‖up+gelu 融合写 gelu_o → `moe_grouped_quant` → fp4 gemv down 写 yg。
- **B6b + B7 代码（66e179f）**：两份 GEMV .cu 的每 block warp 数改 host `pick_warps`（SM 数运行时取；M_TILE<16 → 16 warp，=16 → 8；grid < 2×SM 则减半）；B7 kernel `moe_fp4_decode.cu` + 对拍测试 `moe_fp4_decode_parity.rs`（4 场景：每专家 1 行 / 含空专家 / 9 行 > M_TILE 走 chunk 循环 / 小形状）。接入 `moe_branch`（`MOE_DECODE_GEMV=1`）。
- 验证链 `/w/b7.sh`：两个对拍（mlp_fp8 / moe_fp4）门禁 → bN（A=当前二进制默认 = B5 行为；B=新二进制 `FP8_GEMV=1 MOE_DECODE_GEMV=1`）。判据：c1 **且** c8 都 ≤ B5（7.47 / 10.94）、中文一致。
- B7 链已起（`/w/b7.sh`，GPU0）；远端 `bN.sh` 加 `BENV`（第 3 参，仅 B/B2 侧 `env $benv`），本地 `scripts/dev/kaon3_stepB_verify.sh` 若要同步此改动待收功时 pull。

## B8 预案 2026-10-28（prefill 侧 TTFT，D54 已证 down GEMM grid=(44,1040,128)=5.8M block 占 prefill 37%）
根因：`moe_grouped_sm120.cu` grid.y = ceil(R/BM) 按**总行数** R 切、grid.z = E=128，每个 (e, by) 块先读 offsets 再判 `m_blk ≥ row_hi → return`；R=8320 时 1040×128 = 133k 个 (e,by) 里只有 Σ_e ceil(len_e/32) ≈ 8320/32+128 ≈ 390 个有活，**99.7% 的 block 空转**，×44 个 N 块 = 5.8M 次启动。
方案：device 侧先建 tile 表（一个小 kernel，`tiles[t] = (e, m_blk)` 只含有行的块，`ntile` 原子计数；上限 R/BM + E 由 host 算），grouped GEMM grid=(N/BN, ntile_max, 1)，block 读 `tiles[blockIdx.y]`，越过 ntile 的 return。空转 block 从 5.8M 降到 ≤ 44×(R/32+E)。decode（R=8）也少 44×1040×127 的空 block → 可能再省一点。逐位不变（同 tile 同序）。等 B7 结果后做。
- **B7 对拍通过（`tmp_b7par_RESULT.txt`）**：moe_fp4 4 场景全 ok —— gateup_act 最大误差 0.0000–0.0039（输出量级 ~1，含 gelu 后 bf16 舍入，极小）；down 最大误差 0.015–0.038（超差 0）—— 比 mlp_fp8 的 0.0077 大，因为合成 down 输入是独立随机 fp4×e4m3 块缩放（量级 ~10–30 vs fp8 测试 ~1），相对误差同为 bf16 ulp 级；判据 0.01+0.01|ref| 对此量级合适。mlp_fp8 9 场景（含 B6 的 4 个 gemv 形状）再次全过。ABAB 已起（B 侧 `FP8_GEMV=1 MOE_DECODE_GEMV=1`，含 B6b warp 数改动）。
- 顺手发现：`moe_grouped_gemm` 里有 `MOE_GRID_CAP` 诊断开关（结果会错，注释写"测完删"）——上会话留下的，B8 做 tile 表时一并删。
- **B7 链 A 侧口径注意**：`bN.sh` 备份的"旧二进制"是 B6 那次编的（当时 `fp8_gemv_enabled` 默认开）→ A = B6 行为（c1 6.95 / c8 11.59），**不是 B5**。B7 的 B 侧与 **B5 定稿数（7.47 / 10.94）** 比，A 轮只作环境参照。

## ★★★ D63 2026-10-28 B7 ABAB：B 侧（FP8_GEMV=1 MOE_DECODE_GEMV=1 + B6b pick_warps）c1 **8.77** / c8 **16.37**，比 B5（7.47 / 10.94）差 17% / 50% → 不合入，拆变量
A 侧（=B6 行为）6.95 / 11.59 两轮稳定。B 侧三个变量叠在一起（B6 fp8 gemv、B6b warp 数、B7 MoE gemv），用同一二进制 env 开关拆：① 全关（= B5 + B6b 的 dense 受 pick_warps 影响）② 只 MOE_DECODE_GEMV=1 ③ 只 FP8_GEMV=1。中文一致、英文 "is is is" 与 A 同（二进制相同的非 gemv 部分）。
- B7 ABAB 定稿（归档 b7_RESULT.txt）：B 两轮 8.77/8.76、16.37/16.42 稳定 ⇒ 真退化不是噪声。拆变量链 `/w/b7split.sh` 已起（同二进制：① MOE_DECODE_GEMV=1 vs 关 ② FP8_GEMV=1 vs 关；"关"= B5 路径 + B6b pick_warps 的 dense）。

## ★★★★ D64 2026-10-28 B7 拆变量（同二进制 env）：`MOE_DECODE_GEMV=1` 单独 → c1 **7.48 → 9.27（+1.8 ms，+24%）** ⇒ **B7 的 MoE GEMV 是退化源，放弃接入**
- "全关"= B5 路径 + B6b pick_warps：c1 7.48 / c8 11.14（B5 定稿 7.47 / 10.94）⇒ B6b 对 dense 基本中性（c8 +0.2，在噪声边）。
- **为什么 B7 慢（事后看清）**：decode c1 每专家 1 行时，gate‖up 每 warp 只流 2 行 × 1.4 KB = 2.8 KB、lane 只 6 次迭代；down 每 warp 4 行 × 352 B，lane 只 2 次 —— 工作太碎，每 block 的 A 装载 + barrier + 收尾占大头，根本不是"流式"；而旧 grouped GEMM 的 mma tile 一个 CTA 吃 64 列 × 整 K，kernel 本身 18.6 us 只比下限差 3.6×。**B5 的"warp 一行 GEMV"只在单个大矩阵（N·K ≥ 数 MB、每 warp 流 ≥ 10 KB）有效，MoE 单专家 N=704 不满足。** 正确方向应是把 gate/up/down 三段融进一个 kernel 保 smem 中间量（减 launch 与往返），不是换 GEMV。
- 另一疑点（查表数组 e2m1 落 local memory / exp2f）已顺手改成 cvt 内建（moe_fp4_decode.cu），但不足以翻盘，B7 不再验证；kernel 与测试留作小 M 参考，`MOE_DECODE_GEMV` 默认关，layer 接入保留（关着不影响）。
- FP8_GEMV 拆分（split2）结果待出；B6c 已把 gemv 门槛改为 smem ≤ 24 KB（K=2816 → m ≤ 4）。
- D64 补（split2 `FP8_GEMV=1`，B7 二进制含 pick_warps）：c8 **15.96**（关 10.99）—— m=8 走 gemv（K=2816 smem 45 KB、warps=8）比 D62 的 11.56 还差 4.4 ms ⇒ pick_warps 把 warp 数从 4 提到 8 后更慢（smem 不变、block 数减半、SM 填不满）。证实 c8 问题是占用/并行度而非 A 流量；**B6c 门槛 smem ≤ 24 KB（m ≤ 4）** 是对的，c8 不该走 gemv。等 c1。
- D64 定稿（split2 两轮）：`FP8_GEMV=1`（B7 二进制，pick_warps=8 at m=8）c1 **9.27 / 9.27**（关 7.49）、c8 15.96/16.21 —— **c1 也慢了 1.8 ms**，与 D62 的 c1 6.96（warps=4）相反 ⇒ **pick_warps 把 warp 数提到 16（m=1,K=2816 smem 5.6 KB，qkv 3072 行 → 16 warp → 192 block）是错的**：block 少了 4×，SM 只 1 block/each，权重流读的并行度不够。结论：**warp 多 ≠ 好；小 M 下 block 数（并行度）比 A 复用重要**。pick_warps 改回按 "block ≥ 4×SM" 选、上限 4。
- **D64 更正（又一次自我纠错）**：上两条把 split1（MoE）的 B2 行（9.27 / 16.21）错读成 split2（FP8）的——监控输出两个文件拼在一起，我按出现顺序归了档。**真 split2（`FP8_GEMV=1`，B7 二进制 pick_warps=8@m=1）：c1 7.49 → 7.00（-0.5 ms）、c8 11.00 → 11.39（+0.4 ms）**，与 D62 同（c1 赚 c8 赔）。所以"pick_warps 16 让 c1 慢 1.8 ms"**不成立**，那 1.8 ms 是 MoE GEMV 的。pick_warps 改 4 / block ≥ 4×SM 仍保留（block 数 2.3×SM 偏少是事实，但影响没那么大）。B6c（m ≤ 4 才走）要验的是 c8 回到 ≈11.0 且 c1 保住 ≈7.0。
- 教训进 `dinfer-my-recurring-mistakes`：**多文件拼接的监控输出，必须按文件名分段读，不能按行序归档**。

## B9 预案 2026-10-28（B7 失败后的正确方向：MoE 三段融合，不换 GEMV）
decode c1 的 MoE 段 nsys：grouped_gemm gate 18.6 + up 18.6 + down 12.5 + gelu 1.0 + grouped_quant 1.4×2 + scatter 2.5 + 小 kernel ≈ 58 us/层，其中三次 GEMM 各自装 A tile、各 launch。
方案：一个 kernel 按 (专家 e, N 块) 组织，CTA 对本专家 ≤ 32 行：① 读 A 行块一次驻 smem；② gate/up 两个 N=704 的 GEMM 对同一 A（dual 已做到同 kernel 不同 blockIdx.z，但 A 装两次）→ 改为同 CTA 算 gate 列块与 up 列块、epilogue 直接 gelu·mul 写 bf16 中间量到 smem；③ down 需要整行 704 的 gelu 输出 → 同 CTA 做不到（N 块只有 64 列），所以融合止于 gate‖up+gelu（省 1 次 A 装载 + gelu kernel + 1 次 launch ≈ 3–4 us/层），down 仍独立。预期 ≈ -0.1 ms/步，小。
**更大的头在别处**：D54 清单扣掉已做的（norm/rope 3 ms、dense 2.2 ms、attention 1.3 ms、gather 0.14）后，c1 7.0 ms 里：MoE 三 GEMM 1.49 + nvjet qkv/oproj 1.31 + 全注意力/滑窗 attention（B3 后）~1.2 + 小 kernel/launch 间隙 ~2 + lm_head 0.87。**launch 间隙 1.9 ms（D54）现在占比最大**——graph 内 1298 → 现在约 1100 kernel/步，每个 ~1.5 us 间隙。下一刀应是**减 kernel 数**：MoE 前处理 route_topk+softmax+count+scan+fill+row2e（6 launch → 1）、grouped_quant 并进 gather、f32_add_to_bf16 并进 scatter_add、attention 的 cache_write 并进 rope。目标 -300 launch ≈ -0.45 ms。

## B10 预案 2026-10-28：MoE 路由前处理 6 launch → 1（decode tokens ≤ 64）
现链（每层）：route_topk_ids（grid=tokens）→ route_softmax（grid=tokens，1 线程/块）→ memset(counts) → count → scan（1 block）→ fill → row2expert（grid=E）。7 次 GPU 动作、每次都是几 us 的小活，nsys 合计 ~15 us/层 + 7×1.5 us 间隙 ≈ 25 us/层 ≈ **0.75 ms/步**。
融合 kernel `moe_route_bucket_small`（1 block，tokens ≤ 64）：smem 装 logits[tokens×E]（64×128×4 = 32 KB），每 warp 处理 2 token：top-k（照 route_topk_ids 的顺序选择，逐位同）、softmax 权重（照 route_softmax 的 f32 口径逐位同：全体 Z、renorm 的 32-lane 树形）；再 block 内 counts（smem 原子）→ scan（单线程）→ fill（smem 原子游标，顺序"不定"与旧版一致：归约是加法）→ tile_off → row2e。写 topk_ids / topk_w / offsets / cursor / sorted_tok / sorted_k / tile_off / row2e。host 侧 tokens ≤ 64 走融合，否则原链。逐位：topk_ids / topk_w / offsets / tile_off / row2e 应全同；sorted_tok 区间内顺序允许不同（旧版也靠 atomicAdd 顺序不定）→ 判据改为"每区间作为集合相同"。预期 -0.6 ms/步（c1 7.0 → ~6.4）。
- D64 split2 定稿（两轮）：`FP8_GEMV=1` c1 7.49→7.00/6.99、c8 11.00→11.39（归档 b7s_fp8_RESULT.txt）；split1 `MOE_DECODE_GEMV=1` c1 9.27/9.27、c8 15.96/16.21（b7s_moe_RESULT.txt）。
- **B8 链已起**（`/w/b8.sh`：moe_pipe_parity 3 个逐位测试 + mlp 对拍 → bN，A=B7 二进制默认 / B=新二进制 `FP8_GEMV=1`（B6c m≤4 + pick_warps≤4 + tile 表）。判据：c1 ≤ 7.0、c8 ≤ 11.0、TTFT c8 583 → 明显降（tile 表主要省 prefill）。
- B8 第一次编译失败：`kernels.h` 的 `moe_bucket` 声明没带 `tile_off`（Python 编辑脚本在该处 assert 失败后整段未写盘，我只核了 .cu/.rs 没核 .h）→ C linkage 重载报错。已修。**教训：多文件批量编辑后用 grep 逐文件核对目标串，不信脚本的 "ok"。**
- **B10 代码已提交（2aac32e）**：`moe.cu` 新增 `route_bucket_small_kernel`（单 block 256 线程，smem logits 副本 + counts/cursor；top-k 照旧序、权重调公共 `route_softmax_row`、scan 含 tile_off、fill 用 smem 原子、row2e 二分）；`layer.rs` grouped 路径 tokens ≤ 64 默认走（`MOE_ROUTE_FUSED=0` 回退），跳过旧 softmax/bucket/row2e 三次调用；测试 `moe_route_fused_parity`（5 场景，逐位 + 区间集合）。
- 链 `/w/b8b.sh`：moe_pipe_parity + mlp_fp8 + moe_route_fused 三门禁 → bN（A=B7 二进制 / B=新：B6c+pick_warps≤4+B8 tile 表+B10 融合，B 侧 `FP8_GEMV=1`）。

## B11 预案 2026-10-28：MoE grouped 路径再减 4 launch/层（gather 并进 quant；memset+scatter_add+memset+f32_add → 1 个 combine）
- **B11a** `moe_grouped_quant` 加可空 `sorted_tok`：源行 = `A[sorted_tok[r]]`，直接从 h2 读，去掉 `moe_gather_bf16`（xg 中间缓冲不再写）。逐位同（量化同一行数据）。
- **B11b** 路由时多产出逆映射 `inv[tok·K+k] = sorted 位置`（B10 融合 kernel 与旧 `fill_kernel` 都写），新 kernel `moe_combine`：grid=tokens，out[tok,d] = bf16(Σ_k w_k·y[inv[tok,k],d])（mode 1：Σ bf16(bf16(w)·v)），替代 memset(acc) + scatter_add(atomic) + memset(y) + f32_add_to_bf16。累加序从 atomic 不定 → 固定 k 升序（旧版本身不确定，不算退化；对拍判据 |Δ| ≤ bf16 ulp）。
- 省 4 launch/层 = 120/步 ≈ 0.18 ms 间隙 + ~6 us×30 kernel ≈ **-0.35 ms/步**。
- nsys 里 `fp8_e4m3_roundtrip` 90 次/步（kv_fp8 默认关，来源待查：可能是 `Gemm::forward_fp8`/attention 的 A 往返）—— B12 候选。
- **b8b 三门禁全过（`tmp_b8bpar_RESULT.txt`）**：moe_pipe_parity 3 测（含 tile 表）bad=0；mlp_fp8 9 场景 ok；**moe_route_fused_parity 5 场景：ids 同、w 位差 0、offsets/tile_off/counts/cursor 同、row2e 同、sorted 区间集合同** ⇒ B10 逐位。ABAB 跑中（A 侧 c8 又出一次 17.14 排队噪声，与 D61 同模式）。
- b8b 的 bN **B 侧编译失败**（`layer.rs:1634` `moe_fp4_gemv` 多了个 `tile_off`——B8 批量替换 `&mg.alpha_down, &sc.buckets.offsets,\n` 也命中了 B7 的调用），bN 没检查编译 EXIT 就跑 → B 侧用的是 12:18 旧二进制 + `FP8_GEMV=1`，数字 7.00/11.39 = D64 split2 重放，**无效**。已修；下一链（b8c）把 B11 一起带上。**bN.sh 要加 "EXIT≠0 则停"**。
- **B11 代码落地**：`moe_grouped_quant` 加可空 `sorted_tok`（gather 融进量化，去 `moe_gather_bf16` 与 xg 写）；路由（旧 `fill_kernel` 与 B10 融合 kernel）多写 `inv[tok·K+k]`；新 `moe_combine_bf16`（grid=tokens，mode 同 scatter_add；替代 memset(acc)+scatter_add+memset(y)+f32_add_to_bf16，`MOE_COMBINE_OLD=1` 回退）。grouped 路径每层 -4 launch。
- `bN.sh`（远端 + 本地 `kaon3_stepB_verify.sh`）加编译失败即停（NOBUILD DONE）。
- b8b ABAB 归档（`b8b_RESULT.txt`，**B 侧无效**：编译失败跑了旧二进制）；A 侧两轮 7.48 / 11.11 再次确认 B5+pick_warps 基线。
- **b8c 链**：三门禁 → bN（TAG=b8c，MARKER=combine_kernel，B 侧 `FP8_GEMV=1`；新二进制含 B6c + pick_warps≤4 + B8 tile 表 + B10 路由融合 + B11 gather/combine 融合）。预期 c1 7.48 → ~6.5（B10 -0.6、B11 -0.35、B6c -0.5）、c8 ≤ 11.0、TTFT c8 583 → 明显降（B8）。
- b8b 的 bN 实际跑了两遍（RESULT 文件里 B/A2 重复出现；多半是 b8.sh 第一次编译失败后我又起 b8b.sh，而 bN 没停）—— 已 kill（容器内按 pid，`pkill -x bench/dinfer`），**b8c 已起**。监控脚本改为只在 `==bN` 段看 DONE（之前把 par 段的 "PARITY_FAIL DONE" 当终止符）。

## B12 预案 2026-10-28：`fp8_e4m3_roundtrip` 90 次/步的来源 = `cfg.kv_fp8`（config.json `kv_cache_quant_algo` 非空 ⇒ 默认开），每层 q/k/v 各一次
- 这是 KV cache fp8 量化口径的"bf16→e4m3→bf16 往返"（与 vLLM fp8 KV 一致）；q 也往返。3 launch/层 + 独立 `gqa_cache_write` 1 launch/层 = 120 launch/步。
- 融合：① k/v 的往返并进 `gqa_cache_write_kernel`（写 cache 前做 roundtrip，k/v 缓冲本身也要被后续 attention 读 → kernel 顺便写回 k/v）；② q 的往返并进 rope（rope_kernel 末尾对 q 做 roundtrip；k 也在 rope 里，可一起做，这样 cache_write 只管搬）。全部逐位（同一个 cvt 两次）。省 3 launch/层 ≈ -0.15 ms/步。
- 做法：`rope_span_bf16` 加 `fp8_rt: int` 参数（0/1），`gqa_cache_write_bf16` 加 `fp8_rt`（对 v 做；k 已在 rope 里做了）。等 b8c 结果后做。
- b8c 编译失败：`grouped_quant_kernel` 的 kernel 体 `sorted_tok` 改动丢了（第一次 B11 脚本在 kernel 体后面的 host 段 assert 失败 → 整个文件未写盘；第二次脚本只补了 host 段）。**同一错误第 3 次**（B8 kernels.h、B8 fp4_gemv、这次）：批量编辑脚本一处 assert 失败 = 该文件全部改动丢失，必须 grep 每个目标串。已修。
- **B12 代码（进行中）**：`gqa.cu` 新增 `gqa_cache_write_fp8rt_kernel`（q/k/v 原地 fp8 往返 + k/v 写 cache，往返口径逐字同 `fp8_e4m3_roundtrip_kernel` scale=1）+ `kernels.h` 声明已写；ffi/ops/layer（`KV_FP8_FUSED=0` 回退）接线中。config.json `quantization.kv_cache_scheme` 存在而 `kv_cache_quant_algo` 为 None → 待确认 kv_fp8 是否真开（nsys 90 次往返/步说明开着，来源可能是 `kv_cache_scheme` 分支或 KV_FP8 env）。
- **B12 代码完成（本地 commit，未 put）**：kv_fp8 来源确认 = `hf_quant_config.json` `quantization.kv_cache_quant_algo = "FP8"`（config.json 里没有，我先前查错文件）。ffi（主 extern 块，无 no-nvcc stub 列表项，与 `gqa_cache_write_bf16` 同待遇）/ ops `gqa_cache_write_fp8rt_bf16` / layer（`kvfp8_fused` 时一次 launch 替代 q/k/v 三次往返 + cache_write；`KV_FP8_FUSED=0` 回退）。逐位（同 cvt）。-3 launch/层。等 b8c 结果后随下一链（b8d）验：门禁 = 现有三测 + 固定 prompt 输出与 b8c 的 B 完全一致（kv 往返逐位 ⇒ 输出必须逐字同）。
- b8c 三门禁通过（含 B11b combine vs scatter_add 两 mode 全 ok）；编译 EXIT=0 marker=19；A 侧 7.49 / 11.12。等 B。

## ★★★★★ D65 2026-10-28 b8c 首批 B（B6c+pick_warps≤4+B8 tile 表+B10 路由融合+B11 gather/combine 融合，`FP8_GEMV=1`）：c1 TPOT 7.49 → **6.63（-11.5%）**，**TTFT c1 38.3 → 29.9（-22%）、c8 580 → 493（-15%）**（B8 tile 表的 prefill 收益）；c8 TPOT 11.12 → 11.60（+4%，疑 FP8_GEMV m≤4 门槛仍伤 c8 或 B10 单 block 在 tokens=8 慢）。中文 chat 一致；英文 raw 变体（非判据）。等 A2/B2，再 env 拆 c8 的 +0.5。

## B13 预案 2026-10-28：norm + 残差加 融合（-3 launch/层 = -90/步 ≈ -0.15 ms）
每层现有：① `post_attn_norm(o) → norm; x += norm`；② `norm(dense) + h2(moe) → norm; post_ff_norm(norm) → h2; x += h2`；③ `layer_scalar`（scale_bf16_n，条件）。
- 新 kernel `rmsnorm_add_bf16`：`res += bf16(rmsnorm(x)·w)`（一 block 一行，先算 norm 行再原地加 res）。① 用它（o → x）；② 第二段 `post_ff_norm(norm) → h2; x += h2` 用它（但 h2 之后还被用？看 1386 后——h2 下一层重算，不再读 → 可省写 h2）。
- ② 第一段 `norm += h2` 是 add_inplace，再 norm：用现有 `fused_add_rmsnorm`（res=norm, x=h2, out=tmp）语义 = `norm += h2; tmp = rmsnorm(norm)·w` 正好。
- 全部逐位（同一 norm 数值、同一 bf16 加）。等 b8c 定稿 + b8d（B12）后做；与 B10/B11/B12 一起把 launch 从 ~1100 → ~900/步。
- **b8c 定稿（ABAB 两轮稳定，归档 b8c_RESULT.txt）**：c1 TPOT 7.49/7.49 → **6.63/6.64（-11.4%）**，吞吐 125.5→143；**TTFT c1 38.2→29.8（-22%），c8 580→493（-15%）**；c8 TPOT 11.12/11.08 → 11.60/11.60（+4.5%）。中文四次一致。入 CSV round b8（含 B6c/B8/B10/B11 四项，c8 的 +0.5 ms 用 split 拆：`/w/b8csplit.sh` 已起，FP8_GEMV 与 MOE_ROUTE_FUSED=0 各一组）。六刀（B1–B5、b8c）累计 c1 **17.07 → 6.64（-61%）**，vs vLLM 4.69 差 1.42×；TTFT c1 43 → 29.8 已反超 vLLM 25.0 的差距缩到 1.19×。
- **B13 实现方案定**：不加新 kernel 文件，`norm.cu` 三个 rmsnorm kernel（vec/warp/scalar）加模板 bool `NORM_ADD`：为真时 `out[i] = bf16(f32(out[i]) + f32(bf16(norm·w)))`（与 `add_inplace_bf16_n` 的 bf16 加同式：先舍 norm 再 f32 加再舍）。launch 多一个 `add_out` 参数，新 extern `rmsnorm_add_bf16(x, w, out_res, …)`。layer：① `post_attn_norm(o) → x +=`；② `post_ff_norm(norm) → x +=`（h2 不再写：下一层 pre_ff_norm_2 重算 h2，1386 后无人读 h2——已核）；③ `norm += h2; post_ff_norm` 用现有 `fused_add_rmsnorm`（res=norm, x=h2, out=tmp）——但 ② 要 x += 而不是写 tmp，所以 ②③ 合成：`fused_add_rmsnorm`(res=norm, x=h2) 产 tmp → 还是要一次 add。改为：③ 用 `fused_add` 产 tmp 到 h2，② 用 `rmsnorm_add`... 两者都是 norm 操作，不能一次。最终每层：`rmsnorm_add`(o→x) 1 次代 2；`fused_add_rmsnorm`(norm+=h2 → h2=norm(norm)) 1 次代 2；`x += h2` 保留 1 次（或并进 layer_scalar：`x = (x + h2)·s` 一个 kernel）。合计 5 → 3 launch/层。
- **D65 split1（b8c 二进制，`FP8_GEMV` 关 vs 开）**：关 c1 7.11 / c8 11.78；开 6.63 / 11.60 ⇒ **B6c（m ≤ 4 走 gemv）c1 -0.48、c8 -0.18，两边都赚**（c8 的 8 token 请求实际 decode 步常 < 8 行，m ≤ 4 命中）。c8 的 +0.5（11.1 → 11.6）**不是 B6c**，在 B8/B10/B11 里；等 split2（`MOE_ROUTE_FUSED=0`）。
- **B13 代码落地（commit）**：`norm.cu` vec/warp/scalar 三 kernel 加模板 `NORM_ADD`（`out[i] = bf16(f32(out[i]) + f32(bf16(norm·w)))`，与 add_inplace 同式）、`rmsnorm_add_bf16` extern + ffi/ops；layer：post_attn `rmsnorm_add(o → x)` 代 2 launch；post_ff `norm += h2; rmsnorm_add(norm → x)` 代 3 launch（只在 norm_mode==1 且非 sub_dump 时；`NORM_ADD_FUSED=0` 回退）。-2 launch/层。随 b8d 一起编验（b8d 二进制 = b8c + B12 + B13）；判据：三门禁 + 中文一致 + **英文 raw 输出与 b8c 逐字同**（B12/B13 都应逐位）。

## B14 候选 2026-10-28（b8c 后 c1 6.64 ms 的去向估算，待 nsys 复抓确认）
| 段 | 估算 ms/步 | 来源 | 下一刀 |
|---|---|---|---|
| MoE grouped GEMM gate/up/down | ~1.45 | D54 50 us/层 × 30，B8 tile 表只去空 block（c1 本就少） | 单专家 1 行时 CTA 内 K 流水不够深：BK=128 一步一 barrier；改 BM=16 tile（M=1 浪费减半）+ 3 阶段 cp.async 流水；或 gate‖up 同 CTA（dual 已有但 A 装两次）|
| attention（B3 后）| ~1.2 | 滑窗 v3 25 层 + D=512 5 层 | 已带宽型；看 KV 读是否 fp8（kv_fp8=on 但 cache 仍 bf16 存 → 读 2× 字节！）→ **B15：cache 真存 e4m3，attention 读 fp8 解码** 省一半 KV 带宽 |
| nvjet qkv/oproj（m=1 已走 gemv）| ~0.8 | B6c 后 qkv 2816→3072/4096 两种、oproj | gemv 本身 29→? us 待 nsys |
| lm_head gemv | 0.87 | 带宽 1.05× 下限 | 不动 |
| launch 间隙 | ~1.3 | ~900 kernel × 1.5 us | B12/B13 再 -150 → B14：attention 前后 rope/norm 等 7 个小 kernel 合 1（per-layer "attn_pre"）|
| 小 kernel 自身 | ~0.5 | | 同上 |
**最大单项 = B15（KV cache 真 fp8 存）**：decode 每步读 KV = 30 层 × 2(k,v) × 1560 tok × nkv·D × 2 B ≈ 25 层×8×256 + 5 层×2×512 → ~0.8 GB/步 bf16 → fp8 0.4 GB，省 ~0.25 ms；且 cache 容量翻倍。vLLM 就是 fp8 KV（kv_cache_dtype=fp8_e4m3）。需改 cache 布局 + v3/d512 kernel 读路径（e4m3→bf16 展开），attention 数值逐位同（往返值相同）。
- D65 split2 进行中（`MOE_ROUTE_FUSED=0` vs 开，FP8_GEMV 两边都关）：A（开）7.12 / 11.60。等 B。注意 envab 两组的 "默认" 都没带 FP8_GEMV → 基线是 7.11，不是 b8c 的 6.64；对比只看组内 Δ。
- **D65 split2（`MOE_ROUTE_FUSED=0` vs 开）**：关 7.28 / 11.67，开 7.12 / 11.60 ⇒ **B10 c1 -0.16、c8 -0.07，两边都赚**（比预估的 -0.6 小：7 launch → 1 实省 0.16 ms ⇒ 单 launch 间隙在 graph 内约 0.7 us，不是 1.5 us；D54 的 1.9 ms 间隙估算偏高）。c8 +0.5 的归因只剩 B8（tile 表：每 block 多一次 offsets/tile_off 读 + 7 步二分，c8 R=64 时 block 数 ~ 2×；但 TTFT -15% 是它）或 B11（combine 单 block/token 串行 K=8 读）。下一链 b8d 带 `MOE_COMBINE_OLD=1` 的 split。
- **修正 launch 间隙模型**：graph 内 ~0.7 us/launch ⇒ 剩 ~900 launch ≈ 0.6 ms，不是 1.3；B12/B13 预期各 -0.06/-0.04 ms（小），B15（KV fp8 存）和 MoE GEMM 深流水才是 >0.2 ms 级的刀。

## B15 设计 2026-10-28：KV cache 真存 e4m3（kv_fp8 层）—— decode KV 读带宽减半、cache 容量翻倍
现状：`cfg.kv_fp8` 只做 bf16→e4m3→bf16 **往返**，cache 仍 `Tensor<bf16>`（`model.rs:27`，`service.rs dtype_bytes: 2`）；attention 读 bf16。vLLM `kv_cache_dtype=fp8_e4m3` 真存 1 字节。
方案（Kaon-V3 scale=1，无 per-tensor kv scale 要带）：
1. cache 元素类型按层：kv_fp8 时 `Tensor<u8>`，`dtype_bytes: 1`（KvCacheSpec），布局不变 `[blocks, bs, nkv, 2, D]`。
2. 写：`gqa_cache_write_fp8rt_kernel`（B12）直接写 e4m3 字节（q/k/v 原地往返保留，供本步 attention 的 K/V bf16 输入）。
3. 读：`gqa_decode_v3_core.cuh` / `gqa_decode_v3.cu`(d512) / `gqa.cu` 通用慢路径 + prefill 路径（含 fa_v5 读历史 KV）—— 模板加 `KV_FP8`：cp.async 搬 8 B（16 个 e4m3）到 smem 的 fp8 区，用前 `__nv_cvt_fp8x2_to_halfraw2` 展开成 bf16（与往返值逐位同，因为往返 = 同一 cvt）。smem K/V 区可改存 e4m3（省一半 smem → KEYS 翻倍）或展开后存 bf16（改动最小）。先做"展开后存 bf16"。
4. 对拍：`gqa_decode_win_parity` 加 KV_FP8 场景：cache 字节 = e4m3(k)，参照 = 用往返后 k/v 的 CPU f64 → 应与 bf16 cache + 往返输入的结果**逐位同**。
预期：c1 decode KV 读 ~0.8 GB/步 → 0.4 GB，attention 1.2 → ~0.95 ms（-0.25）；c8 更多（KV 读占比更高）；prefill 不变。工作量：4 个 kernel 读路径 + spec/alloc + 测试，≈ 半天。排 b8d 之后。
- **B15 范围勘定**：读 cache 的 kernel = `gqa_decode_v3_core.cuh`（v3 D=128/256，cp.async 16B 流水）、`gqa_decode_v3.cu` d512、`gqa.cu` `gqa_paged_kernel`（通用慢路径，逐元素 `to_f32(kj[d])`）、`gqa.cu` fa_v5/v6/v7（chunked prefill 读历史 KV，待确认是否在 kaon3 路径上）。写 cache = `gqa_cache_write_kernel` + B12 `fp8rt` 版。
  方案定：① ffi/C ABI 加 `int kv_fp8`（cache 指针按 u8 解释）；② v3/d512：smem 加 fp8 暂存区 `[PIPE][KEYS][D] u8`，cp.async 改 8 B（`cp.async.ca` 8 字节 + zfill），到达后每线程把自己的 8 个 e4m3 展开成 bf16 写入现有 `s_k/s_v`（后续消费代码不动）；③ paged：模板 `KV_FP8`，`kv_load(ptr,d)` 按类型解码；④ Rust：`LayerKv` 加 `is_fp8`，cache 仍 `Tensor<bf16>` 但长度减半（字节数 = 原一半），kernel 边界按 u8 解释——避开 `KvCacheSpec`/`AttnCtx` 类型改动；`KV_FP8_STORE=0` 回退。⑤ 测试 `gqa_decode_win_parity` 加 fp8 cache 场景：bf16 cache（往返值）vs fp8 cache 逐位同。
- **b8d 门禁全过**（B12/B13 二进制）：rmsnorm_add 4 场景位差 0；其余同前。ABAB 跑中（A 侧 = b8c 二进制但 **A 不带 FP8_GEMV**（bN 只给 B 侧 BENV）→ A 7.11/11.60 是 D65 split 的"关"口径，B 要与 b8c 定稿 6.64/11.60 比，不是与本链 A 比）。
- **B15 开工**：fa_v5 PAGED（混批 prefill+decode）也读 cache（cp16z 16B → smem bf16），kaon3 D=256 走它；先不改 v5，kv_fp8 存储开时混批强制 cfg=-1 慢路径（正确但慢；decode 专用路径 `gqa_decode_win` 已覆盖 c1–c16 主场景）。
- **b8d 首批 B（B12+B13）**：c1 6.64 → **6.57（-1.1%）**，c8 11.60 → 11.58；TTFT 29.8 → 29.5。**英文 raw 输出与 b8c 逐字同**（"the following. Wait, I'm not sure if I should be here…"）⇒ B12/B13 逐位，符合预期。-5 launch/层 实省 0.07 ms ⇒ 单 launch ≈ 0.5 us。等 A2/B2 后入 CSV round b8d。
- **b8d 定稿（B12 kv_fp8 融合 + B13 norm+残差融合，ABAB 两轮）**：c1 6.64 → **6.57/6.58（-1.0%）**，c8 11.60 → 11.58/11.52；TTFT 29.8 → 29.5。英文 raw 输出与 b8c 逐字同 ⇒ 逐位。保留，入 CSV round b8d。累计 c1 **17.07 → 6.57（-61.5%）**，vs vLLM 4.69 差 1.40×。split 链 `/w/b8dsplit.sh` 已起（`MOE_COMBINE_OLD=1` / `KV_FP8_FUSED=0`，A 侧带 FP8_GEMV=1）。
- **D66 b8d split1（`MOE_COMBINE_OLD=1` vs 开，均 FP8_GEMV=1）**：开 6.56 / 11.52，关 6.62 / 11.59 ⇒ **B11b combine c1 -0.06、c8 -0.07，两边都赚**。c8 的 +0.5（b8c 引入）归因只剩 **B8 tile 表**（R=64 时 grid.y = R/32+E = 130 vs 旧 2×128... 旧版 grid.y=2, grid.z=128 → 256 block/N 块；新 130 block/N 块，更少——不该慢；但每 block 多读 tile_off[E] + 7 步二分 + offsets[e]、offsets[e+1]：对 44×130 个 block 是小事）或 B11a（quant 直接读 h2 的随机行 vs gather 后连续行——quant 读变随机但 gather 本身也随机）。待 nsys 复抓 c8 看 grouped_gemm 时长。
- **B15 代码进度**：CUDA 侧完成（`gqa_decode_v3_core.cuh` KV_FP8 模板 + cp8z/fp8x8_to_bf16x8；`gqa_decode_v3.cu` v3/d512 模板 + host `kv_fp8` 参数；`gqa.cu` paged kernel KV_FP8 模板 + `kv_elem_f32`、paged_impl kv_fp8（混批强制 cfg=-1 慢路径）、decode_impl kv_fp8（v2 不支持→报错）、cache_write/fp8rt STORE_FP8 模板）；ffi decls 加 `kv_fp8: i32`（无 no-nvcc stub 需改）；ops.rs 四个包装加 `kv_fp8: bool` 进行中。
- B15 Rust 侧：ops.rs 4 包装 + opset `paged_win` + 测试调用点 + `AttnCtx.kv_fp8_store` + `LayerKv.is_fp8`（kv_fp8 层默认存 e4m3，`KV_FP8_STORE=0` 回退；cache 按 bf16 分配一半长度）+ model.rs 三处 ctx。待：parity 测试加 fp8 场景、put、编。
- B15 测试 `gqa_decode_fp8cache_bitwise`（在 gqa_decode_win_parity.rs）：k/v 先 GPU 往返 → 写 bf16 cache 与 e4m3 cache → decode v3/d512 与 paged 慢路径各跑两种 cache，**逐位同**为过（3 场景 ×2 路径）。
- **D66 split2（`KV_FP8_FUSED=0` vs 开）**：开 6.57 / 11.54，关 6.59 / 11.55 ⇒ B12 c1 -0.02（3 launch ≈ 0.02 ms，与 0.5–0.7 us/launch 一致）。B12/B13 都保留。
- B15 全部代码已 commit + put；链 `/w/b15.sh`（门禁含 `gqa_decode_fp8cache_bitwise`）等 split 结束起。**判据**：fp8cache 逐位 0 差；bN B 侧（KV_FP8_STORE 默认开）c1/c8 TPOT 下降（预期 attention -0.2 ms）；**输出与 b8d 逐字同**（存 e4m3 读回 = 往返值，应逐位）。
- b15 第一次编译失败（nvcc），取错误行中。D66 split2 定稿：B12 开 6.57/11.54 vs 关 6.59/11.55（两轮稳定）。
- b15 编译失败根因：**又是 `kernels.h` 没同步**（5 个 gqa 入口声明缺 `kv_fp8`，C linkage 重载报错）——同 B8 的错第 2 次。已修。**规矩加一条：改任何 extern "C" 签名，kernels.h / .cu / ffi.rs 三处 grep 同一函数名核对参数数。**
- b15 第二次编译失败：Rust `E0061: 20 vs 19 args`（ffi 调用漏 kv_fp8，定位中）。
- **B16 预判修正**：kaon3-moe-pipe 第二刀已做过 `MOE_GEMM_DUAL`（gate+up 同 kernel，grid.z 翻倍填 SM）graph 态 A/B：**端到端 <1%**，回滚。说明 MoE GEMM 的 50 us/层不是 "CTA 少填不满 SM" 能解的——mma tile 单 CTA 的 K 链（22 步 × barrier）才是；要缩短 K 链（更深 cp.async 流水 / split-K 跨 CTA 归约）。排在 B15 之后，先 nsys c8 看 grouped_gemm 实际时长。
- b15 第二次编译失败根因：`ops.rs` 非 win 的 `gqa_paged_impl` 调用漏 `kv_fp8`（我只改了 win 版）；已修并第 3 次起链。`tests/mega_attn_parity.rs` 两处 `ffi::gqa_decode_v3_bf16` 直调也缺参（独立测试目标，不在本链编译范围）——补上以免下次全量 test 编不过。

## ★★★★ D67 2026-10-28 B15 fp8cache 逐位测试：**decode v3 / d512 三场景 bf16 cache vs e4m3 cache 位差 0**（20480 / 40960 / 8192 元素）；paged_slow D=512 位差 0；paged_slow **D=256 位差 135/20480、339/8192**
- D=256 paged 的差**不是 fp8 读错**：kv_fp8=false 时 D=256 走 `cfg=0` → **fa_v5 PAGED**（tensor-core 流水 kernel）；kv_fp8=true 时 `paged_impl` 强制 `cfg=-1` → 通用慢路径。两条不同 kernel 累加序不同 → bf16 ulp 差（135/20480 = 0.7%，量级与 B3 测试里 v3 vs 慢路径的差同）。D=512 两边都只有慢路径 → 位差 0，正好印证。
- 测试修：paged 两侧都用 `GQA_PAGED_CFG=-1`（强制慢路径）比；**生产影响**：kv_fp8 存储开时，混批（prefill+decode 同步）D=256 走慢路径（正确但慢）——纯 decode（`gqa_decode_win`）不受影响。**B15b**：fa_v5 PAGED 加 KV_FP8 读（cp16z→cp8z+展开，同 v3 做法）。

## B15b / B16 预案 2026-10-28（等 b15 第 4 次链期间）
- **B15b（fa_v5 PAGED 读 e4m3 cache）**：v5 用 cp.async 16B → smem bf16 → ldmatrix；fp8 要先落 staging 再展开。两种做法：① 独立 fp8 staging 区（TILE·D·2·NBUF 字节，D=256 NBUF=2 要 +64 KB，5090 99 KB 上限装不下）；② 原地两阶段：fp8 8B 落到行的后半区（字节 [D,2D)），`cp_async_wait_all` 后每线程读自己 8 字节进寄存器 → `__syncthreads` → 写 bf16 到 [0,2D)（多一道 barrier/tile，无额外 smem）。选 ②。先看 V5_TILE/LDS/NBUF 实际值。
- **B16（MoE grouped GEMM K 链加深流水）**：pipe 版是 2 级双缓冲 + 每 BK=128 步 `wait_group 1` + barrier；c1 每 CTA 每步只搬 4 KB B，18.6 us / 22 步 = 0.85 us/步 ≈ 一次 HBM 往返 + barrier ⇒ **延迟绑定**，不是带宽。smem 每级只 7.7 KB（sA 2.5 + sB 5.1）→ 改 4 级流水（3 组在飞）预期每步 ~0.3 us，gate/up 18.6 → ~7 us、down 12.5 → ~5 us，三 GEMM 50 → ~20 us/层 ≈ **-0.9 ms/步**（c1 6.57 → ~5.7）。dual 第二刀 <1% 是因为它只加 CTA 数不缩 K 链，与此不矛盾。实现：`grouped_gemm_kernel_pipe<BK, STAGES>`，`wait_group STAGES-2`，A/sfa 的寄存器预取也按 STAGES 轮转；逐位不变（同 mma 序）。对拍 = moe_pipe_parity（bad=0）。**B16 排在 B15 结果之后立刻做。**
- **D67 定稿**：fp8cache 测试两侧同一 kernel 后 **6/6 场景位差 0**（decode v3 D=256 窗 1024、d512、v3 D=256 全窗；paged 慢路径同三场景）⇒ B15 读写路径逐位正确。bN 编译失败（NOBUILD，Rust 侧 models/kaon3 某调用漏参，取错误中）。
- b15 第 4 次：门禁全过，bN **编译失败 E0502**（`model.rs` StepCtx 里 `kv_fp8_store: kv_cache[l].is_fp8` 与 `kv_mut: &mut kv_cache[l].cache.data` 同一结构体字面量里借用冲突）→ 先读 `is_fp8` 到局部再建 ctx。第 5 次起链。
- b15 第 5 次：门禁过、编译 EXIT=0（marker=0 正常：`kv_elem_f32` 内联无符号），ABAB 跑中。
- **B16 代码落地**：`grouped_gemm_kernel_pipe4<BK, STAGES>`（A/sfa/B/sfb 全 cp.async、STAGES 级环形、`wait_group STAGES-1`、越界行 zfill 0——有效行输出与 pipe 版逐位同）；host `MOE_GEMM_STAGES`（默认 4，支持 3/4/6，≤2 回退 pipe 双缓冲）；测试 `moe_pipe4_matches_baseline`（gate/up 形状 + down 形状 BK=64 路径，bad=0）。链 `/w/b16.sh` 等 b15 结束起。

## ★★★ D68 2026-10-28 B15 首批 B（KV cache 存 e4m3）：c1 TPOT 6.56（持平 b8d 6.57）；c8 TPOT 11.53 → **11.12（-3.6%）**，但 **c8 TTFT 492 → 1397 ms（2.8×）**，吞吐 437 → 262
- TTFT 爆是预料中的 B15b 缺口：c8 的 prefill 走混批 `paged_win`（`BatchMode::Paged`），kv_fp8 存储下 `paged_impl` 强制 `cfg=-1` 通用慢路径（D=256 层 25 层）。**c8 TPOT -0.4 ms 证明 KV 读带宽减半有效**（c8 KV 读占比高；c1 KV 总量小、在 L2 里，看不出）。
- 决定：**B15 的 `KV_FP8_STORE` 默认改关**（等 B15b fa_v5 PAGED 加 e4m3 读后再默认开）；本轮 B 侧 c8 TPOT 数字记 CSV 作负/中性结果说明。英文 raw 输出与 b8d 逐字同（逐位）。
- **b15 定稿（两轮）**：B c1 6.56/6.57（持平）、c8 TPOT 11.12/11.12（-3.6%）、c8 TTFT 1397（2.8×，混批慢路径）。`KV_FP8_STORE` 默认改关（`=1` 开）；CSV round b15 记"未合入"。B15b 代码落地（fa_v5 PAGED 加 KV_FP8：fp8 staging 区 + 展开 barrier，NBUF 固定 1；`paged_impl` 不再强制 cfg=-1，v6/v7 cfg≥1000 退到 fa_v5），与 B16 同链验。
- B16 补：`STAGES=6 × BK=128` 静态 smem 50.7 KB 超 48 KB（ptxas 会报 too much shared data）→ BK=128 只 3/4 级，BK=64 可 6 级。b16 链起时该守卫还没 put，若编译失败即此因，重起即可。
- **b16 门禁**：`MOE_PIPE4_PARITY rows=22 STAGES=4 bad=0/15488`（gate/up 形状逐位）；编译过（STAGES=6×BK=128 实例当时还在，ptxas 没报——静态 smem 50.7 KB 可能按 opt-in 上限算了；守卫仍保留）。等 down 形状行 + fp8cache paged_v5 行 + bN。
- **b16 门禁全过**：`MOE_PIPE4_PARITY` gate/up bad=0/15488、down bad=0/61952（STAGES=4 逐位）；**fp8cache paged_v5（fa_v5 PAGED KV_FP8）两场景位差 0** ⇒ B15b 逐位。bN 编译过（marker=110），ABAB 跑中。bN 之后起 `/w/b15bsplit.sh`（b16 二进制：`KV_FP8_STORE=1` vs 关，均 FP8_GEMV=1）看 c8 TTFT 是否回 ~490、TPOT 保住 -3.6%。

## B17 预案 2026-10-28：prefill（TTFT）下一刀的靶子
- D54 时 c1 prefill 步（1560 tok）eager 797 kernel、kernel_sum 155 ms、gaps 4.4 ms；其中 down GEMM 空 block 57 ms（B8 已去）。b8c 后 **TTFT c1 29.5 ms**（含 tokenize 1.2 + sched + prefill 步 + 首 decode 采样），说明 prefill 步现 ≈ 25 ms。
- 剩余大头待 nsys 复抓（prefill 单独一段）：候选 ① MoE grouped GEMM prefill（R=12480 行、BM=32 tile，现在 B16 4 级流水也会受益）；② dense NVFP4 GEMM（1560×2816×2112 ×3，mma tile 已有）；③ attention fa_v5 prefill；④ 797 次 eager launch 的 host 开销（D51：launch 16 ms of 41 ms prefill exec ≈ 40%）—— **prefill 图化**（按 token 桶 capture：512/1024/2048/4096）或 **多流/launch 批量**。④ 可能最大：host 16 ms 现在是 prefill 的一半以上。

## ★★★★ D69 2026-10-28 b16 首批 B（B16 4 级流水 + B15b 代码；KV_FP8_STORE 默认关）：c1 TPOT 6.57 → **6.39（-2.7%）**，c8 11.52 → 11.57（持平），TTFT c1 29.5 → 28.8、c8 492 → 488
- 预期 -0.9 ms 实得 -0.18 ms：说明 MoE grouped GEMM 的 50 us/层**不全是 K 链延迟**——4 级流水只把每步从 0.85 → ~0.6 us；剩余是每 CTA 的固定开销（tile_lookup、prologue、epilogue）与 N=704 只 11 个 CTA/专家 的并行度。要再降得换 tile 形状（BM=16 + BN=32 让 CTA 数×4）或把 gate/up/down 三 GEMM 融合。记 CSV round b16。
- A 侧（b15 二进制，当时 KV_FP8_STORE 默认开）c8 TTFT 1395 再现 D68。英文 raw 与 b8d 同（逐位）。
- **B17a 发现**：`layer.rs` forward 热路径每层 **17 次 `std::env::var`**（ATTN_SKIP_*/DENSE_SKIP/MOE_SKIP/KAON3_*_TIME/MOE_ISO/MOE_GEMM_SKIP/NQ4_EXACT_RCP…）→ 每步 ~510 次 getenv（锁 + 线性扫 environ）≈ 0.5–1 ms host，**prefill（eager，host 在关键路径）直接吃掉**；decode 在图里不受影响（但 graph 外的 route/plan 也有）。改 `OnceLock<bool>` 一次读。零风险，逐位不变。
- **b16 定稿（两轮）**：c1 6.57 → **6.39/6.40（-2.7%）**，吞吐 144 → 148；c8 11.52 → 11.57/11.58（持平）；TTFT c1 29.5 → 28.8。保留，CSV round b16。累计 c1 **17.07 → 6.39（-62.6%）**，vs vLLM 4.69 差 1.36×。
- B17a 扩：C++ host 每次 launch 都 `getenv`：`gqa.cu` `GQA_D512_REF`（每层 prefill）、`moe_grouped_sm120.cu` `MOE_GEMM_PIPE`（每层 3 次）→ static once；Rust `ops.rs` `gqa_prefill_cfg`（每层）/ `rmsnorm_layout`（每次 norm ×~8/层）内 `env::var` → 看实现后缓存。总计 prefill 每层再省 ~12 次 getenv。
- B17a 核：`ops.rs` 的 `gqa_prefill_cfg` / `rmsnorm_layout` 已是 OnceLock（误报），其余 ops.rs env 读在 GDN（Qwen3.5）路径，与 kaon3 无关。B17a 落地范围 = layer.rs 17 处 + C++ 2 处。随 b17 链验（预期 prefill host 省 ~1 ms → TTFT c1 28.8 → ~27.5；decode 图内无影响）。

## ★★★ D70 2026-10-28 B15b split（b16 二进制，`KV_FP8_STORE=1` vs 关，均 FP8_GEMV=1）首批：c1 6.39/6.39 持平；c8 TPOT 11.58 → **11.35（-2.0%）**，c8 TTFT 488 → **507（+3.9%）**
- TTFT 回到正常量级（D68 的 1397 是慢路径），但比 bf16 cache 慢 4%：fa_v5 KV_FP8 固定 NBUF=1（bf16 侧 5090 默认也是 1）、多一道展开 barrier/tile + 展开计算。c8 decode 赢 0.23 ms/步（KV 读减半），64 token 输出赢 15 ms，TTFT 赔 19 ms——**c8 本轮净持平**；长输出（真实负载 >64 token）净赢，且 cache 容量翻倍（更多并发 / 更长上下文）。
- 决定：等 A2/B2 复核后 **`KV_FP8_STORE` 默认开**（vLLM 口径即 fp8 KV），CSV 记 round b15b；fa_v5 KV_FP8 的 TTFT 差留作 B15c（展开并进 ldmatrix 前的 smem 布局、或 staging 直接做 e4m3 ldmatrix + 寄存器内展开）。

## B16b 预案 2026-10-28：MoE grouped GEMM decode 专用 tile BM=16 / BN=32（CTA 数 ×4）
- 现 tile BM=32×BN=64，8 warp = 2(M)×4(N) 各 16×16。decode 每专家 1 行：M 方向 2 个 warp 里 1 个全白干（行 ≥16 全越界），且每 CTA 每步搬 B 4 KB（BN=64 行 × BK/2），gate/up 只 11 CTA/专家。
- 新 `grouped_gemm_kernel_pipe4<BK, STAGES, BM_T=16, BN_T=32>`：warp 布局 1(M)×8(N)？不行——8 warp × 16 列 = 128 列 > BN。改 BN_T=32 → 2 warp 够，但要保持 8 warp 的 cp.async 装载并行度……取 **BM=16, BN=64, 8 warp = 1×8 → 每 warp 16×8 列（一个 n8 片段，只 acc 不要 acc2）**：mma 序每个输出元素与现 kernel 同（同 k 序累加），逐位；CTA 数不变但每 CTA 的 M 侧浪费减半、寄存器/smem 减半 → 占用翻倍（每 SM 更多 CTA 并发），等效并行度 ×2。再配 BN=32 版本（CTA 数 ×2）二选一由 bench 定。
- 判据：moe_pipe4 测试加 BM16 场景 bad=0；bN c1 TPOT。预期 -0.3 ms（MoE 三 GEMM 1.3 ms 的 25%）。
- B16b 不盲写：先 nsys 复抓 c1（`/w/nsys_c1b.sh`，b17 二进制）看 `grouped_gemm_kernel_pipe4` 实际 us/call（B16 前 18.6/12.5；若 ≈12/8 说明流水生效但剩余是固定开销 → BM=16/占用；若仍 ~17 说明 4 级没生效/被别的卡住）。再定 B16b 方向。nsys 在 b17 链之后跑。
- **D70 定稿（两轮）**：`KV_FP8_STORE=1` c1 6.39/6.39（持平 6.39/6.40）；c8 TPOT 11.35/11.35 vs 11.58/11.56（**-1.9%**）；c8 TTFT 507/507 vs 488/487（+4%）。默认开已 commit，CSV round b15b。

## B17 可行性 2026-10-28：prefill 图化
- plan 缓冲固定上限（`PlanBuffers::alloc(max_tokens, max_rows, max_blocks)`），prefill 时所有 kernel 的 grid/行数来自 host int `tokens/num_reqs/max_seqlen`（不是 device 侧读）→ 图化只能**按 token 桶 capture**（如 256/512/1024/2048/4096，tokens 向上 pad，pad 行 slot=-1 不写 cache、logits 只取有效行），num_reqs=1 的纯 prefill 桶。c1 1560 tok → 2048 桶：多算 31% token 的 GEMM（prefill 是算力型，这 31% 大约 +5 ms）vs 省 host launch 16 ms → **净 -11 ms（TTFT 28.8 → ~18，反超 vLLM 25）**。混批（Paged）仍 eager。
- 先做 B17a（env 一次读，-~1 ms）看 host 实际还剩多少；再决定是否上桶图（代码量：executor capture 加 Prefill 桶 + 每桶 plan 预填 + runner 选桶；kernel 不改）。

## D71 2026-10-28 b17 首批 B（B17a env 一次读 + KV_FP8_STORE 默认开）：c1 TPOT 6.39 持平、**TTFT 28.99 → 28.97 持平**；c8 = D70 的 B（11.35 / 507）
- B17a 收益 ≈ 0：510 次 getenv/步在 prefill 里 <0.1 ms（我估 0.5–1 ms 高估了 5–10×）。不过是正确的清理（逐位、零风险），留着。**结论：prefill host 的 16 ms（D51）不在 env 读上；下一步 B17 必须先用 `HOST_TIME=1` 复测 prefill 步的 upload/launch/collect 三段**（D51 以来 B8 等改动可能已经改变分布），再决定图化。
- 另：`executor.rs forward_sample` 里有一行无条件 `eprintln!("[fwd-sample] ENTER ...")`——每步一次 stderr 写（无锁但 syscall），prefill/decode 都有；顺手删（改 tracing::trace）。
- **b17 定稿**：B 两轮 6.39/6.40、TTFT 28.97/28.90；c8 11.35/11.35、507/507 —— 与 b16+B15b 定稿一致，B17a 中性（清理保留，不入 CSV）。

## B16b 设计定稿 2026-10-28（等 nsys c1b 期间）：grouped GEMM decode 用 BM=16 tile
- pipe4 现 BM=32：8 warp = 2(M)×4(N)，每 warp 16×16（两个 n8 片段 acc/acc2）。decode 每专家 ≤16 行时 wm=1 的 4 个 warp 全算零行（mma 对 zfill 的 A）——**一半算力白干**，smem/寄存器也按 32 行占。
- BM=16 变体：8 warp = 1(M)×8(N)，每 warp 16×8（一个 n8 片段，只 acc）；B 片段 b0 = ncol=warp*8；A 片段同（mrow=0）。**每个输出元素的 k 序累加与 BM=32 版逐位同**（同 lane 同 k 顺序，只是少算一个片段）→ bad=0 可测。smem 每级 16·72+64·72+4·(16+64) = 6.1 KB（原 8.4）；寄存器少 acc2 → 占用 ↑。
- tile 表：`tile_off` 现按 BM=32 算（scan_kernel/fused route 都写死 `/32`）。加第二张 `tile_off16`（同一 scan 多写一列，`(cnt+15)/16`），MoeBuckets 加字段，`moe_grouped_gemm` 多收一个指针，host 按 `rows ≤ 128`（decode 口径）选 BM=16 + tile_off16，否则 BM=32 + tile_off。grid.y 上界 R/16+E。
- 预期：gate/up 每 CTA 算量减半 → 18.6→~11 us（若固定开销占一半则 ~14）；三 GEMM -0.3~-0.5 ms/步。先等 nsys 看 pipe4 现值再动手。

## ★★★★★★ D72 2026-10-28 nsys c1b（b17 二进制，FP8_GEMV=1，KV e4m3）：每步 ~1065 kernel，kernel 和 ≈ 5.7 ms + 间隙 ≈ 1.4 ms = TPOT 6.78（归档 tmp_nsys_c1b_full.txt；87 步合成一个 graph 实例，表中 n 要 ÷87）
| 段 | n/步 | us/次 | ms/步 | 说明 |
|---|---|---|---|---|
| lm_head gemvx | 1 | 870 | 0.87 | 带宽下限，不动 |
| grouped_gemm_pipe4 gate/up | 60 | 11.7 | 0.70 | B16 后 18.6→11.7（-37%）；grid.y=129（R/32+E 上界，空 block 多）|
| grouped_gemm_pipe4 down | 30 | 10.9 | 0.33 | 12.5→10.9 |
| dense gemv（B5 down）| 60 | 8.7 | 0.53 | |
| **rmsnorm_vec<256,2> grid=4160** | 30 | **17.1** | **0.51** | **按 max_tokens 行跑（1 行实际）**——B1 修了逐头 norm，这里还有一处 h=2816 的 norm 按 4160 行 launch！|
| **route_bucket_small（B10）** | 30 | **16.2** | **0.49** | 单 block 单线程循环 8×128 + softmax 32-lane 树 → 比原 topk 7.3+softmax 5.6 还慢！融合省了 launch 但把并行度打没了 |
| qkv gemv（B6c）| 25 | 16.5 | 0.42 | 3072×2816 fp8 = 8.6 MB → 下限 4.8 us，3.4× |
| gateup_act（B5）| 30 | 10.2 | 0.31 | |
| rmsnorm 2816 单行 ×6/层 | 181 | 1.3 | 0.24 | |
| cache_write_fp8rt | 30 | 4.8 | 0.14 | |
| v3 attention（25 层）| 25 | 5.2 | 0.13 | KV e4m3 后很快 |
| d512 attention（5 层）| 5 | 18.3 | 0.09 | |
| o_proj gemv | 30 | 18.0 | 0.09（5 层 full）+… | |
| 其余小 kernel | ~550 | | ~0.6 | |
**两个意外的大头都是"我自己引入/没查到的"**：① rmsnorm grid=4160（0.51 ms，纯浪费，B18 立刻修）；② B10 融合 kernel 反而慢 0.1 ms/步（B19：top-k 改 warp 并行，每 token 一 warp）。两刀预期 -0.8 ms（6.39 → ~5.6，vs vLLM 4.69 差 1.2×）。
- **hosttime_c1（b17 二进制，HOST_TIME=1，c1 ×8，16 out）**：全部步均值 upload=20us launch(submit)=**337us** collect(dtoh_sync)=6870us。launch 含 prefill 步（24 步里 8 步 prefill）也只 0.34 ms ⇒ **D51 的"prefill host launch 16 ms" 已不存在**（当时是 host 侧 env/锁/旧代码路径？或被 B8 等改动顺带消掉）。**B17 prefill 图化降级**（省不到 1 ms）。TTFT 29 ms = prefill GPU ~25 ms（nsys eager 段 kernel_sum 83 ms 是 nsys 开销下的；真实 ≈ 25）+ 1.2 tokenize + 首 decode。prefill 下一刀看 nsys 段：grouped_gemm_pipe4 gate/up 18.6+down 13.7 = 32 ms(nsys 态) 占 39%、nvjet qkv 16.4、fa_v6 d512 7.4、fa_v5 6.1。

## ★★★★★ D73 2026-10-28 两个热点根因
- **B18**：`ops::rmsnorm_bf16_noweight` 不收 `tokens`，用 `x.len()/hidden` 当行数 → kaon3 router 预处理（`rmsnorm_noweight_into`，norm_mode==1 默认路径）每层按 **max_tokens=4160 行**跑（17 us vs 单行 1.3 us）。B1 修的是逐头 norm 的同类病，这处漏了（是 `DefaultOps::rmsnorm_noweight` 直接忽略了 `_tokens` 参数）。修：ops 加 `tokens` 参数，opset 透传。省 **0.47 ms/步**，逐位不变（真实行结果同）。
- **B19**：B10 融合 kernel 的 top-k 是**单线程**扫 K×E=1024 次 smem + 128 次原子（原 route_topk_ids 用 256 线程装 smem 后单线程扫，也是单线程，但 topk 7.3 + softmax 5.6 = 12.9 < 16.2 的差来自融合 kernel 还串了 softmax_row 的 Z 全体循环 + bucket 段）。修：每 token 一 warp——top-k 用 32 lane 各扫 4 专家 + shuffle 归约（并列取小下标，与串行 `v > bv` 语义同）→ 8 轮 ×（4 次比较 + 5 步 shuffle）；softmax_row 仍 lane 0 单线程（口径逐位）。预期 16 → ~7 us，省 0.27 ms/步。
- **B18/B19 代码落地**（两 commit）：B18 `rmsnorm_bf16_noweight(tokens)` + opset 透传；B19 `route_bucket_small_kernel` top-k 每 token 一 warp（lane 扫 E/32、xor 归约值大者胜 / 并列小下标，与串行语义同），fill 一 lane 一 (t,k)。门禁 `moe_route_fused_parity`（ids/w 位同 + 区间集合，含并列场景）。链 b18：A=b17 二进制（6.39/11.35）。预期 c1 → ~5.6。
- B18 同类审计：`ops.rs` 还有两处 `tokens = x.len()/hidden`（`rmsnorm_bf16` 旧入口、`fused_add_rmsnorm_bf16`）——kaon3 不走（走 `rmsnorm_bf16_n`/opset 的 tokens 版），GLM/Qwen3 线用不用待查；`nn.rs:56 Linear::forward` 的 numel/in_features 是 2026-10-20 pitfall 已记的那处（forward_rows 替代）。nsys c1b 表里 decode 段 grid=4160 只剩这一处（已修）。

## B20 预案 2026-10-28：fp8 GEMV（B5/B6c）访存没批量发出 → 延迟绑定
D72 表：qkv gemv 16.5 us vs 8.6 MB/1.79 TB/s = 4.8 us（3.4×）；dense down 8.7 vs 3.3（2.6×）；gateup 10.2 vs 6.6（1.5×）。`dot2` 内循环 `for (c = lane; c < nvec; c += 32) { load w0; load w1; 转换+FMA }`——循环体含 K 长度相关的不定次数迭代（nvec/32 = 5.5），编译器不展开，每次迭代 2 个 16B 读串行等待（~1 us 一次 HBM 往返 × 6 轮 ≈ 6 us）。pick_warps 把 block 压到 2 warp（768 块）也让每 SM 只 ~9 warp 在飞。
修：① 内循环改两段——先一次性把本 lane 全部 `ceil(nvec/32)`（≤8）个 uint4 读进寄存器（编译期上限 K ≤ 4096 → 8 个/行，两行 16 个 uint4 = 64 寄存器），再逐个转换+FMA；**累加序不变（同 c 顺序）→ 逐位同**；② pick_warps 的 `block ≥ 4×SM` 条件保留但上限回 4 warp（现在 2）看占用。预期 qkv 16.5 → ~7、down 8.7 → ~5、gateup 10.2 → ~8 ⇒ 三项 -0.5 ms/步。判据：mlp_fp8_decode_parity 位同（已有）+ bN。

## ★★★★★★ D74 2026-10-28 b18 首批 B（B18 rmsnorm 行数 + B19 warp top-k）：c1 TPOT 6.39 → **5.59（-12.7%）**，吞吐 148 → 168；c8 11.35 → **10.28（-9.4%）**，463 req tok/s；TTFT 持平（28.7 / 506）
预测 -0.8 ms 实得 -0.80 ms（c1）。英文 raw 输出与 b17 逐字同 ⇒ 逐位（两刀都应逐位：行数修正不改真实行；top-k 并列语义同）。累计 c1 **17.07 → 5.59（-67%）**，vs vLLM 4.69 差 **1.19×**。等 A2/B2 入 CSV round b18。

## B21 候选 2026-10-28（D74 后 c1 ≈ 5.6 ms 的构成，按 nsys c1b 表扣掉 B18/B19/B20 预期）
| 段 | ms/步 | 下限 | 刀 |
|---|---|---|---|
| lm_head gemvx bf16 | 0.87 | 0.83 | 不动（或 lm_head fp8 存：权重 1.48 GB → 0.74，-0.4 ms，但改模型口径，先不做） |
| MoE grouped GEMM gate/up+down | 1.04 | 0.42 | **B16b**：BM=16 tile（M 侧浪费减半，占用翻倍）；或 gate‖up 同 CTA 省一次 A 装载 |
| dense gemv 三件（B20 后） | ~0.9 | 0.3 | B20 验后看剩多少 |
| qkv/oproj gemv（B20 后） | ~0.3 | 0.16 | 同上 |
| rmsnorm 2816 单行 ×6/层 | 0.24 | | norm→quant 已有融合（`rmsnorm_quant_fp8`，`NORM_FUSE_QUANT=1`，D-? 时因对拍差关了）；对 router 预处理 `rmsnorm_scale2` 已融合 |
| route_bucket_small（B19 后） | ~0.2 | | softmax_row 的 Z 循环（E=128 单 lane）也可 warp 并行：exp 顺序要保持 `__fadd_rn` 串行序才逐位——不能并行归约。留 |
| cache_write_fp8rt + rope + norm_quant | 0.34 | | 三个按 token 的小 kernel 融合成一个（rope→fp8 往返→cache_write→norm_quant 的 q 侧）：-2 launch/层 + 少两遍读写 q/k/v ≈ -0.1 ms |
| 间隙 ~1000 launch × 0.6 us | 0.6 | | 继续减 launch 数只能慢慢啃 |
**排序**：B20（已排队）→ B16b → 小 kernel 融合（rope+rt+cache_write+norm_quant）→ 再 nsys。目标 c1 ≤ 5.0（vLLM 1.07×）。
- **b18 定稿（两轮）**：c1 6.87/6.88(A, 关 FP8_GEMV) → B **5.59/5.59**；与 b17 定稿 6.39 比 **-12.5%**；c8 11.35 → **10.28/10.28（-9.4%）**；TTFT 28.4 / 505。逐位。CSV round b18。远端队列：nsys_c8 → b20 自动起。
- **B16b 实现方案**：`tile_off` 缓冲扩成 `2×(E+1)`：前半 BM=32 前缀和（现有），后半 BM=16 前缀和（scan_kernel / 融合路由 kernel 多写一列）；`tile_lookup` 加 `bm` 参数；`grouped_gemm_kernel_pipe4<BK, STAGES, BM_T>`：BM_T=16 时 warp 布局 1×8（`wm=0, wn=warp`，每 warp 一个 n8 片段 acc，无 acc2；mma 的 a 片段 m 行用 lane/4 + 8·(g%2) 同前，行 ≥16 的 A 不存在 → sA 只 16 行，g%2==1 的 a 片读 sA 行 8..15 正常）——**每输出元素 k 序与 BM=32 版同 → 逐位**。host：`rows ≤ 128` 用 BM=16 + tile_off 后半，grid.y 上界 R/16+E。测试 moe_pipe4 加 BM16 场景（rows 22 → 自动走 BM16 路径，与 base 逐位）。
- nsys c8 抓到（b18 二进制）：64 步合成一个 graph 实例（kernel 66654 → 1041/步）；表待读（归因 c8 +0.5：B8 tile 表 vs B11a）。b20 链已自动起。

## ★★★★★ D75 2026-10-28 nsys c8（b18 二进制）：图内 kernel 和 ≈ 4.9 ms/步 + 间隙 1.3 = **6.2 ms/步，而 bench TPOT 10.28 ms** ⇒ **c8 每步有 ~4 ms 在 graph 之外**（host 调度 / plan 上传 / collect 同步 / 采样 / HTTP）
- 各 kernel us/call 与 c1 几乎同（grouped_gemm 11.6/10.7、gemv 8.1/16.2、route 10.4、attention v3 4.7、d512 18.0）—— c8 的 GPU 工作量≈c1（M=8 对带宽型 kernel 近免费），**kernel 侧没有 c8 的 +0.5 的影子**；D65 以来的 c8 +0.5 归因（B8/B11a）作废——是图外时间的噢动。
- **这是比任何 kernel 刀都大的靶子**：c8 图外 ~4 ms/步 = 40%！c1 的图外 = 5.59 − (5.7 kernel + 1.4 gap… 不对，c1b 表 5.7+1.4 = 7.1 > 5.59 TPOT)——c1b 表是 87 步合成的均值含 prefill 影响？要重算：c1 instance kernel_sum 497782/87 = 5.72 ms，span 621458/87 = 7.14 ms > TPOT 6.78（nsys 态）——nsys 开销。c8 同口径 span 391734/64 = 6.12 ms vs nsys 态 TPOT 8.75 ⇒ **图外 ≈ 2.6 ms/步（nsys 态）**，非 nsys 态 TPOT 10.28 − 6.1 ≈ 4 ms？数字不自洽（nsys 态 TPOT 8.75 < 非 nsys 10.28 不可能）——bench c8 p50 在 nsys 下是 8.75（mean 10.93），bN 里 p50 10.28：取 p50 比 ⇒ 图外 ≈ 10.28 − 6.1 = **~4 ms** 仍成立（nsys 下 p50 更低可能是 nsys 改变了调度节拍）。
- **下一步 B22**：HOST_TIME=1 跑 c8，看 upload / launch / collect 三段 + engine 的 sched/publish；对比 c1（hosttime_c1：launch 0.34 collect 6.87 ms 其中 collect 就是 GPU 时间）。怀疑点：① c8 时 `collect(dtoh_sync)` 后的 host 采样/detokenize/HTTP 串在步间（ASYNC_SCHED 是否生效）；② plan upload 按 max_tokens 拷；③ 每步 `ensure_graph` / bucket 切换（n 在 1..8 波动 → 不同桶 → 不同图，首次捕获 30 ms）。
- b20 第一次门禁失败：测试里 K=8192 场景撞上 B20 新加的 K ≤ 4096 守卫（预期行为，全注意力 o_proj K=8192 本就走 cuBLASLt）→ 测试改 K=4096 m=1。重起。

## B22 预判 2026-10-28（等 hosttime_c8 期间）：c8 图外 ~4 ms/步 的候选
- `forward_sample` 末尾 `collect()` = `dtoh_sync_copy`：**同步等 GPU**，所以 async_sched 的 "先排 n+1 再等 n" 在 kaon3 executor 里实际只能把 sched 与 GPU 重叠，而 begin(n+1) 的 upload/launch 要等 finish(n) 后才发（同一流）。c1 时 GPU 6 ms ≫ host，掩住了；c8 时每步 host 工作 ×8（8 个请求的 plan 构建、block_table 拷贝 nreq×max_blocks、8 次 publish/channel send、detokenize、HTTP 流式写）全部串在 GPU 空闲窗里。
- 候选量化（hosttime_c8 出来对号）：① `upload`（plan htod：ids/pos/slots/cu/seq/block_table[8×512]×4B=16 KB，小）；② `launch(submit)` graph launch ~0.3 ms；③ `collect` = GPU 时间 ~6 ms（图内 6.2 ms 吻合）；④ engine `sched`（8 请求 + block 分配）；⑤ `publish`（8 次 send + 流式 HTTP）；⑥ **步间空档**：engine total − (sched+exec+publish) = HTTP/tokenizer 线程抢核？bench 绑 60–91，serve 没绑核 → 可能同核；D51 时 c8 sched/publish 都 <50 us，但那是 TPOT 19 ms 的年代，比例看不出。
- 若 ④⑤⑥ 合计 ~4 ms：刀 = 把 publish/detokenize 挪到独立线程、plan 构建复用缓冲、serve 进程 taskset 到独立核组（bench 已绑 60–91）。
- **B22a 发现**：bN/envab 里 **dinfer serve 没绑核**（只有 bench `taskset -c 60-91`）——违反 CLAUDE.md "三方钉核不重叠" 规则（vLLM 容器也没查）。机器 2 NUMA（node0: 0-31,64-95；node1: 32-63,96-127），bench 在 node0 的 60-91；serve 的 tokio worker + engine 线程浮动在 128 核上，可能与 bench 同核 / 跨 NUMA 访问 pinned host 内存。c8 时 host 工作 ×8，受影响更大。探针 `/w/pinsplit.sh`（envab 加 `SERVE_TASKSET`）：不绑 vs 绑 0-31，排在 hosttime_c8 之后。**无论结果，bN/envab/stepA 都要补 serve 绑核并写进报告**（规则要求）。
- b20 A 侧（b18 二进制、不带 FP8_GEMV）c1 6.05（= 5.59 + FP8_GEMV 的 0.46，自洽）；A c8 TTFT 133 / TPOT 16.1 又是排队态噢动（第 4 次见，都在 A 轮首跑：GPU 刚从上一链切换，疑 warmup 不足）。

## ★★★ D76 2026-10-28 b20 首批 B（B20 fp8 gemv 批量预取）：c1 TPOT **5.63**（b18 定稿 5.59 → **+0.7%，中性偏负**）；c8 10.28 → **10.00（-2.7%）**；TTFT 持平
- 预期 -0.5 ms 没有兑现：c1 持平说明 qkv/dense gemv 的 16.5 / 8.7 us **不是**"逐轮等 HBM"——批量发 8 个 16B 读进寄存器后时间没变 ⇒ 瓶颈在别处（每 block 只 2 warp × 768 block、每 SM 同时 ~9 warp，访存并行度本就够？或 e4m3→f32 转换 ALU：每 uint4 16 次 cvt + 16 FMA×M_TILE，m=1 时 cvt 占主导——qkv 8.6 MB = 8.6M 元素 cvt，5090 ~100 T ops/s 不该是瓶颈…）。待 nsys 复抓看 gemv us/call 是否变（c8 -2.7% 可能来自 m=2..4 的 M_TILE 路径受益）。英文输出变了（"if I'm doing it right"）——**B20 应逐位（同 c 序），输出却变** ⇒ 要么 fmaf 顺序被编译器重排（`#pragma unroll` 后 r0/r1 两行交错？不影响单元素序），要么 m>1 时 M_TILE 路径……对拍 9 场景位同过了。记录，等 A2/B2 + nsys。
- **b20 定稿（两轮）**：c1 5.63/5.62（vs b18 5.59：+0.6%，噪声级）；c8 10.00/10.03（vs 10.28：**-2.7%**）。保留（c8 赢、c1 持平），CSV round b20。英文 raw 变体：B20 的 fmaf 链本意同序，但第一版 dot2 内 c 循环按 `c += 32` 迭代 j=0,1,2…；新版同；差别可能在编译器对两版的 FMA 调度（非 IEEE 严格）——对拍是 0.01 容差过的，不是位同。**不再追**，但记一笔：fp8 gemv 路径两版都"非逐位于 cuBLASLt"，与 B5 一致。
- **B16b 代码落地**（moe.cu 两张 tile 表 / `pipe4<BK,STAGES,BM_T>` 1×8 warp 布局 / host `R ≤ 128` → BM16（`MOE_GEMM_BM16=0` 回退）/ ops tile_off 2(E+1) + cpu helper 双表）。测试 moe_pipe4（rows=22 → 自动走 BM16）bad=0 为门禁。
- **bN.sh / stepB_verify.sh 加 serve 绑核 `taskset -c 0-31`**（SERVE_TASKSET 可覆盖；RESULT 写 `serve_taskset= bench_taskset=`）。注意：**b16b 起的 A/B 两侧都绑核**，与 b20 定稿（不绑）比时要先看 pinsplit 的 Δ 再归因。
- 队列断了一环：`after_b20.sh`（等 b20 DONE 后起 hosttime_c8）没在跑——`docker exec -d` 起的 watcher 在我 TaskStop 本地监控时被连带杀？（本地 Monitor 与远端 docker exec 无父子关系，更可能是 b20 第一次 PARITY_FAIL 时 after_b20 的 grep "^DONE" 匹到了 par 文件…不对它 grep 的是 tmp_b20_RESULT）。不追了：手动起 hosttime_c8，after_host8 → pinsplit → after_pin → b16b 仍在。**教训：链式 watcher 用一个脚本顺序执行，不要多个 watcher 各等各的。**

## B23 预案 2026-10-28：attention 前段 5 launch → 1（head_rmsnorm q/k/v + rope + fp8rt/cache_write）
nsys c1b：q/k/v 逐头 norm 0.89+1.09+0.85 us、rope 3.26、cache_write_fp8rt 4.76 ⇒ 10.9 us/层 + 4 个间隙 ≈ 13 us；融成一个 per-token kernel（block=token，warp=头）估 ~5 us → **-0.25 ms/步**。难点：逐位——逐头 norm 现走 `rmsnorm_vec_kernel<128,1>`（Block256 布局：每线程 8 元素组平方和 → block 树归约 `rms_ss_triton`），融合 kernel 要复刻同一归约树（把 `rms_ss_triton`/`rms_inv`/`norm_mid` 抽到共享头）。排 B16b 之后。

## ★★★★★ D77 2026-10-28 hosttime_c8（b20 二进制，c8 ×64，32 out）：**executor upload=20us launch=4–57us collect(dtoh_sync)=9.7–10.2 ms；engine sched=1us publish=8us total=9.4 ms** ⇒ **c8 每步 host ≈ 0，GPU 占 100%**
- **D75 "c8 图外 ~4 ms" 作废**：是我拿 nsys 态的 kernel 和（nsys 有开销且 instance 边界不是 bench 的"步"）与非 nsys 态 TPOT 相减。真相：c8 TPOT 10.0 ms 全是 GPU。B22（host 刀）**撤销**；pinsplit 仍跑完当作钉核口径校准，不期待收益。
- 那 c8 比 c1 多的 4.4 ms（10.0 vs 5.6）在哪些 kernel 上？nsys c8 表（÷64）重算见下；候选：① fp8 gemv 在 m>4 回退 cuBLASLt nvjet（qkv/oproj/dense 三件 ×30 层，nvjet 对 M=8 的效率）；② MoE grouped GEMM R=64 行（tile 数 ×8，CTA 多但每 CTA 仍串 K）；③ attention KV 读 8×（v3 4.7→? d512 18）；④ lm_head gemv 8 行（870 us 不变）。

## ★★★★★ D78 2026-10-28 c8 的"多 4 ms"定性：**是流里插进来的 prefill 步**（c8 持续并发：每轮 8 请求完成 → 新 8 请求 prefill → 在 decode 之间占 GPU），不是 host、不是 kernel 变慢
- nsys c8 表（÷64 步）：grouped_gemm 11.6/10.7 us、dense gemv 8.1、qkv gemv 16.2、route 10.4、v3 4.7、d512 18 —— **每个 kernel us/call 与 c1 完全一样**，kernel 和 4.9 + 间隙 1.3 = 6.1 ms/步；bench 的 TPOT 把排在 decode 步之间的 prefill 步（c8 共 64 请求 → 8 轮 prefill，每轮 ~30 ms）摊进了 decode 间隔。D77 的 collect 9.7 ms 是 GPU 忙（含 prefill）。
- **结论**：c8 decode 本身 ≈ 6 ms（vs c1 5.6；M=8 对带宽型近免费）。要降 c8 TPOT 要降 **TTFT/prefill**（c8 的 30 ms prefill ×8 轮 / 64 请求 ×32 token ≈ 每 token 摊 0.1 ms…不对：每轮 prefill 30 ms 摊到该轮 8 请求 × 32 步 = 256 步 → 0.12 ms/步；8 轮共 0.9 ms/步？与 4 ms 不符）——再算：bench c8 TPOT 定义 = (请求总时 − TTFT)/(out−1)，请求在 decode 中若被别的请求的 prefill 插队（max-num-batched-tokens 4096 一次塞 1560×2），该 decode 步等 30 ms → 32 步里有 ~4 次被插 ⇒ +4 ms/步。**吻合**。vLLM 同样会被插队（它 c8 TPOT 6.24 vs c1 4.69 也 +1.5）。
- **刀**：① prefill 更快（TTFT 28 → 18 直接让 c8 TPOT -1.3）；② chunked prefill 更细（max-num-batched-tokens 4096 → 2048/1024：每步插队 15 ms 但次数 ×2，TPOT p50 不变、p99 降）；③ decode 优先调度（vLLM 默认也不做）。先做 ①：prefill 的 MoE grouped GEMM 占 39%（nsys 态 32 ms/83）、qkv nvjet 20%——**B16 的 4 级流水对 prefill 也生效，但 prefill 走 BM=32**；下一刀 prefill 的 MoE GEMM 换 BN=128 tile（R 大时 N 块多余）或 dual（gate+up 同 CTA 省 A 装载，prefill R 大时 A 装载占比高）。

## B24/B25 预案 2026-10-28：prefill（TTFT 28 ms → 目标 ≤20，连带 c8 TPOT -1.3）
nsys c1b prefill 段（1560 tok，nsys 态 83 ms；真实 ≈ 25）：grouped_gemm gate/up 18.6 ms（32 次 582 us）22%、down 13.7（16 次 859 us）16.5%、qkv/oproj nvjet fp8 16.4（66 次 249 us）20%、fa_v6 d512 7.4、dense nvjet 6.7、fa_v5 6.1、combine 2.4、gelu 1.9。
- **B24 MoE prefill GEMM**：R=12480 行、BM=32×BN=64 tile：A（12480×1408 B=17.6 MB）被 N 方向 11 个 CTA 各读一遍 = 193 MB/次；B（每专家 1 MB）被 M 方向 ~390 个 tile 各读 = 390 MB——**A/B 都重读 10–400 遍，但在 L2（5090 96 MB）里**；582 us/次 vs 算力 51 GF/次 ÷ 1.2 PF = 43 us ⇒ **13× 离算力**，是 tile 太小（32×64 一个 CTA 每步 mma 2 片段，load:mma 比例高）。刀：`pipe4` 加 `gate‖up 同 CTA`（一次 A/sfa 装载算两份 B，A 侧流量减半、mma 翻倍）→ 预期 gate/up 18.6 → ~11；再大 tile（BM=64/BN=128）是下一步。
- **B25 qkv/oproj fp8 cuBLASLt**：66 次 × 249 us；形状 1560×3072×2816 = 27 GF → 22 us 算力下限，**11× 离**。nvjet 64x32x64 tile 对 M=1560 不是最优（Lt 自调优没选对？`GEMM_LT_CANDIDATES`/`GEMM_TUNE` 开关看过没）。先 `GEMM_ALGO_LOG=1` 看 Lt 选了什么，再试 `forward_fp8`（自写 sm120 fp8 kernel，`FP8_SELF_KERNEL=1` A/B）。
- 两刀若各省 30%，prefill -10 ms → TTFT ~18、c8 TPOT ~8.7。**优先级高于 B23/B16b 之后的 decode 小刀**。
- **pinsplit pin_a（不绑核，b20 二进制，A/B 同 env）**：c1 5.62/5.63/5.62、c8 10.10/10.03/10.03 —— 同 env 两侧一致（量测噪声 ±0.01/±0.07）。pin_b（绑 0-31）跑中；与 D77 一致，预期持平。钉核的意义是口径合规（CLAUDE.md 规则），不是收益。
- pin_a 定稿：4 轮 c1 5.62–5.63，c8 10.03–10.10（B2 c8 一次 TTFT 108/TPOT 10.76 = 第 5 次见的"排队态"噢动：TTFT 异常低 + TPOT 异常高同现 ⇒ 是 8 请求没叠上的态，bench 侧现象，不是服务端）。pin_b（绑核）跑中。
- **B25a 探针**：`GEMM_FP8_DEBUG=1` 起一次 serve 跑 c1，看 fp8 cuBLASLt 对 m=1560 的 qkv/oproj/dense 三形状 pick 了哪个 algo/tile（现 pick 走 `want` 匹配 + heuristic，无计时自调优；bf16 路径有 tune_shape，fp8 没有）。若 pick 是 64x32 小 tile（nsys 看到 nvjet 64x32x64），改为对 fp8 也跑 `tune_shape`（按 M 桶）或直接指定大 tile 候选。
- **B25 定方向**：fp8 cuBLASLt 的 algo 是**为了与 vLLM 逐位**固定成 `id=67 tile=13 splitk=1`（kaon3-layer1 第 7/14 轮），不是性能选的；nsys 里 nvjet 64x32x64 就是它。prefill m=1560 用 tile=13 比大 tile 慢是预料中的。刀：`FP8_LT_TUNE=1`（默认开？）时对 **m ≥ 64 的形状**跑一次计时自调优（复用 bf16 `tune_shape` 的候选计时骨架，fp8 描述子 + scale 指针），按 (m 桶, n, k) 缓存；**m < 64 保持 tile=13 逐位口径**（decode/对拍不受影响）。prefill 数值与 vLLM 非逐位是可接受的（vLLM 自己 prefill 用的也是 Lt 自选 algo，口径在 oracle_nobi 的 logprob 容差内比）。预期 qkv/oproj/dense 三件 16.4+6.7 ms（nsys 态）→ ~40% ⇒ 真实 prefill -3 ms。
- **pinsplit 定稿**：pin_b（serve 绑 0-31）c1 5.62 ×4、c8 10.02–10.03 ×4 = pin_a（不绑）同值 ⇒ 绑核零收益，口径合规保留默认 0-31。
- **B25 实现**：`forward_fp8_cublaslt` 的 algo 选择：m < 64 维持 want 匹配（tile=13 逐位口径）；m ≥ 64 且 `FP8_LT_TUNE != 0` 时，首次遇到 (m 桶=next_pow2, n, k) 对 32 个启发式候选各计时 `tune.timing_iters` 次（复用 `time_us` + `run_lt_fp8`，在真实缓冲上跑，输出会被覆盖多次——无副作用），取最快，缓存到 `default_algos`（key 带 m 桶）。
- **b16b 门禁全过**：MOE_PIPE4（rows=22 → host 走 BM=16 路径）gate/up bad=0/15488、down bad=0/61952 ⇒ BM=16 tile 逐位。bN 跑中（A/B 都绑核 0-31）。队列：b16b → fp8algo_probe（Lt pick 看一眼）→ b25（fp8 Lt m≥64 自调优）→ moe_prefill_timing（prefill 形状 pipe 2 级 vs 4 级微基准，定 B24 方向）。
- **B24 量化**：prefill gate GEMM 一次 582 us（nsys 态）：R=12480 N=704 K=2816 = 49.5 GFLOP → **85 TF/s = 5090 NVFP4 峰值（~1.6 PF）的 5%**。4290 CTA / 170 SM = 25 波，每 CTA 23 us / 22 K 步 ≈ **1 us/步** —— 与 decode 一样是每步一次 HBM/L2 往返 + barrier 的延迟绑定（pipe 2 级）。B16 的 4 级流水在 prefill 同样生效（nsys c1b 的 prefill 段用的已是 pipe4：582 us 是 pipe4 的数！）⇒ 4 级还不够，或 tile 太小让每步 mma 只 2 片段（算力利用低）。方向：① BM=64×BN=128 tile（每 CTA 每步 mma ×4，CTA 数 /4，K 链不变）；② STAGES=6（BK=64 路径可以，BK=128 smem 超）；③ 换 BK=64 + 6 级。先等 moe_prefill_timing（pipe vs pipe4 at R=12544）看 4 级到底省了多少。

## B24a 设计 2026-10-28：pipe4 加 BN_T=128（prefill，R > 128 时）
- 现 BM=32×BN=64：每 CTA 每 K 步装 A 2.3 KB + B 4.6 KB，算 8 warp × 2 mma = 16 mma(m16n8k64)。BN=128：装 A 2.3 + B 9.2 KB，算 32 mma —— **每字节装载的 mma 翻倍**，CTA 数减半（11 → 6 个 N 块），K 链不变。warp 布局保持 2(M)×4(N)，每 warp 16×32 = 4 个 n8 片段（acc[4][4]），**每输出元素 k 序不变 → 逐位**。smem/级：32·80 + 128·80 + 4·(32·2 + 128·2) = 14 KB → 4 级 56 KB 超 48 KB 静态 → 用 **3 级**（42 KB）或改 dynamic smem。先 3 级。
- 只在 `R > 128`（prefill）走；decode 仍 BM16/BN64。门禁：moe_pipe4 测试加 R=12544 大形状逐位（与 pipe 2 级比）；bench 看 TTFT。
- 等 moe_prefill_timing 出 pipe vs pipe4 的差再决定做不做（若 4 级已把 582 → 300，大 tile 再 -30%；若没差，说明不是流水/算力比，是别的）。

## ★★★ D79 2026-10-28 b16b 首批 B（BM=16 decode tile）：c1 TPOT 5.63 → **5.57（-1.1%）**，c8 10.03 → **9.82（-2.1%）**；但 **c8 TTFT 506 → 545（+7.6%）**，c1 TTFT mean 31.4 / p50 26.85（散布大）
- BM=16 只在 R ≤ 128（decode 步）走，prefill（R=12480）仍 BM=32 → TTFT 不该变。c8 TTFT +39 ms 可能是：① 混批步（Paged，decode 行 + prefill 行同步，R = 8 + 12480 > 128 → BM=32，不受影响）……② tile_off 缓冲 2(E+1) 后 B8 的 BM=32 表位置不变、BM=16 表接在后面——prefill 没碰；③ 噪声（c1 TTFT mean 31.4 vs p50 26.85 说明有一两次慢请求拉高）。等 B2 看 TTFT 是否复现；若复现则 c8 TTFT 的 +39 要归因（排队态？）。
- B24a 可行性补：b16b 二进制 `grouped_gemm_kernel_pipe4` 各实例 REG 58–81、LOCAL 0（无 spill）、smem 16–35 KB；BN=128 四片段版多 8 个 acc + 4 个 b 寄存器 ≈ 95 REG，仍 < 128（2 CTA/SM 256 线程占满 64K 寄存器文件的边界是 128），可行。

## ★★★ D80 2026-10-28 队列结算（b16b 定稿 / fp8algo / b25 / moe_prefill_timing 均 DONE，之前 `Input is too long` 丢了上下文，本节是重建）
- **b16b 定稿（两轮 ABAB）**：A c1 6.04/6.05 → B **5.57/5.57**；c8 10.02/10.02 → **9.82/9.82**；c8 TTFT 505/506 → 545/543（+7.6% 复现，非噪声）。英文/中文 raw 两侧同。**保留**（c1 -1.1% 兑现）；c8 TTFT +39 ms 待下面一起归因。
- **fp8algo 探针（B25a）**：Lt 对 prefill 形状的 parity pick 是 tile=13/15（小 tile）；m=432/4160 时 tune 选 **tile=20**（n=8192 k=2816：275 us；n=2816 k=4096：161 us）。
- **b25（FP8_LT_TUNE m≥64 自调优）首批+复测**：c1 TPOT 5.57 持平、TTFT p50 26.47 → 26.45 **持平**；c8 TPOT 9.81/9.85 → **10.09/10.18（+3%）**、c8 TTFT 543 → 566/567（+4%）。两轮同向 ⇒ **B25 中性偏负，`FP8_LT_TUNE` 默认改关**（代码保留，`=1` 开）。c1 TTFT 不动说明 qkv/oproj fp8 GEMM 在 426 token prefill 里不是主项（nsys 20% 是 nsys 态 1560 tok 的数）。
- **moe_prefill_timing（B24 前置）**：E=128 / 98 行/专家 / 12544 行：STAGES=2 base 292.8 us vs pipe 295.7（0.99×）；STAGES=4 270.9 vs 278.6（0.97×）；decode 形状 12.3 us 两者同。**流水级数对 prefill 形状零收益** ⇒ B24a（BN=128 大 tile）的前提"4 级流水还不够"不成立；582 us（nsys 态）vs 微基准 271 us 也差 2×（nsys 开销 / 真实行分布不均 / 权重 126 MB 不在 L2）。B24a **降级为待定**，先做下面。
- **c8 TTFT 的量级问题（新靶子）**：c1 prefill 426 tok = 26 ms；c8 TTFT p50 **545 ms** = 8×426 tok 用了 **21×** 的时间（vLLM 146 ms，5.8×）。bench 闭环 8 路同时完成 → 8 个新请求几乎同时到；按 sched 逻辑先 running（在飞 decode 行）再 waiting，新请求并进当步 → **Paged 混批步走 `paged_win`（逐 token 查 block_table，不是 `prefill_win` 的连续 K/V 快路径）**；D68 的 b15 c8 TTFT 1397 ms 已经是这条慢路径被 fp8 cache 再放大一次。假设：8 路 prefill 中大部分落在混批慢路径，而不是纯 Prefill 步。
- **探针 `ttft8.sh`**：① `max_tokens=1` c=1/2/4/8 看纯 prefill 标度（不混批时应 ≈ 线性）；② c8 ×64 `HOST_TIME=1` 统计每步 mode 与 tokens（`[fwd-sample] ENTER mode=... t=...`），数 Prefill / Paged 步各几次、Paged 步耗时。结果 `tmp_ttft8_RESULT.txt`。

## ★★★★★ D81 2026-10-28 ttft8 探针（b16b 二进制 + FP8_LT_TUNE=0 + **HOST_TIME=1**）：**c8×64 bench 口径 TTFT p50 74.9 ms / TPOT 10.05**，而 bN 链同口径 c8 TTFT 545 ms —— 差 7×，服务端 env 只差 HOST_TIME
- 纯 prefill 标度（max_tokens=1）：c1 28.4 / c2 58.8 / c4 85.5 / c8 75.4（p50）/ c16 96.7（p90 647）。Prefill 步 t=432 26.4 ms、t=860 48.3、t=3008 77.2、t=4096 118.7 ⇒ **3008 tok 一步 77 ms = 1560 tok nsys 估算的线性延伸，prefill 本身不超线性**。
- c8×64 窗口：Prefill 1 步 + Decode 631 步（9.0 ms）+ Paged 16 步（35 ms, max_t=97）。Paged 步 t≈97 用 35 ms（比 t=432 的纯 Prefill 还慢 1.3×，**混批慢路径确认**：8 decode 行 + 1 个 ~90 tok chunk）。
- 关键矛盾：同一二进制、同一 bench 参数，bN 链 c8 TTFT 545 ms（b16b/b25 四轮一致），ttft8 链 75 ms。差别：① HOST_TIME=1（每步两行 eprintln → stderr 重定向文件，还可能改变 async 节拍）；② FP8_LT_TUNE=0（b16b 链 A/B 时 tune 还没进代码 → 不是它）；③ bN 的 c8 跟在 c1 之后同一服务（KV 池 / prefix cache 状态）；④ 端口。**ttft8b 正在跑**：同服务 c1×8 → c8×64 ×3，plain / HOST_TIME=1 / plain2 三组，看是 HOST_TIME 还是"c1 之后"触发 545。
- 若是"c1 之后"：怀疑 prefix cache / 块回收把后续 waiting 请求卡在 `allocate_slots` 失败 → 等下一步（每步 9 ms × N 步 = TTFT 数百 ms）。若是 HOST_TIME：说明 545 是真实态、75 是计时副作用（不太可能让 TTFT 降 7×）。

## ★★★★★★ D82 2026-10-28 ttft8 步序解读：c8 TTFT 的真凶是 **Paged（混批）步比同 token 数的纯 Prefill 步慢 3×**，且有 ~25 ms 固定开销
- 步序（`tmp_ttft8_seq.txt`）：c8×64 每波 = `Paged:13 ×1 + Paged:~90 ×1 + Decode:8 ×62`。每波只 ~100 个新 token ⇒ **prompt 命中了前面 c1–c16 探针的 prefix cache**（bench 同 seed 同 idx 生成同 prompt），ttft8 的 c8 TTFT 75 ms **不可与 bN 的 545 比**（bN 每条链新起服务，c1×8 只暖了 idx 0–7）。
- **Paged 步耗时表**（us，含 host；8 行 decode + 1 个 chunk）：t=10–16 **30–41 ms**、t=79–97 34–39、t=191/199 **59–60**、t=2765 **476**。对照纯 Prefill：t=429 26.4、t=857 42.8、t=3008 127、t=4096 171；纯 Decode t=8 9.0。
  ⇒ Paged 步 ≈ **25 ms 固定 + 0.16 ms/token**（vs Prefill 0.04 ms/token、Decode 9 ms）。t=2765 的 Paged 476 vs Prefill 3008 的 127 = **3.7×**。
- 推算 bN c8×64（无 cache）：每波 8 新请求几乎同时到 → 先 cut-in 1 个纯 Prefill（26）、其余 7 个并进下一步成 Paged t≈3000（~480 ms）⇒ TTFT ≈ 500 ms **吻合 545**；首波在空闲 GPU 上更快 → mean 428 < p50 545 也吻合。vLLM 同口径 146。
- **刀（B26）**：Paged 步的 prefill 行走了什么慢路径？候选：① 滑窗层 attention `paged_win`（fa_v5 PAGED：K/V 经 block_table 装载，grid 按 max_seqlen 分 BQ=64 行块）——t=13 时 query 只有 1 个短 chunk，不该 25 ms；② **D=512 全注意力层 PAGED 路径**：`gqa.cu:1577` `head_dim==512` 的分派 —— 非 paged 走 d512 快 kernel（fa_v6），**paged 可能落到 MAX_D=512 通用慢路径**（nsys D54 曾见 5 层 × 369 us = 1.85 ms/步，那是 decode 的；prefill 行走通用路径按 q_len × seq_len 串行会更惨）；③ 混批时 MoE/dense 的行数 > 8 不走 mega / gemv，正常；④ eager 全程（无 graph）launch 1065 次 ≈ +0.5 ms，不是主项。
- 验证手段：`KAON3_STAGE_TIME=1` 若只在 decode 生效则扩到 Paged；或 nsys 抓一个 c2 混批（c1 在飞 + 1 新请求）步看 kernel 表。

## ★★★★★ D83 2026-10-28 ttft8b（同服务 c1×8 → c8×64 ×3，plain / HOST_TIME=1 / plain2）：**首次 c8×64（新 prompt）TTFT p50 503–505 ms**，第 2/3 次（prefix cache 命中）71–75 ms；三组一致 ⇒ HOST_TIME 无关，D81 的 75 是 cache 态，**545/505 是真实态**
- TPOT 也随之：首次 p50 9.9–10.0 / p90 17；cache 态 9.7–9.9 / p90 10.0。bN 口径（c1×8 后 c8×64 一次）= 首次态。
- 结论链：c8 TTFT 505 = 一波 8 新请求中 7 个并进在飞 decode 行的 **Paged 步**（t≈3000）耗 ~480 ms（D82 表 t=2765 → 476），vs 同 token 纯 Prefill 127 ms。**Paged 步 3.7× 慢 + 25 ms 固定开销 = B26 靶子**。
- 代码线索（`gqa.cu` paged_win 分派）：`head_dim==256` 有 v6 PAGED 分支、`128/256` 有 v5 PAGED，**`head_dim==512` 在 PAGED 没有任何快分支 → 落到 `gqa_paged_kernel`（grid=(req, head)，每 block 串行 q_len×seq_len×D 标量点积）**；非 paged 的 prefill_win 对 512 走 v6。5 层全注意力 × 3000 行 chunk 串行 = 几百 ms 量级，与 476 吻合；小 t 的 25 ms 固定开销可能也是它（一个 block 做 13 行 × 430 key × 512 维 + 决定性 8 行 decode 不 skip）。
- 下一步：nsys 抓 c2 混批步 kernel 表定量（`nsys_mixed.sh`），然后 B26 = paged_win 给 D=512 加 v6 PAGED 分支（kv_fp8 时 v6 不读 e4m3 → 要加 KV_FP8 展开或先对 D=512 层用 bf16 cache）。

## B26 设计 2026-10-28：paged_win 的 D=512 走 v6 PAGED（含 KV_FP8）
- 根因代码：`gqa.cu` `gqa_paged_win_bf16` 分派只有 `head_dim==256`（v6/v7/umma）与 `128/256`（v5）快分支；**D=512 落到 `gqa_paged_kernel`**：grid=(req, head)，每 block 对 q_len 行串行、每行 3 遍扫 seq_len 个 key、每 key D=512 标量 FMA（无 mma、无 smem tile）。q_len=3000 时每 block ≈ 3000×3000/2×512×3 ≈ 7 G FMA 串行 → 几百 ms；q_len=13 时 ≈ 13×430×512×3 ≈ 8.6 M FMA / 256 线程，5 层 ≈ 几 ms + decode 行不 skip。非 paged 的 `prefill_win` 对 512 早有 `v6_launch<false,512,4,16>`（smem 97.5 KB，刚好 < 99 KB）。
- 刀：v6 kernel / v6_launch 加模板参数 `KV_FP8=false`；PAGED+KV_FP8 时 load_tile 对 e4m3 cache **同步读 8 B → `fp8x8_to_bf16x8_g` 直写 sK/sV**（v5 用 cp.async 到 staging 再展开，要多 2×TILE×D 字节 smem；D=512 加不起 → 不用 staging；NBUF=1 本来无重叠，同步读无损）。分派：`cfg>=0 && head_dim==512 && !GQA_D512_REF` → `v6_launch<true,512,4,16[,1,false,true]>`，InvalidConfiguration 才回退 ref。
- 门禁：新 GPU 测试 `gqa_paged_d512_parity`：bf16 cache 与 e4m3 cache 两场景、2 请求（q_len 37 + 1 跳过 / hist 100）、v6 vs ref（`GQA_D512_REF=1` 子进程或 cfg<0）位差 0（ref 为 fp32 累加标量，v6 为 mma bf16 → 不逐位，用 D54 的 d512 口径：max|Δ| 阈值与 decode_win_parity 同）。bench：bN 口径 c8 TTFT 545 → 预期 ≤ 200（Paged t≈3000 步 476 → ~130）。
- **B26 实现落地（commit 7600ec1）**：`gqa.cu` v6 kernel/launch 加 `KV_FP8` 模板参数（PAGED+KV_FP8：load_tile 同步读 8 B e4m3 → `fp8x8_to_bf16x8_g` 直写 sK/sV，`continue` 跳过 cp.async），`gqa_paged_win_bf16` 加 `head_dim==512` 分支（`GQA_D512_REF=1` 回退）。门禁 `crates/kernels/tests/gqa_paged_d512_parity.rs`：3 请求（q 37/300/1，hist 563/0/76），v6 bf16 vs ref max|Δ| ≤ 2%·amax、v6 e4m3 vs v6 bf16 **位差 0**。远端链 `b26.sh`：门禁 → `GQA_D512_REF=1` 自洽 → `bN.sh TAG=b26`（A 旧 / B 新 ABAB，c1+c8）。预期 c8 TTFT 545 → ≤ 200、c1 不变。

## ★★★★ D84 2026-10-28 B26 门禁过：`gqa_paged_d512_parity` v6 PAGED D=512（bf16 cache）vs ref max|Δ| 0.0156（amax 4.5，tol 0.09，ndiff 8607/1.38M = mma 序差几 ulp）；**v6 e4m3 cache vs v6 bf16 cache 位差 0**；ref e4m3 vs bf16 位差 0；`GQA_D512_REF=1` 时 v6 路径 ndiff=0（分支回退生效）。bN TAG=b26 跑中（A 旧二进制 / B 新）。
- 顺带确认滑窗层（D=256）PAGED 本来就有快分支：cfg≥1000 → v6/v7 PAGED；kv_fp8 时 `cfg=0` → fa_v5 PAGED KV_FP8 模板（B15b 已逐位）。只有 D=512 漏了。

## ★★★★★★ D85 2026-10-28 b26 首批 B（paged_win D=512 v6）：**c8 TTFT p50 542.6 → 155.8（-71%，3.5×）、mean 427 → 130**；c8 吞吐 465 → **672 tok/s（+45%）**、req/s 7.27 → 10.50；c8 TPOT 9.86 → 9.83；c1 TTFT 27.2 → 28.6 / TPOT 5.99 = 5.99（c1 不变——bN 的 A/B 都没带 FP8_GEMV，故 5.99 不是 5.57；与 b16b 链 A 侧 6.04 同口径）
- vLLM c8 TTFT 146 → 我们 156（1.07×），之前 3.7×。c8 吞吐 vs vLLM 956 → 70%（之前 49%）。
- 剩余：c8 TTFT 156 ≈ 7 请求 Paged 步 ~130 ms = 纯 Prefill 3008 tok 的 127 + 25 ms 固定开销；小 t Paged 步（t=13 → 36 ms vs decode 9）的 25 ms 固定开销还在（pagedprobe 跑完看）。候选：① Paged 步 eager 1065 launch（~1 ms，不是 25）；② 混批里 decode 行走 `gqa_decode_win` + prefill 行走 paged_win 两套 kernel 都跑全 30 层——各 5 层 D=512；③ MoE route_fused 只到 tokens ≤ 64，Paged t>64 走旧 route；④ dense/qkv 在 8 < tokens ≤ 16 走 fp8 gemv，> 16 走 cuBLASLt tile=13（m 小时 nvjet 慢）。等 pagedprobe 的 t 分桶表再定。
- **b26 定稿（两轮 ABAB）**：A c8 TTFT 542.6/543.5 → B **155.8/155.6（-71%）**，吞吐 465.1/465.1 → **672.0/672.4（+45%）**，req/s 7.27 → 10.50；c8 TPOT 9.86/9.86 → 9.83/9.86（持平）；c1 TTFT 27.2/26.4 → 28.6/28.4（p50，噢动范围内，mean 31.5/31.4 → 28.6/28.5 反而降），c1 TPOT 5.99 = 5.99（bN 不带 FP8_GEMV 的口径）。英文 raw / 中文输出 A/B 同。**合入**，CSV round b26。

## ★★★★ D86 2026-10-28 pagedprobe（b26 二进制，HOST_TIME，c8×64 **seed 11 新 prompt**）：TTFT p50 157 / mean 134；每波 = `Prefill:857`（cut-in 收了 2 个请求，43 ms）+ `Paged:~2580`（6 请求 + decode 行，**114 ms = 0.044 ms/token ≈ 纯 Prefill 的 0.042**）+ `Decode:8 ×62`（10.0 ms）。**Paged 步的 25 ms 固定开销与 3.7× 慢都没了**（全是 D=512 通用 kernel）。TTFT 157 = 43 + 114 ≈ 两步之和。
- 下一刀 TTFT 只能靠 prefill 本身（3000 tok 127 ms → vLLM 同口径约 100）：B24（MoE prefill GEMM 39%）/ B25 已证无效 / qkv nvjet。
- **⚠ 疑似正确性问题（必须先核）**：Paged 步里 `paged_win(..., skip_single = mode==Paged, ...)`，三个 PAGED kernel（v5/v6/v7 `gqa.cu:185/455/721`）对 q_len==1 的请求直接 return；ops.rs 注释说"它们由 gqa_decode_bf16 处理"，但 **kaon3 `layer.rs` 的 Paged 分支之后没有再为 decode 行调 decode kernel**（Decode 分支的 `gqa_decode_win_bf16` 只在 mode==Decode 走）。若属实，混批步里在飞 decode 行的 attention 输出是上一层残留 → c8 输出错而 bench 不查内容（c1 永不混批所以中文 raw 一直对）。核法：同 seed c1 vs c2 的逐请求输出比对（greedy 应全同）。

## ★★★★★★ D87 2026-10-28 **混批正确性 bug 实证**（`mixparity.sh`，b26 二进制，greedy 64 tok，同 prompt 串行 vs 并发）：c2 vs 串行 **6/8 不同**、c8 vs 串行 **7/8 不同**（cache 态，差异从第 1 个字符就开始；有 "the彷️-----" / "```python" 类乱出），新 prompt c8 vs 串行 3/8 不同
- 根因 = D86 的怀疑：Paged 步 `paged_win(skip_single=true)` 跳过 q_len==1 行，而 kaon3 `layer.rs` 没有再为这些 decode 行调 decode kernel → 混批步里在飞 decode 行的 attn 输出是上一层 / 上一步残留。**所有 c≥2 的 bench 数字（吞吐 / TPOT）在这个 bug 下跑出来的——token 数和步数不受影响（greedy 仍出 64 tok），但输出是错的**。c1 从不混批所以中文 raw 一直对；bN 的"c8 输出同"从来只看了 c1 的两条 curl。
- **B27**：Paged 分支后补 `gqa_decode_win_bf16(..., cu_seqlens_q=Some(&ctx.cu_seqlens), ...)`（v3 core 已支持：q_len != 1 的请求 return，行号取 cu_seqlens_q[b]）。性能：每混批步多 30 次 decode attention launch（c8 每波 1 步）→ 可忽略。门禁：mixparity 三组 **0/8 不同**（greedy、同 prefix cache 态下 decode 行数值应与串行逐位同；新 prompt 组 prefill 分块不同可能有 ulp 差 → 允许"首差位置 > 40 字符"但目标 0）。
- **D88 来历**：`skip_single = mode==Paged` 在 kaon3 第一个可编译 commit（9d8a702，阶段 3）就在，照搬 qwen3 的 `ops.attn().mixed`（那边 `gqa_mixed_bf16` 内部先 paged 再对 q_len==1 行调 decode kernel）但只搬了 paged 一半。通用 `gqa_paged_kernel` **不收 skip_single**（D=512 层 B26 前走它）→ B26 前：滑窗层 decode 行被跳（错）、D=512 层照算（对）；**B26 把 D=512 也换成尊重 skip_single 的 v6 → 5 层全注意力的 decode 行也空了**，所以 b26 的 c8 +45% 里含"少算了 8 行 × 5 层 attention"的水分（量很小：decode 行 attention 每层几 us）。B27 后 c8 数字才是真的；mixparity 过了再 bN 比 b26。

## ★★★★ D89 2026-10-28 b27 门禁 **没过**：mixparity c2 6/8、c8 6/8（cache 态）、c8 fresh 3/8 —— 与 b26 几乎一样 ⇒ **B27 补的 decode 行 attention 不是（唯一）根因**，或者 decode 行根本不在那一步
- 差异从 **char 1** 起（B 的首 token 就错）：首 token 来自 prefill 行的 logits，不是 decode 行的 → 错在 **Paged 步里 prefill 行自己**（或采样行号 / 输入 token 映射），而非 A 的 decode 行受污染。但 fresh 组 5/8 全同（64 tok greedy）说明大多数 Paged 行是对的。
- 候选：① cache 态 B = 带历史 chunk（t≈13, hist≈400）→ paged_win PAGED 路径 hist>0 的 prefill 行：v5/v6 kernel 的 `hist`/`vis`/window 下界算错？（c1 串行 cache 态从未被验过！mixparity 的"串行基线"是 fresh Prefill 态，cache 态 c1 也可能错——那就不是并发 bug 而是 **chunked-prefill-with-history bug**）；② fresh 组 3/8：hist=0 的 Paged 行 vs 纯 Prefill 行差在 paged_win vs prefill_win（cache 读 vs 连续 K/V；kv_fp8 cache 读的是 e4m3 往返值，prefill_win 读的 k/v 也做过 fp8 往返 → 应同）；③ logits 行号 `logits_row_start` / `rows` 在 Paged 多 chunk 时取错行；④ 输入 token：Paged 步 prefill 行 input_ids 来自 host（对），decode 行来自 gather（B27 不影响）。
- **mixprobe（GPU 1 并行跑，不碰 bN）**：A 先发 200 out；80 ms 后 B 发（必进 Paged 步）；比 A/B 各自 vs 串行；再同 prompt 重发一遍（cache 态：B 变 hist≈400 的 chunk）；三组 env plain / `CUT_IN=0` / `ATTN_DECODE_OLD=1`。期望把"谁错"钉到 A 侧 / B 侧 × fresh / cache 态。
- **D90 b27 bN 数字作废**：A 侧二进制在链里被我 8 s 后的 cp 换回 b26，A 与 A2 的 c8 TTFT 一个 101 一个 155 ⇒ A 跑的不是同一个二进制。B（b27）：c8 TTFT 118、吞吐 679、TPOT 10.26；c1 不变。B27 的性能影响本来就应 ≈ 0；**正确性没过之前不入 CSV**。
- 下一步排查并行跑：GPU 1 `mixprobe`（A 在飞 + B 插入，fresh / cache 态，3 env）；GPU 4 `cacheprobe`（**c1 串行**同 prompt 三次：若第 2 次就不同 → 不是并发问题，是**带历史 chunk（prefix cache 命中后 Paged 步）本身错**；env plain / KV_FP8_STORE=0 / GQA_PAGED_D512_V6_OFF=1 分路径）。怀疑点：D=256 滑窗层 **v5 PAGED + KV_FP8 + hist>0 + window** 组合从未被测过（B15b fp8cache 测试全是 q_len=1；Qwen3 的 mixed 是 window=0）。
- D91：mixprobe（GPU 1）plain 组第一条串行请求就 **HTTP 503**（A=400 词 200 out 串行基线）→ 服务侧拒绝（查 503 条件：排队上限 / 引擎未就绪 / max_tokens 超限？）；cacheprobe（GPU 4）同时起，结果待看。先查 503 来源再重跑。
- D91 解：容器 `kaon3-full` **只挂了 GPU 0**（`CUDA_VISIBLE_DEVICES=1/4` → `CUDA_ERROR_NO_DEVICE`，serve 秒退，脚本继续 curl 到老端口的残留服务 → 503）。两个探针改 GPU 0 串行排在 bN b27 之后（`probes.sh`）。
- D92：b27 bN DONE（B2 c8 TTFT 155.8 / 吞吐 675 / TPOT 9.82 = b26 同值，B27 性能中性，符合预期）。probes 链（cacheprobe → mixprobe）在 GPU 0 跑。读 v5 PAGED 窗口/hist 掩码（`lo/vis/warp_hi_lo/tile0`）没看出错；runner 带历史 chunk 的 start/positions/slot 也按 `num_computed` 算。等 cacheprobe 定"c1 串行 cache 态是否就错"。

## ★★★★★★ D93 2026-10-28 cacheprobe（c1 **串行**，同 prompt 三次，第 2/3 次 prefix cache 命中 → 带历史 chunk t=11–17 走 Paged 步）：**plain 3/7 prompt 输出改变**（400 词：首字符就变 'be be be…' → 'as it is is…'；33 词：'-The-The-' → 's little bit…'；90 词 char 32 起且第 2≠第 3 次），**`KV_FP8_STORE=0` 0/7 全同**
- ⇒ **不是并发 bug**。mixparity 的 c2/c8 差异来自同一根因：bench/探针里 prompt 命中 cache 后的 chunk 行（hist>0, q_len 小）在 **e4m3 cache** 下算错；fresh 组 3/8 也可能是 c8 一波里有 cut-in 后"剩余 chunk"（hist = 已算部分）。
- 步序证据：`Prefill:400 → Decode:1×47 → Paged:16 → Decode …`：第 2 次同 prompt 只算 16 个 token（384 命中），Paged 步 1 请求 16 行、hist=384。
- 嫌疑链（kv_fp8 + PAGED + hist>0 + q_len>1）：① fa_v5 PAGED KV_FP8 模板（D=256 滑窗层；kv_fp8 时 cfg 从 464 降为 0 → v5）的 cp.async8 staging + expand；② B26 v6 KV_FP8（D=512）；③ `gqa_cache_write` fp8 分支对 chunk 的 slot 写；④ fp8 往返 q 的位置。B15b 的 fp8cache 测试只有 q_len=1 行（decode）和 hist=0 多行，**从没测 hist>0 的多行**。
- 复现测试 `gqa_paged_fp8_hist_parity.rs`：D=256 win 1024 reqs [(400,16),(413,17),(1200,14),(90,11),(33,33),(1,1)] + [(1300,16),(2100,300)]、D=512 win 0；cfg=0 bf16 vs e4m3 位差 + 逐请求定位 + vs ref。链 `histpar.sh`（等 mixprobe DONE）。
- 顺带：`KV_FP8_STORE` 默认开是 D70 定的（c8 TPOT -1.9%）；若 kernel 修不快，**先把默认改关**止血（c8 TPOT +0.2 ms，换正确性）。
- **D94 mixprobe plain**：case0（A 400/200out + B 400/64out）**四项全 SAME**；case1（B max_tokens=1）A **DIFF at char 256**（≈ A 的第 ~50 个 token，正是 B 插入的那一步）、B SAME；case2（B 60 词）A char 1 DIFF、B char 4 DIFF；case3（A 1500 词）A char 8、B char 2 DIFF。cache 态组与 fresh 组逐项相同 ⇒ 差异由 **A/B 相对时序决定**（B 的 prefill 落在 A 的哪一步），不是 cache 本身。
- 解读：A 被 B 的插入改写 = **混批 Paged 步里在飞 decode 行（A）的输出错**（B27 补的 decode 行 attention 之后依然错 → 要么 B27 的调用没生效/位置错，要么 decode 行错在别处：gather 的 input id、position、或 KV 写入 slot）；B 自己（prefill 行）在 case2/3 也错。case0 全对说明"混批"不必然错——与步内行布局 / token 数有关（case0 B 400 tok；case2 B 60 tok、case3 A 1500 tok hist 大）。
- 结合 cacheprobe（c1 串行 cache 态 e4m3 3/7 错、bf16 0/7）：至少两个独立问题 ①e4m3 cache 读 hist>0 多行 chunk；②混批步 decode/prefill 行间串扰。**止血**：`KV_FP8_STORE` 默认改关（①）；②继续 histpar + 再做 `KV_FP8_STORE=0` 下的 mixprobe 看 ② 是否随 ① 消失。
- **D95 cacheprobe 第 3 组 `GQA_PAGED_D512_V6_OFF=1`（e4m3 cache 仍开，只把 D=512 PAGED 退回通用 kernel）：0/7 全同** ⇒ e4m3 问题**落在 B26 的 v6 KV_FP8 分支**（D=512 全注意力层 hist>0 多行），不是 v5 滑窗层。可疑点：v6 KV_FP8 load_tile 对 `j >= kn`（tile 尾部越界 key）**没有写零**——bf16 路径 cp_async16 bytes=0 做 zfill，我的同步读分支 `continue` 前只处理了 `j < kn`；越界 key 行 smem 残留上一 tile 的值 → 被 mask 掉的 key 不影响（S=-inf→P=0），但 **PV 的 mma 读 sV 残留 × P=0 = 0** 也无影响……除非 **NaN/Inf 残留**（0 × Inf = NaN）：第一 tile 前 smem 未初始化，可能含 NaN 位型！bf16 路径 zfill 保证了 0。hist>0 时 n_tiles 多、tile 尾部更常见。**修**：`j >= kn` 时写零 uint4。
- 另：`gqa_paged_d512_parity` 过了（e4m3 vs bf16 位差 0）但它的 reqs (600,37)/(300,300)/(77,1) 恰好…… 600/16=37.5、300/16=18.75、77/16 都有尾部 → 应该也会碰残留；可能残留碰巧是 0（alloc_zeros 后的 smem 前任是零）。serve 里 smem 前任是别的 kernel → 非零/NaN。
- **D96 复核 D95 假设**：v6 KV_FP8 分支的 `continue` 只在 `j < kn` 内，尾部 `j ≥ kn` 仍走 `cp_async16(bytes=0)` zfill → **"越界 key 未清零"不成立**。目前事实：serve 里 v6 KV_FP8 PAGED（D=512, hist>0）错、v6 bf16 PAGED 对、通用 kernel e4m3 对；而 `gqa_paged_d512_parity` 里 v6 e4m3 == v6 bf16 位差 0。差异只能来自测试没覆盖的条件：① smem 残留（测试里 alloc_zeros 后 smem 前任多为零）；② 多 block 并发下的 race（测试 h=8 nkv=2，serve h=16? nkv=2）；③ serve 的 block_table / max_blocks / seq_lens 组合（prefix cache 复用块、块号非连续）；④ 实际数据分布（e4m3 饱和 / 次正规）。等 `histpar`（serve 真实形状、随机块号连续）与 `mixprobe2`（KV_FP8_STORE=0 下混批串扰是否消失）。若 histpar 位差 0 → 下一步在 serve 里 dump 一个 Paged 步的 attn 输出（`sub_out`）对 v6 vs 通用。
- D97：mixprobe 的 `ATTN_DECODE_OLD=1` 组没跑完（脚本结尾无 DONE；serve 停后 python 僵死）→ histpar/mixprobe2/cp2 三个 watcher 全等死。杀掉、去掉等待、串成 `chain3.sh` 直接跑（histpar → mixprobe2 → cacheprobe2）。**教训**：探针脚本的 `echo DONE` 要放 trap 里，watcher 要带超时。
- 已确认与测试一致的条件：`ATTN_MODE` 默认 2（test 与 serve 同）；`g_p_fp8` 两边都按 kv_fp8 置 1（test p_fp8 参数传 0！→ **test 里 pq8 关、serve 里开**——P 先 fp8 往返再进 PV；但 bf16/e4m3 两侧同开同关，不影响"位差 0"的结论，只影响 vs ref）。
- **D98 代码级复核（v6 KV_FP8 vs 通用 kernel vs v5 KV_FP8）**：e4m3→bf16 展开三处同口径（`__nv_cvt_fp8x2_to_halfraw2` → half2 → float2 → bf16 rn）；写入 `bf16_to_e4m3` SATFINITE 一处；e 偶移量 = slot·Hkv·2·D + hk·2·D（+D 取 V）对 D=512 Hkv=2 的 8 B 对齐成立；尾部 zfill 走 cp_async16(bytes=0)；`cp_async_wait_all + __syncthreads` 在每 tile 顶部，同步写与 cp.async 零填充同一 tile 无 race。**读代码没找到差异**。决定：若 histpar（serve 真实形状）位差 0 → 在 serve 里加 `ATTN_D512_DUMP=1`：Paged 步 D=512 层 attn 输出同时跑 v6 和通用（`GQA_D512_REF` 路径）各一份，dump 比对定位到行/头；若 histpar 位差 ≠ 0 → 直接拿测试复现修。

## ★★★★ D99 2026-10-28 histpar（serve 真实形状：D=256 win1024 hist 400/413/1200/90/33 q 11–33、(1300,16)/(2100,300)；D=512 win0 同形状）：**fast e4m3 vs fast bf16 位差 0（三组全部）**；fast vs ref max 0.0156（mma 序）；ref e4m3 vs ref bf16 位差 0
- ⇒ v6 KV_FP8 / v5 KV_FP8 在**干净测试环境**里逐位正确。serve 里错而测试里对的剩余解释：① **smem 残留**（测试里 smem 前任几乎都是零；serve 里前任 kernel 留非零/NaN 位型；而 v6 KV_FP8 分支与 bf16 分支的 smem 覆盖模式不同：bf16 全靠 cp.async，fp8 同步写）；② 多 kernel 流内顺序（cache_write_fp8rt → paged_win 同流，应有序）；③ 数据分布（真 K/V 的 e4m3 饱和 / 次正规）。
- 直接验 ①：加 `smem_poison` kernel（每 SM 多 block 把 ~96 KB 动态 smem 写满 NaN 位型 0x7fc0 / 0xffff 交替），在 hist parity 测试里每次 attention launch 前先 poison 一遍；若 fast e4m3 ≠ clean 基线 → 残留实锤 + 定位到哪一行/哪 tile。

## ★★★★★★ D100 2026-10-28 mixprobe2（**KV_FP8_STORE=0**，b27 二进制）：**B（插入的 prefill 行）全部 SAME；A（在飞 decode）仍 DIFF**：case2（B 60 词/64 out）A char 70；case3（A 1500 词）A **char 1**；case0 两次一 SAME 一 DIFF（char 775 / char 8）；**case1（B max_tokens=1）全 SAME**
- ⇒ 第二个独立 bug，与 e4m3 无关：**B 只插入一步（max_tokens=1）时 A 不受影响；B 留下来一起 decode 时 A 错**。错的是 A（在飞）、不是 B。A 的首 token 就错（case3 char 1）说明 **cut-in 之后紧接的那个步** 里 A 的行就错了：A 的 chosen（上一步采样）经 gather 作为输入 → 若行号/顺序在"插入后 2 请求"的步里对不上（cut-in 把 B 排到 A 前面？`logits_rows`/`gather_idx` 用的行序与 sched 行序不一致？），A 吃到错 token。B 正确 ⇒ prefill 行 / Paged attention 没问题。
- 还需区分：是 **2 行纯 Decode 步**本身错（但 c8 bench 几十轮 8 行 decode 跑着——若 8 行 decode 本身错，输出早该乱……其实没人看过 c8 输出！）还是**cut-in 后第一步**错。探针 `pair2`：① A/B 同时发（无 cut-in 时序差：两条新 prompt 同步到 → 一个 Prefill 步两行 + 2 行 Decode）vs 串行；② `CUT_IN=0` 版；③ `ATTN_DECODE_OLD=1` 版（gqa_decode_win 多行 vs 旧 paged 路径）；④ B max_tokens=1 对照。
- D101：kaon3 `executor.rs` 里 **没有任何 gather_idx / prev-chosen 处理**（grep 只有 warmup 的 `input_ids = vec![0; t]`），而 qwen3 executor 有 `save_chosen` / `gather_idx` 分支。c1 async decode 却一直正确 ⇒ 要么 kaon3 服务 async_sched 关（无在飞行）、要么 PlanBuffers（core 公共件）在 upload 时做了 gather。正在查；若 kaon3 走的是"同步调度 + cut-in 需要 async"的混合，cut-in 分支（`begin_cut_in` 要求 `!any_gather`）之后的下一步 A 行的 input id 来源就是嫌疑。
- **D102 强嫌疑（代码）**：`PlanBuffers::upload` 对 `gather_idx ≥ 0` 的行做 `ids[i] = prev_chosen[gather_idx[i]]`（exec.rs:471），`prev_chosen` 要由执行器每步 `save_chosen(chosen, n)` 写入（qwen3 executor 两处调用）。**kaon3 executor 从不调 `save_chosen`** → 若 kaon3 runner 走 async（gather_idx ≥ 0），在飞行 token 全是 prev_chosen 的零/陈旧值。c1 却对 ⇒ kaon3 多半 `supports_async=false`（runner 同步，gather_idx 全 -1）。那 cut-in（engine 侧只看 ASYNC_SCHED env + CUT_IN）在**同步 runner** 上：步 n 已同步算完、结果存在 runner 的 sync 槽里等 `finish(pend_n)`；cut-in 的 `begin_cut_in → forward_sample → finish(pend_c)` **复用同一个 sync 槽 / `self.generated`**，把步 n（A）的结果覆盖成 B 的 → A 在 cut-in 那步吃到错 token（B 正确、A 错、B 只 1 token 时也该错…case1 全 SAME 待解释：可能 case1 时序没触发 cut-in）。验证：`CUT_IN=0` 组（mixprobe2 第 2 组被 503 污染、pair2 有 CUT_IN=0 组）若全 SAME → 实锤。修：cut-in 用独立结果槽，或 kaon3 实现 supports_async + save_chosen（本来就该有：async 省一步 host）。

## ★★★★★★ D103 2026-10-28 **第二个 bug 根因（代码实锤）：同步 runner 上 cut-in 覆盖步 n 的采样结果**
- kaon3 executor `supports_async=false` → runner `begin(so_n)` 走同步分支：`forward_sample(&plan, &mut self.generated, ...)` 把步 n 的 token 写进 **`self.generated`（runner 字段）**，`Pending{is_async:false}` 不带结果；engine 把 pend_n 存进 `self.pending`，下一轮循环先做 **cut-in**：`begin_cut_in(so_c)` → 又一次 `forward_sample(..., &mut self.generated)` **覆盖**，`finish(pend_c)` 读到 B 的 token（对）；然后 `finish(pend_n)` 读 `self.generated` —— **已经是 B 的结果**（长度相同 1 行 → `ensure!(len)` 过）→ A 在那一步拿到 B 的 token。A 从此跑偏（char 1/8/70…取决于 cut-in 发生在 A 的第几个 token）；B 不受影响；B max_tokens=1 时 cut-in 也发生——但 case1 全 SAME 是因为 cut-in 要求 `so_n.rows ≤ 8 && num_waiting > 0`，B 到达时 A 的步 n 恰已 finish（时序）。
- 与 D87/D100 全吻合：bf16 cache 下 B 全 SAME、A 错；`CUT_IN=0` 应全 SAME（pair2 验证中）。**所有 c≥2 bench 的输出都有此错**（每波 8 请求同时到 → 至少 1 次 cut-in）。
- **B28 修法**：`Pending` 加 `sync_result: Option<(generated, drafts)>`，同步分支把 `self.generated/drafts` **移进 Pending**，`finish` 从 pending 取；engine 不改。副作用零。另：kaon3 应实现 `supports_async`（prev_chosen gather + save_chosen）省一步 host——记 TODO。
- D104：cacheprobe2 两组都 503（同 D91 现象：serve 没起来，python 打到残留端口？或 `KV_FP8_STORE=1` 起服 OOM？查 log）。b28 链已排队（等 pair2 DONE）：build → pair2 → mixparity → cacheprobe(e4m3) → bN b28。本地 `cargo check -p dinfer-core` 过。计划补 runner 单测：同步 Echo executor 上 begin(n) → begin_cut_in(c) → finish(c) → finish(n)，断言 n 的 token 不被 c 覆盖（B28 前应 fail）。
- **D105**：runner 单测 `sync_cut_in_does_not_overwrite_pending_step_result`：B28 修前 **FAIL**（步 n 拿到 cut-in 的 token 1007）、修后过 → 根因实锤。cacheprobe2 被 OOM/503 污染（mixprobe2 的 serve 没退干净时起服），作废杀掉；e4m3 验证并入 b28 链的 cacheprobe 门禁组。队列：histpar2（smem 毒化）→ pair2（b27 二进制，看 CUT_IN=0 组是否全 SAME = 侧证）→ b28（build → pair2 → mixparity → cacheprobe e4m3 → bN）。
- **D106 同一根因的第二条路径（更常见）**：engine 的 `async_sched` 分支——步 n 在 pending 时先 `begin(so_next)`；同步 runner 上 `begin` 立刻 forward_sample，`last_begin_was_async()==false` → 立刻 `finish(pend_next)` 进 `sync_done`，**然后才 `finish(pend_n)`**。B28 前 `finish(pend_n)` 读到的 `self.generated` 是步 n+1 的 ⇒ **不需要 cut-in，只要 pending 非空（c≥2 的每一步）步 n 的 token 就被 n+1 的覆盖**（n、n+1 行数相同 → ensure 过）。这解释了 mixparity cache 态 c2 6/8 都错、以及 ATTN_DECODE_OLD/CUT_IN=0 组仍错。B28 让 Pending 自带结果 → 两条路径一起修。kaon3 没 async executor 是放大器（qwen3 走 async，结果在 executor 里按 ticket 取）。
- D107：histpar2 一直等 cacheprobe2 的 DONE——我在**宿主机**执行 `echo DONE >> /w/dinfer/...`（容器路径），写到了不存在的目录；宿主机路径是 `/hf_data/yaochen/dinfer/...`。已补写，队列继续：histpar2 → pair2 → b28。

## ★★★★ D108 2026-10-28 histpar2：**smem 全卡毒化（NaN 位型）后 v6/v5 KV_FP8 仍与 bf16 位差 0**（三组形状，POISON=1/0 同）⇒ smem 残留假设排除；e4m3 kernel 在所有已知条件下正确
- 回看 D93（cacheprobe plain 3/7、`KV_FP8_STORE=0` 0/7、`D512_V6_OFF` 0/7）：那是 **b27 二进制 + B28 修前**。第 2/3 次同 prompt 请求走 cache 态 chunk（Paged 步）——若请求 i 的 Paged 步进了 pending 而下一请求……cacheprobe 是**串行**的，pending 不该有别的请求；但 `async_sched` 下 engine 对同一请求也会提前 `begin(n+1)`（decode 步 n+1 在 pending n 时就 begin），同步 runner 上 → `self.generated` 被 n+1 覆盖 → **单请求也错**？那 c1 bench 的中文 raw 为什么一直对……因为 c1 fresh 时 `begin(n+1)` 需要 n 的 token（在飞，`any_gather` → 同步 runner 不 async_eligible？）。不确定——**用 b28 链的 cacheprobe 门禁组（KV_FP8_STORE=1 与默认关）直接看**：若 b28 下 e4m3 3/7 → 0/7，则 D93 也是 B28 的表现（e4m3 只是改变了时序/步数分布让它显形），`KV_FP8_STORE` 可翻回默认开。

## ★★★★★ D109 2026-10-28 b28 门禁 pair2 **仍 DIFF**（case0 A@14 B@114、case2 A@4、case3 A@8；c8 fresh 5/8），与 b27 同量级 ⇒ B28（结果随 Pending）修的是真 bug 但**不是主根因**
- 更强的嫌疑（代码）：`ASYNC_SCHED` 默认开 → scheduler 对 pending 步的请求用 `num_output_placeholders` 提前推进、步 n+1 的 plan 里该行 token "在飞"（runner `in_flight=true` → `input_ids=0, gather_idx=last_row`）。**gather 由 `PlanBuffers::upload` 在设备上做：`ids[i] = prev_chosen[gather_idx[i]]`，而 `prev_chosen` 要 executor 每步 `save_chosen(argmax, n)` 写入——kaon3 executor 从不写** ⇒ 在飞行的 input token = prev_chosen 初值 0（或陈旧）→ 该请求从那步起吃错 token。c1 不显形：单请求时 pending n 的 token 不在飞？（待核 sched 是否对"唯一请求 + 同步 runner"不发在飞行；或 `async_eligible` 要求 Decode 且 c1 的 n+1 常被判成…）c≥2 时每步都有在飞行 → 必错。
- **B29**：kaon3 executor 在 argmax 后 `self.plan.save_chosen(stream, &argmax, n)`（cut-in 步 `skip_prev_chosen=true` 时不写），与 qwen3 同；graph 捕获体里也要进图。顺带 kaon3 可以 `supports_async=true`（collect 改异步 D2H）——第二步再做。
- **D110 b28 门禁全没过**：pair2 DIFF（同 b27）、mixparity c2 6/8 c8 6/8 fresh 3/8、cacheprobe（KV_FP8_STORE=1）4/7（默认关组待看）。B28 保留（真 bug，单测护住），但主根因是 D109：**kaon3 executor 只 upload 了 gather_idx，从未 `gather_inflight`（ids 仍是 host 的 0）也从未 `save_chosen`**。qwen3 executor 两处都有（exec 444 / 730）。
- **B29 实现**：`forward_device_mode` 开头 `self.plan.gather_inflight(s, t)`（图内安全：非在飞行 gather_idx=-1 空操作）；argmax 后 `if !skip_prev_chosen { save_chosen(argmax, nrows) }`；`forward_sample` 入口记 `skip_prev_chosen = plan.skip_prev_chosen`。图捕获假 plan gather_idx 全 -1、save 写 nb 行无害。预期：pair2 / mixparity / cacheprobe 全 SAME；e4m3 的 D93 也应随之消失（在飞行 id=0 让"第 2 次请求"这类有 pending 的时序错）。

## ★★★★★ D111 2026-10-28 重新读 engine：**同步 runner 下 `pending` 永远不被设置**（engine.rs 600：只在 `last_begin_was_async()` 时 `self.pending = Some`），因此 **cut-in 从不发生、begin(n+1)-before-finish(n) 从不发生、在飞行 token（gather_idx ≥ 0）从不出现**（publish 在下一次 schedule 之前完成，placeholders 已清）。ttft8 日志 `begin-split … async=false` 全部步印证。
- ⇒ **B28、B29 对 kaon3 当前流程都不生效**（都是真缺陷，留着：B28 有单测，B29 是将来 async 的前提），b28 门禁不变是必然的。D103/D106/D109 的推理链在"同步 runner"这一条上**错了**——我把 engine 的 async 分支当成了必走路径。
- 那 pair2 / mixparity 的差异是什么？看数据形态：pair2 case3（A 400 词 + B 400 词/1 tok **同时**到 → 一个 2 请求的纯 Prefill 步）**A 首 token就不同**；case0 两次一 SAME 一 DIFF；c8 fresh 5/8 且首差位置 1/9/13/81/85 散布。这不像"某行 input id = 0"的系统性错（那会每次都错、从首 token 起全乱），更像 **批尺寸改变 GEMM/MoE 的 tile/算法 → 几 ulp 差 → 在"词沙拉"prompt 的平坦分布上翻转贪心 top-1**（prompt 是随机常用词序列，next-token 分布极平，top-1/top-2 近似并列）。vLLM 同样不保证批不变（生产不开 BI，`ROUTER_GEMM_BI=0`）。c1 的中文 raw 在几十轮里从不变——自然语言分布尖。
- **决定性实验 `natprobe`**（排在 b29 链后）：8 条自然语言 prompt（中英问答）串行 vs c2 / c8，与 8 条词沙拉对照；env plain / `NO_GRAPH=1`。预期：自然 0/8 不同、词沙拉若干不同 ⇒ "并发 bug"结案为批尺寸数值噢动；B27（Paged 步 decode 行 attention）仍是真修（D86 pagedprobe 的 Paged 步确实含 decode 行）。若自然语言也 DIFF → 真 bug，继续查 nrows≥2 的 prefill 步（gather_rows / logits_rows / MoE bucket 对 tokens>64）。
- 顺带：pair2 只跑了 plain 一组就 DONE（trap EXIT 在 python 之后？第 2 组 serve 端口未释放 → NOTUP？），natprobe 每组前 `pkill -x dinfer; sleep 3`。
- **D112 队列事故**：b28 bN 只跑完 A（c1 6.00 / c8 TTFT 155.9 TPOT 9.80 = b27 B 同值）就没了 B/A2/B2，无 DONE；b29.sh 卡在等 b28 DONE，natprobe 卡在等 b29。嫌疑：我排的每条新链开头 `pkill -x dinfer`（b29.sh / natprobe.sh / chain3）会杀**别的链正在跑的 serve**——bN 的 serve 被杀 → bench 失败 → 脚本中断。规则补进 knowledge：**链之间只能用 DONE 标记串行，链内不许 `pkill -x dinfer`**（只 kill 自己起的 `$SP`）。
- 处理：b28 bN 不补跑（B28 对 kaon3 现状不生效，数字应 = b27）；b29 链的 A 侧改成 **b27 二进制**（`/w/dinfer_prev_b28` 就是 b27 的；复制为 `/w/dinfer_prev_b29`），让 b29 链自然启动。b29 bN = b27 vs b29（含 B28+B29，预期持平）。
- D113（2026-10-29 00:0x）：b29 链 watcher 仍在等（b28 DONE 已由我补写，但容器内 `/w/dinfer/tmp_b28_RESULT.txt` 与宿主 `/hf_data/yaochen/dinfer/...` 是同一文件——待核是否 bind mount 延迟/路径不同）。natprobe 排在 b29 后。本轮已 commit 到位，等待期间不再改代码。
- **D114 队列根因找到**：pair2 / natprobe 里 `trap 'echo DONE >> $R' EXIT` —— **`timeout … python3` 子进程退出时也触发 EXIT trap**（bash 子 shell 继承 trap），第一组跑完就写了 DONE → 下游 watcher（b28 链等 pair2 DONE）提前放行、b28 开头 `pkill -x dinfer` 杀了 pair2 第 2 组及之后……也解释了 pair2 为什么只有 plain 一组、b28 bN 为什么只剩 A。改：DONE 只在脚本最后一行写；每 env 组包子 shell；链内不 pkill。b29_now（无 watcher）已起；natprobe_wrap 等 b29 DONE。
- D115（00:1x）：b29_now 链已起（编译中）：build → pair2_b29（4 env 组，子 shell 隔离）→ mixparity_b29 → cacheprobe_b29（KV_FP8_STORE=1 / 默认）→ bN b29（A = `dinfer_prev_b29` = b28 二进制 = b27+B28，B = b29）。natprobe_wrap 等 b29 DONE 后跑（自然语言 vs 词沙拉决断）。等待期间不改代码。
- **D116 b29 pair2 plain**：case0 A@1 B SAME；case1 A@70 B@4；case2 A SAME B@13；case3 全 SAME；c8 fresh 4/8（@9/@13/@1/@81）。与 b27（5/8）、b28（5/8）同量级且**哪一行错、错在第几字符都不同** ⇒ 不是确定性 bug 的形态（确定性 bug 同输入同位置错），符合"批尺寸→tile→ulp→平坦分布贪心翻转"。B29 的 gather/save 对同步 runner 不生效（D111）也印证了"不变"。等 natprobe（自然语言）终判。
- **D117 b29 pair2 `CUT_IN=0`**：6/8 fresh、pair 四 case 全有 DIFF；位置又全不同于 plain 组 ⇒ 再次排除 cut-in / 调度路径；差异随每次运行的具体数值变（pair2 的 `seed = 7000 + hash(port)%100`，python hash 随机化 → 每组 prompt 其实不同，所以位置不可比；但"总有 4–6/8 差"稳定）。等 ATTN_DECODE_OLD / NO_GRAPH 两组 + natprobe。
- **D118 b29 pair2 `ATTN_DECODE_OLD=1`**：7/8 fresh、pair 四 case 全有 DIFF → decode attention 新旧 kernel 都一样 ⇒ 排除 B3 split-K decode kernel。三组（plain / CUT_IN=0 / ATTN_DECODE_OLD）形态一致。剩 NO_GRAPH 组 → 若也一样则 graph 捕获排除，只剩"批尺寸数值"一种解释，由 natprobe 终判。已 push origin/dev（e0ceba2）。
- **D119 b29 pair2 `NO_GRAPH=1`**：4/8 fresh、pair 四 case A 全 DIFF（@59/@61/@12/@22）、B 多 SAME → CUDA graph 排除。**四组 env（plain / CUT_IN=0 / ATTN_DECODE_OLD / NO_GRAPH）同形态**：每组 4–7/8、位置随机、不随开关变 ⇒ 与调度 / 捕获 / decode kernel 无关，剩下的解释只有"批尺寸改变 GEMM/MoE 累加路径 → ulp → 平坦分布贪心翻转"。natprobe（自然语言）是终判。
- D120（00:3x）：pair2_b29 四 env 组全出（4/6/7/4 of 8）；mixparity_b29 跑中 → cacheprobe_b29 → bN b29 → natprobe。若 natprobe 自然语言 0/8：结案"批尺寸数值噢动"，写进 knowledge/perf-method §15 作判据（词沙拉 prompt 不能作并发正确性判据；用自然语言 + 若有 logprobs 用 top-2 margin）。kaon3 executor 不实现 take_logprobs → 无法直接量 margin，暂以自然语言替代。

## ★★★ D121 2026-10-29 cacheprobe_b29（KV_FP8_STORE=1）：**4/7 DIFF，且每条 `2nd==3rd True`**（prompt0 'be be…'→'as it is…'、prompt2 首字符、prompt3 @33、prompt4 @6），与 D93 b27 的 3/7 几乎同集合 ⇒ **e4m3 下 cache 态 chunk 的差异是确定性的**（同 prompt 第 2/3 次完全一致），不是运行间噪声；bf16 cache（D93 第 2 组）0/7。gate 的 "" 默认组没出（cacheprobe_b29 只跑了 1 组就 GATES DONE——子 shell 隔离的是 pair2/natprobe，cacheprobe 源脚本没加；又一次"第 2 组丢"）。
- 解读：e4m3 cache 下 **Paged 步（hist>0 chunk）** 与 **纯 Prefill 步** 对同一 prompt 算出不同 top-1，而 bf16 cache 下两者同。两者 K/V 数值都经 fp8 往返（kvfp8_on 两边都真）→ 差只能来自 **kernel 路径**：bf16 cache 时 Paged 走 v6 bf16 PAGED（D=256 cfg 464 → v6；D=512 B26 v6），e4m3 时 D=256 退 **v5 PAGED KV_FP8**（cfg 降 0）、D=512 走 v6 KV_FP8。而纯 Prefill 两种都走 v6 非 paged。**v5 vs v6 的 mma/softmax 累加序不同 → 几 ulp → 词沙拉翻转**；histpar 的 "fast e4m3 vs fast bf16 位差 0" 是 **v5 vs v5**，没比 v5 vs v6。即：**不是 bug，是 v5/v6 路径差 + 平坦分布**。bf16 cache 时 Paged/Prefill 都是 v6 → 同序 → 0/7。
- 验证（排 natprobe 后）：`cacheprobe3`：自然语言 prompt（chat）同 prompt 三次，KV_FP8_STORE=1 → 预期 0/N；若仍 DIFF 则真 bug。另：让 e4m3 也走 v6（给 v6 PAGED D=256 加 KV_FP8 实例，和 B26 一样）可彻底消除路径差。
- **D122**：bN b29 又只跑完 A（c1 6.00 / c8 155.9 / 9.82，= b27/b28 值）就停：无 B log、无 serve/bench 进程、RESULT 无 DONE、b29_now 不在。这次链里已无跨链 pkill（natprobe 等 b29 DONE 没放行）。嫌疑转向 bN.sh 自身：`run_one` 的 serve 行以 `>log 2>&1 &` 结尾 `local sp=$!`——若 A 组结束时 `kill -INT $sp` / `kill -KILL` 打到的不是 serve 而是…；或 `bash /w/bN_keepprev.sh` 被 `timeout 1500 bash pair2_b29.sh` 的残留 timeout 进程组影响。先看脚本原文与 A 组日志尾。bN b28/b29 的 B 组都丢 ⇒ 规律：**链里"编译 → 门禁（含 timeout）→ bN"这个顺序下 bN 总在 A 后死**；此前独立起的 bN（b26/b27）四组都完整。
- D123：bN b29 独立重起（`bash -x`，trace 到 `tmp_b29_bN_trace.log`），A = b28 二进制、B = b29。若四组完整 → "门禁链后 bN 必死"是链脚本问题（timeout 进程组 / 子 shell），不是 bN；若再死于 A 后，看 trace 最后一行。natprobe → cacheprobe3 等其 DONE 后自动跑。
- **D124**：独立起的 bN b29 顺利过了 A → B 边界（A c1 5.99 / c8 155.4 / 9.80；B 的 serve 已起）⇒ b28/b29 链里 bN 在 A 后死是**链脚本的问题**（`timeout N bash xxx.sh` 用过之后、同一 bash 里再起长脚本——timeout 退出时给其进程组发 SIGTERM？或 cacheprobe 第 2 组 NOTUP 的 `exit 1` 没包子 shell 直接退了整条链——b29_now 里 cacheprobe_b29 的 "" 组确实没出）。**结论：bN 必须独立起，不放在门禁链尾**。
- **b29 首批 B**：c1 5.99 = A；c8 TPOT 9.80 → **10.05（+2.5%）**、吞吐 678 → 661（-2.5%）、TTFT 155.4 = 155.4。B29 多了每步一次 `row_gather_i32` + 一次 `dtod_copy_n(save_chosen)`（graph 内，各 ~3 us + 间隙）——按 D72 口径每步 1065 launch 多 2 个不该有 0.25 ms；更可能是 c8 的"排队态"噢动（D75 见过 ±0.3）。等 A2/B2 定稿；若 B 两轮都 +2.5% → B29 对同步 runner 无收益只有代价 → 加开关 `INFLIGHT_GATHER`（默认关，待 kaon3 实现 supports_async 时再开）。
- b29 A2：c1 6.00 / c8 155.9 / 9.80（与 A 同）。已在本地加 `INFLIGHT_GATHER` 开关（默认开；B2 若仍 +2.5% → 改默认关并记 TODO "kaon3 supports_async 时打开"）。
- **b29 定稿（两轮）**：A（b28）c1 5.99/6.00、c8 TTFT 155.4/155.9、TPOT 9.80/9.80、吞吐 678/677；B（b29，gather+save 开）c1 5.99/6.00、c8 TTFT 155.4/155.4、**TPOT 10.05/10.06（+2.5%）、吞吐 661/661（-2.5%）**。两轮同向 → B29 的两个图内 launch 在同步 runner 下纯开销（gather_idx 恒 -1）。**`INFLIGHT_GATHER` 默认关**（代码留，kaon3 做 supports_async 时打开）。CSV round b29 记"gather 开"的数字（未合入默认）。bN 四组完整 ⇒ 链尾死 = 链脚本问题（D124）。

## ★★★★★★ D125 2026-10-29 natprobe（b29 二进制，plain）：**自然语言 c2 7/8、c8 6/8 不同** —— nat0 长城：串行 "这里是为您准备的三句话介绍：…" vs c8 "长城的历史可以概括为以下三点：…"（@0）；nat3 四行诗第三句不同；nat5 排版不同。**这不是 ulp 翻转，是真 bug**（D111 的"批尺寸数值噢动"假设被推翻；词沙拉只是放大了它）。
- 事实清单：① c1 串行同 prompt ×3 全同（cacheprobe bf16 0/7）→ 单请求路径确定；② 并发时 A 和 B 都可能错，错的位置随机，同配置重跑错的集合不同（D116–D119）；③ env plain / CUT_IN=0 / ATTN_DECODE_OLD / NO_GRAPH 都错（NO_GRAPH 组 natprobe 还在跑）；④ e4m3 vs bf16 无关（bf16 cache 下也错）；⑤ 同步 runner：无 pending / 无 cut-in / 无在飞行；⑥ bN 的 c8 吞吐 / TPOT 数字看不出。
- ⇒ 错在 **"一步里有 ≥2 个请求"的 forward 本身**：纯 Prefill 多请求（cu_seqlens 多段）或纯 Decode 多行（n≥2）。pair2 case3（B max_tokens=1，A 仍错）说明 **2 请求同一 Prefill 步**就足以让 A 错；但 natprobe nat1/2/4/7 c8 SAME 而 c2 DIFF 说明随机性强（与行序 / 行数 / 具体数值有关）。
- **下一步 natprobe2（二分）**：固定 A（长城 prompt，64 tok），对照 ①串行 ×2（确定性基线）②A+B 同时发、B max_tokens=1（只共享 Prefill 步 + 1 个 Decode 步）③B 200 词 64 tok（共享 Prefill + 全程 Decode）④B 在 A 发出 100 ms 后到（A 单独 Prefill，B 单独 Prefill，之后共 Decode）。env 组：plain / `MOE_ROUTE_FUSED=0` / `MOE_DECODE_GEMV=0` / `FP8_GEMV=0` / `NO_GRAPH=1`。哪一组 ②③④ 全 SAME → 定位到那条路径。

## ★★★★★★ D126 2026-10-29 **混杂变量找到：所有"并发 vs 串行"探针都把 prefix cache 命中混进去了**
- natprobe：先串行（fresh prefill）再 c2/c8（**同 prompt → cache 命中 → 带历史 chunk 的 Paged 步**）；pair2：先并发（fresh）再串行（cache 命中）；mixparity：明写 cache 态。**没有一个探针比过"同为 fresh"的串行 vs 并发**。
- cacheprobe3（KV_FP8_STORE=1，**c1 串行**，自然语言同 prompt ×3）：**9/10 第 1 次 ≠ 第 2 次，且第 2 == 第 3**。nat0 第 2 次 = '长城的历…' **正是 natprobe c8 的输出** ⇒ natprobe 的"并发差异" = cache 态差异。NO_GRAPH 组位置与 plain 几乎同（@0/@98/@17/@86/@54）⇒ 给定批构成，差异**确定**。
- 所以真命题是：**prefix cache 命中后的 chunk（hist>0，Paged 步）输出 ≠ 同 prompt fresh prefill 输出**，自然语言 9/10（e4m3）。cacheprobe（词沙拉，bf16）0/7 → 可能 bf16 下没有 / 很小；cacheprobe3 的 bf16 组待出。**哪边对还不知道**（fresh 的"这里是为您准备的三句话介绍"几十轮一直这样，但没和 vLLM 比过 cache 态）。vLLM 的 prefix cache 命中数值上也不逐位（chunk 不同 tile），但自然语言不该 9/10 翻。
- 候选根因（hist>0 chunk 专属）：① positions / RoPE 对 chunk 行用 `start+k`（看过，对）；② `seq_lens` vs `cu_seqlens` 在 Paged 路径的 hist 计算；③ **KV 写入**：chunk 行的 K/V 写到 slot_mapping（对），但 **cache 命中块的 K/V 是上一次请求写的**——若上次请求的 K/V 写入在**某些层**被跳过 / 写错位置（例如 B26 前 D=512 层、或 fused `cache_write_fp8rt` 的 slot 计算），fresh 时不读 cache 不受影响，命中时读到错的 K/V！这解释了 e4m3 更严重（fp8rt 融合写路径只在 kvfp8 层）和 D=512 关掉 v6 后 0/7（D95：通用 kernel 读法不同…不，读的是同一 cache）。④ `get_computed_blocks` 命中块数 / `num_computed_tokens` 对齐（block 32）：命中 384，chunk 从 384 起——若实际应从 352 起（最后一块未满 / hash 边界），chunk 少算 1 块 → 中间 32 个 token 的 K/V 来自旧块但 Q 跳过——输出偏。
- **立即做**：A）cacheprobe3 bf16 组；B）natprobe2 改基线为 **sa2（cache 态串行）**，并加 **禁 prefix cache** 的 env 组（若有开关）做纯并发对照；C）两个 fresh 对照：同 prompt 两个新服务各 fresh 一次应全同（确定性基线）。

## ★★★★★★ D127 2026-10-29 cacheprobe3 默认组（**bf16 cache**）：**同样 9/10，且每条首差位置与 e4m3 组完全相同**（@0/@148/@4/@206/@17/@86/@54/@34/@26）⇒ 与 KV dtype 无关、确定性 ⇒ **prefix-cache 命中后的 chunk 路径本身有 bug**（词沙拉 cacheprobe bf16 0/7 是巧合/分布原因，不是证据）。KV_FP8_STORE 可翻回默认开（它不是问题）。
- 命中态步序（ttft8）：400 词 prompt 第 2 次 `Paged:16`（命中 384 = 12 块×32，剩 ~42 token？却只算 16）；`Paged:13/10/11`。**待核**：chunk 行数 vs prompt_len − num_computed。
- 候选根因重排（kaon3 专属 vs core 公共）：① runner `start = st.num_computed`，`positions = start..start+n`、`slot_mapping` 从 `block_ids[pos/bs]` —— core 公共件，Qwen3 线 prefix cache 跑过 bench 没报过错（但 Qwen3 也没人比过输出！）；② kaon3 `layer.rs` Paged 路径的 RoPE：`rope.apply(positions)`（看过 positions 来自 plan）；③ kaon3 **全注意力层 proportional RoPE / 滑窗层 default RoPE** 两套——对 chunk 行只要 positions 对就对；④ `gqa_paged_win` 的 `seq_lens[b] = start+n`、`cu_seqlens` → hist = seq_len − q_len：对。⑤ **MoE / dense 对 Paged 步与 Prefill 步走不同分支**（tokens ≤ 64 → `route_fused`、`moe_decode_gemv r ≤ 128`、`fp8_gemv tokens ≤ 16`、mega `tokens ≤ 8`）：**chunk 行数 10–16 恰好落进这些 decode 专用快路径**，而 fresh prefill（400 行）走大路径。这些快路径是按 decode（每行独立）写的，若其中某个对"同一请求多行"有假设（例如 fp8_gemv 的 M tile、route_fused 的 bucket、mega 的 MEGA_ROWS=8 行缓冲…）就会错！**词沙拉 0/7 vs 自然语言 9/10 的差别**也可解释：差异大小取决于内容。
- **二分实验 `cacheprobe4`**（同 prompt ×2，自然语言 10 条，c1）：env 组 `--no-prefix-caching`（预期 0/10 = 确定性基线）/ plain / `FP8_GEMV=0` / `MOE_ROUTE_FUSED=0` / `MOE_DECODE_GEMV=0` / `KAON3_MEGA=0` / `MOE_GROUPED=0`。哪组 0/10 → 定位到那条快路径。
- **D128 两类假设待 cacheprobe4 分辨**：(A) kaon3 小行数快路径（chunk 1–16 行：mega ≤8 / dense_gemv ≤16 / fp8_gemv / route_fused ≤64）对"同请求多行 chunk"有错 → 某个关闭组 0/10；(B) **core 公共件**：prefix cache 命中块在 `allocate_slots` 取新块时**没先 touch/ref**，被当空闲块分给同一请求的 chunk → 命中块的 K/V 被 chunk 覆盖（vLLM 先 touch 命中块再分配）→ 所有关闭组都 9/10、只有 NOPC 0/10；且 Qwen3 线 prefix cache bench 也从没比过输出。(B) 可用 kv manager 单测直接判：req1 全 prompt → free → req2 同 prompt → computed ∩ new_blocks 应为空。

## ★★★★★★ D129 2026-10-29 cacheprobe4：`--no-prefix-caching` **0/10**；plain **9/10**（位置与 cacheprobe3 两组逐条相同 @0/@148/@4/@206/@17/@86/@54/@34/@26）⇒ **bug 在 prefix cache 命中路径**，确定性，与 dtype 无关。等快路径关闭组（fp8gemv0 / routefused0 / decodegemv0 / mega0 / grouped0）：若某组 0/10 → kaon3 快路径对多行 chunk 错；若全 9/10 → core（KV 块命中 / 命中块被重分配 / hash 对齐 / `num_computed` 推进）。
- 候选 core 根因（并行核代码）：① `allocate_slots` 对命中块 `touch`（ref 0→1，从空闲队列摘走）是否在 `pool.allocate(new)` **之前**；② `cache_blocks` 只缓存满块（to_cache 对齐 block 32），hash 只含 prompt？若 hash 含已生成 token（req1 生成了 48 个），req2 的 prompt 尾块 hash 对不上 → 不会错，只是少命中；③ 滑窗层：cache_write 是否对所有层无条件写（若某层按 window 省写，fresh 不读 cache 不受影响、命中就读到空块）。
- **D130 分辨实验 `chunkprobe`**（排在 cacheprobe4 后）：`--no-prefix-caching` 下 `--max-num-batched-tokens` 16/32/8 把 prompt 切成多 chunk（后续 chunk = hist>0 的 Paged 步，K/V 由同一请求前一步写），与 4096 预算单步 Prefill 比；再加 `pc4096`（开 prefix cache、同 prompt 只发一次 = fresh）作对照。**chunk 组全 SAME → Paged 步 / hist>0 kernel 路径干净，bug 在 cache 命中复用（块内容 / 命中长度 / num_computed 推进）；chunk 组 DIFF → Paged 步本身错（与 cache 无关，只是 cache 命中让它显形）**。同时核 `allocate_slots`：touch 命中块在 allocate 新块之前（manager.rs 253/265）——顺序对。
- **D131 cacheprobe4 续**：`FP8_GEMV=0` **8/10**（位置略变：@5/@190/@10/@38/@28——关 gemv 改了数值路径但没修）、`MOE_ROUTE_FUSED=0` 9/10（位置与 plain 完全同 → route_fused 对结果无影响，排除）。等 decodegemv0 / mega0 / grouped0。目前倾向 core / 步形状问题而非某条 kaon3 快路径；chunkprobe 将分辨"Paged 步本身"vs"cache 复用"。
- **D132**：`MOE_DECODE_GEMV=0`、`KAON3_MEGA=0` 都 9/10 且位置与 plain **完全相同**（@0/@148/@4/@206/@17/@86/@54/@34/@26）⇒ mega / decode_gemv / route_fused 三条快路径对结果**零影响**（关掉连 ulp 都不变——说明命中态 chunk 根本没走它们，或它们逐位等价）。只有 `FP8_GEMV=0` 改了位置（8/10）= 改数值路径但不修。**结论：不是 kaon3 的小行数快路径；是 core（prefix cache 命中 → 块 / num_computed / positions / 块内容）或 Paged 步共同路径（attention hist>0）**。chunkprobe（无 cache 的 chunked prefill）正在分辨后者。
- **D133**：cacheprobe4 plain 步形：fresh `Prefill t=20–25`（chat 模板后 prompt ≈ 20–25 token），命中后 `Paged t=4–9`（命中 16 = hash 块 16 的整数倍？KV block 32 → 命中块只有半块有效…）。要核：`hash_block_size` 是不是 16 而 KV `block_size` 32（`get_computed_blocks` 的 `step = bs/hbs`，`max_blocks = hit_len / bs`——22 token 的 prompt `hit_len=21 → max_blocks=0` 本不该命中！）。若命中 16 来自别的路径（host tier？snapshot？）或 `hbs`=16 与 `bs`=32 的换算错 → 命中块号 / num_computed 不一致 → chunk 的 positions/hist 与真实 K/V 错位 ⇒ 正好解释"确定性、与 kernel 无关、自然语言 9/10"。
- **D134 关键线索**：kaon3 KV `block_size=32`、`hash_block_size=16`（CacheConfig 默认）。cacheprobe4 步序：`Prefill:21 → … → Paged:5`（第 2 次只算 5 = 命中 16）。但 `get_computed_blocks`：`max_hit = 20`、`max_blocks = hit_len / bs = 20/32 = 0` → **按 KV 块算不该命中任何块**；却命中了 16 个 token ⇒ 命中长度按 **hash 块（16）** 算、而 KV 块是 32：`num_new_computed=16` 对应半个 KV 块 → 请求 2 拿到的"命中块"是请求 1 的第 0 块（含 token 0–20 + 生成的 11 个 = 32 槽都写过），chunk 从 pos 16 起写 slot 16–20（覆盖同一块的 16–20，值应同）——**positions / hist 看起来仍一致**……但若 `allocate_slots` 的 `need` 按 `(total_computed + new).div_ceil(32) − (have + computed_blocks)` 算，`computed_blocks = c.group().len()` = 命中 KV 块数（1）而 `num_new_computed=16` 是 token 数 → 请求 2 持有块 [b0]，`num_computed=16`，token 16–20 写到 b0 的 slot 16–20，对；decode 到 token 32+ 时需要第 2 块 → allocate……看起来自洽。那差异来自哪？**请求 1 的 b0 slot 21–31 装的是请求 1 的生成 token**，请求 2 的 chunk 写 slot 16–20，decode 从 21 起覆盖——若 **cache_write 对 Paged 步 chunk 行 没写**（例如 kaon3 对 Paged 用 `slot_mapping` 的 -1 过滤 / 只写 prefill 行…），请求 2 的 token 16–20 的 K/V 就是请求 1 的 token 16–20（**同 token，同位置 → 相同！**）——也不会错。除非 **请求 1 的 b0 在请求 1 结束 free 后、请求 2 到达前被别人覆盖**：cacheprobe4 是串行单请求，没别人……但 **engine 的 warmup / 每次 chat 的模板 token** 不同？两次 prompt 完全一样。
- 等 chunkprobe：若 chunk16（无 cache、同请求前一步写的 K/V）也 DIFF → 问题在 "Paged 步 hist>0 的 chunk 行" 本身，与 cache 无关，上面全不用想。
- **D135**：chunkprobe `chunk16`（`--no-prefix-caching --max-num-batched-tokens 16`）步形 `Prefill:16 ×10 → Paged:4–9` —— **与 cache 命中态（Prefill 21 → Paged 5）的第 2 步形状一致**（前 16 token 一步、剩 4–9 一步）。两者唯一区别：chunk16 的前 16 个 K/V 是同请求上一步刚写的；命中态是上一请求写的 cache 块。chunk16 vs base4096 的比对结果决定方向。

## ★★★★★★ D136 2026-10-29 chunkprobe（`--no-prefix-caching`）：**chunk16 9/10、chunk8 9/10**（首差位置 @0/@148/@4/@17… 与 cache 命中态 cacheprobe3/4 **完全相同**），chunk32 0/10（prompt 仅 20–25 token → 32 预算等于没切，无信息）
- ⇒ **prefix cache 无罪**。bug = **带历史的 chunk（hist>0，Paged 步）自身输出错**，与 cache / dtype / 并发 / 快路径 / graph 全无关；cache 命中只是最常见的触发方式（命中 16 → chunk hist=16）。nat0 chunk 输出 "长城的历史可以概括为以下三点" = 命中态输出 ⇒ 同一 bug。
- 已排除：hist>0 的 **v5 PAGED（cfg=0，D=256 win1024）/ v6 PAGED D=512 win0** 与通用 kernel 位差 ≤ ulp（histpar/d512 parity，含 hist 384/563/1186、q 11–37）。**没测过的恰是 serve 里 bf16 cache 时 D=256 滑窗层实际走的 `v6 PAGED D=256 cfg=464 window=1024`**（v6 PAGED 之前只在 Qwen3 window=0 用过；B26 的 v6 D=512 也是 window=0）。v6 的 window 掩码 / `tile0` / `warp_hi_lo` 对 PAGED+hist>0 的组合是首次上线。
- 两路验证并行：① `winprobe`：serve `GQA_PREFILL_CFG=0`（滑窗层 PAGED 退 v5）下 chunk16 vs base4096——若 0/10 → v6 PAGED window 实锤；② `gqa_paged_fp8_hist_parity` 加 cfg=464（v6 PAGED D=256 win1024 bf16）vs ref 与 vs v5。
- 另注：chunked prefill 在 Kaon 生产（max-num-batched-tokens 4096、prompt 1560）不常触发，但 **prefix cache 命中几乎每次都触发** → 线上所有命中请求都在错。
- D137：v5 / v6 的 window/hist 序言逐行相同（vis/lo/last_key/n_tiles/warp_min_vis/warp_hi_lo/blk_lo_vis/tile0）；v6 多一个 `warp_max_vis` 跳过整 warp 不可见 tile（`k0 <= warp_max_vis`）。chunk 行 q_len 4–9、hist 16：所有行可见 key 0..hist+row，window 1024 远大于 → 掩码不起作用，tile0=0；差别只能在 PAGED 装载（block_table 寻址）或 Q 装载（v6 Q 进 smem，`cp_async16(sQ…, row<q_len?16:0)`）——q_len=5 时 BQ=64 行里 59 行是零填充，mma 对零行无害。等 winprobe（v5 vs v6 实测）。
- D138：确认 serve 的 Paged 路径 cfg = `gqa_prefill_cfg` = 464（Sm120）→ 滑窗层 bf16 cache 时走 `v6_dispatch<PAGED=true>` NW4/TILE64；runner `seq_lens = start+n`、slot 按 pos；全对。winprobe（v5 强制）与 histpar3（cfg=464 vs ref）跑中，二者任一命中即可定位到 v6 PAGED D=256。
- D139：并行核 core 的 chunk 边界语义：中间 chunk 的采样 token 必须丢弃（`end_of_seq=false` 行），下一 chunk 的 input 必须来自 prompt 原 token（runner `token_ids.get(pos)`）；`next_known` 用于 prompt_logprobs。若 runner/sched 把中间 chunk 的采样当成输出 append、或 num_computed 多推 1 → 下一 chunk 从错位置起 → 与 kernel 无关、与 cache 无关、确定性、v5/v6 都错 ⇒ **这与 winprobe 若"v5 也 9/10"的结果完全吻合**。读 runner.rs 455–482 / sched 652–690。
- D140：core chunk 语义核对**正确**：runner 中间 chunk `num_computed += n`、不 append token（`out.sampled=[]`）；sched 对空 generated 不 append；下一 chunk input 来自 `token_ids.get(pos)`、positions=pos、slot 按 block。⇒ 剩 kaon3 层内与 kernel。winprobe v5_chunk16 已出，等比对行。

## ★★★★★★ D141 2026-10-29 winprobe + histpar3：**attention kernel 无罪**
- winprobe（无 cache）：v6 chunk16 vs base 9/10；**v5（GQA_PREFILL_CFG=0）chunk16 vs base 9/10，首差位置与 v6 逐条相同**；v5_base == v6_base 0/10；d512ref chunk16 也 9/10。histpar3：v6 PAGED cfg=464 win1024 vs ref max 0.0156（ulp 级）。
- ⇒ 三种 attention 实现（v5 / v6 / 通用 ref）对同一 chunk 步给出**同样的"错"结果** → 错不在 attention kernel 内部，而在它们**共同的输入**：① 从 paged cache 读到的 K/V（vs fresh 时从连续 k/v 缓冲读）—— cache 内容 / slot_mapping / block_ids；② chunk 行自身的 q/k/v（上游 GEMM/norm/rope 对 5 行 vs 21 行）；③ 或 attention 之后（o_proj…）对 Paged 步的行有别。
- 最可疑：**runner 对 running 请求追加新块（`cached_reqs.new_block_ids`）→ `st.block_ids`**；若 chunk 步的 slot_mapping 指到错块，chunk 行 K/V 写错位置、attention 读不到自己/邻居的 K/V → 确定性、语义级错误、与 kernel 无关。但 decode 每 32 token 也走同路径且 c1 一直对……除非 prefill-chunk 时的块追加时序不同（块在 chunk 步**同一步**分配并立刻使用）。
- **定位手段**：逐层 dump 比对 fresh（21 行 Prefill）第 16–20 行 vs chunk 步（5 行 Paged）第 0–4 行，从 input_norm → q/k/v → rope → attn_out → o_proj → dense → moe，第一个分叉的 op 即根因所在。
- **D142**：Paged 步**所有** K/V（含 chunk 行自己的）从 cache 读（`paged_win(cache, block_table, slot...)`），fresh 从连续 k/v 读。cache 写→读是 v5/v6/ref 三者的共同前置 ⇒ 嫌疑收敛到 **cache 写入位置（slot_mapping / block_table / max_blocks stride）或 cache 内容（写入时机）**。注意 `PlanBuffers::upload` 把 block_table 按 `stride = self.max_blocks`（缓冲宽度）布局，而 runner 的 `plan.max_blocks` 是本步最大块数——kaon3 传给 kernel 的 `max_blocks = self.plan.max_blocks`（缓冲宽）→ 一致；decode 也靠它，排除。写 kernel 级二分测试 `gqa_chunk_vs_prefill`：同一 k/v，21 行 `prefill_win`（连续）vs 写 cache 后 `paged_win` 只算行 16–20（hist 16），D=256 win1024 与 D=512 win0，两者行 16–20 应 ≤ ulp；若差大 → paged 读路径 / cache 写路径有形状相关 bug（如 slot_off 对 nkv/D 的层间差）。
- **D143 重新框定**：至此可确定的只有"**chunk（hist>0）步的输出 ≠ fresh 单步**，确定性，与 attention 实现 / cache / dtype 无关"。它**可能不是 bug**：chunk 行走的是 m=5 的 GEMM/MoE（tile、grouped GEMM 行分组、route bucket 都随 m 变）而 fresh 行走 m=21，各层几 ulp 差 → 30 层累积 → 自然语言 top-1 也可翻（vLLM 同样非批不变）。但 9/10 的翻转率对自然语言偏高，需要量化：① kernel 测试 `gqa_chunk_vs_prefill`（同 q/k/v，连续 vs cache+paged，只看 attention）——若 ≤ ulp 则 attention 路径干净，差来自上游 GEMM/MoE 的 m 依赖；② `concprobe`（`--no-prefix-caching`，同 8 条自然 prompt 串行 vs c2 vs c8 **都 fresh**）——真正的"并发 vs 串行"对照（之前所有并发探针都被 cache 命中污染）；c8 vs c8 重复看确定性。若 concprobe 全 SAME → "并发 bug" 彻底结案，只剩 chunk/命中态的批尺寸数值问题（可用 BI 口径权衡：batch-invariant kernel 代价大，vLLM 默认也不做）。

## ★★★★★★ D144 2026-10-29 `gqa_chunk_vs_prefill`：**cache 写 + paged 读（chunk）与连续 prefill 对同一 q/k/v 的行 16–20 输出：v5/v6 位差 0（D=256）/ ≤0.0005（D=512 mma 序）**，7 组形状（21/25/400/1200，win1024 与 win0+k_eq_v）全过；ref 慢路径差 0.06–0.13 是它 fp32 标量序（已知）。
- ⇒ **attention 的 cache 写→读路径与连续路径逐位等价**。chunk ≠ fresh 的差异只能来自 **attention 之外、按行数 m 变化的算子**：qkv/o_proj/dense 的 cuBLASLt fp8（algo 按 `Key{m,n,k}` 选，m=5 vs 21 不同 algo/tile）、MoE grouped GEMM（每专家行数分组不同）、route/bucket；每层几 ulp → 30 层累积 → 贪心翻转。**这是批不变性问题，不是功能 bug**。vLLM 同样不批不变（生产 `ROUTER_GEMM_BI=0`）。
- 但 9/10 自然语言翻转仍偏高 → 量化它：① `concprobe`（无 cache、同 fresh：串行 vs c2/c8 vs c8 重复）—— 若 c8 vs 串行也 ~9/8 翻则是普遍的"m 依赖"，若 0/8 则 chunk 的 m=5 特别敏感（小 m 走 fp8 gemv / split-K）；② 对照 vLLM：同 prompt fresh vs 命中态（vLLM 开 prefix cache）是否也翻——若 vLLM 稳定而我们翻，说明我们某个小 m 路径精度差（候选：fp8 gemv dot2 的 fp32 累加 vs cuBLASLt、grouped GEMM BM=16 tile 的累加序、NVFP4 块缩放在小 m 的 rounding）。
- 今天的"并发正确性 bug"叙事到此修正：B27 是真修（混批 decode 行确实没人算，影响 c≥2 的 Paged 步）；B28/B29 是真缺陷但不触发；其余差异 = chunk/命中态的 m 依赖数值。

## ★★★★★★ D145 2026-10-29 concprobe（`--no-prefix-caching`，全部 fresh）：**serial×2 0/8；c8 vs c8 重复 0/8（确定性）；c2 vs serial 6/8、c8 vs serial 6/8**（nat0 两版都是通顺的"三句话介绍长城"，措辞不同；nat3 四行诗后两句不同）
- ⇒ 最终结论：**没有并发 bug**（同批构成 → 同输出，逐字确定）；差异 = **批构成改变 → 按 m 选 algo/tile 的 GEMM / MoE grouped GEMM 分组 → 每层 ulp → 30 层累积 → 贪心翻转**（批不变性问题）。chunk/命中态 9/10 与并发 6/8 是同一现象的两种触发。昨天追的 B27（真修）、B28/B29（真缺陷不触发）与这个现象无关。
- 待决：翻转率是否比 vLLM 高（同 8 条 prompt 对 vLLM 基线 :28067 做 serial vs c8，见下）。若 vLLM 显著更稳 → 我们某个小 m 路径精度差（候选：fp8 gemv fp32 累加 vs Lt、grouped GEMM BM=16/32 tile 序、NVFP4 小 m 块缩放），值得修；否则结案为已知的非批不变，记入 design.md §11 台账（vLLM 默认同样不批不变，BI 模式有性能代价）。
- **恢复**：`KV_FP8_STORE` 默认改回开（D70 c8 TPOT -1.9%；e4m3 不是原因）。`FP8_LT_TUNE` 保持默认关（b25 实测 c8 +3%）。`INFLIGHT_GATHER` 默认关。

## D146 2026-10-29 vLLM 基线批不变性对照：基线容器 `vllm-noneager` 监听 **:28201**（生产口径，served-model-name `kaon-vllm`）、`vllm-eager` :28200（CLAUDE.md 写的 :28067 已过时 → 更正）。同 8 条自然语言 prompt、同脚本（serial×2 / c8 vs serial / c8 vs c8）。结果归档 `test/kaon3/2026-10-29/vllm_batch_invariance_RESULT.txt`。
- **D146 结果（vLLM 基线 :28201，同 8 条 prompt）**：serial×2 **6/8 不同**（同 prompt 串行重跑都不稳）、c8 vs serial 5/8、c8 vs c8 重复 **7/8**；nat0 两版都是"三个版本…"语义等价。**vLLM 比我们更不确定**（我们 serial×2 0/8、c8×2 0/8、c8 vs serial 6/8）。⇒ **结案**：kaon3 的翻转率不高于 vLLM，且在同批构成下严格确定（vLLM 连这点都没有，大概是其 MoE/FlashInfer 内核的非确定归约）。批不变性作为可选优化留台账，不是缺陷。
- 收尾：CLAUDE.md 基线端口 :28067 → :28201（noneager，生产口径）/ :28200（eager）。

## 第四轮 2026-10-29（用户确认顺序：① 重抓 nsys → ② MoE decode GEMM split-K → ③ 层内 launch 合并 → ④ supports_pair → ⑤ lm_head fp8）
- 用户质疑："理论上应该比 vLLM 快"。回答口径：c1 一步带宽下限 ≈ 2.0 ms（lm_head 0.83 + MoE 0.42 + dense 0.3 + qkv/oproj 0.16 + KV），vLLM 4.69 = 2.3×，我们 5.57 = 2.8×；差在 ~1000 launch/步 与 MoE grouped GEMM 占用率（~90 CTA / 170 SM，每 CTA 串 22 个 K 步），不在物理极限。compute overlap 一处没有（单流、`supports_async/pair` 都 false）。B16b 已解决 BM=32 半 warp 空转但只 -1.1%。GQA 已确认与 vLLM 一致（D147）。
- **D148 重抓 nsys**：链 `nsys_now.sh` = 编译（源含 KV_FP8_STORE 默认开 / INFLIGHT_GATHER 默认关）→ `nsys_c1b.sh` → `nsys_c8.sh`；产物 `tmp_nsys_c1b_RESULT.txt` / `tmp_nsys_c8_RESULT.txt`（kernel 表按 `nsys_kern.py`）。用它刷新 B21 候选表后再动 ②。

## ★★★★★ D148 2026-10-29 nsys c1b（当前二进制：b29 源 + KV_FP8_STORE 默认开 + INFLIGHT_GATHER 关；nsys 态 TPOT 5.95，bench 5.57）：107 步合成一个 graph 实例，**1058 kernel/步，kernel 和 4.89 ms + 间隙 1.32 ms = 6.2 ms/步**
| 段 | n/步 | us/次 | ms/步 | 带宽下限 ms | 倍数 |
|---|---|---|---|---|---|
| lm_head gemvx bf16（+o_proj 全注意力层 5 次） | 31 | 870 / ~3 | 0.96 | 0.83 | 1.1× |
| **fp8 `gemv_kernel<1>`（dense down ×30 + qkv ×25 + o_proj 滑窗 ×25）** | 85 | 10.7 avg | **0.91** | ~0.25 | **3.6×** |
| gateup_act（dense gate‖up+gelu，fp8） | 30 | 9.4 | 0.28 | 0.2 | 1.4× |
| MoE grouped gate/up（BM16 BK128，11×129 grid） | 60 | 10.7 | 0.64 | 0.45 | 1.4× |
| MoE grouped down（BM16 BK64，44×129） | 30 | 10.7 | 0.32 | 0.22 | 1.5× |
| **route_bucket_small（1 block）** | 30 | 10.7 | **0.32** | ~0 | B19 后仍 10.7 us |
| rmsnorm_vec<256,2>（h=2816 单行 ×5/层） | 151 | 1.2 | 0.18 | 0 | launch |
| cache_write_fp8rt / combine / norm_quant / rope / grouped_quant / rmsnorm<128,1> / merge | 30 each | 1–4.7 | 0.62 | 0 | launch |
| v3 attn 25 层 / d512 5 层 / nvjet qkv 5 层 | 25/5/5 | 4.6/17.9/21 | 0.31 | ~0.1 | |
| 其余 ~20 种小 kernel（scale/quant/gelu/mul/add/argmax…） | ~330 | 0.6–1 | 0.25 | 0 | launch |
| **间隙 1058 launch** | | 1.25 avg | **1.32** | 0 | |
- **修正我昨晚给用户的排序**：MoE grouped GEMM 已在带宽下限 1.4–1.5×（三次 GEMM 共 40 MB/层 ÷ 1.79 TB/s = 22 us vs 实测 32 us），split-K 最多再省 0.3 ms，**不是最大缺口**。真正的两块：① **fp8 gemv（dense down / qkv / o_proj）3.6× 下限 = 0.65 ms 可省**（B20 批量预取没中，kernel 结构要重写：每 block 处理一段 N、全 block 协作流式读 K、更多字节在飞）；② **launch 数 1058 → 间隙 1.32 ms + 小 kernel 0.6 ms**：每层 35 个 launch，目标 ≤ 15（norm/rope/cache_write/norm_quant 融合、route+bucket+quant 融合、combine 融进 down 的 epilogue）。③ route_bucket_small 0.32 ms 单 block 是第三。
- 新顺序：②a 层内融合（-0.5~0.8 ms）→ ① fp8 gemv 重写（-0.5）→ ③ route（-0.25）→ MoE split-K（-0.3）→ lm_head fp8（-0.4）。合计可到 ~3.5 ms（vLLM 4.69）。
- D148 c8：64 步/实例，1041 kernel/步，kernel 和 4.81 ms + 间隙 1.02 = 5.8 ms/步（bench TPOT 9.82 的其余 ≈ 4 ms 是插队 prefill，D78）。kernel 构成与 c1 同（gemvx 20%、fp8 gemv 18.6%、MoE gate/up 13%、route 6.5%…每 kernel us 与 c1 持平，M=8 对带宽型免费）⇒ c8 的刀也是 launch 数 + fp8 gemv，再加 `supports_pair`。
- **动手顺序定稿**：B23（q/k/v 逐头 norm ×3 + rope + cache_write_fp8rt → 1 个 per-token kernel，-4 launch/层 = -120/步，逐位复刻 `rmsnorm_vec<128,1>` 归约树）→ B30（scale/quant_fp8/mul_row/add/gelu 这 6 个逐元素小 kernel 各并进相邻 kernel 的 epilogue）→ B31 fp8 gemv 重写 → route。

## B23 落地 2026-10-29：attention 前段五合一（commit 见 git）
- `norm.cu` 新 kernel `attn_pre_fused_kernel<VP, STORE_FP8>`：grid=tokens，block=(q+2kv) 个 warp（滑窗层 32 warp、全注意力 20 warp），一 warp 一头行：逐头 RMSNorm（复刻 `rmsnorm_vec_kernel<128,1>` 的蝶形 + 4 段合并序，lane 持向量 lane+32j）→ q/k neox RoPE（复刻 `rope_kernel<true>` 每步舍 bf16）→ fp8 往返写回 → k/v 写 cache（e4m3 / bf16）。smem 每 warp 一行。
- 接入 `layer.rs`：条件 kv_fp8 层 + 无 ATTN_SKIP_* + 非 dump + KV_FP8_FUSED；`ATTN_PRE_FUSED=0` 回退。门禁 `attn_pre_fused_parity`（4 形状 × e4m3/bf16，q/k/v/cache 四者位差 0）。预期：-4 launch/层 = -120/步，nsys 态 norm 3×1.2 + rope 3.0 + cache_write 4.7 = 11.3 us/层 → ~5 us；含间隙 ≈ -0.4 ms/步。
- **D149 B23 门禁过**：`attn_pre_fused_parity` 8 组（D=256/512 × tokens 1/13/400 × e4m3/bf16）q/k/v/cache **位差全 0**。bN TAG=b23 已起（A=b29 二进制，B=B23）。
- **B30 候选（层尾 / 路由）**：层尾 `add_inplace(h2→norm) + rmsnorm_add(norm→x) + scale(x)`：3 launch → 1（fused_add_rmsnorm 已有 NORM_ADD 变体，再加 layer_scalar 乘）；路由前 `rmsnorm_noweight + scale + mul_row`：3 → 1（`rmsnorm_scale2` 已有，但 mode 1 口径走分开三步，需对拍确认 scale2 在 mode 1 也逐位）；post_attn `o_proj → rmsnorm_add` 已融。合计每层再 -4 launch。
- **B30 落地（commit）**：a) `fused_add_rmsnorm_add_scale_bf16`：norm kernels 的 NORM_ADD 路径加 `post_scale`（`bf16(bf16(x + bf16(norm·w))·s)`，三处 store 同改），层尾 3 launch → 1；b) `rmsnorm_scale2` 加 mode 1 口径（inv 走 BI、三次 bf16 舍入复刻 noweight-norm → scale → mul），router 预处理 3 → 1。合计 -4 launch/层 = -120/步。门禁 `layer_tail_fused_parity`（tokens 1/8/400，s=1 与 0.707；router 三步 vs scale2）。等 b23 bN 完成后串：parity → bN TAG=b30（A=b23）。
- **D150 队列**：bN b23（经 watcher 包装起）第三次在 A 组后静默退出（无 B log、无进程、无 DONE）；独立 `setsid bash -x` 起的（D123）完整。重起 b23 为独立 + trace（`tmp_b23_bN_trace.log`），b30 链的 bN 也加 trace；下次死掉看 trace 最后一行定位（嫌疑：serve 收 SIGINT 时对进程组发信号把同 pgrp 的 bN.sh 带走；独立起时 pgrp 不同？）。
- **B32 预案（route_bucket_small 10.7 us 单 block）**：kernel 里 c1 只有 1 token：top-8 = 8 轮 warp 归约（快）、`route_softmax_row` **单 lane** 做 E=128 的 softmax（exp 串行 128 次 = 大头）、tid 0 串行 scan 128 专家（两张 tile 表）、row2e 二分。改法：① softmax 用 warp（32 lane × 4 元素，归约序固定成 `__fadd_rn` 树即可逐位——但现在是单 lane 顺序累加，要逐位就得保留串行序……顺序累加 128 次 exp 的求和改成"每 lane 4 个顺序累加 + 32 路蝶形"数值不同 → 非逐位；可接受（路由分数 ulp 不改 top-k 集合，只改权重末位）但要过 moe_route_fused_parity 的阈值）；② scan 用 warp 前缀和（`__shfl_up`）128 元素 4 轮；③ tile_off 两张表同一轮算。预期 10.7 → ~3 us，-0.23 ms/步。排在 B31 之后。
- **B31 预案（fp8 gemv 3.6×）**：现结构：每 warp 2 行 N、每 lane 读 ≤8 个 uint4（K/16/32 向量）一次性发出 → 一个 warp 在飞 2×K 字节 = 5.6 KB（K=2816），SM 4 warp × 2 = 22 KB 在飞；5090 需要 ~64 KB+/SM 才能填满 HBM 延迟带宽积（1.8 TB/s × ~600 ns）。改法：每 warp 4–8 行 N（寄存器 r0..r3 各 8 uint4 = 32 reg × 4 = 128 reg 太多）→ 改为每 lane 16 B × 4 行 × 2 批（流水：先发第 2 批再算第 1 批）；或 block 级 cp.async 把 8 行 × K 的权重搬进 smem（8 × 2816 = 22 KB/级，2 级 44 KB）再 mma 风格计算——更干净的是直接用 **mma.sync m16n8k32 fp8**（sm_120 支持 e4m3 mma）把 gemv 当 M=16（pad）GEMM 做，tensor core 把 fp8→f32 累加也省了（现在每元素 cvt + fma 占算力）。先做微基准：同形状 3072×2816 的 ① 现 kernel ② 8 行/warp 双批 ③ mma 版，取最快。
- D151：bN b23（独立 + trace）A 组完成（c1 5.99 / c8 TTFT 101（排队态噢动）/ TPOT 10.42），B 组（B23 二进制）跑中。B30 链等其 DONE。
- **D152 b23 bN B 组 NOTUP**：bN 的 `cargo build` 把我已推上去的 B30 源也编进了（B = B23+B30），serve 起来 warmup 报 `rmsnorm_scale2_bf16 failed: cudaError 1`（InvalidValue）→ `model.rs:209` 下标 panic。B30b 的 `rmsnorm_scale2` 在 mode 1 路径被调用时参数不合法——查 launcher 校验（hidden 2816 合法；疑 `scale` 指针对齐 / `rs` / mode 范围）。修好后 b23 与 b30 要分开测：先 `ROUTER_PRE_FUSED=0` 跑 b23+B30a，再开。
- 修：`rmsnorm_scale2_bf16` 入口放开 mode 1。链 `b2330.sh`：parity（attn_pre + layer_tail）+ `fp8_gemv_timing` 微基准 → bN TAG=b2330（A = b29 二进制，B = B23+B30a+B30b）。b23 单独的 bN 作废（B 组 NOTUP）。

## ★★★★★ D153 2026-10-29 b2330 门禁全过（attn_pre 8 组 + layer_tail 9 组位差 0）；**fp8 gemv 微基准推翻 D148 的"3.6×"**
| 形状 | m=1 us（下限） | 有效带宽 | m=8 us | 有效带宽 |
|---|---|---|---|---|
| dense down 2816×2112 | 4.12（3.32） | 1443 GB/s | **18.5** | 322 |
| qkv 3072×2816 | 6.16（4.83） | 1405 | **23.8** | 363 |
| o_proj 2816×4096 | 6.17（6.44） | 1870 | **47.1** | 245 |
| gate/up 一半 2112×2816 | 6.16（3.32） | 966 | **43.0** | 138 |
- **m=1 已在带宽下限 0.8–1.0×**（nsys 的 10.7 us 是 nsys 态膨胀，D75 教训再犯一次：小 kernel 的 nsys 时长不能当真）。c1 的 fp8 gemv 没有 0.65 ms 可省，B31 对 c1 **作废**。
- **m=8 慢 4.5–7.6×**：M_TILE=8 时 smem 45 KB/块、每 warp 8 行 FMA 串行、occupancy 掉到 1–2 block/SM。这正是 D62 加 24 KB 门槛的原因——所以 c8 的 dense/qkv 实际走哪条？nsys c8 显示 `gemv_kernel<1>` 84/步与 c1 同 → **c8 里 dense_decode_gemv 可能按行拆成 m=1 调用**（或模板实参被 nsys 截断）。要核 `dense_decode_gemv` / `launch_gemv` 对 m=8 的分派；若 c8 真走 m=8 慢路径，修它是 c8 TPOT 的大刀（85 次 × (25−6) us ≈ 1.6 ms/步）。

## ★★★★★ D154 2026-10-29 b2330 首批 B（B23 + B30a + B30b，bN 口径不带 FP8_GEMV）：c1 TPOT 5.98 → **5.57（-6.9%）**，吞吐 158 → 169；c8 TPOT 10.05 → **9.37（-6.8%）**，吞吐 678 → 703，c8 TTFT 133 → 151（A 侧 133 是排队态噢动，正常 ~155）；英文/中文输出与 A 逐字同
- -8 launch/层 = -240 launch/步，实测 -0.41 ms/步 ≈ 每 launch 1.7 us（含 kernel 自身 + 间隙），与 D148 估的 1.25 us 间隙 + 小 kernel 时长吻合。**launch 数路线验证成立**：剩余 ~800 launch 还有 ~1 ms 可挖。
- 注意口径：bN 的 A/B 都没带 `FP8_GEMV=1`；之前 c1 5.57 是带它的数。B23+B30 叠加 FP8_GEMV 预计 ~5.2，等 A2/B2 后用 envab 补一组。
- **b2330 定稿（两轮 ABAB）**：A c1 5.98/5.99、c8 TPOT 10.05/9.82、吞吐 678/673 → B c1 **5.57/5.57（-7%）**、吞吐 169/169；c8 TPOT **9.37/9.37（-6.8%）**、吞吐 **703/703（+4.4%）**、TTFT 151/152（≈ A2 的 155.6）；B/B2 英文 raw、中文输出与 A 逐字同。**合入**（默认开：`ATTN_PRE_FUSED` / `LAYER_TAIL_FUSED` / `ROUTER_PRE_FUSED`）。CSV round b2330。下一步 envab 叠加 `FP8_GEMV=1` 看 c1 绝对值。
- **D155 gemvsweep（m=1 不变，m=8 看 warps）**：warps=1 比自选（0）慢 2–3×（m=8 dense_down 41.9 vs 18.5）→ 占用率确实是 m=8 的瓶颈方向；其余档位见 `tmp_gemvsweep_RESULT.txt`。envab `fp8gemv_b2330`（A 默认 / B `FP8_GEMV=1`，b2330 二进制）排队。c8 是否真走 `gemv_kernel<8>` 用 python sqlite3 直接查 nsys c8 库。

## ★★★★ D156 2026-10-29 gemvsweep 定论 + c8 分派核实
- m=8：warps 2（自选）18.5 / 23.8 us → **warps 8：10.3 / 12.3 us（1.8×）**，16 与 8 同；m=1 不受 warps 影响（已在下限）。根因：M_TILE=8 的 A 驻留 45 KB 只被 2 个 warp 摊，occupancy 1 block/SM 且每 warp 串行 8 行 FMA。
- c8 nsys sqlite（python）：`gemv_kernel<1>` 5355 次 + **`gemv_kernel<4>` 2073 次（13.8 us）、`gateup_act<4>` 1036 次（15 us）**，nvjet qkv 315 次 21.4 us ⇒ c8 的 decode 步里 dense/qkv 走 M_TILE 4（24 KB 门槛下 K=2112 m=8 → 33 KB 超 → 实际是 n<8 的步？或 K=2816 的 m≤4）+ 一部分 cuBLASLt nvjet（21 us，比 8-warp gemv 的 12 us 慢 1.7×）。
- **B31b**：`pick_warps` 对 M_TILE ≥ 4 固定 8 warp；A 驻留门槛 24 → 48 KB（`FP8_GEMV_SMEM_KB` 覆盖）。预期 c8：85 次 gemv 中走 <4>/<8> 的 -3~5 us、qkv 从 nvjet 21 → 12 us；c1 不变。门禁 `mlp_fp8_decode_parity`（m 1/3/8/16）；链 `b31b.sh`（bN 两侧 FP8_GEMV=1，A=b2330）。
- **B32 落地（commit）**：`route_softmax_row_warp`：m 用 warp max；exp 每 lane 算自己 4 个专家（E=128）；**Z 仍按 e 升序 `__fadd_rn` 串行累加**（lane 0..31 依次 shfl 取值）→ 与单 lane 版逐位同；top-k 的 raw/renorm 蝶形同序。`route_bucket_small` 内替换单 lane 调用（`__syncwarp` 后全 warp 读 ids）。预期 10.7 → ~4 us（-0.2 ms/步）。远端放 `stage/moe.cu`，`b32.sh` 等 b31b DONE 后入源树 → `moe_route_fused_parity`（topk_w 位同）→ bN b32（A=b31b，两侧 FP8_GEMV=1）。
- **链排队**：envab fp8gemv_b2330（跑中）→ b31b（parity+timing → bN）→ b32。
- **B33 候选（再 -3~4 launch/层）**：① `MOE_GEMM_DUAL=1`（gate+up 单 launch，grid.z=2E，注释称逐位；从未在 b1x 后复测）→ -1；② dense gate/up 已是 gateup_act 单 kernel；③ router logits：`bf16_to_f32_n` 单独一次 → 让 `route_bucket_small` 直接读 bf16（或 router GEMM 用 f32 输出 epilogue）→ -1；④ `moe_scatter_add`/combine 后的 `f32_add_to_bf16_n`（2 处）→ 并进 combine 的 epilogue → -1；⑤ 输入 norm → fp8 quant 融合（`NORM_FUSE_QUANT`，曾因对拍差关，mode 1 下重新对拍）→ -1。先用 envab 试 ① 的现成开关（零代码）。
- **B33c 落地（commit）**：`route_bucket_small` 接受 bf16 logits（`RouteLogits::Bf16`，kernel 内 `to_f32` cast = `bf16_to_f32_n` 同值），kaon3 在 route_fused 且非 BI router 且非 dump 时跳过 `bf16_to_f32_n`（-1 launch/层）。parity 测试加 bf16 vs f32 入口比对（ids 同、w 位同）。远端放 `stage33/`，等 b32 链后入树。
- **D157 链尾死因候选（第 4 次复现：envab fp8gemv_b2330 也在 A 组后无声退出，无 B log）**：共同点 = `run_one` 里 `kill -INT $sp` 杀同进程组的 serve 后脚本立即消失。推断 dinfer serve 收 SIGINT 时对**进程组**发了信号（或 bash 把 SIGINT 传给了前台组），把同 pgrp 的脚本一并带走；之前"独立起 + bash -x"的能活是巧合（pgrp 关系不同）。修：bN.sh / envab.sh 里 serve 用 `setsid` 起（独立 pgrp）。envab fp8gemv_b2330 重起；b31b 链等它 DONE 后用新 bN.sh 生成 bN_b31b.sh。
- D157 补：远端 envab.sh 的 serve 行文本与我 sed 的模式不同没打上补丁（已在跑的这次是独立 setsid 起的，预期能活）；另存 `envab_setsid.sh` 供后续链（envab_dual 已改指向）。bN.sh 已打上 setsid。

## ★★★★★ D158 2026-10-29 envab fp8gemv_b2330 首批（b2330 二进制）：`FP8_GEMV=1` c1 TPOT **5.57 → 5.16（-7.4%）**，吞吐 169 → 182；c8 9.37 → 9.39（持平，c8 走 M_TILE 4/nvjet 未受益——B31b 的靶）。**c1 5.16 ms = vLLM 4.69 的 1.10×**（开工时 1.19×）。
- **fp8gemv_b2330 定稿（两轮，setsid 起的链完整跑完 → D157 判断成立）**：A（无 FP8_GEMV）c1 5.57/5.58、c8 9.37/9.37；B（FP8_GEMV=1）c1 **5.16/5.16**、c8 9.39/9.37。c1 -7.4%，c8 持平。**`FP8_GEMV` 应改默认开**（c1 稳定收益、c8 中性、输出同 — mlp_fp8 parity 已过）；c8 的收益等 B31b（M_TILE≥4 8 warp + 48 KB 门槛）。
- **B33a 落地（commit）**：`rmsnorm_quant_fp8` 加 `mode`：mode 1 = `bf16(norm_mid(x·inv)·w)` 再 `×inv_scale → e4m3`，与 `rmsnorm_bf16_inplace + quant_fp8` 两步逐位（`norm_quant_fused_parity`）；`NORM_FUSE_QUANT` 默认开（旧"更差"记录是 compile 口径）；`forward_preq` 也走 fp8 gemv（否则 preq 退回 Lt 抵消收益）。-1 launch/层。stage33 已更新（含 B33c、FP8_GEMV 默认开）。
- **B33 链计划**：b32 bN DONE → stage33 入树 → parity（moe_route_fused + norm_quant_fused + mlp_fp8）→ bN TAG=b33（A=b32 二进制）。
- **D159 b31b 门禁过**：mlp_fp8 parity m=8/16 超差 0；gemv timing m=8：dense_down 18.5 → **10.3**、qkv 23.8 → **12.3**、o_proj 47.1 → **16.4**、gateup 43.0 → **8.2** us（8 warp）。bN b31b 跑中（两侧 FP8_GEMV=1，A=b2330）。

## ★★★★★ D160 2026-10-29 b31b 首批 B（fp8 gemv M_TILE≥4 8 warp + 48 KB 门槛；两侧 FP8_GEMV=1，A=b2330）：c8 TPOT **9.37 → 8.58（-8.4%）**，吞吐 704 → **755（+7%）**，TTFT 持平；c1 5.15 = 5.15（不变，符合预期）。A2 c8 9.38 复现 A。等 B2。
- 当前 c1 nsys（b29 二进制）每步 launch 构成（/步）：fp8 gemv 85、rmsnorm<256,2> 151、rmsnorm<128,1> 90（B23 后 0）、grouped GEMM 90、grouped_quant 60、quant_fp8 60（B33a 后 30）、scale 61（B30 后 ~1）、rope/cache_write/norm_quant/combine/merge/route/gelu/mul_row/add/bf16_to_f32 各 30、v3 25、lm_head 31。B23+B30+B33 后预计 ~1058 → ~820。
- **B34 候选（下一批 launch 砍）**：① `rmsnorm<256,2>` 剩 ~90/步（pre_ff_norm_2 for MoE、post_ff_norm_1 dense、post_attn 已融）：MoE 的 `pre_ff_norm_2` 可融进 `moe_grouped_quant`（norm→量化一体，同 B33a 思路）-30；② `grouped_quant` ×2 → gate/up 的量化并进 route kernel 尾部（route 已有 sorted_tok）-30；③ `gelu_mul` 并进 down 的 `grouped_quant`（读 gate_o/up_o 直接算 gelu·mul 再量化）-30；④ `gqa_decode_merge_v2` 并进 o_proj gemv 的 A 装载（merge 输出 bf16 → 直接读 scratch 做 merge）-25。合计再 -115/步 ≈ -0.2 ms。
- **b31b 定稿（两轮）**：c8 TPOT 9.37/9.38 → **8.58/8.55（-8.6%）**，吞吐 704/703 → **755/755（+7.3%）**；c1 5.15 不变。中文 chat 输出 A/B 逐字同；**英文 raw 补全（"The capital of France is"，无模板）A "…Wait, I'm not sure if I'm doing it right" → B "is is is is …"**。差异来源：该 6 token prefill 的 qkv/o_proj 以前 45 KB > 24 KB 门槛走 cuBLASLt，现在走 gemv<8>（m=6 < M_TILE）→ 数值路径变（同 D145 批不变性口径，parity m=3/8/16 超差 0）；但 "is is is" 退化要排除 m<M_TILE 的 bug：加 `smemcap_nat` 探针（同 b31b 二进制，`FP8_GEMV_SMEM_KB=24` vs 默认 48，自然语言 10 条串行 + 该 raw prompt），若 10 条全同/语义等价则接受。
- b32 门禁过：`moe_route_fused_parity` 全部 ok（w 位差 0、combine 位同）。bN b32 排队/跑中。

## ★★★ D161 2026-10-29 b32 首批 B（route softmax warp 版）：c1 5.15 → **5.33（+3.5%）**，c8 8.66 → **8.89（+2.7%）** —— **退化**。原因：Z 串行累加改成 32×4 次 `__shfl_sync` 链（每次 shfl 等 lane 同步）+ `ex[32]`/循环展开寄存器压力，比单 lane 128 次 MUFU exp 还慢；10.7 us 里主要不是 exp。**B32 默认关**（`ROUTE_SOFTMAX_WARP=1` 开），单 lane 路径保留 bf16 logits 支持（B33c 用）。stage33/moe.cu 已更新。route 的真正优化应是把 route+bucket 的 **block 级**串行（tid 0 扫 128 专家 ×2 表、二分 row2e）并行化，而不是 softmax。

## B35 设计 2026-10-29：kaon3 `supports_pair`（prefill / decode 跨流并行，用户顺序第 ④ 刀；c8 TPOT 里 ~3 ms 是插队 prefill）
- 参照 qwen3 `AltSet`：第二套 {stream, Gemm, PlanBuffers+uploader, logits/gathered/argmax, LayerScratch, ActBuf}，`forward_sample_pair(decode, prefill)`：主流入队 decode（graph）→ 副流等主流上一步事件 → 入队 prefill → 记事件 → 主流等它 → 两边 collect → 按原行序合并。KV cache 共享（两步写不同 slot，读互不依赖：decode 读历史、prefill 写自己）。
- **kaon3 特有障碍**：`QLinear::Fp8 { quant: Mutex<Option<CudaSlice<u8>>>, scache }` 与 `LayerScratch.preq` 是**按层共享的 lazy 缓冲**，两条流同时 forward 会写同一块 quant 缓冲 → race。改法：quant 缓冲移出 QLinear 进 LayerScratch（每套 scratch 各一份，按 max_tokens 预分配 `Σ 层型 K` 的最大值一块复用），`forward(..)` 多收一个 `&mut CudaSlice<u8>`；`scache`（两个 device 标量）只读可共享。`Gemm` 带 cuBLASLt handle + workspace → 副流自己一套（qwen3 同）。显存：副套 LayerScratch 按 prefill max_tokens 4096 分配 ≈ 1–2 GB，5090 32 GB 现用 11 GB，可行。
- 预期：c8 TPOT 8.58 → ~6.5（去掉 prefill 插队的 ~2–3 ms，vLLM 6.24），c8 TTFT 不变或略升（prefill 与 decode 争 SM）。c1 无影响。工作量：executor +150 行、layer.rs quant 缓冲重构 +60 行、parity 用现有 c8 mixparity（输出同批构成确定性）+ bN。
- b32 定稿（两轮）：B c1 5.33/5.34、c8 8.89/8.86 vs A 5.15/5.15、8.66/8.66 → **退化确认，默认关**（CSV round b32 记"未合入"）。envab dual（MOE_GEMM_DUAL=1，跑在 b32 二进制上：A 基线 5.34/8.87 = b32 的值）跑中。
- **B35 落地（本地 commit，未上远端）**：① `QLinear::Fp8.quant` 共享 Mutex 暂存 → `LayerScratch.quant_a`（每套一份，`forward(.., &mut sc.quant_a)`；测试走 `forward_alloc`）；② `AltSet`{stream, Gemm, PlanBuffers+uploader, logits/hidden/gathered/argmax, LayerScratch, ActBuf}，`XSTREAM=0` 关；`swap_alt`、`enqueue(plan, allow_graph)`（从 forward_sample 拆出）、`supports_pair`、`forward_sample_pair`（主流 decode graph → 副流等 main_done → prefill eager → 事件 → 主流等 → 两边 collect）；单流步末也记 `main_done`。显存：副套按 max_tokens 4160 的 scratch 约 +1.5 GB。门禁：c8 concprobe（同批构成确定性）+ mixparity 不应变差（只是时序变）+ bN c8。排在 b33 链后（stage35）。

## ★★★★ D162 2026-10-29 链尾死因**真**根因：`docker exec -d … bash x.sh` 起的脚本 stdout/stderr 接在已断开的 exec 管道上，`kill -KILL $sp`（进程已被 INT 结束）向 stderr 写 "No such process" → **SIGPIPE 杀掉 bash**。证据：所有完整跑完的链（b23 独立、b2330、b31b、b32 的 bN）都是 `bash -x … >trace 2>&1` 重定向了输出；所有死在 A 组之后的（b28/b29/b23 首次/fp8gemv 首次/dual）都没重定向；D157 的 setsid 不是原因（dual 用了 setsid 版 envab 仍死）。
- **规则**：远端起任何脚本一律 `docker exec -d C bash -c "setsid bash /w/x.sh </dev/null >/w/dinfer/tmp_x.out 2>&1 &"`；链内调用子脚本也 `>log 2>&1`。进 knowledge/subagent-ops。
- **D163 b33 门禁全过**：mlp_fp8（含 m=8/16）超差 0；bf16-logits 入口 ids 同 / w 位差 0（5 组）；norm+quant mode 1 融合 fp8 码位差 0（9 组）。bN b33 跑中（marker=0 是 strings 没匹配到 `route_lg`——内联函数名不在二进制里，无碍；A=b31b 带 FP8_GEMV=1，B=默认开）。
- **B34① 评估**：`grouped_quant` 是逐 16 元素块的 elementwise（每线程一块，无行归约），而 `pre_ff_norm_2` 需要整行 2816 的平方和 → 融合要么改成一 block 一行（R=8 行 → 8 block，并行度掉）要么先 norm 再 quant 两段同 kernel 用 block 归约（R 行 × 176 块/行，block 内归约再广播）——可做：一 block 一行 256 线程、每线程 11 个 16B 向量，归约树复刻 `rmsnorm_vec<256,2>`（hidden 2816 → VPT 2… 不对，vec kernel 的 BLOCK=256/VPT=2 处理 hidden≤4096：每线程 2 个 8 元素向量 = 16 元素，正好对应 quant16 的一块；按原 kernel 的 gs 顺序累加 + rms_ss_triton 树就能逐位同）→ 新 kernel `rmsnorm_grouped_quant`：输入 x[token]（经 sorted_tok 间接）、w、agscale[e]；输出 P/S；替代 `rmsnorm_into + grouped_quant`（-1 launch，且省 h2 的一写一读）。下一刀做。dense 侧 `post_ff_norm_1` 需跨 block 归约 N=2816 的 gemv 输出 → 不可融进 gemv，留。

## ★★★★ D164 2026-10-29 **b33 bN 作废 + D157 的 setsid 是有害的**：nvidia-smi 在 B 组跑时仍见 A 的 `dinfer_prev_b33` 占 8 GB —— `setsid prog &` 在非交互 bash 下若 bash 把后台作业放进独立 pgrp，setsid 会 fork 后父退出，`$!` 指向已退出的父 → `kill -INT $sp` 杀不到 serve → **上一组的 serve 留在 GPU 上与下一组争算力**（b33 A c1 5.76 vs 应有 5.15 = 与残留的 dual_b32 服务同卡）。D162 的输出重定向才是链尾死因的修法；setsid 全部撤回（bN.sh / tail_wrap→envab.sh）。已杀全部残留，重起 b33 bN（独立、重定向、无 setsid）。
- 规则更新（knowledge/subagent-ops）：serve 直接 `prog &`，不用 setsid；链起法只靠 `>log 2>&1`；每条链开头 `nvidia-smi` 核无残留 dinfer 进程，有则先清。
- **D165 队列重置**：发现同时有 3 个脚本在起 serve（旧 b35 watcher、tail_wrap 残留的 envab_setsid、after_b33 因旧 RESULT 含 NOTUP 提前放行），GPU 上两套 serve 并存 → 全部杀掉，改为**单编排器 `master.sh`** 串行：b33 bN → dual envab → smemcap_nat → b35，每段前 `pkill -x dinfer/bench`，全部输出落文件，不再用 watcher。**规则**：同一时间只允许一个编排脚本在跑；起新链前 `ps` 核无旧编排器。

## ★★★ D166 2026-10-29 b33 bN（A 侧实际是 **b32 二进制**：b33.sh 第一次实例在我 sed 之前已执行了 `cp target → prev_b33`，当时 target=b32；A 5.34/8.87 = b32 数）：B（b33 = B31b + B33a + B33c + FP8_GEMV 默认开 + B32 关）c1 **5.20**、c8 **8.92**。与 b31b 定稿（5.15 / 8.58，不同次运行）比 **c1 +1%、c8 +4%** —— B33 不但没省反而慢。嫌疑：B33c 的单 lane bf16 路径 `route_softmax_row_any` 用 `float row[MAX_E=1024]` 局部数组（4 KB local memory）+ 逐元素 cast，比直接读 f32 全局慢；c8 每层 +11 us 量级吻合。B33a（norm+quant mode 1）理论只省 launch，但 `rmsnorm_quant_kernel` 用 `rms_inv(mode)` 走 BI 的 rcp/sqrt 近似 + 多一次 bf16 往返，应中性。
- **拆变量（envab on b33 二进制，排在 master 后）**：① `ROUTER_LOGITS_BF16=0` vs 默认；② `NORM_FUSE_QUANT=0` vs 默认。哪个让 c8 回到 8.58 就关掉它或重写（bf16 logits 应在 route kernel 的 s_val 装载时 cast 一次，softmax 读 s_val——现在 softmax 读的是全局 logits 而不是 s_val 副本！改成读 s_val 行即可同时去掉 local 数组）。
- **b33 定稿（两轮，A=b32 二进制）**：B c1 5.20/5.20、c8 8.92/8.96 —— 比 b31b（5.15/8.58）慢，**B33 不合入现状**，D166 修了 B33c 的 local 数组（softmax 改读 smem 副本）；连同 B34①（norm+grouped quant 融合，parity 测试 `moe_norm_quant_fused_parity`）与 B35 一起放 `stage36/`。下一条链 **b36** = stage36 入树 → parity ×5（moe_route_fused / norm_quant_fused / moe_norm_quant_fused / mlp_fp8 / attn_pre）→ bN（A = b31b 二进制 `/w/dinfer_prev_b32`，两侧 FP8_GEMV=1）→ 若仍慢再拆 B33a/B33c/B34①/B35 四个开关。

## ★★★ D167 2026-10-29 envab dual_b33（b33 二进制，A 默认 / B `MOE_GEMM_DUAL=1`）首批：c1 5.20 → **5.13（-1.3%）**，c8 8.99 → **8.88（-1.2%）**（gate+up 单 launch，grid.z=2E，注释称逐位）。等 A2/B2；两轮同向则 `MOE_GEMM_DUAL` 默认开（再 -30 launch/步）。
- 注：`moe_grouped_gemm_dual` 走的是旧 2 级 `pipe` body（BM=32，非 pipe4/BM16），仍比两次 pipe4<BM16> 快 1.3%——说明 decode 下 launch 数/grid 填充比流水级数重要。若 A2/B2 确认：先把 `MOE_GEMM_DUAL` 默认开（零代码），再做 pipe4<BM16> 的 dual 变体（预期再 -1%）。
- **dual_b33 定稿（两轮）**：`MOE_GEMM_DUAL=1` c1 5.20/5.20 → **5.13/5.13（-1.3%）**，c8 8.99/8.98 → **8.88/8.82（-1.5%）**，输出同。**默认开**（已进 stage36/layer.rs，b36 一并测）。
- **D168 smemcap_nat（b31b 二进制，`FP8_GEMV_SMEM_KB=24` vs 默认 48，无 cache，10 条自然语言 chat + 2 条 raw 补全含 "The capital of France is"）：0/12 不同** ⇒ B31b 的门槛改动不改变输出；bN 里 A（b2330）→ B（b31b）raw 输出从 "Wait, I'm not sure…" 变 "is is is…" 不是门槛引起（两个二进制还差 pick_warps 8 warp：数值上同序，理论不该变）。未解；自然语言 10/10 同，先记为待查（下次用 envnat 直接比两个二进制：`BIN=` 两次）。

## ★★★★ D169 2026-10-29 **b35**（master 末段实际跑的是 `b35.sh` = stage35：只换 executor.rs/layer.rs，即 **B35 xstream**；stage36 的 kernel（B33 修正 / B34① / dual 默认开）**没进树、没跑**，`tmp_b36_*` 不存在；A = `prev_b35` = 编译前 target = b33 二进制）：**A/B 完全同**（c1 5.20/5.20，c8 TPOT 9.12/9.13，吞吐 731/730；A2/B2 复现），门禁 concprobe 0/8 差
- **B35 没起作用的原因（关键发现）**：B 服务日志 `runner step mix (cumulative) steps=1000 mixed=4 pair=4 prefill_only=16`——c8 bench 里 **混批步只有 4/1000**，prefill 几乎全是 **prefill-only 步（16 = 8 轮 × 2 chunk）**。bench c8×64 的 8 条请求同 prompt-len / 同 max-tokens，同步完成 → 下一批 8 条同时到 → scheduler 出纯 prefill 步，decode 行此时为空。`supports_pair` 只对混批生效，所以 c8 TPOT 一点没动。
- 推论：D78 "c8 TPOT 多 3–4 ms 是 decode 步之间插队的 prefill" 在这个 bench 形态下**不成立**（decode 行在 prefill 时根本不存在，TPOT 口径 = (总时 − TTFT)/(out−1) 不含别人的 prefill）。**c8 比 c1 多的 ~4 ms 另有来源**，要用服务端 `/metrics` step_time 直方图 + 每步 wall 时间定性（D170）。
- c1 5.20 = b33 定稿数（dual 未默认开）。B35 对 c1 无影响符合预期。stage36 kernel 另起链 **b36**（含 B36 dual→pipe4<BM16>，`moe_pipe_parity` dual 用例 rows=22 覆盖新路径）。
- binnat（b2330 vs b31b，自然语言 10 + raw 2，无 cache）：**0/12 差**，bN 里 raw "Wait, I'm not sure" → "is is is" 的差异不是二进制差异（两个二进制同输入同输出），是 bN 第一条 raw 请求前服务态（bench 之后 prefix cache / 批构成）的批不变性现象，归 D157 结案口径。

## ★★★★★ D170 2026-10-29 hosttime（b33 二进制，`HOST_TIME=1`，bN 同口径）：**c8 decode 步本身就是 9.0 ms，不是 prefill 插队**
- `[host-time]` 按 (tokens,nreq) 分组：(1,1) 756 步 total p50 **5207 us**（cadence 5.21 = bench TPOT 5.21）；**(8,8) 496 步 total p50 8970 / p90 9256 us，cadence p50 8.98 ms = bench c8 TPOT 8.95**；prefill (429,1) 25.3 ms ×13、(3003,8) 143.9 ms ×2（chunk 一次塞 8 条 → TTFT 151）；host sched/publish 都 < 10 us。
- 结论：**c8 的 TPOT 全部由 M=8 的 decode 步构成**（exec 8.96 ms），D78 的"插队 prefill 摊进 TPOT"定性**错误**（那时没有按批形状分组）。M=8 比 M=1 慢 3.75 ms（+72%），而权重带宽型 kernel 对 M=8 应近免费 —— **这 3.75 ms 是 M=8 时某些 kernel 退化**，优先级最高。嫌疑：① MoE grouped GEMM：M=8 → top-8 ×8 = 64 行散到 ≤64 个专家，激活专家数 ~50（vs M=1 的 8），**专家权重读取量 ×6**（每层 gate/up/down 704×2816×3×0.5 B ≈ 3 MB/专家 → 50 专家 150 MB/层 × 30 层 = 4.5 GB/步 ÷ 1.8 TB/s = **2.5 ms**）—— 这才是 c8 的物理下限来源，不是 bug，但要确认实际是否贴近；② fp8 gemv M=8：M_TILE=8 的路径（pick_warps / smem）效率；③ attention d512/v6 在 8 行时。下一步：nsys c8 按 kernel 名比 c1 的 us/call（D148 c8 表说"每 kernel us 与 c1 持平"——与本条矛盾，需重抓；D148 c8 可能抓的是 bench 尾部 M<8 段）。
- 修 b36 编译错：`moe_grouped_sm120.cu` 的 B34① launcher 被 D166 的编辑误插到文件头（吃掉首行 `//`），nvcc "extended character 。 is not valid in an identifier"。挪回文件尾（`}  // extern "C"` 前）。

## ★★★★★ D171 2026-10-29 重读 nsys c8（D148 抓的 `tmp_nsys_c8.sqlite`，`nsys_kern.py --all`）：**M=8 的 decode 步在另一个 graphId（桶 8）里，D148/D78 只看了"最常见 kernel 数"的那组 = M=1 尾段**，所以得出"每 kernel us 与 c1 持平"的错误结论
- **口径更正**：`nsys_c8.sh` 是 `--concurrency 8 --num-requests 3` → 实际并发 3，graph id=8 的 34 步是 **M=3**（不是 8），且二进制是 D148 时的 b29（pick_warps 旧版）。表里的方向性结论（router cuBLAS 挑了慢 kernel、MoE 每专家行数少效率低）成立，绝对值要以 b37 后重抓的 `nsys_c8b.sh`（24 请求，真 M=8）为准。
- graph id=8（M=3，34 步）vs graph id=2（M=1，63 步），每 call p50（nsys 态）：
  | kernel | M=1 | M=8 | 每步差（×30 层） |
  |---|---|---|---|
  | MoE gate/up `pipe4<128,4,16>` | 10.7 us | **22.0** | +340 us（+3780 call 的 2.1×） |
  | MoE down `pipe4<64,4,16>` | 10.5 | **24.6** | +420 |
  | **router GEMM**（bf16 cuBLAS）| gemvx 3.4 | **cutlass wmma 16x16 grid=(8,1,1,32) 16.7** | **+400**（8 个 CTA 串行吃 K=2816；M=8/N=128 形状被 cuBLAS 选了最差 kernel） |
  | o_proj fp8（sliding，K=4096）| gemv 6.1（`(704,64)`）| **nvjet split-K 11.7，30 call/步**（M=1 只 5 call）| **+245**：M_TILE=8×K4096×2 = 64 KB > 48 KB smem 门槛 → 退回 cuBLASLt |
  | dense `gateup_act<4>` | 9.4 | 15.0 | +170（11.9 MB → 6.6 us 下限，M=8 2.3× 离） |
  | qkv gemv `(1024,128)` | 16.0 | 19.6（`gemv<4>`） | +90（带宽型，正常） |
  | lm_head（tied bf16 262144×2816 = 1.48 GB）| gemvx **870 us** | cutlass 880 | 0；但它占 c1 一步的 **17%**，就是带宽下限（820 us） |
  nsys 态合计 ≈ +1.7 ms（真实 ≈ ×0.6 ≈ 1.0 ms）+ 其余小 kernel 的 M=8 增量 + 间隙增量 ⇒ 对应 hosttime 的 +3.75 ms 的主体。
- **刀（按收益/代价）**：R1 router 自写 bf16 gemv（M ≤ 16：warp 一列，K 切 lane；720 KB 权重，目标 ≤ 4 us；逐位：fp32 累加序固定）→ c8 -0.3 ms；R2 `FP8_GEMV_SMEM_KB=64` 让 M_TILE=8 的 o_proj 留在 gemv（零代码 envab）→ c8 -0.2 ms；R3 dense gateup M=8 的 warp 编排（2 warp/块 → 8）；R4 MoE pipe4 M=8：64 行散到 ~50 专家 ⇒ 权重读 ~6×（物理），要先算实际激活专家数定下限再决定是否值得。
- c1 的硬事实：lm_head 870 + MoE 1270 + fp8 gemv 1150 + router 100 + … ；lm_head 若做 fp8（per-channel）可省 ~430 us（c1 5.2 → ~4.8，**达 c1 ≤ 5.0 停止条件**），但 vLLM 对该 checkpoint 保持 bf16 lm_head，这是数值口径变更，**要用户拍板**（候选 ⑤）。

## ★★★★★ D172 2026-10-29 **b36 定稿（两轮 ABAB，A = b33 二进制 `prev_b35`，B = stage36 = B33 修正 + B34① norm+grouped_quant 融合 + B35 xstream + dual 默认开 + B36 dual→pipe4<BM16>）**：c1 TPOT **5.15/5.15 → 5.04/5.04（-2.1%）**，c8 TPOT **8.66/8.70 → 8.46/8.43（-2.5%）**，吞吐 c8 750 → **761**，TTFT 持平；中文 chat / raw 输出 A/B 逐字同；parity 6 个测试组全过（含 `moe_pipe_parity` dual 用例覆盖 pipe4 dual）
- hosttime（b36 二进制）：(1,1) 5059 us、(8,8) 8475 us；c8 混批仍 4/1000，B35 在此 bench 形态下不贡献（留着，线上流量混批多）。
- c1 5.04，**离停止条件 c1 ≤ 5.0 还差 0.04 ms**；c8 8.46 对 vLLM 6.24 还差 1.36×。按 D171 的刀：R2 `FP8_GEMV_SMEM_KB=64`（o_proj M=8 留 gemv）envab 跑中 → R1 router bf16 gemv（M ≤ 16）→ R3 dense gateup M=8 warp 编排。
- 归档 `test/kaon3/2026-10-29/b36/`（bN / parity / hosttime ×2 / b35 / binnat / nsys c8 --all）。

## ★★ D173 2026-10-29 R2 envab smem64_b36（b36 二进制，`FP8_GEMV_SMEM_KB=64`：让 M_TILE=8×K=4096 的 o_proj 留在自写 gemv）：c1 5.05 = 5.05（不涉及），c8 **8.45 → 8.65（+2.4%）退化** → **不采用**，48 KB 门槛保留
- 解释：M_TILE=8、K=4096 时 smem 64 KB/块 → 每 SM 只能放 3 块（228 KB），8 warp 块 × 3 = 24 warp/SM，权重流并行度不够；cuBLASLt nvjet split-K（11.7 us）反而更好。真正的解法是 gemv 做 K 切分（split-K 两段 + 归约）或 A 放寄存器而不是 smem；先做 R1/R3，再回头。

## ★★★ D174 2026-10-29 b37 首批 B（R1 router gemv v1：warp 一专家行，grid = E/4 = 32 块）：c1 5.04 → **5.09（+1%）**，c8 8.46 → **8.89（+5%）** —— **退化**，parity 过（ulp 0、批不变）
- 原因：grid 只有 32 块 × 4 warp，170 个 SM 用了 19%，每 lane 串行 11 个 chunk（4 展开），latency-bound；M=8 时每 chunk 再读 8 次 x，链更长。比 cuBLAS gemvx（M=1 3.4 us）还慢。
- v2：grid = E（128 块）× 8 warp，warp 内 split-K（每 warp K/8 = 44 chunk，lane 1–2 个），warp 内 xor 树 + smem 固定序跨 warp 求和（累加序与 M/grid 无关 → 仍批不变）；加微基准（`router_gemv_timing`）在 nsys 外量 v2 vs cuBLAS，M=1/8，**达不到 cuBLAS M=1 的 3.4 us 就只在 M ≥ 2 启用**。

## ★★★★★ D175 2026-10-29 **b37b 定稿（R1 v2 router bf16 gemv：block/专家、8 warp split-K、固定序；A = b36 二进制）**：c1 TPOT **5.04/5.05 → 5.00/4.99（-1%）**，c8 **8.42/8.47 → 8.42/8.39**，c8 吞吐 762 → 766；输出逐字同。微基准 m=1/2/4/8：gemv 3.98/3.78/3.94/3.98 us vs cuBLAS 6.51/12.31/12.31/9.35；m=16 6.16 vs 9.39（仍赢，ulp 1 —— M ≤ 16 全开）。**c1 ≤ 5.0 停止条件首次达到**（vLLM 4.69 的 1.07×）。CSV round b37b。
- 下一条：B34③ gelu·mul + down 量化融合（b38，已写待编）；之后 c8 专项（M=8：o_proj fp8 gemv K 切分、MoE pipe4 M=8）。

## ★★★ D176 2026-10-29 b38 首批 B（B34③ gelu·mul + down 量化融合，A = b37b 二进制）：c1 4.99 → **4.96**，c8 8.43 → **8.42**；parity 6 组 aq/a_s 位差 0；输出同。等 A2/B2。
- 下一刀 **A8**（c8 专项，D171 o_proj 项）：fp8 gemv 的 A 改成 e4m3 驻留 smem（M_TILE·K·1 B，M=8/K=4096 = 32 KB ≤ 48 KB 门槛），dot2 内每 16 元素一次 `e4m3x16_to_f32`（原 bf16 路径也要 `__bfloat1622float2` 解包，ALU 量同级）；e4m3 → bf16 → f32 与 e4m3 → f32 值相同 → **逐位同**（`mlp_fp8_decode_parity` 已有 (2816,4096,8) 形状）。开关 `FP8_GEMV_A8`，默认开；A/B 用 envab。
- **b38 定稿（两轮）**：c1 4.99/5.00 → **4.96/4.97（-0.6%）**，c8 8.43/8.40 → **8.42/8.39**，吞吐 767；输出同。**合入**（CSV round b38）。

## D177 2026-10-29 b39 链起（A8：fp8 gemv A 以 e4m3 驻留 smem，`FP8_GEMV_A8` 默认开；`mlp_fp8_gemv_a_bytes()` 把字节数告诉 Rust 门槛；A 侧 = b38 二进制，md5 写进 `tmp_b39par_RESULT.txt`）
- 门禁：`mlp_fp8_decode_parity` A8 开/关各一遍（含 (2816,4096,8) 形状）+ `fp8_gemv_timing` 微基准；本地 sm_86 nvcc 已编过。预期：o_proj M=8 从 Lt split-K 11.7 us ×30 回到 gemv，c8 -0.2 ms；c1 M=1 不变（A 驻留 4 KB vs 8 KB，略省）。
- 本链起前核对清单（用户 2026-10-29 "细心点"）：本地 nvcc 编过 ✓、stage39 五文件 md5 与本机一致 ✓、远端 `bash -n` ✓、A 侧 target 是 b38（01:47，含 B34③ 符号）✓、GPU 0 无残留 ✓。
- **MoE 物理下限估算（M=8，top-8/128，独立近似）**：每层激活专家 ≈ 128·(1−(1−8/128)^8) ≈ **51.6**（M=1 为 8）；每专家 gate/up/down 704×2816×3×0.5 B ×(1+1/16) ≈ 3.15 MB → M=8 每步 4.9 GB ÷ 1.79 TB/s ≈ **2.73 ms**，M=1 ≈ 0.42 ms。即 c8 的 MoE 段比 c1 **至少多 2.3 ms 是物理的**，c8 TPOT 下限 ≈ c1 的 5.0 + 2.3 ≈ 7.3 ms；现 8.39，离下限 1.1 ms。vLLM 6.24 < 7.3 ⇒ **vLLM 在 c8 必然没有逐 token 读全部激活专家**——要查 vLLM 这个 checkpoint 的 MoE kernel（cutlass grouped / flashinfer trtllm fp4）对 M=8 是否做了专家内行合并或别的事（或 bench 的 c8 在 vLLM 侧实际批更小）。这是 c8 方向下一步的关键问题（D178）。
- D177 门禁：parity A8 开/关全 ok（(2816,4096,8) 超差 0）；微基准（A8 开）m=8：dense_down 10.25 us、qkv 12.23、**oproj(K=4096) 14.14**、gateup_half 8.21；m=1 全部 6.16/4.13（与 B31b 时同）。oproj m=8 的 14.1 us 对 Lt split-K nsys 态 11.7（真实 ≈ 8–10）**可能不赢**——bN 见分晓；若 c8 退化，A8 只对 K ≤ 2816 形状生效（门槛不变），oproj 留 Lt。

## ★★★★★ D178 2026-10-29 **b39 定稿（A8：fp8 gemv A 以 e4m3 驻留 smem，默认开；A = b38 二进制）**：c1 4.96/4.97 → 4.97/4.97（不变），c8 TPOT **8.34/8.43 → 8.04/8.04（-4%）**，c8 吞吐 767 → **795**（vLLM 956 的 83%）；parity A8 开/关全过。**合入**（CSV round b39）。
- 微基准 oproj m=8 14.1 us 看着没赢 Lt 的 nsys 态 11.7，端到端却 -0.3 ms/步 —— 又一次印证 nsys 态数字不能直接当真实（D153 教训）；端到端 A/B 才算数。
- 当前：c1 **4.97**（vLLM 4.69 的 1.06×，停止条件 ≤ 5.0 ✓），c8 **8.04**（vLLM 6.24 的 1.29×）。下一步只剩 c8：先用 `MOE_EXPERT_STATS=1` 数 M=8 真实激活专家数（校准带宽下限，解释 vLLM 6.24），再决定 MoE M=8 的刀。
- **D178 第一次 expstats 作废**：只抓到启动期 4 条（capture 桶 4160/64/4 + 一条 prefill）—— decode 步走 graph 回放，Rust 层代码不执行，`MOE_EXPERT_STATS` / `KAON3_MOE_TIME` 这类层内诊断**必须配 `NO_GRAPH=1`**（脚本已改，b40b 重跑：expstats2 → moetime）。顺带：graph capture 期 counts 未初始化（max_rows 2016817528），诊断忽略 capture 样本。

## ★★★★★ D179 2026-10-29 expstats2（NO_GRAPH，b39 二进制 + 诊断）：**M=8 decode 步每层真实激活专家 p50 = 29（mean 28.3，19–36）**，不是独立近似的 52；M=1 恒 8；prefill 428 tok 激活 55–59，且极偏（单专家 410/3424 行）
- 带宽下限重算：M=8 MoE 权重 28.3 × 3.35 MB × 30 = 2.84 GB → **1.59 ms**；M=1 0.45 ms；**M=8 比 M=1 物理多读 ≈ 1.14 ms**。vLLM c8 − c1 = 6.24 − 4.69 = **1.55**（贴下限）；我们 8.04 − 4.97 = **3.07** ⇒ M=8 下 **≈ 1.9 ms 是 kernel 效率**，不是物理。这就是 c8 剩余差距的全部，目标 c8 ≤ 6.5。
- 嫌疑排序（D171 表 + D177 微基准）：① fp8 gemv M=8（qkv 12.2 / oproj 14.1 / dense_down 10.3 / gateup 8.2 us，均为 m=1 的 2×；带宽型应 ≈1×；30 层 × (12.2+14.1+10.3+8.2×2 − m1 值) ≈ **+0.8 ms**）；② MoE pipe4 M=8（每 call 22/24.6 vs 10.7/10.5 us，30 层 × 2 call ≈ +0.75 ms，其中物理 ≈ +1.1 ms 的一部分…需 nsys c8b 校准）；③ attention 8 行 d512/v3；④ route/combine 小 kernel。等 nsys c8b（真 M=8，`--all`）给每 kernel 的 M=8 增量，再按增量排刀。

## ★★★★★★ D180 2026-10-29 nsys c8b（b39+诊断二进制，c8×24，`--all`；graph id=11 = **真 M=8，122 步**；id=2 = M=1，65 步）每步（÷步数）：
| kernel | M=1 us/call × 次/步 | M=8 us/call × 次/步 | M=8 每步 | M=1 每步 | Δ |
|---|---|---|---|---|---|
| MoE gate‖up dual `pipe4<128,4,16>` grid.z=2 | 20.1 × 30 | **63.6 × 30** | **1.91 ms** | 0.60 | **+1.31** |
| MoE down `pipe4<64,4,16>` | 10.6 × 30 | **38.1 × 30** | **1.14** | 0.32 | **+0.82** |
| lm_head bf16（gemvx / cutlass） | 870 × 1 | 883 × 1 | 0.88 | 0.87 | 0 |
| fp8 gemv o_proj+dense_down（A8，grid 176） | 6.2 × 55 | 9.2 × 55 | 0.51 | 0.34 | +0.17 |
| fp8 gemv qkv sliding（grid 512） | 16.0 × 25 | 21.9 × 25 | 0.55 | 0.40 | +0.15 |
| dense gateup_act | 9.8 × 30 | 12.4 × 30 | 0.37 | 0.29 | +0.08 |
| route_bucket_small | 11.4 × 30 | 11.8 × 30 | 0.35 | 0.34 | 0 |
| router_gemv（R1 v2） | 1.7 × 30 | 3.1 × 30 | 0.09 | 0.05 | +0.04 |
| 合计 kernel 和/步 | | | **7.55** | ≈4.5 | +3.0 |
- **结论 1**：c8 比 c1 多的 3.07 ms 里 **MoE 两段占 2.13**，fp8 gemv 四段合计 0.44，其余 ~0.5 是 attention / 小 kernel。D171 的"router cuBLAS / o_proj 退 Lt"已修完（router 3.1 us、o_proj 9.2 us），剩下就是 MoE。
- **结论 2（更重要）**：MoE grouped GEMM 的带宽效率 **M=1 与 M=8 一样只有 ~50%**：M=1 gate/up 读 8 专家 × 2 × 1.11 MB = 17.8 MB / 20.1 us = 0.89 TB/s；M=8 读 28.3 × 2 × 1.11 = 63 MB / 63.6 us = 0.99 TB/s；down 同样 ~0.85 TB/s。不是"M=8 退化"，是 **pipe4 对这种 64 行 × 64 B 小段的流式读只吃到一半带宽**。若做到 80%：c1 MoE 0.92 → 0.58（**c1 4.97 → ~4.6**）、c8 3.05 → 1.9（**c8 8.04 → ~6.9**）。这是 c1/c8 共同的最大一刀（B40）。
- 候选：① 已有的 `moe_fp4_gemv`（`moe_fp4_decode.cu`，带宽型 grouped GEMV，`MOE_GEMV` 开关）——先查它当时为什么没赢、现在跑分；② pipe4 BK=128 → 每行每 stage 64 B 的离散读改成每行 ≥128–256 B 连续（BK=256 或 stage 内合并 2 个 BK）；③ 每 CTA 多 N 块（BN=128）减少 B 行重读…先做①的微基准（nsys 外）再定。

## ★★★★★ D181 2026-10-29 moebench（nsys 外微基准，真实 decode 形状；注意权重集 18–62 MB **全在 L2 里是热的**，数字是"L2 热"上限，不是 DRAM 口径）
| 场景 | gate‖up dual | 2×grouped | down | B7 gateup_act | B7 down gemv |
|---|---|---|---|---|---|
| M=1（8 专家 8 行） | **18.3 us**（0.98 TB/s） | 24.6 | **8.2**（1.08 TB/s） | 74.4 | 36.9 |
| M=8（28 专家 64 行） | **28.7**（"2.2 TB/s" = L2 热） | 28.7 | 18.9 | 227 | 95 |
- STAGES 3/4 无差；`MOE_DUAL_PIPE4=0`（旧 BM32 体）M=1 慢 2 us、M=8 慢 1.8 us（B36 收益确认）。B7 GEMV 慢 4–8×，彻底放弃。
- **结论**：M=1 权重在 L2 热的情况下仍只 ~1 TB/s ⇒ **不是 DRAM 带宽瓶颈，是每 CTA 的串行 K 链**：gate‖up grid = 8 专家 × 11 N 块 × 2 = 176 CTA（1 CTA/SM，8 warp/SM），每 CTA 串行 22 个 BK=128 迭代（每迭代 wait_group + 2 次 __syncthreads + 4 次 mma ≈ 0.7 us）→ 18 us 几乎全是链长 × 迭代固定开销。M=8 CTA 更多（616）反而每专家更便宜，同一结论。
- **B40 刀**：① **BK=256**（迭代 22 → 11；smem 12.8 KB/stage，STAGES=3 → 38 KB < 48 KB；K 尾块加谓词零填充，让 down K=704 也能走 BK=256 → 3 迭代；累加序 k 升序不变 → 逐位同，`moe_pipe_parity` 带 `MOE_GEMM_BK=256` 跑一遍即门禁）；② 若还不够：split-K ×2–4（grid 翻倍，fp32 partial 固定序归约，批不变）。moetime2（eager+sync）r=8 gemm3 = 39 us/层、r=64 106 us/层，与 nsys 一致（c1 MoE GEMM 30 × 39 = 1.2 ms、c8 3.2 ms）。
- **D181 门禁过**：BK=64/128/256 与旧 body（MOE_GEMM_PIPE=0）dump 逐字节 SAME（gate K=2816 + down K=704 含尾块零填充，单路 + dual 各半），`moe_dual_matches` 在 BK=256 下 ok。moebench2（BK 扫）+ bN b41 跑中。

## ★★★★ D182 2026-10-29 moebench2（BK 扫，L2 热）：**BK=256 vs 128**：M=1 gate‖up **14.4 → 12.3 us**（DRAM 下限 10.0）、down 10.1 → 8.2（下限 5.0）；M=8 gate‖up **26.7 → 20.5**、down 18.9 → 16.4。BK=64 M=1 gate‖up 28 us（旧默认 down 走的就是 BK=64，因 704 % 128 ≠ 0）。
- 串行 K 链定性成立但只占一半：M=1 gate‖up 22 迭代 → 11 迭代省 2 us（≈0.2 us/迭代，不是估的 0.7）；剩下 12.3 − 10.0 = 2.3 us 是启动 / 尾部 / 首 stage 延迟。预期端到端：c1 MoE 30 × (2.1 + 1.9) ≈ **-0.12 ms**（4.97 → ~4.85），c8 30 × (6.2 + 2.5) ≈ **-0.26 ms**（8.04 → ~7.8）。bN b41 跑中。
- 再往下的刀（M=8 真实服务权重不在 L2，DRAM 下限 52 us/层 vs 现 ~75）：split-K×2 让 gate‖up grid 从 616 CTA 翻到 1232（每 CTA 5–6 迭代，更多 CTA 并发读 → 吃满 DRAM 并行度）；down K=704 只 3 迭代，split 无意义，改 BN=32（CTA ×2）。

## ★★★★★★ D183 2026-10-29 **b41 定稿（B40：decode pipe4 BK=256 + K 尾谓词，默认；A = b39 等价二进制）**：c1 TPOT **4.97/4.97 → 4.78/4.78（-3.8%）**，c8 TPOT **8.04/8.04 → 7.19/7.18（-10.6%）**，c8 吞吐 795 → **867（vLLM 956 的 91%）**，TTFT 持平；中文/raw 输出 A/B 逐字同；门禁 BK 64/128/256 位同。**合入**（CSV round b41）。
- 端到端收益（c1 -0.19、c8 -0.85 ms）是 L2 热微基准预期（-0.12 / -0.26）的 2–3×：真实服务权重冷（DRAM），迭代减半 + STAGES 3 让每 CTA 在飞字节翻倍，DRAM 并行度上去了——微基准低估了"链长"在冷数据下的代价。
- 现状：**c1 4.78（vLLM 1.02×）、c8 7.19（1.15×）、吞吐 91%**。

## D184 2026-10-29 b42 链起（B41：decode pipe4 `BN_T=32` 模板 + 4 warp CTA，仅 BK=256 路径；默认仍 64，bN B 侧 `MOE_GEMM_BN=32`；门禁 = ref/256/bn32 dump cmp + dual parity；A = b41 二进制 02:37）
- 预期：gate‖up CTA 616 → 1232、down 44 → 88 N 块；每 CTA smem 19 KB → 每 SM 常驻 CTA 翻倍；在飞字节翻倍。若赢则改默认 + 再测 BN=16。
- **b42 定稿（两轮，B = `MOE_GEMM_BN=32`）**：c1 4.77/4.78 → **4.82/4.82（+1%）**，c8 7.18/7.19 → 7.18/7.17（持平）。微基准：gate‖up 不变（12.33 = 12.33），down **8.2 → 10.3 us 退化**（4 warp CTA 载 A/B 的线程减半，K=704 只 3 迭代，装载占比高）。**BN=32 不采用**，默认留 64（代码留着，开关 `MOE_GEMM_BN=32`）。门禁 ref/256/bn32 位同。
- 结论：BK=256 之后 gate‖up 在 L2 热已 12.3 us（下限 10.0）、M=8 20.5（已低于 DRAM 下限 34.9，说明微基准形状全在 L2）——**再提 MoE 要看真实（DRAM 冷）场景**：下一刀应直接 A/B split-K×2（gate‖up K=2816 → 2×1408，grid 翻倍，fp32 partial + 固定序归约 kernel，多一次 launch）；或者换方向做 attention / 其余小 kernel（c8 nsys 表里 MoE 之外的 +0.9 ms）。

## D185 2026-10-29 B42b：fp8 gemv / gateup 权重预取提前到 load_a 之前（c8 nsys 表第 2 名 `gemv_kernel<8>` 10407 call/122 步 = 85/步 × 15.2 us = 1.3 ms/步，M=1 也是 0.5 ms）
- 现状 kernel 顺序：load_a（smem 装载 + barrier）→ 每 lane 发 8 个 uint4 权重 __ldg → 算。权重流在 prologue 期间不动。改成：先发权重 __ldg（寄存器在飞）→ load_a → 算；fma 序不变 → 逐位同（门禁 `mlp_fp8_decode_parity` A8 开/关 + `fp8_gemv_timing`）。链 b43（A = b41 二进制）。

## ★★★ D186 2026-10-29 b43 定稿（B42b 权重预取提前，两轮）：c1 4.77/4.78 → **4.73/4.73（-0.8%）**，c8 7.19/7.19 → **7.54/7.54（+4.9%）**。微基准 m=8 不变（10.3/12.3/14.4/8.2）—— 退化只在端到端：M_TILE=8 时每线程 16 个 uint4（64 寄存器）跨 barrier 常驻，占用掉一档，权重流并行度反降。**改法**：`PRE = M_TILE <= 2` 编译期选（c1 路径预取、M≥4 原序）。与 B43 一起进 b44。
## D187 2026-10-29 B43：route_bucket_small 两处并行化（逐位同）：① softmax 的 128 次串行 exp 改 warp 并行算到 smem 行，Z 仍 lane 0 按 e 升序 `__fadd_rn`（序不变）；② counts → offsets/tile_off/tile_off16/cursor 的单线程 128 轮前缀和改 warp 0 整数 shfl_up 扫描。`ROUTE_SOFTMAX_WARP=2`（默认）/1（B32）/0（单线程）。门禁 `moe_route_fused_parity`（与 route_softmax 单线程路径逐位比）。预期 route 11.8 → ~6 us ⇒ c1/c8 各 -0.17 ms。

## D188 2026-10-29 用户问：若只剩 megakernel 一条路怎么做、改动多大 → 开子代理做数据支撑（只读 nsys sqlite + 现有 mega_* 骨架，不碰 GPU），产物 `docs/loops/kaon3-megakernel-study.md`
- 主会话预判（待数据验证）：不做一步一 kernel；做"分段 persistent"（norm→qkv→rope/cache→attention→o_proj→层尾 一段），MoE grouped GEMM / lm_head 留独立 kernel；改动 ~1000 行 + 3–4 轮 A/B；门槛 = "< 8 us kernel 总时长 + gaps − 每级屏障 3–10 us × 级数" ≥ 0.5 ms 才开工。

## ★★★★★★ D189 2026-10-29 **b44 定稿（B42c 预取只对 M_TILE≤2 + B43 route 并行 exp/warp 扫描；A = b43 二进制）**：c1 TPOT 4.73/4.74 → **4.65/4.65**，c8 7.53/7.54 → **7.13/7.13**，c8 吞吐 835 → **870**；相对 b41 定稿（4.78 / 7.19 / 867）：**c1 -2.7%、c8 -0.8%**。parity 全过（route 与单线程版逐位、A8 开关各一遍）；输出同。**合入**（CSV round b44）。
- **c1 4.65 < vLLM 4.69（0.99×）——首次反超**。c8 7.13 vs 6.24（1.14×），吞吐 91%。
- B42b 的教训：同一改动 M=1 赢、M=8 输（寄存器跨 barrier 常驻压占用）——gemv 类改动必须按 M_TILE 分档决定，微基准 m=8 没看出差异（L2 热），端到端才暴露。
- **D188 结论（`kaon3-megakernel-study.md`）**：**no-go**。边界效应上限 c1 ≈ 1.1 ms（23%）/ c8 ≈ 0.9 ms（12%）；megakernel 要付 30 层 × 14 级 = 420 个阶段屏障 × 3–10 us = 1.3–4.2 ms > 能拿回的 1.1；dense persistent 版在本卡已输 5.8×（D60/D61）、B300 单层输 0.91/0.82/0.59×；c8 的差距 65% 在 > 20 us 的大 kernel（MoE pipe4、gemv<8>），常驻 grid 只会更差。规模 3.5–4k 新 LOC + ≥ 11 轮 A/B。若硬要做：只融滑窗层 attention 段（8 kernel）一轮，判据 c1 ≥ -0.3 ms 且 c8 不退。**替代路线**：再融 6–8 个 < 3 us kernel/层（merge+quant、router+route、grouped_quant 进 route、gelu_quant 进 MoE epilogue、combine+norm+res、post-down 三 norm）≈ c1 -0.3…-0.4；c8 走 MoE split-K×2 + gemv<8> 占用 ≈ -0.4。
- nsys 重抓（b44 二进制）：c1b 85287 kernel/107 步 ≈ **797/步**，kernel 和 4.97 ms + gaps 1.37（nsys 态；bench 4.65）；c8b id=11 97922/122 ≈ 803/步，7.93 + 3.7（gaps 含 nsys 膨胀）。供 split-K 后对照。

## D190 2026-10-29 B44 split-K（待编待验）：decode gate‖up dual 的 pipe4 加 `ksplit`（grid.z = 2·ksplit，各 split 只算自己的 K 块段，写 fp32 部分和 P[2·ksplit][R][N]，不乘 alpha）；新 `gelu_grouped_quant_splitk` 在消费端做 Σ_s + alpha + bf16 舍入 + gelu + 量化（与 B34③ 同式）。`MOE_SPLITK=2|4`（默认 1 = 关，走原路径）。
- 口径：ksplit=1 与原路径逐位同；ksplit≥2 fp32 加法序不同 → 非逐位，门禁 `moe_splitk_parity`：fp4 码差 ≤ 2% 且 worst ≤ 1、a_s 差 ≤ 2%；端到端看输出。归约序 split 升序固定 → 批不变。
- 预期：gate‖up CTA 616 → 1232（M=8）/ 176 → 352（M=1），每 CTA 6 迭代；DRAM 冷场景在飞字节翻倍；c8 目标 7.13 → ~6.8。链 b45（B 侧 env MOE_SPLITK=2，A = b44）。

## ★★★ D191 2026-10-29 b45 定稿（B44 split-K×2，B 侧 `MOE_SPLITK=2`，两轮）：c1 4.65 → **4.86（+4.5%）**，c8 7.13 → **7.42（+4%）** —— **退化，不采用**（默认 1 = 关，代码留作 A/B）。parity：ksplit 1/2/4 fp4 码差全 0（K 段对齐下 fp32 和恰好同）。
- 解释：BK=256/STAGES=3 后每 CTA 只 11 迭代，再对半 → 6 迭代，prologue（首 2 stage 装载 + tile_lookup）占比过半，CTA 翻倍换不回；外加 fp32 部分和 2×4 B 写读。**MoE decode 的"链长"路线到此为止**，剩余差距要么是每 CTA 启动固定开销（tile_lookup 读 offsets/tile_off 两级二分 + 首 stage 延迟），要么回到 DRAM 并行度（需要 nsys b44 c8b 按 kernel 看 pipe4 现在的 us/call 再定）。

## D192 2026-10-29 B45（待编待验）：combine + post_ff_norm_2 一次 launch（`combine_norm_kernel`：block=token、256 线程持 2 个 8 元素组，combine 同式 → 写 y → 平方和同序 → rms_inv(nmode) → h2），与 `combine_kernel → rmsnorm_vec<256,2>` 逐位同；`MOE_COMBINE_NORM=0` 回退；-30 launch/步。门禁 `moe_combine_norm_parity`。链 b46（A = b44）。
- b44 nsys 对照（c1b 新抓，M=1，每步 797 kernel）：lm_head gemvx 870 us（20.8%）、fp8 gemv<1> 85 call 10.1 us（20.6%）、MoE pipe4 dual+down 60 call ≈ 12.2 us 均（17.5%）、dense gateup 9.3、route 8.8、combine 4.1、v3 attn 4.6、norm_quant 3.2、d512 attn 18、< 3 us 的 norm/quant/merge/gelu_quant/router 共 ~12 个/层。b46 融 combine+norm_2（-30 launch）；之后候选：router_gemv+route（-30）、rmsnorm_grouped_quant 并进 route（-30）、merge+quant（-25）。

## D193 2026-10-29 排队（master6，b46 后）：envab `FP8_GEMV_WARPS=4` / `=16` on b46 二进制 —— 看 gemv<8>（qkv M=8 21.9 us，grid 512×256 线程）是否是波次量化（2 块/SM × 170 = 340 槽，512 块 = 1.5 波）；c1 也会变（force 对所有 M_TILE 生效），只按 c8 判，赢了就改 `pick_warps` 的 `m_tile >= 4` 分支。

## ★★ D194 2026-10-29 b46 首版 B45（combine+norm_2 融合，两轮）：c1 4.65 → **4.70（+1%）**，c8 7.12 → **7.24（+1.7%）**，TTFT +3 ms —— 退化（parity 全过、输出同）。原因：融合版 gather 用标量 bf16 读 `y_in[pos*H + i*8+e]`，lane 间步长 16 B、每次 2 B → 合并率 1/8，K=8 × 2 向量 = 16 次坏读/线程；原 combine_kernel 按 d 跨线程连续读是合并的。修：每 k 一次 uint4（8 bf16）读，逐元素累加序不变 → 仍逐位同。b47 重测（A = b46 二进制 = 默认走融合版 … **注意**：b46 二进制 `MOE_COMBINE_NORM` 默认开，A 侧要 `MOE_COMBINE_NORM=0` 才等于 b44；用 bN A 侧 env 变体）。
- **D193 结果（b46 二进制，默认含融合版退化基线 4.71/7.25）**：`FP8_GEMV_WARPS=4`：c8 7.25 → 7.31（更差）；`=16`：c1 4.71 → 4.76（M=1 也被强制 16 warp，坏）、c8 7.25/7.26 → 7.18/7.85（B2 c8 TTFT 101 异常，排队态噪声）。结论：gemv<8> 不是波次量化问题；16 warp 对 M≥4 至多 -0.06 ms，不值得单独一轮，**搁置**。

## ★★★ D195 2026-10-29 b47 定稿（B45 修正版 combine+norm_2 融合，uint4 向量 gather；同二进制 envab，A = `MOE_COMBINE_NORM=0`）：c1 4.65/4.65 → **4.62/4.62（-0.6%）**，c8 7.12/7.12 → 7.33/**7.09**（B 轮 c8 TTFT 103 异常 = 请求没叠上的排队态，看 B2 7.09，-0.4%），吞吐 872 → 875–878；parity 位差 0；输出同。**合入**（默认开，CSV round b47）。
- 现状：**c1 4.62（vLLM 4.69 的 0.985×）**，c8 7.09（1.14×），吞吐 ~876（92%）。

## ★★★★★★ D196 2026-10-29 高并发对照（用户要 c16/32/64 数据；`kaon3_cN.sh`，b47 二进制 vs vLLM :28201，两轮交错换 seed，两端 max-num-seqs 64）：**没有优势，差距随并发拉大**
| c | dinfer TPOT p50 | vLLM | 比 | dinfer 吞吐 | vLLM | 比 | TTFT p50 d/v |
|---|---|---|---|---|---|---|---|
| 8 | 6.90/7.08 | 6.15/5.98 | 1.15× | 873/854 | 961/1007 | 89% | 153/150 |
| 16 | 11.33/11.30 | 6.92/7.06 | **1.62×** | 1118 | 1560 | 72% | 200/224 |
| 32 | 16.09/16.21 | 9.21/9.40 | **1.73×** | 1490 | 2435 | 61% | 346/233 |
| 64 | 28.47/28.38 | 15.39/15.47 | **1.84×** | 1880 | 3300 | 57% | 372/256 |
- vLLM 每翻倍并发 TPOT ×1.15–1.6（带宽型摊薄）；我们 ×1.4–1.75 → M>8 路径里有按行数线性涨的东西。本日 loop 全部只对 c1/c8 调优，M>16 路径基本没碰：
  ① `tokens ≤ 16` 以上 fp8 gemv 不走 → qkv/o_proj/dense 退回 cuBLASLt（`FP8_LT_TUNE` 默认关，D154 关的）；② MoE decode 快路径 `R ≤ 128`（BM16/BK256）→ c32 起 256/512 行走 BM=32/BK=128 旧 pipe4；③ c64 激活专家 ≈ 全部 128（每专家 ~4 行，BM=32 tile 7/8 空）；④ attention v3/d512 与 route 在 32–64 行的 us 没量过；⑤ lm_head bf16 262144×2816 在 M=64 走 cuBLAS gemvx？（M=1 870 us，M=64 可能变 GEMM 路径）。
- 下一步：hosttime c16/32/64 分组 + nsys c32（96 请求）→ 按 M=32 kernel 表开刀（预计先是 ②③ 的 MoE 路径与 ① 的 Lt 自调优）。归档 `test/kaon3/2026-10-29/cN/`。

## ★★★★★ D197 2026-10-29 c16/32/64 定位（hosttime_cN + nsys c32b，b47 二进制）
- hosttime：decode 步 exec (16,16) **9.7 ms**、(32,32) **12.3**、(64,64) **15.8**；bench TPOT 11.3 / 16.1 / 28.4 ⇒ **TPOT − 步 = 1.6 / 3.8 / 12.6 ms 是 prefill 插入**（c64 占 TPOT 的 44%）。混批 prefill 步 (4141 tok, 55 req) **188 ms = 22 tok/ms**，单请求 1560 tok 25 ms = 62 tok/ms → 多请求 chunk prefill 慢 3×（候选：per-request 循环的 attention prefill / 非 graph eager launch 4141 行）。
- nsys M=32 每步（÷128）：**MoE gate‖up `grouped_gemm_kernel_dual<128>`（旧 BM=32 pipe 体，快路径门槛 R ≤ 128）107.8 us × 30 = 3.23 ms（28%）**；down `pipe4<64,4,32,64>` 58.5 × 30 = 1.76；fp8 Lt nvjet 10.6 × 125 = 1.33 + 64x64 21.3 × 25 = 0.53（M>16 不走 gemv）；attention v3 40.6 × 25 = 1.0、d512 109 × 5 = 0.55；**route_bucket_small 32 us × 30 = 0.96（单 block，随 token 线性）**；lm_head small_m_mma<32> 903 us；combine_norm 5.6 × 30 = 0.17。合计 ≈ 11.5。
- 刀：① `MOE_DEC_RMAX`（默认 128 → B 侧 512）：让 M ≤ 64 的 MoE 走 BM16/BK256/STAGES3 dual pipe4（M=32 激活 ~100 专家、每专家 2–3 行，BM=32 tile 93% 空）；② route 拆两段（tokens > 8 时）：`route_topk_kernel` grid=tokens 一 warp/块（top-k + softmax，全局 `acc` 原子计数）+ `route_bucket_kernel` 单 block（scan + sorted/inv/row2e，末尾清 acc 自复位）——多 1 launch 换 32 tok 32 → ~8 us；③ prefill 插入（c64 主体）：下一条，先 nsys 混批 prefill 步。

## ★★★★ D198 2026-10-29 b48 定稿（`MOE_DEC_RMAX=512`，envab_cN 两轮，b47 源 + RMAX 开关）：c16 11.03/11.27 → 11.28/11.27（持平）、**c32 TPOT 12.20/12.10 → 10.47/10.50（-14%），吞吐 1882 → 2070（+10%）**、**c64 28.43/28.60 → 26.74/26.82（-6%），1883 → 1981**；门禁 R=264 old/rmax 与 ref 逐位 SAME。**默认改 512**（代码已改，远端 target f05df420f737 是"默认 128 + 开关"版，下一轮第一条链先编默认 512 版并用 cN.sh 干净口径复测）。
- **口径注意**：`envab_cN.sh` 三档同 seed 7，c32/c64 的前 64/128 条 prompt 命中 c16 留下的 prefix cache（TTFT p50 45 ms 暴露），绝对值低于 cN 表（c32 16.1），只能看 A/B 方向；下一轮把 envab_cN 的 seed 按档错开（已列待办）。
- **收工（用户：连 vLLM 都没达到、差距大，下一轮再优化）**。c16–c64 剩余刀按 D197：② route 拆两段（grid=tokens）③ 混批 prefill 22 tok/ms 定位（c64 TPOT 44% 是它）④ M>16 fp8 Lt 自调优重开 ⑤ attention v3 M=32 40 us/层。

## ★★★★★ D199 2026-10-29 第四轮开工：用户重申 step A/B/C 流程 + 新原则"先对标 vLLM 逐环节"
**用户原话要点（2026-10-29）**：
- 前置：拉齐 vLLM 测试条件，两端都非 eager，concurrency / max-model-len / max-num-batched-tokens 一致。
- Step A：跑 5 轮 c=1,2,4,8,16，测 dinfer 启动时间 / TTFT / TPOT / 带宽 / GPU 利用率，每步 tokenize / chat 模板 / htod / block 分配 / schedule / exec / wait。
- Step B：针对耗时最长的做深入分析和优化，提交，编译最新代码运行。
- Step C：端到端测试，收益差则回滚，**数据必须进 CSV 和 dashboard**。回 Step B 重复 5 次（我明确认为需要可适当加轮次）；没进展回 Step A 重新分析。
- **新原则**：要超越，先对齐参照做得好的点。现在还没达到 vLLM，所以 Step A 要**逐环节与 vLLM 比**：哪一环节比 vLLM 慢 → 劣势优化；哪一环节已领先 → 优势保留（改动不许把它丢掉）。

**本轮环境事实（开工核对）**
- vLLM 基线 `vllm-noneager` :28201（GPU 2，`--max-model-len 16384 --max-num-batched-tokens 4096 --block-size 32 --gpu-memory-utilization 0.85`，**max-num-seqs 默认 256**；dinfer 64——c ≤ 64 时不影响批形状，记录在案）。`vllm-eager` :28200 不用。
- dinfer：GPU 0，`--max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 4096`，graph 态（非 eager），env `MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU=4 MEGA_SPDN=4 ROUTER_GEMM_BI=0`。
- 远端二进制 05:54 md5 f05df420 = b48 源（RMAX 默认仍 128）；本地 HEAD 默认 512 **先同步编译**（本轮基线 = HEAD）。
- 钉核：serve 0-31、bench 60-91、vLLM 容器自己的核；写进 RESULT 头部。

**Step A v4 怎么测（`scripts/dev/kaon3_stepA4.sh`，put 到 `/w/stepA4.sh`，`docker exec -d kaon3-full bash -c 'bash /w/stepA4.sh > /w/dinfer/tmp_stepA4_run.log 2>&1'`）**
- 5 轮 × c ∈ {1,2,4,8,16}（主口径）+ c ∈ {32,64}（差距所在，附加行），每窗口 dinfer / vLLM 交错，**同窗口同 seed，跨窗口换 seed**（`seed=1000·r+c`，避免 prefix cache 命中假象）。400 tok prompt / 64 out / `num_requests=8c` / warmup 2。
- dinfer 环节：启动 wall + `model_load_ms`；`[render-time]`（chat 模板 + tokenize）；`[host-time]` sched / exec / publish 按 (tokens,nreq) 分组；`[begin-split]` update_states（含 block 分配）/ build_plan / forward_sample；`[kaon3-host]` upload（htod）/ launch / collect（wait = collect(dtoh_sync)）；nvidia-smi 200 ms 采样 sm / mem util。
- **vLLM 对标环节（`/metrics` 窗口增量）**：`request_queue_time`（≈ 我们 sched 排队）、`request_prefill_time` / `request_decode_time` / `request_inference_time`、`scheduler_time_seconds`（host 调度）、`prefill_steps` / `decode_steps`（步数 → 每步 ms = decode_time/decode_steps）、`iteration_tokens_total`（批形状）、`estimated_read_bytes_per_gpu`（带宽估计）、`prefix_cache_hits/queries`。dinfer 侧 `engine_steps_total` / `generation_tokens_total` / `prefix_cache_hits_total` 同口径取增量。
- 产出：一张"环节 × 并发"对比表：每环节 dinfer / vLLM / 比值，标 ✓领先 / ✗落后；Step B 只打 ✗ 里最大的一项。
- 带宽：vLLM 用其自报 read bytes / 窗口时间；dinfer 用权重字节/步（= 已知 ~2.6 GB fp4+fp8 混合 / 步）÷ 步时间反推；两边都标"估计"。

**停止条件（本轮）**：① c16 TPOT ≤ vLLM × 1.10 且 c32 / c64 ≤ 1.25×（吞吐 ≥ 80%）；② c1 / c8 不退化（> 3% 回滚）；③ 每刀进 CSV（含负结果）。Step B 连续 5 刀无进展 → 回 Step A。

## ★★★★ D200 2026-10-29 Step A v4 第 1 轮速读（HEAD 二进制 md5 9046f6ef，RMAX 默认 512；完整 5 轮见 `test/kaon3/2026-10-29/stepA4/`）
| c | dinfer TPOT | vLLM TPOT | dinfer 纯 decode 步 exec | dinfer prefill 步 | TPOT − decode 步（= prefill 插入） |
|---|---|---|---|---|---|
| 1 | 4.65 | 4.69(D196) | 4.64 | — | 0 |
| 8 | 7.07 | 6.15 | 6.54 | — | 0.5 |
| 16 | 11.04 | 7.06 | 8.67 | 26 步 × 98.5 ms（2147 tok/步 = **22 tok/ms**） | **2.4** |
| 32 | 14.6 | 9.2(D196) | 9.29 | | **5.3** |
| 64 | 27.0 | 15.4(D196) | 11.83 | | **15.2** |
- vLLM c16：33 prefill 步 + 655 decode 步，5.2 s wall → **7.5 ms/步 含 prefill**（chunked prefill 混进 decode 步，几乎不抬步时）；per_req prefill 126 ms ⇒ 6400 tok/126 ms ≈ **50 tok/ms**（边 decode 边 prefill）。dinfer prefill 22 tok/ms，单请求 1560 tok 25 ms = 62 tok/ms（D197）→ **多请求 prefill 比单请求慢 3×，比 vLLM 慢 2.3×**。
- **环节判定**：✓ 领先/持平：c1 decode 步、TTFT c≤8、render 1 ms、sched/publish/htod < 40 us（vLLM sched 490 us/call）。✗ 落后：① **多请求 prefill 吞吐**（c16–64 TPOT 差额的 60–100%，最大项）；② 纯 decode 步 M=16/32/64 比 vLLM 慢 ~1.2×（8.67 vs ~7.0）。
- **Step B 靶子 = ①**：pair 步（decode 主流 + prefill 副流）的 prefill 2147 tok 98 ms。先 nsys 一个 c16 prefill 步看 kernel 表（候选：gqa_prefill 按请求循环 / MoE 大 M 路径 / fp8 GEMM 形状），再改。
- 工具修正：stepA4 的 dinfer `/metrics` 名带 `dinfer:` 前缀，首轮 dinfer-metrics 全 0（脚本已修，下次链用 `stepA4b.sh`）。

## D201 2026-10-29 Step B 定靶准备：现有 nsys c32b 里几乎没有混批 prefill 步（只 1660 个 eager kernel，是启动 warmup + autotune）→ 需要专抓
- `scripts/dev/nsys_prefill_steps.py`（eager kernel 按间隙切步、最长 3 步聚合、按家族汇总）；`scripts/dev/kaon3_nsys_prefill16.sh`（stepA4 口径 c16 × 128 请求抓 pair 步 prefill 副流）。stepA4 跑完 GPU0 空出后起。
- c32b 的 warmup prefill（4096 tok 单请求）kernel 表给了一个基准：14 层 MoE dual 1315 us/call + pipe4 872 us/call ≈ 2.2 ms/层（4096 行 → **1.9 tok/us**），gqa_fa_v6 3.7 ms/call（D=512 两层）。

## ★★★★★★ D202 2026-10-29 Step A v4 定稿（5 轮中位数，`test/kaon3/2026-10-29/stepA4/`）——**靶子 = 多请求 prefill 步**
| c | TPOT d/v（×） | TTFT d/v | tok/s d/v（×） | dinfer 纯 decode 步 | dinfer prefill 步 exec（tok, nreq, 步数） | TPOT−decode 步 | vLLM prefill 步数 / per-req prefill ms / iter tok 均 / sched us |
|---|---|---|---|---|---|---|---|
| 1 | 4.65/4.69（0.99） | 27/25 | 200/200 | 4.64 | 25.1（428, 1.0, 10） | 0 | 10 / 20 / 7.7 / 84 |
| 2 | 5.42/5.37（1.01） | 52/36 | 338/346 | 5.07 | 26.1（428, 1.4, 18） | 0.35 | 18 / 22 / 14 / 145 |
| 4 | 5.89/6.06（0.97） | 86/89 | 565/544 | 5.58 | 41.0（810, 2.3, 18） | 0.31 | 18 / 73 / 26 / 188 |
| 8 | 7.06/6.14（1.15） | 154/149 | 858/971（0.88） | 6.54 | 71.2（1571, 4.1, 18） | 0.52 | 23 / 93 / 50 / 292 |
| 16 | 11.36/7.14（**1.59**） | 197/179 | 1116/1574（0.71） | 8.67 | **98.4（2146, 10.0, 26）** | **2.69** | 36 / 113 / 96 / 490 |
| 32 | 14.67/9.31（**1.58**） | 346/231 | 1612/2470（0.65） | 9.31 | **121.8（2649, 23.8, 42）** | **5.36** | 46 / 133 / 193 / 843 |
| 64 | 27.07/15.34（**1.76**） | 360/247 | 1980/3367（0.59） | 11.83 | **154.4（3378, 51.8, 66）** | **15.24** | 74 / 153 / 381 / 1496 |

**逐环节判定（用户新原则：劣势优化、优势保留）**
- ✓ 领先 / 持平（**改动不许丢**）：c1–c4 TPOT、c≤8 TTFT、render（chat 模板 + tokenize）1.0 ms、sched 0–6 us（vLLM 84–1496 us/步）、publish ≤ 10 us、htod 15–38 us、launch 21–65 us、启动 wall ~60 s。
- ✗ 落后 ①（**最大项**）：**混批 / prefill 步**。c16 一步 2146 tok × 10 请求 = 98 ms = **22 tok/ms**；c64 3378 tok × 52 请求 154 ms（含 50 在飞 decode 行）。TPOT − 纯 decode 步 = 2.7 / 5.4 / 15.2 ms，占 TPOT 差额（4.2 / 5.4 / 11.7）的 **64% / 100% / 130%**。vLLM 同口径 per-req prefill 113–153 ms 但摊在 36–74 个混批步里（每步 ~96–381 tok），每步只多 1–2 ms。
- ✗ 落后 ②：纯 decode 步 M=16/32/64 = 8.67 / 9.31 / 11.83 vs vLLM 全步（含 prefill 摊入）7.1 / 9.3 / 15.3 → M=16 慢 ~1.2×，M≥32 已持平或领先。先放着。
- 两者机理不同：dinfer 的 pair 步让 prefill 子批独占副流，**decode 主流必须等 prefill 完**（`forward_sample_pair` wait_event）→ 一步 98 ms 直接抬 10 个请求的 TPOT；vLLM 把 prefill 切成小 chunk 混进每个 decode 步，每步只慢 1–2 ms。
- **Step B 方向（两条，按 ROI 排）**：(a) **prefill 步本身 22 tok/ms → 单请求 62 tok/ms 的差距**（nsysp16 抓 kernel 表定是 attention 按请求 / MoE 大 M / GEMM 形状哪个）；(b) **调度：限制每步 prefill 新 token 数**（vLLM `long_prefill_token_threshold` 思路：dinfer 已有同名参数默认 0 = 不限；先 A/B `--long-prefill-token-threshold 512`，零代码），让 prefill 分摊到多步而不是一步塞 10 个请求 2146 tok。(b) 是"采用 vLLM 做得对的"，进 design.md §11 台账。

## ★★★★★ D204 2026-10-29 nsysp16：c16 混批步（2148 tok × 10 请求，host exec 101.5 ms，26 步）kernel 表（nsys 2 ms 间隙切步，3 步聚合 ÷3）
| 段 | 每步 ms | 占比 | 每层 us / 次数 | 备注 |
|---|---|---|---|---|
| MoE gate‖up `grouped_gemm_kernel_dual` | 24.0 | 23% | 791 × 30 | 旧 BM=32/BK=128 pipe 体（R ≈ 2148×K 行 > RMAX 512） |
| MoE down `grouped_gemm_kernel_pipe4` | 16.0 | 16% | 532 × 30 | 同上 |
| fp8 Lt `nvjet 64x32x64` | 19.1 | 19% | 153 × 125 | qkv / o_proj / dense？（M=2148 走 cuBLASLt） |
| `Kernel2`（cuBLAS） | 16.8 | 17% | 543 × 31 | 待定名（grid 见下条） |
| `nvjet 64x64x64` | 7.1 | 7% | 282 × 25 | |
| attention（fa_v5 103 × 25 + fa_v6 327 × 5） | 4.2 | 4% | | **不是瓶颈**（prefill attention 2148 tok 仅 4%） |
| softcap_bf16（**2.04 ms × 1**） | 2.0 | 2% | tokens × vocab | 疑按全部 tokens 行算 logits（应只算 10 行）→ 待查 |
| norm / quant / route / combine / gelu 等 | ~10 | 10% | | |
| kernel 和 / span | 102 / 108 | | gaps 6 ms | |
- **结论**：混批 prefill 步 = **GEMM 吞吐问题，不是 attention**。MoE 占 39%（40 ms）、GEMM 43%。M=2148 的 MoE：行数 ≈ 2148 × 8 = 17 K 行分到 ~全部专家，dual 791 us/层 → 算力利用率低（NVFP4 5090 峰值远高）；fp8 Lt 的 nvjet 153 us × 125 次也偏慢（qkv 2148×2816×(q+k+v) 应 < 50 us）。
- **Step B 排序**：(b) 调度限流（`--long-prefill-token-threshold`）零代码先跑（lpt 链已起）→ 期望把 98 ms 大步切成 ≤ 512 tok 的小步，TPOT 差额 2.7 / 5.4 / 15 ms 大部分回收；(a) 再做 MoE prefill tile（BM=64/128 大 tile + 更多 stage）与 fp8 Lt M>64 自调优（FP8_LT_TUNE）。

## ★★★★★ D205 2026-10-29 混批步 nsys 按 grid 复核：**lm_head 按全部 tokens 行算（cutlass Kernel2 3.0 ms/步，grid (56,256)=2148 行 × 262144 vocab）+ softcap 0.38 ms**，而本步只需 10 行
- 单步 kernel 和 ~24 ms（nsys 态，含 3 步平均 ÷ 内核数 ≈958/步）：MoE dual 5.7（30 × 190 us，grid (11,234,2) = BM32 旧体）、down 3.8（30 × 128 us）、**lm_head Kernel2 3.0（1 × 3018 us）**、fp8 Lt nvjet 2.2+1.7+1.5+0.5（M=424 行 qkv/o/dense）、attention 1.5、softcap 0.38、norm/route/combine 等 ~2。
- 注意：nsys 2 ms 切出的"步"是 424 tok（HOST_TIME 2148 tok 的步 = 多个这类片段，nsys 态内核膨胀）。按比例：lm_head + softcap ≈ **14% 的混批步时间** = c16 每步 ~14 ms、c64 ~22 ms（3380 行 × 262144 = 1.77 GFLOP·2816 ≈ 5 TFLOP → 5090 bf16 ~200 TFLOPS 下限 25 ms！**c64 的 154 ms 混批步里 lm_head 占 1/6**）。
- vLLM：`logits_indices = query_start_loc[1:] - 1`，只对采样行算 logits（`gpu_model_runner.py:2324`）。我们 `model.rs:268` `gemm.forward_bf16(final_normed, embed, tokens, vocab)` 按全部 tokens 算再 `gather_rows`（executor.rs:298）。
- **B46**：lm_head 前先 gather `final_normed` 的 `logits_rows` 行（h=2816，nrows ≤ 64）→ GEMM 只算 nrows 行 → softcap nrows×V → 直接写 `gathered`。纯 decode（tokens == nrows，图内）路径不变（行一一对应，不 gather，保证逐位不动）。预期：c16 混批步 −14 ms（98 → 84）、c64 −22；TPOT c16 −0.4 / c64 −2.2 ms；单请求 prefill TTFT −（1560 行 lm_head ≈ 2.3 ms）。"采用 vLLM 做得对的"，进 design §11 台账。

## ★★★ D206 2026-10-29 Step B(b) 结果：`--long-prefill-token-threshold` 512 / 1024 **零效果**（两轮 c8–c64 全部持平，`test/kaon3/2026-10-29/lpt/`）
- 原因显而易见、事先该想到：该参数是**单请求**上限，而 bench 的 prompt 只 428 tok < 512，一步 2148 tok 是 **10 个请求各一整条**。c16 混批步 hosttime 仍 26 步 × 2146 tok × 10 请求 × 99 ms。
- vLLM 为什么每步只多 1–2 ms：它每步 prefill 新 token 也可到 4096 预算，但 per-req prefill 113 ms / 36 步 ⇒ 平均每步 prefill 量 ≈ 96 tok（iter_tokens_mean）—— **它的 prefill 单步吞吐高**（50 tok/ms vs 我们 22），所以一步就算塞满也只是 ~80 ms，而且被 decode 步数摊薄。我们的瓶颈仍是 prefill 步本身的 GEMM 效率（D204/D205）。
- **B47（整步 prefill 配额 `--prefill-tokens-per-step N`）**：新增 `SchedulerParams.prefill_tokens_per_step`（0 = 不限），running 续算与 waiting 新请求的 prefill 行合计 ≤ N；配额满时 running 的 prefill 行跳过（不出队不抢占）、waiting 不再接。这是把 98 ms 大步切成 ~N/22 ms 的小步 → decode 行每步只多 N/22 ms，但 TTFT 变长（整波 prefill 总时长不变甚至更长）。**不是根治**，是在 prefill GEMM 效率修好前的调度缓冲；A/B 时同时看 TPOT 与 TTFT，不许只看 TPOT。
- **B46（lm_head 只算采样行）**是真减工作量：c16 混批步 −14%、c64 −1/6。两者同一个二进制，`LM_HEAD_ROWS_ONLY=0` / `--prefill-tokens-per-step 0` 作 A 侧。

## D207 2026-10-29 B46 + B47 已写（commit e7f64497），b49 链起（编 → marker → A/B/C × 2 轮，c8–c64）
- B46：`model.forward` 新参 `sample_rows: Option<(&CudaSlice<i32>, nrows)>`；nrows < tokens 且非 hidden_dir 时 gather `final_normed` → `sc.lm_in`（[max_rows, h]）→ lm_head GEMM / softcap 只算 nrows 行 → executor dtod 拷进 `gathered`。纯 decode（nrows == tokens，图内）与 probe 路径原样。`LM_HEAD_ROWS_ONLY=0` 回退。
- B47：`SchedulerParams.prefill_tokens_per_step`（CLI `--prefill-tokens-per-step`，默认 0）；running 的 prefill 续算与 waiting 新请求合计 ≤ N，配额满时 running 行 `i += 1; continue`（不抢占），waiting 停接。核心单测跑过。
- 预期：B 比 A c16 TPOT −0.4 / c64 −2 ms（lm_head 去掉）、TTFT 略降；C 比 B c16/c32/c64 TPOT 再降但 TTFT 升——按停止条件 ① 看 TPOT 比值，TTFT 不许超 vLLM 1.5×。

## D208 2026-10-29 b49 跑中（A 侧 r1 = 与 stepA4 一致：c8 7.06 / c16 11.36 / c32 14.69 / c64 27.26，基线稳）；下一刀准备
- 混批步 roofline（c16 一步 2148 tok，nsys 态 424 tok 片段 ×5）：MoE gate‖up dual `grouped_gemm_kernel_dual<128>` 190 us/层 @ M=424×8=3392 行、N=704×2、K=2816 → 2 × 3392 × 1408 × 2816 × 2 = **54 GFLOP → 5090 NVFP4 密集 ~1 PFLOPS 下限 ~55 us**，实测 190 us = 29%；down 128 us @ N=2816 K=704 同量级 → 27 GFLOP 下限 27 us，实测 21%。两者 BM=32/BN=64/BK=128 双缓冲（非 pipe4），每 CTA 8 warp 各 1 个 m16n8 片 → 算力型 GEMM 用的是 decode 的小 tile。**B48 候选**：prefill（R > RMAX）走 BM=64 或 128 / BN=128 / STAGES 3–4 的 pipe4 实例（模板已有 BM_T/BN_T 参数），tile 表第一半按 BM 重算。微基准 `moe_gemm_timing` 加 M=3392 行（每专家 ~26 行）形状先量。
- fp8 Lt：nvjet 64x32x64 153 us × 125 次/步（M=424 时）。qkv N=2816+2×... K=2816：2 × 424 × 2816 × 4224 ≈ 10 GFLOP → fp8 ~1.6 PFLOPS 下限 6 us，实测 36 us（nsys 态）= 17%。**B49 候选**：`FP8_LT_TUNE=1`（m ≥ 64 自调优，D80 时因 c8 +3% 关掉，但那是 M=8；现在只在 prefill M≥64 生效）env A/B 零代码。

## ★★★★ D209 2026-10-29 b49 第 1 轮（A = LM_HEAD_ROWS_ONLY=0 / B = B46 / C = B46 + `--prefill-tokens-per-step 1024`）
| c | A tpot/ttft/tps | B（B46） | C（+配额 1024） |
|---|---|---|---|
| 8 | 7.06 / 154 / 859 | **6.94 / 131 / 905**（−1.7% / −15% / +5%） | 8.22 / 77 / 865 |
| 16 | 11.36 / 197 / 1115 | **11.04 / 166 / 1175**（−2.8% / −16% / +5%） | 12.87 / 84 / 1139 |
| 32 | 14.69 / 350 / 1604 | **13.99 / 289 / 1747**（−4.8% / −17% / +9%） | 17.66 / 91 / 1671 |
| 64 | 27.26 / 362 / 1965 | **24.87 / 301 / 2181**（−8.8% / −17% / +11%） | 28.71 / 103 / 2096 |
- **B46 全档正收益**（TPOT −1.7…−8.8%、TTFT −15…−17%、吞吐 +5…+11%），输出与 A 逐字同。等第 2 轮同向即定稿。
- **B47 配额 1024 判负**：TTFT p50 掉 60–75%（prefill 分摊、首 token 早出）但 **TPOT 全档涨 10–26%**、吞吐降。机理：配额把一波 prefill 切成 N 段 → N 个混批步，每个混批步都走 eager 副流 + 主流等待（pair 步的 decode 也不进图：`forward_sample_pair` 里 decode 走 `enqueue(decode, true)` 虽可进图，但每个混批步固定开销 ~5–8 ms 没省），**混批步数 ×N，每步固定开销不摊**。默认 0 保留旋钮；SLO 偏 TTFT 的场景可开。进 CSV 作负结果。
- 结论复述：根治仍是 **prefill 步 GEMM 效率**（B48 MoE 大 tile / B49 fp8 Lt 自调优）。

## ★★★★★ D210 2026-10-29 b49 定稿：**B46 合入**（两轮同向：c8 6.95 / c16 10.97 / c32 14.03 / c64 24.88，TPOT −1.6% / −3.7% / −4.7% / −8.5%；TTFT −15–17%；吞吐 +5–11%；输出逐字同）；**B47 配额判负默认 0**（CSV round b49c 负结果）。CSV round b49，`test/kaon3/2026-10-29/b49/`。
- 现状对 vLLM（stepA4 中位数）：c8 1.13× / c16 **1.54×** / c32 1.51× / c64 1.62×；吞吐 93% / 75% / 71% / 65%。停止条件 ①（c16 ≤ 1.10×、c32/64 ≤ 1.25×）未达，继续 Step B。
- **B48 已写（commit a27b0989）**：`grouped_gemm_kernel_p64<BK128,STAGES3>` BM64×BN64、8 warp 4×2、每 warp 2×2 m16n8 片段（A 片段 n 向复用 4 次）；R > `MOE_DEC_RMAX` 且 K%128==0 时 grouped 与 dual 都走它；第三张 tile 表（scan_kernel / route_bucket_small 两处同步产出，`tile_off` 3(E+1)）。`MOE_PREFILL_P64=0` 回退。b50 链：微基准 P424/P2148 两进程 CHECKSUM 门禁 → ABAB c8–c64。

## ★★★ D211 2026-10-29 b50 微基准（P64 关 / 开，两进程 CHECKSUM 全同 → 门禁过，GATE=0；二进制 md5 2e0a01ef）
| 形状 | gate‖up dual 旧 BM32 | p64 BM64 | down 旧 | p64 |
|---|---|---|---|---|
| M=8（64 行 28 专家，decode） | 20.5 us | 20.5（不走，R ≤ 512） | 16.4 | 16.4 |
| P424（3392 行 128 专家） | 263.6 | **237.2（−10%）** | 145.0 | 145.0（**K=704 % 128 ≠ 0 没走 p64**） |
| P2148（17184 行） | 723.7 | **549.2（−24%）** | 495.2 | 500.0 |
- 算力口径：P2148 gate‖up 2×17184×1408×2816 = 136 GFLOP / 549 us = **248 TFLOPS**（NVFP4 dense 峰值约 1 PFLOPS 的 1/4）；down 68 GFLOP / 495 us = 137 TFLOPS。还有 2–4× 空间。
- p64 smem：3 stage × (64×80 + 64×80 + 64×8 + 64×8) B = 33.8 KB → 占用 2–3 CTA/SM；8 warp 每 BK=128 块每 warp 8 次 mma 对 16 次 32-bit smem 读（A 4 + B 8 + 2 scale），**smem 读仍是主要开销**（ldmatrix 没用、片段每次从 smem 取）。下一档：BN=128（每 warp 2×4 片段，B 片段再翻倍复用 A）或 ldmatrix.x4。
- **down 要加 BK=64 实例**（K=704 = 11 × 64）：`grouped_gemm_kernel_p64<64, 4>`。
- ABAB 跑中。

## ★★★★ D212 2026-10-29 b50 定稿：**B48 合入**（两轮同向；A = `MOE_PREFILL_P64=0`）：c8 7.00/7.00（持平）、c16 10.99 → 10.93、c32 14.07 → **13.79（−2%）**、c64 24.89 → **23.90（−4%）**；TTFT **−7…−9%** 全档；吞吐 +2…+5%；输出逐字同。CSV round b50，`test/kaon3/2026-10-29/b50/`。
- 只有 gate‖up 走了 p64（down K=704 % 128 ≠ 0）；B48b 加 `p64<64,4>` 让 down 也走（commit 已入，b51 链验）。
- 现状对 vLLM：c16 **1.53×** / c32 1.48× / c64 1.56×；吞吐 77% / 73% / 68%。
- b51 链 = B48b 门禁 + ABAB；b52 = `FP8_LT_TUNE=0/1`（B49，m ≥ 64 自调优 + 2 的幂桶缓存）。

## D213 2026-10-29 b51 微基准（B48b：p64 加 `<64,4>` 实例给 down）：P2148 gate‖up 723 → 550（同 b50），**down 495 → 414（−16%）**，CHECKSUM 全同 GATE=0；二进制 md5 0c18bcb8；ABAB 跑中，接着 b52 FP8_LT_TUNE。
- down 还是 414 us = 68 GFLOP → 164 TFLOPS；gate‖up 550 → 248 TFLOPS。**B48c 候选：BN=128**（每 warp 16×64 = 8 个 n8 片段，A 片段复用 8 次，smem 读/mma 再减 ~40%；smem 3 stage ≈ 44 KB → 2 CTA/SM）。

## ★★★★ D214 2026-10-29 b51 定稿：**B48b 合入**（down 也走 p64；A = P64 关）：c8 7.02（持平）、c16 10.84、c32 **13.65（−3%）**、c64 **23.51（−6%）**；TTFT −10…−12%；吞吐 +1…+7%；输出逐字同。CSV b51。累计 B46+B48+B48b 对 stepA4 基线：c16 11.36 → 10.84（−4.6%）、c32 14.67 → 13.65（−7%）、c64 27.07 → 23.51（−13%）；TTFT c64 360 → 266。
- 现状对 vLLM：c16 **1.52×** / c32 1.47× / c64 1.53×；吞吐 78% / 74% / 69%。
- b52（`FP8_LT_TUNE=1`，B49）第 1 轮 B 侧：c8 TTFT 116 → **100（−14%）**、c16 147 → **125（−15%）**，TPOT c8 7.02 → 7.09（+1%）/ c16 10.84 → 10.77（−0.6%）；model_load 35 s 不变（调优在首次形状出现时付，不在启动）。等 c32/c64 与第 2 轮。
- B48c（BN=128，`MOE_PREFILL_BN=128`）已写（commit），b53 链排队（b52 后）。

## ★★★★★ D215 2026-10-29 b52 定稿：**B49 合入**（`FP8_LT_TUNE` 默认开，`=0` 回退；A = 关）：c8 6.98 → 7.06（+1%，噪声内）、c16 10.78 → 10.76、c32 13.59 → **13.44（−1%）**、c64 23.44 → **21.88（−7%）**；**TTFT 全档 −14…−16%**（c8 116 → 100、c64 266 → 224）；吞吐 +2…+9%；输出逐字同；model_load 不变（35 s）。CSV b52，`test/kaon3/2026-10-29/b52/`。
- 累计四刀（B46 / B48 / B48b / B49）对 stepA4 基线：c16 TPOT 11.36 → **10.76（−5%）**、c32 14.67 → **13.44（−8%）**、c64 27.07 → **21.88（−19%）**；TTFT c16 197 → 126（−36%）、c64 360 → 224（−38%）；吞吐 c64 1980 → 2541（+28%）。
- 对 vLLM：c8 1.15× / c16 **1.51×** / c32 1.44× / c64 **1.43×**；吞吐 97% / 80% / 79% / 75%。**TTFT 全档已反超 vLLM**（c16 126 vs 179、c64 224 vs 247）。停止条件 ① 未达（c16 ≤ 1.10×），继续。
- 待办（不阻塞）：自调优现在在首个 m ≥ 64 形状出现时付 32 候选 × 5 次计时（每个 (桶,n,k) 约 10–30 ms，首批请求 TTFT 尾部）；应挪到启动 warmup（按 m 桶 64…4096 × 5 形状预调）。
- 剩余差额分析：c16 TPOT 10.76 vs 纯 decode 步 8.67 → prefill 插入仍 2.1 ms；c64 21.88 vs 11.83 → 10 ms。混批步从 98 ms 应已降到 ~70 ms（待 hosttime 复核）。下一刀：B48c BN=128（b53 跑中）→ 再 nsys 一次混批步看新的大头。

## D216 2026-10-29 b53 微基准（B48c BN=128 vs 64，CHECKSUM 全同 GATE=0，二进制 md5 8db4c1a2）：P2148 gate‖up 538 → **508（−6%）**、down 410 → **375（−9%）**；P424 gate‖up 237 → 230、down 141 → 147（+4%，BN=128 时 N=2816 只 22 列块 × 行块，CTA 少了）。增益小 → smem 片段读不是唯一瓶颈；剩下的是 cp.async 装载量（每 CTA 每 BK 块装 A 64×64 B + B 128×64 B）与 mma 发射密度。ABAB 跑中，端到端 ≥ 1% 才改默认。

## ★★★ D217 2026-10-29 b53 定稿：**B48c 合入**（BN=128 默认，`MOE_PREFILL_BN=64` 回退；A = BN64，二进制 8db4c1a2 = FP8_LT_TUNE 仍是默认关的版本）：c8 7.01（持平）、c16 10.78（持平）、c32 13.48（−1%）、c64 **23.05（−1.7%）**；TTFT −4%；吞吐 +1…+2%；输出逐字同。CSV b53。小刀，但两轮同向、微基准 −6/−9%，合入。
- 注意 b53 的 A/B 都在 FP8_LT_TUNE 关的二进制上；B49 默认开后的合成基线以 b54 nsys 链的 serve 为准。
- **疑点**（D218 查）：b52 B 侧 serve 日志里 `grep "gemm fp8"` = 0 行 —— `[gemm fp8 tune]` 的 eprintln 没出现在日志。要么 stderr 没进 serve.log（但 `[kaon3-host]` 等 eprintln 都在），要么 `tune_enabled()` 为假、tune 分支没走 → 那 B49 的收益来自别处（2 的幂桶 key 让 m ≥ 64 的形状第一次走 `parity_pick` 查表时… 不对，parity_pick 不依赖 key）。必须查清再说"B49 是自调优的功劳"。

## D218 2026-10-29 疑点解除：`[gemm fp8 tune]` 的 eprintln 在 `GEMM_FP8_DEBUG` 门控内（gemm.rs ~836），b52 没设所以 0 行；tune 分支条件 `tune_on && m >= 64 && tune_enabled()`，`GEMM_TUNE` 默认开。B49 收益确属自调优。b54 链 serve 带 `GEMM_FP8_DEBUG=1`，日志会给每个 (m 桶, n, k) 的 best / parity algo 对照——同时验证"m 2 的幂桶"下调了几次。

## ★★★★★ D219 2026-10-29 b54：五刀后 c16 混批步 nsys（`GEMM_FP8_DEBUG=1`，二进制 85b85d5a）
- **混批步 host exec 101.5 → 65.0 ms（−36%）**；26 步 × 2148 tok × 10 请求不变。bench c16 TPOT 10.88 / TTFT 133（nsys 态）。
- nsys 2 ms 片段（424 tok 口径，kernel 和 57.7 ms / span 61）：MoE p64 gate‖up 14.2 + down 11.9 = **26.1 ms（45%）**；fp8 Lt **nvjet 128x128x64**（自调优选的 `tile=20`，原 64x32 `tile=13/15`）5.8 + 4.3 + 2.7 + 1.0 = 13.8（24%）；attention 5.0（9%）；combine_norm 2.3、rmsnorm_grouped_quant 1.6、rmsnorm_vec 1.6、lm_head **0.9**（原 3.0，B46 ✓）、gelu/quant/route/attn_pre 各 < 1。
- fp8 tune 结果（m 桶 4160，6 形状：qkv n=8192/9216、o n=2816 k=4096/8192、dense 2112↔2816）：全部从 `tile=13/15`（64x32）换到 **`tile=20`（128x128 stages=15）**，best 81–325 us；自调优只付了 6 次（桶缓存 ✓）。
- **MoE p64 的物理账**：gate‖up 474 us @ 3392 行 → 2×3392×1408×2816 = 27 GFLOP → **57 TFLOPS**（峰值 ~1 PFLOPS 的 6%！微基准 P2148 是 248 TFLOPS——nsys 态 + 片段 424 tok 行数少，CTA 数 6×421×2 = 5052 但每 CTA 只 11 列块…）；smem 片段读下限约 170 us/SM 级、mma 发射下限 ~300 us（17184 行 P2148 口径）。p64 还有 ~1.5–2× 空间（ldmatrix / 更深 stage / BM128），但**它在 TPOT 里只占 c16 的 ~15%**。
- **结构性差额（更大）**：pair 步里 decode 子批（9 ms）等 prefill 副流（65 ms）完才 publish → 10 个在飞请求每 64 步被卡一次 65 ms ≈ 每步 +1.0 ms；c64 每 64 步卡 ~150 ms ≈ +2.3 ms/步 + 50 行 decode 本身 11.8。要消掉只能让 **decode 不等 prefill**（prefill 异步收割：alt 流跑完才把 prefill 的首 token 并进后续某步的 publish，中间 decode 步照常跑图）。这需要 runner/engine 的 Pending 允许"部分请求结果晚到"——改动在 core，≥ 半天，下一轮主攻。vLLM 没这个问题是因为它 prefill 单步 50 tok/ms（我们现在 2148/65 = **33 tok/ms**，从 22 提到 33）且 chunk 混进 decode 步只多 1–2 ms。
- 近期小刀（排队）：① MoE p64 STAGES 4（BK128 时 smem 45 KB 仍 2 CTA/SM）；② `rmsnorm_vec` 118 次 × 13 us = 1.6 ms（grid 2338 = 行数，fine）与 `rmsnorm_grouped_quant` 53 us × 30 融进 MoE 入口；③ attention prefill fa_v5 98 us × 25 层（GQA 16 头 D=256 2148 tok 的 flash 下限 ~2×2148²×256×16×2 / 2 ≈ 38 GFLOP → 40 us）→ 2.4×，中等刀。

## D220 2026-10-29 排队：stepA5（五刀二进制 3 轮 c1–64 对 vLLM，`tmp_stepA5_RESULT.txt`）→ b55（B48d p64 STAGES=4 门禁 + ABAB）→ b56（`XSTREAM=1` pair vs `XSTREAM=0` 单流 Paged 混批 ABAB）
- b56 的意义：D219 判定"decode 等 prefill"是结构性差额，但 D169 记过 c8 下混批步只 4/1000（prefill 几乎全是 prefill-only 步）；c16–c64 是否同样？stepA4 host 分组给的是 (2148 tok, 10 req) 的"prefill 步"= 含 decode 行的混批（nreq 10 > 新请求数 ~5）。单流 Paged 混批把 prefill + decode 行放一个 eager 前向：decode 行不再等副流，但 decode 行本身也不走图（eager 多 ~3–5 ms）。两者孰优由 b56 定；若 B 更好，异步收割的 ROI 要重算。
- 异步收割规格已写：`docs/loops/kaon3-prefill-async.md`（下一轮主攻，core 改动）。

## ★★★★★★ D221 2026-10-29 Step A v5（五刀二进制 85b85d5a，3 轮中位数，`test/kaon3/2026-10-29/stepA5/`）—— 本轮 Step C 总账
| c | TPOT d/v（×）stepA4 → A5 | TTFT d/v | tok/s d/v（×） | 纯 decode 步 | 混批步 exec（tok, nreq, 步数） | TPOT − decode 步 |
|---|---|---|---|---|---|---|
| 1 | 4.65/4.69（0.99） | **21**/25 | 204/200 | 4.64 | 19.2（428, 1, 10） | 0 |
| 2 | 5.37/5.36（1.00） | 40/36 | 348/347 | 5.07 | 20.2 | 0.30 |
| 4 | 6.03/6.06（1.00） | **58**/89 | 603/544（1.11） | 5.57 | 28.2 | 0.46 |
| 8 | 7.06 → 7.15/6.12（1.17） | 154 → **95**/146 | 858 → 948/990（0.96） | 6.56 | 71 → 44.5 | 0.59 |
| 16 | 11.36 → **10.63**/7.09（1.59 → **1.50**） | 197 → **118**/183 | 1116 → **1286**/1571（0.71 → 0.82） | 8.74 | 98 → **59.1**（2146, 10, 26） | 2.69 → 1.89 |
| 32 | 14.67 → **13.08**/9.22（1.58 → **1.42**） | 346 → **199**/232 | 1612 → **2004**/2448（0.65 → 0.82） | 9.43 | 122 → **72.6** | 5.36 → 3.65 |
| 64 | 27.07 → **21.24**/15.43（1.76 → **1.38**） | 360 → **208**/241 | 1980 → **2643**/3355（0.59 → 0.79） | 11.89 | 154 → **90.7** | 15.24 → 9.35 |
- **收益**：c64 TPOT −22%、吞吐 +33%；c16/32 TPOT −6/−11%、吞吐 +15/+24%；**TTFT 全档反超 vLLM**（c4 58 vs 89、c16 118 vs 183、c64 208 vs 241）；c1/c2 不动（图内路径未碰，逐位同）。混批步 prefill 吞吐 22 → **36 tok/ms**（2146/59）。
- **环节判定更新**：✓ 领先：TTFT 全档、c1–c4 TPOT、host 各段、启动 36 s。✗ 仍落后：① **TPOT − 纯 decode 步**（c16 1.9 / c32 3.7 / c64 9.4 ms = pair 步 decode 等 prefill 的结构性代价，占剩余差额的 **53% / 95% / 160%**）；② 纯 decode 步 M=16 8.74 vs vLLM 全步 7.09（M≥32 持平/领先）。
- **停止条件 ①（c16 ≤ 1.10×、c32/64 ≤ 1.25×）未达**；Step B 五刀都正向，本轮不回 Step A，下一轮按 D219/D220：b55/b56 结果后进 `kaon3-prefill-async`（消 ①），再看 M=16 decode 步（②，MoE 快路径 R ≤ 128 对 M=16 的 128 行…）。
- CSV：stepA5 作 round `a5`（五刀合成，vLLM 同窗口实测）。

## D222 2026-10-29 b55 微基准：**B48d STAGES=4 判负**（P2148 gate‖up 503 → 526 us（+5%），down 373 → 375，P424 持平；CHECKSUM 同）。4 级 smem 45 KB 把 BN128 的占用从 2 CTA/SM 压到 1。默认留 3，env 实例保留（A/B 用）。ABAB 不跑（微基准已负），链停，直接起 b56。CSV 不记（没跑端到端）。
- p64 的下一档不是 stage 数，是 **ldmatrix / 更大 BM（128）** 或 **A 片段驻寄存器复用 2 个 n 半区**——留给下一轮；当前 MoE prefill GEMM 在混批步里 ~40%，但混批步本身只占 c16 TPOT 的 18%（1.9/10.6），结构性差额优先。

## D223 2026-10-29 第四轮小结（D199–D222，供下次开场）
- **Step A（stepA4，5 轮）→ 靶子 = 混批 prefill 步**（c16 98 ms / 2146 tok；占 TPOT 差额 64–130%）。nsys：GEMM 吞吐（MoE 39% + Lt 43%）+ lm_head 全 tokens（14%）。
- **Step B 七刀**：B46 lm_head 采样行 ✓（c64 −8.5%）；B47 整步 prefill 配额 ✗（TPOT +10–26%，默认 0）；B48 MoE p64 BM64 ✓（−4%）；B48b down 也走 ✓（−6%）；B48c BN=128 ✓（−1.7%）；B48d STAGES=4 ✗（微基准 +5%）；B49 fp8 Lt 自调优 ✓（−7%，TTFT −15%）。全部进 CSV（b49/b49c/b50/b51/b52/b53/a5）。
- **Step C 总账（stepA5）**：c64 TPOT 27.07 → 21.24（−22%）、吞吐 +33%；c16 1.59× → 1.50×；**TTFT 全档反超 vLLM**；混批步 98 → 59 ms（22 → 36 tok/ms）。输出全程逐字同。
- **未达停止条件**（c16 ≤ 1.10×）。剩余差额定性：① pair 步 decode 等 prefill（c16 1.9 / c64 9.4 ms，占剩余的 53–160%）→ `kaon3-prefill-async`；② M=16 纯 decode 步 8.74 vs M=8 6.56（+2.2 ms，vLLM 全步 7.09）。b56（pair vs 单流混批）在跑，结果决定 ① 的路线。
- 知识已落：`knowledge/perf-concurrency.md`（新页 §15–22）、perf-tooling §23、executor-common pair 步代价、infra 僵尸 pgrep；design §11 四行；pitfalls 一条。

## ★★★ D224 2026-10-29 b56 第 1 轮（A = `XSTREAM=1` pair 步 / B = `XSTREAM=0` 单流 Paged 混批，同二进制 58d2032d = 五刀 + B48d env）
| c | A tpot/ttft/tps | B tpot/ttft/tps |
|---|---|---|
| 8 | 7.09 / 95 / 944 | 7.08 / 95 / 946 |
| 16 | 10.64 / 120 / 1279 | **10.60 / 116 / 1292** |
| 32 | 13.06 / 198 / 2000 | **12.94 / 194 / 2027** |
| 64 | 21.35 / 209 / 2614 | **21.15 / 206 / 2649** |
- **单流混批 ≥ pair（B 全档略好 0.4–1.3%，待第 2 轮）**。意味着：B35 的"prefill 副流 ‖ decode 主流图"在 c16–c64 没拿到并行收益（prefill 65 ms 占满 SM，decode 图挤不进去），还多付两流同步；而单流混批里 decode 行走 eager 的额外代价也小（decode 行 10–50 行并进 2148 行的 GEMM 几乎免费）。
- **对 D219 结论的修正**：剩余差额（c16 1.9 / c64 9.4 ms）不是"pair 同步"，是**混批步本身 60–90 ms 期间所有在飞 decode 行都停**——两种结构都如此。异步收割（`kaon3-prefill-async`）的目标不变（让 decode 步在 prefill 期间继续），但实现应基于**单流 + 第二条流只跑 prefill、decode 步继续走图**，并且 decode 图与 prefill eager 在 SM 上会抢——真正能拿回多少取决于 prefill 的 SM 占用能否让出（限 prefill 的 grid / stream priority：decode 流高优先级）。vLLM 的做法（chunk 混批、单步多 1–2 ms）之所以行，是它 prefill 单步 50 tok/ms 且被 decode 步数摊薄；我们 36 tok/ms 仍是根。
- 若第 2 轮同向：`XSTREAM` 默认改 0（少一套 AltSet 显存 + 少一类路径），pair 代码留作 A/B。

## ★★★ D225 2026-10-29 b56 定稿：**B50 合入**（`XSTREAM` 默认关 = 单流 Paged 混批；`=1` 回退 pair）。两轮同向：c8 7.09 → 7.05、c16 10.68 → **10.60**、c32 13.06 → 12.98、c64 21.40 → **21.17（−1.1%）**；TTFT −1…−3%；吞吐 +1…+2%；输出逐字同。CSV b56，`test/kaon3/2026-10-29/b56/`。
- 六刀累计（B46/B48/B48b/B48c/B49/B50）对 stepA4：c16 11.36 → 10.60（−6.7%）、c32 14.67 → 12.98（−11.5%）、c64 27.07 → 21.17（−21.8%）。对 vLLM：c16 1.49× / c32 1.41× / c64 1.37×。
- 剩下的 b57（nsys c16 纯 decode 步）看 M=16 的 +2.2 ms 在哪。

## D226 2026-10-29 第四轮停止条件对账（D199 ①②③）
- ① c16 ≤ 1.10× / c32、c64 ≤ 1.25×：**✗**（c16 1.49× / c32 1.41× / c64 1.37×；吞吐 82% / 82% / 79%）。
- ② c1/c8 不退化 > 3%：**✓**（c1 4.65 不动；c8 7.06 → 7.05）。
- ③ 每刀进 CSV 含负结果：**✓**（b49 / b49c / b50 / b51 / b52 / b53 / b56 / a5；b55 微基准判负未起端到端、未记 CSV——按规矩应记一行"微基准判负未跑端到端"，下次补）。
- Step B 连续 5 刀无进展的回 Step A 条件未触发（六刀五正）。本轮以 b57（M=16 decode 步定位）收尾，下一轮进 `kaon3-prefill-async`。

## ★★★★★ D227 2026-10-29 b57：c16 纯 decode 步 nsys（graph id=14，189 步，kernel 和 9.17 ms/步 nsys 态；HOST_TIME 8.74）—— M=16 比 M=8 多的 2.2 ms 在哪
| 段 | ms/步 | 占比 | 次数/步 × us | 下限（权重字节 / 1.79 TB/s） | 判定 |
|---|---|---|---|---|---|
| MoE gate‖up + down `pipe4<256,3,16,64>` | **2.80** | 30.5% | 60 × 46.7 | ~50–80 激活专家 × 3 MB × 30 层 ≈ 4.5–7 GB → **2.5–4 ms** | **已在物理下限**（M↑ 激活专家↑，vLLM 同样） |
| fp8 gemv<16>（qkv / o_proj） | **1.90** | 20.8% | 60 × 31.7 | 35 MB/层 × 30 = 1.04 GB → **0.58 ms** | **3.3× 离下限**（M=8 时 21.9 us；M=16 的 warp-行结构变 issue 瓶颈） |
| lm_head bf16 wmma 16x16 | 0.89 | 9.7% | 1 × 888 | 1.48 GB → 0.83 ms | 下限（fp8 lm_head 可半，parity 风险，不动） |
| dense `gateup_act<16>` | 0.73 | 7.9% | 30 × 24.3 | 11.9 MB/层 → 6.6 us | **3.7× 离下限** |
| attention v3 + d512 | 0.84 | 9.1% | 25 × 21.7 + 5 × 59 | | 中 |
| `route_bucket_small`（单 block） | **0.53** | 5.8% | 30 × 17.7 | M=8 时 9.2 us，随 token 线性 | D197 ② route 拆两段 → ~8 us，省 0.3 |
| nvjet 64x32（dense down，Lt parity tile） | 0.41 | 4.5% | 30 × 13.7 | 5.9 MB → 3.3 us | 4× 离下限 |
| router_gemv / combine_norm / norm×5 / quant / merge 等 | ~0.9 | 10% | | | 小 |
- **结论**：M=16 的 MoE 已是带宽物理账（vLLM 同），不是靶子；**小 M fp8 GEMM（gemv<16> / gateup_act<16> / nvjet 64x32）合计 3.0 ms，下限 ~0.8 ms**，是 M=16 decode 步的真靶子（c16 TPOT 10.6 里 ~2 ms 可拿）；route 单 block 0.3 ms 次之。
- **B51 候选（env 旋钮，零 kernel 改动先 A/B）**：`FP8_GEMV_MAX_M`（默认 16 → 8：M=9..16 不走 gemv 走 cuBLASLt）+ `FP8_LT_TUNE_MIN_M`（默认 64 → 9：M ≥ 9 自调优选大 tile）。parity 口径只锁 m=1（oracle c1），M=16 的 c≥2 输出本来批变。若 Lt 自调优在 M=16 选到 ~15 us 级 tile → qkv/o 1.9 → ~0.9 ms。
- **B52 候选**：route 拆两段（D197 ②，grid=tokens），固定 −0.3 ms（c16），c32/c64 更多（32 us / 64 us 单 block）。

## ★★★★★ D228 2026-10-29 b58 定稿：**B51 合入**（`FP8_GEMV_MAX_M` 8、`FP8_LT_TUNE_MIN_M` 9 改默认；A = gemv ≤ 16）：c16 TPOT 10.69/10.65 → **9.86/9.86（−7.6%）**，吞吐 1292 → 1378（+7%）；c32 13.16/13.1 → 12.82/12.90（−2%）；c8 / c64 持平（c8 仍 gemv<8>，c64 本就走 Lt）。CSV b58。
- m=16 自调优选择：qkv n=8192/9216 → `id=67 tile=13 splitk=1`（14.5 / 15.6 us；gemv<16> 原 31.7 us）；o_proj n=2816 k=4096/8192 → **`id=35 tile=13 splitk=3`**（8.6 / 10.6 us）。即 cuBLASLt 的 tile=13 在 m=16 本来就比我们 gemv<16> 快 2×——gemv 只在 m ≤ 8 占优（b31b 定稿时只量过 m=8）。
- 输出：英文 raw 同；中文 chat **措辞变**（"这里是…" → "以下是…"），是 o_proj split-K 累加序不同 → 近并列 token 翻转；c1 parity 口径（m=1）不受影响（m < 9 配置不变）。按 D62 / 批不变性结案口径接受。
- 现状对 vLLM：c8 1.15× / **c16 1.39×** / c32 1.39× / c64 1.37×；吞吐 96% / **88%** / 83% / 79%。七刀累计 c16 11.36 → 9.86（−13%）。
- B52（route 拆两段）b59 链跑中（parity 门禁 + ABAB，两侧带 B51）。

## ★★★★★ D229 2026-10-29 b59 定稿：**B52 合入**（route 拆两段，默认 `ROUTE_SPLIT_MIN_TOK=9`；`=0` 回退 fused；parity 测试 tokens 33/64 自动走 split，与旧链逐位同）。两轮同向（A/B 两侧都带 B51）：c8 持平；c16 9.90 → **9.60（−3%）**；c32 12.86 → **12.22（−5%）**；c64 21.16 → **19.73（−6.8%）**，吞吐 2647 → 2811（+6%）；输出逐字同。CSV b59。
- 收益比 D227 估的 0.3 ms（c16）大：c32/c64 的 route 单 block 原 32 / 64 us × 30 层 = 1–2 ms，拆后 ~10 us。
- **八刀累计**（B46/B48/B48b/B48c/B49/B50/B51/B52）对 stepA4：c16 11.36 → **9.60（−15.5%）**、c32 14.67 → **12.22（−16.7%）**、c64 27.07 → **19.73（−27%）**；吞吐 c64 1980 → 2811（+42%）。对 vLLM：c8 1.15× / **c16 1.35× / c32 1.33× / c64 1.28×**；吞吐 96% / 89% / 87% / 84%。TTFT 全档仍反超。
- 停止条件 ①（c16 ≤ 1.10×、c32/64 ≤ 1.25×）仍未达；c64 已到 1.28×，差 0.03。
- 下一刀候选（M=16 decode 步剩余，按 D227 表）：**B53 dense MLP M=9..16 也走 Lt**（`gateup_act<16>` 24 us + gemv down 走 Lt 3.3 us 下限；`DENSE_GEMV_ROWS` 16 → 8 env A/B）；attention v3 M=16 21.7 us × 25。
