#!/usr/bin/env python3
"""stepA RESULT.txt → "环节 × 并发" 阶段耗时对标表（中位数）。
dinfer 列来自 [dinfer-host]（render / 纯 decode 步 sched,exec,publish,upd,build,htod,launch,wait / 混批步 exec,tok,nreq）与 [dinfer-metrics]；
vLLM 列来自 [vllm-metrics]（per-req queue / prefill / decode / infer、每步 sched、prefill 步数、iter tokens）。
用法：python3 -I scripts/dev/kaon3_stage_table.py test/kaon3/<date>/stepA6/RESULT.txt"""
import re, sys, statistics as st, collections
R = collections.defaultdict(lambda: collections.defaultdict(list)); c = None
def num(s, k, default=float('nan')):
    m = re.search(k + r'=([0-9.]+)', s); return float(m.group(1)) if m else default
for l in open(sys.argv[1]):
    m = re.match(r'=== round=(\d+) c=(\d+)', l)
    if m: c = int(m.group(2)); continue
    if c is None: continue
    if '[dinfer] ' in l:
        R[c]['tpot'].append(float(re.search(r'TPOT mean [\d.]+ p50 ([\d.]+)', l).group(1))); R[c]['ttft'].append(float(re.search(r'TTFT mean [\d.]+ p50 ([\d.]+)', l).group(1)))
    if '[vllm] ' in l:
        R[c]['v_tpot'].append(float(re.search(r'TPOT mean [\d.]+ p50 ([\d.]+)', l).group(1))); R[c]['v_ttft'].append(float(re.search(r'TTFT mean [\d.]+ p50 ([\d.]+)', l).group(1)))
    if '[dinfer-host]' in l:
        R[c]['render'].append(num(l, 'render') / 1e3)
        d = re.search(r'decode: (.*?)(\|\||$)', l); p = re.search(r'prefill: (.*?)(\|\||$)', l)
        if d and 'none' not in d.group(1):
            s = d.group(1)
            for k in ('sched', 'exec', 'publish', 'upd', 'build', 'htod', 'launch', 'wait'): R[c]['d_' + k].append(num(s, k) / 1e3)
        if p and 'none' not in p.group(1):
            s = p.group(1); R[c]['p_exec'].append(num(s, 'exec') / 1e3); R[c]['p_tok'].append(num(s, 'toks/step')); R[c]['p_nreq'].append(num(s, 'nreq')); R[c]['p_steps'].append(num(s, 'steps'))
    if '[dinfer-metrics]' in l:
        R[c]['d_step_ms'].append(num(l, 'step_ms')); R[c]['d_sched_us'].append(num(l, 'sched')); R[c]['d_steps'].append(num(l, 'steps'))
    if '[vllm-metrics]' in l:
        pr = re.search(r'per_req queue=([\d.]+)ms prefill=([\d.]+)ms decode=([\d.]+)ms infer=([\d.]+)ms', l)
        if pr:
            R[c]['v_queue'].append(float(pr.group(1))); R[c]['v_prefill'].append(float(pr.group(2))); R[c]['v_decode'].append(float(pr.group(3))); R[c]['v_infer'].append(float(pr.group(4)))
        R[c]['v_sched_us'].append(num(l, 'sched')); R[c]['v_psteps'].append(num(l, 'steps prefill')); R[c]['v_dsteps'].append(float(re.search(r'decode=(\d+) \|', l).group(1)) if re.search(r'decode=(\d+) \|', l) else float('nan')); R[c]['v_iter'].append(num(l, 'iter_tokens_mean'))
    if '[dinfer-gpu]' in l: R[c]['d_sm'].append(num(l, 'sm') )
    if '[vllm-gpu]' in l: R[c]['v_sm'].append(num(l, 'sm'))
def M(c, k, f=1.0, d=2):
    v = [x for x in R[c].get(k, []) if x == x]
    return f'{st.median(v) * f:.{d}f}' if v else '-'
print('## dinfer 各阶段（ms，中位数）—— 纯 decode 步 / 混批步 / 每请求')
print('| c | TPOT | TTFT | render | decode 步 total | sched | exec | publish | upd | build | htod | launch | wait(GPU) | 混批步 exec（tok, nreq, n） | TPOT−decode 步 | GPU sm% |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for c in sorted(R):
    tot = M(c, 'd_sched'); ex = st.median(R[c]['d_exec']) if R[c]['d_exec'] else float('nan'); tp = st.median(R[c]['tpot'])
    print(f"| {c} | {M(c,'tpot')} | {M(c,'ttft',1,0)} | {M(c,'render')} | {M(c,'d_step_ms')} | {M(c,'d_sched',1,3)} | {M(c,'d_exec')} | {M(c,'d_publish',1,3)} | {M(c,'d_upd',1,3)} | {M(c,'d_build',1,3)} | {M(c,'d_htod',1,3)} | {M(c,'d_launch',1,3)} | {M(c,'d_wait')} | {M(c,'p_exec',1,1)}（{M(c,'p_tok',1,0)}, {M(c,'p_nreq',1,1)}, {M(c,'p_steps',1,0)}） | {tp-ex:.2f} | {M(c,'d_sm',1,0)} |")
print()
print('## vLLM 各阶段（/metrics 窗口增量；per-req ms，sched us/步）')
print('| c | TPOT | TTFT | queue/req | prefill/req | decode/req | infer/req | sched us/步 | prefill 步数 | decode 步数 | iter tok 均 | GPU sm% |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|')
for c in sorted(R):
    print(f"| {c} | {M(c,'v_tpot')} | {M(c,'v_ttft',1,0)} | {M(c,'v_queue',1,1)} | {M(c,'v_prefill',1,1)} | {M(c,'v_decode',1,1)} | {M(c,'v_infer',1,1)} | {M(c,'v_sched_us',1,0)} | {M(c,'v_psteps',1,0)} | {M(c,'v_dsteps',1,0)} | {M(c,'v_iter',1,1)} | {M(c,'v_sm',1,0)} |")
print()
print('## 环节对标（dinfer / vLLM）')
print('| c | TPOT 比 | TTFT 比 | 调度 us/步（d / v） | prefill 吞吐 tok/ms（d 混批步实测 / v 粗估） | 每步 ms（d 纯 decode 步 / v per-req decode÷63） |')
print('|---|---|---|---|---|---|')
for c in sorted(R):
    dt = st.median(R[c]['tpot']); vt = st.median(R[c]['v_tpot']); dtt = st.median(R[c]['ttft']); vtt = st.median(R[c]['v_ttft'])
    dp = (st.median(R[c]['p_tok']) / st.median(R[c]['p_exec'])) if R[c]['p_exec'] else float('nan')
    # vLLM per-req prefill 时间是请求从首次调度到 prefill 完的墙钟（含与其它请求同批）；一波 c 个请求同时 prefill 时每步 ≈ 这个时间 / 该请求参与的 prefill 步数。粗口径：窗口总 prompt tok / (prefill 步数 × 每步墙钟≈per-req prefill/步数)。
    vpf = st.median(R[c]['v_prefill']) if R[c]['v_prefill'] else float('nan'); vps = st.median(R[c]['v_psteps']) if R[c]['v_psteps'] else float('nan')
    vp = (428.0 * 8 * c) / (vpf * (8 * c) / max(1.0, 8 * c / max(1.0, vps / 1.0))) if vpf == vpf else float('nan')  # = 428·8c / (vpf · 8c/ reqs_per_step) ；reqs_per_step = 8c/vps
    vp = 428.0 * (8 * c / vps) / vpf if vps == vps and vpf == vpf else float('nan')  # tok per prefill 步 / 每步时长（≈ per-req prefill）
    vd = (st.median(R[c]['v_decode']) / 63.0) if R[c]['v_decode'] else float('nan')  # per-req decode 时间 / 63 个 decode 步 = 该请求看到的每步时长（含混批）
    print(f"| {c} | {dt/vt:.2f}× | {dtt/vtt:.2f}× | {M(c,'d_sched_us',1,0)} / {M(c,'v_sched_us',1,0)} | {dp:.0f} / ~{vp:.0f}（粗） | {M(c,'d_exec')} / {vd:.2f} |")
