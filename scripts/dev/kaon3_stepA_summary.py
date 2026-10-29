#!/usr/bin/env python3
"""stepA4/stepA5 RESULT.txt → 每并发中位数表（dinfer vs vLLM：TPOT / TTFT / tok/s；dinfer 纯 decode 步与混批步 exec；vLLM prefill 步数 / per-req prefill / sched）。
用法：python3 -I scripts/dev/kaon3_stepA_summary.py test/kaon3/<date>/stepA5/RESULT.txt"""
import re, sys, statistics as st, collections
rows = collections.defaultdict(lambda: collections.defaultdict(list)); c = None
for l in open(sys.argv[1]):
    m = re.match(r'=== round=(\d+) c=(\d+)', l)
    if m: c = int(m.group(2)); continue
    if c is None: continue
    for side in ('dinfer', 'vllm'):
        if l.strip().startswith(f'[{side}] '):
            tp = re.search(r'TPOT mean ([\d.]+) p50 ([\d.]+)', l); tt = re.search(r'TTFT mean ([\d.]+) p50 ([\d.]+)', l); th = re.search(r'([\d.]+) output tok/s', l)
            if tp: rows[c][side + '_tpot'].append(float(tp.group(2))); rows[c][side + '_ttft'].append(float(tt.group(2))); rows[c][side + '_tps'].append(float(th.group(1)))
    if '[dinfer-host]' in l:
        d = re.search(r'decode: steps=(\d+) toks/step=(\d+) nreq=([\d.]+) sched=(\d+) exec=(\d+)', l)
        p = re.search(r'prefill: steps=(\d+) toks/step=(\d+) nreq=([\d.]+) sched=(\d+) exec=(\d+)', l)
        if d: rows[c]['d_exec'].append(float(d.group(5)) / 1e3)
        if p: rows[c]['p_exec'].append(float(p.group(5)) / 1e3); rows[c]['p_steps'].append(int(p.group(1))); rows[c]['p_toks'].append(int(p.group(2))); rows[c]['p_nreq'].append(float(p.group(3)))
    if '[vllm-metrics]' in l:
        pf = re.search(r'steps prefill=(\d+) decode=(\d+)', l); pr = re.search(r'prefill=([\d.]+)ms decode=([\d.]+)ms', l); sc = re.search(r'sched=(\d+)us', l)
        if pf: rows[c]['v_psteps'].append(int(pf.group(1))); rows[c]['v_prefill_ms'].append(float(pr.group(1))); rows[c]['v_sched_us'].append(int(sc.group(1)))
def med(v): return st.median(v) if v else float('nan')
print("c | d_TPOT v_TPOT ratio | d_TTFT v_TTFT | d_tps v_tps ratio | d_decode_step d_mixed_step(toks,nreq,n) | TPOT-dstep | v_psteps v_prefill/req v_sched_us")
for c in sorted(rows):
    r = rows[c]
    print(f"{c:2d} | {med(r['dinfer_tpot']):5.2f} {med(r['vllm_tpot']):5.2f} {med(r['dinfer_tpot'])/med(r['vllm_tpot']):.2f}× | {med(r['dinfer_ttft']):5.0f} {med(r['vllm_ttft']):5.0f} | {med(r['dinfer_tps']):5.0f} {med(r['vllm_tps']):5.0f} {med(r['dinfer_tps'])/med(r['vllm_tps']):.2f} | {med(r['d_exec']):5.2f} {med(r['p_exec']):6.1f}({med(r['p_toks']):.0f},{med(r['p_nreq']):.1f},{med(r['p_steps']):.0f}) | {med(r['dinfer_tpot'])-med(r['d_exec']):5.2f} | {med(r['v_psteps']):3.0f} {med(r['v_prefill_ms']):5.1f} {med(r['v_sched_us']):4.0f}")
