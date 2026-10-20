import sys,csv,re,collections
txt=sys.stdin.read().splitlines()
hdr=[i for i,l in enumerate(txt) if l.startswith("Start (ns)")]
rows=list(csv.reader(txt[hdr[0]:])); h=rows[0]; rows=[r for r in rows[1:] if r and r[0].strip()]
si=h.index("Start (ns)"); di=h.index("Duration (ns)"); ni=h.index("Name")
rows.sort(key=lambda r:int(r[si]))
# 稳态窗口：最后一次 >= 8 MB 的 H2D（权重加载）之后才算（否则加载期拷贝被摊进"decode 步"——engine-gap 6c 误读）
bi=h.index("Bytes (MB)") if "Bytes (MB)" in h else None
if bi is not None:
  big=[int(r[si]) for r in rows if "Host-to-Device" in r[ni] and r[bi] and float(r[bi])>=8.0]
  if big:
    t_ss=max(big); rows=[r for r in rows if int(r[si])>t_ss]
    print(f"steady-state window: after last >=8MB H2D, {len(rows)} events")
steps=[]; cur=[]
for r in rows:
  cur.append(r)
  if "sample_fused" in r[ni] or "argmax" in r[ni]: steps.append(cur); cur=[]
PRE=("wy_","ho_","conv1d_prefill","gqa_fa","prep_kernel")
DEC=("recurrent_decode","gqa_decode","conv1d_update")
def cls(s):
  p=any(any(k in r[ni] for k in PRE) for r in s); d=any(any(k in r[ni] for k in DEC) for r in s)
  return "mixed" if p and d else ("prefill" if p else "decode")
by=collections.defaultdict(list)
for s in steps: by[cls(s)].append(s)
top=int(sys.argv[1]) if len(sys.argv)>1 else 10
for tag in ["decode","prefill","mixed"]:
  ss=by.get(tag,[]); N=len(ss)
  if not N: print(f"[{tag}] none"); continue
  tot=collections.Counter(); cnt=collections.Counter(); span=0; busy=0
  for s in ss:
    t0=int(s[0][si]); t1=int(s[-1][si])+int(s[-1][di]); span+=t1-t0
    for r in s:
      n=re.sub(r"<.*","",r[ni]).strip()[:46]; d=int(r[di]); tot[n]+=d; cnt[n]+=1; busy+=d
  print(f"[{tag}] steps={N} span_ms={span/N/1e6:.2f} kernel_sum_ms={busy/N/1e6:.2f} kernels/step={sum(cnt.values())/N:.0f}")
  for n,d in tot.most_common(top): print(f"  {d/N/1e3:7.0f}us {cnt[n]/N:6.1f}/step {d/cnt[n]/1e3:7.1f}us {n}")
