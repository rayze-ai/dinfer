#!/usr/bin/env python3
"""单发 TTFT 按长度探针（engine-gap 8e–8n，probe ttft_len）：python3 -I ttft_probe.py <port> [n=5] [words=430]；words 430 ≈ 1 K token、860 ≈ 6 K；每次换 seed（避免 prefix cache 命中），max_tokens=1，temperature 0；钉核 taskset -c 100-103。"""
import json,time,urllib.request,sys,random
port=sys.argv[1]; n=int(sys.argv[2]) if len(sys.argv)>2 else 5; words=int(sys.argv[3]) if len(sys.argv)>3 else 430
model=json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models"))["data"][0]["id"]
res=[]; pt=None
for i in range(n):
    random.seed(int(time.time()*1e6)%1000000+i)
    prompt=" ".join(str(random.randint(100000,999999)) for _ in range(words))
    body=json.dumps({"model":model,"prompt":prompt,"max_tokens":1,"temperature":0}).encode()
    req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",body,{"Content-Type":"application/json"})
    t=time.perf_counter(); r=json.load(urllib.request.urlopen(req)); dt=(time.perf_counter()-t)*1e3
    res.append(dt); pt=r.get("usage",{}).get("prompt_tokens")
print(f"port {port} model {model} prompt_tokens {pt}: TTFT ms "+" ".join(f"{x:.1f}" for x in res)+f" | min {min(res):.1f} med {sorted(res)[n//2]:.1f}")
