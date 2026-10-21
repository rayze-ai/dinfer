#!/usr/bin/env python3
"""对拍：同一 chat prompt，基线服务 logprobs vs 参考实现 logits。"""
import json,sys,subprocess,urllib.request,os,glob
os.environ["HF_HUB_OFFLINE"]="1"
import torch
from transformers import AutoTokenizer
SNAP=glob.glob("/models/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/*")[0]
MSGS=[{"role":"user","content":"What is the capital of France?"}]
tok=AutoTokenizer.from_pretrained(SNAP)
prompt_ids=tok.apply_chat_template(MSGS,tokenize=True,add_generation_prompt=True)
print("prompt ids:",prompt_ids[:12],"... len",len(prompt_ids))
# 基线：拿 prompt 最后一个位置的 top-k（chat + echo + logprobs）
port=sys.argv[1] if len(sys.argv)>1 else "28067"
model=json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models"))["data"][0]["id"]
body=json.dumps({"model":model,"messages":MSGS,"max_tokens":1,"temperature":0,
                 "logprobs":True,"top_logprobs":10,"echo":True,"add_generation_prompt":True}).encode()
req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",body,{"Content-Type":"application/json"})
r=json.load(urllib.request.urlopen(req,timeout=180))
jp=r["choices"][0]["logprobs"]["content"]
print("baseline prompt tokens:",len(jp))
last=jp[-1]
base={a["token"]:a["logprob"] for a in last["top_logprobs"]}
print("baseline last-pos top10:",sorted(base.items(),key=lambda x:-x[1])[:10])
json.dump({"prompt_ids":list(prompt_ids),"baseline_top":base},open("/w/parity_base.json","w"))
