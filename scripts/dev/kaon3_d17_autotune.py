#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跑一次前向（触发 FlashInfer autotune），然后读它对该形状选中的配置。

关键：autotune 缓存写在容器内 /root/.cache，容器重启即丢，
所以"跑前向"和"读配置"必须在**同一个进程**里完成（本脚本）。
"""
import glob
import json
import os
import sys

os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch

import vllm.model_executor.models.gemma4 as g4
from vllm import LLM, SamplingParams

model_dir = os.environ["MODEL"]
ids = [int(x) for x in sys.argv[1:]]
M, N, K = 24, 8192, 2816  # 目标形状（L0 qkv_proj）

llm = LLM(model=model_dir, enforce_eager=True, tensor_parallel_size=1, gpu_memory_utilization=0.85,
          dtype="auto", max_model_len=4096, enable_prefix_caching=False, block_size=32, max_num_seqs=1)
print("[autotune] 前向一次 ...", flush=True)
llm.generate([{"prompt_token_ids": ids}], SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False)

files = glob.glob("/root/.cache/vllm/flashinfer_autotune_cache/**/autotune_configs.json", recursive=True)
print(f"[autotune] 找到 {len(files)} 个缓存文件", flush=True)
if not files:
    print("FAIL 没有缓存", flush=True)
    sys.exit(2)

for p in files:
    print(f"  文件 {p}", flush=True)
    with open(p) as f:
        data = json.load(f)
    print(f"  顶层类型 {type(data).__name__}", flush=True)

    hits = []

    def walk(node, path=""):
        if isinstance(node, dict):
            blob = json.dumps(node, ensure_ascii=False)
            if str(M) in blob and str(N) in blob and str(K) in blob and len(blob) < 3000:
                hits.append((path, node))
            for k, v in node.items():
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(data)
    print(f"  命中 {len(hits)} 条含 ({M},{N},{K}) 的配置", flush=True)
    for path, node in hits[:6]:
        print(f"  [命中] {path}", flush=True)
        print(f"         {json.dumps(node, ensure_ascii=False)[:700]}", flush=True)

    if not hits:
        s = json.dumps(data, ensure_ascii=False)
        print(f"  预览前 1200 字符: {s[:1200]}", flush=True)

print("DONE", flush=True)
