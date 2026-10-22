import sys, os
os.environ.setdefault("VLLM_BATCH_INVARIANT","1"); os.environ.setdefault("VLLM_USE_V1","1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING","0")
import numpy as np
import torch
from vllm import LLM, SamplingParams
S=sys.argv[1]; out=os.environ["OUT"]; os.makedirs(out, exist_ok=True)
llm = LLM(model=S, enforce_eager=True, tensor_parallel_size=1, gpu_memory_utilization=0.85,
          dtype="auto", max_model_len=4096, enable_prefix_caching=False, block_size=32, max_num_seqs=1)
m = llm.llm_engine.model_executor.driver_worker.model_runner.model
cand = m
while not hasattr(cand, "layers"):
    nxt = None
    for name in ("language_model", "model", "text_model"):
        if hasattr(cand, name):
            nxt = getattr(cand, name); break
    if nxt is None:
        raise SystemExit(f"no layers; type={type(cand).__name__} attrs={[n for n in dir(cand) if not n.startswith('_')][:25]}")
    cand = nxt
layers = cand.layers
print("layers:", len(layers), flush=True)
for L in (0, 11):
    attn = layers[L].self_attn
    qkv = attn.qkv_proj
    w = getattr(qkv, "weight", None)
    ws = getattr(qkv, "weight_scale", None)
    is_ = getattr(qkv, "input_scale", None)
    print(f"--- L{L} qkv_proj weight {type(w).__name__} dtype={getattr(w,'dtype',None)} shape={tuple(getattr(w,'shape',()))}")
    print(f"    weight_scale={None if ws is None else (ws.dtype, tuple(ws.shape), ws.flatten()[:4].tolist())}")
    print(f"    input_scale ={None if is_ is None else is_.tolist()}")
    if w is not None:
        try:
            ww = w.detach().to('cpu')
            u8 = ww.view(torch.uint8).numpy().copy()
            np.save(f"{out}/qkv_L{L}.npy", u8)
            print(f"    写出 {out}/qkv_L{L}.npy {u8.shape}", flush=True)
        except Exception as e:
            print(f"    导出失败: {e}", flush=True)
