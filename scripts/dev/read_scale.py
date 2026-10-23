#!/usr/bin/env python3
# 读 checkpoint 里某个标量张量（对拍口径用）。用法：S=<ckpt> NAME=<tensor> python3 scripts/dev/read_scale.py
import os, json
from safetensors import safe_open
S = os.environ["S"]; name = os.environ["NAME"]
idx = json.load(open(os.path.join(S, "model.safetensors.index.json")))
with safe_open(os.path.join(S, idx["weight_map"][name]), framework="pt") as f:
    t = f.get_tensor(name)
    print(name.split(".")[-1], "=", t.item(), t.dtype)
